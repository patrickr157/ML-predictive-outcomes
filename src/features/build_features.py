"""Derive adjudication age, exertional capacity, and diagnostic group for SSDI applicants.

``pu2014ssa.dta`` has no application year. ``TONSET_AGE`` and ``TLIMWRKAGE`` are
ages when the main condition or work limitation began. They are used as an
adjudication-age proxy only when the reported age is 18 through 66. Otherwise
the extract keeps ``age_at_interview``.
"""

from pathlib import Path
from sys import path as sys_path

sys_path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pandas as pd

from schema import TARGET_NAME, assert_no_forbidden_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = PROJECT_ROOT / "data" / "processed" / "ssdi_claims_clean.parquet"
RAW_DTA = PROJECT_ROOT / "data" / "raw" / "pu2014ssa.dta"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ssdi_features.parquet"

MIN_GRID_AGE = 18
MAX_GRID_AGE = 66
# Ages 60 and older are capped at 59. In this survey that band mixes SSDI
# outcomes with Old-Age and Survivors Insurance early-retirement substitution
# and Full Retirement Age conversions, which otherwise pull age toward denial.
AGE_CAP = 59
EXERTIONAL_TIERS = ("sedentary_only", "light_max", "medium_or_above")

# Census codes from TMAINCONDW / TMAIN in 2014SSA_Metadata_AllSections.pdf.
# ``unreported`` is the reference level and is not one-hot encoded downstream.
DIAGNOSIS_CODE_TO_GROUP: dict[str, str] = {
    "03": "musculoskeletal",
    "04": "musculoskeletal",
    "06": "musculoskeletal",
    "14": "musculoskeletal",
    "21": "musculoskeletal",
    "25": "musculoskeletal",
    "40": "musculoskeletal",
    "13": "circulatory",
    "15": "circulatory",
    "27": "circulatory",
    "34": "circulatory",
    "01": "mental",
    "17": "mental",
    "19": "mental",
    "20": "mental",
    "23": "mental",
    "31": "mental",
    "08": "neurological",
    "11": "neurological",
    "12": "neurological",
    "33": "neurological",
    "36": "neurological",
    "18": "respiratory",
    "10": "endocrine",
    "28": "endocrine",
    "38": "endocrine",
    "05": "sensory",
    "07": "cancer",
    "29": "cancer",
    "02": "other",
    "16": "other",
    "26": "other",
    "30": "other",
    "32": "other",
    "35": "other",
    "37": "other",
    "39": "other",
    "41": "none",
}
DIAGNOSIS_LEVELS = (
    "musculoskeletal",
    "circulatory",
    "mental",
    "neurological",
    "respiratory",
    "endocrine",
    "sensory",
    "cancer",
    "other",
    "none",
)


def load_claims(path: Path) -> pd.DataFrame:
    claims = pd.read_parquet(path).reset_index(drop=True)
    if TARGET_NAME not in claims.columns:
        raise KeyError(f"{path.name} is missing {TARGET_NAME}")
    return claims


def load_aligned_timing(claims: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Attach onset and work-limitation ages in applicant-file order.

    The clean extract has no person id. Alignment is checked against interview
    age, which was taken from the same ``EAPPLYSS == 1`` row order.
    """
    timing = pd.read_stata(
        raw_path,
        columns=["eapplyss", "tage_s", "tonset_age", "tlimwrkage", "tmaincondw", "tmain"],
        convert_categoricals=False,
    )
    timing.columns = [column.lower() for column in timing.columns]
    applicants = timing.loc[timing["eapplyss"].eq(1)].reset_index(drop=True)
    if len(applicants) != len(claims):
        raise AssertionError(
            f"Applicant rows changed: clean extract {len(claims)}, raw filter {len(applicants)}"
        )

    interview_age = pd.to_numeric(applicants["tage_s"], errors="raise").astype("int64")
    clean_age = claims["age_at_interview"].astype("int64")
    if not interview_age.reset_index(drop=True).equals(clean_age.reset_index(drop=True)):
        raise AssertionError("Raw applicant order does not match ssdi_claims_clean.parquet")

    attached = claims.copy()
    attached["tonset_age"] = pd.to_numeric(applicants["tonset_age"], errors="raise")
    attached["tlimwrkage"] = pd.to_numeric(applicants["tlimwrkage"], errors="raise")
    attached["tmaincondw"] = applicants["tmaincondw"].to_numpy()
    attached["tmain"] = applicants["tmain"].to_numpy()
    return attached


def _valid_grid_age(age: pd.Series) -> pd.Series:
    return age.ge(MIN_GRID_AGE) & age.le(MAX_GRID_AGE)


def derive_adjudication_age(claims: pd.DataFrame) -> pd.DataFrame:
    """Prefer condition-onset age, then work-limitation age, then interview age."""
    onset_ok = _valid_grid_age(claims["tonset_age"])
    limitation_ok = _valid_grid_age(claims["tlimwrkage"])

    age = claims["age_at_interview"].astype("float64")
    source = pd.Series("age_at_interview", index=claims.index, dtype="object")

    age = age.mask(limitation_ok, claims["tlimwrkage"])
    source = source.mask(limitation_ok, "tlimwrkage")
    age = age.mask(onset_ok, claims["tonset_age"])
    source = source.mask(onset_ok, "tonset_age")

    claims = claims.copy()
    claims["adjudication_age"] = age.round().astype("int64").clip(upper=AGE_CAP)
    claims["adjudication_age_source"] = source
    return claims


def _condition_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return None
    return text.zfill(2)


def assign_diagnosis_group(claims: pd.DataFrame) -> pd.Series:
    """Use the main work-limitation condition, then the main activity condition.

    ``TMAINCONDW`` is the condition the person names as the main reason for a work
    limitation. When that item is blank, ``TMAIN`` supplies the main condition
    behind activity limits or fair/poor health.
    """
    work_code = claims["tmaincondw"].map(_condition_code)
    activity_code = claims["tmain"].map(_condition_code)
    chosen = work_code.where(work_code.notna(), activity_code)
    group = chosen.map(DIAGNOSIS_CODE_TO_GROUP).fillna("unreported")
    unknown = sorted(set(group.unique()) - set(DIAGNOSIS_LEVELS) - {"unreported"})
    if unknown:
        raise AssertionError(f"Unmapped diagnosis groups: {unknown}")
    return group


def assign_exertional_tier(claims: pd.DataFrame) -> pd.Series:
    """Map the 10 lb and 25 lb lift items to one exertional tier.

    Sedentary: cannot lift 10 pounds, or reports difficulty lifting 10 pounds.
    Light: 10 pounds is unrestricted, but 25 pounds is difficult or impossible.
    Medium or above: 25 pounds is possible and reported without difficulty.
    Inability at 10 pounds overrides an inconsistent "able to lift 25" answer.
    """
    difficulty_10 = claims["difficulty_lift_10lb"].eq(1)
    cannot_lift_10 = claims["able_lift_10lb"].eq(2)
    difficulty_25 = claims["difficulty_lift_25lb"].eq(1)
    cannot_lift_25 = claims["able_lift_25lb"].eq(2)
    unrestricted_25 = claims["able_lift_25lb"].eq(1) & claims["difficulty_lift_25lb"].eq(2)

    tier = pd.Series("light_max", index=claims.index, dtype="object")
    tier.loc[unrestricted_25 & ~difficulty_10 & ~cannot_lift_10] = "medium_or_above"
    tier.loc[difficulty_10 | cannot_lift_10] = "sedentary_only"
    light_limit = ~tier.eq("sedentary_only") & (difficulty_25 | cannot_lift_25)
    tier.loc[light_limit] = "light_max"

    unknown = ~tier.isin(EXERTIONAL_TIERS)
    if unknown.any():
        raise AssertionError(f"{int(unknown.sum())} rows have no exertional tier")
    return tier


def add_functional_features(claims: pd.DataFrame) -> pd.DataFrame:
    claims = claims.copy()
    claims["exertional_tier"] = assign_exertional_tier(claims)
    claims["severe_ambulation_limit"] = (
        claims["difficulty_walk_quarter_mile"].eq(1) & claims["able_walk_quarter_mile"].eq(2)
    ).astype("int8")
    return claims


def build_feature_matrix(claims: pd.DataFrame) -> pd.DataFrame:
    timed = derive_adjudication_age(claims)
    featured = add_functional_features(timed)
    featured["diagnosis_group"] = assign_diagnosis_group(featured)
    # Onset ages and raw condition codes are inputs, not model features.
    # Age stays continuous so it is not repeated as collinear one-hot bins.
    featured = featured.drop(columns=["tonset_age", "tlimwrkage", "tmaincondw", "tmain"])
    assert_no_forbidden_columns(featured.columns)
    return featured


def report(features: pd.DataFrame) -> None:
    print(f"rows {len(features)}")
    print("adjudication_age_source")
    for name, count in features["adjudication_age_source"].value_counts().items():
        print(f"  {name} {int(count)}")
    print("exertional_tier")
    for name in EXERTIONAL_TIERS:
        print(f"  {name} {int(features['exertional_tier'].eq(name).sum())}")
    print(f"severe_ambulation_limit {int(features['severe_ambulation_limit'].sum())}")
    print("diagnosis_group")
    for name, count in features["diagnosis_group"].value_counts().items():
        print(f"  {name} {int(count)}")


def main() -> None:
    claims = load_claims(CLAIMS_PATH)
    aligned = load_aligned_timing(claims, RAW_DTA)
    features = build_feature_matrix(aligned)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}")
    report(features)


if __name__ == "__main__":
    main()
