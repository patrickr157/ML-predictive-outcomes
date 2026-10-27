"""Build the applicant-level SSDI extract from the 2014 SIPP SSA Supplement."""

from pathlib import Path

import pandas as pd

from schema import (
    ANY_DISABILITY_INCOME_SOURCE,
    APPLICANT_SOURCE,
    EXPECTED_APPLICANT_ROWS,
    FEATURES,
    SOURCE_TO_CLEAN_MAP,
    SSDI_RECEIPT_SOURCE,
    SSI_RECEIPT_SOURCE,
    TARGET_NAME,
    assert_no_forbidden_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DTA = PROJECT_ROOT / "data" / "raw" / "pu2014ssa.dta"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ssdi_claims_clean.parquet"

LABEL_SOURCES = (
    APPLICANT_SOURCE,
    ANY_DISABILITY_INCOME_SOURCE,
    SSDI_RECEIPT_SOURCE,
    SSI_RECEIPT_SOURCE,
)


def load_source(path: Path) -> pd.DataFrame:
    """Read the columns required to filter, label, and build features."""
    usecols = list(dict.fromkeys([*SOURCE_TO_CLEAN_MAP, *LABEL_SOURCES]))
    frame = pd.read_stata(path, columns=usecols, convert_categoricals=False)
    frame.columns = [column.lower() for column in frame.columns]
    missing = [column for column in usecols if column not in frame.columns]
    if missing:
        raise KeyError(f"Expected columns missing from {path.name}: {missing}")
    return frame


def label_terminal_ssdi_award(applicants: pd.DataFrame) -> pd.Series:
    """Code 1 for ever-received SSDI and 0 for applicants who never received it.

    SSI-only receipt stays in class 0. An SSDI yes outranks a conflicting
    ``EDINC == 2`` if both are present.
    """
    awarded = applicants[SSDI_RECEIPT_SOURCE].eq(1)
    never_received_ssdi = applicants[SSDI_RECEIPT_SOURCE].eq(2) | applicants[
        ANY_DISABILITY_INCOME_SOURCE
    ].eq(2)
    unlabeled = ~(awarded | never_received_ssdi)
    if unlabeled.any():
        raise ValueError(f"{int(unlabeled.sum())} applicants have no SSDI label")

    labels = pd.Series(pd.NA, index=applicants.index, dtype="Int8")
    labels.loc[never_received_ssdi] = 0
    labels.loc[awarded] = 1

    ssi_only = applicants[SSI_RECEIPT_SOURCE].eq(1) & ~awarded
    if not labels.loc[ssi_only].eq(0).all():
        raise AssertionError("Applicants who received only SSI must remain in class 0")
    return labels


def build_analysis_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Keep ever-applicants and replace source names with the clean schema."""
    applicants = source.loc[source[APPLICANT_SOURCE].eq(1)].copy()
    if len(applicants) != EXPECTED_APPLICANT_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_APPLICANT_ROWS} applicants, found {len(applicants)}"
        )

    labels = label_terminal_ssdi_award(applicants)
    clean = pd.DataFrame(index=applicants.index)
    for feature in FEATURES:
        clean[feature.clean_name] = pd.to_numeric(
            applicants[feature.source_name], errors="raise"
        ).astype("Int64")
    clean[TARGET_NAME] = labels
    assert_no_forbidden_columns(clean.columns)
    return clean.reset_index(drop=True)


def write_dataset(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def report(frame: pd.DataFrame) -> None:
    """Print row count, class balance, and feature missingness."""
    target = frame[TARGET_NAME]
    positives = int(target.eq(1).sum())
    negatives = int(target.eq(0).sum())
    feature_names = [feature.clean_name for feature in FEATURES]
    missing_pct = frame[feature_names].isna().mean().mul(100)

    print(f"rows {len(frame)}")
    print(f"terminal_ssdi_award 1 {positives} ({positives / len(frame):.1%})")
    print(f"terminal_ssdi_award 0 {negatives} ({negatives / len(frame):.1%})")
    print("missing_pct")
    for name, pct in missing_pct.items():
        print(f"  {name} {pct:.1f}")


def main() -> None:
    source = load_source(RAW_DTA)
    clean = build_analysis_frame(source)
    write_dataset(clean, OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")
    report(clean)


if __name__ == "__main__":
    main()
