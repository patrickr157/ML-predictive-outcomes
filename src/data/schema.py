"""Column contract for the 2014 SIPP SSA Supplement applicant extract.

Names in ``pu2014ssa.dta`` are lowercase. The Census metadata report prints the
same names in uppercase. Value labels are not stored in the Stata file; the
codes below are from ``data/raw/2014SSA_Metadata_AllSections.pdf``.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """One pre-adjudication survey item and its analysis name."""

    source_name: str
    clean_name: str
    description: str


# No TDIS* or TAPP* column exists on pu2014ssa.dta, so age at SSDI application
# cannot be derived. TONSET_AGE, TLIMWRKAGE, TAGELASTW, and TPRVWRKAGE are ages
# at condition onset or work limitation, not at application. TAGE_S is age at
# the September-November 2014 supplement interview and is only an interview-age proxy.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        source_name="tage_s",
        clean_name="age_at_interview",
        description=(
            "Interview-age proxy from TAGE_S. The file has no SSDI application "
            "year, so age at application cannot be derived."
        ),
    ),
    FeatureSpec(
        source_name="eeduc",
        clean_name="education_level",
        description="Highest degree or grade completed. Census codes 31-46.",
    ),
    FeatureSpec(
        source_name="esex_s",
        clean_name="sex",
        description="Sex. 1 = Male, 2 = Female.",
    ),
    FeatureSpec(
        source_name="ehlthstat",
        clean_name="general_health",
        description="General health. 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor.",
    ),
    FeatureSpec(
        source_name="edif10",
        clean_name="difficulty_lift_10lb",
        description="Difficulty lifting or carrying 10 pounds. 1 = Yes, 2 = No.",
    ),
    FeatureSpec(
        source_name="ecant10",
        clean_name="able_lift_10lb",
        description=(
            "Able to lift or carry 10 pounds at all. 1 = Yes, 2 = No. "
            "The instrument asks this only when EDIF10 = 1. In the public file, "
            "people who report no difficulty are coded 1."
        ),
    ),
    FeatureSpec(
        source_name="edif25",
        clean_name="difficulty_lift_25lb",
        description="Difficulty lifting or carrying 25 pounds. 1 = Yes, 2 = No.",
    ),
    FeatureSpec(
        source_name="ecant25",
        clean_name="able_lift_25lb",
        description=(
            "Able to lift or carry 25 pounds at all. 1 = Yes, 2 = No. "
            "Instrument universe: ECANT10 = 1 or EDIF25 = 1."
        ),
    ),
    FeatureSpec(
        source_name="ewalkd",
        clean_name="difficulty_walk_quarter_mile",
        description="Difficulty walking a quarter mile. 1 = Yes, 2 = No.",
    ),
    FeatureSpec(
        source_name="ewalk_c",
        clean_name="able_walk_quarter_mile",
        description=(
            "Able to walk a quarter mile at all. 1 = Yes, 2 = No. "
            "The instrument asks this only when EWALKD = 1. In the public file, "
            "people who report no difficulty are coded 1."
        ),
    ),
    FeatureSpec(
        source_name="esitd",
        clean_name="difficulty_sit_one_hour",
        description="Difficulty sitting for one hour. 1 = Yes, 2 = No.",
    ),
    FeatureSpec(
        source_name="estandd",
        clean_name="difficulty_stand_one_hour",
        description="Difficulty standing or being on feet for one hour. 1 = Yes, 2 = No.",
    ),
)

SOURCE_TO_CLEAN_MAP: dict[str, str] = {
    feature.source_name: feature.clean_name for feature in FEATURES
}

TARGET_NAME = "terminal_ssdi_award"
APPLICANT_SOURCE = "eapplyss"
SSDI_RECEIPT_SOURCE = "edinc_typ1"
ANY_DISABILITY_INCOME_SOURCE = "edinc"
SSI_RECEIPT_SOURCE = "edinc_typ2"
EXPECTED_APPLICANT_ROWS = 3011

# Post-determination receipt and monthly income or transfer payments.
# Median and standard-deviation companions repeat the same payment amounts.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "edinc",
        "edinc_typ1",
        "edinc_typ2",
        "edinc_typ3",
        "edinc_typ4",
        "edinc_typ5",
        "edinc_typ6",
        "e12m_dinc",
        "tpenamt1",
        "tpenamt1_med",
        "tpenamt1_std",
        "tpensamt",
        "tpensamt_med",
        "tpensamt_std",
        "tpayamt",
        "tpayamt_med",
        "tpayamt_std",
        "elumpsum",
        "eseverance",
        "adinc",
        "adinc_typ1",
        "adinc_typ2",
        "adinc_typ3",
        "adinc_typ4",
        "adinc_typ5",
        "adinc_typ6",
        "a12m_dinc",
        "apenamt1",
        "apensamt",
        "apayamt",
        "alumpsum",
        "aseverance",
    }
)


def assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    """Raise if a post-determination or payment field is in the feature matrix."""
    forbidden = {name.lower() for name in FORBIDDEN_COLUMNS}
    present = sorted({column.lower() for column in columns} & forbidden)
    if present:
        joined = ", ".join(present)
        raise AssertionError(f"Forbidden columns in feature matrix: {joined}")
