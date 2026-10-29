"""Score one SSDI applicant with the calibrated LightGBM model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import sys
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR / "data"))
sys.path.insert(0, str(SRC_DIR / "features"))
sys.path.insert(0, str(SRC_DIR / "models"))

from build_features import AGE_CAP, DIAGNOSIS_LEVELS
from train import design_matrix
from train_final import FINAL_MODEL_PATH

FAVORABLE_PROBABILITY = 0.60
BORDERLINE_PROBABILITY = 0.40
PROBABILITY_FLOOR = 0.0001
PROBABILITY_CEILING = 0.9999

SEX_CODES = {"male": 1, "m": 1, "1": 1, "female": 2, "f": 2, "2": 2}
HEALTH_CODES = {
    "excellent": 1,
    "very_good": 2,
    "very good": 2,
    "good": 3,
    "fair": 4,
    "poor": 5,
}
CONDITION_GROUPS = set(DIAGNOSIS_LEVELS) | {"unreported"}
FEATURE_LABELS = {
    "sex": "sex",
    "adjudication_age": "age",
    "education_level": "education",
    "general_health": "general health",
    "severe_ambulation_limit": "unable to walk a quarter mile",
    "exertional_sedentary_only": "sedentary exertional limit",
    "exertional_light_max": "light exertional limit",
    "exertional_medium_or_above": "medium or heavier work capacity",
    "diagnosis_musculoskeletal": "musculoskeletal condition",
    "diagnosis_circulatory": "circulatory condition",
    "diagnosis_mental": "mental condition",
    "diagnosis_neurological": "neurological condition",
    "diagnosis_respiratory": "respiratory condition",
    "diagnosis_endocrine": "endocrine condition",
    "diagnosis_sensory": "sensory condition",
    "diagnosis_cancer": "cancer",
    "diagnosis_other": "other condition",
    "diagnosis_none": "no reported condition",
}


class SSDIClaimInput(BaseModel):
    """Disability-report fields accepted by the serving model.

    Functional items and the primary condition may be omitted. Omitted lift and
    walking flags are treated as no difficulty, omitted health as good, and an
    omitted condition as unreported.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(
        ...,
        ge=18,
        le=66,
        description="Age in years. Applicants age 60 through 66 are scored as 59.",
    )
    education_level: int = Field(
        ...,
        ge=31,
        le=46,
        description="Census education code. 31 is less than first grade and 46 is a doctorate.",
    )
    sex: int = Field(..., description="Sex. 1 = male, 2 = female.")
    general_health: int = Field(
        3,
        ge=1,
        le=5,
        description="Self-rated health. 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor.",
    )
    lift_10lb_diff: bool = Field(
        False,
        description="True when the person has difficulty lifting or carrying 10 pounds.",
    )
    lift_25lb_diff: bool = Field(
        False,
        description="True when the person has difficulty lifting or carrying 25 pounds.",
    )
    walk_quarter_mile_diff: bool = Field(
        False,
        description="True when the person has difficulty walking a quarter mile.",
    )
    primary_condition: str = Field(
        "unreported",
        description="Broad diagnostic group, or unreported when the primary condition is blank.",
    )

    @field_validator("sex", mode="before")
    @classmethod
    def _normalize_sex(cls, value: object) -> int:
        if isinstance(value, str):
            key = value.strip().lower()
            if key not in SEX_CODES:
                raise ValueError("sex must be 1, 2, male, or female")
            return SEX_CODES[key]
        return int(value)

    @field_validator("general_health", mode="before")
    @classmethod
    def _normalize_health(cls, value: object) -> int:
        if isinstance(value, str):
            key = value.strip().lower()
            if key not in HEALTH_CODES:
                raise ValueError("general_health must be 1-5 or excellent, very good, good, fair, or poor")
            return HEALTH_CODES[key]
        return int(value)

    @field_validator("primary_condition", mode="before")
    @classmethod
    def _normalize_condition(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("primary_condition must be a diagnostic group name")
        group = value.strip().lower()
        if group not in CONDITION_GROUPS:
            allowed = ", ".join(sorted(CONDITION_GROUPS))
            raise ValueError(f"primary_condition must be one of: {allowed}")
        return group

    @computed_field
    @property
    def adjudication_age(self) -> int:
        """Age presented to the model after the retirement-age cap."""
        return min(self.age, AGE_CAP)


class DrivingFactor(BaseModel):
    """One local SHAP contribution for the scored applicant."""

    feature: str
    direction: Literal["helps", "hurts"]
    shap: float


class SSDIClaimOutput(BaseModel):
    """Calibrated award score and the three strongest local factors."""

    probability_awarded: float = Field(..., gt=0.0, lt=1.0)
    risk_tier: Literal["Favorable", "Borderline", "High Risk of Denial"]
    driving_factors: list[DrivingFactor] = Field(..., min_length=3, max_length=3)


def exertional_tier(lift_10lb_diff: bool, lift_25lb_diff: bool) -> str:
    """Map the two lift-difficulty flags onto the training exertional tiers."""
    if lift_10lb_diff:
        return "sedentary_only"
    if lift_25lb_diff:
        return "light_max"
    return "medium_or_above"


def applicant_frame(claim: SSDIClaimInput) -> pd.DataFrame:
    """Map a validated claim onto one pre-dummy model row."""
    return pd.DataFrame(
        [
            {
                "sex": claim.sex,
                "adjudication_age": claim.adjudication_age,
                "education_level": claim.education_level,
                "general_health": claim.general_health,
                "severe_ambulation_limit": int(claim.walk_quarter_mile_diff),
                "exertional_tier": exertional_tier(claim.lift_10lb_diff, claim.lift_25lb_diff),
                "diagnosis_group": claim.primary_condition,
            }
        ]
    )


def risk_tier(probability: float) -> Literal["Favorable", "Borderline", "High Risk of Denial"]:
    """Band the calibrated award probability for a reviewer."""
    if probability >= FAVORABLE_PROBABILITY:
        return "Favorable"
    if probability >= BORDERLINE_PROBABILITY:
        return "Borderline"
    return "High Risk of Denial"


def _factor_name(feature: str, value: float) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    if feature == "sex":
        return "male" if value == 1 else "female"
    binary = feature.startswith(("exertional_", "diagnosis_")) or feature == "severe_ambulation_limit"
    if binary and value == 0:
        return f"no {label}"
    return label


def driving_factors(contributions: pd.Series, values: pd.Series) -> list[DrivingFactor]:
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    factors: list[DrivingFactor] = []
    seen: set[str] = set()
    for feature, shap_value in ranked.items():
        name = _factor_name(str(feature), float(values[feature]))
        if name in seen:
            continue
        seen.add(name)
        factors.append(
            DrivingFactor(
                feature=name,
                direction="helps" if float(shap_value) > 0 else "hurts",
                shap=round(float(shap_value), 4),
            )
        )
        if len(factors) == 3:
            break
    if len(factors) != 3:
        raise RuntimeError("Expected three distinct SHAP factors")
    return factors


def _coerce_input(applicant_data: SSDIClaimInput | dict[str, object]) -> SSDIClaimInput:
    if isinstance(applicant_data, SSDIClaimInput):
        return applicant_data
    if isinstance(applicant_data, dict):
        return SSDIClaimInput.model_validate(applicant_data)
    raise TypeError("applicant_data must be a dict or SSDIClaimInput")


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, object]:
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the final model first. Missing {FINAL_MODEL_PATH}")
    return joblib.load(FINAL_MODEL_PATH)


def _bounded_probability(probability: float) -> float:
    bounded = min(max(round(float(probability), 4), PROBABILITY_FLOOR), PROBABILITY_CEILING)
    return float(bounded)


def local_contributions(model: object, matrix: pd.DataFrame) -> pd.Series:
    """TreeSHAP values from LightGBM, excluding the base-value column."""
    raw = model.booster_.predict(matrix, pred_contrib=True)
    values = np.asarray(raw)[0, :-1]
    if values.shape != (matrix.shape[1],):
        raise RuntimeError(f"Expected {matrix.shape[1]} SHAP values, received {values.shape}")
    return pd.Series(values, index=matrix.columns)


def predict_claim(applicant_data: SSDIClaimInput | dict[str, object]) -> SSDIClaimOutput:
    """Return a calibrated award probability, risk tier, and the top local factors."""
    claim = _coerce_input(applicant_data)
    artifact = load_artifact()
    matrix = design_matrix(applicant_frame(claim))
    expected = list(artifact["feature_columns"])
    if list(matrix.columns) != expected:
        raise AssertionError("Inference features do not match the saved model")

    raw_probability = float(artifact["model"].predict_proba(matrix)[0, 1])
    probability = float(artifact["calibrator"].predict(np.array([raw_probability]))[0])
    contributions = local_contributions(artifact["model"], matrix)
    return SSDIClaimOutput(
        probability_awarded=_bounded_probability(probability),
        risk_tier=risk_tier(probability),
        driving_factors=driving_factors(contributions, matrix.iloc[0]),
    )


def main() -> None:
    cases = {
        "25-year-old sedentary worker": {
            "age": 25,
            "education_level": 39,
            "sex": "male",
            "general_health": "good",
            "lift_10lb_diff": True,
            "lift_25lb_diff": True,
            "walk_quarter_mile_diff": False,
            "primary_condition": "musculoskeletal",
        },
        "58-year-old limited-education laborer": {
            "age": 58,
            "education_level": 34,
            "sex": "male",
            "general_health": "poor",
            "lift_10lb_diff": True,
            "lift_25lb_diff": True,
            "walk_quarter_mile_diff": True,
            "primary_condition": "musculoskeletal",
        },
    }
    for name, applicant in cases.items():
        print(name)
        print(json.dumps(predict_claim(applicant).model_dump(), indent=2))


if __name__ == "__main__":
    main()
