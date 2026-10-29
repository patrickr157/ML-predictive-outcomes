"""Freeze one SSDI intake into the facts a later model is allowed to see.

This module does not score. A partial screener returns the missing fact names
and no feature row. Callers must not fill those gaps with healthy defaults.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

EducationCut = Literal["high_school_or_less", "more_than_high_school"]
GeneralHealth = Literal["excellent", "very_good", "good", "fair", "poor"]
PriorOutcome = Literal[
    "denied_initial",
    "denied_recon",
    "denied_alj",
    "pending",
    "never",
]
ConditionCategory = Literal[
    "musculoskeletal",
    "mental_health",
    "cardiovascular",
    "neurological",
    "autoimmune",
    "cancer",
    "respiratory",
    "other",
]

MIN_ONSET_AGE = 18
MAX_ONSET_AGE = 120

CONDITION_CATEGORIES = set(get_args(ConditionCategory))
PRIOR_OUTCOMES = set(get_args(PriorOutcome))
HEALTH_VALUES = set(get_args(GeneralHealth))
EDUCATION_CUTS = set(get_args(EducationCut))

# Advocate and webapp grade values at or below a high-school diploma or GED.
HIGH_SCHOOL_OR_LESS_GRADES = {
    "none",
    "never_attended",
    "1-6",
    "1-8",
    "7-12_no_diploma",
    "9",
    "10",
    "11",
    "12",
    "hs_diploma_ged",
}
MORE_THAN_HIGH_SCHOOL_GRADES = {
    "some_college",
    "some-college",
    "associate",
    "associates",
    "bachelor_or_higher",
    "bachelors",
    "masters",
}

HEALTH_ALIASES = {
    "excellent": "excellent",
    "very_good": "very_good",
    "very good": "very_good",
    "good": "good",
    "fair": "fair",
    "poor": "poor",
}

GAP_FIELDS = (
    "onset_age",
    "education_cut",
    "primary_conditions",
    "prior_application_outcome",
    "can_lift_10lb",
    "can_lift_25lb",
    "can_walk_quarter_mile",
    "general_health",
)


class IntakeFacts(BaseModel):
    """Partial screener or claim fields. Missing answers stay missing."""

    model_config = ConfigDict(extra="ignore")

    date_of_birth: date | None = None
    date_disability_began: date | None = None
    onset_age: int | None = None
    education_cut: str | None = None
    highest_grade_completed: str | None = None
    education_level: int | None = None
    primary_conditions: list[str] | str | None = None
    applied_before: bool | None = None
    prior_application_outcome: str | None = None
    can_lift_10lb: bool | None = None
    can_lift_25lb: bool | None = None
    can_walk_quarter_mile: bool | None = None
    general_health: str | None = None

    @field_validator(
        "date_of_birth",
        "date_disability_began",
        mode="before",
    )
    @classmethod
    def _parse_date(cls, value: object) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return None
            return date.fromisoformat(text[:10])
        raise ValueError("dates must be ISO calendar dates")

    @field_validator(
        "applied_before",
        "can_lift_10lb",
        "can_lift_25lb",
        "can_walk_quarter_mile",
        mode="before",
    )
    @classmethod
    def _parse_bool(cls, value: object) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"true", "yes", "y", "1"}:
                return True
            if key in {"false", "no", "n", "0"}:
                return False
        raise ValueError("expected yes or no")

    @field_validator("onset_age", "education_level", mode="before")
    @classmethod
    def _parse_int(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("expected an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        raise ValueError("expected an integer")

    @field_validator(
        "education_cut",
        "highest_grade_completed",
        "prior_application_outcome",
        "general_health",
        mode="before",
    )
    @classmethod
    def _blank_string(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        raise ValueError("expected text")


class IntakeFeatureRow(BaseModel):
    """Complete pre-decision facts. This is not a score."""

    model_config = ConfigDict(extra="forbid")

    onset_age: int = Field(..., ge=MIN_ONSET_AGE, le=MAX_ONSET_AGE)
    education_cut: EducationCut
    primary_conditions: list[ConditionCategory] = Field(..., min_length=1)
    prior_application_outcome: PriorOutcome
    can_lift_10lb: bool
    can_lift_25lb: bool
    can_walk_quarter_mile: bool
    general_health: GeneralHealth


class IntakeReadiness(BaseModel):
    """Either a complete feature row or the names of the facts still missing."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    features: IntakeFeatureRow | None = None
    gaps: list[str]


def _completed_years(born: date, on: date) -> int | None:
    if on < born:
        return None
    years = on.year - born.year
    if (on.month, on.day) < (born.month, born.day):
        years -= 1
    return years


def _onset_age(facts: IntakeFacts) -> int | None:
    if facts.date_of_birth is not None and facts.date_disability_began is not None:
        years = _completed_years(facts.date_of_birth, facts.date_disability_began)
    else:
        years = facts.onset_age
    if years is None or years < MIN_ONSET_AGE or years > MAX_ONSET_AGE:
        return None
    return years


def _education_cut(facts: IntakeFacts) -> EducationCut | None:
    if facts.education_cut is not None:
        key = facts.education_cut.strip().lower().replace(" ", "_")
        if key in EDUCATION_CUTS:
            return key  # type: ignore[return-value]
        return None
    if facts.highest_grade_completed is not None:
        grade = facts.highest_grade_completed.strip().lower()
        if grade in HIGH_SCHOOL_OR_LESS_GRADES:
            return "high_school_or_less"
        if grade in MORE_THAN_HIGH_SCHOOL_GRADES:
            return "more_than_high_school"
        return None
    if facts.education_level is not None:
        if 31 <= facts.education_level <= 39:
            return "high_school_or_less"
        if 40 <= facts.education_level <= 46:
            return "more_than_high_school"
    return None


def _conditions(facts: IntakeFacts) -> list[ConditionCategory] | None:
    raw = facts.primary_conditions
    if raw is None:
        return None
    items = [raw] if isinstance(raw, str) else list(raw)
    cleaned: list[ConditionCategory] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            return None
        key = item.strip().lower()
        if key == "":
            continue
        if key not in CONDITION_CATEGORIES:
            return None
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)  # type: ignore[arg-type]
    if not cleaned:
        return None
    return cleaned


def _prior_outcome(facts: IntakeFacts) -> PriorOutcome | None:
    if facts.prior_application_outcome is not None:
        key = facts.prior_application_outcome.strip().lower()
        if key in PRIOR_OUTCOMES:
            return key  # type: ignore[return-value]
        return None
    if facts.applied_before is False:
        return "never"
    return None


def _health(facts: IntakeFacts) -> GeneralHealth | None:
    if facts.general_health is None:
        return None
    key = HEALTH_ALIASES.get(facts.general_health.strip().lower().replace("-", " "))
    if key is None:
        key = HEALTH_ALIASES.get(facts.general_health.strip().lower())
    if key in HEALTH_VALUES:
        return key  # type: ignore[return-value]
    return None


def assess_intake(payload: IntakeFacts | dict[str, object]) -> IntakeReadiness:
    """Return a feature row only when every required fact is present.

    The result has no award probability. ``predict_claim`` is intentionally
    not called.
    """
    facts = payload if isinstance(payload, IntakeFacts) else IntakeFacts.model_validate(payload)
    resolved = {
        "onset_age": _onset_age(facts),
        "education_cut": _education_cut(facts),
        "primary_conditions": _conditions(facts),
        "prior_application_outcome": _prior_outcome(facts),
        "can_lift_10lb": facts.can_lift_10lb,
        "can_lift_25lb": facts.can_lift_25lb,
        "can_walk_quarter_mile": facts.can_walk_quarter_mile,
        "general_health": _health(facts),
    }
    gaps = [name for name in GAP_FIELDS if resolved[name] is None]
    if gaps:
        return IntakeReadiness(ready=False, features=None, gaps=gaps)
    return IntakeReadiness(
        ready=True,
        features=IntakeFeatureRow.model_validate(resolved),
        gaps=[],
    )
