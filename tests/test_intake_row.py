"""The intake row records facts and refuses to score a partial screener."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "features"))

from intake_row import assess_intake

COMPLETE = {
    "date_of_birth": "1970-06-15",
    "date_disability_began": "2024-06-15",
    "highest_grade_completed": "12",
    "primary_conditions": ["musculoskeletal", "mental_health"],
    "applied_before": False,
    "can_lift_10lb": False,
    "can_lift_25lb": False,
    "can_walk_quarter_mile": False,
    "general_health": "poor",
}


def test_complete_screener_is_ready_and_has_no_score() -> None:
    result = assess_intake(COMPLETE)
    assert result.ready is True
    assert result.gaps == []
    assert result.features is not None
    assert result.features.onset_age == 54
    assert result.features.education_cut == "high_school_or_less"
    assert result.features.prior_application_outcome == "never"
    dumped = result.model_dump()
    assert "probability_awarded" not in dumped
    assert "probability_awarded" not in dumped["features"]


def test_missing_limits_and_education_return_gaps_and_no_row() -> None:
    result = assess_intake(
        {
            "date_of_birth": "1980-01-01",
            "date_disability_began": "2020-01-01",
            "primary_conditions": ["cancer"],
            "prior_application_outcome": "denied_initial",
        }
    )
    assert result.ready is False
    assert result.features is None
    assert result.gaps == [
        "education_cut",
        "can_lift_10lb",
        "can_lift_25lb",
        "can_walk_quarter_mile",
        "general_health",
    ]


def test_college_grade_is_more_than_high_school() -> None:
    payload = {**COMPLETE, "highest_grade_completed": "bachelors"}
    result = assess_intake(payload)
    assert result.features is not None
    assert result.features.education_cut == "more_than_high_school"


def test_onset_before_birth_is_a_gap() -> None:
    payload = {
        **COMPLETE,
        "date_of_birth": "1980-01-01",
        "date_disability_began": "1970-01-01",
    }
    result = assess_intake(payload)
    assert result.ready is False
    assert "onset_age" in result.gaps


def test_unknown_condition_is_a_gap() -> None:
    payload = {**COMPLETE, "primary_conditions": ["musculoskeletal", "not-a-condition"]}
    result = assess_intake(payload)
    assert result.features is None
    assert result.gaps == ["primary_conditions"]


def test_very_good_health_alias_is_accepted() -> None:
    payload = {**COMPLETE, "general_health": "very good"}
    result = assess_intake(payload)
    assert result.features is not None
    assert result.features.general_health == "very_good"


def test_prior_denial_is_kept_when_they_applied_before() -> None:
    payload = {
        **COMPLETE,
        "applied_before": True,
        "prior_application_outcome": "denied_alj",
    }
    result = assess_intake(payload)
    assert result.features is not None
    assert result.features.prior_application_outcome == "denied_alj"
