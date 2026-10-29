"""Serving checks for the calibrated SSDI claim scorer."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "data"))

from predict import SSDIClaimInput, applicant_frame, predict_claim

SAMPLE = {
    "age": 58,
    "education_level": 34,
    "sex": "male",
    "general_health": "poor",
    "lift_10lb_diff": True,
    "walk_quarter_mile_diff": True,
    "primary_condition": "musculoskeletal",
}


def test_probability_is_strictly_inside_unit_interval() -> None:
    result = predict_claim(SAMPLE)
    assert 0.0 < result.probability_awarded < 1.0


def test_shap_returns_three_distinct_features() -> None:
    factors = predict_claim(SSDIClaimInput.model_validate(SAMPLE)).driving_factors
    names = [factor.feature for factor in factors]
    assert len(names) == 3
    assert len(set(names)) == 3


def test_age_over_60_is_capped_at_59() -> None:
    claim = SSDIClaimInput(age=65, education_level=39, sex=1)
    assert claim.adjudication_age == 59
    scored_age = int(applicant_frame(claim)["adjudication_age"].iloc[0])
    assert scored_age == 59
    assert 0.0 < predict_claim(claim).probability_awarded < 1.0


def test_model_execution_completes_within_100ms() -> None:
    predict_claim(SAMPLE)
    started = time.perf_counter()
    predict_claim(SAMPLE)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.1
