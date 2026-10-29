"""SHAP values for the out-of-fold LightGBM models.

Each applicant is explained by the fold model that did not train on that row.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import shap

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR / "models"))

from train import MODEL_PATH, design_matrix, load_features

DIRECTION_FEATURES = (
    "exertional_sedentary_only",
    "exertional_light_max",
    "exertional_medium_or_above",
)
PRIOR_AGE_60_PLUS_SHAP = -0.514448


def positive_class_shap(model: object, matrix: pd.DataFrame) -> np.ndarray:
    """Return one SHAP row per applicant for the allowance class."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(matrix)
    if hasattr(values, "values"):
        array = np.asarray(values.values)
    else:
        array = values
    if isinstance(array, list):
        array = np.asarray(array[-1])
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[:, :, -1]
    expected = (len(matrix), matrix.shape[1])
    if array.shape != expected:
        raise ValueError(f"SHAP shape {array.shape} does not match {expected}")
    return array


def stacked_shap(matrix: pd.DataFrame, payload: dict[str, object]) -> pd.DataFrame:
    saved_columns = list(payload["feature_columns"])
    if saved_columns != list(matrix.columns):
        raise AssertionError("Saved fold features do not match the current design matrix")

    values = np.full(matrix.shape, np.nan, dtype="float64")
    for fold in payload["folds"]:
        test_index = np.asarray(fold["test_index"])
        held_out = matrix.iloc[test_index]
        values[test_index, :] = positive_class_shap(fold["model"], held_out)

    if not np.isfinite(values).all():
        raise AssertionError("SHAP values were not produced for every out-of-fold row")
    return pd.DataFrame(values, columns=matrix.columns, index=matrix.index)


def report(features: pd.DataFrame, matrix: pd.DataFrame, values: pd.DataFrame) -> None:
    importance = values.abs().mean().sort_values(ascending=False)
    print("top_10_mean_abs_shap")
    for rank, (name, score) in enumerate(importance.head(10).items(), start=1):
        print(f"  {rank} {name} {score:.6f} mean_signed {values[name].mean():.6f}")

    age = features["adjudication_age"]
    bands = {
        "under_50": age.lt(50),
        "age_50_54": age.between(50, 54),
        "age_55_59": age.between(55, 59),
        "age_60_plus": age.ge(60),
    }
    print("mean_adjudication_age_shap_by_age")
    band_means: dict[str, float] = {}
    for name, active in bands.items():
        mean_age_shap = float(values.loc[active.to_numpy(), "adjudication_age"].mean())
        band_means[name] = mean_age_shap
        print(f"  {name} n {int(active.sum())} {mean_age_shap:.6f}")

    print("mean_shap_when_feature_is_active")
    for name in DIRECTION_FEATURES:
        active = matrix[name].eq(1)
        print(f"  {name} n {int(active.sum())} {float(values.loc[active, name].mean()):.6f}")

    age_60_shap = band_means["age_60_plus"]
    anomaly_resolved = age_60_shap >= -0.05
    print(f"prior_age_60_plus_shap {PRIOR_AGE_60_PLUS_SHAP:.6f}")
    print(f"age_60_plus_shap {age_60_shap:.6f}")
    print(f"age_60_plus_anomaly_resolved {anomaly_resolved}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Run src/models/train.py before SHAP. Missing {MODEL_PATH}")
    features = load_features()
    matrix = design_matrix(features)
    payload = joblib.load(MODEL_PATH)
    values = stacked_shap(matrix, payload)
    report(features, matrix, values)


if __name__ == "__main__":
    main()
