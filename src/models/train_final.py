"""Fit the deployable LightGBM model and an isotonic probability map.

The booster is fit on every applicant. The calibrator is fit on out-of-fold
probabilities from the same hyperparameters, so the saved probabilities are
not graded on in-sample scores.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR / "models"))

from train import design_matrix, fit_out_of_fold, load_features, make_estimator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "ssdi_lightgbm_v1.joblib"


def fit_final_model() -> dict[str, object]:
    features = load_features()
    matrix = design_matrix(features)
    target = features["terminal_ssdi_award"].astype("int64")
    labels = target.to_numpy()

    oof_probabilities, _folds = fit_out_of_fold(matrix, target)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof_probabilities, labels)
    calibrated_oof = calibrator.predict(oof_probabilities)

    model = make_estimator()
    model.fit(matrix, labels)
    print(f"oof_brier_raw {brier_score_loss(labels, oof_probabilities):.4f}")
    print(f"oof_brier_calibrated {brier_score_loss(labels, calibrated_oof):.4f}")

    return {
        "model": model,
        "calibrator": calibrator,
        "feature_columns": list(matrix.columns),
    }


def main() -> None:
    payload = fit_final_model()
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, FINAL_MODEL_PATH)
    print(f"wrote {FINAL_MODEL_PATH}")
    print(f"features {payload['feature_columns']}")


if __name__ == "__main__":
    main()
