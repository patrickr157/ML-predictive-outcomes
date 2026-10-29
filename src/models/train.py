"""Five-fold LightGBM baseline for terminal SSDI award.

Features are continuous adjudication age, demographics, education, exertional
tier, ambulation, general health, and broad diagnostic group. One-hot age bins
are omitted because they duplicated ``adjudication_age``.
"""

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR / "data"))
sys.path.insert(0, str(SRC_DIR / "features"))
sys.path.insert(0, str(SRC_DIR / "models"))

from baseline_grid import FEATURES_PATH, grid_decision
from build_features import DIAGNOSIS_LEVELS
from schema import EXPECTED_APPLICANT_ROWS, TARGET_NAME, assert_no_forbidden_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = PROJECT_ROOT / "data" / "processed" / "ssdi_lgbm_oof.parquet"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "lgbm_fold_models.joblib"
PR_CURVE_PATH = PROJECT_ROOT / "data" / "processed" / "lgbm_precision_recall.png"

RANDOM_STATE = 42
N_SPLITS = 5
DECISION_THRESHOLD = 0.5
PREVIOUS_ROC_AUC = 0.5966
PREVIOUS_PR_AUC = 0.5607

IDENTIFIER_COLUMNS = ("ssuid", "pnum", "adjudication_age_source")
EXERTIONAL_LEVELS = ("sedentary_only", "light_max", "medium_or_above")
NUMERIC_FEATURES = (
    "sex",
    "adjudication_age",
    "education_level",
    "general_health",
    "severe_ambulation_limit",
)


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    features = pd.read_parquet(path)
    if len(features) != EXPECTED_APPLICANT_ROWS:
        raise AssertionError(f"Expected {EXPECTED_APPLICANT_ROWS} rows, found {len(features)}")
    if TARGET_NAME not in features.columns:
        raise KeyError(f"Missing target column {TARGET_NAME}")
    return features


def design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix from the approved pre-adjudication columns.

    ``age_at_interview`` is not a separate input. It is already the fallback
    inside ``adjudication_age``. Diagnostic group ``unreported`` is the reference
    level, so its indicator is left out of the matrix.
    """
    required = (*NUMERIC_FEATURES, "exertional_tier", "diagnosis_group")
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise KeyError(f"Feature columns missing from the extract: {missing}")

    unknown_groups = set(features["diagnosis_group"].unique()) - set(DIAGNOSIS_LEVELS) - {"unreported"}
    if unknown_groups:
        raise AssertionError(f"Unexpected diagnosis groups: {sorted(unknown_groups)}")

    matrix = features.loc[:, list(NUMERIC_FEATURES)].astype("int64").copy()
    for level in EXERTIONAL_LEVELS:
        matrix[f"exertional_{level}"] = features["exertional_tier"].eq(level).astype("int64")
    for level in DIAGNOSIS_LEVELS:
        matrix[f"diagnosis_{level}"] = features["diagnosis_group"].eq(level).astype("int64")

    leaked = set(matrix.columns) & set(IDENTIFIER_COLUMNS)
    if leaked:
        raise AssertionError(f"Identifier columns entered the model matrix: {sorted(leaked)}")
    assert_no_forbidden_columns(matrix.columns)
    if matrix.isna().any().any():
        raise ValueError("Model matrix contains missing values")
    return matrix


def make_estimator() -> LGBMClassifier:
    """Shallow boosted trees sized for about 3,000 applicants."""
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        objective="binary",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )


def fit_out_of_fold(matrix: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return OOF probabilities and the fitted fold models."""
    probabilities = np.full(len(matrix), np.nan, dtype="float64")
    folds: list[dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    labels = target.astype("int64").to_numpy()

    for fold_id, (train_index, test_index) in enumerate(splitter.split(matrix, labels), start=1):
        model = make_estimator()
        model.fit(matrix.iloc[train_index], labels[train_index])
        probabilities[test_index] = model.predict_proba(matrix.iloc[test_index])[:, 1]
        folds.append({"fold": fold_id, "model": model, "test_index": test_index})

    if not np.isfinite(probabilities).all():
        raise AssertionError("Out-of-fold probabilities were not produced for every row")
    return probabilities, folds


def classification_metrics(target: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= DECISION_THRESHOLD).astype("int64")
    labels = target.astype("int64")
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
    }


def operating_point(
    target: pd.Series,
    probabilities: np.ndarray,
    *,
    minimum_recall: float | None = None,
    minimum_precision: float | None = None,
) -> dict[str, float] | None:
    """Pick the threshold that best meets a recall floor or a precision floor."""
    precision, recall, thresholds = precision_recall_curve(target.astype("int64"), probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    if minimum_recall is not None:
        eligible = np.flatnonzero(recall >= minimum_recall)
        if eligible.size == 0:
            return None
        chosen = int(eligible[np.argmax(precision[eligible])])
    else:
        eligible = np.flatnonzero(precision >= minimum_precision)
        if eligible.size == 0:
            return None
        chosen = int(eligible[np.argmax(recall[eligible])])
    return {
        "threshold": float(thresholds[chosen]),
        "precision": float(precision[chosen]),
        "recall": float(recall[chosen]),
    }


def print_threshold_tradeoffs(target: pd.Series, probabilities: np.ndarray) -> None:
    at_recall = operating_point(target, probabilities, minimum_recall=0.80)
    at_precision = operating_point(target, probabilities, minimum_precision=0.75)
    print("threshold_tradeoff")
    print(f"{'target':<28} {'threshold':>10} {'precision':>10} {'recall':>8}")
    if at_recall is None:
        print(f"{'precision at 80% recall':<28} {'not reached':>10}")
    else:
        print(
            f"{'precision at 80% recall':<28} {at_recall['threshold']:10.4f} "
            f"{at_recall['precision']:10.4f} {at_recall['recall']:8.4f}"
        )
    if at_precision is None:
        print(f"{'recall at 75% precision':<28} {'not reached':>10}")
    else:
        print(
            f"{'recall at 75% precision':<28} {at_precision['threshold']:10.4f} "
            f"{at_precision['precision']:10.4f} {at_precision['recall']:8.4f}"
        )
    save_precision_recall_plot(target, probabilities, at_recall, at_precision)


def save_precision_recall_plot(
    target: pd.Series,
    probabilities: np.ndarray,
    at_recall: dict[str, float] | None,
    at_precision: dict[str, float] | None,
) -> None:
    precision, recall, _thresholds = precision_recall_curve(target.astype("int64"), probabilities)
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(recall, precision, color="#1f4e79", linewidth=2)
    if at_recall is not None:
        axis.scatter(at_recall["recall"], at_recall["precision"], color="#c45911", zorder=3)
        axis.annotate("80% recall", (at_recall["recall"], at_recall["precision"]), textcoords="offset points", xytext=(8, -12))
    if at_precision is not None:
        axis.scatter(at_precision["recall"], at_precision["precision"], color="#548235", zorder=3)
        axis.annotate("75% precision", (at_precision["recall"], at_precision["precision"]), textcoords="offset points", xytext=(8, 8))
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("LightGBM out-of-fold precision-recall")
    figure.tight_layout()
    figure.savefig(PR_CURVE_PATH, dpi=140)
    plt.close(figure)
    print(f"wrote {PR_CURVE_PATH}")


def print_comparison(target: pd.Series, grid_probability: np.ndarray, model_probability: np.ndarray) -> None:
    rows = {
        "Grid baseline": classification_metrics(target, grid_probability),
        "LightGBM OOF": classification_metrics(target, model_probability),
    }
    header = f"{'model':<16} {'roc_auc':>8} {'pr_auc':>8} {'brier':>8} {'precision':>10} {'recall':>8} {'f1':>8}"
    print(header)
    for name, metrics in rows.items():
        print(
            f"{name:<16} {metrics['roc_auc']:8.4f} {metrics['pr_auc']:8.4f} "
            f"{metrics['brier']:8.4f} {metrics['precision']:10.4f} {metrics['recall']:8.4f} {metrics['f1']:8.4f}"
        )

    labels = target.astype("int64")
    for name, probabilities in (("Grid baseline", grid_probability), ("LightGBM OOF", model_probability)):
        predicted = (probabilities >= DECISION_THRESHOLD).astype("int64")
        matrix = confusion_matrix(labels, predicted, labels=[0, 1])
        print(f"confusion_matrix {name} rows=actual cols=predicted labels=0,1")
        print(matrix)


def main() -> None:
    features = load_features()
    matrix = design_matrix(features)
    target = features[TARGET_NAME]
    probabilities, folds = fit_out_of_fold(matrix, target)

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            TARGET_NAME: target.astype("int8").to_numpy(),
            "oof_probability": probabilities,
        }
    ).to_parquet(OOF_PATH, index=False)
    joblib.dump({"feature_columns": list(matrix.columns), "folds": folds}, MODEL_PATH)

    grid_probability = grid_decision(features)["prediction"].astype("float64").to_numpy()
    model_metrics = classification_metrics(target, probabilities)
    print(f"rows {len(features)}")
    print(f"features {list(matrix.columns)}")
    print(f"wrote {OOF_PATH}")
    print(
        f"previous_iteration roc_auc {PREVIOUS_ROC_AUC:.4f} pr_auc {PREVIOUS_PR_AUC:.4f}"
    )
    print(
        f"change roc_auc {model_metrics['roc_auc'] - PREVIOUS_ROC_AUC:+.4f} "
        f"pr_auc {model_metrics['pr_auc'] - PREVIOUS_PR_AUC:+.4f}"
    )
    print_comparison(target, grid_probability, probabilities)
    print_threshold_tradeoffs(target, probabilities)


if __name__ == "__main__":
    main()
