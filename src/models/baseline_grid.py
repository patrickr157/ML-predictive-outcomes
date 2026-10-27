"""Deterministic SSA medical-vocational grid baseline.

This is a reduced grid. The supplement has no SVP or transferable-skills
items, so the rules use age, sedentary restriction, high-school education,
general health, and the quarter-mile walking limit.

Census ``EEDUC`` code 39 is a high-school diploma or GED. Codes 31-39 are
high school or less. General health codes 4 and 5 are fair and poor.

Rule 1 runs before Rule 2, and both override Rule 3.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ssdi_features.parquet"
TARGET_NAME = "terminal_ssdi_award"
HIGH_SCHOOL_OR_LESS = 39
FAIR_OR_POOR_HEALTH = 4


def grid_decision(features: pd.DataFrame) -> pd.DataFrame:
    """Return hard 0/1 predictions and the rule that produced each one."""
    age = features["adjudication_age"]
    sedentary = features["exertional_tier"].eq("sedentary_only")
    high_school_or_less = features["education_level"].le(HIGH_SCHOOL_OR_LESS)
    fair_or_poor = features["general_health"].ge(FAIR_OR_POOR_HEALTH)
    severe_walk = features["severe_ambulation_limit"].eq(1)

    prediction = pd.Series(0, index=features.index, dtype="int8")
    rule = pd.Series("rule_3_denial", index=features.index, dtype="object")

    # Rule 3: residual cases. Fair or poor health plus a severe exertional
    # or walking limit is an allowance. Every other residual case is a denial.
    rule_3_allowance = fair_or_poor & (sedentary | severe_walk)
    prediction = prediction.mask(rule_3_allowance, 1)
    rule = rule.mask(rule_3_allowance, "rule_3_allowance")

    # Rule 1: a younger worker who is not limited to sedentary work is denied.
    rule_1 = age.lt(50) & ~sedentary
    prediction = prediction.mask(rule_1, 0)
    rule = rule.mask(rule_1, "rule_1_younger_denial")

    # Rule 2: advanced age, sedentary restriction, and high school or less.
    rule_2 = age.ge(55) & sedentary & high_school_or_less
    prediction = prediction.mask(rule_2, 1)
    rule = rule.mask(rule_2, "rule_2_advanced_age_allowance")

    return pd.DataFrame({"prediction": prediction, "rule": rule})


def evaluate(features: pd.DataFrame, decision: pd.DataFrame) -> None:
    y_true = features[TARGET_NAME].astype("int8")
    y_pred = decision["prediction"].astype("int8")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("confusion_matrix rows=actual cols=predicted labels=0,1")
    print(matrix)
    print(f"precision {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"recall {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"f1 {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"brier {brier_score_loss(y_true, y_pred.astype('float64')):.4f}")
    print("rule_counts")
    for name, count in decision["rule"].value_counts().items():
        print(f"  {name} {int(count)}")


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    decision = grid_decision(features)
    evaluate(features, decision)


if __name__ == "__main__":
    main()
