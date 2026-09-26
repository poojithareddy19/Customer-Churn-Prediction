from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

from .features import FALSE_NEGATIVE_COST_RUPEES, FALSE_POSITIVE_COST_RUPEES


def cross_validate_roc_auc(model, features, target, cv) -> float:
    scores = cross_val_score(model, features, target, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


def summarize_predictions(target: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | str]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions).ravel()
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "accuracy": float((predictions == target).mean()),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "classification_report": classification_report(target, predictions, zero_division=0),
    }


def expected_cost_curve(
    target: np.ndarray,
    probabilities: np.ndarray,
    false_positive_cost: float = FALSE_POSITIVE_COST_RUPEES,
    false_negative_cost: float = FALSE_NEGATIVE_COST_RUPEES,
    steps: int = 101,
) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, steps)
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, predictions).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "expected_cost": float(fp * false_positive_cost + fn * false_negative_cost),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            }
        )
    return pd.DataFrame(rows)


def best_cost_threshold(cost_frame: pd.DataFrame) -> dict[str, float]:
    best_row = cost_frame.loc[cost_frame["expected_cost"].idxmin()]
    return {
        "threshold": float(best_row["threshold"]),
        "expected_cost": float(best_row["expected_cost"]),
        "false_positives": float(best_row["false_positives"]),
        "false_negatives": float(best_row["false_negatives"]),
    }


def calibration_table(target: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    fraction_of_positives, mean_predicted_value = calibration_curve(target, probabilities, n_bins=bins, strategy="uniform")
    return pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted_value,
            "fraction_of_positives": fraction_of_positives,
        }
    )


def permutation_importance_table(model, features, target, feature_names: list[str], n_repeats: int = 10) -> pd.DataFrame:
    result = permutation_importance(model, features, target, scoring="roc_auc", n_repeats=n_repeats, random_state=42, n_jobs=-1)
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return frame.reset_index(drop=True)
