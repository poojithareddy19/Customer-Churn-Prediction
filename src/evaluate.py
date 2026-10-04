from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

from .features import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST, readable_feature_name


DEFAULT_CAPACITIES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def cross_validate_roc_auc_scores(model, features, target, cv) -> np.ndarray:
    """Per-fold ROC-AUC scores."""
    return cross_val_score(model, features, target, cv=cv, scoring="roc_auc", n_jobs=-1)


def cross_validate_roc_auc(model, features, target, cv) -> float:
    return float(np.mean(cross_validate_roc_auc_scores(model, features, target, cv)))


def fold_score_summary(scores) -> dict[str, float | list[float]]:
    """Per-fold scores with their mean and sample standard deviation (ddof=1)."""
    values = np.asarray(scores, dtype=float)
    return {
        "fold_scores": [float(value) for value in values],
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


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
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
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


def _rank_order(probabilities: np.ndarray) -> np.ndarray:
    # Highest probability first; a stable sort keeps ties in their original order.
    return np.argsort(-np.asarray(probabilities, dtype=float), kind="mergesort")


def lift_table(target: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Lift and gains by equal-size bins of predicted probability, decile 1 being the highest risk."""
    target = np.asarray(target).astype(int)
    sorted_target = target[_rank_order(probabilities)]
    overall_rate = float(sorted_target.mean())
    total_churners = int(sorted_target.sum())

    rows = []
    cumulative_customers = 0
    cumulative_churners = 0
    for decile, chunk in enumerate(np.array_split(sorted_target, n_bins), start=1):
        customers = len(chunk)
        churners = int(chunk.sum())
        cumulative_customers += customers
        cumulative_churners += churners
        churn_rate = churners / customers if customers else 0.0
        cumulative_rate = cumulative_churners / cumulative_customers if cumulative_customers else 0.0
        rows.append(
            {
                "decile": decile,
                "customers": customers,
                "churners": churners,
                "churn_rate": float(churn_rate),
                "lift": float(churn_rate / overall_rate) if overall_rate else 0.0,
                "cumulative_churners": cumulative_churners,
                "cumulative_capture": float(cumulative_churners / total_churners) if total_churners else 0.0,
                "cumulative_lift": float(cumulative_rate / overall_rate) if overall_rate else 0.0,
            }
        )
    return pd.DataFrame(rows)


def capacity_table(
    target: np.ndarray,
    probabilities: np.ndarray,
    capacities: tuple[float, ...] = DEFAULT_CAPACITIES,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> pd.DataFrame:
    """Outcome of contacting the top share of customers ranked by predicted probability."""
    target = np.asarray(target).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    order = _rank_order(probabilities)
    sorted_target = target[order]
    sorted_probabilities = probabilities[order]
    total = len(target)
    total_churners = int(target.sum())

    rows = []
    for capacity in capacities:
        if not 0 < capacity <= 1:
            raise ValueError(f"capacity must be in (0, 1], got {capacity}")
        contacted = min(total, math.ceil(round(capacity * total, 9)))
        true_positives = int(sorted_target[:contacted].sum())
        false_positives = contacted - true_positives
        false_negatives = total_churners - true_positives
        rows.append(
            {
                "capacity": float(capacity),
                "customers_contacted": int(contacted),
                "probability_cutoff": float(sorted_probabilities[contacted - 1]),
                "recall": float(true_positives / total_churners) if total_churners else 0.0,
                "precision": float(true_positives / contacted),
                "expected_cost": float(false_positives * false_positive_cost + false_negatives * false_negative_cost),
            }
        )
    return pd.DataFrame(rows)


def _point_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    positives = target == 1
    true_positives = int((predictions & positives).sum())
    false_positives = int((predictions & ~positives).sum())
    false_negatives = int((~predictions & positives).sum())
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "recall": float(true_positives / (true_positives + false_negatives)) if true_positives + false_negatives else 0.0,
        "precision": float(true_positives / (true_positives + false_positives)) if true_positives + false_positives else 0.0,
        "expected_cost": float(false_positives * false_positive_cost + false_negatives * false_negative_cost),
        "top_decile_lift": float(lift_table(target, probabilities).loc[0, "lift"]),
    }


def bootstrap_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict:
    """Point estimates with 95% percentile bootstrap intervals; single-class resamples are skipped."""
    target = np.asarray(target).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    point = _point_metrics(target, probabilities, threshold, false_positive_cost, false_negative_cost)

    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {name: [] for name in point}
    skipped = 0
    for _ in range(n_boot):
        indices = rng.integers(0, len(target), len(target))
        resampled_target = target[indices]
        if resampled_target.min() == resampled_target.max():
            skipped += 1
            continue
        metrics = _point_metrics(resampled_target, probabilities[indices], threshold, false_positive_cost, false_negative_cost)
        for name, value in metrics.items():
            samples[name].append(value)

    return {
        "n_boot": n_boot,
        "n_skipped": skipped,
        "confidence_level": 0.95,
        "random_state": random_state,
        "threshold": float(threshold),
        "metrics": {
            name: {
                "estimate": point[name],
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
            for name, values in samples.items()
        },
    }


def logistic_odds_ratios(pipeline, features, target, top_n: int = 15) -> pd.DataFrame:
    """Fit a copy of a preprocess + logistic pipeline and return its largest odds ratios.

    Numeric inputs are standardised, so their odds ratio is per one standard deviation.
    """
    fitted = clone(pipeline).fit(features, target)
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    coefficients = fitted.named_steps["model"].coef_[0]
    frame = pd.DataFrame(
        {
            "feature": [readable_feature_name(name) for name in names],
            "log_odds": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )
    order = frame["log_odds"].abs().sort_values(ascending=False).index
    return frame.loc[order].head(top_n).reset_index(drop=True)


def profit_curve(
    target: np.ndarray,
    probabilities: np.ndarray,
    monthly_charges: np.ndarray,
    offer_cost: float,
    success_rate: float,
    months: float,
    steps: int = 101,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Expected retention profit when everyone at or above each threshold gets an offer.

    profit = sum over contacted churners of (success_rate * monthly_charges * months)
             - offer_cost * customers contacted
    Returns the curve and its profit-maximising row.
    """
    target = np.asarray(target).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    monthly_charges = np.asarray(monthly_charges, dtype=float)
    rows = []
    for threshold in np.linspace(0.0, 1.0, steps):
        contacted = probabilities >= threshold
        contacted_churners = contacted & (target == 1)
        saved_revenue = float(success_rate * months * monthly_charges[contacted_churners].sum())
        offer_spend = float(offer_cost * contacted.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "customers_contacted": int(contacted.sum()),
                "churners_contacted": int(contacted_churners.sum()),
                "expected_saved_revenue": saved_revenue,
                "offer_spend": offer_spend,
                "profit": saved_revenue - offer_spend,
            }
        )
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["profit"].idxmax()]
    return frame, {key: float(value) for key, value in best.items()}
