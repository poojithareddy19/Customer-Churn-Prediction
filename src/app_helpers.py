"""Helpers for app.py that do not depend on Streamlit, so they can be unit tested."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import original_feature_name


def report_cost(report: dict[str, Any], kind: str) -> float:
    """Read "false_positive" or "false_negative" cost; artifacts trained before the rename used *_cost_rupees."""
    if f"{kind}_cost" in report:
        return float(report[f"{kind}_cost"])
    return float(report[f"{kind}_cost_rupees"])


def profit_threshold(report: dict[str, Any]) -> float | None:
    """Validation-chosen profit-maximising threshold; None for artifacts trained before the profit analysis."""
    profit_analysis = report.get("profit_analysis")
    if profit_analysis is None:
        return None
    return float(profit_analysis["validation_threshold"])


def log_prediction(
    log_path: Path,
    inputs: dict[str, Any],
    probability: float,
    decision_rule: str,
    threshold: float,
    verdict: str,
) -> dict[str, Any]:
    """Append one prediction as a JSON line with a UTC timestamp and return the record."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "probability": float(probability),
        "decision_rule": decision_rule,
        "threshold": float(threshold),
        "verdict": verdict,
    }
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record


def explain_prediction(
    calibrated_model,
    input_frame: pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Top SHAP contributions for one customer, summed over each original feature's one-hot columns.

    Explains the first fitted pipeline inside the CalibratedClassifierCV, so the values describe the
    uncalibrated model score. Returns the table and the unit of the values. A background frame of raw
    features is required when the model is a logistic regression.
    """
    import shap

    pipeline = calibrated_model.calibrated_classifiers_[0].estimator
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocess.transform(input_frame)

    if isinstance(model, LogisticRegression):
        if background is None:
            raise ValueError("a background frame is needed to explain a logistic regression")
        explainer = shap.LinearExplainer(model, preprocess.transform(background))
        unit = "log-odds"
    else:
        explainer = shap.TreeExplainer(model)
        # Gradient boosting is explained in log-odds; sklearn forests are explained in probability.
        unit = "probability" if hasattr(model, "estimators_") else "log-odds"

    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:
        values = values[..., 1]
    contributions = pd.Series(values[0], index=preprocess.get_feature_names_out())

    by_feature = contributions.groupby([original_feature_name(name) for name in contributions.index]).sum()
    top = by_feature.reindex(by_feature.abs().sort_values(ascending=False).index).head(top_n)
    frame = pd.DataFrame(
        {
            "feature": [f"{feature} = {input_frame.iloc[0][feature]}" for feature in top.index],
            "contribution": top.to_numpy(),
            "direction": ["raises risk" if value > 0 else "lowers risk" for value in top.to_numpy()],
        }
    )
    return frame, unit
