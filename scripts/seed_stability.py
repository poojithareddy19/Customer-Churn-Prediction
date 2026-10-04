"""Re-run split, calibration and evaluation of the selected configuration under several seeds.

The hyperparameters of the deployed model are read from artifacts/training_report.json and kept fixed;
only the data split and the model's random_state change. The deployed artifact is not modified.

Usage: python scripts/seed_stability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import load_dataset, split_features_target  # noqa: E402
from src.train import REPORT_PATH, build_selected_pipeline, calibrate_and_evaluate, split_data  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
OUTPUT_PATH = ROOT / "reports" / "seed_stability.csv"


def _recall(metrics: dict) -> float:
    positives = metrics["true_positives"] + metrics["false_negatives"]
    return metrics["true_positives"] / positives if positives else 0.0


def run(seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    model_name = report["best_model_name"]
    best_params = next(row["best_params"] for row in report["tuned_models"] if row["model"] == model_name)
    features, target = split_features_target(load_dataset())

    rows = []
    for seed in seeds:
        splits = split_data(features, target, seed)
        positive_ratio = float((splits.train_target == 0).sum() / max((splits.train_target == 1).sum(), 1))
        pipeline = build_selected_pipeline(model_name, best_params, seed, positive_ratio)
        evaluation = calibrate_and_evaluate(pipeline, splits)
        rows.append(
            {
                "seed": str(seed),
                "model": model_name,
                "threshold": evaluation.threshold["threshold"],
                "test_roc_auc": evaluation.test_metrics["roc_auc"],
                "test_brier": evaluation.test_metrics["brier_score"],
                "test_recall": _recall(evaluation.test_metrics),
            }
        )
        print(f"seed {seed}: ROC-AUC {rows[-1]['test_roc_auc']:.4f}, Brier {rows[-1]['test_brier']:.4f}, recall {rows[-1]['test_recall']:.4f}")

    frame = pd.DataFrame(rows)
    metric_columns = ["threshold", "test_roc_auc", "test_brier", "test_recall"]
    summary = [
        {"seed": "mean", "model": model_name, **frame[metric_columns].mean().to_dict()},
        {"seed": "std", "model": model_name, **frame[metric_columns].std(ddof=1).to_dict()},
    ]
    return pd.concat([frame, pd.DataFrame(summary)], ignore_index=True)


if __name__ == "__main__":
    result = run()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, float_format="%.6f")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
