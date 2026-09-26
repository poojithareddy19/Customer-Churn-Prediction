from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.evaluate import (
    best_cost_threshold,
    calibration_table,
    cross_validate_roc_auc,
    expected_cost_curve,
    permutation_importance_table,
    summarize_predictions,
)
from src.features import (
    DATASET_PATH,
    FALSE_NEGATIVE_COST_RUPEES,
    FALSE_POSITIVE_COST_RUPEES,
    FEATURE_COLUMNS,
    build_preprocessor,
    load_dataset,
    split_features_target,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
REPORT_PATH = ARTIFACT_DIR / "training_report.json"


@dataclass(frozen=True)
class TrainingBundle:
    model: Any
    threshold: float
    report: dict[str, Any]


def _candidate_pipelines(random_state: int, positive_ratio: float) -> dict[str, Any]:
    logistic = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    rf_balanced = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_smote = ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            (
                "model",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb_balanced = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    random_state=random_state,
                    n_estimators=250,
                    tree_method="hist",
                    eval_metric="auc",
                    n_jobs=-1,
                    scale_pos_weight=positive_ratio,
                ),
            ),
        ]
    )
    xgb_smote = ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            (
                "model",
                XGBClassifier(
                    random_state=random_state,
                    n_estimators=250,
                    tree_method="hist",
                    eval_metric="auc",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "logistic_balanced": logistic,
        "rf_balanced": rf_balanced,
        "rf_smote": rf_smote,
        "xgb_balanced": xgb_balanced,
        "xgb_smote": xgb_smote,
    }


def _rf_search_space() -> dict[str, list[Any]]:
    return {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
    }


def _rf_smote_search_space() -> dict[str, list[Any]]:
    space = _rf_search_space()
    space["smote__k_neighbors"] = [3, 5]
    return space


def _xgb_search_space() -> dict[str, list[Any]]:
    return {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    }


def _search_model(pipeline, search_space: dict[str, list[Any]], features, target, cv) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
        verbose=0,
    )
    search.fit(features, target)
    return search


def train_model(random_state: int = 42) -> TrainingBundle:
    frame = load_dataset(DATASET_PATH)
    features, target = split_features_target(frame)

    train_features, temp_features, train_target, temp_target = train_test_split(
        features,
        target,
        test_size=0.4,
        random_state=random_state,
        stratify=target,
    )
    validation_features, test_features, validation_target, test_target = train_test_split(
        temp_features,
        temp_target,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_target,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    positive_ratio = float((train_target == 0).sum() / max((train_target == 1).sum(), 1))
    candidates = _candidate_pipelines(random_state, positive_ratio)

    baseline_rows = []
    for candidate_name, candidate_pipeline in candidates.items():
        score = cross_validate_roc_auc(candidate_pipeline, train_features, train_target, cv=cv)
        baseline_rows.append({"model": candidate_name, "cv_roc_auc": score})

    baseline_frame = pd.DataFrame(baseline_rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    rf_search = _search_model(candidates["rf_balanced"], _rf_search_space(), train_features, train_target, cv)
    rf_smote_search = _search_model(candidates["rf_smote"], _rf_smote_search_space(), train_features, train_target, cv)
    xgb_search = _search_model(candidates["xgb_balanced"], _xgb_search_space(), train_features, train_target, cv)
    xgb_smote_search = _search_model(candidates["xgb_smote"], _xgb_search_space(), train_features, train_target, cv)

    tuned_frame = pd.DataFrame(
        [
            {"model": "rf_balanced_tuned", "cv_roc_auc": float(rf_search.best_score_), "best_params": rf_search.best_params_},
            {"model": "rf_smote_tuned", "cv_roc_auc": float(rf_smote_search.best_score_), "best_params": rf_smote_search.best_params_},
            {"model": "xgb_balanced_tuned", "cv_roc_auc": float(xgb_search.best_score_), "best_params": xgb_search.best_params_},
            {"model": "xgb_smote_tuned", "cv_roc_auc": float(xgb_smote_search.best_score_), "best_params": xgb_smote_search.best_params_},
        ]
    ).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    best_search = {
        "rf_balanced_tuned": rf_search,
        "rf_smote_tuned": rf_smote_search,
        "xgb_balanced_tuned": xgb_search,
        "xgb_smote_tuned": xgb_smote_search,
    }[tuned_frame.loc[0, "model"]]

    # The deployed model is calibrated, so the threshold must be chosen on calibrated probabilities.
    # Calibrate on the training split only and pick the threshold on the untouched validation split.
    validation_calibrated = CalibratedClassifierCV(estimator=best_search.best_estimator_, method="sigmoid", cv=3)
    validation_calibrated.fit(train_features, train_target)
    validation_probabilities = validation_calibrated.predict_proba(validation_features)[:, 1]
    cost_frame = expected_cost_curve(validation_target.to_numpy(), validation_probabilities)
    best_threshold = best_cost_threshold(cost_frame)
    validation_summary = summarize_predictions(validation_target.to_numpy(), validation_probabilities, best_threshold["threshold"])

    final_features = pd.concat([train_features, validation_features], axis=0)
    final_target = pd.concat([train_target, validation_target], axis=0)
    final_calibrated = CalibratedClassifierCV(estimator=best_search.best_estimator_, method="sigmoid", cv=3)
    final_calibrated.fit(final_features, final_target)

    test_probabilities = final_calibrated.predict_proba(test_features)[:, 1]
    test_summary = summarize_predictions(test_target.to_numpy(), test_probabilities, best_threshold["threshold"])
    test_cost_frame = expected_cost_curve(test_target.to_numpy(), test_probabilities)
    test_threshold = best_cost_threshold(test_cost_frame)

    importance_frame = permutation_importance_table(
        final_calibrated,
        test_features,
        test_target.to_numpy(),
        FEATURE_COLUMNS,
    )
    calibration_frame = calibration_table(test_target.to_numpy(), test_probabilities)

    report = {
        "metric": "roc_auc",
        "selection_reason": "ROC-AUC is threshold-independent and is the most reliable selector for the imbalanced churn target.",
        "false_positive_cost_rupees": FALSE_POSITIVE_COST_RUPEES,
        "false_negative_cost_rupees": FALSE_NEGATIVE_COST_RUPEES,
        "baseline_models": baseline_frame.to_dict(orient="records"),
        "tuned_models": tuned_frame.to_dict(orient="records"),
        "best_model_name": tuned_frame.loc[0, "model"],
        "validation_threshold": best_threshold,
        "validation_metrics": validation_summary,
        "test_threshold_from_cost_curve": test_threshold,
        "test_metrics": test_summary,
        "calibration_curve": calibration_frame.to_dict(orient="records"),
        "permutation_importance": importance_frame.head(20).to_dict(orient="records"),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_calibrated,
            "threshold": best_threshold["threshold"],
            "report": report,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return TrainingBundle(model=final_calibrated, threshold=best_threshold["threshold"], report=report)


def load_or_train_bundle(force_retrain: bool = False) -> TrainingBundle:
    if MODEL_PATH.exists() and not force_retrain:
        bundle = joblib.load(MODEL_PATH)
        return TrainingBundle(model=bundle["model"], threshold=float(bundle["threshold"]), report=bundle["report"])
    return train_model()


if __name__ == "__main__":
    bundle = train_model()
    print(json.dumps(bundle.report["test_metrics"], indent=2))
