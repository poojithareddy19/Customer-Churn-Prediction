from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
    bootstrap_metrics,
    calibration_table,
    capacity_table,
    cross_validate_roc_auc_scores,
    expected_cost_curve,
    fold_score_summary,
    lift_table,
    logistic_odds_ratios,
    permutation_importance_table,
    profit_curve,
    summarize_predictions,
)
from src.features import (
    DATASET_PATH,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    FEATURE_COLUMNS,
    OFFER_COST,
    OFFER_SUCCESS_RATE,
    RETENTION_MONTHS,
    build_preprocessor,
    load_dataset,
    split_features_target,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
REPORT_PATH = ARTIFACT_DIR / "training_report.json"
PROFIT_SENSITIVITY_RATES = (0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class TrainingBundle:
    model: Any
    threshold: float
    report: dict[str, Any]
    feature_columns: list[str]


@dataclass(frozen=True)
class DataSplits:
    train_features: pd.DataFrame
    train_target: pd.Series
    validation_features: pd.DataFrame
    validation_target: pd.Series
    test_features: pd.DataFrame
    test_target: pd.Series


@dataclass(frozen=True)
class CalibratedEvaluation:
    validation_probabilities: np.ndarray
    threshold: dict[str, float]
    validation_metrics: dict[str, Any]
    model: Any
    test_probabilities: np.ndarray
    test_metrics: dict[str, Any]


def split_data(features: pd.DataFrame, target: pd.Series, random_state: int = 42) -> DataSplits:
    """Stratified 60/20/20 train, validation and test split."""
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
    return DataSplits(train_features, train_target, validation_features, validation_target, test_features, test_target)


def calibrate_and_evaluate(estimator, splits: DataSplits) -> CalibratedEvaluation:
    """Pick the cost threshold on validation, refit on train plus validation and score the test split."""
    # The deployed model is calibrated, so the threshold must be chosen on calibrated probabilities.
    # Calibrate on the training split only and pick the threshold on the untouched validation split.
    validation_calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)
    validation_calibrated.fit(splits.train_features, splits.train_target)
    validation_probabilities = validation_calibrated.predict_proba(splits.validation_features)[:, 1]
    cost_frame = expected_cost_curve(splits.validation_target.to_numpy(), validation_probabilities)
    best_threshold = best_cost_threshold(cost_frame)
    validation_summary = summarize_predictions(
        splits.validation_target.to_numpy(), validation_probabilities, best_threshold["threshold"]
    )

    final_features = pd.concat([splits.train_features, splits.validation_features], axis=0)
    final_target = pd.concat([splits.train_target, splits.validation_target], axis=0)
    final_calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)
    final_calibrated.fit(final_features, final_target)

    test_probabilities = final_calibrated.predict_proba(splits.test_features)[:, 1]
    test_summary = summarize_predictions(splits.test_target.to_numpy(), test_probabilities, best_threshold["threshold"])
    return CalibratedEvaluation(
        validation_probabilities=validation_probabilities,
        threshold=best_threshold,
        validation_metrics=validation_summary,
        model=final_calibrated,
        test_probabilities=test_probabilities,
        test_metrics=test_summary,
    )


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


def build_selected_pipeline(model_name: str, best_params: dict[str, Any], random_state: int, positive_ratio: float):
    """Rebuild a tuned candidate, for example "xgb_balanced_tuned", with its saved hyperparameters."""
    base_name = model_name.removesuffix("_tuned")
    pipeline = _candidate_pipelines(random_state, positive_ratio)[base_name]
    return pipeline.set_params(**best_params)


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


def _logistic_search_space() -> dict[str, list[Any]]:
    # liblinear supports both l1 and l2 penalties.
    return {
        "model__C": [float(value) for value in np.logspace(-3, 2, 20)],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"],
    }


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


def _search_model(pipeline, search_space: dict[str, list[Any]], features, target, cv, n_iter: int = 10) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
        verbose=0,
    )
    search.fit(features, target)
    return search


def _search_fold_scores(search: RandomizedSearchCV) -> list[float]:
    return [float(search.cv_results_[f"split{fold}_test_score"][search.best_index_]) for fold in range(search.n_splits_)]


def _profit_analysis(evaluation: CalibratedEvaluation, splits: DataSplits) -> dict[str, Any]:
    """Choose a profit-maximising threshold on validation and report test profit at it."""
    validation_target = splits.validation_target.to_numpy()
    validation_charges = splits.validation_features["MonthlyCharges"].to_numpy()
    test_target = splits.test_target.to_numpy()
    test_charges = splits.test_features["MonthlyCharges"].to_numpy()

    def at_threshold(success_rate: float, threshold: float) -> dict[str, float]:
        curve, _ = profit_curve(test_target, evaluation.test_probabilities, test_charges, OFFER_COST, success_rate, RETENTION_MONTHS)
        row = curve.iloc[(curve["threshold"] - threshold).abs().idxmin()]
        return {key: float(value) for key, value in row.items()}

    sensitivity = []
    for success_rate in sorted(set(PROFIT_SENSITIVITY_RATES) | {OFFER_SUCCESS_RATE}):
        _, validation_best = profit_curve(
            validation_target, evaluation.validation_probabilities, validation_charges, OFFER_COST, success_rate, RETENTION_MONTHS
        )
        test_row = at_threshold(success_rate, validation_best["threshold"])
        sensitivity.append(
            {
                "success_rate": float(success_rate),
                "validation_threshold": validation_best["threshold"],
                "validation_profit": validation_best["profit"],
                "test_customers_contacted": int(test_row["customers_contacted"]),
                "test_churners_contacted": int(test_row["churners_contacted"]),
                "test_profit": test_row["profit"],
            }
        )
    base = next(row for row in sensitivity if row["success_rate"] == OFFER_SUCCESS_RATE)
    deployed = at_threshold(OFFER_SUCCESS_RATE, evaluation.threshold["threshold"])
    return {
        "assumptions": {
            "offer_cost": OFFER_COST,
            "offer_success_rate": OFFER_SUCCESS_RATE,
            "retention_months": RETENTION_MONTHS,
            "note": "offer_success_rate is an assumption, not data; profit = success_rate * MonthlyCharges * retention_months "
            "summed over contacted churners, minus offer_cost per contacted customer",
        },
        "validation_threshold": base["validation_threshold"],
        "validation_profit": base["validation_profit"],
        "test_customers_contacted": base["test_customers_contacted"],
        "test_churners_contacted": base["test_churners_contacted"],
        "test_profit": base["test_profit"],
        "test_profit_at_deployed_cost_threshold": deployed["profit"],
        "success_rate_sensitivity": sensitivity,
    }


def _git_state() -> dict[str, Any]:
    """Commit hash of the code that trained the model, or None outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}


def _log_to_mlflow(report: dict[str, Any], best_params: dict[str, Any]) -> None:
    """Log the run to MLflow when it is installed (requirements-dev.txt); otherwise do nothing."""
    try:
        import mlflow
    except ImportError:
        return
    test_metrics = report["test_metrics"]
    positives = test_metrics["true_positives"] + test_metrics["false_negatives"]
    flagged = test_metrics["true_positives"] + test_metrics["false_positives"]
    try:
        mlflow.set_experiment("customer-churn")
        with mlflow.start_run(run_name=report["best_model_name"]):
            mlflow.log_params(
                {
                    "model_name": report["best_model_name"],
                    "false_positive_cost": report["false_positive_cost"],
                    "false_negative_cost": report["false_negative_cost"],
                    "threshold": report["validation_threshold"]["threshold"],
                    **{name.removeprefix("model__"): value for name, value in best_params.items()},
                }
            )
            mlflow.log_metrics(
                {
                    "cv_roc_auc": report["tuned_models"][0]["cv_roc_auc"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_brier_score": test_metrics["brier_score"],
                    "test_recall": test_metrics["true_positives"] / positives if positives else 0.0,
                    "test_precision": test_metrics["true_positives"] / flagged if flagged else 0.0,
                }
            )
    except Exception as error:  # Tracking problems must not fail training.
        print(f"MLflow logging skipped: {error}")


def train_model(
    random_state: int = 42,
    data: pd.DataFrame | None = None,
    n_iter: int = 10,
    artifact_dir: Path | None = None,
) -> TrainingBundle:
    """Train, select, calibrate and save the model.

    data must look like load_dataset() output; it defaults to the full dataset. n_iter is the number of
    randomised search iterations per model family. artifact_dir defaults to artifacts/.
    """
    run_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame = load_dataset(DATASET_PATH) if data is None else data
    features, target = split_features_target(frame)

    splits = split_data(features, target, random_state)
    train_features, train_target = splits.train_features, splits.train_target
    test_features, test_target = splits.test_features, splits.test_target

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    positive_ratio = float((train_target == 0).sum() / max((train_target == 1).sum(), 1))
    candidates = _candidate_pipelines(random_state, positive_ratio)

    baseline_rows = []
    cv_fold_scores: dict[str, dict[str, Any]] = {}
    for candidate_name, candidate_pipeline in candidates.items():
        fold_scores = cross_validate_roc_auc_scores(candidate_pipeline, train_features, train_target, cv=cv)
        cv_fold_scores[candidate_name] = fold_score_summary(fold_scores)
        baseline_rows.append({"model": candidate_name, "cv_roc_auc": float(np.mean(fold_scores))})

    baseline_frame = pd.DataFrame(baseline_rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    rf_search = _search_model(candidates["rf_balanced"], _rf_search_space(), train_features, train_target, cv, n_iter)
    rf_smote_search = _search_model(candidates["rf_smote"], _rf_smote_search_space(), train_features, train_target, cv, n_iter)
    xgb_search = _search_model(candidates["xgb_balanced"], _xgb_search_space(), train_features, train_target, cv, n_iter)
    xgb_smote_search = _search_model(candidates["xgb_smote"], _xgb_search_space(), train_features, train_target, cv, n_iter)
    logistic_search = _search_model(
        candidates["logistic_balanced"], _logistic_search_space(), train_features, train_target, cv, n_iter
    )

    tuned_frame = pd.DataFrame(
        [
            {"model": "rf_balanced_tuned", "cv_roc_auc": float(rf_search.best_score_), "best_params": rf_search.best_params_},
            {"model": "rf_smote_tuned", "cv_roc_auc": float(rf_smote_search.best_score_), "best_params": rf_smote_search.best_params_},
            {"model": "xgb_balanced_tuned", "cv_roc_auc": float(xgb_search.best_score_), "best_params": xgb_search.best_params_},
            {"model": "xgb_smote_tuned", "cv_roc_auc": float(xgb_smote_search.best_score_), "best_params": xgb_smote_search.best_params_},
            {"model": "logistic_balanced_tuned", "cv_roc_auc": float(logistic_search.best_score_), "best_params": logistic_search.best_params_},
        ]
    ).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)

    searches = {
        "rf_balanced_tuned": rf_search,
        "rf_smote_tuned": rf_smote_search,
        "xgb_balanced_tuned": xgb_search,
        "xgb_smote_tuned": xgb_smote_search,
        "logistic_balanced_tuned": logistic_search,
    }
    for search_name, search in searches.items():
        cv_fold_scores[search_name] = fold_score_summary(_search_fold_scores(search))
    best_search = searches[tuned_frame.loc[0, "model"]]

    evaluation = calibrate_and_evaluate(best_search.best_estimator_, splits)
    best_threshold = evaluation.threshold
    final_calibrated = evaluation.model
    test_probabilities = evaluation.test_probabilities
    test_cost_frame = expected_cost_curve(test_target.to_numpy(), test_probabilities)
    test_threshold = best_cost_threshold(test_cost_frame)

    importance_frame = permutation_importance_table(
        final_calibrated,
        test_features,
        test_target.to_numpy(),
        FEATURE_COLUMNS,
    )
    calibration_frame = calibration_table(test_target.to_numpy(), test_probabilities)
    test_lift = lift_table(test_target.to_numpy(), test_probabilities)
    test_capacity = capacity_table(test_target.to_numpy(), test_probabilities)
    test_bootstrap = bootstrap_metrics(test_target.to_numpy(), test_probabilities, best_threshold["threshold"])
    model_comparison = [
        {"model": name, "cv_roc_auc_mean": cv_fold_scores[name]["mean"], "cv_roc_auc_std": cv_fold_scores[name]["std"]}
        for name in tuned_frame["model"]
    ]
    odds_ratios = logistic_odds_ratios(logistic_search.best_estimator_, train_features, train_target)

    report = {
        "metric": "roc_auc",
        "selection_reason": "ROC-AUC is threshold-independent and is the most reliable selector for the imbalanced churn target.",
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "baseline_models": baseline_frame.to_dict(orient="records"),
        "tuned_models": tuned_frame.to_dict(orient="records"),
        "best_model_name": tuned_frame.loc[0, "model"],
        "validation_threshold": best_threshold,
        "validation_metrics": evaluation.validation_metrics,
        "test_threshold_from_cost_curve": test_threshold,
        "test_metrics": evaluation.test_metrics,
        "calibration_curve": calibration_frame.to_dict(orient="records"),
        "permutation_importance": importance_frame.head(20).to_dict(orient="records"),
        "cv_fold_scores": cv_fold_scores,
        "test_lift_table": test_lift.to_dict(orient="records"),
        "test_capacity_table": test_capacity.to_dict(orient="records"),
        "test_bootstrap_ci": test_bootstrap,
        "model_comparison": model_comparison,
        "logistic_odds_ratios": odds_ratios.to_dict(orient="records"),
        "profit_analysis": _profit_analysis(evaluation, splits),
        "run_timestamp": run_timestamp,
        **_git_state(),
    }

    output_dir = ARTIFACT_DIR if artifact_dir is None else Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_calibrated,
            "threshold": best_threshold["threshold"],
            "report": report,
            "feature_columns": FEATURE_COLUMNS,
        },
        output_dir / MODEL_PATH.name,
    )
    (output_dir / REPORT_PATH.name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log_to_mlflow(report, best_search.best_params_)

    return TrainingBundle(
        model=final_calibrated,
        threshold=best_threshold["threshold"],
        report=report,
        feature_columns=list(FEATURE_COLUMNS),
    )


def load_or_train_bundle(force_retrain: bool = False) -> TrainingBundle:
    if MODEL_PATH.exists() and not force_retrain:
        bundle = joblib.load(MODEL_PATH)
        return TrainingBundle(
            model=bundle["model"],
            threshold=float(bundle["threshold"]),
            report=bundle["report"],
            # Artifacts saved before feature_columns was stored fall back to the current schema.
            feature_columns=list(bundle.get("feature_columns", FEATURE_COLUMNS)),
        )
    return train_model()


if __name__ == "__main__":
    bundle = train_model()
    print(json.dumps(bundle.report["test_metrics"], indent=2))
