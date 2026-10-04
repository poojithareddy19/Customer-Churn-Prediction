import json

import joblib
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS, load_dataset
from src.train import MODEL_PATH, load_or_train_bundle, train_model


def test_load_or_train_bundle_returns_saved_feature_columns():
    saved = joblib.load(MODEL_PATH)
    bundle = load_or_train_bundle()
    assert bundle.feature_columns == list(saved["feature_columns"])
    assert bundle.feature_columns == FEATURE_COLUMNS


def test_train_model_runs_end_to_end_on_a_small_sample(tmp_path, monkeypatch):
    # Keep any MLflow run out of the repository.
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    frame = load_dataset()
    sample, _ = train_test_split(frame, train_size=800, stratify=frame["Churn"], random_state=42)

    bundle = train_model(data=sample, n_iter=2, artifact_dir=tmp_path)

    report = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))
    assert (tmp_path / "churn_model.joblib").exists()
    assert 0.0 < bundle.threshold < 1.0
    for key in [
        "cv_fold_scores",
        "test_lift_table",
        "test_capacity_table",
        "test_bootstrap_ci",
        "model_comparison",
        "logistic_odds_ratios",
        "profit_analysis",
        "run_timestamp",
        "git_commit",
        "false_positive_cost",
        "false_negative_cost",
    ]:
        assert key in report
    assert len(report["model_comparison"]) == 5
    assert sum(row["customers"] for row in report["test_lift_table"]) == 160

    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=["customer-churn"])
    assert len(runs) == 1
    assert runs.loc[0, "metrics.test_roc_auc"] == report["test_metrics"]["roc_auc"]
