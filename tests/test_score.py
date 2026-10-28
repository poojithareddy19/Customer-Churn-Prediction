import numpy as np
import pandas as pd
import pytest

from src import score
from src.app_helpers import profit_threshold
from src.features import DATASET_PATH, FEATURE_COLUMNS, apply_internet_guard, build_input_frame, clean_frame
from src.train import load_or_train_bundle

EXPECTED_COLUMNS = ["customerID", "churn_probability", "risk_rank", "decision_rule", "threshold", "flagged", "charges_warning"]


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("score") / "customers.csv"
    pd.read_csv(DATASET_PATH).head(200).to_csv(path, index=False)
    return path


def _run(tmp_path, input_path, *options):
    output_path = tmp_path / "scored.csv"
    code = score.main(["--input", str(input_path), "--output", str(output_path), *options])
    return code, output_path


def test_scoring_200_rows_keeps_ids_and_columns(tmp_path, sample_csv):
    code, output_path = _run(tmp_path, sample_csv)

    scored = pd.read_csv(output_path)
    assert code == 0
    assert len(scored) == 200
    assert list(scored.columns) == EXPECTED_COLUMNS
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["customerID"].tolist() == pd.read_csv(sample_csv)["customerID"].tolist()
    assert sorted(scored["risk_rank"]) == list(range(1, 201))


def test_scores_match_the_app_path(tmp_path, sample_csv):
    bundle = load_or_train_bundle()
    _, output_path = _run(tmp_path, sample_csv)
    scored = pd.read_csv(output_path)

    rows = clean_frame(pd.read_csv(sample_csv))[FEATURE_COLUMNS].to_dict(orient="records")
    app_probabilities = [
        float(bundle.model.predict_proba(build_input_frame(apply_internet_guard(row), bundle.feature_columns))[0, 1])
        for row in rows
    ]
    assert np.allclose(scored["churn_probability"], app_probabilities, rtol=0, atol=1e-12)


def test_capacity_rule_flags_top_share(tmp_path, sample_csv):
    code, output_path = _run(tmp_path, sample_csv, "--rule", "capacity", "--capacity", "0.2")

    scored = pd.read_csv(output_path)
    assert code == 0
    assert scored["flagged"].sum() == 40
    assert set(scored.loc[scored["flagged"], "risk_rank"]) == set(range(1, 41))
    assert scored["threshold"].isna().all()


def test_missing_feature_column_fails_and_names_it(tmp_path, sample_csv, capsys):
    broken = tmp_path / "broken.csv"
    pd.read_csv(sample_csv).drop(columns=["Contract"]).to_csv(broken, index=False)

    code, output_path = _run(tmp_path, broken)

    assert code != 0
    assert "Contract" in capsys.readouterr().err
    assert not output_path.exists()


def test_profit_rule_uses_report_profit_threshold(tmp_path, sample_csv):
    cutoff = profit_threshold(load_or_train_bundle().report)
    code, output_path = _run(tmp_path, sample_csv, "--rule", "profit")

    scored = pd.read_csv(output_path)
    assert code == 0
    assert (scored["threshold"] == cutoff).all()
    assert (scored["flagged"] == (scored["churn_probability"] >= cutoff)).all()


def test_missing_artifact_fails_instead_of_training(tmp_path, sample_csv, monkeypatch, capsys):
    monkeypatch.setattr(score, "MODEL_PATH", tmp_path / "missing.joblib")

    code, output_path = _run(tmp_path, sample_csv)

    assert code != 0
    assert "python -m src.train" in capsys.readouterr().err
    assert not output_path.exists()
