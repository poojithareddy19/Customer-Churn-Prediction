import joblib
import numpy as np
import pytest

from src.app_helpers import capacity_rows, expected_contact_value, explain_prediction, profit_threshold, report_cost
from src.features import FEATURE_COLUMNS, load_dataset, original_feature_name, readable_feature_name
from src.train import MODEL_PATH


def test_report_cost_reads_current_keys():
    report = {"false_positive_cost": 65.0, "false_negative_cost": 780.0}
    assert report_cost(report, "false_positive") == 65.0
    assert report_cost(report, "false_negative") == 780.0


def test_report_cost_falls_back_to_old_rupee_keys():
    report = {"false_positive_cost_rupees": 100.0, "false_negative_cost_rupees": 1500.0}
    assert report_cost(report, "false_positive") == 100.0
    assert report_cost(report, "false_negative") == 1500.0


def test_profit_threshold_reads_validation_threshold():
    assert profit_threshold({"profit_analysis": {"validation_threshold": 0.34}}) == 0.34


def test_profit_threshold_is_none_for_older_artifacts():
    assert profit_threshold({"false_positive_cost": 65.0}) is None


def test_expected_contact_value_is_minus_offer_cost_at_zero_probability():
    assert expected_contact_value(0.0, 70.0, 65.0, 0.3, 12) == -65.0


def test_expected_contact_value_hand_computed():
    assert expected_contact_value(0.5, 70.0, 65.0, 0.3, 12) == pytest.approx(0.5 * 0.3 * 70 * 12 - 65)
    assert expected_contact_value(0.5, 70.0, 65.0, 0.3, 12) == pytest.approx(61.0)


def test_expected_contact_value_matches_profit_curve():
    from src.evaluate import profit_curve

    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 50)
    probabilities = rng.uniform(0, 1, 50)
    charges = rng.uniform(20, 110, 50)
    threshold = 0.5

    curve, _ = profit_curve(target, probabilities, charges, offer_cost=65.0, success_rate=0.3, months=12)
    curve_profit = curve.loc[np.isclose(curve["threshold"], threshold), "profit"].item()

    contacted = probabilities >= threshold
    # With the true label in place of the probability, the summed per-customer values are the realised profit.
    summed = sum(
        expected_contact_value(float(label), float(charge), 65.0, 0.3, 12)
        for label, charge in zip(target[contacted], charges[contacted])
    )
    assert summed == pytest.approx(curve_profit)


def test_capacity_rows_prefers_validation_table():
    validation = [{"capacity": 0.2, "probability_cutoff": 0.55}]
    test = [{"capacity": 0.2, "probability_cutoff": 0.56}]
    assert capacity_rows({"validation_capacity_table": validation, "test_capacity_table": test}) == (validation, "validation")


def test_capacity_rows_falls_back_to_test_table_for_older_artifacts():
    test = [{"capacity": 0.2, "probability_cutoff": 0.56}]
    assert capacity_rows({"test_capacity_table": test}) == (test, "test")


def test_capacity_rows_empty_when_absent():
    assert capacity_rows({"false_positive_cost": 65.0}) == ([], "")


@pytest.mark.parametrize(
    "transformed,original,readable",
    [
        ("numeric__tenure", "tenure", "tenure"),
        ("categorical__Contract_Two year", "Contract", "Contract = Two year"),
        ("categorical__PaymentMethod_Bank transfer (automatic)", "PaymentMethod", "PaymentMethod = Bank transfer (automatic)"),
        ("categorical__OnlineSecurity_No internet service", "OnlineSecurity", "OnlineSecurity = No internet service"),
    ],
)
def test_feature_name_mapping(transformed, original, readable):
    assert original_feature_name(transformed) == original
    assert readable_feature_name(transformed) == readable


def test_explain_prediction_returns_top_five_readable_contributions():
    bundle = joblib.load(MODEL_PATH)
    frame = load_dataset()[FEATURE_COLUMNS]
    row = frame.head(1)

    explanation, unit = explain_prediction(bundle["model"], row, background=frame.sample(200, random_state=0), top_n=5)

    assert len(explanation) == 5
    assert explanation["contribution"].abs().is_monotonic_decreasing
    assert np.isfinite(explanation["contribution"]).all()
    assert unit in {"log-odds", "probability"}
    for label in explanation["feature"]:
        feature, value = label.split(" = ", 1)
        assert feature in FEATURE_COLUMNS
        assert value == str(row.iloc[0][feature])


def test_explain_prediction_supports_logistic_regression():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.features import build_preprocessor, split_features_target

    features, target = split_features_target(load_dataset().sample(600, random_state=42))
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=2000))])
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3).fit(features, target)

    explanation, unit = explain_prediction(calibrated, features.head(1), background=features, top_n=5)

    assert unit == "log-odds"
    assert len(explanation) == 5
    with pytest.raises(ValueError):
        explain_prediction(calibrated, features.head(1), background=None)


def test_log_prediction_appends_json_lines(tmp_path):
    import json

    from src.app_helpers import log_prediction

    log_path = tmp_path / "logs" / "predictions.jsonl"
    log_prediction(log_path, {"tenure": 12, "Contract": "Month-to-month"}, 0.42, "Cost threshold", 0.08, "high risk")
    log_prediction(log_path, {"tenure": 60, "Contract": "Two year"}, 0.03, "Capacity (top N%)", 0.67, "lower risk")

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["inputs"]["Contract"] == "Month-to-month"
    assert records[1]["decision_rule"] == "Capacity (top N%)"
    assert records[0]["timestamp_utc"].endswith("+00:00")
    assert set(records[0]) == {"timestamp_utc", "inputs", "probability", "decision_rule", "threshold", "verdict", "warning"}
    assert records[0]["warning"] is None


def test_log_prediction_records_optional_warning(tmp_path):
    import json

    from src.app_helpers import log_prediction

    log_path = tmp_path / "predictions.jsonl"
    log_prediction(log_path, {"tenure": 0}, 0.4, "Cost threshold", 0.08, "high risk", warning="tenure is 0")

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["warning"] == "tenure is 0"
