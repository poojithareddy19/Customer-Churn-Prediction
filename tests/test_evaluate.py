import numpy as np
import pytest

from src.evaluate import best_cost_threshold, bootstrap_metrics, capacity_table, expected_cost_curve, lift_table


def _toy_scores(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, n)
    probabilities = np.clip(0.3 * target + rng.uniform(0, 0.7, n), 0, 1)
    return target, probabilities


def test_expected_cost_curve_picks_minimum_cost_threshold():
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.9])

    cost_frame = expected_cost_curve(target, probabilities, false_positive_cost=1, false_negative_cost=10, steps=3)
    best = best_cost_threshold(cost_frame)

    assert best["threshold"] == 0.5
    assert best["expected_cost"] == 0.0


def test_lift_table_perfect_ranking_puts_highest_lift_in_first_decile():
    # 20 churners out of 100, all ranked above every non-churner.
    target = np.array([1] * 20 + [0] * 80)
    probabilities = np.linspace(1.0, 0.0, 100)

    table = lift_table(target, probabilities)

    assert list(table["decile"]) == list(range(1, 11))
    assert table["customers"].sum() == 100
    assert table.loc[0, "lift"] == table["lift"].max()
    assert table.loc[0, "lift"] == pytest.approx(5.0)
    assert table["cumulative_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["cumulative_lift"].iloc[-1] == pytest.approx(1.0)


def test_capacity_table_full_contact_gives_full_recall():
    target, probabilities = _toy_scores()

    table = capacity_table(target, probabilities, capacities=(0.1, 0.5, 1.0), false_positive_cost=1, false_negative_cost=10)

    full = table.loc[table["capacity"] == 1.0].iloc[0]
    assert full["customers_contacted"] == len(target)
    assert full["recall"] == pytest.approx(1.0)
    assert full["probability_cutoff"] == pytest.approx(probabilities.min())
    assert table["recall"].is_monotonic_increasing


def test_bootstrap_metrics_interval_contains_estimate_and_is_reproducible():
    target, probabilities = _toy_scores()

    first = bootstrap_metrics(target, probabilities, threshold=0.5, false_positive_cost=1, false_negative_cost=10, n_boot=200)
    second = bootstrap_metrics(target, probabilities, threshold=0.5, false_positive_cost=1, false_negative_cost=10, n_boot=200)

    assert first == second
    assert first["n_skipped"] == 0
    assert set(first["metrics"]) == {"roc_auc", "brier_score", "recall", "precision", "expected_cost", "top_decile_lift"}
    for interval in first["metrics"].values():
        assert interval["lower"] <= interval["estimate"] <= interval["upper"]


def test_bootstrap_metrics_skips_single_class_resamples():
    target = np.array([0] * 9 + [1])
    probabilities = np.linspace(0.0, 1.0, 10)

    result = bootstrap_metrics(target, probabilities, threshold=0.5, false_positive_cost=1, false_negative_cost=10, n_boot=50)

    assert result["n_skipped"] > 0
