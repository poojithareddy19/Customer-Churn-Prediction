import numpy as np

from src.evaluate import best_cost_threshold, expected_cost_curve


def test_expected_cost_curve_picks_minimum_cost_threshold():
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.9])

    cost_frame = expected_cost_curve(target, probabilities, false_positive_cost=1, false_negative_cost=10, steps=3)
    best = best_cost_threshold(cost_frame)

    assert best["threshold"] == 0.5
    assert best["expected_cost"] == 0.0
