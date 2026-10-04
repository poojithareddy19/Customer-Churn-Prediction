import numpy as np

from src.monitoring import PSI_SIGNIFICANT, PSI_STABLE, psi, psi_status


def test_psi_is_zero_for_identical_samples():
    values = np.random.default_rng(0).normal(size=5000)
    assert psi(values, values) == 0.0


def test_psi_is_small_for_same_distribution_and_large_for_shifted_one():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=5000)
    assert psi(reference, rng.normal(size=5000)) < PSI_STABLE
    assert psi(reference, rng.normal(loc=1.0, size=5000)) > PSI_SIGNIFICANT


def test_psi_handles_discrete_values_and_empty_bins():
    reference = np.repeat([0, 1, 2], [500, 300, 200])
    shifted = np.repeat([2, 3], [500, 500])
    value = psi(reference, shifted)
    assert np.isfinite(value) and value > PSI_SIGNIFICANT


def test_psi_status_thresholds():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.1) == "moderate"
    assert psi_status(0.25) == "moderate"
    assert psi_status(0.3) == "significant"
