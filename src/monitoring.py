"""Population stability index (PSI) for comparing a reference distribution with a newer one."""
from __future__ import annotations

import numpy as np

PSI_STABLE = 0.1
PSI_SIGNIFICANT = 0.25
PSI_EPSILON = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI with bin edges at the quantiles of expected; empty bins get a small epsilon share."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    # Discrete columns such as tenure can repeat quantiles, so keep only distinct inner edges.
    inner_edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1))[1:-1])
    expected_counts = np.bincount(np.searchsorted(inner_edges, expected, side="right"), minlength=len(inner_edges) + 1)
    actual_counts = np.bincount(np.searchsorted(inner_edges, actual, side="right"), minlength=len(inner_edges) + 1)
    expected_share = np.maximum(expected_counts / expected_counts.sum(), PSI_EPSILON)
    actual_share = np.maximum(actual_counts / actual_counts.sum(), PSI_EPSILON)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def psi_status(value: float) -> str:
    """Usual reading: below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 significant shift."""
    if value < PSI_STABLE:
        return "stable"
    if value <= PSI_SIGNIFICANT:
        return "moderate"
    return "significant"
