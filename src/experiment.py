"""Pure helpers for designing and analysing a two-arm retention experiment."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chisquare
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize, proportions_ztest


def sample_size_two_proportions(p_control: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers needed per arm to detect an absolute lift of mde over p_control (two-sided, equal arms)."""
    if not 0 < p_control < 1 or not 0 < p_control + mde < 1:
        raise ValueError("p_control and p_control + mde must both be strictly between 0 and 1")
    effect_size = proportion_effectsize(p_control + mde, p_control)
    per_arm = NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0, alternative="two-sided")
    return int(math.ceil(per_arm))


def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> float:
    """Chi-square p-value for a sample-ratio mismatch; expected_ratio is the planned treatment share."""
    total = n_control + n_treatment
    expected = [total * (1 - expected_ratio), total * expected_ratio]
    return float(chisquare([n_control, n_treatment], f_exp=expected).pvalue)


def analyze_ab(conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05) -> dict[str, float]:
    """Absolute lift (treatment minus control) with a Wald CI and a pooled two-proportion z-test."""
    rate_control = conv_c / n_c
    rate_treatment = conv_t / n_t
    ci_lower, ci_upper = confint_proportions_2indep(conv_t, n_t, conv_c, n_c, method="wald", compare="diff", alpha=alpha)
    z_stat, p_value = proportions_ztest([conv_t, conv_c], [n_t, n_c], alternative="two-sided")
    return {
        "control_rate": float(rate_control),
        "treatment_rate": float(rate_treatment),
        "absolute_lift": float(rate_treatment - rate_control),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
    }


def simulate_experiment(eligible_n: int, base_retention: float, true_effect: float, random_state: int = 42) -> dict[str, int]:
    """SIMULATED: randomise customers 50/50 and draw retention outcomes with an assumed treatment effect."""
    rng = np.random.default_rng(random_state)
    treated = rng.random(eligible_n) < 0.5
    retention_probability = np.where(treated, base_retention + true_effect, base_retention)
    retained = rng.random(eligible_n) < retention_probability
    return {
        "n_control": int((~treated).sum()),
        "n_treatment": int(treated.sum()),
        "retained_control": int((retained & ~treated).sum()),
        "retained_treatment": int((retained & treated).sum()),
    }
