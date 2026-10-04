import pytest

from src.experiment import analyze_ab, sample_size_two_proportions, srm_check


def test_sample_size_grows_as_mde_shrinks():
    sizes = [sample_size_two_proportions(0.6, mde) for mde in (0.10, 0.05, 0.03, 0.02)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_sample_size_rejects_impossible_rates():
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.98, 0.05)


def test_srm_check_flags_only_a_real_mismatch():
    assert srm_check(5000, 5000) > 0.05
    assert srm_check(5000, 5500) < 0.001


def test_analyze_ab_interval_contains_observed_difference():
    result = analyze_ab(conv_c=600, n_c=1000, conv_t=650, n_t=1000)
    assert result["absolute_lift"] == pytest.approx(0.05)
    assert result["ci_lower"] < result["absolute_lift"] < result["ci_upper"]
    assert 0 < result["p_value"] < 1

