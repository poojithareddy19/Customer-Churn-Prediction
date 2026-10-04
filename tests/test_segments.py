import pytest

from src.segments import SQL_DIR, add_wilson_columns, run_sql_file, wilson_interval


def test_overall_query_matches_dataset_totals():
    frame = run_sql_file(SQL_DIR / "01_overall.sql")
    assert frame.loc[0, "customers"] == 7043
    assert frame.loc[0, "churners"] == 1869


def test_tenure_cohorts_cover_every_customer():
    frame = run_sql_file(SQL_DIR / "02_tenure_cohorts.sql")
    assert list(frame["tenure_cohort"]) == ["0-6", "7-12", "13-24", "25-48", "49+"]
    assert frame["customers"].sum() == 7043


@pytest.mark.parametrize("successes,trials", [(0, 10), (3, 10), (10, 10), (1869, 7043)])
def test_wilson_interval_stays_in_unit_range_and_contains_estimate(successes, trials):
    lower, upper = wilson_interval(successes, trials)
    assert 0.0 <= lower <= successes / trials <= upper <= 1.0


def test_add_wilson_columns_places_bounds_after_rate():
    frame = add_wilson_columns(run_sql_file(SQL_DIR / "01_overall.sql"))
    columns = list(frame.columns)
    assert columns[columns.index("churn_rate") + 1 : columns.index("churn_rate") + 3] == ["churn_rate_ci_lower", "churn_rate_ci_upper"]
