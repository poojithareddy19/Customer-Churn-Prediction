from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"


def sql_files(sql_dir: Path = SQL_DIR) -> list[Path]:
    """SQL files in run order (they are numbered)."""
    return sorted(sql_dir.glob("*.sql"))


def run_sql_file(path: Path, data_dir: Path = ROOT) -> pd.DataFrame:
    """Run one query; relative CSV paths inside it resolve against data_dir."""
    connection = duckdb.connect()
    try:
        connection.execute(f"SET file_search_path = '{data_dir.as_posix()}'")
        frame = connection.execute(path.read_text(encoding="utf-8")).df()
    finally:
        connection.close()
    # DuckDB sums come back as floats; customer and churner counts are whole numbers.
    for column in ("customers", "churners"):
        if column in frame:
            frame[column] = frame[column].astype(int)
    return frame


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    # At 0 or 100% the matching bound is exactly 0 or 1; set it directly to avoid rounding drift.
    lower = 0.0 if successes == 0 else max(0.0, centre - half_width)
    upper = 1.0 if successes == trials else min(1.0, centre + half_width)
    return lower, upper


def add_wilson_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add churn_rate_ci_lower and churn_rate_ci_upper next to churn_rate."""
    result = frame.copy()
    bounds = [wilson_interval(int(row.churners), int(row.customers)) for row in result.itertuples()]
    position = result.columns.get_loc("churn_rate") + 1
    result.insert(position, "churn_rate_ci_lower", [lower for lower, _ in bounds])
    result.insert(position + 1, "churn_rate_ci_upper", [upper for _, upper in bounds])
    return result
