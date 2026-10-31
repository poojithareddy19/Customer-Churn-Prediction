"""Score a CSV of customers in batch with the saved model.

    python -m src.score --input customers.csv --output scored.csv [--rule profit|cost|capacity]
                        [--capacity 0.2] [--threshold 0.4]

The input needs the 19 feature columns of the Telco dataset; customerID is copied to the output when present.
Rows go through the same cleaning and consistency guards as the app. Rules:

- profit: flag customers at or above the validation profit threshold saved in the report.
- cost: flag customers at or above the validation cost threshold saved with the model.
- capacity: flag the top share of this input file by score (ties keep file order). This differs from the
  app, which applies a fixed probability cutoff from the training report to one customer at a time.

--threshold overrides the profit or cost threshold, and decision_rule then reads "profit (override)" or
"cost (override)". Rows with a blank or unknown category value are still scored (the encoder ignores unknown
values and the imputer fills blanks) and named in the category_warning column. The command never trains: it
fails if the artifact is missing. Batch predictions are not written to the app's prediction log.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.app_helpers import profit_threshold
from src.evaluate import _rank_order
from src.features import CUSTOMER_ID_COLUMN, apply_internet_guard, category_warning, charges_warning, clean_frame
from src.train import MODEL_PATH, TrainingBundle, load_or_train_bundle

RULES = ("profit", "cost", "capacity")
REQUIRED_NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges")


class ScoringError(Exception):
    """Expected problem with the input or options, reported without a traceback."""


def _check_numeric(frame: pd.DataFrame) -> None:
    for column in REQUIRED_NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad_rows = frame.index[values.isna()]
        if len(bad_rows):
            examples = ", ".join(str(row + 2) for row in bad_rows[:5])
            raise ScoringError(f"column {column} must be numeric; bad value on line {examples} of the input")
        frame[column] = values
    total = frame["TotalCharges"]
    blank = total.isna() | total.astype(str).str.strip().eq("")
    bad_rows = frame.index[pd.to_numeric(total, errors="coerce").isna() & ~blank]
    if len(bad_rows):
        examples = ", ".join(str(row + 2) for row in bad_rows[:5])
        raise ScoringError(f"column TotalCharges must be numeric or blank; bad value on line {examples} of the input")


def score_frame(
    raw: pd.DataFrame,
    bundle: TrainingBundle,
    rule: str = "profit",
    capacity: float = 0.2,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score a raw Telco-format frame and apply one decision rule."""
    missing = [column for column in bundle.feature_columns if column not in raw.columns]
    if missing:
        raise ScoringError(f"missing feature columns: {', '.join(missing)}")
    if raw.empty:
        raise ScoringError("the input has no rows")
    if rule not in RULES:
        raise ScoringError(f"unknown rule {rule}; choose from {', '.join(RULES)}")

    raw = raw.reset_index(drop=True)
    checked = raw[bundle.feature_columns].copy()
    _check_numeric(checked)
    cleaned = clean_frame(checked)
    guarded = [apply_internet_guard(row) for row in cleaned[bundle.feature_columns].to_dict(orient="records")]
    model_input = pd.DataFrame(guarded, columns=bundle.feature_columns)
    probabilities = bundle.model.predict_proba(model_input)[:, 1]

    ranks = np.empty(len(probabilities), dtype=int)
    ranks[_rank_order(probabilities)] = np.arange(1, len(probabilities) + 1)

    if rule == "capacity":
        if threshold is not None:
            raise ScoringError("--threshold cannot be combined with the capacity rule")
        if not 0 < capacity <= 1:
            raise ScoringError(f"--capacity must be in (0, 1], got {capacity}")
        # Same rounding as evaluate.capacity_table.
        flagged_count = min(len(probabilities), math.ceil(round(capacity * len(probabilities), 9)))
        flagged = ranks <= flagged_count
        used_threshold: float | None = None
        rule_label = rule
    else:
        rule_label = rule if threshold is None else f"{rule} (override)"
        if threshold is None:
            threshold = profit_threshold(bundle.report) if rule == "profit" else bundle.threshold
            if threshold is None:
                raise ScoringError("this artifact has no profit threshold; use --rule cost or pass --threshold")
        used_threshold = float(threshold)
        flagged = probabilities >= used_threshold

    output: dict[str, Any] = {}
    if CUSTOMER_ID_COLUMN in raw.columns:
        output[CUSTOMER_ID_COLUMN] = raw[CUSTOMER_ID_COLUMN].to_numpy()
    output["churn_probability"] = probabilities
    output["risk_rank"] = ranks
    output["decision_rule"] = rule_label
    output["threshold"] = used_threshold
    output["flagged"] = flagged
    output["charges_warning"] = [charges_warning(row) for row in guarded]
    output["category_warning"] = [category_warning(row) for row in guarded]
    return pd.DataFrame(output)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.score", description="Score a CSV of customers with the saved churn model."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV with the Telco feature columns")
    parser.add_argument("--output", required=True, type=Path, help="where to write the scored CSV")
    parser.add_argument("--rule", choices=RULES, default="profit", help="decision rule (default: profit)")
    parser.add_argument("--capacity", type=float, default=0.2, help="share to flag with the capacity rule (default: 0.2)")
    parser.add_argument("--threshold", type=float, default=None, help="override the profit or cost threshold")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if not MODEL_PATH.exists():
            raise ScoringError(f"no trained model at {MODEL_PATH}; run python -m src.train first")
        try:
            raw = pd.read_csv(args.input)
        except FileNotFoundError:
            raise ScoringError(f"input file not found: {args.input}") from None
        except pd.errors.EmptyDataError:
            raise ScoringError(f"input file is empty: {args.input}") from None
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            raise ScoringError(f"could not read {args.input} as CSV: {error}") from None
        scored = score_frame(raw, load_or_train_bundle(), args.rule, args.capacity, args.threshold)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.output, index=False)
        except OSError as error:
            raise ScoringError(f"could not write {args.output}: {error}") from None
    except ScoringError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Scored {len(scored)} customers, flagged {int(scored['flagged'].sum())}, wrote {args.output}")
    unknown = int(scored["category_warning"].notna().sum())
    if unknown:
        print(
            f"Warning: blank or unknown category values in {unknown} of {len(scored)} rows; "
            "see the category_warning column"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
