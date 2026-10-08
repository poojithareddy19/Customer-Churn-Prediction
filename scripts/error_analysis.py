"""Profile the churners the model misses at the profit threshold and draw a precision-recall curve.

Scores the test split with the deployed model, splits the actual churners into caught (probability at or above
the profit threshold) and missed, and compares the two groups. Writes reports/missed_churners.md and a light and
dark precision-recall chart to docs/images/.

Usage: python scripts/error_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from readme_charts import OUTPUT_DIR, THEMES, _style_axes  # noqa: E402
from src.app_helpers import profit_threshold  # noqa: E402
from src.features import load_dataset, split_features_target  # noqa: E402
from src.train import load_or_train_bundle, split_data  # noqa: E402

REPORT_PATH = ROOT / "reports" / "missed_churners.md"
NUMERIC_PROFILE = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_PROFILE = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]


def numeric_profile(caught: pd.DataFrame, missed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_PROFILE:
        rows.append(
            {
                "column": column,
                "caught_median": caught[column].median(),
                "missed_median": missed[column].median(),
                "caught_mean": caught[column].mean(),
                "missed_mean": missed[column].mean(),
            }
        )
    return pd.DataFrame(rows)


def categorical_profile(caught: pd.DataFrame, missed: pd.DataFrame, column: str) -> pd.DataFrame:
    shares = pd.DataFrame(
        {
            "caught": caught[column].value_counts(normalize=True),
            "missed": missed[column].value_counts(normalize=True),
        }
    ).fillna(0.0)
    shares.index.name = "value"
    return shares.sort_values("missed", ascending=False).reset_index()


def draw_pr_curve(target, probabilities, threshold: float, cost_threshold: float, theme_name: str) -> Path:
    theme = THEMES[theme_name]
    precision, recall, cutoffs = precision_recall_curve(target, probabilities)
    base_rate = float(target.mean())

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.patch.set_facecolor(theme["surface"])
    _style_axes(ax, theme)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.plot(recall, precision, color=theme["series"], linewidth=2)
    ax.axhline(base_rate, color=theme["muted"], linewidth=1.2, linestyle="--")
    ax.annotate(f"Baseline {base_rate:.1%} (churn rate)", (0.02, base_rate), xytext=(0, -14), textcoords="offset points",
                color=theme["text_secondary"], fontsize=10)
    for cutoff, label in ((threshold, "profit"), (cost_threshold, "cost-only")):
        flagged = probabilities >= cutoff
        point_recall = float(target[flagged].sum() / target.sum())
        point_precision = float(target[flagged].mean())
        ax.plot(point_recall, point_precision, marker="o", markersize=8, color=theme["text"],
                markeredgecolor=theme["surface"], markeredgewidth=2)
        ax.annotate(f"{cutoff:.2f} {label}: {point_recall:.0%} recall, {point_precision:.0%} precision",
                    (point_recall, point_precision), xytext=(-10, -18), textcoords="offset points", ha="right",
                    color=theme["text"], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Recall (share of churners flagged)", color=theme["text_secondary"], fontsize=10)
    ax.set_ylabel("Precision", color=theme["text_secondary"], fontsize=10)
    average_precision = average_precision_score(target, probabilities)
    ax.set_title(f"Precision-recall, test split (AP {average_precision:.3f})", loc="left", color=theme["text"],
                 fontsize=12, pad=12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"pr_curve_{theme_name}.png"
    fig.savefig(output_path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return output_path


def _markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    bundle = load_or_train_bundle()
    threshold = profit_threshold(bundle.report)
    features, target = split_features_target(load_dataset())
    splits = split_data(features, target, 42)
    test = splits.test_features.copy()
    test_target = splits.test_target.to_numpy()
    probabilities = bundle.model.predict_proba(test[bundle.feature_columns])[:, 1]
    test["probability"] = probabilities

    churners = test.loc[test_target == 1]
    caught = churners.loc[churners["probability"] >= threshold]
    missed = churners.loc[churners["probability"] < threshold]

    numeric = numeric_profile(caught, missed)
    numeric_view = pd.DataFrame(
        {
            "Column": numeric["column"],
            "Caught median": numeric["caught_median"].map("{:.1f}".format),
            "Missed median": numeric["missed_median"].map("{:.1f}".format),
            "Caught mean": numeric["caught_mean"].map("{:.1f}".format),
            "Missed mean": numeric["missed_mean"].map("{:.1f}".format),
        }
    )

    lines = [
        "# Missed churners",
        "",
        "Generated by `python scripts/error_analysis.py`. Test split, deployed model, profit threshold "
        f"{threshold:.2f} (the app default).",
        "",
        f"Of {len(churners):,} churners in the test split, the model flags **{len(caught):,}** and misses "
        f"**{len(missed):,}**. Missed churners have a median predicted probability of "
        f"{missed['probability'].median():.2f}, against {caught['probability'].median():.2f} for caught ones.",
        "",
        "## Numeric columns",
        "",
        _markdown(numeric_view),
        "",
    ]
    for column in CATEGORICAL_PROFILE:
        shares = categorical_profile(caught, missed, column)
        view = pd.DataFrame(
            {
                column: shares["value"],
                f"Caught ({len(caught)})": shares["caught"].map("{:.1%}".format),
                f"Missed ({len(missed)})": shares["missed"].map("{:.1%}".format),
            }
        )
        lines += [f"## {column}", "", _markdown(view), ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    for theme_name in THEMES:
        print(draw_pr_curve(test_target, probabilities, threshold, bundle.threshold, theme_name))
    print(numeric.round(1).to_string(index=False))
    for column in CATEGORICAL_PROFILE:
        print(categorical_profile(caught, missed, column).round(3).to_string(index=False))
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
