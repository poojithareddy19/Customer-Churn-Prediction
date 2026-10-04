"""Lift and calibration charts for the README, drawn from artifacts/training_report.json.

Writes a light and a dark version to docs/images/ so GitHub can show the one matching the reader's theme.

Usage: python scripts/readme_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "training_report.json"
OUTPUT_DIR = ROOT / "docs" / "images"

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text_secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "series": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "text_secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "series": "#3987e5"},
}


def _style_axes(ax, theme: dict[str, str]) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.tick_params(colors=theme["text_secondary"], length=0, labelsize=10)


def draw(report: dict, theme_name: str) -> Path:
    theme = THEMES[theme_name]
    lift_rows = report["test_lift_table"]
    calibration = report["calibration_curve"]
    average_rate = sum(row["churners"] for row in lift_rows) / sum(row["customers"] for row in lift_rows)

    fig, (lift_ax, calibration_ax) = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=100, gridspec_kw={"wspace": 0.25})
    fig.patch.set_facecolor(theme["surface"])

    # Churn rate by risk decile, with the average churn rate as the reference line.
    deciles = [row["decile"] for row in lift_rows]
    rates = [row["churn_rate"] for row in lift_rows]
    _style_axes(lift_ax, theme)
    lift_ax.bar(deciles, rates, width=0.45, color=theme["series"])
    lift_ax.axhline(average_rate, color=theme["muted"], linewidth=1.2, linestyle="--")
    lift_ax.annotate(f"Average {average_rate:.1%}", (10.4, average_rate), xytext=(0, 5), textcoords="offset points",
                     ha="right", color=theme["text_secondary"], fontsize=10)
    lift_ax.annotate(f"{rates[0]:.1%} ({lift_rows[0]['lift']:.1f}x average)", (deciles[0], rates[0]), xytext=(-8, 6),
                     textcoords="offset points", ha="left", color=theme["text"], fontsize=10)
    lift_ax.set_xticks(deciles)
    lift_ax.set_xlim(0.4, 10.6)
    lift_ax.set_ylim(0, 0.85)
    lift_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    lift_ax.set_xlabel("Risk decile (1 = highest predicted risk)", color=theme["text_secondary"], fontsize=10)
    lift_ax.set_title("Churn rate by risk decile, test split", loc="left", color=theme["text"], fontsize=12, pad=12)

    # Predicted probability vs observed churn rate; the diagonal is perfect calibration.
    predicted = [point["mean_predicted_probability"] for point in calibration]
    observed = [point["fraction_of_positives"] for point in calibration]
    _style_axes(calibration_ax, theme)
    calibration_ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    calibration_ax.plot([0, 1], [0, 1], color=theme["muted"], linewidth=1.2, linestyle="--")
    calibration_ax.annotate("Perfect calibration", (0.9, 0.9), xytext=(-8, 4), textcoords="offset points",
                            ha="right", color=theme["text_secondary"], fontsize=10)
    calibration_ax.plot(predicted, observed, color=theme["series"], linewidth=2, marker="o", markersize=8,
                        markeredgecolor=theme["surface"], markeredgewidth=2)
    calibration_ax.set_xlim(0, 1)
    calibration_ax.set_ylim(0, 1)
    calibration_ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    calibration_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    calibration_ax.set_xlabel("Mean predicted churn probability", color=theme["text_secondary"], fontsize=10)
    calibration_ax.set_ylabel("Observed churn rate", color=theme["text_secondary"], fontsize=10)
    calibration_ax.set_title("Calibration, test split", loc="left", color=theme["text"], fontsize=12, pad=12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"model_charts_{theme_name}.png"
    fig.savefig(output_path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return output_path


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    for theme_name in THEMES:
        print(draw(report, theme_name))


if __name__ == "__main__":
    main()
