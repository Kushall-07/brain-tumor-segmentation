"""
Generate Graph 2 for the report: class-wise (region-wise) Dice score bar chart.

Source of truth: outputs/exp_swinunetr_4class_et_fixed/checkpoints/best_mean_dice.json
  -> the "best_scores" block.

This block is written by training/train.py at the end of training and records,
for each metric independently, the best value that metric ever reached across
all 75 logged epochs (validation set, see training_metrics.csv):
    best_scores.mean_dice = 0.8796766569217047
    best_scores.wt_dice   = 0.8314783483743667
    best_scores.tc_dice   = 0.9166000187397003
    best_scores.et_dice   = 0.8920635253190994

These values were verified against the repository (not hard-coded from the
prompt) and match exactly. A second, independent evaluation artifact also
exists at outputs/exp_swinunetr_4class_et_fixed/metrics/per_case_metrics.csv
(a 20-case held-out post-hoc audit); its aggregate means differ
(WT=0.8400, TC=0.8934, ET=0.8842, Mean=0.8725) because it averages per-case
Dice over a fixed 20-case set at a single checkpoint, rather than taking each
metric's best value across epochs. This script plots the checkpoint's
best_scores block since those are the values that match the repository's
documented headline results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSON = PROJECT_ROOT / "outputs" / "exp_swinunetr_4class_et_fixed" / "checkpoints" / "best_mean_dice.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "graph2_classwise_dice.png"
OUT_CSV = OUTPUT_DIR / "graph2_classwise_dice_data.csv"

COLOR_WT = "#eb6834"  # orange
COLOR_TC = "#1baf7a"  # aqua
COLOR_ET = "#eda100"  # yellow
COLOR_MEAN_LINE = "#52514e"  # secondary ink (distinguishes Mean from the regions)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main() -> None:
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {SOURCE_JSON}")

    with open(SOURCE_JSON) as f:
        metadata = json.load(f)

    best = metadata["best_scores"]
    print(f"[SOURCE] {SOURCE_JSON} -> best_scores")

    regions = ["WT", "TC", "ET"]
    values = [best["wt_dice"], best["tc_dice"], best["et_dice"]]
    mean_dice = best["mean_dice"]
    colors = [COLOR_WT, COLOR_TC, COLOR_ET]

    out_df = pd.DataFrame({
        "region": regions + ["Mean"],
        "definition": [
            "Whole Tumor (NCR/NET + Edema + ET)",
            "Tumor Core (NCR/NET + ET)",
            "Enhancing Tumor",
            "Mean of WT, TC, ET",
        ],
        "dice_score": values + [mean_dice],
        "dice_percent": [v * 100 for v in values] + [mean_dice * 100],
    })
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[OUTPUT] Underlying data saved to: {OUT_CSV}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(len(regions))
    bars = ax.bar(x, [v * 100 for v in values], width=0.55, color=colors,
                   edgecolor=SURFACE, linewidth=2, zorder=3)

    for rect, v in zip(bars, values):
        ax.annotate(f"{v * 100:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK_PRIMARY, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    ax.set_xlim(-0.7, len(regions) - 1 + 0.7)

    # Mean Dice shown as a distinct reference line, not as a 4th tumor-region bar.
    ax.axhline(mean_dice * 100, color=COLOR_MEAN_LINE, linewidth=1.6, linestyle="--", zorder=2)
    ax.annotate(f"Mean Dice = {mean_dice * 100:.2f}%",
                xy=(-0.65, mean_dice * 100), xytext=(0, 6),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10, color=COLOR_MEAN_LINE, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(["WT\n(Whole Tumor)", "TC\n(Tumor Core)", "ET\n(Enhancing Tumor)"], fontsize=10.5)
    ax.set_ylabel("Dice Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Class-wise Validation Dice Score by Tumor Region\n(SwinUNETR, BraTS 2023 — best value per metric across training)",
                 fontsize=12.5, fontweight="bold", pad=12)

    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, color=GRID, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OUTPUT] Figure saved to: {OUT_PNG}")


if __name__ == "__main__":
    main()
