"""
Generate Graph 1 for the report: Dice score vs. epoch.

Source of truth: outputs/exp_swinunetr_4class_et_fixed/logs/training_metrics.csv
This CSV is written once per epoch by utils.experiment_logger.ExperimentLogger
during training/train.py. Per epoch it logs:
    - train_loss  : average training loss over the training set (NOT a Dice score)
    - wt_dice, tc_dice, et_dice, mean_dice : Dice scores computed by validate()
      on the held-out validation loader (val_loader), i.e. VALIDATION Dice.

IMPORTANT: train.py never computes a Dice score on the training set (see
train_one_epoch(), which returns only a scalar loss). No "training Dice"
value exists anywhere in this repository (no CSV, JSON, or TensorBoard log
contains one), so this figure plots the real per-epoch VALIDATION Dice curve
only (mean + per-region) rather than fabricating a training-Dice series.

No values are invented, smoothed, or interpolated: every point plotted here
is a row of the source CSV, verbatim.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = PROJECT_ROOT / "outputs" / "exp_swinunetr_4class_et_fixed" / "logs" / "training_metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "graph1_validation_dice_vs_epoch.png"
OUT_CSV = OUTPUT_DIR / "graph1_validation_dice_vs_epoch_data.csv"

# Fixed categorical palette (validated: scripts/validate_palette.js), light-mode chart
COLOR_MEAN = "#2a78d6"  # blue
COLOR_WT = "#eb6834"    # orange
COLOR_TC = "#1baf7a"    # aqua
COLOR_ET = "#eda100"    # yellow
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Training history not found: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)
    print(f"[SOURCE] {SOURCE_CSV}  ({len(df)} epochs, columns={list(df.columns)})")

    # Persist exactly what is plotted, for verification.
    out_df = df[["epoch", "wt_dice", "tc_dice", "et_dice", "mean_dice"]].copy()
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[OUTPUT] Underlying data saved to: {OUT_CSV}")

    epochs = df["epoch"]
    best_idx = df["mean_dice"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_mean = float(df.loc[best_idx, "mean_dice"])

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

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(epochs, df["mean_dice"], color=COLOR_MEAN, linewidth=2.4, label="Mean Validation Dice", zorder=4)
    ax.plot(epochs, df["wt_dice"], color=COLOR_WT, linewidth=1.6, label="WT Validation Dice", zorder=3)
    ax.plot(epochs, df["tc_dice"], color=COLOR_TC, linewidth=1.6, label="TC Validation Dice", zorder=3)
    ax.plot(epochs, df["et_dice"], color=COLOR_ET, linewidth=1.6, label="ET Validation Dice", zorder=3)

    ax.scatter([best_epoch], [best_mean], color=INK_PRIMARY, edgecolor=SURFACE, s=70, zorder=5)
    ax.annotate(
        f"Best mean Dice: {best_mean:.4f} (epoch {best_epoch})",
        xy=(best_epoch, best_mean), xytext=(10, -14), textcoords="offset points",
        fontsize=9.5, color=INK_PRIMARY,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INK_MUTED, linewidth=0.6),
    )

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dice Score", fontsize=12, fontweight="bold")
    ax.set_title("Validation Dice Score Across Epochs\n(SwinUNETR, BraTS 2023 — no training-set Dice was logged)",
                 fontsize=13, fontweight="bold", pad=12)

    ax.set_xlim(0, float(epochs.max()) + 1)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.7, color=GRID, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.9, edgecolor=GRID)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[OUTPUT] Figure saved to: {OUT_PNG}")


if __name__ == "__main__":
    main()
