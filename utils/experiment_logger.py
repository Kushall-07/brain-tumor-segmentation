from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class MetricRow:
    epoch: int
    train_loss: float
    learning_rate: float
    wt_dice: float
    tc_dice: float
    et_dice: float
    mean_dice: float
    # Back-compat alias for older plotters
    val_dice: float | None = None

    def as_dict(self) -> dict[str, float | int]:
        mean = float(self.mean_dice if self.val_dice is None else self.val_dice)
        return {
            "epoch": int(self.epoch),
            "train_loss": float(self.train_loss),
            "wt_dice": float(self.wt_dice),
            "tc_dice": float(self.tc_dice),
            "et_dice": float(self.et_dice),
            "mean_dice": mean,
            "learning_rate": float(self.learning_rate),
        }


class ExperimentLogger:
    """CSV metrics logger + matplotlib curve plotting (region Dice)."""

    fieldnames: tuple[str, ...] = (
        "epoch",
        "train_loss",
        "wt_dice",
        "tc_dice",
        "et_dice",
        "mean_dice",
        "learning_rate",
    )

    def __init__(self, log_dir: str | Path, filename: str = "training_metrics.csv") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / filename

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.fieldnames))
                writer.writeheader()

    def log(self, row: MetricRow) -> None:
        data = row.as_dict()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.fieldnames))
            writer.writerow(data)

    def read_rows(self) -> list[dict[str, float]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out: list[dict[str, float]] = []
            for r in reader:
                if not r:
                    continue
                # Support legacy val_dice-only CSVs
                if "mean_dice" in r and r["mean_dice"] not in (None, ""):
                    mean = float(r["mean_dice"])
                elif "val_dice" in r and r["val_dice"] not in (None, ""):
                    mean = float(r["val_dice"])
                else:
                    mean = 0.0
                out.append(
                    {
                        "epoch": float(r["epoch"]),
                        "train_loss": float(r["train_loss"]),
                        "wt_dice": float(r.get("wt_dice") or 0.0),
                        "tc_dice": float(r.get("tc_dice") or 0.0),
                        "et_dice": float(r.get("et_dice") or 0.0),
                        "mean_dice": mean,
                        "learning_rate": float(r["learning_rate"]),
                    }
                )
            return out

    def plot_curves(self) -> dict[str, Path]:
        rows = self.read_rows()
        if len(rows) == 0:
            return {}

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(r["epoch"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        outputs: dict[str, Path] = {}

        def _style_ax(ax, xlabel: str, ylabel: str) -> None:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        ax.plot(epochs, train_loss, label="Train loss", linewidth=2.0)
        _style_ax(ax, xlabel="Epoch", ylabel="Loss")
        ax.legend()
        fig.tight_layout()
        loss_path = self.log_dir / "loss_curve.png"
        fig.savefig(str(loss_path), bbox_inches="tight")
        plt.close(fig)
        outputs["loss_curve"] = loss_path

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        ax.plot(epochs, [r["wt_dice"] for r in rows], label="WT Dice", linewidth=2.0)
        ax.plot(epochs, [r["tc_dice"] for r in rows], label="TC Dice", linewidth=2.0)
        ax.plot(epochs, [r["et_dice"] for r in rows], label="ET Dice", linewidth=2.0)
        ax.plot(epochs, [r["mean_dice"] for r in rows], label="Mean Dice", linewidth=2.0)
        _style_ax(ax, xlabel="Epoch", ylabel="Dice")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        fig.tight_layout()
        dice_path = self.log_dir / "brats_region_dice.png"
        fig.savefig(str(dice_path), bbox_inches="tight")
        plt.close(fig)
        outputs["brats_region_dice"] = dice_path

        return outputs


def _write_csv(path: Path, rows: Iterable[Mapping[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExperimentLogger.fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
