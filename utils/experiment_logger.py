from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class MetricRow:
    epoch: int
    train_loss: float
    val_dice: float
    learning_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "epoch": int(self.epoch),
            "train_loss": float(self.train_loss),
            "val_dice": float(self.val_dice),
            "learning_rate": float(self.learning_rate),
        }


class ExperimentLogger:
    """CSV metrics logger + matplotlib curve plotting."""

    fieldnames: tuple[str, ...] = ("epoch", "train_loss", "val_dice", "learning_rate")

    def __init__(self, log_dir: str | Path, filename: str = "metrics.csv") -> None:
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
                out.append(
                    {
                        "epoch": float(r["epoch"]),
                        "train_loss": float(r["train_loss"]),
                        "val_dice": float(r["val_dice"]),
                        "learning_rate": float(r["learning_rate"]),
                    }
                )
            return out

    def plot_curves(self) -> dict[str, Path]:
        """Generate paper-ready PNG plots in the log directory."""
        rows = self.read_rows()
        if len(rows) == 0:
            return {}

        # Use a non-interactive backend (works on servers/Windows without display).
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(r["epoch"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        val_dice = [float(r["val_dice"]) for r in rows]

        outputs: dict[str, Path] = {}

        def _style_ax(ax, xlabel: str, ylabel: str) -> None:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

        # Loss curve
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        ax.plot(epochs, train_loss, label="Train loss", linewidth=2.0)
        _style_ax(ax, xlabel="Epoch", ylabel="Loss")
        ax.legend()
        fig.tight_layout()
        loss_path = self.log_dir / "loss_curve.png"
        fig.savefig(str(loss_path), bbox_inches="tight")
        plt.close(fig)
        outputs["loss_curve"] = loss_path

        # Dice curve
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        ax.plot(epochs, val_dice, label="Val Dice", linewidth=2.0)
        _style_ax(ax, xlabel="Epoch", ylabel="Dice")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        fig.tight_layout()
        dice_path = self.log_dir / "dice_curve.png"
        fig.savefig(str(dice_path), bbox_inches="tight")
        plt.close(fig)
        outputs["dice_curve"] = dice_path

        return outputs


def _write_csv(path: Path, rows: Iterable[Mapping[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExperimentLogger.fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))

