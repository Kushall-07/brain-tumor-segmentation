from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        epochs: list[float] = []
        train_loss: list[float] = []
        wt_dice: list[float] = []
        tc_dice: list[float] = []
        et_dice: list[float] = []
        mean_dice: list[float] = []
        learning_rate: list[float] = []

        for r in reader:
            if not r:
                continue
            epochs.append(float(r["epoch"]))
            train_loss.append(float(r.get("train_loss", "nan")))
            wt_dice.append(float(r.get("wt_dice", "nan")))
            tc_dice.append(float(r.get("tc_dice", "nan")))
            et_dice.append(float(r.get("et_dice", "nan")))
            if "mean_dice" in r and r["mean_dice"]:
                mean_dice.append(float(r["mean_dice"]))
            elif "val_dice" in r and r["val_dice"]:
                mean_dice.append(float(r["val_dice"]))
            else:
                mean_dice.append(float("nan"))
            learning_rate.append(float(r.get("learning_rate", "nan")))

    return {
        "epoch": epochs,
        "train_loss": train_loss,
        "wt_dice": wt_dice,
        "tc_dice": tc_dice,
        "et_dice": et_dice,
        "mean_dice": mean_dice,
        "learning_rate": learning_rate,
    }


def plot_metrics(csv_path: Path, out_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _read_csv(csv_path)
    if len(series["epoch"]) == 0:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(e) for e in series["epoch"]]
    outputs: dict[str, Path] = {}

    def _style(ax, xlabel: str, ylabel: str) -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(epochs, series["train_loss"], linewidth=2.0, label="Train loss")
    _style(ax, "Epoch", "Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "train_loss.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    outputs["train_loss"] = p

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    ax.plot(epochs, series["wt_dice"], linewidth=2.0, label="WT")
    ax.plot(epochs, series["tc_dice"], linewidth=2.0, label="TC")
    ax.plot(epochs, series["et_dice"], linewidth=2.0, label="ET")
    ax.plot(epochs, series["mean_dice"], linewidth=2.2, linestyle="--", label="Mean")
    _style(ax, "Epoch", "Dice")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "brats_region_dice.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    outputs["brats_region_dice"] = p

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(epochs, series["learning_rate"], linewidth=2.0, label="LR")
    _style(ax, "Epoch", "Learning rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "learning_rate.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    outputs["learning_rate"] = p

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BraTS training curves from metrics CSV")
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else (Path.cwd() / "outputs" / "plots")
    paths = plot_metrics(csv_path=csv_path, out_dir=out_dir)
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
