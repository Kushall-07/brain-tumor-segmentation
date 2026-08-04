from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r]


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _format_float(v: float, ndigits: int = 6) -> str:
    if v != v:
        return "nan"
    return f"{v:.{ndigits}f}"


def generate_metrics_summary(csv_path: Path) -> dict[str, float | int]:
    rows = _read_rows(csv_path)
    if len(rows) == 0:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    def col(name: str, fallback: str | None = None) -> list[float]:
        out: list[float] = []
        for r in rows:
            v = r.get(name)
            if (v is None or str(v).strip() == "") and fallback:
                v = r.get(fallback)
            out.append(_to_float(v))
        return out

    epochs = col("epoch")
    train_loss = col("train_loss")
    wt = col("wt_dice")
    tc = col("tc_dice")
    et = col("et_dice")
    mean_d = col("mean_dice", "val_dice")
    lr = col("learning_rate")

    best_idx = max(range(len(mean_d)), key=lambda i: mean_d[i])

    return {
        "Best Mean Dice": mean_d[best_idx],
        "Best WT Dice": max(wt) if wt else float("nan"),
        "Best TC Dice": max(tc) if tc else float("nan"),
        "Best ET Dice": max(et) if et else float("nan"),
        "Final Mean Dice": mean_d[-1],
        "Final WT Dice": wt[-1] if wt else float("nan"),
        "Final TC Dice": tc[-1] if tc else float("nan"),
        "Final ET Dice": et[-1] if et else float("nan"),
        "Best Epoch": int(_to_float(epochs[best_idx], 0)),
        "Minimum Train Loss": min(train_loss) if train_loss else float("nan"),
        "Final Learning Rate": lr[-1] if lr else float("nan"),
    }


def write_performance_metrics_csv(metrics: dict[str, float | int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([str(k), str(v)])


def write_performance_metrics_md(metrics: dict[str, float | int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in metrics.items():
        vv = _format_float(v, 6) if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {vv} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_performance_metrics_png(metrics: dict[str, float | int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [[str(k), _format_float(v, 6) if isinstance(v, float) else str(v)] for k, v in metrics.items()]
    fig_h = max(2.0, 0.45 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(7.0, fig_h), dpi=300)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BraTS region performance summary table")
    parser.add_argument("--csv", type=str, required=True, help="Path to training_metrics.csv")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = generate_metrics_summary(csv_path)
    write_performance_metrics_csv(metrics, out_dir / "performance_metrics.csv")
    write_performance_metrics_md(metrics, out_dir / "performance_metrics.md")
    write_performance_metrics_png(metrics, out_dir / "performance_metrics.png")

    print(f"Saved metrics to: {out_dir}")


if __name__ == "__main__":
    main()
