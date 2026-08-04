from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def _find_case_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(str(root))
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _find_pred_files_flat(pred_dir: Path) -> list[Path]:
    return sorted(pred_dir.glob("*_pred.nii.gz")) + sorted(pred_dir.glob("*_pred.nii"))


def _find_single_file(case_dir: Path, token: str) -> Path:
    matches = sorted(case_dir.glob(f"*{token}*.nii.gz")) + sorted(case_dir.glob(f"*{token}*.nii"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching '*{token}*.nii[.gz]' in: {case_dir}")
    if len(matches) > 1:
        raise FileExistsError(f"Multiple files matching '*{token}*' in {case_dir}: {[m.name for m in matches]}")
    return matches[0]


def _load_mask_to_zyx(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D mask NIfTI at {path}, got ndim={data.ndim}")

    # Repo convention: [X,Y,Z] -> [Z,X,Y] == [D,H,W]
    data_zyx = np.transpose(data, (2, 0, 1))
    return data_zyx.astype(np.int64, copy=False)


def _collect_pairs(pred_dir: Path, gt_dir: Path) -> list[tuple[str, Path, Path]]:
    pred_cases = _find_case_dirs(pred_dir)

    # Support two layouts:
    # 1) pred_dir/<case_id>/*pred*.nii.gz
    # 2) pred_dir/*_pred.nii.gz (flat)
    if len(pred_cases) == 0:
        pred_files = _find_pred_files_flat(pred_dir)
        if len(pred_files) == 0:
            raise ValueError(f"No case subfolders or '*_pred.nii[.gz]' files found in pred_dir: {pred_dir}")

        out: list[tuple[str, Path, Path]] = []
        for pred_path in pred_files:
            name = pred_path.name
            case_id = name
            if case_id.endswith("_pred.nii.gz"):
                case_id = case_id[: -len("_pred.nii.gz")]
            elif case_id.endswith("_pred.nii"):
                case_id = case_id[: -len("_pred.nii")]

            gt_case_dir = gt_dir / case_id
            if not gt_case_dir.exists():
                raise FileNotFoundError(f"GT case folder not found: {gt_case_dir}")
            gt_mask = _find_single_file(gt_case_dir, "seg")
            out.append((case_id, pred_path, gt_mask))
        return out

    out: list[tuple[str, Path, Path]] = []
    for case in pred_cases:
        case_id = case.name
        pred_mask = _find_single_file(case, "pred")

        gt_case_dir = gt_dir / case_id
        if not gt_case_dir.exists():
            raise FileNotFoundError(f"GT case folder not found: {gt_case_dir}")
        gt_mask = _find_single_file(gt_case_dir, "seg")

        out.append((case_id, pred_mask, gt_mask))
    return out


def compute_voxel_confusion_matrix(
    pred_dir: Path,
    gt_dir: Path,
    *,
    ignore_background: bool = False,
    labels: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    pairs = _collect_pairs(pred_dir=pred_dir, gt_dir=gt_dir)

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for case_id, pred_path, gt_path in pairs:
        pred = _load_mask_to_zyx(pred_path)
        gt = _load_mask_to_zyx(gt_path)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred={pred.shape} gt={gt.shape}")

        y_true_all.append(gt.reshape(-1))
        y_pred_all.append(pred.reshape(-1))

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    if labels is None:
        labels = sorted(set(np.unique(y_true).tolist() + np.unique(y_pred).tolist()))

    labels = [int(x) for x in labels]
    if ignore_background and 0 in labels:
        labels = [x for x in labels if x != 0]

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    return cm, labels


def save_confusion_matrix_csv(cm: np.ndarray, labels: Sequence[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_s = [str(int(l)) for l in labels]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *labels_s])
        for i, row in enumerate(cm.tolist()):
            writer.writerow([labels_s[i], *row])


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[int],
    out_path: Path,
    *,
    title: str = "Voxel-wise Confusion Matrix",
    normalize: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_plot = cm.astype(np.float64)
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sum + 1e-12)

    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=300)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels_s = [str(int(l)) for l in labels]
    ax.set(
        xticks=np.arange(len(labels_s)),
        yticks=np.arange(len(labels_s)),
        xticklabels=labels_s,
        yticklabels=labels_s,
        xlabel="Predicted class",
        ylabel="Ground-truth class",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate
    fmt = ".2f" if normalize else "d"
    thresh = cm_plot.max() * 0.6 if cm_plot.size > 0 else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j] if normalize else int(cm[i, j])
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    ax.grid(False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute voxel-wise confusion matrix for segmentation masks")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing per-case predicted masks")
    parser.add_argument("--gt_dir", type=str, required=True, help="BraTS validation directory containing GT masks")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (writes CSV + PNG)")
    parser.add_argument("--ignore_background", action="store_true", help="Ignore class 0")
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit class labels (e.g. 0 1 2 3). If omitted, inferred from data.",
    )
    parser.add_argument("--normalize", action="store_true", help="Row-normalize confusion matrix for plotting")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.labels is None:
        cm, labels = compute_voxel_confusion_matrix(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            ignore_background=bool(args.ignore_background),
            labels=None,
        )
    else:
        labels_in = [int(x) for x in args.labels]
        cm, labels = compute_voxel_confusion_matrix(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            ignore_background=bool(args.ignore_background),
            labels=labels_in,
        )

    csv_path = out_dir / "confusion_matrix.csv"
    png_path = out_dir / "confusion_matrix.png"

    save_confusion_matrix_csv(cm=cm, labels=labels, out_path=csv_path)
    plot_confusion_matrix(cm=cm, labels=labels, out_path=png_path, normalize=bool(args.normalize))

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
