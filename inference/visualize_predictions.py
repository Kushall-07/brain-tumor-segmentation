from __future__ import annotations

import argparse
from pathlib import Path

from utils.visualization import AxialComparisonSpec, save_axial_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 3D segmentation predictions (axial slices)")
    parser.add_argument("--mri", type=str, required=True, help="Path to input MRI NIfTI (.nii/.nii.gz)")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground-truth mask NIfTI (.nii/.nii.gz)")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted mask NIfTI (.nii/.nii.gz)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "visualizations"),
        help="Output directory for visualization PNGs",
    )
    parser.add_argument("--out_name", type=str, default=None, help="Optional output filename (PNG)")
    parser.add_argument("--num_slices", type=int, default=12, help="Number of axial slices to render")
    parser.add_argument("--cols", type=int, default=4, help="Number of slice columns (each slice uses 3 panels)")
    parser.add_argument("--mri_channel", type=int, default=0, help="If MRI is 4D, which channel to display")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for masks")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title")
    args = parser.parse_args()

    spec = AxialComparisonSpec(
        num_slices=int(args.num_slices),
        cols=int(args.cols),
        mri_channel=int(args.mri_channel),
        overlay_alpha=float(args.alpha),
        title=args.title,
    )

    out_path = save_axial_comparison(
        mri_path=Path(args.mri),
        gt_mask_path=Path(args.gt),
        pred_mask_path=Path(args.pred),
        out_dir=Path(args.out_dir),
        out_name=args.out_name,
        spec=spec,
    )

    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()

