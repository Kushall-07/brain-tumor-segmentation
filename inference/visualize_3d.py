"""
Interactive 3D brain tumor visualization (Plotly + marching cubes).

Example:
    python -m inference.visualize_3d ^
        --mri path/to/t1c.nii.gz ^
        --pred path/to/case_pred.nii.gz ^
        --exp_name exp_residual_weighted

Outputs HTML (and optional PNG) under outputs/<exp_name>/visualizations/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from utils.mesh_utils import (
    MeshData,
    build_brain_mesh,
    build_class_meshes,
    crop_around_mask,
    export_meshes_obj,
    load_mask_volume,
    load_mri_volume,
    mesh_to_plotly_trace_arrays,
)

DEFAULT_TITLE = "3D Brain Tumor Segmentation Visualization"

# Camera presets (Plotly scene.camera eye/center/up)
CAMERA_PRESETS: dict[str, dict[str, list[float]]] = {
    "default": {"eye": {"x": 1.6, "y": 1.6, "z": 1.3}, "center": {"x": 0, "y": 0, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}},
    "front": {"eye": {"x": 0.0, "y": 2.2, "z": 0.2}, "center": {"x": 0, "y": 0, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}},
    "side": {"eye": {"x": 2.4, "y": 0.0, "z": 0.2}, "center": {"x": 0, "y": 0, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}},
    "top": {"eye": {"x": 0.0, "y": 0.0, "z": 2.6}, "center": {"x": 0, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0}},
}


@dataclass(frozen=True)
class VisualizationConfig:
    """Rendering and performance options."""

    title: str = DEFAULT_TITLE
    brain_opacity: float = 0.15
    tumor_opacity: float = 0.85
    dark_mode: bool = True
    show_axes: bool = False
    camera_preset: str = "default"
    auto_rotate: bool = False
    rotate_frames: int = 48
    downsample_brain: int = 2
    downsample_tumor: int = 1
    marching_step_brain: int = 2
    marching_step_tumor: int = 1
    crop_margin_vox: int = 16
    class_ids: tuple[int, ...] = (1, 2, 3)
    # Lighting
    ambient: float = 0.55
    diffuse: float = 0.85
    specular: float = 0.35
    roughness: float = 0.55


def _hex_to_rgb(color: str) -> str:
    c = color.strip()
    if c.startswith("rgba") or c.startswith("rgb"):
        return c
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        return f"rgb({r},{g},{b})"
    return c


def _mesh_trace(
    mesh: MeshData,
    *,
    opacity: float,
    visible: bool | Literal["legendonly"] = True,
    lighting: Optional[dict[str, float]] = None,
    lightposition: Optional[dict[str, float]] = None,
    hovertemplate: str = "%{fullData.name}<extra></extra>",
) -> "object":
    import plotly.graph_objects as go

    arrays = mesh_to_plotly_trace_arrays(mesh)
    lit = lighting or {}
    lp = lightposition or {"x": 100, "y": 200, "z": 0}

    return go.Mesh3d(
        **arrays,
        name=mesh.name,
        color=_hex_to_rgb(mesh.color_rgba),
        opacity=float(opacity),
        flatshading=False,
        visible=visible,
        hovertemplate=hovertemplate,
        lighting={
            "ambient": float(lit.get("ambient", 0.55)),
            "diffuse": float(lit.get("diffuse", 0.85)),
            "specular": float(lit.get("specular", 0.35)),
            "roughness": float(lit.get("roughness", 0.55)),
            "fresnel": 0.2,
        },
        lightposition=lp,
    )


def _scene_layout(
    cfg: VisualizationConfig,
    *,
    domain_x: Optional[list[float]] = None,
    domain_y: Optional[list[float]] = None,
) -> dict:
    preset = CAMERA_PRESETS.get(cfg.camera_preset, CAMERA_PRESETS["default"])
    bg = "#0b0f19" if cfg.dark_mode else "#ffffff"
    grid = "#334155" if cfg.dark_mode else "#cbd5e1"
    font = "#e2e8f0" if cfg.dark_mode else "#0f172a"

    scene: dict = {
        "xaxis": {
            "visible": cfg.show_axes,
            "title": "X" if cfg.show_axes else "",
            "gridcolor": grid,
            "color": font,
            "backgroundcolor": bg,
        },
        "yaxis": {
            "visible": cfg.show_axes,
            "title": "Y" if cfg.show_axes else "",
            "gridcolor": grid,
            "color": font,
            "backgroundcolor": bg,
        },
        "zaxis": {
            "visible": cfg.show_axes,
            "title": "Z" if cfg.show_axes else "",
            "gridcolor": grid,
            "color": font,
            "backgroundcolor": bg,
        },
        "bgcolor": bg,
        "camera": preset,
        "aspectmode": "data",
    }
    if domain_x is not None:
        scene["domain"] = {"x": domain_x, "y": domain_y or [0.0, 1.0]}
    return scene


def _orbit_camera_frames(n_frames: int, radius: float = 2.2, height: float = 1.2) -> list[dict]:
    """Generate camera 'eye' positions on a circle for auto-rotation."""
    frames: list[dict] = []
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_frames), endpoint=False)
    for theta in angles:
        frames.append(
            {
                "eye": {
                    "x": float(radius * np.cos(theta)),
                    "y": float(radius * np.sin(theta)),
                    "z": float(height),
                },
                "center": {"x": 0, "y": 0, "z": 0},
                "up": {"x": 0, "y": 0, "z": 1},
            }
        )
    return frames


def build_meshes_from_volumes(
    mri: np.ndarray,
    pred: np.ndarray,
    *,
    cfg: VisualizationConfig,
) -> tuple[list[MeshData], list[MeshData]]:
    """Build brain + tumor meshes (with optional crop for speed)."""
    pred_i = np.rint(pred).astype(np.int16)
    mri_c, pred_c = crop_around_mask(mri.astype(np.float32), pred_i, margin_vox=cfg.crop_margin_vox)

    brain = build_brain_mesh(
        mri_c,
        downsample=cfg.downsample_brain,
        step_size=cfg.marching_step_brain,
    )
    tumors = build_class_meshes(
        pred_c,
        class_ids=cfg.class_ids,
        downsample=cfg.downsample_tumor,
        step_size=cfg.marching_step_tumor,
    )
    return [brain], tumors


def create_interactive_figure(
    brain_meshes: Sequence[MeshData],
    tumor_meshes: Sequence[MeshData],
    *,
    cfg: VisualizationConfig,
    gt_tumor_meshes: Optional[Sequence[MeshData]] = None,
) -> "object":
    """
    Build a Plotly Figure with brain + tumor Mesh3d traces.

    If gt_tumor_meshes is provided, renders side-by-side Pred vs GT scenes.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lighting = {
        "ambient": cfg.ambient,
        "diffuse": cfg.diffuse,
        "specular": cfg.specular,
        "roughness": cfg.roughness,
    }

    compare_gt = gt_tumor_meshes is not None and len(gt_tumor_meshes) > 0

    if compare_gt:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("Prediction", "Ground Truth"),
            horizontal_spacing=0.04,
        )
        fig.update_layout(
            scene=_scene_layout(cfg, domain_x=[0.0, 0.46]),
            scene2=_scene_layout(cfg, domain_x=[0.54, 1.0]),
        )
    else:
        fig = go.Figure()
        fig.update_layout(scene=_scene_layout(cfg))

    def _add_traces(brain: Sequence[MeshData], tumors: Sequence[MeshData], *, col: int, suffix: str) -> None:
        for bm in brain:
            trace = _mesh_trace(
                bm,
                opacity=cfg.brain_opacity,
                lighting=lighting,
                hovertemplate=f"Brain envelope{suffix}<extra></extra>",
            )
            if compare_gt:
                fig.add_trace(trace, row=1, col=col)
            else:
                fig.add_trace(trace)

        for tm in tumors:
            trace = _mesh_trace(
                tm,
                opacity=cfg.tumor_opacity,
                lighting=lighting,
                hovertemplate=f"{tm.name}{suffix}<extra></extra>",
            )
            if compare_gt:
                fig.add_trace(trace, row=1, col=col)
            else:
                fig.add_trace(trace)

    if compare_gt:
        _add_traces(brain_meshes, tumor_meshes, col=1, suffix=" (Pred)")
        _add_traces(brain_meshes, gt_tumor_meshes or [], col=2, suffix=" (GT)")
    else:
        _add_traces(brain_meshes, tumor_meshes, col=1, suffix="")

    paper_bg = "#0b0f19" if cfg.dark_mode else "#f8fafc"
    font_col = "#e2e8f0" if cfg.dark_mode else "#0f172a"

    fig.update_layout(
        title={"text": cfg.title, "x": 0.5, "xanchor": "center", "font": {"size": 20, "color": font_col}},
        paper_bgcolor=paper_bg,
        font={"color": font_col, "family": "Segoe UI, Roboto, Helvetica, Arial, sans-serif"},
        margin=dict(l=8, r=8, t=56, b=8),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(0,0,0,0)" if cfg.dark_mode else "rgba(255,255,255,0.8)",
        },
        hovermode="closest",
    )

    updatemenus: list[dict] = []

    # Auto-rotation animation (single-scene layouts only)
    if cfg.auto_rotate and not compare_gt:
        import plotly.graph_objects as go

        frames_cam = _orbit_camera_frames(cfg.rotate_frames)
        fig.frames = [
            go.Frame(name=f"f{i:03d}", layout=go.Layout(scene=dict(camera=cam)))
            for i, cam in enumerate(frames_cam)
        ]
        updatemenus.append(
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.98,
                "buttons": [
                    {
                        "label": "▶ Rotate",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "■ Stop",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        )

    # Camera preset dropdown
    cam_buttons = []
    for name, cam in CAMERA_PRESETS.items():
        if compare_gt:
            relayout = {"scene.camera": cam, "scene2.camera": cam}
        else:
            relayout = {"scene.camera": cam}
        cam_buttons.append({"label": name.capitalize(), "method": "relayout", "args": [relayout]})

    updatemenus.append(
        {
            "type": "dropdown",
            "direction": "down",
            "x": 0.98,
            "y": 0.98,
            "xanchor": "right",
            "buttons": cam_buttons,
        }
    )

    fig.update_layout(updatemenus=updatemenus)

    return fig


def save_html(fig: "object", out_path: Path, *, include_plotlyjs: str = "cdn") -> Path:
    import plotly.io as pio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_path), include_plotlyjs=include_plotlyjs, full_html=True, auto_open=False)
    return out_path


def save_png(fig: "object", out_path: Path, *, width: int = 1600, height: int = 1000, scale: int = 2) -> Path:
    """Export static PNG (requires kaleido: pip install kaleido)."""
    import plotly.io as pio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pio.write_image(fig, str(out_path), width=int(width), height=int(height), scale=int(scale))
    except Exception as e:
        raise RuntimeError(
            "PNG export failed. Install kaleido: pip install kaleido"
        ) from e
    return out_path


def resolve_out_dir(out_dir: Optional[str], exp_name: Optional[str]) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "outputs"
    if out_dir is not None:
        return Path(out_dir)
    if exp_name is not None:
        return outputs_root / str(exp_name) / "visualizations"
    return outputs_root / "visualizations"


def run_visualization(
    *,
    mri_path: Path,
    pred_path: Path,
    out_dir: Path,
    gt_path: Optional[Path] = None,
    out_stem: Optional[str] = None,
    cfg: VisualizationConfig,
    mri_channel: int = 0,
    write_png: bool = False,
    export_obj: bool = False,
) -> dict[str, Path]:
    """End-to-end: load volumes → meshes → Plotly → export."""
    mri_vol = load_mri_volume(mri_path, channel=mri_channel)
    pred_vol = load_mask_volume(pred_path)

    if pred_vol.array.shape != mri_vol.array.shape:
        raise ValueError(
            f"MRI shape {mri_vol.array.shape} != mask shape {pred_vol.array.shape}. "
            "Use matching NIfTI pairs."
        )

    brain_meshes, tumor_meshes = build_meshes_from_volumes(mri_vol.array, pred_vol.array, cfg=cfg)

    gt_tumors: list[MeshData] = []
    if gt_path is not None:
        gt_vol = load_mask_volume(gt_path)
        if gt_vol.array.shape != mri_vol.array.shape:
            raise ValueError(f"GT shape {gt_vol.array.shape} != MRI shape {mri_vol.array.shape}")
        _, gt_tumors = build_meshes_from_volumes(mri_vol.array, gt_vol.array, cfg=cfg)

    fig = create_interactive_figure(
        brain_meshes,
        tumor_meshes,
        cfg=cfg,
        gt_tumor_meshes=gt_tumors if gt_tumors else None,
    )

    stem = out_stem or pred_path.stem.replace(".nii", "").replace("_pred", "")
    if stem.endswith(".nii"):
        stem = stem[:-4]
    html_path = out_dir / f"{stem}_3d.html"
    save_html(fig, html_path)

    outputs: dict[str, Path] = {"html": html_path}

    if write_png:
        png_path = out_dir / f"{stem}_3d.png"
        save_png(fig, png_path)
        outputs["png"] = png_path

    if export_obj:
        obj_dir = out_dir / f"{stem}_meshes"
        all_meshes = list(brain_meshes) + list(tumor_meshes)
        export_meshes_obj(all_meshes, obj_dir, prefix="pred")
        outputs["obj_dir"] = obj_dir

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive 3D brain tumor visualization (Plotly HTML export)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mri", type=str, required=True, help="MRI NIfTI path (.nii/.nii.gz)")
    p.add_argument("--pred", type=str, required=True, help="Predicted mask NIfTI")
    p.add_argument("--gt", type=str, default=None, help="Optional ground-truth mask for side-by-side view")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory for HTML/PNG")
    p.add_argument("--exp_name", type=str, default=None, help="Experiment name → outputs/<exp_name>/visualizations/")
    p.add_argument("--out_name", type=str, default=None, help="Output file stem (without extension)")
    p.add_argument("--mri_channel", type=int, default=0, help="4D MRI channel index")
    p.add_argument("--brain_opacity", type=float, default=0.15)
    p.add_argument("--tumor_opacity", type=float, default=0.85)
    p.add_argument(
        "--light_mode",
        action="store_true",
        help="Use light background (default is dark presentation mode)",
    )
    p.add_argument("--show_axes", action="store_true", help="Show 3D axis lines and labels")
    p.add_argument("--camera", type=str, default="default", choices=sorted(CAMERA_PRESETS.keys()))
    p.add_argument("--auto_rotate", action="store_true", help="Add orbit animation controls")
    p.add_argument("--save_png", action="store_true", help="Also export PNG (requires kaleido)")
    p.add_argument("--export_obj", action="store_true", help="Export meshes as OBJ files")
    p.add_argument("--downsample_brain", type=int, default=2, help="Stride downsample before brain meshing")
    p.add_argument("--downsample_tumor", type=int, default=1, help="Stride downsample before tumor meshing")
    p.add_argument("--title", type=str, default=DEFAULT_TITLE)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    dark = not bool(args.light_mode)
    cfg = VisualizationConfig(
        title=str(args.title),
        brain_opacity=float(args.brain_opacity),
        tumor_opacity=float(args.tumor_opacity),
        dark_mode=dark,
        show_axes=bool(args.show_axes),
        camera_preset=str(args.camera),
        auto_rotate=bool(args.auto_rotate),
        downsample_brain=int(args.downsample_brain),
        downsample_tumor=int(args.downsample_tumor),
    )

    out_dir = resolve_out_dir(args.out_dir, args.exp_name)
    gt_path = Path(args.gt) if args.gt else None

    outputs = run_visualization(
        mri_path=Path(args.mri),
        pred_path=Path(args.pred),
        out_dir=out_dir,
        gt_path=gt_path,
        out_stem=args.out_name,
        cfg=cfg,
        mri_channel=int(args.mri_channel),
        write_png=bool(args.save_png),
        export_obj=bool(args.export_obj),
    )

    print("Saved interactive 3D visualization:")
    for k, p in outputs.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
