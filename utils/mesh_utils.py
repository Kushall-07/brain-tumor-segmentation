from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from skimage.measure import marching_cubes


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshData:
    """Triangle mesh suitable for Plotly Mesh3d."""

    vertices: np.ndarray  # [N, 3] float32
    faces: np.ndarray  # [M, 3] int32 (triangle vertex indices)
    name: str
    color_rgba: str  # '#RRGGBB' or 'rgba(r,g,b,a)'


@dataclass(frozen=True)
class VolumeData:
    """Canonical [D, H, W] volume loaded from NIfTI."""

    array: np.ndarray  # float32 [D, H, W]
    affine: np.ndarray
    source_path: Path


# ---------------------------------------------------------------------------
# Loading (repo convention: [X,Y,Z] -> [Z,X,Y] == [D,H,W])
# ---------------------------------------------------------------------------


def load_nifti_volume(path: str | Path, *, dtype: np.dtype = np.float32) -> VolumeData:
    import nibabel as nib

    p = Path(path)
    img = nib.as_closest_canonical(nib.load(str(p)))
    data = img.get_fdata(dtype=dtype)

    if data.ndim == 3:
        vol = np.transpose(data, (2, 0, 1))
    elif data.ndim == 4:
        # Caller may pass 4D MRI; default channel 0 unless specified elsewhere.
        vol = np.transpose(data[..., 0], (2, 0, 1))
    else:
        raise ValueError(f"Unsupported NIfTI ndim={data.ndim} for {p}")

    vol = np.asarray(vol, dtype=np.float32)
    return VolumeData(array=vol, affine=np.asarray(img.affine), source_path=p)


def load_mri_volume(path: str | Path, *, channel: int = 0) -> VolumeData:
    import nibabel as nib

    p = Path(path)
    img = nib.as_closest_canonical(nib.load(str(p)))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 3:
        vol = np.transpose(data, (2, 0, 1))
    elif data.ndim == 4:
        c = int(channel)
        if c < 0 or c >= int(data.shape[-1]):
            raise ValueError(f"mri channel={c} out of range for shape {data.shape}")
        vol = np.transpose(data[..., c], (2, 0, 1))
    else:
        raise ValueError(f"Unsupported MRI ndim={data.ndim} for {p}")

    return VolumeData(array=np.asarray(vol, dtype=np.float32), affine=np.asarray(img.affine), source_path=p)


def load_mask_volume(path: str | Path) -> VolumeData:
    vol = load_nifti_volume(path, dtype=np.float32)
    # Preserve integer class labels after load.
    vol = VolumeData(array=np.rint(vol.array).astype(np.int16), affine=vol.affine, source_path=vol.source_path)
    return vol


# ---------------------------------------------------------------------------
# Preprocessing for marching cubes (performance + stability)
# ---------------------------------------------------------------------------


def normalize_volume(vol: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    """Percentile-based normalization to [0, 1]."""
    v = np.asarray(vol, dtype=np.float32)
    lo = float(np.percentile(v, pmin))
    hi = float(np.percentile(v, pmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            return np.zeros_like(v, dtype=np.float32)
    return np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def downsample_volume(vol: np.ndarray, factor: int = 2) -> np.ndarray:
    """Fast stride downsampling for large MRI volumes."""
    f = max(1, int(factor))
    if f == 1:
        return vol
    return vol[::f, ::f, ::f]


def smooth_volume(vol: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Light Gaussian smoothing to stabilize marching cubes."""
    if sigma <= 0:
        return vol
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(vol, sigma=float(sigma)).astype(np.float32)


def crop_around_mask(
    mri: np.ndarray,
    mask: np.ndarray,
    margin_vox: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop volumes to a bounding box around foreground (+margin)."""
    fg = np.argwhere(mask > 0)
    if fg.size == 0:
        return mri, mask

    z0, y0, x0 = fg.min(axis=0)
    z1, y1, x1 = fg.max(axis=0)
    m = int(margin_vox)

    d, h, w = mask.shape
    zz0 = max(0, int(z0) - m)
    yy0 = max(0, int(y0) - m)
    xx0 = max(0, int(x0) - m)
    zz1 = min(d, int(z1) + m + 1)
    yy1 = min(h, int(y1) + m + 1)
    xx1 = min(w, int(x1) + m + 1)

    sl = (slice(zz0, zz1), slice(yy0, yy1), slice(xx0, xx1))
    return mri[sl], mask[sl]


def brain_envelope_mask(mri_norm: np.ndarray, percentile: float = 18.0) -> np.ndarray:
    """Build a soft brain/skull envelope from normalized MRI intensities."""
    v = np.asarray(mri_norm, dtype=np.float32)
    nz = v[v > 0]
    if nz.size == 0:
        return np.zeros_like(v, dtype=bool)

    thr = float(np.percentile(nz, percentile))
    mask = v >= thr

    # Fill holes and keep largest connected component for cleaner surface.
    from scipy.ndimage import binary_closing, binary_fill_holes, label

    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, iterations=2)

    labeled, n = label(mask)
    if n > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        keep = int(counts.argmax())
        mask = labeled == keep

    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def marching_cubes_mesh(
    volume: np.ndarray,
    *,
    level: float = 0.5,
    step_size: int = 2,
    allow_degenerate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run marching cubes and return (vertices [N,3], faces [M,3]).

    step_size > 1 speeds up extraction on large volumes.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.size == 0:
        raise ValueError("Empty volume passed to marching_cubes")

    verts, faces, _normals, _vals = marching_cubes(
        vol,
        level=float(level),
        step_size=max(1, int(step_size)),
        allow_degenerate=bool(allow_degenerate),
    )
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    return verts, faces


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int = 120_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform face decimation by stride (fast, no extra deps)."""
    if faces.shape[0] <= max_faces:
        return vertices, faces

    stride = int(np.ceil(faces.shape[0] / float(max_faces)))
    faces_sub = faces[::stride]
    used = np.unique(faces_sub.reshape(-1))
    remap = -np.ones(int(vertices.shape[0]), dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    verts_sub = vertices[used]
    faces_remap = remap[faces_sub]
    return verts_sub, faces_remap


def build_brain_mesh(
    mri: np.ndarray,
    *,
    downsample: int = 2,
    smooth_sigma: float = 1.0,
    brain_percentile: float = 18.0,
    step_size: int = 2,
    max_faces: int = 120_000,
) -> MeshData:
    """Extract a semi-transparent brain envelope mesh from MRI."""
    mri_ds = downsample_volume(mri, downsample)
    mri_norm = normalize_volume(mri_ds)
    if smooth_sigma > 0:
        mri_norm = smooth_volume(mri_norm, sigma=smooth_sigma)

    mask = brain_envelope_mask(mri_norm, percentile=brain_percentile)
    field = mask.astype(np.float32)

    verts, faces = marching_cubes_mesh(field, level=0.5, step_size=step_size)
    verts, faces = simplify_mesh(verts, faces, max_faces=max_faces)

    # Scale vertices back to original voxel spacing indices.
    f = max(1, int(downsample))
    verts = verts * float(f)

    return MeshData(
        vertices=verts,
        faces=faces,
        name="Brain",
        color_rgba="#b0c4de",  # light steel blue (opacity set at render time)
    )


def build_class_meshes(
    mask: np.ndarray,
    class_ids: Sequence[int] = (1, 2, 3),
    *,
    class_colors: dict[int, str] | None = None,
    class_names: dict[int, str] | None = None,
    downsample: int = 1,
    step_size: int = 1,
    max_faces_per_class: int = 80_000,
    smooth_sigma: float = 0.6,
) -> list[MeshData]:
    """One surface mesh per tumor class."""
    default_colors = {1: "#e41a1c", 2: "#4daf4a", 3: "#ffd700"}
    default_names = {
        1: "NCR/NET (Red)",
        2: "Edema (Green)",
        3: "Enhancing Tumor / ET (Yellow)",
    }
    colors = default_colors if class_colors is None else class_colors
    names = default_names if class_names is None else class_names

    mask_ds = downsample_volume(mask.astype(np.int16), downsample)
    meshes: list[MeshData] = []

    for cid in class_ids:
        cid = int(cid)
        binary = (mask_ds == cid).astype(np.float32)
        if not np.any(binary):
            continue

        if smooth_sigma > 0:
            binary = smooth_volume(binary, sigma=smooth_sigma)

        try:
            verts, faces = marching_cubes_mesh(binary, level=0.5, step_size=step_size)
        except (ValueError, RuntimeError):
            continue

        verts, faces = simplify_mesh(verts, faces, max_faces=max_faces_per_class)
        f = max(1, int(downsample))
        verts = verts * float(f)

        color = colors.get(cid, "#ff7f00")
        meshes.append(
            MeshData(
                vertices=verts,
                faces=faces,
                name=names.get(cid, f"Class {cid}"),
                color_rgba=str(color),
            )
        )

    return meshes


def mesh_to_plotly_trace_arrays(mesh: MeshData) -> dict[str, np.ndarray]:
    """Convert MeshData to kwargs for plotly.graph_objects.Mesh3d."""
    v = mesh.vertices
    f = mesh.faces
    return {
        "x": v[:, 0],
        "y": v[:, 1],
        "z": v[:, 2],
        "i": f[:, 0],
        "j": f[:, 1],
        "k": f[:, 2],
    }


# ---------------------------------------------------------------------------
# Optional mesh export
# ---------------------------------------------------------------------------


def export_obj(mesh: MeshData, out_path: str | Path) -> Path:
    """Write a minimal OBJ file (vertices + triangular faces)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    v = mesh.vertices
    f = mesh.faces
    lines: list[str] = [f"# {mesh.name}", f"o {mesh.name}"]
    for x, y, z in v:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for a, b, c in f + 1:  # OBJ is 1-indexed
        lines.append(f"f {a} {b} {c}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def export_meshes_obj(meshes: Iterable[MeshData], out_dir: str | Path, prefix: str = "mesh") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, mesh in enumerate(meshes):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in mesh.name)
        p = out_dir / f"{prefix}_{i:02d}_{safe}.obj"
        paths.append(export_obj(mesh, p))
    return paths
