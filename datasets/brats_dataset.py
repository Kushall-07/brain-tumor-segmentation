
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BraTSCaseFiles:
    """Resolved file paths for a single BraTS case."""

    t1: Path
    t1ce: Path
    t2: Path
    flair: Path
    seg: Path


def _load_nii(path: Path) -> np.ndarray:
    """Load a NIfTI volume and return a numpy array as float32."""
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)

    # Convert to canonical orientation
    vol = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)

    # Ensure shape is [D, H, W]
    vol = np.transpose(vol, (2, 0, 1))

    return np.asarray(vol, dtype=np.float32)


def _zscore_normalize(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize a volume."""
    mean = float(vol.mean())
    std = float(vol.std())
    if std < eps:
        return vol - mean
    return (vol - mean) / (std + eps)


def _remap_brats_labels(mask: np.ndarray) -> np.ndarray:
    """Remap BraTS labels to consecutive integers.

    Mapping:
    - 0 -> 0 (background)
    - 1 -> 1
    - 2 -> 2
    - 4 -> 3
    """

    mask_i = mask.astype(np.int16, copy=False)
    out = np.zeros_like(mask_i, dtype=np.int16)
    out[mask_i == 1] = 1
    out[mask_i == 2] = 2
    out[mask_i == 4] = 3
    return out


def _pad_to_min_shape(vol: np.ndarray, min_shape: tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    d, h, w = vol.shape
    pd = max(0, min_shape[0] - d)
    ph = max(0, min_shape[1] - h)
    pw = max(0, min_shape[2] - w)

    if pd == 0 and ph == 0 and pw == 0:
        return vol

    pad_width = (
        (pd // 2, pd - (pd // 2)),
        (ph // 2, ph - (ph // 2)),
        (pw // 2, pw - (pw // 2)),
    )
    return np.pad(vol, pad_width=pad_width, mode="constant", constant_values=pad_value)


def _compute_patch_slices(
    vol_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    center_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    slices: list[slice] = []
    for dim, p, c in zip(vol_shape, patch_size, center_zyx):
        start = int(c) - (p // 2)
        start = max(0, min(start, dim - p))
        end = start + p
        slices.append(slice(start, end))
    return slices[0], slices[1], slices[2]


def _choose_patch_center(
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
    foreground_prob: float,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    d, h, w = mask.shape

    use_fg = rng.random() < foreground_prob
    if use_fg:
        fg = np.argwhere(mask > 0)
        if fg.size > 0:
            zyx = fg[rng.integers(0, fg.shape[0])]
            return int(zyx[0]), int(zyx[1]), int(zyx[2])

    z = int(rng.integers(0, max(1, d)))
    y = int(rng.integers(0, max(1, h)))
    x = int(rng.integers(0, max(1, w)))
    return z, y, x


def _find_modality_file(case_dir: Path, modality: str) -> Path:
    """Find the NIfTI file for a modality inside a case folder.

    Supports typical BraTS naming like:
    - *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz
    - *_seg.nii.gz (or *_label.nii.gz)
    - BraTS-GLI-xxxxx-t1n.nii.gz, BraTS-GLI-xxxxx-t1c.nii.gz, etc.
    """
    import re

    # Mapping from simplified modality names to their BraTS equivalents
    modality_map = {
        "t1": ["t1", "t1n"],
        "t1ce": ["t1ce", "t1c"],
        "t2": ["t2", "t2w"],
        "flair": ["flair", "t2f"],
        "seg": ["seg", "label"],
    }

    # Get the possible names for this modality
    possible_names = modality_map.get(modality, [modality])

    # Build regex pattern that matches the modality as a whole word
    # This prevents "t1" from matching "t1ce"
    patterns = []
    for name in possible_names:
        # Match the modality followed by a non-alphanumeric character or end of string
        # Also match if preceded by a non-alphanumeric character or start of string
        pattern = re.compile(rf"(?:^|[^a-zA-Z0-9]){re.escape(name)}(?:[^a-zA-Z0-9]|$)")
        patterns.append(pattern)

    # Find all NIfTI files in the directory
    all_files = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))

    # Filter files that match any of the patterns
    matches = []
    for file_path in all_files:
        filename = file_path.name
        # Remove extensions to get the base filename
        if filename.endswith(".nii.gz"):
            filename = filename[:-7]
        elif filename.endswith(".nii"):
            filename = filename[:-4]
        
        for pattern in patterns:
            if pattern.search(filename):
                matches.append(file_path)
                break

    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find '{modality}' NIfTI file in: {case_dir}")
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple candidate '{modality}' files found in {case_dir}: {[m.name for m in matches]}"
        )
    return matches[0]


class BraTSDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, str]]):
    """PyTorch Dataset for BraTS-style 3D MRI tumor segmentation."""

    def __init__(
        self,
        root_dir: str | Path,
        case_dirs: Optional[Sequence[str | Path]] = None,
        modalities: Sequence[str] = ("t1n", "t1c", "t2w", "t2f"),
        seg_name: str = "seg",
        transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        normalize: bool = True,
        patch_size: Optional[tuple[int, int, int]] = None,
        foreground_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """Create the dataset.

        Args:
            root_dir: Root directory containing one subfolder per patient/case.
            case_dirs: Optional explicit list of case directories (relative to root_dir or absolute).
            modalities: Modality identifiers used to locate files in each case dir.
            seg_name: Identifier used to locate the segmentation file (default: 'seg').
            transforms: Optional callable applied as transforms(image, mask) -> (image, mask).
            normalize: If True, apply per-modality z-score normalization.
        """

        self.root_dir = Path(root_dir)
        self.modalities = tuple(modalities)
        if len(self.modalities) != 4:
            raise ValueError(f"Expected 4 modalities, got {len(self.modalities)}: {self.modalities}")

        self.seg_name = seg_name
        self.transforms = transforms
        self.normalize = normalize
        self.patch_size = patch_size
        self.foreground_prob = float(foreground_prob)
        self.rng = np.random.default_rng(seed)

        if case_dirs is None:
            self.case_dirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        else:
            resolved: list[Path] = []
            for p in case_dirs:
                pp = Path(p)
                resolved.append(pp if pp.is_absolute() else (self.root_dir / pp))
            self.case_dirs = resolved

        if len(self.case_dirs) == 0:
            raise ValueError(f"No case directories found under: {self.root_dir}")

        self.cases: list[BraTSCaseFiles] = [self._resolve_case_files(d) for d in self.case_dirs]

    def _resolve_case_files(self, case_dir: Path) -> BraTSCaseFiles:
        if len(self.modalities) != 4:
            raise ValueError(f"Expected 4 modalities, got {len(self.modalities)}: {self.modalities}")

        t1 = _find_modality_file(case_dir, self.modalities[0])
        t1ce = _find_modality_file(case_dir, self.modalities[1])
        t2 = _find_modality_file(case_dir, self.modalities[2])
        flair = _find_modality_file(case_dir, self.modalities[3])
        seg = _find_modality_file(case_dir, self.seg_name)
        return BraTSCaseFiles(t1=t1, t1ce=t1ce, t2=t2, flair=flair, seg=seg)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        case = self.cases[idx]
        case_id = self.case_dirs[idx].name

        vols = [
            _load_nii(case.t1),
            _load_nii(case.t1ce),
            _load_nii(case.t2),
            _load_nii(case.flair),
        ]

        shapes = {v.shape for v in vols}
        if len(shapes) != 1:
            raise ValueError(f"Modality volumes have inconsistent shapes for case index {idx}: {shapes}")

        if self.normalize:
            vols = [_zscore_normalize(v) for v in vols]

        mask_np = _load_nii(case.seg)
        mask_np = _remap_brats_labels(mask_np)
        mask_np = mask_np.astype(np.int64, copy=False)

        if mask_np.shape != vols[0].shape:
            raise ValueError(f"Mask shape {mask_np.shape} does not match image shape {vols[0].shape}")

        if self.patch_size is not None:
            ps = self.patch_size

            vols = [_pad_to_min_shape(v, ps, pad_value=0.0) for v in vols]
            mask_np = _pad_to_min_shape(mask_np, ps, pad_value=0)

            center = _choose_patch_center(mask_np, ps, self.foreground_prob, self.rng)
            sz, sy, sx = _compute_patch_slices(mask_np.shape, ps, center)

            vols = [v[sz, sy, sx] for v in vols]
            mask_np = mask_np[sz, sy, sx]

        # Stack into [4, D, H, W]
        image_np = np.stack(vols, axis=0).astype(np.float32, copy=False)

        if image_np.shape[0] != 4:
            raise ValueError("Expected 4 MRI modalities")

        image = torch.from_numpy(image_np)
        mask = torch.from_numpy(mask_np)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask, case_id

