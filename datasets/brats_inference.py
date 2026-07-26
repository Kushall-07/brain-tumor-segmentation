from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.brats_dataset import _find_modality_file, _load_nii, _zscore_normalize


@dataclass(frozen=True)
class BraTSInferenceCaseFiles:
    """Resolved file paths for a single BraTS inference case (no segmentation)."""

    t1: Path
    t1ce: Path
    t2: Path
    flair: Path


class BraTSInferenceDataset(Dataset[Tuple[torch.Tensor, str]]):
    """PyTorch Dataset for BraTS-style 3D MRI tumor segmentation inference.

    This dataset is designed for inference only and does not require segmentation masks.
    """

    def __init__(
        self,
        root_dir: str | Path,
        case_dirs: Optional[Sequence[str | Path]] = None,
        modalities: Sequence[str] = ("t1", "t1ce", "t2", "flair"),
        normalize: bool = True,
    ) -> None:
        """Create the inference dataset.

        Args:
            root_dir: Root directory containing one subfolder per patient/case.
            case_dirs: Optional explicit list of case directories (relative to root_dir or absolute).
            modalities: Modality identifiers used to locate files in each case dir.
            normalize: If True, apply per-modality z-score normalization.
        """
        self.root_dir = Path(root_dir)
        self.modalities = tuple(modalities)
        if len(self.modalities) != 4:
            raise ValueError(f"Expected 4 modalities, got {len(self.modalities)}: {self.modalities}")

        self.normalize = normalize

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

        self.cases: list[BraTSInferenceCaseFiles] = [
            self._resolve_case_files(d) for d in self.case_dirs
        ]

    def _resolve_case_files(self, case_dir: Path) -> BraTSInferenceCaseFiles:
        """Resolve the file paths for the four MRI modalities in a case directory.

        Args:
            case_dir: Path to the case directory

        Returns:
            BraTSInferenceCaseFiles with paths to t1, t1ce, t2, flair files
        """
        if len(self.modalities) != 4:
            raise ValueError(f"Expected 4 modalities, got {len(self.modalities)}: {self.modalities}")

        t1 = _find_modality_file(case_dir, self.modalities[0])
        t1ce = _find_modality_file(case_dir, self.modalities[1])
        t2 = _find_modality_file(case_dir, self.modalities[2])
        flair = _find_modality_file(case_dir, self.modalities[3])

        return BraTSInferenceCaseFiles(t1=t1, t1ce=t1ce, t2=t2, flair=flair)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Load and return the MRI image tensor and case ID for the given index.

        Args:
            idx: Index of the case to load

        Returns:
            Tuple of (image_tensor, case_id) where image_tensor has shape [4, D, H, W]
        """
        case = self.cases[idx]
        case_id = self.case_dirs[idx].name

        # Load the four MRI modalities
        vols = [
            _load_nii(case.t1),
            _load_nii(case.t1ce),
            _load_nii(case.t2),
            _load_nii(case.flair),
        ]

        # Validate that all modalities have the same shape
        shapes = {v.shape for v in vols}
        if len(shapes) != 1:
            raise ValueError(
                f"Modality volumes have inconsistent shapes for case index {idx}: {shapes}"
            )

        # Apply z-score normalization if enabled
        if self.normalize:
            vols = [_zscore_normalize(v) for v in vols]

        # Stack into [4, D, H, W]
        image_np = np.stack(vols, axis=0).astype(np.float32, copy=False)

        if image_np.shape[0] != 4:
            raise ValueError("Expected 4 MRI modalities")

        # Convert to torch tensor
        image = torch.from_numpy(image_np)

        return image, case_id
