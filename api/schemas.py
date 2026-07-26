from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel


# Modality constants
MODALITY_T1 = "t1"
MODALITY_T1CE = "t1ce"
MODALITY_T2 = "t2"
MODALITY_FLAIR = "flair"

# Valid file extensions
VALID_EXTENSIONS = {".nii", ".nii.gz"}


@dataclass(frozen=True)
class ModalityPaths:
    """Explicit paths for MRI modalities - no filename discovery."""
    t1: Path
    t1ce: Path
    t2: Path
    flair: Path


class PredictRequest(BaseModel):
    data_dir: str
    checkpoint_path: str
    output_dir: str
    case_index: int = 0
    save_probabilities: bool = False


class PredictResponse(BaseModel):
    status: str
    result: dict


class PredictUploadResponse(BaseModel):
    status: str
    result: dict
