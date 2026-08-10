from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple

import nibabel as nib
import numpy as np

from fastapi import UploadFile

from api.schemas import ModalityPaths


def create_upload_session(base_dir: Union[str, Path] = "uploads") -> Path:
    """Create a unique upload session folder.

    Args:
        base_dir: Base directory for uploads (default: "uploads")

    Returns:
        Path to the newly created upload session folder
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    session_name = f"patient_{timestamp}_{unique_id}"

    session_path = base_path / session_name
    session_path.mkdir(parents=True, exist_ok=True)

    return session_path


def create_prediction_dir(base_dir: Union[str, Path] = "outputs/predictions") -> Path:
    """Create a unique permanent prediction output folder.

    Args:
        base_dir: Base directory for predictions (default: "outputs/predictions")

    Returns:
        Path to the newly created prediction folder
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    prediction_name = f"prediction_{timestamp}_{unique_id}"

    prediction_path = base_path / prediction_name
    prediction_path.mkdir(parents=True, exist_ok=True)

    return prediction_path


def save_uploaded_file(upload_file: UploadFile, destination: Union[str, Path]) -> Path:
    """Save an uploaded file to disk.

    Args:
        upload_file: FastAPI UploadFile object
        destination: Destination path (can be directory or full file path)

    Returns:
        Path to the saved file
    """
    dest_path = Path(destination)

    # If destination is a directory, use the original filename
    if dest_path.is_dir() or (not dest_path.suffix and not dest_path.exists()):
        dest_path = dest_path / upload_file.filename

    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the file content
    with dest_path.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)

    # Reset file pointer for potential re-reading
    upload_file.file.seek(0)

    return dest_path


def cleanup_upload_session(folder: Union[str, Path]) -> None:
    """Delete an upload session folder recursively.

    Args:
        folder: Path to the upload session folder to delete
    """
    folder_path = Path(folder)
    if folder_path.exists():
        shutil.rmtree(folder_path, ignore_errors=True)


def save_modalities(
    t1: UploadFile,
    t1ce: UploadFile,
    t2: UploadFile,
    flair: UploadFile,
    patient_folder: Path,
) -> ModalityPaths:
    """Save all four MRI modalities preserving original file extensions.

    Args:
        t1: Uploaded T1 MRI file
        t1ce: Uploaded T1ce MRI file
        t2: Uploaded T2 MRI file
        flair: Uploaded FLAIR MRI file
        patient_folder: Directory to save the modalities

    Returns:
        ModalityPaths object with paths to saved files
    """
    def _get_extension(filename: str | None) -> str:
        if not filename:
            return ".nii.gz"
        fn_lower = filename.lower()
        if fn_lower.endswith(".nii.gz"):
            return ".nii.gz"
        if fn_lower.endswith(".nii"):
            return ".nii"
        return ".nii.gz"

    t1_ext = _get_extension(t1.filename)
    t1ce_ext = _get_extension(t1ce.filename)
    t2_ext = _get_extension(t2.filename)
    flair_ext = _get_extension(flair.filename)
    
    t1_path = save_uploaded_file(t1, patient_folder / f"t1{t1_ext}")
    t1ce_path = save_uploaded_file(t1ce, patient_folder / f"t1ce{t1ce_ext}")
    t2_path = save_uploaded_file(t2, patient_folder / f"t2{t2_ext}")
    flair_path = save_uploaded_file(flair, patient_folder / f"flair{flair_ext}")

    return ModalityPaths(
        t1=t1_path,
        t1ce=t1ce_path,
        t2=t2_path,
        flair=flair_path,
    )


def calculate_tumor_volume(mask_path: Union[str, Path]) -> dict[str, float | int | list[float]]:
    """Calculate tumor volume from a segmentation mask NIfTI file.

    Args:
        mask_path: Path to the predicted segmentation mask (.nii.gz)

    Returns:
        Dictionary containing:
            - tumor_voxel_count: Number of non-zero voxels
            - voxel_spacing_mm: Voxel spacing [sx, sy, sz] in mm
            - voxel_volume_mm3: Physical volume of one voxel in mm³
            - tumor_volume_mm3: Total tumor volume in mm³
            - tumor_volume_cm3: Total tumor volume in cm³

    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If NIfTI file is invalid or voxel spacing cannot be read
    """
    mask_path_obj = Path(mask_path)
    
    if not mask_path_obj.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path_obj}")
    
    try:
        # Load the NIfTI mask
        mask_img = nib.load(mask_path_obj)
        mask_data = mask_img.get_fdata()
        
        # Get voxel spacing from header
        zooms = mask_img.header.get_zooms()
        if len(zooms) < 3:
            raise ValueError(f"Invalid voxel spacing in NIfTI header: {zooms}")
        
        voxel_spacing_mm = [float(zooms[0]), float(zooms[1]), float(zooms[2])]
        
        # Calculate physical volume of one voxel
        voxel_volume_mm3 = voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]
        
        # Count tumor voxels (all non-zero labels)
        tumor_voxel_count = int(np.count_nonzero(mask_data > 0))
        
        # Calculate tumor volume
        tumor_volume_mm3 = tumor_voxel_count * voxel_volume_mm3
        tumor_volume_cm3 = round(tumor_volume_mm3 / 1000.0, 2)
        
        return {
            "tumor_voxel_count": tumor_voxel_count,
            "voxel_spacing_mm": voxel_spacing_mm,
            "voxel_volume_mm3": round(voxel_volume_mm3, 6),
            "tumor_volume_mm3": round(tumor_volume_mm3, 2),
            "tumor_volume_cm3": tumor_volume_cm3,
        }
        
    except Exception as e:
        raise ValueError(f"Failed to calculate tumor volume from {mask_path_obj}: {str(e)}") from e


def compute_tumor_dimensions_and_geometry_from_data(
    mask_data: np.ndarray, affine: np.ndarray
) -> tuple[dict[str, float] | None, dict[str, dict[str, list[float] | float]] | None]:
    """Calculate 3D tumor dimensions (length, width, height) and physical measurement geometry
    (start_mm, end_mm, midpoint_mm for major axis length) using PCA and NIfTI affine matrix.

    Args:
        mask_data: 3D numpy array of predicted mask
        affine: 4x4 NIfTI affine transformation matrix

    Returns:
        Tuple of (dimensions_dict, geometry_dict) or (None, None) if empty/invalid mask.
    """
    try:
        tumor_mask = mask_data > 0
        tumor_ijk = np.argwhere(tumor_mask)  # Shape (N, 3)

        num_voxels = tumor_ijk.shape[0]
        if num_voxels == 0:
            return None, None

        # 3x3 linear component of NIfTI affine matrix
        M = affine[:3, :3]  # Columns a1, a2, a3 are physical voxel basis vectors
        a1, a2, a3 = M[:, 0], M[:, 1], M[:, 2]

        # Transform voxel indices to physical world mm coordinates
        translation = affine[:3, 3]
        P = tumor_ijk.astype(np.float64) @ M.T + translation  # Shape (N, 3)
        centroid = np.mean(P, axis=0)

        if num_voxels == 1:
            basis_vectors = [a1, a2, a3]
            axis_records = []
            for vec in basis_vectors:
                norm_v = float(np.linalg.norm(vec))
                u = vec / norm_v if norm_v > 0 else np.array([1.0, 0.0, 0.0])
                half_extent = norm_v / 2.0
                s_mm = centroid - u * half_extent
                e_mm = centroid + u * half_extent
                m_mm = (s_mm + e_mm) / 2.0
                axis_records.append({
                    "physical_extent": norm_v,
                    "value_mm": round(norm_v, 2),
                    "start_mm": [round(float(x), 2) for x in s_mm],
                    "end_mm": [round(float(x), 2) for x in e_mm],
                    "midpoint_mm": [round(float(x), 2) for x in m_mm],
                })
        else:
            Q = P - centroid
            cov = np.cov(Q, rowvar=False)

            # Numerically stable eigendecomposition for symmetric covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            sort_idx = np.argsort(eigenvalues)[::-1]
            sorted_vectors = eigenvectors[:, sort_idx]

            axis_records = []
            for col in range(3):
                u = sorted_vectors[:, col]
                norm_u = np.linalg.norm(u)
                if norm_u > 0:
                    u = u / norm_u
                else:
                    u = np.array([1.0 if i == col else 0.0 for i in range(3)])

                # Project centered physical coordinates onto principal axis u
                projections = Q @ u
                p_min = float(np.min(projections))
                p_max = float(np.max(projections))
                center_extent = p_max - p_min

                # Finite physical voxel extent projected onto principal axis u
                voxel_extent = float(
                    abs(np.dot(u, a1)) + abs(np.dot(u, a2)) + abs(np.dot(u, a3))
                )
                half_voxel_extent = voxel_extent / 2.0

                physical_extent = center_extent + voxel_extent

                # Compute physical endpoints in mm (including boundary half-voxel extent)
                start_mm = centroid + u * (p_min - half_voxel_extent)
                end_mm = centroid + u * (p_max + half_voxel_extent)
                midpoint_mm = (start_mm + end_mm) / 2.0

                axis_records.append({
                    "physical_extent": physical_extent,
                    "value_mm": round(float(physical_extent), 2),
                    "start_mm": [round(float(x), 2) for x in start_mm],
                    "end_mm": [round(float(x), 2) for x in end_mm],
                    "midpoint_mm": [round(float(x), 2) for x in midpoint_mm],
                })

        # Sort complete axis records by physical_extent descending
        axis_records.sort(key=lambda r: r["physical_extent"], reverse=True)

        dimensions = {
            "length": axis_records[0]["value_mm"],
            "width": axis_records[1]["value_mm"],
            "height": axis_records[2]["value_mm"],
        }

        geometry = {
            "length": {
                "start_mm": axis_records[0]["start_mm"],
                "end_mm": axis_records[0]["end_mm"],
                "midpoint_mm": axis_records[0]["midpoint_mm"],
                "value_mm": axis_records[0]["value_mm"],
            }
        }

        return dimensions, geometry

    except Exception:
        return None, None


def compute_tumor_dimensions_from_data(mask_data: np.ndarray, affine: np.ndarray) -> dict[str, float] | None:
    """Calculate 3D tumor dimensions directly from mask array and 4x4 affine matrix using PCA."""
    dims, _ = compute_tumor_dimensions_and_geometry_from_data(mask_data, affine)
    return dims


def calculate_tumor_dimensions_and_geometry(
    mask_path: Union[str, Path]
) -> tuple[dict[str, float] | None, dict[str, dict[str, list[float] | float]] | None]:
    """Calculate 3D tumor dimensions and length measurement geometry from a NIfTI mask file."""
    mask_path_obj = Path(mask_path)
    if not mask_path_obj.exists():
        return None, None

    try:
        mask_img = nib.load(mask_path_obj)
        mask_data = mask_img.get_fdata()
        affine = mask_img.affine
        return compute_tumor_dimensions_and_geometry_from_data(mask_data, affine)
    except Exception:
        return None, None


def calculate_tumor_dimensions(mask_path: Union[str, Path]) -> dict[str, float] | None:
    """Calculate automatic 3D tumor dimensions (Length x Width x Height in mm)."""
    dims, _ = calculate_tumor_dimensions_and_geometry(mask_path)
    return dims


