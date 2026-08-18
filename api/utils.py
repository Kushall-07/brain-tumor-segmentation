from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple

import nibabel as nib
import numpy as np

from fastapi import UploadFile

from api.schemas import ModalityPaths

logger = logging.getLogger(__name__)


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


def save_ground_truth(seg: UploadFile, patient_folder: Path) -> Path:
    """Save optional ground-truth segmentation mask, preserving original file format."""
    filename = seg.filename or "seg.nii.gz"
    fn_lower = filename.lower()
    
    # Preserve the original file extension
    if fn_lower.endswith(".nii.gz"):
        ext = ".nii.gz"
    elif fn_lower.endswith(".nii"):
        ext = ".nii"
    else:
        # Unknown format, default to .nii.gz but warn
        ext = ".nii.gz"
        logger.warning(f"Unknown ground-truth file extension: {filename}, defaulting to .nii.gz")
    
    saved_path = save_uploaded_file(seg, patient_folder / f"seg{ext}")
    
    # Log what was saved
    logger.info(f"[GT] Uploaded filename: {filename}")
    logger.info(f"[GT] Saved path: {saved_path.as_posix()}")
    logger.info(f"[GT] Detected/expected format: {ext}")
    
    return saved_path


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


def calculate_tumor_dimensions(mask_path: Union[str, Path]) -> dict[str, float]:
    """Calculate physical tumor dimensions (height, width, length) from segmentation mask.

    Uses the bounding box of the tumor region and voxel spacing to compute physical dimensions.
    Handles anisotropic voxel spacing correctly.

    Args:
        mask_path: Path to the predicted segmentation mask (.nii.gz)

    Returns:
        Dictionary containing:
            - height_mm: Physical height in mm (typically anterior-posterior)
            - width_mm: Physical width in mm (typically left-right)
            - length_mm: Physical length in mm (typically inferior-superior)

    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If NIfTI file is invalid or no tumor present
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
        
        # Find all non-zero voxels (tumor region)
        tumor_coords = np.argwhere(mask_data > 0)
        
        if tumor_coords.size == 0:
            raise ValueError("No tumor region found in mask (all voxels are background)")
        
        # Calculate bounding box in voxel coordinates
        min_coords = tumor_coords.min(axis=0)
        max_coords = tumor_coords.max(axis=0)
        
        # Calculate bounding box dimensions in voxels
        bbox_voxels = max_coords - min_coords + 1  # +1 to include both endpoints
        
        # Convert to physical dimensions using voxel spacing
        # Assuming NIfTI standard orientation: x=left-right, y=anterior-posterior, z=inferior-superior
        width_mm = round(bbox_voxels[0] * voxel_spacing_mm[0], 2)   # left-right
        height_mm = round(bbox_voxels[1] * voxel_spacing_mm[1], 2) # anterior-posterior
        length_mm = round(bbox_voxels[2] * voxel_spacing_mm[2], 2) # inferior-superior
        
        return {
            "height_mm": height_mm,
            "width_mm": width_mm,
            "length_mm": length_mm,
        }
        
    except Exception as e:
        raise ValueError(f"Failed to calculate tumor dimensions from {mask_path_obj}: {str(e)}") from e


def calculate_tumor_volumes_4class(mask_path: Union[str, Path]) -> dict[str, float]:
    """Calculate volumes for Whole Tumor, Tumor Core, and Enhancing Tumor from 4-class mask.

    Class mapping:
        - 0: Background
        - 1: NCR/NET (Necrotic Core)
        - 2: Edema
        - 3: Enhancing Tumor (ET)

    Regions:
        - Whole Tumor (WT): classes 1 + 2 + 3
        - Tumor Core (TC): classes 1 + 3
        - Enhancing Tumor (ET): class 3 only

    Args:
        mask_path: Path to the predicted segmentation mask (.nii.gz)

    Returns:
        Dictionary containing:
            - whole_tumor_cm3: Whole Tumor volume in cm³
            - tumor_core_cm3: Tumor Core volume in cm³
            - enhancing_tumor_cm3: Enhancing Tumor volume in cm³
            - ncr_net_cm3: NCR/NET volume in cm³
            - edema_cm3: Edema volume in cm³

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
        voxel_volume_mm3 = voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]
        
        # Count voxels for each class
        ncr_net_voxels = int(np.count_nonzero(mask_data == 1))
        edema_voxels = int(np.count_nonzero(mask_data == 2))
        et_voxels = int(np.count_nonzero(mask_data == 3))
        
        # Calculate derived region volumes
        tumor_core_voxels = ncr_net_voxels + et_voxels  # TC = NCR/NET + ET
        whole_tumor_voxels = ncr_net_voxels + edema_voxels + et_voxels  # WT = all tumor classes
        
        # Convert to physical volumes (mm³ then cm³)
        ncr_net_cm3 = round((ncr_net_voxels * voxel_volume_mm3) / 1000.0, 2)
        edema_cm3 = round((edema_voxels * voxel_volume_mm3) / 1000.0, 2)
        et_cm3 = round((et_voxels * voxel_volume_mm3) / 1000.0, 2)
        tumor_core_cm3 = round((tumor_core_voxels * voxel_volume_mm3) / 1000.0, 2)
        whole_tumor_cm3 = round((whole_tumor_voxels * voxel_volume_mm3) / 1000.0, 2)
        
        return {
            "whole_tumor_cm3": whole_tumor_cm3,
            "tumor_core_cm3": tumor_core_cm3,
            "enhancing_tumor_cm3": et_cm3,
            "ncr_net_cm3": ncr_net_cm3,
            "edema_cm3": edema_cm3,
        }
        
    except Exception as e:
        raise ValueError(f"Failed to calculate 4-class tumor volumes from {mask_path_obj}: {str(e)}") from e


def filter_segmentation_mask(
    mask_path: Union[str, Path],
    enabled_classes: set[int],
    output_dir: Union[str, Path] = None,
) -> Path:
    """Filter a segmentation mask to show only specific classes (visualization-only).

    Args:
        mask_path: Path to the original 4-class segmentation mask
        enabled_classes: Set of class IDs to keep (e.g., {1, 2, 3})
        output_dir: Directory to save the filtered mask (defaults to same dir as original)

    Returns:
        Path to the filtered visualization mask

    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If NIfTI file is invalid or enabled_classes is invalid
    """
    mask_path_obj = Path(mask_path)
    
    if not mask_path_obj.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path_obj}")
    
    # Validate enabled classes
    valid_classes = {0, 1, 2, 3}
    invalid_classes = enabled_classes - valid_classes
    if invalid_classes:
        raise ValueError(f"Invalid class IDs: {invalid_classes}. Valid classes: {valid_classes}")
    
    try:
        # Load the original NIfTI mask
        original_img = nib.load(str(mask_path_obj))
        original_data = original_img.get_fdata()
        
        # Create filtered data: keep enabled classes, set others to 0 (background)
        if enabled_classes:
            filtered_data = np.where(np.isin(original_data, list(enabled_classes)), original_data, 0)
        else:
            # If no classes enabled, return all zeros
            filtered_data = np.zeros_like(original_data)
        
        # Preserve original data type
        if original_data.dtype != filtered_data.dtype:
            filtered_data = filtered_data.astype(original_data.dtype)
        
        # Create new NIfTI image with same header and affine
        filtered_img = nib.Nifti1Image(
            filtered_data,
            original_img.affine,
            original_img.header.copy()
        )
        
        # Generate output filename based on enabled classes
        if output_dir is None:
            output_dir = mask_path_obj.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create deterministic filename - FIXED: preserve .nii.gz extension
        if enabled_classes:
            class_suffix = "_".join(sorted(str(c) for c in enabled_classes))
        else:
            class_suffix = "none"
        original_name = mask_path_obj.stem
        # Handle both .nii and .nii.gz extensions
        if mask_path_obj.suffix == '.gz':
            base_name = mask_path_obj.name[:-7]  # Remove .nii.gz
            ext = '.nii.gz'
        else:
            base_name = mask_path_obj.stem
            ext = mask_path_obj.suffix
        filtered_name = f"{base_name}_classes_{class_suffix}{ext}"
        filtered_path = output_dir / filtered_name
        
        # Save the filtered mask
        nib.save(filtered_img, str(filtered_path))
        
        return filtered_path
        
    except Exception as e:
        raise ValueError(f"Failed to filter segmentation mask from {mask_path_obj}: {str(e)}") from e


def generate_class_specific_masks(
    mask_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
) -> dict[int, Path]:
    """Generate separate visualization masks for each tumor class.

    Args:
        mask_path: Path to the original 4-class segmentation mask
        output_dir: Directory to save the class-specific masks (defaults to same dir as original)

    Returns:
        Dictionary mapping class IDs to their visualization mask paths:
        {1: Path to NCR/NET mask, 2: Path to Edema mask, 3: Path to ET mask}

    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If NIfTI file is invalid
    """
    mask_path_obj = Path(mask_path)
    
    if not mask_path_obj.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path_obj}")
    
    class_masks = {}
    
    try:
        # Load the original NIfTI mask
        original_img = nib.load(str(mask_path_obj))
        original_data = original_img.get_fdata()
        
        # Generate mask for each class
        for class_id in [1, 2, 3]:
            # Create binary mask: 1 for this class, 0 for everything else
            class_data = np.where(original_data == class_id, 1, 0).astype(original_data.dtype)
            
            # Create new NIfTI image with same header and affine
            class_img = nib.Nifti1Image(
                class_data,
                original_img.affine,
                original_img.header.copy()
            )
            
            # Generate output filename
            if output_dir is None:
                output_dir = mask_path_obj.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            class_names = {1: 'ncr_net', 2: 'edema', 3: 'et'}
            # Handle .nii.gz extension
            if mask_path_obj.suffix == '.gz':
                base_name = mask_path_obj.name[:-7]
                ext = '.nii.gz'
            else:
                base_name = mask_path_obj.stem
                ext = mask_path_obj.suffix
            
            class_name = f"{base_name}_{class_names[class_id]}{ext}"
            class_path = output_dir / class_name
            
            # Save the class-specific mask
            nib.save(class_img, str(class_path))
            class_masks[class_id] = class_path
        
        return class_masks
        
    except Exception as e:
        raise ValueError(f"Failed to generate class-specific masks from {mask_path_obj}: {str(e)}") from e


def calculate_class_analysis(
    mask_path: Union[str, Path],
    classes: set[int],
) -> dict[str, float | dict[str, float]]:
    """Calculate volume and dimensions for specific tumor classes.

    Args:
        mask_path: Path to the 4-class segmentation mask
        classes: Set of class IDs to analyze (e.g., {1, 2, 3})

    Returns:
        Dictionary with volume_cm3 and dimensions_mm for the selected classes

    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If NIfTI file is invalid
    """
    mask_path_obj = Path(mask_path)
    
    if not mask_path_obj.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path_obj}")
    
    try:
        # Load the NIfTI mask
        img = nib.load(str(mask_path_obj))
        data = img.get_fdata()
        
        # Get voxel spacing
        header = img.header
        sx = float(header['pixdim'][1])
        sy = float(header['pixdim'][2])
        sz = float(header['pixdim'][3])
        
        # Find voxels belonging to the selected classes
        if classes:
            class_mask = np.isin(data, list(classes))
        else:
            class_mask = np.zeros_like(data, dtype=bool)
        
        if not np.any(class_mask):
            # No voxels for selected classes
            return {
                'volume_cm3': 0.0,
                'dimensions_mm': {
                    'height_mm': 0.0,
                    'width_mm': 0.0,
                    'length_mm': 0.0,
                },
            }
        
        # Calculate volume
        voxel_count = np.sum(class_mask)
        voxel_volume_mm3 = sx * sy * sz
        volume_mm3 = voxel_count * voxel_volume_mm3
        volume_cm3 = volume_mm3 / 1000.0
        
        # Calculate bounding box dimensions
        coords = np.argwhere(class_mask)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        
        bbox_voxels = max_coords - min_coords + 1
        width_mm = bbox_voxels[0] * sx
        height_mm = bbox_voxels[1] * sy
        length_mm = bbox_voxels[2] * sz
        
        return {
            'volume_cm3': float(volume_cm3),
            'dimensions_mm': {
                'height_mm': float(height_mm),
                'width_mm': float(width_mm),
                'length_mm': float(length_mm),
            },
        }
        
    except Exception as e:
        raise ValueError(f"Failed to calculate class analysis from {mask_path_obj}: {str(e)}") from e


def calculate_individual_class_analysis(
    mask_path: Union[str, Path],
) -> dict[int, dict[str, float]]:
    """Calculate volume and dimensions for each individual tumor class.

    Args:
        mask_path: Path to the 4-class segmentation mask

    Returns:
        Dictionary mapping class IDs to their individual analysis:
        {
            1: {"volume_cm3": ..., "dimensions_mm": {...}},
            2: {"volume_cm3": ..., "dimensions_mm": {...}},
            3: {"volume_cm3": ..., "dimensions_mm": {...}}
        }

    Raises:
        FileNotFoundError: If mask file does not exist
        ValueError: If NIfTI file is invalid
    """
    mask_path_obj = Path(mask_path)
    
    if not mask_path_obj.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path_obj}")
    
    class_analysis = {}
    
    try:
        # Load the NIfTI mask
        img = nib.load(str(mask_path_obj))
        data = img.get_fdata()
        
        # Get voxel spacing
        header = img.header
        sx = float(header['pixdim'][1])
        sy = float(header['pixdim'][2])
        sz = float(header['pixdim'][3])
        voxel_volume_mm3 = sx * sy * sz
        
        # Calculate analysis for each class
        for class_id in [1, 2, 3]:
            # Find voxels for this class
            class_mask = (data == class_id)
            
            if not np.any(class_mask):
                # No voxels for this class
                class_analysis[class_id] = {
                    'volume_cm3': 0.0,
                    'dimensions_mm': {
                        'height_mm': 0.0,
                        'width_mm': 0.0,
                        'length_mm': 0.0,
                    },
                }
                continue
            
            # Calculate volume
            voxel_count = int(np.sum(class_mask))
            volume_mm3 = voxel_count * voxel_volume_mm3
            volume_cm3 = volume_mm3 / 1000.0
            
            # Calculate bounding box dimensions
            coords = np.argwhere(class_mask)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            bbox_voxels = max_coords - min_coords + 1
            width_mm = bbox_voxels[0] * sx
            height_mm = bbox_voxels[1] * sy
            length_mm = bbox_voxels[2] * sz
            
            class_analysis[class_id] = {
                'volume_cm3': float(volume_cm3),
                'dimensions_mm': {
                    'height_mm': float(height_mm),
                    'width_mm': float(width_mm),
                    'length_mm': float(length_mm),
                },
            }
        
        return class_analysis
        
    except Exception as e:
        raise ValueError(f"Failed to calculate individual class analysis from {mask_path_obj}: {str(e)}") from e
