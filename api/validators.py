"""Validation utilities for MRI file uploads and NIfTI files."""

import hashlib
import logging
from pathlib import Path

import nibabel as nib
from fastapi import UploadFile

from api.exceptions import ValidationError
from api.schemas import VALID_EXTENSIONS

logger = logging.getLogger(__name__)


class UploadValidator:
    """Validator for MRI file uploads."""

    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MIN_FILE_SIZE = 1024  # 1KB

    @staticmethod
    def validate_upload(upload_file: UploadFile, modality: str) -> None:
        """Validate a single uploaded file.

        Args:
            upload_file: FastAPI UploadFile object
            modality: Modality name (e.g., "t1", "t1ce", "t2", "flair")

        Raises:
            ValidationError: If validation fails
        """
        if upload_file is None:
            raise ValidationError(
                f"{modality.upper()} file is required",
                client_message=f"{modality.upper()} file is required",
            )

        filename = upload_file.filename
        if not filename:
            raise ValidationError(
                f"{modality.upper()} filename is empty",
                client_message=f"{modality.upper()} filename is empty",
            )

        # Validate file extension
        filename_lower = filename.lower()
        has_valid_ext = any(
            filename_lower.endswith(ext) for ext in VALID_EXTENSIONS
        )
        if not has_valid_ext:
            raise ValidationError(
                f"Invalid file extension for {modality.upper()}: {filename}",
                client_message=f"Invalid file extension for {modality.upper()}. Must be .nii or .nii.gz",
            )

        # Validate file size
        if upload_file.size is None:
            # Read content to get size if not provided
            upload_file.file.seek(0, 2)
            size = upload_file.file.tell()
            upload_file.file.seek(0)
        else:
            size = upload_file.size

        if size == 0:
            raise ValidationError(
                f"{modality.upper()} file is empty",
                client_message=f"{modality.upper()} file is empty",
            )

        if size < UploadValidator.MIN_FILE_SIZE:
            raise ValidationError(
                f"{modality.upper()} file is too small: {size} bytes",
                client_message=f"{modality.upper()} file is too small (minimum 1KB)",
            )

        if size > UploadValidator.MAX_FILE_SIZE:
            raise ValidationError(
                f"{modality.upper()} file is too large: {size} bytes",
                client_message=f"{modality.upper()} file exceeds 500MB limit",
            )

    @staticmethod
    def validate_no_duplicate_uploads(modality_files: dict[str, UploadFile]) -> None:
        """Check for duplicate file uploads using SHA-256 hashing.

        Args:
            modality_files: Dictionary mapping modality names to UploadFile objects

        Raises:
            ValidationError: If duplicate files are detected
        """
        hashes = {}
        duplicates = []

        for modality, upload_file in modality_files.items():
            if upload_file is None:
                continue

            try:
                # Compute SHA-256 hash of file content
                upload_file.file.seek(0)
                hasher = hashlib.sha256()
                while chunk := upload_file.file.read(8192):
                    hasher.update(chunk)
                file_hash = hasher.hexdigest()
                upload_file.file.seek(0)
            except Exception as e:
                logger.error(f"Failed to compute hash for {modality}: {e}")
                raise ValidationError(
                    f"Failed to read {modality.upper()} file",
                    client_message=f"Failed to read {modality.upper()} file",
                )

            if file_hash in hashes:
                existing_modality = hashes[file_hash]
                duplicates.append((existing_modality, modality))
            else:
                hashes[file_hash] = modality

        if duplicates:
            duplicate_pairs = ", ".join(
                f"{a.upper()} and {b.upper()}" for a, b in duplicates
            )
            raise ValidationError(
                f"Duplicate files detected: {duplicate_pairs}",
                client_message=f"Duplicate files detected: {duplicate_pairs}. Each modality must be a unique file.",
            )

    @staticmethod
    def validate_nifti_file(file_path: Path) -> None:
        """Validate that a file is a valid NIfTI file.

        Args:
            file_path: Path to the NIfTI file

        Raises:
            ValidationError: If the file is not a valid NIfTI file
        """
        if not file_path.exists():
            raise ValidationError(
                f"File does not exist: {file_path}",
                client_message=f"File not found: {file_path.name}",
            )

        try:
            img = nib.load(str(file_path))
            # Validate that the image has data
            if img.get_fdata().size == 0:
                raise ValidationError(
                    f"NIfTI file is empty: {file_path}",
                    client_message=f"NIfTI file is empty: {file_path.name}",
                )
        except nib.imagefuncs.ImageFileError as e:
            raise ValidationError(
                f"Invalid NIfTI file: {file_path} - {str(e)}",
                client_message=f"Invalid NIfTI file: {file_path.name}",
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to load NIfTI file {file_path}: {str(e)}",
                client_message=f"Failed to load NIfTI file: {file_path.name}",
            )

    @staticmethod
    def validate_modality_shapes(modality_paths: dict[str, Path]) -> None:
        """Validate that all modality files have matching shapes.

        Args:
            modality_paths: Dictionary mapping modality names to file paths

        Raises:
            ValidationError: If shapes do not match
        """
        shapes = {}

        for modality, file_path in modality_paths.items():
            try:
                img = nib.load(str(file_path))
                shape = img.shape
                shapes[modality] = shape
                logger.info(f"{modality.upper()} shape: {shape}")
            except Exception as e:
                raise ValidationError(
                    f"Failed to load {modality.upper()} for shape validation: {str(e)}",
                    client_message=f"Failed to load {modality.upper()} for validation",
                )

        # Check that all shapes are the same
        reference_shape = next(iter(shapes.values()))
        for modality, shape in shapes.items():
            if shape != reference_shape:
                raise ValidationError(
                    f"Shape mismatch: {modality.upper()} has shape {shape}, "
                    f"expected {reference_shape}",
                    client_message=f"Shape mismatch: {modality.upper()} dimensions do not match other modalities",
                )

        logger.info(f"All modalities have matching shape: {reference_shape}")
