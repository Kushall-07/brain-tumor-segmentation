from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union

from fastapi import UploadFile


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
