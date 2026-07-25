from __future__ import annotations

from pathlib import Path

from inference.predict import predict_case


def predict_case_service(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    case_index: int = 0,
    save_probabilities: bool = False,
) -> dict[str, str | None]:
    """Service layer for brain tumor segmentation inference.

    Args:
        data_dir: Root directory containing case subfolders
        checkpoint_path: Path to trained checkpoint (.pt)
        output_dir: Output directory for results
        case_index: Index of case in data_dir to run inference on
        save_probabilities: If True, save class probabilities as NIfTI

    Returns:
        Dictionary with keys: case_id, mask_path, probability_path

    Raises:
        FileNotFoundError: If data_dir or checkpoint_path does not exist
        RuntimeError: If inference fails for any reason
    """
    # Validate input paths
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir_path}")

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path_obj}")

    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Call the inference function
    try:
        result = predict_case(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            out_dir=output_dir,
            case_index=case_index,
            save_probs=save_probabilities,
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}") from e
