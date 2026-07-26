from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from api.exceptions import CheckpointError, InferenceError, ValidationError
from api.schemas import PredictRequest
from api.services import predict_case_service, predict_case_upload_service

router = APIRouter()

# Predictions directory for secure file serving
PREDICTIONS_DIR = Path("outputs/predictions")


@router.get("/")
def root():
    return {
        "message": "Brain Tumor Segmentation API is running"
    }


@router.get("/health")
def health():
    return {
        "status": "healthy"
    }


@router.post("/predict")
def predict(request: PredictRequest):
    try:
        result = predict_case_service(
            data_dir=request.data_dir,
            checkpoint_path=request.checkpoint_path,
            output_dir=request.output_dir,
            case_index=request.case_index,
            save_probabilities=request.save_probabilities,
        )

        return {
            "status": "success",
            "result": result,
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.client_message,
        )

    except CheckpointError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.client_message,
        )

    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.client_message,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/predict/upload")
def predict_upload(
    flair: UploadFile,
    t1: UploadFile,
    t1ce: UploadFile,
    t2: UploadFile,
    checkpoint_path: str,
    save_probabilities: bool = False,
):
    try:
        result = predict_case_upload_service(
            flair=flair,
            t1=t1,
            t1ce=t1ce,
            t2=t2,
            checkpoint_path=checkpoint_path,
            save_probabilities=save_probabilities,
        )

        return {
            "status": "success",
            "result": result,
        }

    except ValidationError as e:
        # Validation errors (missing modality, invalid extension, empty file, shape mismatch)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.client_message,
        )

    except CheckpointError as e:
        # Checkpoint not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.client_message,
        )

    except InferenceError as e:
        # Inference failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.client_message,
        )

    except Exception as e:
        # Unexpected internal error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/download/{file_path:path}")
def download_prediction(file_path: str):
    """Download a prediction file securely.
    
    Args:
        file_path: Relative path within outputs/predictions/ directory
        
    Returns:
        FileResponse with the requested file
        
    Raises:
        HTTPException 403: If path is outside predictions directory
        HTTPException 404: If file does not exist
    """
    # Resolve the requested path
    requested_path = (PREDICTIONS_DIR / file_path).resolve()
    
    # Resolve the predictions directory to absolute path
    predictions_dir = PREDICTIONS_DIR.resolve()
    
    # Security check: ensure the requested path is within predictions directory
    try:
        requested_path.relative_to(predictions_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: file is outside predictions directory",
        )
    
    # Check if file exists
    if not requested_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )
    
    # Check if it's a file (not a directory)
    if not requested_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not a file",
        )
    
    # Return the file
    return FileResponse(
        path=requested_path,
        filename=requested_path.name,
        media_type="application/octet-stream",
    )