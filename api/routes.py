import logging
import traceback

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from api.config import DATA_ROOTS, DEFAULT_CHECKPOINT_PATH, OUTPUTS_ROOT, PREDICTIONS_ROOT
from api.exceptions import CheckpointError, InferenceError, ValidationError
from api.jobs import get_job
from api.research_info import get_methods_summary, get_model_info
from api.schemas import PredictRequest, ClassAnalysisRequest, ValidationMetricsRequest
from api.security import UnsafePathError, resolve_within, resolve_within_any
from api.services import predict_case_service, start_prediction_job, validate_case_metrics
from api.utils import calculate_class_analysis, calculate_individual_class_analysis

logger = logging.getLogger(__name__)

router = APIRouter()


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
    # These paths come straight from the request body, so they must be
    # confined to known-safe directories before touching the filesystem or
    # loading a checkpoint with torch.load — never pass raw client input
    # to either.
    try:
        data_dir = resolve_within_any(DATA_ROOTS, request.data_dir)
        output_dir = resolve_within(OUTPUTS_ROOT, request.output_dir)
        checkpoint_path = resolve_within(OUTPUTS_ROOT, request.checkpoint_path)
    except UnsafePathError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    try:
        result = predict_case_service(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
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
        logger.exception("[PREDICT] Prediction failed with exception")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.post("/predict/upload")
def predict_upload(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    seg: UploadFile | None = File(default=None),
    save_probabilities: bool = Query(default=False),
):
    """Start an asynchronous prediction job and return a job_id for polling.

    The checkpoint is a server-side deployment decision (DEFAULT_CHECKPOINT_PATH)
    and is intentionally not a client-supplied parameter.
    """
    try:
        logger.info(
            "[GT] Ground truth upload received: %s",
            seg.filename if seg and seg.filename else None,
        )
        job = start_prediction_job(
            flair=flair,
            t1=t1,
            t1ce=t1ce,
            t2=t2,
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            save_probabilities=save_probabilities,
            seg=seg,
        )

        return job

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.client_message,
        )

    except CheckpointError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.client_message,
        )

    except Exception as e:
        logger.exception("[UPLOAD] Prediction upload failed with exception")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.get("/predict/status/{job_id}")
def predict_status(job_id: str):
    """Return the current status and stage of a prediction job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction job not found",
        )
    return job


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
    # Security check: ensure the requested path stays within the predictions directory
    try:
        requested_path = resolve_within(PREDICTIONS_ROOT, file_path)
    except UnsafePathError:
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


@router.post("/predict/class-analysis")
def class_analysis(request: ClassAnalysisRequest):
    """Calculate volume and dimensions for specific tumor classes.

    Args:
        request: ClassAnalysisRequest with mask_path and classes to analyze

    Returns:
        Dictionary with volume_cm3 and dimensions_mm for the selected classes

    Raises:
        HTTPException 400: If request is invalid
        HTTPException 404: If mask file does not exist
        HTTPException 500: If analysis fails
    """
    try:
        # Validate that mask_path stays within the predictions directory
        try:
            requested_path = resolve_within(PREDICTIONS_ROOT, request.mask_path)
        except UnsafePathError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: mask is outside predictions directory",
            )

        # Validate that file exists
        if not requested_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Mask file not found: {requested_path}",
            )

        if not requested_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path is not a file: {requested_path}",
            )

        # Validate classes
        valid_classes = {1, 2, 3}  # Only tumor classes
        invalid_classes = set(request.classes) - valid_classes
        if invalid_classes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid class IDs: {invalid_classes}. Valid tumor classes: {valid_classes}",
            )
        
        # Calculate class analysis
        analysis = calculate_class_analysis(
            mask_path=requested_path,
            classes=request.classes,
        )
        
        # Generate class names
        class_names_map = {1: "NCR/NET", 2: "Edema", 3: "Enhancing Tumor"}
        class_names = [class_names_map[c] for c in sorted(request.classes)] if request.classes else []
        
        return {
            "status": "success",
            "classes": sorted(list(request.classes)),
            "class_names": class_names,
            "volume_cm3": analysis["volume_cm3"],
            "dimensions_mm": analysis["dimensions_mm"],
        }
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mask file not found",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except HTTPException:
        # Re-raise HTTPExceptions raised above (403/404) as-is instead of
        # letting the catch-all below flatten them into a 500.
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate class analysis",
        )


@router.post("/predict/individual-class-analysis")
def individual_class_analysis(request: ClassAnalysisRequest):
    """Calculate volume and dimensions for each individual tumor class.

    Args:
        request: ClassAnalysisRequest with mask_path (classes parameter is ignored)

    Returns:
        Dictionary with volume_cm3 and dimensions_mm for each class

    Raises:
        HTTPException 400: If request is invalid
        HTTPException 404: If mask file does not exist
        HTTPException 500: If analysis fails
    """
    try:
        # Validate that mask_path stays within the predictions directory
        try:
            requested_path = resolve_within(PREDICTIONS_ROOT, request.mask_path)
        except UnsafePathError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: mask is outside predictions directory",
            )

        # Validate that file exists
        if not requested_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Mask file not found: {requested_path}",
            )

        if not requested_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path is not a file: {requested_path}",
            )

        # Calculate individual class analysis
        class_analysis = calculate_individual_class_analysis(
            mask_path=requested_path,
        )
        
        # Generate class names
        class_names_map = {1: "NCR/NET", 2: "Edema", 3: "Enhancing Tumor"}
        
        return {
            "status": "success",
            "class_analysis": {
                str(class_id): {
                    "name": class_names_map[class_id],
                    "volume_cm3": analysis["volume_cm3"],
                    "dimensions_mm": analysis["dimensions_mm"],
                }
                for class_id, analysis in class_analysis.items()
            },
        }
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mask file not found",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except HTTPException:
        # Re-raise HTTPExceptions raised above (403/404) as-is instead of
        # letting the catch-all below flatten them into a 500.
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate individual class analysis",
        )


@router.post("/predict/validate-case")
def validate_case(request: ValidationMetricsRequest):
    """Compute validation metrics for an existing prediction and ground-truth mask pair."""
    try:
        result = validate_case_metrics(
            prediction_mask_path=request.prediction_mask_path,
            ground_truth_mask_path=request.ground_truth_mask_path,
        )
        return {
            "status": "success",
            "validation": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("[VALIDATION] validate-case endpoint failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Validation computation failed",
        ) from e


@router.get("/research/methods")
def research_methods():
    """Return methods and reproducibility summary from project configuration."""
    return {
        "status": "success",
        "methods": get_methods_summary(),
    }


@router.get("/research/model-info")
def research_model_info(checkpoint_path: str | None = None):
    """Return model metadata; optionally read validation scores from checkpoint.

    `checkpoint_path`, if supplied, must resolve inside OUTPUTS_ROOT — it is
    loaded with `torch.load`, so arbitrary client-supplied paths are rejected
    rather than passed through.
    """
    ckpt = None
    if checkpoint_path:
        try:
            ckpt = resolve_within(OUTPUTS_ROOT, checkpoint_path)
        except UnsafePathError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
    return {
        "status": "success",
        "model": get_model_info(ckpt),
    }
