from fastapi import APIRouter, HTTPException, UploadFile

from api.schemas import PredictRequest
from api.services import predict_case_service, predict_case_upload_service

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

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


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

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")