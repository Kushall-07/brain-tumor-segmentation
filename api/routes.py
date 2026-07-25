from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services import predict_case_service

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


class PredictRequest(BaseModel):
    data_dir: str
    checkpoint_path: str
    output_dir: str
    case_index: int = 0
    save_probabilities: bool = False


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