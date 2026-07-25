from fastapi import APIRouter

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