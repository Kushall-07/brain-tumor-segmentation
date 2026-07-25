from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="API for Deep Learning-Based Brain Tumor Segmentation using Multi-Modal MRI",
    version="1.0.0",
)

app.include_router(router)