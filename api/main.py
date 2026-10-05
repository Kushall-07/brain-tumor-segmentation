from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import CORS_ALLOWED_ORIGINS
from api.routes import router

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="API for Deep Learning-Based Brain Tumor Segmentation using Multi-Modal MRI",
    version="1.0.0",
)

# CORS configuration — set CORS_ALLOWED_ORIGINS (comma-separated) in production;
# defaults to local Vite dev ports when unset.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)