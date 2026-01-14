"""FastAPI application for trash image classification inference."""

from contextlib import asynccontextmanager
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import Tensor

from trashsorting.model import TrashModel
from trashsorting.predict import predict_from_tensor, CLASS_NAMES
from trashsorting.utils.transform import image_transform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model state
model_state: Dict[str, Any] = {
    "model": None,
    "error": None,
    "checkpoint_path": None,
}


# Pydantic response models
class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": "plastic",
                "confidence": 0.92,
                "all_probabilities": {
                    "cardboard": 0.02,
                    "glass": 0.01,
                    "metal": 0.03,
                    "paper": 0.01,
                    "plastic": 0.92,
                    "trash": 0.01
                }
            }
        }
    )

    predicted_class: str = Field(..., description="The predicted trash category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all 6 classes")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    model_error: Optional[str] = Field(None, description="Model loading error if any")


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    model_name: str
    num_classes: int
    classes: list[str]
    checkpoint_path: str
    checkpoint_size_mb: float
    architecture: str


# Lifespan context manager for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup and cleanup on shutdown."""
    # Startup: Load model
    checkpoint_path = Path("models/model.ckpt")

    logger.info(f"Loading model from {checkpoint_path}")
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        model = TrashModel.load_from_checkpoint(
            str(checkpoint_path),
            pretrained=False  # Don't download pretrained weights, use checkpoint
        )
        model.eval()

        # Ensure model is in evaluation mode with no gradient computation
        torch.set_grad_enabled(False)

        model_state["model"] = model
        model_state["checkpoint_path"] = str(checkpoint_path)
        logger.info("Model loaded successfully")

    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        model_state["error"] = error_msg

    yield

    # Shutdown: cleanup
    logger.info("Shutting down API")
    model_state["model"] = None


# Initialize FastAPI app
app = FastAPI(
    title="Trash Image Classification API",
    description="API for classifying trash images into recycling categories using MobileNetV3",
    version="0.0.1",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Welcome endpoint with API documentation links."""
    return {
        "message": "Trash Image Classification API",
        "version": "0.0.1",
        "description": "Classify trash images into 6 recycling categories using MobileNetV3",
        "endpoints": {
            "predict": "POST /predict - Upload image for classification",
            "health": "GET /health - Check API health",
            "model_info": "GET /model/info - Get model information",
            "docs": "GET /docs - Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API and model health status.

    Returns information about model loading state and any errors.
    Useful for Docker health checks and monitoring.
    """
    return HealthResponse(
        status="healthy" if model_state["model"] is not None else "degraded",
        model_loaded=model_state["model"] is not None,
        model_error=model_state["error"]
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.

    Returns model architecture, classes, and checkpoint details.
    """
    if model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {model_state['error']}"
        )

    checkpoint_path = Path(model_state["checkpoint_path"])
    checkpoint_size = checkpoint_path.stat().st_size / (1024 * 1024)  # Convert to MB

    return ModelInfoResponse(
        model_name=model_state["model"].model_name,
        num_classes=model_state["model"].num_classes,
        classes=CLASS_NAMES,
        checkpoint_path=str(checkpoint_path),
        checkpoint_size_mb=round(checkpoint_size, 2),
        architecture="MobileNetV3-Small (timm)"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Classify a trash image into recycling categories.

    Accepts: JPEG, PNG, etc. (any PIL-compatible format)
    Returns: Predicted class and probabilities for all categories

    Args:
        file: Image file upload (multipart/form-data)

    Returns:
        PredictionResponse with predicted class, confidence, and all probabilities

    Raises:
        HTTPException:
            - 503 if model not loaded
            - 400 if invalid file type or corrupted image
            - 500 if prediction fails
    """
    # 1. Check model loaded
    if model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {model_state['error']}"
        )

    # 2. Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/* (JPEG, PNG, etc.)"
        )

    # 3. Read and process image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )

    # 4. Transform and predict
    try:
        transform = image_transform()
        image_tensor = cast(Tensor, transform(image))

        logger.info(f"Processing image: {file.filename}, size: {image.size}, tensor shape: {image_tensor.shape}")

        # Use predict_from_tensor with loaded model
        result = predict_from_tensor(
            image_tensor=image_tensor,
            model=model_state["model"],
            class_names=CLASS_NAMES
        )

        logger.info(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")

        # Return only the fields needed for PredictionResponse
        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"]
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
