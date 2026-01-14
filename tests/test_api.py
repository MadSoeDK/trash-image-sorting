"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from unittest.mock import patch, MagicMock

from trashsorting.api import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock TrashModel for testing."""
    model = MagicMock()
    model.eval.return_value = None
    model.model_name = "mobilenetv3_small_100"
    model.num_classes = 6
    return model


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_root_endpoint(client):
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    assert "version" in data
    assert data["message"] == "Trash Image Classification API"


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


@patch('trashsorting.api.predict_from_tensor')
@patch('trashsorting.api.model_state')
def test_predict_success(mock_state, mock_predict, client, sample_image, mock_model):
    """Test successful prediction."""
    # Setup mocks
    mock_state.__getitem__.side_effect = lambda key: {
        "model": mock_model,
        "error": None,
        "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
    }[key]

    mock_predict.return_value = {
        "predicted_class": "plastic",
        "confidence": 0.92,
        "all_probabilities": {
            "cardboard": 0.02,
            "glass": 0.01,
            "metal": 0.03,
            "paper": 0.01,
            "plastic": 0.92,
            "trash": 0.01
        },
        "predicted_idx": 4
    }

    # Make request
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "plastic"
    assert data["confidence"] == 0.92
    assert len(data["all_probabilities"]) == 6
    assert "predicted_idx" not in data  # Should not be in response


@patch('trashsorting.api.model_state')
def test_predict_invalid_file_type(mock_state, client, mock_model):
    """Test prediction with non-image file."""
    # Mock model as loaded so we can test file type validation
    mock_state.__getitem__.side_effect = lambda key: {
        "model": mock_model,
        "error": None,
        "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
    }[key]

    file_content = b"This is not an image"
    response = client.post(
        "/predict",
        files={"file": ("test.txt", file_content, "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


@patch('trashsorting.api.model_state')
def test_predict_model_not_loaded(mock_state, client, sample_image):
    """Test prediction when model not loaded."""
    mock_state.__getitem__.side_effect = lambda key: {
        "model": None,
        "error": "Checkpoint not found",
        "checkpoint_path": None
    }[key]

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


@patch('trashsorting.api.model_state')
def test_predict_corrupted_image(mock_state, client, mock_model):
    """Test prediction with corrupted image."""
    # Mock model as loaded so we can test image processing
    mock_state.__getitem__.side_effect = lambda key: {
        "model": mock_model,
        "error": None,
        "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
    }[key]

    corrupted = io.BytesIO(b"corrupted image data")

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", corrupted, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Failed to process image" in response.json()["detail"]


@patch('trashsorting.api.model_state')
def test_model_info_success(mock_state, client, mock_model):
    """Test model info endpoint with loaded model."""
    mock_state.__getitem__.side_effect = lambda key: {
        "model": mock_model,
        "error": None,
        "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
    }[key]

    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 6
    assert data["model_name"] == "mobilenetv3_small_100"
    assert len(data["classes"]) == 6
    assert "cardboard" in data["classes"]
    assert data["architecture"] == "MobileNetV3-Small (timm)"


@patch('trashsorting.api.model_state')
def test_model_info_not_loaded(mock_state, client):
    """Test model info endpoint when model not loaded."""
    mock_state.__getitem__.side_effect = lambda key: {
        "model": None,
        "error": "Checkpoint not found",
        "checkpoint_path": None
    }[key]

    response = client.get("/model/info")
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


@patch('trashsorting.api.predict_from_tensor')
@patch('trashsorting.api.model_state')
def test_predict_with_png_image(mock_state, mock_predict, client, mock_model):
    """Test prediction with PNG image format."""
    # Create PNG image
    img = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Setup mocks
    mock_state.__getitem__.side_effect = lambda key: {
        "model": mock_model,
        "error": None,
        "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
    }[key]

    mock_predict.return_value = {
        "predicted_class": "glass",
        "confidence": 0.85,
        "all_probabilities": {
            "cardboard": 0.02,
            "glass": 0.85,
            "metal": 0.05,
            "paper": 0.01,
            "plastic": 0.05,
            "trash": 0.02
        },
        "predicted_idx": 1
    }

    # Make request
    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "glass"
    assert data["confidence"] == 0.85


@patch('trashsorting.api.predict_from_tensor')
@patch('trashsorting.api.model_state')
def test_predict_inference_error(mock_state, mock_predict, client, sample_image, mock_model):
    """Test prediction when inference fails."""
    # Setup mocks
    mock_state.__getitem__.side_effect = lambda key: {
        "model": mock_model,
        "error": None,
        "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
    }[key]

    # Make predict_from_tensor raise an exception
    mock_predict.side_effect = RuntimeError("Inference error")

    # Make request
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")}
    )

    # Assertions
    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]


def test_health_endpoint_structure(client):
    """Test health endpoint returns correct structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Check all required fields are present
    assert "status" in data
    assert "model_loaded" in data
    assert "model_error" in data

    # Check types
    assert isinstance(data["status"], str)
    assert isinstance(data["model_loaded"], bool)
    assert data["model_error"] is None or isinstance(data["model_error"], str)
