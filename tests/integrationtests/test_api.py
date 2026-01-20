"""API integration tests for trash image classification.

These tests verify the API functionality by simulating real API calls
using FastAPI's TestClient and httpx.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from trashsorting.api import app


@pytest.fixture
def client():
    """Create FastAPI test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


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
    """Create a simple RGB test image."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_png_image():
    """Create a PNG test image."""
    img = Image.new('RGB', (300, 300), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_rgba_image():
    """Create an RGBA test image to test conversion."""
    img = Image.new('RGBA', (200, 200), color=(255, 0, 0, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


class TestRootEndpoint:
    """Tests for the root API endpoint."""

    def test_api_root_returns_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/api/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data

        assert data["message"] == "Trash Image Classification API"
        assert data["version"] == "0.0.1"

    def test_root_endpoint_structure(self, client):
        """Test that root endpoint has all expected fields."""
        response = client.get("/api/")
        data = response.json()

        endpoints = data["endpoints"]
        assert "predict" in endpoints
        assert "health" in endpoints
        assert "model_info" in endpoints
        assert "docs" in endpoints


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_check_structure(self, client):
        """Test health endpoint returns correct structure."""
        response = client.get("/api/health")
        data = response.json()

        # Check all required fields are present
        assert "status" in data
        assert "model_loaded" in data
        assert "model_error" in data

        # Check types
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert data["model_error"] is None or isinstance(data["model_error"], str)

    def test_health_check_status_values(self, client):
        """Test health endpoint returns valid status values."""
        response = client.get("/api/health")
        data = response.json()

        # Status should be either 'healthy' or 'degraded'
        assert data["status"] in ["healthy", "degraded"]


class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""

    @patch('trashsorting.api.model_state')
    @patch('trashsorting.api.Path')
    def test_model_info_success(self, mock_path, mock_state, client, mock_model):
        """Test model info endpoint with loaded model."""
        mock_state.__getitem__.side_effect = lambda key: {
            "model": mock_model,
            "error": None,
            "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
        }[key]

        # Mock Path().stat().st_size to return a fake file size
        mock_path_instance = MagicMock()
        mock_path_instance.stat.return_value.st_size = 10 * 1024 * 1024  # 10 MB
        mock_path.return_value = mock_path_instance

        response = client.get("/api/model/info")
        assert response.status_code == 200

        data = response.json()
        assert data["num_classes"] == 6
        assert data["model_name"] == "mobilenetv3_small_100"
        assert len(data["classes"]) == 6
        assert "cardboard" in data["classes"]
        assert "plastic" in data["classes"]
        assert data["architecture"] == "MobileNetV3-Small (timm)"

    @patch('trashsorting.api.model_state')
    def test_model_info_not_loaded(self, mock_state, client):
        """Test model info endpoint when model not loaded."""
        mock_state.__getitem__.side_effect = lambda key: {
            "model": None,
            "error": "Checkpoint not found",
            "checkpoint_path": None
        }[key]

        response = client.get("/api/model/info")
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


class TestPredictionEndpoint:
    """Tests for the prediction endpoint."""

    @patch('trashsorting.api.predict_from_tensor')
    @patch('trashsorting.api.model_state')
    def test_predict_success(self, mock_state, mock_predict, client, sample_image, mock_model):
        """Test successful prediction with JPEG image."""
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
            "/api/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "plastic"
        assert data["confidence"] == 0.92
        assert len(data["all_probabilities"]) == 6
        assert "predicted_idx" not in data  # Should not be in response

    @patch('trashsorting.api.predict_from_tensor')
    @patch('trashsorting.api.model_state')
    def test_predict_with_png_image(self, mock_state, mock_predict, client, sample_png_image, mock_model):
        """Test prediction with PNG image format."""
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
            "/api/predict",
            files={"file": ("test.png", sample_png_image, "image/png")}
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] == "glass"
        assert data["confidence"] == 0.85

    @patch('trashsorting.api.predict_from_tensor')
    @patch('trashsorting.api.model_state')
    def test_predict_all_classes(self, mock_state, mock_predict, client, sample_image, mock_model):
        """Test that prediction returns all expected trash classes."""
        mock_state.__getitem__.side_effect = lambda key: {
            "model": mock_model,
            "error": None,
            "checkpoint_path": "models/model.ckpt"
        }[key]

        mock_predict.return_value = {
            "predicted_class": "cardboard",
            "confidence": 0.90,
            "all_probabilities": {
                "cardboard": 0.90,
                "glass": 0.02,
                "metal": 0.02,
                "paper": 0.02,
                "plastic": 0.02,
                "trash": 0.02
            },
            "predicted_idx": 0
        }

        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        # Check all expected classes are present
        expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        for cls in expected_classes:
            assert cls in data["all_probabilities"]

    @patch('trashsorting.api.model_state')
    def test_predict_invalid_file_type(self, mock_state, client, mock_model):
        """Test prediction with non-image file."""
        # Mock model as loaded so we can test file type validation
        mock_state.__getitem__.side_effect = lambda key: {
            "model": mock_model,
            "error": None,
            "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
        }[key]

        file_content = b"This is not an image"
        response = client.post(
            "/api/predict",
            files={"file": ("test.txt", file_content, "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    @patch('trashsorting.api.model_state')
    def test_predict_model_not_loaded(self, mock_state, client, sample_image):
        """Test prediction when model not loaded."""
        mock_state.__getitem__.side_effect = lambda key: {
            "model": None,
            "error": "Checkpoint not found",
            "checkpoint_path": None
        }[key]

        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    @patch('trashsorting.api.model_state')
    def test_predict_corrupted_image(self, mock_state, client, mock_model):
        """Test prediction with corrupted image data."""
        # Mock model as loaded so we can test image processing
        mock_state.__getitem__.side_effect = lambda key: {
            "model": mock_model,
            "error": None,
            "checkpoint_path": "models/best-epoch=15-val_loss=0.54.ckpt"
        }[key]

        corrupted = io.BytesIO(b"corrupted image data")

        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", corrupted, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "Failed to process image" in response.json()["detail"]

    @patch('trashsorting.api.predict_from_tensor')
    @patch('trashsorting.api.model_state')
    def test_predict_inference_error(self, mock_state, mock_predict, client, sample_image, mock_model):
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
            "/api/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )

        # Assertions
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]

    @patch('trashsorting.api.predict_from_tensor')
    @patch('trashsorting.api.model_state')
    def test_predict_confidence_range(self, mock_state, mock_predict, client, sample_image, mock_model):
        """Test that confidence is always between 0 and 1."""
        mock_state.__getitem__.side_effect = lambda key: {
            "model": mock_model,
            "error": None,
            "checkpoint_path": "models/model.ckpt"
        }[key]

        mock_predict.return_value = {
            "predicted_class": "metal",
            "confidence": 0.75,
            "all_probabilities": {
                "cardboard": 0.05,
                "glass": 0.05,
                "metal": 0.75,
                "paper": 0.05,
                "plastic": 0.05,
                "trash": 0.05
            },
            "predicted_idx": 2
        }

        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        # Check confidence is in valid range
        assert 0.0 <= data["confidence"] <= 1.0

        # Check all probabilities are in valid range
        for prob in data["all_probabilities"].values():
            assert 0.0 <= prob <= 1.0


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema_accessible(self, client):
        """Test that OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Trash Image Classification API"

    def test_docs_endpoint(self, client):
        """Test that Swagger UI docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

