"""Performance/load tests for the trash sorting API using Locust.

These tests simulate multiple users interacting with the API to measure
performance characteristics like response time, throughput, and stability
under load.

To run these tests locally:
    locust -f tests/performancetests/locustfile.py

To run headless:
    locust -f tests/performancetests/locustfile.py \
        --headless --users 10 --spawn-rate 1 --run-time 1m --host http://localhost:8000

Environment variables:
    DEPLOYED_MODEL_URL or MYENDPOINT: The URL of the API to test
"""

import io
import os
from pathlib import Path
from random import choice

from locust import HttpUser, task, between, events
from PIL import Image


def create_test_image(color='red', size=(224, 224), format='JPEG'):
    """Create a test image in memory.

    Args:
        color: Color of the image (e.g., 'red', 'blue', 'green')
        size: Tuple of (width, height)
        format: Image format (e.g., 'JPEG', 'PNG')

    Returns:
        BytesIO object containing the image data
    """
    img = Image.new('RGB', size, color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes


class TrashSortingUser(HttpUser):
    """Simulates a user of the trash sorting API.

    This user performs various tasks on the API with different frequencies
    to simulate realistic usage patterns.
    """

    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts. Can be used for setup."""
        # Pre-generate some test images for reuse
        self.test_images = {
            'jpeg_small': create_test_image('red', (224, 224), 'JPEG'),
            'jpeg_medium': create_test_image('blue', (500, 500), 'JPEG'),
            'jpeg_large': create_test_image('green', (1000, 1000), 'JPEG'),
            'png': create_test_image('yellow', (300, 300), 'PNG'),
        }

    @task(5)
    def predict_image(self):
        """Main task: Predict trash category for an image.

        This task has weight 5, making it the most common operation.
        """
        # Randomly select an image type to test different scenarios
        image_type = choice(list(self.test_images.keys()))
        img_bytes = self.test_images[image_type]
        img_bytes.seek(0)  # Reset to beginning

        # Determine content type based on image type
        content_type = "image/jpeg" if 'jpeg' in image_type else "image/png"
        filename = f"test.{'jpg' if 'jpeg' in image_type else 'png'}"

        with self.client.post(
            "/api/predict",
            files={"file": (filename, img_bytes, content_type)},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate response structure
                    if "predicted_class" in data and "confidence" in data:
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                except Exception as e:
                    response.failure(f"Failed to parse response: {e}")
            elif response.status_code == 503:
                # Model not loaded - expected in some scenarios
                response.failure("Model not loaded (503)")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(2)
    def check_health(self):
        """Check API health status.

        This task has weight 2, making it moderately common.
        """
        with self.client.get("/api/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and "model_loaded" in data:
                        response.success()
                    else:
                        response.failure("Invalid health response structure")
                except Exception as e:
                    response.failure(f"Failed to parse health response: {e}")
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def get_model_info(self):
        """Get model information.

        This task has weight 1, making it the least common.
        """
        with self.client.get("/api/model/info", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "model_name" in data and "num_classes" in data:
                        response.success()
                    else:
                        response.failure("Invalid model info response")
                except Exception as e:
                    response.failure(f"Failed to parse model info: {e}")
            elif response.status_code == 503:
                # Model not loaded - this is expected in some scenarios
                response.failure("Model not loaded (503)")
            else:
                response.failure(f"Model info failed: {response.status_code}")

    @task(1)
    def get_api_root(self):
        """Access the API root endpoint.

        This task has weight 1.
        """
        with self.client.get("/api/", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "message" in data and "endpoints" in data:
                        response.success()
                    else:
                        response.failure("Invalid root response structure")
                except Exception as e:
                    response.failure(f"Failed to parse root response: {e}")
            else:
                response.failure(f"Root endpoint failed: {response.status_code}")


class HighLoadUser(HttpUser):
    """Simulates a high-load scenario with rapid predictions.

    This user only performs predictions with minimal wait time,
    useful for stress testing.
    """

    # Very short wait time to create high load
    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Prepare test images."""
        self.test_image = create_test_image('red', (224, 224), 'JPEG')

    @task
    def rapid_predictions(self):
        """Rapidly send prediction requests."""
        self.test_image.seek(0)
        self.client.post(
            "/api/predict",
            files={"file": ("test.jpg", self.test_image, "image/jpeg")}
        )


# Event handlers for custom reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 60)
    print("Starting load test for Trash Sorting API")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 60)
    print("Load test completed")
    print("=" * 60)

    # Print summary statistics
    stats = environment.stats
    print("\nSummary Statistics:")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    if stats.total.num_requests > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"Failure rate: {failure_rate:.2f}%")
    print(f"Average response time: {stats.total.avg_response_time:.2f} ms")
    print(f"Median response time: {stats.total.median_response_time:.2f} ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f} ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f} ms")
    if stats.total.total_rps:
        print(f"Requests per second: {stats.total.total_rps:.2f}")

