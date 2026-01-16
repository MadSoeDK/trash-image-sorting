#!/bin/bash
set -e

echo "Starting GCP training job..."

# Environment variables (set these when running the job)
GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-"trash_classification_data"}
GCS_DATA_PATH=${GCS_DATA_PATH:-"data"}
GCS_MODEL_PATH=${GCS_MODEL_PATH:-"models"}

# Download data from GCS
echo "Downloading training data from gs://${GCS_BUCKET_NAME}/${GCS_DATA_PATH}..."
mkdir -p /app/data
gsutil -m rsync -r gs://${GCS_BUCKET_NAME}/${GCS_DATA_PATH} /app/data

# Run training
echo "Starting model training..."
uv run src/trashsorting/train.py

# Upload trained model to GCS
echo "Uploading trained model to gs://${GCS_BUCKET_NAME}/${GCS_MODEL_PATH}..."
gsutil -m rsync -r /app/models gs://${GCS_BUCKET_NAME}/${GCS_MODEL_PATH}

echo "Training job completed successfully!"
