#! /usr/bin/env bash

# Install Google Cloud SDK via apt-get
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update
sudo apt-get install -y google-cloud-cli
gcloud auth configure-docker gcr.io
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install --install-hooks
