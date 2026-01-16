# docker build -t "train_1" -f "./dockerfiles/train.dockerfile" .
# docker run --rm -v ./data:/app/data train_1:latest

FROM ghcr.io/astral-sh/uv:debian-slim AS base

# Install Google Cloud SDK for GCS access
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the project into `/app`
WORKDIR /app

# Fix Python version because of pytorch compatibility
COPY .python-version /app/.python-version
ARG PYTHON_VERSION
ENV UV_PYTHON=${PYTHON_VERSION}


ENV HF_HOME="/app/data/raw"
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Copy training script wrapper
COPY scripts/train_gcp.sh /app/train_gcp.sh
RUN chmod +x /app/train_gcp.sh

ENTRYPOINT ["/app/train_gcp.sh"]
