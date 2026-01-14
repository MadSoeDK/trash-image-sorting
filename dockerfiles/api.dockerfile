# Multi-stage build for optimized production image
# Stage 1: Builder - install dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install production dependencies only (exclude dev group)
# --no-install-project: only install dependencies, not the project itself yet
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and models
COPY src src/
COPY models models/

# Install the project itself
RUN uv sync --frozen --no-dev

# Stage 2: Runtime - minimal production image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Copy only the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy project files and models
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models /app/models
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Expose port
EXPOSE 8000

# Run the API
ENTRYPOINT ["/app/.venv/bin/uvicorn", "src.trashsorting.api:app", "--host", "0.0.0.0", "--port", "8000"]
