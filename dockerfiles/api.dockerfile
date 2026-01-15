# Multi-stage build for optimized production image
# Stage 1: Builder - install dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install production dependencies only (exclude dev group)
# --no-install-project: only install dependencies, not the project itself yet
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and models
COPY src src/
COPY models models/

# Install project
RUN uv sync --frozen --no-dev

# Stage 2: Runtime - minimal production image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Copy only the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy project files and models
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models /app/models
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

EXPOSE 8080

# Run the API - use PORT env var if set, otherwise default to 8080
CMD /app/.venv/bin/uvicorn src.trashsorting.api:app --host 0.0.0.0 --port ${PORT:-8080}
