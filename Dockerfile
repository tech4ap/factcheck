FROM python:3.12-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY uv.lock ./

# Install the project and its dependencies (excluding dev dependencies)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Stage 2: Runtime
FROM python:3.12-slim-bookworm AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the application and its virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY src/ ./src/
COPY models/ ./models/ 

# Make sure we have the pyproject.toml for proper Python path setup
COPY pyproject.toml ./

# Create necessary directories
RUN mkdir -p /tmp/deepfake_cache /tmp/deepfake_s3_cache logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PATH="/app/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port (optional, for monitoring)
EXPOSE 8080

# Default command - run SQS consumer
CMD ["python", "-m", "src.aws.sqs_deepfake_consumer"] 
