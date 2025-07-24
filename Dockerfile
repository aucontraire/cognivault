# Multi-stage Dockerfile for CogniVault API
# Optimized for both development and production use

# Development stage with all tools and hot reload
FROM python:3.12-slim as development

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Copy source code and project files
COPY src/ ./src/
COPY examples/ ./examples/
COPY README.md ./

# Install the project
RUN poetry install

# Set Python path
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Default command for development
CMD ["poetry", "run", "uvicorn", "cognivault.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage with minimal dependencies
FROM python:3.12-slim as production

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from development stage
COPY --from=development /app/.venv /app/.venv

# Copy source code and project files
COPY src/ ./src/
COPY README.md ./

# Set environment variables
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "cognivault.api.main:app", "--host", "0.0.0.0", "--port", "8000"]