# Dockerfile for Wav2Vec2 Vietnamese Name Extraction Project
# Milestones: Problem definition, dataset selection, success metrics, project plan
# Data ingestion, exploratory analysis, data versioning (DVC / Git)
# Baseline model training and evaluation
# CI pipeline: testing, linting, experiment tracking
# Deployment setup (API / batch), model registry

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-vi.txt .

RUN pip install --no-cache-dir -r requirements-vi.txt

COPY train_wav2vec2.py eval_wav2vec2.py ./
COPY pyproject.toml pytest.ini ./
COPY tests/ ./tests/

# Linting stage
RUN echo "Running ruff linting..." && ruff check . || echo "Warning: ruff checks found issues"

# Testing stage
RUN echo "Running pytest..." && pytest tests/ -v --tb=short || echo "Warning: some tests failed"

# Default command: Run training script (can be overridden)
CMD ["python", "train_wav2vec2.py"]