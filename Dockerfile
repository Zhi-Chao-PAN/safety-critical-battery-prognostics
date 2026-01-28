# Dockerfile for Spatial Bayesian vs Deep Learning
# Ensures fully reproducible environment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command: run tests
CMD ["pytest", "tests/", "-v"]

# Alternative commands:
# docker run spatial-bayes python src/evaluate_rigor.py
# docker run spatial-bayes python src/train_bayes_hierarchical.py
