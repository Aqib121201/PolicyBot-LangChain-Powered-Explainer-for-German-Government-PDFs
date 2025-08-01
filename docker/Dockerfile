# PolicyBot Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-deu \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Development dependencies
FROM base as development

# Install Python development dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy German model
RUN python -m spacy download de_core_news_sm

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external \
    models/embeddings models/classifiers models/summarizers \
    visualizations logs temp

# Stage 3: Production image
FROM base as production

# Install only production Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy German model
RUN python -m spacy download de_core_news_sm

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external \
    models/embeddings models/classifiers models/summarizers \
    visualizations logs temp

# Create non-root user
RUN useradd --create-home --shell /bin/bash policybot && \
    chown -R policybot:policybot /app
USER policybot

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 