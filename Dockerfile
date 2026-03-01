

# ── Stage 1: Base Python image ──────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for faiss, sentence-transformers, pypdf
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml .
COPY uv.lock* .

# Install uv for fast installs
RUN pip install uv

# Install all Python dependencies
RUN uv pip install --system \
    fastapi \
    uvicorn \
    python-dotenv \
    python-multipart \
    faiss-cpu \
    sentence-transformers \
    langchain \
    langchain-groq \
    langchain-community \
    pypdf \
    numpy \
    pydantic

# Copy the rest of the project
COPY . .

# Create upload directory
RUN mkdir -p data/uploaded

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the API server
CMD ["python", "api.py"]