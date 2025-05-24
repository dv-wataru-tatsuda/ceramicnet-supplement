# NVIDIA CUDA Runtime (Ubuntu 22.04) – CUDA 12.3
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODE=GPU

# Image license metadata
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Upgrade pip to a fixed version
RUN python -m pip install --no-cache-dir pip==23.3.1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# Copy the Python script and data
COPY CeramicNet+PointTransformer.py .
COPY ceramicnet_data/ ./ceramicnet_data/

# Set the default command
CMD ["python", "CeramicNet+PointTransformer.py"]