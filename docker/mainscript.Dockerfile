FROM python:3.11.6-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODE=CPU

# Image license metadata
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to a fixed version
RUN pip install --no-cache-dir pip==23.3.1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter and nbconvert
RUN pip install --no-cache-dir jupyter nbconvert

# Copy the notebook and data
COPY mainscript.ipynb .
COPY ceramicnet_data/ ./ceramicnet_data/

# Create output directory
RUN mkdir -p output

# Set the default command to run the notebook
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "mainscript.ipynb", "--output", "output/mainscript_executed.ipynb"] 