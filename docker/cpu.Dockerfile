FROM python:3.11.6-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODE=CPU

# Image license metadata
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Upgrade pip to a fixed version
RUN pip install --no-cache-dir pip==23.3.1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script and data
COPY CeramicNet+PointTransformer.py .
COPY ceramicnet_data/ ./ceramicnet_data/

# Set the default command
CMD ["python", "CeramicNet+PointTransformer.py"] 