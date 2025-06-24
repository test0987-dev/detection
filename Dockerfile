FROM python:3.10-slim

WORKDIR /app

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install detectron2 (this takes time, so do it before copying code)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads static/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
