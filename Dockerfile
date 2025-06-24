FROM python:3.10-slim


WORKDIR/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Install detectron2 from source
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    pip install -e .

# Copy the project files
COPY . .

# Expose the port your Flask app runs on (modify if needed)
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
