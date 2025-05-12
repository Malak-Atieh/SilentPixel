FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirments.txt

# Copy code and model files
COPY . . 

# Create models directory if it doesn't exist
RUN mkdir -p models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=6000
ENV STEG_MODEL_PATH=/app/models/steg_model.pth

EXPOSE 6000

CMD ["python", "app.py"]
