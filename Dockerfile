# Base image with CPU-only PyTorch + torchvision + numpy already installed
FROM pytorch/pytorch:2.2.0-cpu

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed by opencv, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install ONLY the light Python deps
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your project (app.py, model .pth, templates, static, etc.)
COPY . .

# Railway uses PORT env; default to 8000 for local
ENV PORT=8000

EXPOSE 8000

# Start Flask via gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
