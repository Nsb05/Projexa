# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed by opencv, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the project (including app.py and .pth model)
COPY . .

# Railway sets PORT env; default to 8000 for local runs
ENV PORT=8000

# Expose the port (not strictly needed for Railway but nice for local)
EXPOSE 8000

# Start the Flask app via gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
