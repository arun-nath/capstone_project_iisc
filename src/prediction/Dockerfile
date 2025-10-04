# Base image with Python 3.11
FROM python:3.10-slim

# FROM tensorflow/tensorflow:latest-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY lstm_1hr_prediction_2_atr_multiplier.keras .
COPY scaler.joblib .

# Copy app files
COPY main.py .

# Default command
CMD ["python", "main.py"]
