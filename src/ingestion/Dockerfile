FROM python:3.11-slim

# Install system deps needed for psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    zip \
 && rm -rf /var/lib/apt/lists/*

# Create Lambda layer directory
WORKDIR /opt
RUN mkdir -p python

# Install Python libs
RUN pip install --no-cache-dir \
    yfinance \
    sqlalchemy \
    psycopg2-binary \
    -t python/

# Package into zip
RUN zip -r9 /layer.zip python