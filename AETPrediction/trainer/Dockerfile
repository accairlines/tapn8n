FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py preprocess.py codification.py ./

# Create directories
RUN mkdir -p /app/logs /app/data

# Run training script
CMD ["python", "train.py"] 