FROM python:3.12-slim

# mysqlclient runtime headers (safe to include)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb-dev pkg-config build-essential \
 && rm -rf /var/lib/apt/lists/*

# upgrade pip/setuptools/wheel so PEP 517 builds (if any) work smoothly
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Run Django server
CMD ["gunicorn", "--workers=2", "--threads=8", "--timeout=60", "--keep-alive=5", "--bind=0.0.0.0:8000", "aet_predictor.wsgi:application"]