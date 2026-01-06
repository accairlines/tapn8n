FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . . 

# Create directories
RUN mkdir -p /app/logs /app/data && touch /app/logs/run_train.log

# Copy automation scripts
COPY system/run_train.sh /app/system/run_train.sh
RUN chmod +x /app/system/run_train.sh

# Register cron job to run every 12 hours
RUN (crontab -l 2>/dev/null; echo "0 */12 * * * bash /app/system/run_train.sh >> /app/logs/run_train.log 2>&1") | crontab -

# Start cron daemon and stream logs to keep container alive
CMD ["bash", "-c", "cron && tail -f /app/logs/run_train.log"]
