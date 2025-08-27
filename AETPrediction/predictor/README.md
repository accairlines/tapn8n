# AET Prediction API

This is a Django-based API for predicting Aircraft Elapsed Time (AET) using machine learning models.

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Test the API:**
   ```bash
   python test_api.py
   ```

### Manual Docker Build

1. **Build the image:**
   ```bash
   docker build -t aet-predictor .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -e DB_NAME=flight_ops \
     -e DB_USER=root \
     -e DB_PASSWORD=your_password \
     -e DB_HOST=your_db_host \
     -e DB_PORT=3306 \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/logs:/app/logs \
     aet-predictor
   ```

## API Endpoints

### Root Endpoint
- **GET /** - API information and available endpoints

### Health Check
- **GET /health/** - Health check for monitoring

### Predictions
- **GET /predict/{flight_id}/** - Predict AET for a specific flight
- **POST /predict/batch/** - Batch prediction for recent flights

### Admin
- **GET /admin/** - Django admin interface

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `False` | Django debug mode |
| `DJANGO_SECRET_KEY` | `your-secret-key-here` | Django secret key |
| `DB_NAME` | `flight_ops` | Database name |
| `DB_USER` | `root` | Database user |
| `DB_PASSWORD` | `` | Database password |
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `3306` | Database port |
| `AET_MODEL_PATH` | `/app/models/model.pkl` | Path to trained model |
| `AET_DATA_PATH` | `/app/data` | Path to data files |
| `AET_LOG_PATH` | `/app/logs` | Path to log files |

## Troubleshooting

### "Not Found" Errors
The "Not Found" errors you were seeing were because the API didn't have a root endpoint (`/`) defined. This has been fixed by adding:
- Root endpoint (`/`) - Shows API information
- Health endpoint (`/health/`) - For monitoring

### Model Loading Issues
- Ensure the model file exists at `/app/models/model.pkl`
- Check the `AET_MODEL_PATH` environment variable
- Verify the model file has the correct format

### Database Connection Issues
- Verify database credentials and connection details
- Ensure the database is accessible from the container
- Check firewall and network settings

## Testing

Run the test script to verify all endpoints:
```bash
python test_api.py
```

This will test:
- Root endpoint (should return API info)
- Health endpoint (should return health status)
- Admin endpoint (should redirect to login)
- Prediction endpoints (will fail without DB connection - this is expected)

## Logs

Logs are written to:
- Console output (for Docker logs)
- `/app/logs/django.log` (Django application logs)
- `/app/logs/training.log` (Training logs)

## Model Information

The API loads a pre-trained machine learning model that predicts:
- Taxi out time
- Airborne time  
- Taxi in time

The model is loaded at startup and can be reloaded if needed.
