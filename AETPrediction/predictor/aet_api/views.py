from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
from django.utils import timezone
from datetime import datetime, timedelta
import json
import logging
from .model_loader import ModelLoader
from .db_extractor import get_flight_data_for_prediction
from django.conf import settings

logger = logging.getLogger(__name__)

# Initialize model loader
model_loader = ModelLoader(settings.MODEL_PATH)

@csrf_exempt
def predict_flight(request, flight_id):
    """Predict AET for a specific flight"""
    try:
        # Get flight data from database
        flight_data = get_flight_data(flight_id)
        
        if not flight_data:
            return JsonResponse({
                'error': 'Flight not found'
            }, status=404)
        
        # Make prediction
        prediction = model_loader.predict(flight_data)
        
        # Format response
        response = format_prediction_response(flight_id, prediction, flight_data)
        
        return JsonResponse(response)
        
    except Exception as e:
        logger.error(f"Prediction error for flight {flight_id}: {str(e)}")
        return JsonResponse({
            'error': 'Prediction failed',
            'message': str(e)
        }, status=500)

def predict_batch(request):
    """Batch prediction for recent flights (called by cron)"""
    try:
        # Get flights from last 30 minutes
        recent_flights = get_recent_flights(minutes=30)
        
        predictions = []
        for flight_id in recent_flights:
            try:
                flight_data = get_flight_data(flight_id)
                if flight_data:
                    prediction = model_loader.predict(flight_data)
                    result = format_prediction_response(flight_id, prediction, flight_data)
                    predictions.append(result)
            except Exception as e:
                logger.error(f"Batch prediction error for last 30 minutes flights: {str(e)}")
        
        return JsonResponse({
            'timestamp': timezone.now().isoformat(),
            'flight_count': len(predictions),
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return JsonResponse({
            'error': 'Batch prediction failed',
            'message': str(e)
        }, status=500)

def get_flight_data(flight_id):
    # Set start date to current UTC time
    start_date = (timezone.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (timezone.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    flight_data = get_flight_data_for_prediction(start_date=start_date, end_date=end_date, days_back=None, flight_id=flight_id)
        
    return flight_data

def get_recent_flights(minutes=30):
    # Set start date to current UTC time
    start_date = (timezone.now() - timedelta(minutes=30)).strftime('%Y-%m-%d')
    end_date = (timezone.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    flight_data = get_flight_data_for_prediction(start_date=start_date, end_date=end_date, days_back=None, flight_id=None)

    return flight_data

def format_prediction_response(flight_id, prediction, flight_data):
    """Format prediction into API response"""
    # Convert predictions from minutes to HH:MM:SS format
    def minutes_to_time(minutes):
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    
    # Calculate total predicted AET
    total_predicted = (
        prediction['taxi_out'] + 
        prediction['airborne'] + 
        prediction['taxi_in']
    )
    
    # Calculate planned EET
    planned_eet = (
        (flight_data.get('TAXI_OUT_TIME') or 15) +
        (flight_data.get('FLIGHT_TIME') or 60) +
        (flight_data.get('TAXI_IN_TIME') or 10)
    )
    
    # Calculate delta
    delta_minutes = total_predicted - planned_eet
    
    return {
        'flight_id': flight_id,
        'predicted_aet': minutes_to_time(total_predicted),
        'predicted_breakdown': {
            'taxi_out': minutes_to_time(prediction['taxi_out']),
            'airborne_time': minutes_to_time(prediction['airborne']),
            'taxi_in': minutes_to_time(prediction['taxi_in'])
        },
        'planned_eet': minutes_to_time(planned_eet),
        'delta_minutes': round(delta_minutes, 1)
    } 