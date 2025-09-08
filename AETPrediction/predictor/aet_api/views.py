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
        
        logger.info(f"Flight data details: {json.dumps(flight_data, default=str)}")
        logger.info(f"Flight data: {flight_data}")
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
    """Get flight data for prediction - returns dictionary format expected by model"""
    try:
        logger.info(f"Getting flight data for ID: {flight_id} (type: {type(flight_id)})")
        # Set start date to current UTC time
        start_date = (timezone.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (timezone.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        flight_df = get_flight_data_for_prediction(start_date=start_date, end_date=end_date, days_back=None, flight_id=flight_id)
        
        # Check if we got any data
        if flight_df.empty:
            return None
        
        # Convert DataFrame to dictionary format expected by the model
        # Take the first row (should be only one for single flight)
        flight_row = flight_df.iloc[0]
        
        # Map database extractor columns to model expected format
        flight_data = {
            'ID': flight_row.get('ID'),
            'AIRCRAFT_ICAO_TYPE': flight_row.get('AIRCRAFT_ICAO_TYPE', 'A320'),
            'DEPARTURE_AIRP': flight_row.get('FROM_IATA', 'UNK'),
            'ARRIVAL_AIRP': flight_row.get('TO_IATA', 'UNK'),
            'STD': flight_row.get('STD'),
            'TAXI_OUT_TIME': flight_row.get('TAXI_OUT_TIME', 15),  # Default 15 minutes
            'FLIGHT_TIME': flight_row.get('FLIGHT_TIME', 60),      # Default 60 minutes
            'TAXI_IN_TIME': flight_row.get('TAXI_IN_TIME', 10),    # Default 10 minutes
            'CALLSIGN': flight_row.get('CALL_SIGN', ''),
            'AC_REGISTRATION': flight_row.get('AC_REGISTRATION', ''),
            'OPERATOR': flight_row.get('OPERATOR', ''),
            'FLT_NR': flight_row.get('FLT_NR', ''),
            'FROM_TERMINAL': flight_row.get('FROM_TERMINAL', ''),
            'TO_TERMINAL': flight_row.get('TO_TERMINAL', ''),
            'FROM_GATE': flight_row.get('FROM_GATE', ''),
            'FROM_STAND': flight_row.get('FROM_STAND', ''),
            'TO_STAND': flight_row.get('TO_STAND', ''),
            'CAPTAIN': flight_row.get('CAPTAIN', ''),
            'AIRLINE_SPEC': flight_row.get('AIRLINE_SPEC', ''),
            'PERFORMANCE_FACTOR': flight_row.get('PERFORMANCE_FACTOR', 1.0),
            'ROUTE_NAME': flight_row.get('ROUTE_NAME', ''),
            'ROUTE_OPTIMIZATION': flight_row.get('ROUTE_OPTIMIZATION', ''),
            'CRUISE_CI': flight_row.get('CRUISE_CI', 0),
            'CLIMB_PROC': flight_row.get('CLIMB_PROC', ''),
            'CRUISE_PROC': flight_row.get('CRUISE_PROC', ''),
            'DESCENT_PRO': flight_row.get('DESCENT_PRO', ''),
            'GREAT_CIRC': flight_row.get('GREAT_CIRC', 0),
            'ZERO_FUEL_WEIGHT': flight_row.get('ZERO_FUEL_WEIGHT', 0),
            'BODYTYPE': flight_row.get('BODYTYPE', ''),
            'EQUIPTYPE': flight_row.get('EQUIPTYPE', ''),
            'EQUIPTYPE2': flight_row.get('EQUIPTYPE2', ''),
            # Add calculated fields that the model might expect
            'route_distance': flight_row.get('GREAT_CIRC', 500),
            'max_flight_time': flight_row.get('FLIGHT_TIME', 90),
            'mel_count': 0,  # Default - would need to calculate from MEL data
            'avg_wind_speed': 10,  # Default - would need to calculate from waypoints
            'max_wind_speed': 20,  # Default - would need to calculate from waypoints
            'avg_temperature': 15,  # Default - would need to calculate from waypoints
            'max_altitude': 35000   # Default - would need to calculate from waypoints
        }
        
        return flight_data
        
    except Exception as e:
        logger.error(f"Error getting flight data for ID {flight_id}: {str(e)}")
        return None

def get_recent_flights(minutes=30):
    """Get flight IDs that departed in the last N minutes"""
    try:
        # Set start date to current UTC time
        start_date = (timezone.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d')
        end_date = (timezone.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        flight_df = get_flight_data_for_prediction(start_date=start_date, end_date=end_date, days_back=None, flight_id=None)
        
        # Extract flight IDs from the DataFrame
        if flight_df.empty:
            return []
        
        # Return list of flight IDs
        return flight_df['ID'].tolist() if 'ID' in flight_df.columns else []
        
    except Exception as e:
        logger.error(f"Error getting recent flights: {str(e)}")
        return []

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