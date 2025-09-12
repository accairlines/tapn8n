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
import pandas as pd
import traceback

logger = logging.getLogger(__name__)

# Initialize model loader
model_loader = ModelLoader(settings.MODEL_PATH)

@csrf_exempt
def predict_flight(request, flight_id):
    """Predict AET for a specific flight"""
    try:
        # Validate flight_id
        if not flight_id or (isinstance(flight_id, str) and not flight_id.strip()):
            return JsonResponse({
                'error': 'Invalid flight ID provided'
            }, status=400)
        
        # Convert to string and strip whitespace
        flight_id = str(flight_id).strip()
        
        # Get flight data from database
        flight_data = get_flight_data(flight_id)
        
        if not flight_data:
            return JsonResponse({
                'error': 'Flight not found'
            }, status=404)
        
        logger.info(f"Flight data details: {json.dumps(flight_data, default=str)}")
        # Make prediction
        prediction = model_loader.predict(flight_data)
        
        # Format response
        response = format_prediction_response(flight_id, prediction, flight_data)
        
        return JsonResponse(response)
        
    except Exception as e:
        error = e.args[0]
        logging.error(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        return JsonResponse({
            'error': 'Prediction failed',
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
            logger.warning(f"No flight data found for ID: {flight_id}")
            return None
        
        # Convert DataFrame to dictionary format expected by the model
        # Take the first row (should be only one for single flight)
        flight_row = flight_df.iloc[0]
        
        # Calculate planned and actual times
        flight_data = calculate_planned_actual_times(flight_row)
        
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
        if minutes is None or pd.isna(minutes):
            return "00:00:00"
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    
    # Validate prediction structure
    if not isinstance(prediction, dict):
        logger.error(f"Invalid prediction structure for flight {flight_id}: {prediction}")
        return {
            'flight_id': flight_id,
            'error': 'Invalid prediction data',
            'predicted_aet': "00:00:00",
            'predicted_breakdown': {
                'taxi_out': "00:00:00",
                'airborne_time': "00:00:00",
                'taxi_in': "00:00:00"
            },
            'planned_eet': "00:00:00",
            'delta_minutes': 0,
            'predict': str(prediction)
        }
    
    logger.info(f"Prediction output structure for flight {flight_id}: {str(prediction)}")
    
    # Calculate total predicted AET
    total_predicted = (
        prediction.get('taxi_out', 0) + 
        prediction.get('airborne', 0) + 
        prediction.get('taxi_in', 0)
    )
    
    # Calculate planned EET using more robust method
    def safe_get_time(data, key, default_minutes):
        value = data.get(key)
        if value is None or pd.isna(value):
            return default_minutes
        # If it's already a number, return it
        if isinstance(value, (int, float)):
            return value
        # If it's a time string, convert to minutes
        try:
            if isinstance(value, str) and ':' in value:
                h, m, s = map(int, value.split(':'))
                return h * 60 + m + s / 60
        except (ValueError, AttributeError):
            pass
        return default_minutes
    
    planned_eet = (
        safe_get_time(flight_data, 'TAXI_OUT_TIME', 15) +
        safe_get_time(flight_data, 'FLIGHT_TIME', 60) +
        safe_get_time(flight_data, 'TAXI_IN_TIME', 10)
    )
    
    # Calculate delta
    delta_minutes = total_predicted - planned_eet
    
    return {
        'flight_id': flight_id,
        'predicted_aet': minutes_to_time(total_predicted),
        'predicted_breakdown': {
            'taxi_out': minutes_to_time(prediction.get('taxi_out', 0)),
            'airborne_time': minutes_to_time(prediction.get('airborne', 0)),
            'taxi_in': minutes_to_time(prediction.get('taxi_in', 0))
        },
        'planned_eet': minutes_to_time(planned_eet),
        'delta_minutes': round(delta_minutes, 1)
    } 
    
def calculate_planned_actual_times(flight_row):
    """Calculate actual taxi_out, airborne, and taxi_in times using correct logic, and add them to each flight dict."""
    # Helper to parse datetimes safely
    def parse_dt(val):
        try:
            return pd.to_datetime(val, errors='coerce')
        except Exception:
            return pd.NaT

    # Helper to parse time string to seconds
    def parse_time_to_seconds(time_str, default='00:00:00'):
        try:
            if pd.isna(time_str) or time_str is None:
                time_str = default
            h, m, s = map(int, str(time_str).split(':'))
            return h * 3600 + m * 60 + s
        except (ValueError, AttributeError):
            return 0

    # Parse all relevant datetimes
    offblock = parse_dt(flight_row.get('OFFBLOCK'))
    mvt = parse_dt(flight_row.get('MVT'))
    ata = parse_dt(flight_row.get('ATA'))
    onblock = parse_dt(flight_row.get('ONBLOCK'))
    eta = parse_dt(flight_row.get('ETA'))
    
    # Calculate planned times (in seconds)
    planned_taxi_out = parse_time_to_seconds(flight_row.get('TAXI_OUT_TIME'))
    flight_time = flight_row.get('FLIGHT_TIME') or flight_row.get('TRIP_DURATION')
    planned_airborne = parse_time_to_seconds(flight_time)
    planned_taxi_in = parse_time_to_seconds(flight_row.get('TAXI_IN_TIME'))

    # Calculate actual times (in minutes)
    actual_taxi_out = (mvt - offblock).total_seconds() / 60 if pd.notnull(mvt) and pd.notnull(offblock) else None
    actual_airborne = (ata - mvt).total_seconds() / 60 if pd.notnull(ata) and pd.notnull(mvt) else None
    actual_taxi_in = (onblock - ata).total_seconds() / 60 if pd.notnull(onblock) and pd.notnull(ata) else None
    
    # Calculate totals
    actual_total_time = sum(
        x for x in [actual_taxi_out, actual_airborne, actual_taxi_in] if x is not None
    )
    planned_total_time = planned_taxi_out + planned_airborne + planned_taxi_in
    
    # AET: ATA - MVT (in minutes)
    aet = (ata - mvt).total_seconds() / 60 if pd.notnull(ata) and pd.notnull(mvt) else None
    # EET: planned total time (in minutes)
    eet = planned_total_time / 60
    # Delta: AET - EET
    actual_delta = (aet - eet) if aet is not None and eet is not None else None

    # Create a copy of the flight_row to avoid SettingWithCopyWarning
    flight_data = flight_row.copy()
    
    # Add calculated values to the copy
    flight_data['actual_taxi_out'] = actual_taxi_out
    flight_data['actual_airborne'] = actual_airborne
    flight_data['actual_taxi_in'] = actual_taxi_in
    flight_data['actual_total_time'] = actual_total_time
    flight_data['planned_taxi_out'] = planned_taxi_out
    flight_data['planned_airborne'] = planned_airborne
    flight_data['planned_taxi_in'] = planned_taxi_in
    flight_data['planned_total_time'] = planned_total_time
    flight_data['AET'] = aet
    flight_data['EET'] = eet
    flight_data['actual_delta'] = actual_delta

    return flight_data
