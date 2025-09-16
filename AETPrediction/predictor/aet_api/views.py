from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
from django.utils import timezone
from datetime import datetime, timedelta
import json
import logging
from .model_loader import ModelLoader
from .db_extractor import get_flight_data_for_prediction 
from .db_extractor import get_flight_hist_aeteet
from django.conf import settings
import pandas as pd
import traceback

logger = logging.getLogger(__name__)

# Initialize model loader
model_loader = ModelLoader(settings.MODEL_PATH)

@csrf_exempt
def predict_flight(request, flight_id):
    """Predict AET for a specific flight"""
    start_time = datetime.now()
    try:
        # Validate flight_id
        if not flight_id or (isinstance(flight_id, str) and not flight_id.strip()):
            return JsonResponse({
                'error': 'Invalid flight ID provided'
            }, status=400)
        
        # Convert to string and strip whitespace
        flight_id = str(flight_id).strip()
        
        half_way = datetime.now()
        # Get flight data from database
        flight_data = get_flight_data(flight_id)
        logger.info(f"Prediction for flight {flight_id}: Flight data: {len(flight_data)}, processing time: {datetime.now() - half_way}")
        
        if flight_data is None:
            return JsonResponse({
                'error': 'Flight not found'
            }, status=404)
        
        logger.debug(f"Flight data details: {json.dumps(flight_data, default=str)}")
        # Make prediction
        half_way = datetime.now()
        prediction = model_loader.predict(flight_data)
        logger.info(f"Prediction for flight {flight_id}: Prediction: {len(prediction)}, processing time: {datetime.now() - half_way}")
        
        half_way = datetime.now()
        hist_aeteet = get_flight_hist_aeteet(flight_id)
        logger.info(f"Prediction for flight {flight_id}: Hist: {len(hist_aeteet)}, processing time: {datetime.now() - half_way}")
        
        end_time = datetime.now()
        
        # Format response
        response = format_prediction_response(flight_id, prediction, flight_data, hist_aeteet, (end_time - start_time).total_seconds())
        
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
        logger.debug(f"Getting flight data for ID: {flight_id} (type: {type(flight_id)})")
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
        raise e

def format_prediction_response(flight_id, prediction, flight_data, hist_aeteet, processing_time):
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
            'delta_percentage': 0,
            'predict': str(prediction)
        }
    
    logger.info(f"Prediction output structure for flight {flight_id}: {str(prediction)}, processing time: {processing_time}")
        
    return {
        'flight_id': flight_id,
        'std': flight_data['STD'],
        'sta': flight_data['STA'],
        'planned_taxi_out': round(flight_data['planned_taxi_out']) if flight_data['planned_taxi_out'] is not None else 0,
        'planned_airborne': round(flight_data['planned_airborne']) if flight_data['planned_airborne'] is not None else 0,
        'planned_taxi_in': round(flight_data['planned_taxi_in']) if flight_data['planned_taxi_in'] is not None else 0,
        'planned_total_time': round(flight_data['planned_total_time']) if flight_data['planned_total_time'] is not None else 0,
        'actual_taxi_out': round(flight_data['actual_taxi_out']) if flight_data['actual_taxi_out'] is not None else -1,
        'actual_airborne': round(flight_data['actual_airborne']) if flight_data['actual_airborne'] is not None else -1,
        'actual_taxi_in': round(flight_data['actual_taxi_in']) if flight_data['actual_taxi_in'] is not None else -1,
        'actual_total_time': round(flight_data['actual_total_time']) if flight_data['actual_total_time'] is not None else -1,
        'aet': round(flight_data['AET']) if flight_data['AET'] is not None else -1,
        'eet': round(flight_data['EET']) if flight_data['EET'] is not None else -1,
        'delta_percentage': round(prediction['delta']) if prediction['delta'] is not None else -1,
        'hist_aeteet': round(hist_aeteet['DELTA'].iloc[0]/60) if hist_aeteet['DELTA'].iloc[0] is not None else -1
    } 
    
def calculate_planned_actual_times(flight_row):
    """Calculate actual taxi_out, airborne, and taxi_in times using correct logic, and add them to each flight dict."""
    # Helper to parse datetimes safely
    def parse_dt(val):
        try:
            return pd.to_datetime(val, errors='coerce')
        except Exception:
            return pd.NaT

    # Parse all relevant datetimes
    offblock = parse_dt(flight_row.get('OFFBLOCK'))
    mvt = parse_dt(flight_row.get('MVT'))
    ata = parse_dt(flight_row.get('ATA'))
    onblock = parse_dt(flight_row.get('ONBLOCK'))
    from_timediff = flight_row.get('FROM_TIMEDIFF')
    to_timediff = flight_row.get('TO_TIMEDIFF')
    
    # Calculate planned times (in seconds)
    planned_taxi_out = flight_row.get('TAXI_OUT_TIME').total_seconds() / 60 if flight_row.get('TAXI_OUT_TIME') is not None else 0
    flight_time = flight_row.get('FLIGHT_TIME') or flight_row.get('TRIP_DURATION')
    planned_airborne = flight_time.total_seconds() / 60 if flight_time is not None else 0
    planned_taxi_in = flight_row.get('TAXI_IN_TIME').total_seconds() / 60 if flight_row.get('TAXI_IN_TIME') is not None else 0

    # Calculate actual times (in minutes)
    actual_taxi_out = (mvt - offblock).total_seconds() / 60 if pd.notnull(mvt) and pd.notnull(offblock) else None
    actual_airborne = ((ata - mvt).total_seconds() - (to_timediff * 60)  + (from_timediff * 60)) / 60 if pd.notnull(ata) and pd.notnull(mvt) else None
    actual_taxi_in = (onblock - ata).total_seconds() / 60 if pd.notnull(onblock) and pd.notnull(ata) else None
        
    # Calculate totals
    actual_total_time = actual_taxi_out + actual_airborne + actual_taxi_in if actual_taxi_out is not None and actual_airborne is not None and actual_taxi_in is not None else None
    planned_total_time = planned_taxi_out + planned_airborne + planned_taxi_in
    
    # AET: ATA - MVT (in minutes)
    aet = actual_total_time
    # EET: planned total time (in minutes)
    eet = planned_total_time
    # Delta: AET - EET (in minutes)
    raw_delta = (aet - eet) if aet is not None and eet is not None else None
    # Convert delta to percentage of EET
    actual_delta = (raw_delta / eet * 100) if raw_delta is not None and eet is not None and eet != 0 else None

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
    flight_data['actual_delta'] = raw_delta

    return flight_data
