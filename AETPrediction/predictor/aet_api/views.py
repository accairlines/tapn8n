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
            logger.warning(f"No flight data found for ID: {flight_id}")
            return None
        
        # Convert DataFrame to dictionary format expected by the model
        # Take the first row (should be only one for single flight)
        flight_row = flight_df.iloc[0]
        
        # Calculate planned and actual times
        flight_data = calculate_planned_actual_times(flight_row)
        
        # Map database extractor columns to model expected format
        model_data = {
            # Basic flight information
            'ID': flight_data.get('ID'),
            'OPERATOR': flight_data.get('OPERATOR', ''),
            'FLT_NR': flight_data.get('FLT_NR', ''),
            'AC_REGISTRATION': flight_data.get('AC_REGISTRATION', ''),
            'FROM_IATA': flight_data.get('FROM_IATA', 'UNK'),
            'TO_IATA': flight_data.get('TO_IATA', 'UNK'),
            'STD': flight_data.get('STD'),
            'ETD': flight_data.get('ETD'),
            'ATD': flight_data.get('ATD'),
            'STA': flight_data.get('STA'),
            'ETA': flight_data.get('ETA'),
            'FROM_STAND': flight_data.get('FROM_STAND', ''),
            'TO_STAND': flight_data.get('TO_STAND', ''),
            'AC_READY': flight_data.get('AC_READY'),
            'TSAT': flight_data.get('TSAT'),
            'TOBT': flight_data.get('TOBT'),
            'CTOT': flight_data.get('CTOT'),
            'CALL_SIGN': flight_data.get('CALL_SIGN', ''),
            'SERV_TYP_COD': flight_data.get('SERV_TYP_COD', ''),
            'MVT': flight_data.get('MVT'),
            
            # Flight plan data with fp_ prefix
            'fp_CAPTAIN': flight_data.get('fp_CAPTAIN', ''),
            'fp_AIRCRAFT_ICAO_TYPE': flight_data.get('fp_AIRCRAFT_ICAO_TYPE', ''),
            'fp_AIRLINE_SPEC': flight_data.get('fp_AIRLINE_SPEC', ''),
            'fp_PERFORMANCE_FACTOR': flight_data.get('fp_PERFORMANCE_FACTOR', 1.0),
            'fp_ROUTE_NAME': flight_data.get('fp_ROUTE_NAME', ''),
            'fp_ROUTE_OPTIMIZATION': flight_data.get('fp_ROUTE_OPTIMIZATION', ''),
            'fp_CRUISE_CI': flight_data.get('fp_CRUISE_CI', 0),
            'fp_CLIMB_PROC': flight_data.get('fp_CLIMB_PROC', ''),
            'fp_CRUISE_PROC': flight_data.get('fp_CRUISE_PROC', ''),
            'fp_DESCENT_PROC': flight_data.get('fp_DESCENT_PROC', ''),
            'fp_GREAT_CIRC': flight_data.get('fp_GREAT_CIRC', 0),
            'fp_ZERO_FUEL_WEIGHT': flight_data.get('fp_ZERO_FUEL_WEIGHT', 0),
            
            # Equipment data with eq_ prefix
            'eq_BODYTYPE': flight_data.get('eq_BODYTYPE', ''),
            'eq_EQUIPMENTTYPE': flight_data.get('eq_EQUIPMENTTYPE', ''),
            'eq_EQUIPMENTTYPE2': flight_data.get('eq_EQUIPMENTTYPE2', ''),
            
            # Waypoint data (wp1_ to wp50_)
            **{f'wp{i}_SEG_WIND_DIRECTION': flight_data.get(f'wp{i}_SEG_WIND_DIRECTION', -1) for i in range(1, 51)},
            **{f'wp{i}_SEG_WIND_SPEED': flight_data.get(f'wp{i}_SEG_WIND_SPEED', -1) for i in range(1, 51)},
            **{f'wp{i}_SEG_TEMPERATURE': flight_data.get(f'wp{i}_SEG_TEMPERATURE', -1) for i in range(1, 51)},
            
            # ACARS data (acars1_ to acars20_)
            **{f'acars{i}_WINDDIRECTION': flight_data.get(f'acars{i}_WINDDIRECTION', -1) for i in range(1, 21)},
            **{f'acars{i}_WINDSPEED': flight_data.get(f'acars{i}_WINDSPEED', -1) for i in range(1, 21)},
            
            # Actual times and AET (from calculated values)
            'actual_taxi_out': flight_data.get('actual_taxi_out', 15),
            'actual_airborne': flight_data.get('actual_airborne', 60),
            'actual_taxi_in': flight_data.get('actual_taxi_in', 10),
            'AET': flight_data.get('AET'),
            'OFFBLOCK': flight_data.get('OFFBLOCK'),
            'ATA': flight_data.get('ATA'),
            'ONBLOCK': flight_data.get('ONBLOCK'),
            
            # Additional fields for compatibility
            'CALL_SIGN': flight_data.get('CALL_SIGN', ''),
            'FROM_TERMINAL': flight_data.get('FROM_TERMINAL', ''),
            'TO_TERMINAL': flight_data.get('TO_TERMINAL', ''),
            'FROM_GATE': flight_data.get('FROM_GATE', ''),
            'CAPTAIN': flight_data.get('CAPTAIN', ''),
            'AIRCRAFT_ICAO_TYPE': flight_data.get('AIRCRAFT_ICAO_TYPE', 'A320'),
            'AIRLINE_SPEC': flight_data.get('AIRLINE_SPEC', ''),
            'PERFORMANCE_FACTOR': flight_data.get('PERFORMANCE_FACTOR', 1.0),
            'ROUTE_NAME': flight_data.get('ROUTE_NAME', ''),
            'ROUTE_OPTIMIZATION': flight_data.get('ROUTE_OPTIMIZATION', ''),
            'CRUISE_CI': flight_data.get('CRUISE_CI', 0),
            'CLIMB_PROC': flight_data.get('CLIMB_PROC', ''),
            'CRUISE_PROC': flight_data.get('CRUISE_PROC', ''),
            'DESCENT_PRO': flight_data.get('DESCENT_PRO', ''),
            'GREAT_CIRC': flight_data.get('GREAT_CIRC', 0),
            'ZERO_FUEL_WEIGHT': flight_data.get('ZERO_FUEL_WEIGHT', 0),
            'BODYTYPE': flight_data.get('BODYTYPE', ''),
            'EQUIPTYPE': flight_data.get('EQUIPTYPE', ''),
            'EQUIPTYPE2': flight_data.get('EQUIPTYPE2', ''),
            
            # Legacy fields for backward compatibility
            'DEPARTURE_AIRP': flight_data.get('FROM_IATA', 'UNK'),
            'ARRIVAL_AIRP': flight_data.get('TO_IATA', 'UNK'),
            'TAXI_OUT_TIME': flight_data.get('TAXI_OUT_TIME', 15),
            'FLIGHT_TIME': flight_data.get('FLIGHT_TIME', 60),
            'TAXI_IN_TIME': flight_data.get('TAXI_IN_TIME', 10),
            'CALLSIGN': flight_data.get('CALL_SIGN', '')
        }
        
        return model_data
        
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
    if not isinstance(prediction, dict) or not all(key in prediction for key in ['taxi_out', 'airborne', 'taxi_in']):
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
            'delta_minutes': 0
        }
    
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