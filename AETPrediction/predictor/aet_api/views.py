from time import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
from django.utils import timezone
from datetime import datetime, timedelta, time
import logging
from .model_loader import ModelLoader
from .db_extractor import get_flight_data_for_prediction 
from .db_extractor import get_flight_hist_aeteet
from django.conf import settings
import pandas as pd
import traceback
import aet_api.train as train

logger = logging.getLogger(__name__)

# Initialize model loader
model_loader = ModelLoader(settings.MODEL_PATH)

@csrf_exempt
def predict_flight(request, flight_id):
    """Predict AET for a specific flight
    
    Query parameters:
        model: 'xgb', 'ft_transformer', or 'ensemble' (default: 'ensemble')
    """
    start_time = datetime.now()
    try:
        # Validate flight_id
        if not flight_id or (isinstance(flight_id, str) and not flight_id.strip()):
            return JsonResponse({
                'error': 'Invalid flight ID provided'
            }, status=400)
        
        # Convert to string and strip whitespace
        flight_id = str(flight_id).strip()
        
        # Get model type from query parameters (default: ensemble)
        model_type = request.GET.get('model', 'xgb').lower()
        if model_type not in ['xgb', 'ft_transformer', 'ensemble']:
            model_type = 'xgb'
            logger.warning(f"Invalid model type requested, defaulting to 'ensemble'")
        
        # Get flight data from database
        parcial_start = datetime.now()
        flight_data = get_flight_data(flight_id)
        parcial_end = datetime.now()
        logger.debug(f"Prediction for flight {flight_id}: Flight data: {len(flight_data)}, processing time: {parcial_end - parcial_start}")
        
        if flight_data is None:
            return JsonResponse({
                'error': 'Flight not found'
            }, status=404)
        
        # Make prediction
        parcial_start = datetime.now()
        prediction = model_loader.predict(flight_data, model_type=model_type)
        parcial_end = datetime.now()
        logger.debug(f"Prediction for flight {flight_id}: Model={model_type}, Prediction: {len(prediction)}, processing time: {parcial_end - parcial_start}")
        
        parcial_start = datetime.now()
        hist_aeteet = get_flight_hist_aeteet(flight_id)
        parcial_end = datetime.now()
        logger.debug(f"Prediction for flight {flight_id}: Hist: {len(hist_aeteet)}, processing time: {parcial_end - parcial_start}")
        
        end_time = datetime.now()
        
        # Format response
        response = format_prediction_response(flight_id, prediction, flight_data, hist_aeteet, (end_time - start_time).total_seconds(), model_type)
        
        return JsonResponse(response)
        
    except Exception as e:
        error = e.args[0]
        logging.error(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        return JsonResponse({
            'error': 'Prediction failed',
            'message': str(e)
        }, status=500)
    finally:
        flight_data = None
        prediction = None
        hist_aeteet = None

@csrf_exempt
def train_model(request):
    """Train the model"""
    logging.info("=== Starting montlhy training ===")
    
    try:
        # Load cached data if it exists
        cached_features, cached_targets = train.load_features_targets_from_csv()
        
        # Always process new data
        logging.info("=== Loading Data ===")
        flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations, folders_paths = train.load_data()
        logging.info("=== Loading Data Completed ===")
                
        # Prepare training data
        logging.info("=== Preparing Training Data ===")
        flights = train.prepare_training_data(
            flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations
        )
        logging.info("=== Preparing Training Data Completed ===")
        
        # Remove flights with no flight_plan
        logging.info("=== Removing Flights with No Flight Plan ===")
        flights = [flight for flight in flights if flight.get('flight_plan')]
        logging.info("=== Removing Flights with No Flight Plan Completed ===")
        
        # Calculate actual times
        logging.info("=== Calculating Actual Times ===")
        flights = train.calculate_planned_actual_times(flights)
        logging.info("=== Calculating Actual Times Completed ===")

        # Extract targets and features from new data
        logging.info("=== Extracting Targets and Features ===")
        new_features, new_targets = train.extract_targetsfeatures_from_flights(flights)
        logging.info("=== Extracting Targets and Features Completed ===")
        
        # Append new data to cached data if it exists
        if len(new_features) == 0:
            logging.info("Nothing to process as features are empty")
            return 0
        elif cached_features is not None and cached_targets is not None:
            logging.info("Appending new data to existing cached data...")
            features = pd.concat([cached_features, new_features], ignore_index=True)
            targets = pd.concat([cached_targets, new_targets], ignore_index=True)
            logging.info(f"Combined dataset - Features: {features.shape}, Targets: {targets.shape}")
        else:
            logging.info("No cached data found, using only new data")
            features, targets = new_features, new_targets
        
        # Save combined features and targets to CSV for caching
        logging.info(f"Features: {features.shape}, Targets: {targets.shape}")
        logging.info("=== Saving Combined Features and Targets to CSV ===")
        train.save_features_targets_to_csv(features, targets)
        logging.info("=== Saving Combined Features and Targets to CSV Completed ===")

        # Rename CSV files to .done
        logging.info("=== Renaming CSV Files to .done ===")
        train.rename_csv_files_to_done(folders_paths)
        logging.info("=== Renaming CSV Files to .done Completed ===")
                
        # Train models
        logging.info("=== Training Models ===")
        models, scaler, metrics, ft_transformer_models, ft_transformer_metrics = train.train_models(features, targets)
        logging.info("=== Training Models Completed ===")
        
        # Save models
        logging.info("=== Saving Models ===")
        train.save_models(models, scaler, metrics, ft_transformer_models, ft_transformer_metrics)
        logging.info("=== Saving Models Completed ===")
        
        # Log completion
        logging.info("=== Training completed successfully ===")
        return JsonResponse({'result': True})
    except Exception as e:
        error = e.args[0]
        logging.error(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        return JsonResponse({'result': False, 'error': str(e)})
    
    
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
        parcial_start = datetime.now()
        flight_data = calculate_planned_actual_times(flight_row)
        parcial_end = datetime.now()
        logger.debug(f"Calculating times for flightid: {flight_id}: Flight data: {len(flight_data)}, processing time: {parcial_end - parcial_start}")
        
        return flight_data
        
    except Exception as e:
        raise e

def format_prediction_response(flight_id, prediction, flight_data, hist_aeteet, processing_time, model_type):
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
        
    # Helper function to safely round values, handling NaN and None
    def safe_round(value, default=0):
        if value is None or pd.isna(value):
            return default
        try:
            return round(value)
        except (ValueError, TypeError):
            return default
    
    return {
        'flight_id': flight_id,
        'std': flight_data['STD'],
        'sta': flight_data['STA'],
        'planned_taxi_out': safe_round(flight_data['planned_taxi_out']),
        'planned_airborne': safe_round(flight_data['planned_airborne']),
        'planned_taxi_in': safe_round(flight_data['planned_taxi_in']),
        'planned_total_time': safe_round(flight_data['planned_total_time']),
        'actual_taxi_out': safe_round(flight_data['actual_taxi_out'], -1),
        'actual_airborne': safe_round(flight_data['actual_airborne'], -1),
        'actual_taxi_in': safe_round(flight_data['actual_taxi_in'], -1),
        'actual_total_time': safe_round(flight_data['actual_total_time'], -1),
        'aet': safe_round(flight_data['AET'], -1),
        'eet': safe_round(flight_data['EET'], -1),
        'delta_percentage': safe_round(prediction['xgb'], -1) if model_type == 'xgb' else 0,
        'delta_percentage_ft': safe_round(prediction['ft_transformer'], -1) if model_type == 'ft_transformer' else 0,
        'hist_aeteet': safe_round(hist_aeteet['DELTA'].iloc[0]/60, -1) if not hist_aeteet.empty and len(hist_aeteet) > 0 and hist_aeteet['DELTA'].iloc[0] is not None else -1
    } 
    
def calculate_planned_actual_times(flight_row):
    """Calculate actual taxi_out, airborne, and taxi_in times using correct logic, and add them to each flight dict."""
    # Helper to parse datetimes safely
    def parse_dt(val):
        try:
            return pd.to_datetime(val, errors='coerce')
        except Exception:
            return pd.NaT
    
    # Helper to convert time string to seconds
    def parse_time_to_seconds(t):
        if t is None:
            return 0
        if isinstance(t, (int, float)):
            return t
        if isinstance(t, time):
            total_seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            return total_seconds
        # Try to parse as string if it's a string
        if isinstance(t, str):
            try:
                # Try to parse as time string (HH:MM:SS format)
                time_parts = t.split(':')
                if len(time_parts) >= 2:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
                    return hours * 3600 + minutes * 60 + seconds
            except (ValueError, IndexError):
                pass
        return 0
    
    # Parse all relevant datetimes
    offblock = parse_dt(flight_row.get('OFFBLOCK'))
    mvt = parse_dt(flight_row.get('MVT'))
    ata = parse_dt(flight_row.get('ATA'))
    onblock = parse_dt(flight_row.get('ONBLOCK'))
    from_timediff = flight_row.get('FROM_TIMEDIFF')
    to_timediff = flight_row.get('TO_TIMEDIFF')
    std = parse_dt(flight_row.get('STD'))
    
    # Calculate planned times (in seconds)
    planned_taxi_out = parse_time_to_seconds(flight_row.get('TAXI_OUT_TIME')) / 60 if flight_row.get('TAXI_OUT_TIME') is not None else 0
    flight_time = flight_row.get('FLIGHT_TIME') or flight_row.get('TRIP_DURATION')
    planned_airborne = parse_time_to_seconds(flight_time) / 60 if flight_time is not None else 0
    planned_taxi_in = parse_time_to_seconds(flight_row.get('TAXI_IN_TIME')) / 60 if flight_row.get('TAXI_IN_TIME') is not None else 0

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
    
    # Delay departure: STD - MVT (in minutes)
    delay_dep = (std - offblock).total_seconds() / 60 if pd.notnull(std) and pd.notnull(offblock) else None

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
    flight_data['delay_dep'] = delay_dep
    flight_data['actual_delta'] = raw_delta

    return flight_data
