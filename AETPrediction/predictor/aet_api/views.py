from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
from django.utils import timezone
from datetime import datetime, timedelta
import json
import logging
from .model_loader import ModelLoader
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
                logger.error(f"Batch prediction error for flight {flight_id}: {str(e)}")
        
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
    """Query flight data from MySQL database"""
    with connection.cursor() as cursor:
        # Get flight plan data
        cursor.execute("""
            SELECT 
                fp.ID,
                fp.FLP_FILE_NAME,
                fp.CALLSIGN,
                fp.STD,
                fp.DEPARTURE_AIRP,
                fp.ARRIVAL_AIRP,
                fp.AIRCRAFT_ICAO_TYPE,
                fp.TAXI_OUT_TIME,
                fp.FLIGHT_TIME,
                fp.TAXI_IN_TIME
            FROM FP_ARINC633 fp
            WHERE fp.ID = %s
        """, [flight_id])
        
        result = cursor.fetchone()
        if not result:
            return None
        
        # Convert to dictionary
        columns = [col[0] for col in cursor.description]
        flight_data = dict(zip(columns, result))
        
        # Get waypoint data
        cursor.execute("""
            SELECT 
                COUNT(*) as waypoint_count,
                MAX(CUMULATIVE_FLIGHT_TIME) as max_flight_time
            FROM FP_ARINC633_WP
            WHERE FLP_FILE_NAME = %s
        """, [flight_data['FLP_FILE_NAME']])
        
        wp_data = cursor.fetchone()
        if wp_data:
            flight_data['waypoint_count'] = wp_data[0]
            flight_data['max_flight_time'] = wp_data[1] or 0
        
        # Get MEL count
        cursor.execute("""
            SELECT COUNT(*) as mel_count
            FROM FP_ARINC633_MEL
            WHERE FLP_FILE_NAME = %s
        """, [flight_data['FLP_FILE_NAME']])
        
        mel_data = cursor.fetchone()
        if mel_data:
            flight_data['mel_count'] = mel_data[0]
        
        # Get latest ACARS data
        cursor.execute("""
            SELECT 
                AVG(WINDSPEED) as avg_wind_speed,
                MAX(WINDSPEED) as max_wind_speed,
                AVG(TEMPERATURE) as avg_temperature,
                MAX(ALTITUDE) as max_altitude
            FROM TAP_ACARS_PLANE
            WHERE FLIGHT = %s
            AND REPORTTIME >= %s
        """, [flight_data['CALLSIGN'], flight_data['STD']])
        
        acars_data = cursor.fetchone()
        if acars_data and acars_data[0] is not None:
            flight_data['avg_wind_speed'] = acars_data[0]
            flight_data['max_wind_speed'] = acars_data[1]
            flight_data['avg_temperature'] = acars_data[2]
            flight_data['max_altitude'] = acars_data[3]
        else:
            # Default values if no ACARS data
            flight_data['avg_wind_speed'] = 10
            flight_data['max_wind_speed'] = 20
            flight_data['avg_temperature'] = 15
            flight_data['max_altitude'] = 35000
        
        return flight_data

def get_recent_flights(minutes=30):
    """Get flight IDs that departed in the last N minutes"""
    with connection.cursor() as cursor:
        cutoff_time = timezone.now() - timedelta(minutes=minutes)
        
        cursor.execute("""
            SELECT DISTINCT fp.ID
            FROM FP_ARINC633 fp
            JOIN FLT_STATUS_HIST fsh ON fsh.FLT_INFOID = fp.ID
            WHERE fsh.STATUS = 'OFF_BLOCKS'
            AND fsh.STATUS_DATETIME >= %s
            ORDER BY fsh.STATUS_DATETIME DESC
        """, [cutoff_time])
        
        return [row[0] for row in cursor.fetchall()]

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