"""
Feature extraction and preprocessing functions
"""

import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_flight_data(flights):
    """Preprocess all flight features for model training: encode all columns as category codes, fill missing with -1."""
    # Create DataFrame from flights
    features_data = pd.DataFrame(flights)
    targets_data = pd.DataFrame()

    # Ensure all columns are present
    all_cols = [
        'OFFBLOCK', 'MVT', 'ATA', 'ONBLOCK', 'ETA',
        'FROM_IATA', 'STD', 'CALL_SIGN', 'AC_REGISTRATION',
        'OPERATOR', 'FLT_NR', 'TO_IATA', 'ETD', 'ATD', 'STA', 'FROM_STAND', 'TO_STAND',
        'AC_READY', 'TSAT', 'TOBT', 'CTOT', 'SERV_TYP_COD',
        'FROM_TERMINAL', 'TO_TERMINAL', 'FROM_GATE', 'CAPTAIN', 'AIRCRAFT_ICAO_TYPE',
        'AIRLINE_SPEC', 'PERFORMANCE_FACTOR', 'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CRUISE_CI', 'CLIMB_PROC',
        'CRUISE_PROC', 'DESCENT_PRO', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT',
        'BODYTYPE', 'EQUIPTYPE', 'EQUIPTYPE2'
    ]
    for col in all_cols:
        if col not in features_data.columns:
            features_data[col] = None

    # Encode all columns as category codes
    for col in all_cols:
        features_data[col + '_code'] = features_data[col].astype('category').cat.codes

    # Waypoint columns: wp1_... to wp50_... for each in waypoints_cols
    waypoints_cols = ['SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE']
    waypoint_data = {}
    for base in waypoints_cols:
        for i in range(1, 51):
            col = f'wp{i}_{base}'
            if col not in features_data.columns:
                waypoint_data[col] = None
            # Use a Series to allow .astype and .cat.codes
            waypoint_data[col + '_code'] = pd.Series(waypoint_data.get(col, None)).astype('category').cat.codes
    waypoint_df = pd.DataFrame(waypoint_data)
    features_data = pd.concat([features_data, waypoint_df], axis=1)
    
    # Acars columns: ac1_... to ac20_... for each in acars_cols
    acars_cols = ['WINDDIRECTION', 'WINDSPEED']
    acars_data = {}
    for base in acars_cols:
        for i in range(1, 21):
            col = f'ac{i}_{base}'
            if col not in features_data.columns:
                acars_data[col] = None
            acars_data[col + '_code'] = pd.Series(acars_data.get(col, None)).astype('category').cat.codes
    acars_df = pd.DataFrame(acars_data)
    features_data = pd.concat([features_data, acars_df], axis=1)
    # Drop actual_delta from features since it's the target
    features_data = features_data.drop('delta', axis=1, errors='ignore')
    # Create targets dataframe from flight dict array
    targets_data = pd.DataFrame([{'delta': flight.get('delta')} for flight in flights])
        
    # Fill missing values with -1
    features_data = features_data.fillna(-1)
    targets_data = targets_data.fillna(-1)
    return features_data, targets_data 