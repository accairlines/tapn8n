"""
Feature extraction and preprocessing functions
"""

import pandas as pd
import codification

category_cols_all = [
    'OPERATOR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA', 'DIV_IATA', 'FROM_TERMINAL', 'FROM_GATE', 'FROM_STAND', 
    'TO_TERMINAL', 'TO_STAND', 'AC_READY', 'CALL_SIGN', 'SERV_TYP_COD', 'CHG_REASON', 'fp_FLP_FILE_NAME', 'fp_STD', 
    'fp_CALLSIGN', 'fp_CAPTAIN', 'fp_DEPARTURE_AIRP', 'fp_ARRIVAL_AIRP', 'fp_AIRCRAFT_ICAO_TYPE', 'fp_AIRLINE_SPEC',
    'fp_ROUTE_NAME', 'fp_ROUTE_OPTIMIZATION', 'fp_CLIMB_PROC', 'fp_CRUISE_PROC', 'fp_DESCENT_PROC', 'eq_BODYTYPE', 
    'eq_EQUIPTYPE', 'eq_EQUIPTYPE2'
]
category_cols = [
    'OPERATOR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA', 'fp_CRUISE_PROC', 'eq_EQUIPTYPE', 'eq_EQUIPTYPE2'
]

numeric_cols_all = [
    'FLT_NR', 'PAX_BOARDED', 'CARGO', 'CAPACITY', 'fp_PERFORMANCE_FACTOR', 'fp_CLIMB_CI', 'fp_CRUISE_CI', 'fp_DESCENT_CI', 
    'fp_GREAT_CIRC', 'fp_ZERO_FUEL_WEIGHT', 'fp_TAXI_OUT_TIME', 'fp_TAXI_IN_TIME', 'fp_FLIGHT_TIME', 'actual_taxi_out', 
    'actual_airborne', 'actual_taxi_in', 'actual_total_time', 'planned_taxi_out', 'planned_airborne', 'planned_taxi_in', 
    'planned_total_time', 'AET', 'EET'
]
numeric_cols = [
    'FLT_NR', 'fp_CRUISE_CI'
]

waypoints_cols_all = [
    'SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE'
]
waypoints_cols = []

date_cols_all = [
    'STD', 'ETD', 'ATD', 'STA', 'ETA', 'ATA', 'ONBLOCK', 'AC_READY', 'TSAT', 'OFFBLOCK', 'TOBT', 'CTOT', 'MVT'
]
date_cols = [
    'STD'
]

acars_cols_all = ['WINDDIRECTION', 'WINDSPEED']
acars_cols = []
    
def preprocess_flight_data(flights):
    """Preprocess all flight features for model training: encode all columns as category codes, fill missing with -1."""
    # Create DataFrame from flights
    features_data = pd.DataFrame(flights)
    targets_data = pd.DataFrame()
    
    # Verify all category columns exist in category_cols_all
    invalid_cols = [col for col in category_cols if col not in category_cols_all]
    if invalid_cols:
        raise ValueError(f"The following category columns are not in category_cols_all: {invalid_cols}")
    
    base_data = {}
    for col in category_cols:
        if col not in features_data.columns:
            base_data[col + '_code'] = -1
        else:
            base_data[col + '_code'] = codification.categorialcoding(features_data[col])
    
    # Verify all numeric columns exist in numeric_cols_all
    invalid_cols = [col for col in numeric_cols if col not in numeric_cols_all]
    if invalid_cols:
        raise ValueError(f"The following numeric columns are not in numeric_cols_all: {invalid_cols}")
    for col in numeric_cols:
        if col not in features_data.columns:
            base_data[col] = -1
        else:
            # Convert to numeric, handling NaN values
            numeric_series = pd.to_numeric(features_data[col], errors='coerce')
            base_data[col] = numeric_series.fillna(-1).astype(int)
        
    # Verify all date columns exist in date_cols_all
    invalid_cols = [col for col in date_cols if col not in date_cols_all]
    if invalid_cols:
        raise ValueError(f"The following date columns are not in date_cols_all: {invalid_cols}")
    for col in date_cols:
        if col not in features_data.columns:
            base_data[col] = -1
        else:
            # For date columns, convert to numeric or keep as is
            date_series = features_data[col]
            # Remove the index name to avoid duplicate column issues
            if hasattr(date_series, 'name'):
                date_series.name = None
            # Convert to datetime
            date = pd.to_datetime(features_data[col], errors='coerce')
            # Extract month and year, filling NaN with -1
            base_data[col + '_month'] = date.dt.month.fillna(-1).astype(int)
            base_data[col + '_year'] = date.dt.year.fillna(-1).astype(int)
            base_data[col + '_hour'] = date.dt.hour.fillna(-1).astype(int)
            base_data[col + '_minute'] = date.dt.minute.fillna(-1).astype(int)
            base_data[col + '_second'] = date.dt.second.fillna(-1).astype(int)
            base_data[col + '_weekday'] = date.dt.weekday.fillna(-1).astype(int)
            base_data[col + '_day'] = date.dt.day.fillna(-1).astype(int)
            base_data[col + '_dayofyear'] = date.dt.dayofyear.fillna(-1).astype(int)
            base_data[col + '_dayofweek'] = date.dt.dayofweek.fillna(-1).astype(int)
    flight_data_df = pd.DataFrame(base_data, index=range(len(flights)))
        
    # Verify all waypoints columns exist in waypoints_cols_all
    invalid_cols = [col for col in waypoints_cols if col not in waypoints_cols_all]
    if invalid_cols:
        raise ValueError(f"The following waypoints columns are not in waypoints_cols_all: {invalid_cols}")
    waypoint_data = {}
    for base in waypoints_cols:
        for i in range(1, 51):
            col = f'wp{i}_{base}'
            if col not in features_data.columns:
                waypoint_data[col] = -1
            else:
                # Convert to numeric, handling NaN values
                numeric_series = pd.to_numeric(features_data[col], errors='coerce')
                waypoint_data[col] = numeric_series.fillna(-1).astype(int)
    waypoint_df = pd.DataFrame(waypoint_data, index=range(len(flights)))
    flight_data_df = pd.concat([flight_data_df, waypoint_df], axis=1)
    
    
    # Verify all acars columns exist in acars_cols_all
    invalid_cols = [col for col in acars_cols if col not in acars_cols_all]
    if invalid_cols:
        raise ValueError(f"The following acars columns are not in acars_cols_all: {invalid_cols}")
    acars_data = {}
    for base in acars_cols:
        for i in range(1, 21):
            col = f'ac{i}_{base}'
            if col not in features_data.columns:
                acars_data[col] = -1
            else:
                # Convert to numeric, handling NaN values
                numeric_series = pd.to_numeric(features_data[col], errors='coerce')
                acars_data[col] = numeric_series.fillna(-1).astype(int)
                
    acars_df = pd.DataFrame(acars_data, index=range(len(flights)))
    features_data = pd.concat([flight_data_df, acars_df], axis=1)
    
    # Create targets dataframe from flight dict array
    targets_data = pd.DataFrame([{'delta': flight.get('delta')} for flight in flights])
        
    # Fill missing values with -1
    features_data = features_data.fillna(-1)
    targets_data = targets_data.fillna(-1)
    return features_data, targets_data 