"""
Feature extraction and preprocessing functions
"""

import pandas as pd

def preprocess_flight_data(flights):
    """Preprocess all flight features for model training: encode all columns as category codes, fill missing with -1."""
    # Create DataFrame from flights
    features_data = pd.DataFrame(flights)
    targets_data = pd.DataFrame()

    # Ensure all columns are present
    category_cols = [
        'OPERATOR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA', 'DIV_IATA', 'FROM_TERMINAL', 'FROM_GATE', 'FROM_STAND', 
        'TO_TERMINAL', 'TO_STAND', 'AC_READY', 'CALL_SIGN', 'SERV_TYP_COD', 'CHG_REASON', 'fp_FLP_FILE_NAME', 'fp_STD', 
        'fp_CALLSIGN', 'fp_CAPTAIN', 'fp_DEPARTURE_AIRP', 'fp_ARRIVAL_AIRP', 'fp_AIRCRAFT_ICAO_TYPE', 'fp_AIRLINE_SPEC',
        'fp_ROUTE_NAME', 'fp_ROUTE_OPTIMIZATION', 'fp_CLIMB_PROC', 'fp_CRUISE_PROC', 'fp_DESCENT_PROC' 'eq_BODYTYPE', 
        'eq_EQUIPTYPE', 'eq_EQUIPTYPE2'
    ]
    base_data = {}
    for col in category_cols:
        if col not in features_data.columns:
            base_data[col + '_code'] = -1
        else:
            base_data[col + '_code'] = codification.categorialcoding(features_data[col])
    
    numeric_cols = [
        'FLT_NR', 'PAX_BOARDED', 'CARGO', 'CAPACITY', 'fp_PERFORMANCE_FACTOR', 'fp_CLIMB_CI', 'fp_CRUISE_CI', 'fp_DESCENT_CI', 
        'fp_GREAT_CIRC', 'fp_ZERO_FUEL_WEIGHT', 'fp_TAXI_OUT_TIME', 'fp_TAXI_IN_TIME', 'fp_FLIGHT_TIME'
    ]
    for col in numeric_cols:
        if col not in features_data.columns:
            base_data[col + '_code'] = -1
        else:
            # Convert to numeric, handling NaN values
            numeric_series = pd.to_numeric(features_data[col], errors='coerce')
            base_data[col + '_code'] = numeric_series.fillna(-1).astype(int)
    
    date_cols = [
        'STD', 'ETD', 'ATD', 'STA', 'ETA', 'ATA', 'ONBLOCK', 'AC_READY', 'TSAT', 'OFFBLOCK', 'TOBT', 'CTOT', 'MVT'
    ]
    for col in date_cols:
        if col not in features_data.columns:
            base_data[col + '_code'] = -1
        else:
            # For date columns, convert to numeric or keep as is
            base_data[col + '_code'] = features_data[col]
    flight_data_df = pd.DataFrame(base_data, index=range(len(flights)))
    
    # Waypoint columns: wp1_... to wp50_... for each in waypoints_cols
    waypoints_cols = ['SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE']
    waypoint_data = {}
    for base in waypoints_cols:
        for i in range(1, 51):
            col = f'wp{i}_{base}'
            if col not in features_data.columns:
                waypoint_data[col + '_code'] = -1
            else:
                # Convert to numeric, handling NaN values
                numeric_series = pd.to_numeric(features_data[col], errors='coerce')
                waypoint_data[col + '_code'] = numeric_series.fillna(-1).astype(int)
    waypoint_df = pd.DataFrame(waypoint_data, index=range(len(flights)))
    flight_data_df = pd.concat([flight_data_df, waypoint_df], axis=1)
    
    # Acars columns: ac1_... to ac20_... for each in acars_cols
    acars_cols = ['WINDDIRECTION', 'WINDSPEED']
    acars_data = {}
    for base in acars_cols:
        for i in range(1, 21):
            col = f'ac{i}_{base}'
            if col not in features_data.columns:
                acars_data[col + '_code'] = -1
            else:
                # Convert to numeric, handling NaN values
                numeric_series = pd.to_numeric(features_data[col], errors='coerce')
                acars_data[col + '_code'] = numeric_series.fillna(-1).astype(int)
                
    acars_df = pd.DataFrame(acars_data, index=range(len(flights)))
    features_data = pd.concat([flight_data_df, acars_df], axis=1)
    
    # Create targets dataframe from flight dict array
    targets_data = pd.DataFrame([{'delta': flight.get('delta')} for flight in flights])
        
    # Fill missing values with -1
    features_data = features_data.fillna(-1)
    targets_data = targets_data.fillna(-1)
    return features_data, targets_data 