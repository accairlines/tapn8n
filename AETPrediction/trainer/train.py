#!/usr/bin/env python3
"""
Daily model training script for AET prediction
Reads from CSV files and trains XGBoost model
"""

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from preprocess import preprocess_flight_data
import glob
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set base path from environment or default
DATA_PATH = os.environ.get('AET_DATA_PATH')
LOG_PATH = os.environ.get('AET_LOG_PATH')
MODEL_PATH = os.environ.get('AET_MODEL_PATH')

# Ensure log directory exists
os.makedirs(LOG_PATH, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH, 'training.log')),
        logging.StreamHandler()
    ]
)

def load_data(dynamic_data_path=None):
    """Load CSV data files as lists of dicts (supporting multiple files per type)
    
    Args:
        dynamic_data_path: Path for dynamic CSV files (flight_*.csv, fp_*.csv, acars_*.csv)
                          If None, uses global DATA_PATH for all files
    """
    logging.info("Loading CSV data files as lists of dicts...")
    
    # Use dynamic_data_path for dynamic files, original DATA_PATH for static files
    data_path_for_dynamic = dynamic_data_path if dynamic_data_path else DATA_PATH
    data_path_for_static = DATA_PATH  # Always use original DATA_PATH for static files

    # Define required columns for each file type
    flights_cols = [
        'OFFBLOCK', 'MVT', 'ATA', 'ONBLOCK', 'ETA',
        'FROM_IATA', 'STD', 'CALL_SIGN', 'AC_REGISTRATION',
        'OPERATOR', 'FLT_NR', 'TO_IATA', 'ETD', 'ATD', 'STA', 'FROM_STAND', 'TO_STAND',
        'AC_READY', 'TSAT', 'TOBT', 'CTOT', 'SERV_TYP_COD',
        'FROM_TERMINAL', 'TO_TERMINAL', 'FROM_GATE'
    ]
    flight_plan_cols = [
        'CALLSIGN', 'DEPARTURE_AIRP', 'STD', 'TS', 'FLP_FILE_NAME',
        'CAPTAIN', 'AIRCRAFT_ICAO_TYPE', 'AIRLINE_SPEC', 'PERFORMANCE_FACTOR',
        'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CRUISE_CI', 'CLIMB_PROC',
        'CRUISE_PROC', 'DESCENT_PRO', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT',
        'TRIP_DURATION', 'TAXI_OUT_TIME', 'FLIGHT_TIME', 'TAXI_IN_TIME', 
        'ARRIVAL_AIRP'
    ]
    waypoints_cols = [
        'ALTITUDE', 'SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE',
        'FLP_FILE_NAME', 'CUMULATIVE_FLIGHT_TIME'
    ]
    mel_cols = ['FLP_FILE_NAME']
    acars_cols = ['FLIGHT', 'REPORTTIME', 'WINDDIRECTION', 'WINDSPEED']
    equipments_cols = ['ID', 'BODYTYPE', 'EQUIPTYPE', 'EQUIPTYPE2']
    aircrafts_cols = ['ACREGISTRATION', 'EQUIPTYPEID']
    stations_cols = ['STATION', 'TIMEDIFF_MINUTES', 'DAY_NUM']
    stations_cols = ['STATION', 'TIMEDIFF_MINUTES', 'DAY_NUM']

    def read_multi_csv_to_dicts(pattern, usecols=None):
        files = glob.glob(pattern)
        logging.warning(f"File: {files}, starting to read")
        if not files:
            logging.warning(f"No files found for pattern: {pattern}, returning empty list")
            return []
        df_list = []
        for f in files:
            df = pd.read_csv(f, encoding='latin1', low_memory=False)
            if usecols is not None:
                missing_cols = [col for col in usecols if col not in df.columns]
                for col in missing_cols:
                    df[col] = None
                df = df.reindex(columns=usecols)
            df_list.append(df)
        df_list = [df for df in df_list if not df.empty and not all(df.isna().all())]
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            # handle the case where all files are empty
            df = pd.DataFrame(columns=usecols if usecols else [])
        logging.warning(f"File: {files}, read successfully, returning {len(df)} records")
        return df.to_dict('records')

    def read_single_csv_to_dicts(file_path, usecols=None):
        """Read a single CSV file and return as list of dicts, return empty list if file doesn't exist"""
        logging.warning(f"File: {file_path}, starting to read")
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}, returning empty list")
            return []
        try:
            df = pd.read_csv(file_path, usecols=usecols, low_memory=False)
            
            logging.warning(f"File: {file_path}, read successfully, returning {len(df)} records")
            return df.to_dict('records')
        except Exception as e:
            logging.warning(f"Error reading {file_path}: {str(e)}, returning empty list")
            return []

    def read_acars_files():
        """Read ACARS files and return as list of dicts, return empty list if no files found"""
        acars_files = glob.glob(os.path.join(data_path_for_dynamic, 'acars_*.csv'))
        logging.warning(f"File: {acars_files}, starting to read")
        if not acars_files:
            logging.warning("No ACARS files found, returning empty list")
            return []
        
        try:
            acars_df = pd.concat([
                pd.read_csv(f, usecols=acars_cols, low_memory=False) for f in acars_files
            ], ignore_index=True)
            acars_df['CALLSIGN'] = acars_df['FLIGHT'].str.replace('TP', 'TAP')
            logging.warning(f"File: {acars_files}, read successfully, returning {len(acars_df)} records")
            return acars_df.to_dict('records')
        except Exception as e:
            logging.warning(f"Error reading ACARS files: {str(e)}, returning empty list")
            return []

    # Load main tables (supporting multiple files) - use dynamic path
    flights = read_multi_csv_to_dicts(os.path.join(data_path_for_dynamic, 'flight_*.csv'), usecols=flights_cols)
    flight_plan = read_multi_csv_to_dicts(os.path.join(data_path_for_dynamic, 'fp_arinc633_fp_*.csv'), usecols=flight_plan_cols)
    waypoints = read_multi_csv_to_dicts(os.path.join(data_path_for_dynamic, 'fp_arinc633_wp_*.csv'), usecols=waypoints_cols)
    mel = read_multi_csv_to_dicts(os.path.join(data_path_for_dynamic, 'fp_arinc633_mel_*.csv'), usecols=mel_cols)
    
    # Load ACARS data and replace TP with TAP in FLIGHT column
    acars = read_acars_files()
        
    # Load base tables - use static path (original DATA_PATH)
    equipments = read_single_csv_to_dicts(os.path.join(data_path_for_static, 'equipments.csv'), usecols=equipments_cols)
    aircrafts = read_single_csv_to_dicts(os.path.join(data_path_for_static, 'aircrafts.csv'), usecols=aircrafts_cols)
    stations = read_single_csv_to_dicts(os.path.join(data_path_for_static, 'stations_utc.csv'), usecols=stations_cols)

    logging.info(f"Loaded {len(flights)} flights (as dicts)")

    return flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations


def calculate_planned_actual_times(flights):
    """Calculate actual taxi_out, airborne, and taxi_in times using correct logic, and add them to each flight dict."""
    logging.info("Calculating actual elapsed times (dict version)...")

    # Helper to parse datetimes safely
    def parse_dt(val):
        try:
            return pd.to_datetime(val, errors='coerce')
        except Exception:
            return pd.NaT

    for flight in flights:
        # Parse all relevant datetimes
        offblock = parse_dt(flight.get('OFFBLOCK'))
        mvt = parse_dt(flight.get('MVT'))
        ata = parse_dt(flight.get('ATA'))
        onblock = parse_dt(flight.get('ONBLOCK'))
        eta = parse_dt(flight.get('ETA'))
        
        
        # Planned Taxi out
        taxi_out_str = flight.get('TAXI_OUT_TIME', '00:00:00')
        h, m, s = map(int, taxi_out_str.split(':'))
        planned_taxi_out = h * 3600 + m * 60 + s if taxi_out_str else 0
        # Planned Airborne
        flight_time = flight.get('FLIGHT_TIME') or flight.get('TRIP_DURATION', '00:00:00')
        h, m, s = map(int, flight_time.split(':'))
        planned_airborne = h * 3600 + m * 60 + s if flight_time else 0
        # Planned Taxi in  
        taxi_in_str = flight.get('TAXI_IN_TIME', '00:00:00')
        h, m, s = map(int, taxi_in_str.split(':'))
        planned_taxi_in = h * 3600 + m * 60 + s if taxi_in_str else 0
        
        
        # Planned Taxi out
        taxi_out_str = flight.get('TAXI_OUT_TIME', '00:00:00')
        h, m, s = map(int, taxi_out_str.split(':'))
        planned_taxi_out = h * 3600 + m * 60 + s if taxi_out_str else 0
        # Planned Airborne
        flight_time = flight.get('FLIGHT_TIME') or flight.get('TRIP_DURATION', '00:00:00')
        h, m, s = map(int, flight_time.split(':'))
        planned_airborne = h * 3600 + m * 60 + s if flight_time else 0
        # Planned Taxi in  
        taxi_in_str = flight.get('TAXI_IN_TIME', '00:00:00')
        h, m, s = map(int, taxi_in_str.split(':'))
        planned_taxi_in = h * 3600 + m * 60 + s if taxi_in_str else 0

        # Taxi out: MVT - OFFBLOCK
        actual_taxi_out = (mvt - offblock).total_seconds() / 60 if pd.notnull(mvt) and pd.notnull(offblock) else None
        # Airborne: ATA - MVT
        actual_airborne = (ata - mvt).total_seconds() / 60 if pd.notnull(ata) and pd.notnull(mvt) else None
        # Taxi in: ONBLOCK - ATA
        actual_taxi_in = (onblock - ata).total_seconds() / 60 if pd.notnull(onblock) and pd.notnull(ata) else None
        # Total AET
        actual_total_time = sum(
            x for x in [actual_taxi_out, actual_airborne, actual_taxi_in] if x is not None
        )
        # AET: ATA - MVT
        aet = (ata - mvt).total_seconds() / 60 if pd.notnull(ata) and pd.notnull(mvt) else None
        # EET: ETA - MVT
        eet = (planned_taxi_out + planned_airborne + planned_taxi_in) / 60
        eet = (planned_taxi_out + planned_airborne + planned_taxi_in) / 60
        # delta: AET - EET
        actual_delta = (aet - eet) if aet is not None and eet is not None else None

        # Add to dict
        flight['actual_taxi_out'] = actual_taxi_out
        flight['actual_airborne'] = actual_airborne
        flight['actual_taxi_in'] = actual_taxi_in
        flight['actual_total_time'] = actual_total_time
        flight['planned_taxi_out'] = planned_taxi_out
        flight['planned_airborne'] = planned_airborne
        flight['planned_taxi_in'] = planned_taxi_in
        flight['planned_total_time'] = planned_taxi_out + planned_airborne + planned_taxi_in
        flight['planned_taxi_out'] = planned_taxi_out
        flight['planned_airborne'] = planned_airborne
        flight['planned_taxi_in'] = planned_taxi_in
        flight['planned_total_time'] = planned_taxi_out + planned_airborne + planned_taxi_in
        flight['AET'] = aet
        flight['EET'] = eet
        flight['delta'] = actual_delta
        flight['delta'] = actual_delta

    return flights


def prepare_training_data(flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations):
    """Prepare features and targets for training using list-of-dicts and manual merging."""
    logging.info("Preparing training data (dict version)...")

    # Helper to parse datetimes safely
    def parse_dt(val):
        try:
            return pd.to_datetime(val, errors='coerce')
        except Exception:
            return pd.NaT

    # --- 1. Build station TIMEDIFF map ---
    logging.debug(f"Stations: {len(stations)}")
    station_timediff = {(str(s.get('STATION', '')).strip(), s.get('DAY_NUM')): s.get('TIMEDIFF_MINUTES', 0) for s in stations}
    station_timediff = {(str(s.get('STATION', '')).strip(), s.get('DAY_NUM')): s.get('TIMEDIFF_MINUTES', 0) for s in stations}

    # --- 2. Preprocess flight_plan: keep only latest TS for each (CALLSIGN, DEPARTURE_AIRP, STD) ---
    flight_plan_latest = {}
    logging.debug(f"FlightPlans: {len(flight_plan)}")
    for fp in flight_plan:
        key = (
            str(fp.get('CALLSIGN', '')).strip(),
            str(fp.get('DEPARTURE_AIRP', '')).strip(),
            str(parse_dt(fp.get('STD')))
        )
        ts = parse_dt(fp.get('TS'))
        if key not in flight_plan_latest or (ts and ts > parse_dt(flight_plan_latest[key].get('TS'))):
            flight_plan_latest[key] = fp
    # Now flight_plan_latest is a dict of the latest flight_plan per key

    # --- 3. Merge flight_plan into flights using UTC STD ---
    logging.debug(f"Flights: {len(flights)}")
    for flight in flights:
        # Get TIMEDIFF for this flight's FROM_IATA
        from_iata = str(flight.get('FROM_IATA', '')).strip()
        # Convert STD (local) to UTC
        std_local = parse_dt(flight.get('STD'))
        timediff = station_timediff.get((from_iata, std_local.dayofyear if pd.notnull(std_local) else None), 0)
        timediff = station_timediff.get((from_iata, std_local.dayofyear if pd.notnull(std_local) else None), 0)
        std_utc = std_local - pd.to_timedelta(timediff, unit='m') if pd.notnull(std_local) else pd.NaT
        # Build key for matching
        key = (
            str(flight.get('CALL_SIGN', '')).strip(),
            from_iata,
            str(std_utc)
        )
        flight['STD_UTC'] = std_utc
        flight['flight_plan'] = flight_plan_latest.get(key)

    # --- 4. Merge aircraft into flights ---
    # Build index for aircrafts by ACREGISTRATION
    logging.debug(f"Aircrafts: {len(aircrafts)}")
    aircraft_index = {str(a.get('ACREGISTRATION', '')).strip(): a for a in aircrafts}
    for flight in flights:
        ac_reg = str(flight.get('AC_REGISTRATION', '')).strip()
        flight['aircraft'] = aircraft_index.get(ac_reg)

    # --- 5. Merge equipment into flights ---
    # Build index for equipments by ID
    logging.debug(f"Equipments: {len(equipments)}")
    equipment_index = {str(e.get('ID', '')).strip(): e for e in equipments}
    for flight in flights:
        equiptype_id = None
        if flight.get('aircraft'):
            equiptype_id = str(flight['aircraft'].get('EQUIPTYPEID', '')).strip()
        flight['equipment'] = equipment_index.get(equiptype_id)

    # --- 6. Merge mel and waypoints into flights by flight_plan FLP_FILE_NAME ---
    # Build index for mel and waypoints by FLP_FILE_NAME
    mel_index = {}
    logging.debug(f"MELs: {len(mel)}")
    logging.debug(f"WPs: {len(waypoints)}")
    for m in mel:
        fname = str(m.get('FLP_FILE_NAME', '')).strip()
        mel_index.setdefault(fname, []).append(m)
    waypoints_index = {}
    for wp in waypoints:
        fname = str(wp.get('FLP_FILE_NAME', '')).strip()
        waypoints_index.setdefault(fname, []).append(wp)
    for flight in flights:
        file_name = None
        if flight.get('flight_plan'):
            file_name = str(flight['flight_plan'].get('FLP_FILE_NAME', '')).strip()
        flight['mel'] = mel_index.get(file_name, [])
        flight['waypoints'] = waypoints_index.get(file_name, [])
        logging.debug(f"Flight: {i}, flight_plan: {file_name}, of {len(flights)} flights, {len(flight['mel'])} mel and {len(flight['waypoints'])} waypoints")

    # --- 7. Merge acars into flights by CALLSIGN/FLIGHT and REPORTTIME window ---
    logging.debug(f"ACARSs: {len(acars)}")
    i = 0
    for flight in flights:
        flight_cs = str(flight.get('CALL_SIGN', '')).strip().replace('TAP', 'TP')
        flt_nr = str(flight.get('FLT_NR', '')).strip()
        flight_fid = 'TP' + flt_nr.zfill(4)
        flight_id = 'TP' + flt_nr
        std_utc = flight.get('STD_UTC')
        std_utc_end = std_utc + pd.Timedelta(hours=12) if pd.notnull(std_utc) else None
        acars_matches = []
        logging.debug(f"Flight: {i}, flight_id: {flight_id}, std_utc: {std_utc}, std_utc_end: {std_utc_end}, of {len(flights)} flights, {len(acars)} acars")
        
        for a in acars:
            if a.get('FLIGHT') == flight_id or a.get('FLIGHT') == flight_fid or a.get('FLIGHT') == flight_cs:
                report_time = pd.to_datetime(a.get('REPORTTIME'), errors='coerce')
                if (
                    pd.notnull(report_time) and
                    pd.notnull(std_utc) and
                    std_utc <= report_time <= std_utc_end
                ):
                    acars_matches.append(a)
                    acars.remove(a)
        flight['acars'] = acars_matches
        i += 1

    # All merges done
    return flights


def train_models(features, targets):
    """Train separate XGBoost models for each time component"""
    logging.info("Training models...")

    # Ensure only numeric columns are used
    features_numeric = features.select_dtypes(include=[np.number])

    # Ensure only numeric columns are used
    features_numeric = features.select_dtypes(include=[np.number])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_numeric, targets, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    metrics = {}
    
    # Train model for each target
    for target in ['delta']:
        logging.info(f"Training model for {target}...")
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train[target])
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)
        
        models[target] = model
        metrics[target] = {'mae': mae, 'r2': r2}
        
        logging.info(f"{target} - MAE: {mae:.2f} minutes, R2: {r2:.3f}")
    
    return models, scaler, metrics


def save_models(models, scaler, metrics):
    """Save trained models and scaler"""
    logging.info("Saving models...")
    
    model_data = {
        'models': models,
        'scaler': scaler,
        'metrics': metrics,
        'training_date': datetime.now().isoformat(),
        'feature_names': scaler.feature_names_in_.tolist()
    }
    
    model_file = os.path.join(MODEL_PATH, 'model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    logging.info("Models saved successfully")

def extract_targetsfeatures_from_flights(flights):
    """Extract features DataFrame from enriched flights list, flattening selected flight, selected flight_plan fields, selected equipment fields, up to 50 relevant waypoint features, and up to 20 acars features."""
    # Now build features DataFrame
    data = []
    flight_fields = [
        'OPERATOR', 'FLT_NR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA',
        'STD', 'ETD', 'ATD', 'STA', 'ETA', 'FROM_STAND', 'TO_STAND',
        'AC_READY', 'TSAT', 'TOBT', 'CTOT', 'CALL_ SIGN', 'SERV_TYP_COD', 'MVT'
    ]
    flight_plan_fields = [
        'CAPTAIN', 'AIRCRAFT_ICAO_TYPE', 'AIRLINE_SPEC', 'PERFORMANCE_FACTOR',
        'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CRUISE_CI', 'CLIMB_PROC',
        'CRUISE_PROC', 'DESCENT_PRO', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT'
    ]
    """Extract targets DataFrame from enriched flights list."""
    for flight in flights:
        if not flight.get('flight_plan'):
            continue  # Only include if flight_plan is present
        row = {}
        # Add only selected flight fields
        for k in flight_fields:
            v = flight.get(k)
            if not isinstance(v, (dict, list)):
                row[k] = v
        # Add only selected flight_plan fields, prefixed with 'fp_'
        for k in flight_plan_fields:
            v = flight['flight_plan'].get(k)
            if not isinstance(v, (dict, list)):
                row[f'fp_{k}'] = v
        # Add selected equipment fields, prefixed with 'eq_'
        eq = flight.get('equipment')
        if eq:
            for ek in ['BODYTYPE', 'EQUIPMENTTYPE', 'EQUIPMENTTYPE2']:
                row[f'eq_{ek}'] = eq.get(ek)
        # Add up to 50 waypoints with ALTITUDE > 299, extracting 3 fields
        wp_fields = ['SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE']
        waypoints = [wp for wp in (flight.get('waypoints') or []) if wp.get('ALTITUDE') is not None and float(wp.get('ALTITUDE', 0)) > 299]
        for i in range(50):
            if i < len(waypoints):
                wp = waypoints[i]
                for f in wp_fields:
                    row[f'wp{i+1}_{f}'] = wp.get(f)
            else:
                for f in wp_fields:
                    row[f'wp{i+1}_{f}'] = None
        # Add up to 20 acars, extracting WINDDIRECTION and WINDSPEED
        acars_fields = ['WINDDIRECTION', 'WINDSPEED']
        acars_list = flight.get('acars') or []
        for i in range(20):
            if i < len(acars_list):
                ac = acars_list[i]
                for f in acars_fields:
                    row[f'acars{i+1}_{f}'] = ac.get(f)
            else:
                for f in acars_fields:
                    row[f'acars{i+1}_{f}'] = None
        row['actual_taxi_out'] = flight.get('actual_taxi_out')
        row['actual_airborne'] = flight.get('actual_airborne')
        row['actual_taxi_in'] = flight.get('actual_taxi_in')
        row['AET'] = flight.get('AET')
        row['delta'] = flight.get('delta')
        row['delta'] = flight.get('delta')
        data.append(row)
    feactures_processed, targets_processed = preprocess_flight_data(data)
    return feactures_processed, targets_processed


def save_features_targets_to_csv(features, targets):
    """Save features and targets to CSV files for caching"""
    logging.info("Saving features and targets to CSV files for caching...")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(DATA_PATH, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save features to a clean file
    features_path = os.path.join(cache_dir, 'features.csv')
    if os.path.exists(features_path):
        os.remove(features_path)  # Remove existing file first
    features.to_csv(features_path, index=False)
    logging.info(f"Features saved to: {features_path}")
    
    # Save targets
    targets_path = os.path.join(cache_dir, 'targets.csv')
    if os.path.exists(targets_path):
        os.remove(targets_path)  # Remove existing file first
    targets.to_csv(targets_path, index=False)
    logging.info(f"Targets saved to: {targets_path}")
    
    # Save metadata with timestamp
    metadata = {
        'features_shape': features.shape,
        'targets_shape': targets.shape,
        'saved_at': datetime.now().isoformat(),
        'feature_columns': features.columns.tolist(),
        'target_columns': targets.columns.tolist()
    }
    
    import json
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        os.remove(metadata_path)  # Remove existing file first
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Metadata saved to: {metadata_path}")


def rename_csv_files_to_done(dynamic_data_path=None):
    """Rename all CSV files in the data directory to .done extension"""
    logging.info("Renaming all CSV files to .done extension...")
    
    # Use provided dynamic_data_path or fall back to DATA_PATH
    data_path_for_renaming = dynamic_data_path if dynamic_data_path else DATA_PATH
    
    # Get all CSV files in the data directory
    csv_patterns = [
        os.path.join(data_path_for_renaming, 'flight_*.csv'),
        os.path.join(data_path_for_renaming, 'fp_arinc633_fp_*.csv'),
        os.path.join(data_path_for_renaming, 'fp_arinc633_wp_*.csv'),
        os.path.join(data_path_for_renaming, 'fp_arinc633_mel_*.csv'),
        os.path.join(data_path_for_renaming, 'acars_*.csv')
    ]
    
    renamed_count = 0
    for pattern in csv_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                # Create new filename with .done extension
                directory = os.path.dirname(file_path)
                filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(filename)[0]
                new_filename = f"{name_without_ext}.done"
                new_file_path = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(file_path, new_file_path)
                logging.info(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
                
            except Exception as e:
                logging.error(f"Failed to rename {file_path}: {str(e)}")
    
    logging.info(f"Successfully renamed {renamed_count} CSV files to .done extension")


def load_features_targets_from_csv():
    """Load features and targets from CSV files if they exist and are not empty"""
    cache_dir = os.path.join(DATA_PATH, 'cache')
    features_path = os.path.join(cache_dir, 'features.csv')
    targets_path = os.path.join(cache_dir, 'targets.csv')
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    
    # Check if both files exist and are not empty
    def is_file_valid(file_path):
        if not os.path.exists(file_path):
            return False
        # Check if file is empty (size == 0) or contains only whitespace/headers
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                return len(content) > 0
        except Exception:
            return False
    
    if is_file_valid(features_path) and is_file_valid(targets_path):
        logging.info("Loading features and targets from cached CSV files...")
        
        try:
            features = pd.read_csv(features_path)
            targets = pd.read_csv(targets_path)
            
            # Check if DataFrames are empty
            if features.empty or targets.empty:
                logging.info("Cached CSV files are empty, will process new data")
                return None, None
            
            # Load and log metadata
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logging.info(f"Loaded cached data - Features: {metadata['features_shape']}, Targets: {metadata['targets_shape']}")
                logging.info(f"Cached data saved at: {metadata['saved_at']}")
            
            return features, targets
            
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logging.warning(f"Error reading cached CSV files: {e}")
            logging.info("Will process new data instead")
            return None, None
        except Exception as e:
            logging.warning(f"Unexpected error reading cached CSV files: {e}")
            logging.info("Will process new data instead")
            return None, None
    else:
        logging.info("No valid cached CSV files found")
        return None, None


def main():
    """Main training pipeline - processes all folders in DATA_PATH"""
    logging.info("=== Starting daily training for all folders ===")
    
    # Get all folders in DATA_PATH
    if not os.path.exists(DATA_PATH):
        logging.error(f"DATA_PATH does not exist: {DATA_PATH}")
        return
    
    folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    logging.info(f"Found {len(folders)} folders in DATA_PATH: {folders}")
    
    processed_folders = []
    
    for folder in folders:
        folder_path = os.path.join(DATA_PATH, folder)
        logging.info(f"=== Processing folder: {folder} ===")
        
        # Check if folder has CSV files
        csv_files = []
        csv_patterns = [
            os.path.join(folder_path, 'flight_*.csv'),
            os.path.join(folder_path, 'fp_arinc633_fp_*.csv'),
            os.path.join(folder_path, 'fp_arinc633_wp_*.csv'),
            os.path.join(folder_path, 'fp_arinc633_mel_*.csv'),
            os.path.join(folder_path, 'acars_*.csv')
        ]
        
        for pattern in csv_patterns:
            csv_files.extend(glob.glob(pattern))
        
        if not csv_files:
            logging.info(f"Folder {folder} has no CSV files, assuming already processed, skipping...")
            continue
        
        logging.info(f"Folder {folder} has {len(csv_files)} CSV files, processing...")
        
        try:
            # Store original DATA_PATH for static files and cache
            original_data_path = DATA_PATH
            
            # Create a separate variable for current folder's data path
            current_folder_data_path = folder_path
            processed_folders.append(current_folder_data_path)
            
            # LOG_PATH and MODEL_PATH remain unchanged (global)
            # Only DATA_PATH will be temporarily changed for dynamic CSV files
            
            # Load cached data if it exists for this folder
            cached_features, cached_targets = load_features_targets_from_csv()
            
            # Always process new data
            logging.info("=== Loading Data ===")
            flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations = load_data(current_folder_data_path)
            logging.info("=== Loading Data Completed ===")
            
            # Calculate actual times
            logging.info("=== Calculating Actual Times ===")
            flights = calculate_planned_actual_times(flights)
            logging.info("=== Calculating Actual Times Completed ===")
            
            # Prepare training data
            logging.info("=== Preparing Training Data ===")
            flights = prepare_training_data(
                flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations
            )
            logging.info("=== Preparing Training Data Completed ===")

            # Extract targets and features from new data
            logging.info("=== Extracting Targets and Features ===")
            new_features, new_targets = extract_targetsfeatures_from_flights(flights)
            logging.info("=== Extracting Targets and Features Completed ===")
            
            # Append new data to cached data if it exists
            if cached_features is not None and cached_targets is not None:
                logging.info("Appending new data to existing cached data...")
                cached_features = pd.concat([cached_features, new_features], ignore_index=True)
                cached_targets = pd.concat([cached_targets, new_targets], ignore_index=True)
                logging.info(f"Combined dataset - Features: {new_features.shape}, Targets: {new_targets.shape}")
            else:
                logging.info("No cached data found, using only new data")
                cached_features, cached_targets = new_features, new_targets
            
        except Exception as e:
            logging.debug(f"Training failed for folder {folder}: {str(e)}")
            continue  # Continue with next folder instead of stopping
    
    try:
        # Save combined features and targets to CSV for caching
        logging.info("=== Saving Combined Features and Targets to CSV ===")
        save_features_targets_to_csv(cached_features, cached_targets)
        logging.info("=== Saving Combined Features and Targets to CSV Completed ===")

        # Rename CSV files to .done (use current folder path)
        logging.info("=== Renaming CSV Files to .done ===")
        for folder in processed_folders:
            rename_csv_files_to_done(folder)
        logging.info("=== Renaming CSV Files to .done Completed ===")
                
        # Train models
        logging.info("=== Training Models ===")
        models, scaler, metrics = train_models(features, targets)
        logging.info("=== Training Models Completed ===")
        
        # Save models
        logging.info("=== Saving Models ===")
        save_models(models, scaler, metrics)
        logging.info("=== Saving Models Completed ===")
        
        # Log completion for this folder
        logging.info("=== Completed processing all folders ===")
    except Exception as e:
        logging.error(f"Training failed for folder {folder}: {str(e)}")
        

if __name__ == "__main__":
    main() 