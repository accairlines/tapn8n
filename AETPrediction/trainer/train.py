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
BASE_PATH = os.environ.get('AET_BASE_PATH', r'C:\Users\PedroSarmento\source\repos\hubtmsuite')
DATA_PATH = os.environ.get('AET_DATA_PATH', os.path.join(BASE_PATH, 'AETPrediction', 'data'))
LOG_PATH = os.environ.get('AET_LOG_PATH', os.path.join(BASE_PATH, 'AETPrediction', 'logs'))
MODEL_PATH = os.environ.get('AET_MODEL_PATH', os.path.join(BASE_PATH, 'AETPrediction', 'model.pkl'))

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

def load_data():
    """Load CSV data files as lists of dicts (supporting multiple files per type)"""
    logging.info("Loading CSV data files as lists of dicts...")

    # Define required columns for each file type
    flights_cols = [
        'OPERATOR', 'FLT_NR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA', 'DIV_IATA', 'STD', 'ETD', 'ATD', 'STA',
        'ETA', 'ATA', 'ONBLOCK', 'FROM_TERMINAL', 'FROM_GATE', 'FROM_STAND', 'TO_TERMINAL', 'TO_STAND', 'AC_READY',
        'TSAT', 'PAX_BOARDED', 'CARGO', 'CAPACITY', 'CALL_SIGN', 'OFFBLOCK', 'TOBT', 'CTOT', 'SERV_TYP_COD', 'MVT',
        'CHG_REASON', 
    ]
    flight_plan_cols = [
        'FLP_FILE_NAME', 'STD', 'CALLSIGN', 'CAPTAIN', 'DEPARTURE_AIRP', 'ARRIVAL_AIRP', 'AIRCRAFT_ICAO_TYPE',
        'AIRLINE_SPEC','PERFORMANCE_FACTOR', 'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CLIMB_PROC', 'CLIMB_CI', 'CRUISE_PROC',
        'CRUISE_CI', 'DESCENT_PROC', 'DESCENT_CI', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT', 'TAXI_OUT_TIME', 'TAXI_IN_TIME',
        'FLIGHT_TIME'
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

    def read_multi_csv_to_dicts(root, pattern, usecols=None):
        files = glob.glob(os.path.join(root, pattern))
        if not files:
            logging.warning(f"No files found for pattern: {pattern}, returning empty list")
            return []
        df_list = []
        for f in files:
            logging.info(f"Reading file: {f}")
            df = pd.read_csv(f, encoding='latin1')
            if usecols is not None:
                missing_cols = [col for col in usecols if col not in df.columns]
                for col in missing_cols:
                    df[col] = None
                df = df.reindex(columns=usecols)
            df_list.append(df)
            logging.debug(f"End of reading file: {f}")
        df_list = [df for df in df_list if not df.empty and not all(df.isna().all())]
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            # handle the case where all files are empty
            df = pd.DataFrame(columns=usecols if usecols else [])
        return df.to_dict('records')

    def read_single_csv_to_dicts(file_path, usecols=None):
        """Read a single CSV file and return as list of dicts, return empty list if file doesn't exist"""
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}, returning empty list")
            return []
        
        logging.info(f"Reading file: {file_path}")
        try:
            df = pd.read_csv(file_path, usecols=usecols)
            logging.debug(f"End of reading file: {file_path}")
            return df.to_dict('records')
        except Exception as e:
            logging.warning(f"Error reading {file_path}: {str(e)}, returning empty list")
            return []

    def read_acars_files(root, pattern, usecols=None):
        """Read ACARS files and return as list of dicts, return empty list if no files found"""
        files = glob.glob(os.path.join(root, pattern))
        if not files:
            logging.warning(f"No files found for pattern: {pattern}, returning empty list")
            return []
        df_list = []
        for f in files:
            logging.info(f"Reading file: {f}")
            df = pd.read_csv(f, encoding='latin1')
            if usecols is not None:
                missing_cols = [col for col in usecols if col not in df.columns]
                for col in missing_cols:
                    df[col] = None
                df = df.reindex(columns=usecols)
                df['CALLSIGN'] = df['FLIGHT'].str.replace('TP', 'TAP')
            df_list.append(df)
            logging.debug(f"End of reading file: {f}")
        df_list = [df for df in df_list if not df.empty and not all(df.isna().all())]
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            # handle the case where all files are empty
            df = pd.DataFrame(columns=usecols if usecols else [])
        return df.to_dict('records')
    

    # Load main tables from all subdirectories except cache
    all_flights = []
    all_flight_plans = []
    all_waypoints = []
    all_mel = []
    all_acars = []
    folders_paths = []
    
    for root, dirs, files in os.walk(DATA_PATH):
        if 'cache' in root:
            continue
        folders_paths.append(root)
        logging.info(f"Processing files for path{str(root)}")
        flights_pattern = os.path.join(root, 'flight_*.csv') 
        fp_pattern = os.path.join(root, 'fp_arinc633_fp_*.csv')
        wp_pattern = os.path.join(root, 'fp_arinc633_wp_*.csv')
        mel_pattern = os.path.join(root, 'fp_arinc633_mel_*.csv')
        acars_pattern = os.path.join(root, 'acars_*.csv')
        
        all_flights.extend(read_multi_csv_to_dicts(root, flights_pattern, usecols=flights_cols))
        all_flight_plans.extend(read_multi_csv_to_dicts(root, fp_pattern, usecols=flight_plan_cols))
        all_waypoints.extend(read_multi_csv_to_dicts(root, wp_pattern, usecols=waypoints_cols))
        all_mel.extend(read_multi_csv_to_dicts(root, mel_pattern, usecols=mel_cols))   
        # Load ACARS data and replace TP with TAP in FLIGHT column
        all_acars.extend(read_acars_files(root, acars_pattern, usecols=acars_cols))
        
    flights = all_flights
    flight_plan = all_flight_plans  
    waypoints = all_waypoints
    mel = all_mel
    acars = all_acars
    
    logging.info(f"Loaded {len(all_flights)} flights (as dicts)")
    logging.info(f"Loaded {len(all_flight_plans)} flight plans (as dicts)")
    logging.info(f"Loaded {len(all_waypoints)} waypoints (as dicts)")
    logging.info(f"Loaded {len(all_mel)} mel (as dicts)")
    
 
        
    # Load base tables
    equipments = read_single_csv_to_dicts(os.path.join(DATA_PATH, 'equipments.csv'), usecols=equipments_cols)
    aircrafts = read_single_csv_to_dicts(os.path.join(DATA_PATH, 'aircrafts.csv'), usecols=aircrafts_cols)
    stations = read_single_csv_to_dicts(os.path.join(DATA_PATH, 'stations_utc.csv'), usecols=stations_cols)

    logging.info(f"Loaded {len(flights)} flights (as dicts)")

    return flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations, folders_paths


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
    station_timediff = {(str(s.get('STATION', '')).strip(), s.get('DAY_NUM')): s.get('TIMEDIFF_MINUTES', 0) for s in stations}
    # --- 2. Preprocess flight_plan: keep only latest TS for each (CALLSIGN, DEPARTURE_AIRP, STD) ---
    flight_plan_latest = {}
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
    logging.debug(f"flight_plan_latest... {len(flight_plan_latest)}")
    # --- 3. Merge flight_plan into flights using UTC STD ---
    for flight in flights:
        # Get TIMEDIFF for this flight's FROM_IATA
        from_iata = str(flight.get('FROM_IATA', '')).strip()
        # Convert STD (local) to UTC
        std_local = parse_dt(flight.get('STD'))
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
    logging.debug(f"flights... {len(flights)}")
    # --- 4. Merge aircraft into flights ---
    # Build index for aircrafts by ACREGISTRATION
    aircraft_index = {str(a.get('ACREGISTRATION', '')).strip(): a for a in aircrafts}
    for flight in flights:
        ac_reg = str(flight.get('AC_REGISTRATION', '')).strip()
        flight['aircraft'] = aircraft_index.get(ac_reg)
    logging.debug(f"aircrafts flights... {len(flights)}")
    # --- 5. Merge equipment into flights ---
    # Build index for equipments by ID
    equipment_index = {str(e.get('ID', '')).strip(): e for e in equipments}
    for flight in flights:
        equiptype_id = None
        if flight.get('aircraft'):
            equiptype_id = str(flight['aircraft'].get('EQUIPTYPEID', '')).strip()
        flight['equipment'] = equipment_index.get(equiptype_id)
    logging.debug(f"equipments flights... {len(flights)}")
    # --- 6. Merge mel and waypoints into flights by flight_plan FLP_FILE_NAME ---
    # Build index for mel and waypoints by FLP_FILE_NAME
    mel_index = {}
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
    logging.debug(f"mel and waypoints flights... {len(flights)}")
    # --- 7. Merge acars into flights by CALLSIGN/FLIGHT and REPORTTIME window ---
    for flight in flights:
        callsign = str(flight.get('CALLSIGN', '')).strip()
        std_utc = flight.get('STD_UTC')
        std_utc_end = std_utc + pd.Timedelta(hours=12) if pd.notnull(std_utc) else None
        acars_matches = []
        for a in acars:
            acars_flight = str(a.get('FLIGHT', '')).replace('TP', 'TAP').strip()
            report_time = pd.to_datetime(a.get('REPORTTIME'), errors='coerce')
            if (
                acars_flight == callsign and
                pd.notnull(report_time) and
                pd.notnull(std_utc) and
                std_utc <= report_time <= std_utc_end
            ):
                acars_matches.append(a)
        flight['acars'] = acars_matches
    logging.debug(f"acars flights... {len(flights)}")
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
    
    with open(MODEL_PATH, 'wb') as f:
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
    
    # Save features
    features_path = os.path.join(cache_dir, 'features.csv')
    features.to_csv(features_path, index=False)
    logging.info(f"Features saved to: {features_path}")
    
    # Save targets
    targets_path = os.path.join(cache_dir, 'targets.csv')
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
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Metadata saved to: {metadata_path}")


def rename_csv_files_to_done(folders_paths):
    """Rename all CSV files in the data directory to .done extension"""
    logging.info("Renaming all CSV files to .done extension...")
    
    # Get all CSV files in the data directory
    for folder_path in folders_paths:
        csv_patterns = [
            os.path.join(folder_path, 'flight_*.csv'),
            os.path.join(folder_path, 'fp_arinc633_fp_*.csv'),
            os.path.join(folder_path, 'fp_arinc633_wp_*.csv'),
            os.path.join(folder_path, 'fp_arinc633_mel_*.csv'),
            os.path.join(folder_path, 'acars_*.csv')
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
    """Load features and targets from CSV files if they exist"""
    cache_dir = os.path.join(DATA_PATH, 'cache')
    features_path = os.path.join(cache_dir, 'features.csv')
    targets_path = os.path.join(cache_dir, 'targets.csv')
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    
    if os.path.exists(features_path) and os.path.exists(targets_path):
        logging.info("Loading features and targets from cached CSV files...")
        
        features = pd.read_csv(features_path)
        targets = pd.read_csv(targets_path)
        
        # Load and log metadata
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Loaded cached data - Features: {metadata['features_shape']}, Targets: {metadata['targets_shape']}")
            logging.info(f"Cached data saved at: {metadata['saved_at']}")
        
        return features, targets
    else:
        logging.info("No cached CSV files found")
        return None, None


def main():
    """Main training pipeline"""
    logging.info("=== Starting daily training ===")
    
    try:
        # Load cached data if it exists
        cached_features, cached_targets = load_features_targets_from_csv()
        
        # Always process new data
        logging.info("=== Loading Data ===")
        flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations, folders_paths = load_data()
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
            features = pd.concat([cached_features, new_features], ignore_index=True)
            targets = pd.concat([cached_targets, new_targets], ignore_index=True)
            logging.info(f"Combined dataset - Features: {features.shape}, Targets: {targets.shape}")
        else:
            logging.info("No cached data found, using only new data")
            features, targets = new_features, new_targets
        
        # Save combined features and targets to CSV for caching
        logging.info("=== Saving Combined Features and Targets to CSV ===")
        save_features_targets_to_csv(features, targets)
        logging.info("=== Saving Combined Features and Targets to CSV Completed ===")

        # Rename CSV files to .done
        logging.info("=== Renaming CSV Files to .done ===")
        rename_csv_files_to_done(folders_paths)
        logging.info("=== Renaming CSV Files to .done Completed ===")
                
        # Train models
        logging.info("=== Training Models ===")
        models, scaler, metrics = train_models(features, targets)
        logging.info("=== Training Models Completed ===")
        
        # Save models
        logging.info("=== Saving Models ===")
        save_models(models, scaler, metrics)
        logging.info("=== Saving Models Completed ===")
        
        # Log completion
        logging.info("=== Training completed successfully ===")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 