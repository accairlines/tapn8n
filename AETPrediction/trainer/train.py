#!/usr/bin/env python3
"""
Daily model training script for AET prediction
Reads from CSV files and trains XGBoost model
"""
import sys
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
from ft_transformer import FTTransformerTrainer
import glob
import os
from dotenv import load_dotenv
import traceback
from db_extract import extract_data_per_month

# Load environment variables from .env file
load_dotenv()

# Set base path from environment or default
BASE_PATH = os.environ.get('AET_BASE_PATH')
DATA_PATH = os.environ.get('AET_DATA_PATH')
LOG_PATH = os.environ.get('AET_LOG_PATH')
MODEL_PATH = os.path.join(os.environ.get('AET_MODEL_PATH'), 'model.pkl')

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

flights_cols_all = [
    'OPERATOR', 'FLT_NR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA', 'DIV_IATA', 'STD', 'ETD', 'ATD', 'STA',
    'ETA', 'ATA', 'ONBLOCK', 'FROM_TERMINAL', 'FROM_GATE', 'FROM_STAND', 'TO_TERMINAL', 'TO_STAND', 'AC_READY',
    'TSAT', 'PAX_BOARDED', 'CARGO', 'CAPACITY', 'CALL_SIGN', 'OFFBLOCK', 'TOBT', 'CTOT', 'SERV_TYP_COD', 'MVT',
    'CHG_REASON', 
]
flight_plan_cols_all = [
    'FLP_FILE_NAME', 'STD', 'CALLSIGN', 'CAPTAIN', 'DEPARTURE_AIRP', 'ARRIVAL_AIRP', 'AIRCRAFT_ICAO_TYPE',
    'AIRLINE_SPEC','PERFORMANCE_FACTOR', 'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CLIMB_PROC', 'CLIMB_CI', 'CRUISE_PROC',
    'CRUISE_CI', 'DESCENT_PROC', 'DESCENT_CI', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT', 'TRIP_DURATION', 'TAXI_OUT_TIME', 
    'TAXI_IN_TIME', 'FLIGHT_TIME'
]
waypoints_cols_all = [
    'ALTITUDE', 'SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE',
    'FLP_FILE_NAME', 'CUMULATIVE_FLIGHT_TIME'
]
mel_cols_all = ['FLP_FILE_NAME']
acars_cols_all = ['FLIGHT', 'REPORTTIME', 'WINDDIRECTION', 'WINDSPEED']
equipments_cols_all = ['ID', 'BODYTYPE', 'EQUIPTYPE', 'EQUIPTYPE2']
aircrafts_cols_all = ['ACREGISTRATION', 'EQUIPTYPEID']
stations_cols_all = ['STATION', 'TIMEDIFF_MINUTES', 'DAY_NUM']

def load_data():
    """Load CSV data files as lists of dicts (supporting multiple files per type)"""
    logging.info("Loading CSV data files as lists of dicts...")

    # Define required columns for each file type
    flights_cols = flights_cols_all
    flight_plan_cols = flight_plan_cols_all
    waypoints_cols = waypoints_cols_all
    mel_cols = mel_cols_all
    acars_cols = acars_cols_all
    equipments_cols = equipments_cols_all
    aircrafts_cols = aircrafts_cols_all
    stations_cols = stations_cols_all

    def read_multi_csv_to_dicts(root, pattern, usecols=None):
        # Also check for .done files (files that were already processed)
        files = glob.glob(os.path.join(root, pattern))
        # Check for .done files by replacing .csv with .done in the pattern
        done_pattern = pattern.replace('*.csv', '*.done')
        done_files = glob.glob(os.path.join(root, done_pattern))
        # Combine both .csv and .done files
        files = files + done_files
        if not files:
            logging.warning(f"No files found for pattern: {pattern} or .done files, returning empty list")
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
                # Replace NaN values with None
                df = df.replace({pd.NA: None, pd.NaT: None, np.nan: None})
                df = df.where(pd.notna(df), None)
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
        # Also check for .done files (files that were already processed)
        files = glob.glob(os.path.join(root, pattern))
        # Check for .done files by replacing .csv with .done in the pattern
        done_pattern = pattern.replace('*.csv', '*.done')
        done_files = glob.glob(os.path.join(root, done_pattern))
        # Combine both .csv and .done files
        files = files + done_files
        if not files:
            logging.warning(f"No files found for pattern: {pattern} or .done files, returning empty list")
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
        if 'cache' in root or 'emails' in root:
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

    # Helper to parse time string to seconds
    def parse_time_to_seconds(time_str, default='00:00:00'):
        try:
            if pd.isna(time_str) or time_str is None:
                time_str = default
            h, m, s = map(int, str(time_str).split(':'))
            return h * 3600 + m * 60 + s
        except (ValueError, AttributeError):
            return 0

    for flight in flights:
        # Parse all relevant datetimes
        offblock = parse_dt(flight.get('OFFBLOCK'))
        mvt = parse_dt(flight.get('MVT'))
        ata = parse_dt(flight.get('ATA'))
        onblock = parse_dt(flight.get('ONBLOCK'))
        eta = parse_dt(flight.get('ETA'))
        from_timediff = flight.get('FROM_TIMEDIFF')
        to_timediff = flight.get('TO_TIMEDIFF')
        std = parse_dt(flight.get('STD'))
        
        flight_plan = flight.get('flight_plan')
        # Calculate planned times (in seconds)
        planned_taxi_out = parse_time_to_seconds(flight_plan.get('TAXI_OUT_TIME')) / 60 if flight_plan.get('TAXI_OUT_TIME') is not None else 0
        flight_time = flight_plan.get('FLIGHT_TIME') or flight_plan.get('TRIP_DURATION')
        planned_airborne = parse_time_to_seconds(flight_time) / 60 if flight_time is not None else 0
        planned_taxi_in = parse_time_to_seconds(flight_plan.get('TAXI_IN_TIME')) / 60 if flight_plan.get('TAXI_IN_TIME') is not None else 0

        # Calculate actual times (in minutes)
        actual_taxi_out = (mvt - offblock).total_seconds() / 60 if pd.notnull(mvt) and pd.notnull(offblock) else None
        actual_airborne = ((ata - mvt).total_seconds() - (to_timediff * 60)  + (from_timediff * 60)) / 60 if pd.notnull(ata) and pd.notnull(mvt) else None
        actual_taxi_in = (onblock - ata).total_seconds() / 60 if pd.notnull(onblock) and pd.notnull(ata) else None
        
        # Calculate totals
        actual_total_time = actual_taxi_out + actual_airborne + actual_taxi_in if actual_taxi_out is not None and actual_airborne is not None and actual_taxi_in is not None else None
        planned_total_time = planned_taxi_out + planned_airborne + planned_taxi_in
        
        # AET: ATA - MVT (in minutes)
        aet = actual_airborne
        # EET: planned total time (in minutes)
        eet = planned_total_time
        # Delta: AET - EET (in minutes)
        raw_delta = (aet - eet) if aet is not None and eet is not None else None
        # Convert delta to percentage of EET
        actual_delta = (raw_delta / eet * 100) if raw_delta is not None and eet is not None and eet != 0 else None
        
        # Delay departure: STD - MVT (in minutes)
        delay_dep = (std - offblock).total_seconds() / 60 if pd.notnull(std) and pd.notnull(offblock) else None

        # Add calculated values to the copy
        flight['actual_taxi_out'] = actual_taxi_out
        flight['actual_airborne'] = actual_airborne
        flight['actual_taxi_in'] = actual_taxi_in
        flight['actual_total_time'] = actual_total_time
        flight['planned_taxi_out'] = planned_taxi_out
        flight['planned_airborne'] = planned_airborne
        flight['planned_taxi_in'] = planned_taxi_in
        flight['planned_total_time'] = planned_total_time
        flight['AET'] = aet
        flight['EET'] = eet
        flight['delay_dep'] = delay_dep
        flight['actual_delta'] = actual_delta

    return flights


def prepare_training_data(flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations):
    """Prepare features and targets for training using list-of-dicts and manual merging."""
    logging.debug("Preparing training data (dict version)...")

    # Helper to parse datetimes safely
    def parse_dt(val):
        try:
            return pd.to_datetime(val, errors='coerce')
        except Exception:
            return pd.NaT

    # --- 1. Build station TIMEDIFF map ---
    stations_timediff = {(str(s.get('STATION', '')).strip(), s.get('DAY_NUM')): s.get('TIMEDIFF_MINUTES', 0) for s in stations}
    # --- 2. Preprocess flight_plan: keep only latest TS for each (CALLSIGN, DEPARTURE_AIRP, STD) ---
    flight_plan_latest = {}
    logging.debug(f"flight_plan to be processed... {len(flight_plan)}")
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
    logging.debug(f"flights to be processed... {len(flights)}")
    # --- 3. Merge flight_plan into flights using UTC STD ---
    for flight in flights:
        # Get TIMEDIFF for this flight's FROM_IATA
        from_iata = str(flight.get('FROM_IATA', '')).strip()
        # Convert STD (local) to UTC
        std_local = parse_dt(flight.get('STD'))
        from_timediff = stations_timediff.get((from_iata, std_local.day_of_year), 0)
        std_utc = std_local - pd.to_timedelta(from_timediff, unit='m') if pd.notnull(std_local) else pd.NaT
        # Build key for matching
        key = (
            str(flight.get('CALL_SIGN', '')).strip(),
            from_iata,
            str(std_utc)
        )
        flight['FROM_TIMEDIFF'] = from_timediff
        flight['STD_UTC'] = std_utc
        # Build key for matching
        key = (
            str(flight.get('CALL_SIGN', '')).strip(),
            from_iata,
            str(std_utc)
        )
        to_iata = str(flight.get('TO_IATA', '')).strip()
        sta_local = parse_dt(flight.get('STA'))
        to_timediff = stations_timediff.get((to_iata, sta_local.day_of_year), 0)
        flight['TO_TIMEDIFF'] = to_timediff
        flight['flight_plan'] = flight_plan_latest.get(key)
    logging.debug(f"flights... {len(flights)}")
    # --- 4. Merge aircraft into flights ---
    # Build index for aircrafts by ACREGISTRATION
    logging.debug(f"aircrafts to be processed... {len(aircrafts)}")
    aircraft_index = {str(a.get('ACREGISTRATION', '')).strip(): a for a in aircrafts}
    for flight in flights:
        ac_reg = str(flight.get('AC_REGISTRATION', '')).strip()
        flight['aircraft'] = aircraft_index.get(ac_reg)
    logging.debug(f"aircrafts flights... {len(flights)}")
    # --- 5. Merge equipment into flights ---
    # Build index for equipments by ID
    logging.debug(f"equipments to be processed... {len(equipments)}")
    equipment_index = {str(e.get('ID', '')).strip(): e for e in equipments}
    for flight in flights:
        equiptype_id = None
        if flight.get('aircraft'):
            equiptype_id = str(flight['aircraft'].get('EQUIPTYPEID', '')).strip()
        flight['equipment'] = equipment_index.get(equiptype_id)
    logging.debug(f"equipments flights... {len(flights)}")
    # --- 6. Merge mel and waypoints into flights by flight_plan FLP_FILE_NAME ---
    # Build index for mel and waypoints by FLP_FILE_NAME
    logging.debug(f"mel to be processed... {len(mel)}")
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
    logging.debug(f"acars to be processed... {len(acars)}")
    acars_index = {}
    for a in acars:
        fname = str(a.get('FLIGHT', '')).replace('TP', 'TAP').strip()
        acars_index.setdefault(fname, []).append(a)
    # --- 7. Merge acars into flights by CALLSIGN/FLIGHT and REPORTTIME window ---
    for flight in flights:
        callsign = str(flight.get('CALLSIGN', '')).strip()
        std_utc = flight.get('STD_UTC')
        std_utc_end = std_utc + pd.Timedelta(hours=12) if pd.notnull(std_utc) else None
        acars_matches = []
        for a in acars_index.get(callsign, []):
            report_time = pd.to_datetime(a.get('REPORTTIME'), errors='coerce')
            if (
                pd.notnull(report_time) and
                pd.notnull(std_utc) and
                std_utc <= report_time <= std_utc_end
            ):
                # Get TIMEDIFF for this flight's FROM_IATA
                from_iata = str(flight.get('FROM_IATA', '')).strip()
                # Convert STD (local) to UTC
                std_local = parse_dt(flight.get('STD'))
                # Get flight's TIMEDIFF_MINUTES and STA
                sta = pd.to_datetime(flight.get('STA'), errors='coerce')
                eta = pd.to_datetime(flight.get('ETA'), errors='coerce')
                
                # Calculate STA_UTC by subtracting TIMEDIFF_MINUTES from STA
                if pd.notnull(sta):
                    sta_utc = sta - pd.Timedelta(minutes=to_timediff)
                    
                    # Calculate minutes between REPORTTIME and STA_UTC
                    report_time = pd.to_datetime(a.get('REPORTTIME'), errors='coerce')
                    if pd.notnull(report_time) and pd.notnull(sta_utc):
                        minutes_to_sta = (sta_utc - report_time).total_seconds() / 60
                        minutes_to_eta = (sta_utc - report_time).total_seconds() / 60
                        a['MINUTES_TO_STA'] = minutes_to_sta
                        a['MINUTES_TO_ETA'] = minutes_to_eta
                    else:
                        a['MINUTES_TO_STA'] = -1
                        a['MINUTES_TO_ETA'] = -1
                else:
                    a['MINUTES_TO_STA'] = -1
                    a['MINUTES_TO_ETA'] = -1
                
                acars_matches.append(a)
        flight['acars'] = acars_matches
    logging.debug(f"acars flights... {len(flights)}")
    # All merges done
    return flights


def train_models(features, targets):
    """Train both XGBoost and FT-Transformer models for each time component"""
    logging.info("Training models (XGBoost and FT-Transformer)...")

    # Ensure only numeric columns are used for XGBoost
    features_numeric = features.select_dtypes(include=[np.number])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_numeric, targets, test_size=0.2, random_state=42
    )
    
    # Scale features for XGBoost
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare data for FT-Transformer: separate categorical and numerical
    # Identify categorical columns (those ending with '_code')
    categorical_cols = [col for col in features.columns if col.endswith('_code')]
    numerical_cols = [col for col in features.columns if col not in categorical_cols and col in features_numeric.columns]
    
    X_train_cat = X_train[categorical_cols] if categorical_cols else pd.DataFrame()
    X_train_num = X_train[numerical_cols] if numerical_cols else pd.DataFrame()
    X_test_cat = X_test[categorical_cols] if categorical_cols else pd.DataFrame()
    X_test_num = X_test[numerical_cols] if numerical_cols else pd.DataFrame()
    
    models = {}
    ft_transformer_models = {}
    metrics = {}
    ft_transformer_metrics = {}
    
    # Train model for each target
    for target in ['delta']:
        logging.info(f"Training XGBoost model for {target}...")
        
        # Train XGBoost
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
        
        # Evaluate XGBoost
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)
        
        models[target] = model
        metrics[target] = {'mae': mae, 'r2': r2}
        
        logging.info(f"XGBoost {target} - MAE: {mae:.2f}%, R2: {r2:.3f}")
        
        # Train FT-Transformer
        logging.info(f"Training FT-Transformer model for {target}...")
        
        num_numerical = len(numerical_cols) if numerical_cols else 0
        num_categories = len(categorical_cols) if categorical_cols else 0
        
        if num_numerical == 0 and num_categories == 0:
            logging.warning("No features available for FT-Transformer, skipping...")
            ft_transformer_models[target] = None
            ft_transformer_metrics[target] = {'mae': None, 'r2': None}
        else:
            trainer = FTTransformerTrainer(
                num_numerical=num_numerical,
                num_categories=num_categories,
                d_token=192,
                n_layers=3,
                n_heads=8,
                d_ff=768,
                dropout=0.1,
                learning_rate=1e-4,
                batch_size=256,
                n_epochs=50,  # Reduced for faster training
                device=None
            )
            
            # Train FT-Transformer
            ft_model, ft_scaler = trainer.train(
                X_train_num if not X_train_num.empty else None,
                X_train_cat if not X_train_cat.empty else None,
                y_train[target].values
            )
            
            # Evaluate FT-Transformer
            y_pred_ft = trainer.predict(
                X_test_num if not X_test_num.empty else None,
                X_test_cat if not X_test_cat.empty else None
            )
            mae_ft = mean_absolute_error(y_test[target], y_pred_ft)
            r2_ft = r2_score(y_test[target], y_pred_ft)
            
            ft_transformer_models[target] = {
                'model': ft_model,
                'trainer': trainer,
                'scaler': ft_scaler
            }
            ft_transformer_metrics[target] = {'mae': mae_ft, 'r2': r2_ft}
            
            logging.info(f"FT-Transformer {target} - MAE: {mae_ft:.2f}%, R2: {r2_ft:.3f}")
    
    return models, scaler, metrics, ft_transformer_models, ft_transformer_metrics


def save_models(models, scaler, metrics, ft_transformer_models=None, ft_transformer_metrics=None):
    """Save trained models and scaler (both XGBoost and FT-Transformer)"""
    logging.info("Saving models...")
    
    model_data = {
        'models': models,
        'scaler': scaler,
        'metrics': metrics,
        'training_date': datetime.now().isoformat(),
        'feature_names': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else None
    }
    
    # Add FT-Transformer models if provided
    if ft_transformer_models is not None:
        # Save FT-Transformer models (save state dicts, not full models)
        ft_model_data = {}
        for target, ft_data in ft_transformer_models.items():
            if ft_data is not None:
                ft_model_data[target] = {
                    'model_state_dict': ft_data['model'].state_dict(),
                    'model_config': {
                        'num_numerical': ft_data['trainer'].num_numerical,
                        'num_categories': ft_data['trainer'].num_categories,
                        'd_token': ft_data['trainer'].d_token,
                        'n_layers': ft_data['trainer'].n_layers,
                        'n_heads': ft_data['trainer'].n_heads,
                        'd_ff': ft_data['trainer'].d_ff,
                        'dropout': ft_data['trainer'].dropout
                    },
                    'numerical_scaler': ft_data['scaler']
                }
            else:
                ft_model_data[target] = None
        
        model_data['ft_transformer_models'] = ft_model_data
        model_data['ft_transformer_metrics'] = ft_transformer_metrics
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    logging.info("Models saved successfully")

def extract_targetsfeatures_from_flights(flights):
    """Extract features DataFrame from enriched flights list, flattening selected flight, selected flight_plan fields, selected equipment fields, up to 50 relevant waypoint features, and up to 20 acars features."""
    # Process in batches to avoid memory exhaustion
    BATCH_SIZE = 10000
    all_features = []
    all_targets = []
    
    # Define required columns for each file type
    flights_cols = flights_cols_all
    flight_plan_cols = flight_plan_cols_all
    waypoints_cols = waypoints_cols_all
    mel_cols = mel_cols_all
    acars_cols = acars_cols_all
    equipments_cols = equipments_cols_all
    
    # Filter flights with flight_plan first
    flights_with_fp = [f for f in flights if f.get('flight_plan')]
    total_flights = len(flights_with_fp)
    logging.info(f"Processing {total_flights} flights in batches of {BATCH_SIZE}")
    
    if total_flights == 0:
        logging.warning("No flights with flight_plan found, returning empty DataFrames")
        return pd.DataFrame(), pd.DataFrame()
    
    """Extract targets DataFrame from enriched flights list."""
    for batch_start in range(0, total_flights, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_flights)
        batch_flights = flights_with_fp[batch_start:batch_end]
        logging.info(f"Processing batch {batch_start//BATCH_SIZE + 1}: flights {batch_start} to {batch_end-1}")
        
        data = []
        for flight in batch_flights:
            row = {}
            # Add only selected flight fields
            for k in flights_cols:
                v = flight.get(k)
                if not isinstance(v, (dict, list)):
                    row[k] = v
            # Add only selected flight_plan fields, prefixed with 'fp_'
            for k in flight_plan_cols:
                v = flight['flight_plan'].get(k)
                if not isinstance(v, (dict, list)):
                    row[f'fp_{k}'] = v
            # Add selected equipment fields, prefixed with 'eq_'
            eq = flight.get('equipment')
            if eq:
                for ek in equipments_cols:
                    row[f'eq_{ek}'] = eq.get(ek)
            # Add up to 50 waypoints with ALTITUDE > 299, extracting 3 fields
            waypoints = [wp for wp in (flight.get('waypoints') or []) if wp.get('ALTITUDE') is not None and float(wp.get('ALTITUDE', 0)) > 299]
            for i in range(50):
                if i < len(waypoints):
                    wp = waypoints[i]
                    for f in waypoints_cols:
                        row[f'wp{i+1}_{f}'] = wp.get(f)
                else:
                    for f in waypoints_cols:
                        row[f'wp{i+1}_{f}'] = None
            # Add up to 20 acars, extracting WINDDIRECTION and WINDSPEED
            acars_list = flight.get('acars') or []
            for i in range(20):
                if i < len(acars_list):
                    ac = acars_list[i]
                    for f in acars_cols:
                        row[f'acars{i+1}_{f}'] = ac.get(f)
                else:
                    for f in acars_cols:
                        row[f'acars{i+1}_{f}'] = None
            row['actual_taxi_out'] = flight.get('actual_taxi_out')
            row['actual_airborne'] = flight.get('actual_airborne')
            row['actual_taxi_in'] = flight.get('actual_taxi_in')
            row['actual_total_time'] = flight.get('actual_total_time')
            row['planned_taxi_out'] = flight.get('planned_taxi_out')
            row['planned_airborne'] = flight.get('planned_airborne')
            row['planned_taxi_in'] = flight.get('planned_taxi_in')
            row['planned_total_time'] = flight.get('planned_total_time')
            row['AET'] = flight.get('AET')
            row['EET'] = flight.get('EET')
            row['delta'] = flight.get('actual_delta')
            data.append(row)
        
        # Process batch and append to results
        batch_features, batch_targets = preprocess_flight_data(data)
        all_features.append(batch_features)
        all_targets.append(batch_targets)
        
        # Clear batch data to free memory
        del data, batch_features, batch_targets
        
    # Concatenate all batches
    logging.info("Concatenating all batches...")
    if len(all_features) == 0:
        logging.warning("No features extracted, returning empty DataFrames")
        return pd.DataFrame(), pd.DataFrame()
    
    features_processed = pd.concat(all_features, ignore_index=True)
    targets_processed = pd.concat(all_targets, ignore_index=True)
    logging.info(f"Successfully processed {len(features_processed)} flights into features and targets")
    
    # Clear intermediate lists
    del all_features, all_targets
    
    return features_processed, targets_processed


def get_cache_paths(cache_type='current'):
    """Get cache file paths for static (past months) or current month cache"""
    cache_dir = os.path.join(DATA_PATH, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    if cache_type == 'static':
        features_path = os.path.join(cache_dir, 'features.csv')
        targets_path = os.path.join(cache_dir, 'targets.csv')
        metadata_path = os.path.join(cache_dir, 'metadata.json')
    else:  # current month
        features_path = os.path.join(cache_dir, 'current_month_features.csv')
        targets_path = os.path.join(cache_dir, 'current_month_targets.csv')
        metadata_path = os.path.join(cache_dir, 'current_month_metadata.json')
    
    return features_path, targets_path, metadata_path


def get_last_cache_month():
    """Get the last month that was cached (from current_month_metadata.json)"""
    _, _, metadata_path = get_cache_paths('current')
    if os.path.exists(metadata_path):
        import json
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if 'month' in metadata:
                    return metadata['month']
        except Exception as e:
            logging.warning(f"Error reading last cache month: {str(e)}")
    return None


def check_and_handle_month_transition():
    """Check if month has changed and handle transition by appending current month to static cache"""
    current_date = datetime.now()
    current_month = (current_date.year, current_date.month)
    
    last_month = get_last_cache_month()
    
    if last_month is not None and last_month != current_month:
        logging.info(f"Month transition detected: {last_month} -> {current_month}")
        logging.info("Appending current month data to static cache...")
        
        # Load current month cache
        features_path, targets_path, metadata_path = get_cache_paths('current')
        
        if os.path.exists(features_path) and os.path.exists(targets_path):
            current_features = pd.read_csv(features_path)
            current_targets = pd.read_csv(targets_path)
            
            # Load static cache (past months)
            static_features_path, static_targets_path, static_metadata_path = get_cache_paths('static')
            
            if os.path.exists(static_features_path) and os.path.exists(static_targets_path):
                # Append current month to static cache
                static_features = pd.read_csv(static_features_path)
                static_targets = pd.read_csv(static_targets_path)
                
                combined_features = pd.concat([static_features, current_features], ignore_index=True)
                combined_targets = pd.concat([static_targets, current_targets], ignore_index=True)
                
                # Save updated static cache
                combined_features.to_csv(static_features_path, index=False)
                combined_targets.to_csv(static_targets_path, index=False)
                
                # Update static metadata
                import json
                static_metadata = {
                    'features_shape': combined_features.shape,
                    'targets_shape': combined_targets.shape,
                    'last_updated_at': datetime.now().isoformat(),
                    'last_completed_month': last_month,
                    'feature_columns': combined_features.columns.tolist(),
                    'target_columns': combined_targets.columns.tolist()
                }
                with open(static_metadata_path, 'w') as f:
                    json.dump(static_metadata, f, indent=2)
                
                logging.info(f"Appended {len(current_features)} rows to static cache")
            else:
                # First time - move current month to static cache
                current_features.to_csv(static_features_path, index=False)
                current_targets.to_csv(static_targets_path, index=False)
                
                import json
                static_metadata = {
                    'features_shape': current_features.shape,
                    'targets_shape': current_targets.shape,
                    'last_updated_at': datetime.now().isoformat(),
                    'last_completed_month': last_month,
                    'feature_columns': current_features.columns.tolist(),
                    'target_columns': current_targets.columns.tolist()
                }
                with open(static_metadata_path, 'w') as f:
                    json.dump(static_metadata, f, indent=2)
                
                logging.info(f"Moved {len(current_features)} rows to static cache (first time)")
        
        # Clear current month cache files
        if os.path.exists(features_path):
            os.remove(features_path)
        if os.path.exists(targets_path):
            os.remove(targets_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        logging.info("Current month cache cleared for new month")
        return True
    
    return False


def save_features_targets_to_csv(features, targets):
    """Save features and targets to current month cache files (recreated each run)"""
    logging.info("Saving features and targets to current month cache files...")
    
    # Check for month transition first
    check_and_handle_month_transition()
    
    # Get current month cache paths
    features_path, targets_path, metadata_path = get_cache_paths('current')
    
    # Always recreate current month cache files (remove if exist)
    if os.path.exists(features_path):
        os.remove(features_path)
    if os.path.exists(targets_path):
        os.remove(targets_path)
    
    # Save features and targets to current month cache
    features.to_csv(features_path, index=False)
    logging.info(f"Current month features saved to: {features_path}")
    
    targets.to_csv(targets_path, index=False)
    logging.info(f"Current month targets saved to: {targets_path}")
    
    # Save metadata with timestamp and current month
    current_date = datetime.now()
    current_month = (current_date.year, current_date.month)
    
    metadata = {
        'features_shape': features.shape,
        'targets_shape': targets.shape,
        'saved_at': current_date.isoformat(),
        'month': current_month,
        'feature_columns': features.columns.tolist(),
        'target_columns': targets.columns.tolist()
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Current month metadata saved to: {metadata_path}")


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

def main():
    """Main training pipeline"""
    logging.info("=== Starting daily training ===")
    
    try:
        # Extract data from database per month
        logging.info("=== Extracting Data from Database ===")
        try:
            # Extract data from to current month
            extract_data_per_month(2024, 7)
            logging.info("=== Database Extraction Completed ===")
        except Exception as e:
            logging.warning(f"Database extraction failed or skipped: {str(e)}")
            logging.info("Continuing with existing data files...")
                
        # Always process new data
        logging.info("=== Loading Data ===")
        flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations, folders_paths = load_data()
        logging.info("=== Loading Data Completed ===")
                
        # Prepare training data
        logging.info("=== Preparing Training Data ===")
        flights = prepare_training_data(
            flights, flight_plan, waypoints, mel, acars, equipments, aircrafts, stations
        )
        logging.info("=== Preparing Training Data Completed ===")
        
        # Remove flights with no flight_plan
        logging.info("=== Removing Flights with No Flight Plan ===")
        flights = [flight for flight in flights if flight.get('flight_plan')]
        logging.info("=== Removing Flights with No Flight Plan Completed ===")
        
        # Calculate actual times
        logging.info("=== Calculating Actual Times ===")
        flights = calculate_planned_actual_times(flights)
        logging.info("=== Calculating Actual Times Completed ===")

        # Extract targets and features from new data
        logging.info("=== Extracting Targets and Features ===")
        new_features, new_targets = extract_targetsfeatures_from_flights(flights)
        logging.info("=== Extracting Targets and Features Completed ===")
        
        # Check if there's new data to process
        if len(new_features) == 0:
            logging.info("Nothing to process as features are empty")
            return 0
        
        # Separate cached data: static (past months) and current month
        # cached_features/targets from load_features_targets_from_csv() contains static + current month combined
        # We need to extract only current month data to rebuild current month cache
        
        # Get current month cache separately
        current_features_path, current_targets_path, _ = get_cache_paths('current')
        current_month_cache_features = None
        current_month_cache_targets = None
        
        if os.path.exists(current_features_path) and os.path.exists(current_targets_path):
            current_month_cache_features = pd.read_csv(current_features_path)
            current_month_cache_targets = pd.read_csv(current_targets_path)
        
        # Rebuild current month cache: combine existing current month cache + new data
        # This is recreated each run with all current month data processed so far
        if current_month_cache_features is not None and current_month_cache_targets is not None:
            logging.info("Rebuilding current month cache: combining existing current month cache with new data...")
            current_month_features = pd.concat([current_month_cache_features, new_features], ignore_index=True)
            current_month_targets = pd.concat([current_month_cache_targets, new_targets], ignore_index=True)
        else:
            logging.info("Creating new current month cache with new data...")
            current_month_features = new_features
            current_month_targets = new_targets
        
        # Save to current month cache (this replaces any existing current month cache)
        # The save function will handle month transition automatically
        logging.info(f"Current month cache - Features: {current_month_features.shape}, Targets: {current_month_targets.shape}")
        logging.info("=== Saving to Current Month Cache ===")
        save_features_targets_to_csv(current_month_features, current_month_targets)
        logging.info("=== Saving to Current Month Cache Completed ===")
        
        # For training, combine static cache (past months) + current month cache
        static_features_path, static_targets_path, _ = get_cache_paths('static')
        static_features = None
        static_targets = None
        
        if os.path.exists(static_features_path) and os.path.exists(static_targets_path):
            static_features = pd.read_csv(static_features_path)
            static_targets = pd.read_csv(static_targets_path)
        
        # Combine for training
        if static_features is not None and static_targets is not None:
            logging.info("Combining static cache (past months) with current month cache for training...")
            features = pd.concat([static_features, current_month_features], ignore_index=True)
            targets = pd.concat([static_targets, current_month_targets], ignore_index=True)
            logging.info(f"Final training dataset - Features: {features.shape}, Targets: {targets.shape}")
        else:
            logging.info("Using only current month cache for training (no static cache found)")
            features, targets = current_month_features, current_month_targets
            logging.info(f"Training dataset - Features: {features.shape}, Targets: {targets.shape}")

        # Rename CSV files to .done
        logging.info("=== Renaming CSV Files to .done ===")
        rename_csv_files_to_done(folders_paths)
        logging.info("=== Renaming CSV Files to .done Completed ===")
                
        # Train models
        logging.info("=== Training Models ===")
        models, scaler, metrics, ft_transformer_models, ft_transformer_metrics = train_models(features, targets)
        logging.info("=== Training Models Completed ===")
        
        # Save models
        logging.info("=== Saving Models ===")
        save_models(models, scaler, metrics, ft_transformer_models, ft_transformer_metrics)
        logging.info("=== Saving Models Completed ===")
        
        # Log completion
        logging.info("=== Training completed successfully ===")
        return 0
    except Exception as e:
        error = e.args[0]
        logging.error(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        return 1

if __name__ == "__main__":
    success = main()     
    sys.exit(0 if success else 1)
