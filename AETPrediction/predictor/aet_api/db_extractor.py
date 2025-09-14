"""
Database extraction module for AET prediction
Connects to Django database and extracts data for model training/prediction
"""

import pandas as pd
import numpy as np
from django.conf import settings
from django.db import connections
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

class DatabaseExtractor:
    """Extract and process data from the database for AET prediction"""
    
    def __init__(self):
        self.engine = connections['default']
        # Create SQLAlchemy engine from Django database settings
        db_config = settings.DATABASES['default']
        
        # Build connection string with SSL parameters
        connection_string = f"mysql+pymysql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
        
        # Add SSL parameters if they exist in Django settings
        ssl_params = {}
        if 'OPTIONS' in db_config and 'ssl' in db_config['OPTIONS']:
            ssl_config = db_config['OPTIONS']['ssl']
            if 'ca' in ssl_config and ssl_config['ca']:
                ssl_params['ssl_ca'] = ssl_config['ca']
            # Add other SSL parameters as needed
            ssl_params['ssl_verify_cert'] = True
            ssl_params['ssl_verify_identity'] = True
        
        self.engine = create_engine(connection_string, connect_args=ssl_params)
        
    def extract_flight_data(self, start_date=None, end_date=None, days_back=2, flight_id=None):
        """
        Extract flight data from database and create DataFrame with features for XGBoost model
        
        Args:
            start_date: Start date for data extraction (YYYY-MM-DD format)
            end_date: End date for data extraction (YYYY-MM-DD format) 
            days_back: Number of days back from today if dates not provided
            flight_id: Specific flight ID from OSUSR_UUK_FLT_INFO table
            
        Returns:
            pandas.DataFrame: Processed data ready for XGBoost model
        """
        try:
            if flight_id:
                logger.info(f"Extracting data for specific flight ID: {flight_id}")
                # Extract data for specific flight
                flights_df = self._extract_single_flight(flight_id)
                if flights_df.empty:
                    logger.warning(f"No flight found with ID: {flight_id}")
                    return pd.DataFrame()
                
                # Get the flight's date range for related data
                flight_date = flights_df['STD'].iloc[0]
                start_date = flight_date.strftime('%Y-%m-%d')
                end_date = flight_date.strftime('%Y-%m-%d')
                stationFrom = flights_df['FROM_IATA'].iloc[0]
                stationTo = flights_df['TO_IATA'].iloc[0]
            else:
                # Set default date range if not provided
                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                logger.info(f"Extracting flight data from {start_date} to {end_date}")
                flights_df = self._extract_flights_table(start_date, end_date)
            
            # Check if we have any flight data
            if flights_df.empty:
                logger.warning("No flight data found for the given criteria")
                return pd.DataFrame()
            
            stations_df = self._extract_stations_table(stationFrom=stationFrom, stationTo=stationTo)
            
            callsign = flights_df['CALL_SIGN'].iloc[0].replace('TAP', 'TP') if not flights_df.empty and 'CALL_SIGN' in flights_df.columns else None
            std = flights_df['STD'].iloc[0] if not flights_df.empty and 'STD' in flights_df.columns else None
            timediff_df = stations_df[(stations_df['STATION'] == stationFrom) & (stations_df['DAYS_NUM'] == std.day_of_year)]['timediff_minutes'].iloc[0] if not stations_df.empty and 'timediff_minutes' in stations_df.columns else None
            std_utc = std - timedelta(minutes=int(timediff_df) if timediff_df is not None else 0)
            
            # Extract data from all tables
            callsign = flights_df['CALL_SIGN'].iloc[0] if not flights_df.empty else None
            flight_plans_df = self._extract_flight_plans_table(std_utc.replace(hour=0, minute=0, second=0, microsecond=0), 
                                                               std_utc.replace(hour=0, minute=0, second=0, microsecond=0), 
                                                               callsign=callsign)
            
            flt_file_name = flight_plans_df['FLP_FILE_NAME'].iloc[0] if not flight_plans_df.empty else None
            waypoints_df = self._extract_waypoints_table(start_date, end_date, flt_file_name=flt_file_name)
            mel_df = self._extract_mel_table(start_date, end_date, flt_file_name=flt_file_name)
            
            
            acars_df = self._extract_acars_table(start_date, end_date, callsign=callsign, std_utc=std_utc)
            equipments_df = self._extract_equipments_table()
            aircrafts_df = self._extract_aircrafts_table()
            
            # Join all data together
            combined_df = self._join_all_data(
                flights_df, flight_plans_df, waypoints_df, mel_df, 
                acars_df, equipments_df, aircrafts_df, stations_df
            )
                
            logger.info(f"Successfully extracted {len(combined_df)} records with {len(combined_df.columns)} features")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error extracting flight data: {str(e)}")
            raise
    
    def extract_flight_hist_aeteet(self, flight_id):
        """Extract data from ACARS table"""
        query = """
        SELECT FLT_NO, DELTA
        FROM osusr_uuk_flt_info f
        INNER JOIN view_hist_aet_eet h ON f.FLT_NR = h.FLT_NO
        WHERE f.ID = %s"""
        
        params = [flight_id]
        logger.info(f"Looking for flight with ID: {flight_id} (type: {type(flight_id)})")
        return pd.read_sql(query, self.engine, params=tuple(params))
    
    
    def _extract_single_flight(self, flight_id):
        """Extract data for a specific flight by ID"""
        logger.info(f"Looking for flight with ID: {flight_id} (type: {type(flight_id)})")
        query = """
        SELECT ID,OPERATOR,FLT_NR,AC_REGISTRATION,FROM_IATA,TO_IATA,DIV_IATA,STD,ETD,ATD,STA,ETA,ATA,ONBLOCK,
               FROM_TERMINAL,FROM_GATE,FROM_STAND,TO_TERMINAL,TO_STAND,TO_BELT,DOOROPEN,DOORCLOSED,HULLOPEN,HULLCLOSED,
               AC_READY,TSAT,ASSIGNUSER,ISOTPDISMISS,PAX_BOOKED,PAX_CHECKED,PAX_BOARDED,CARGO,CAPACITY,STATUS,
               LAST_BAGMSGS_PROC,SEMAPHORE,CALL_SIGN,OFFBLOCK,TOBT,CTOT,HAUL_IND,AC_OWNER,SERV_TYP_COD,NI,MVT,TSAT_STATUS,CHG_REASON
        FROM osusr_uuk_flt_info
        WHERE ID = %s
        """
        result = pd.read_sql(query, self.engine, params=(flight_id,))
        logger.info(f"Found {len(result)} records for flight ID {flight_id}")
        return result
    
    def _extract_flights_table(self, start_date, end_date):
        """Extract data from flights table"""
        query = """
        SELECT ID,OPERATOR,FLT_NR,AC_REGISTRATION,FROM_IATA,TO_IATA,DIV_IATA,STD,ETD,ATD,STA,ETA,ATA,ONBLOCK,
               FROM_TERMINAL,FROM_GATE,FROM_STAND,TO_TERMINAL,TO_STAND,TO_BELT,DOOROPEN,DOORCLOSED,HULLOPEN,HULLCLOSED,
               AC_READY,TSAT,ASSIGNUSER,ISOTPDISMISS,PAX_BOOKED,PAX_CHECKED,PAX_BOARDED,CARGO,CAPACITY,STATUS,
               LAST_BAGMSGS_PROC,SEMAPHORE,CALL_SIGN,OFFBLOCK,TOBT,CTOT,HAUL_IND,AC_OWNER,SERV_TYP_COD,NI,MVT,TSAT_STATUS,CHG_REASON
        FROM osusr_uuk_flt_info
        WHERE STD BETWEEN %s AND %s
        """
        return pd.read_sql(query, self.engine, params=(start_date, end_date))
    
    def _extract_flight_plans_table(self, start_date, end_date, callsign=None):
        """Extract data from flight plans table"""
        query = """
        SELECT * FROM (
            SELECT ID,FLP_FILE_NAME,PLAN_NBR,CATEGORY,COMP_AT,STD,DEP_DATE,CALLSIGN,CAPTAIN,FLT_NBR,DEPARTURE_AIRP,ARRIVAL_AIRP,TAIL,
                   AIRCRAFT_ICAO_TYPE,AIRLINE_SPEC,AUTHOR_DISPACHER,TELEX_ADDR,DISPATCH_OFFICER,PERFORMANCE_FACTOR,AVERAGE_FUEL_FLOW,
                   TAXI_FUEL_FLOW,HOLDING_FUEL_FLOW,ROUTE_NAME,ROUTE_OPTIMIZATION,CLIMB_PROC,CLIMB_CI,CRUISE_PROC,CRUISE_CI,DESCENT_PROC,
                   DESCENT_CI,ROUTE_DESC,GROUND_DIST,AIR_DIST,GREAT_CIRC,TRIP_FUEL,TRIP_DURATION,CONTINGENCY_FUEL,CONTINGENCY_FUEL_DURATION,
                   TAKE_OFF_FUEL,TAKE_OFF_DURATION,TAXI_FUEL,TAXI_FUEL_DURATION,BLOCK_FUEL,BLOCK_FUEL_DURATION,LANDING_FUEL,ARRIVAL_FUEL,
                   DRY_OP_WEIGHT,LOAD_WEIGHT,ZERO_FUEL_WEIGHT,ZERO_FUEL_W_LIMIT,TAKEOFF_WEIGHT,TAKEOFF_W_OP_LIMIT,TAKEOFF_W_STRUCT_LIMIT,
                   LANDING_WEIGHT,LANDING_W_OP_LIMIT,LANDING_W_STRUCT_LIMIT,TS, TAXI_OUT_TIME, TAXI_IN_TIME, FLIGHT_TIME
            FROM osusr_fam_fp_arinc633 
            WHERE STR_TO_DATE(
                SUBSTRING_INDEX(SUBSTRING_INDEX(FLP_FILE_NAME, '.', 3), '.', -1),
                '%%d%%b%%Y'
            ) BETWEEN %s AND %s
            """ + (" AND CALLSIGN = %s" if callsign is not None else "") + """
            ORDER BY ID DESC
            LIMIT 1
        ) sub
        """
        
        params = [start_date, end_date]
        if callsign is not None:
            params.append(callsign)
            
        return pd.read_sql(query, self.engine, params=tuple(params))
    
    def _extract_waypoints_table(self, start_date, end_date, flt_file_name=None):
        """Extract data from waypoints table"""
        query = """
        SELECT ID,FLP_FILE_NAME,SEQUENCE,WAYPOINTID,WAYPOINT_NAME,LATITUDE,LONGITUDE,COORDINATES,AIRWAY_TYPE,AIRWAY,ALTITUDE,
               MINIMUM_SAFE_ALT,SEG_WIND_DIRECTION,SEG_WIND_SPEED,SEG_TEMPERATURE,SEG_ISA_DEVIATION,SPEED,TROPOPAUSE,TRUE_AIR_SPEED,
               MACH_NUMBER,IND_AIR_SPEED,GROUND_SPPED,OUTBOUND_TRACK_TRUE,OUTBOUND_TRACK_MAGNETIC,SEG_TRACK_TRUE,SEG_TRACK_MAGNETIC,
               GROUND_DISTANCE,AIR_DISTANCE,REMAIN_GROUND_DIST,REMAIN_AIR_DIST,TIME_FROM_PREV_WAY,CUMULATIVE_FLIGHT_TIME,REMAIN_FLIGHT_TIME,
               BURNOFF_WEIGHT,CUMULATIVE_BURNOFF_W,ESTIMATED_WEIGHT,AIRCRAFT_WEIGHT,FUELON_ON_BOAR,MINIMUM_FUEL_ONBOARD,AIRSPA_AIRSPACE_ICAO,
               AIRSPA_AIRSPACE_NAME,TS
        FROM osusr_fam_fp_arinc633_wp 
        WHERE STR_TO_DATE(
            SUBSTRING_INDEX(SUBSTRING_INDEX(FLP_FILE_NAME, '.', 3), '.', -1),
            '%%d%%b%%Y'
            ) BETWEEN %s AND %s
            """ + (" AND FLP_FILE_NAME = %s" if flt_file_name is not None else "")
            
        params = [start_date, end_date]
        if flt_file_name is not None:
            params.append(flt_file_name)
            
        return pd.read_sql(query, self.engine, params=tuple(params))
    
    def _extract_mel_table(self, start_date, end_date, flt_file_name=None):
        """Extract data from MEL table"""
        query = """
        SELECT ID,FLP_FILE_NAME,REFERENCE_ID,TITLE,TS
        FROM osusr_fam_fp_arinc633_mel 
        WHERE STR_TO_DATE(
            SUBSTRING_INDEX(SUBSTRING_INDEX(FLP_FILE_NAME, '.', 3), '.', -1),
            '%%d%%b%%Y'
        ) BETWEEN %s AND %s
        """ + (" AND FLP_FILE_NAME = %s" if flt_file_name is not None else "")
        
        params = [start_date, end_date]
        if flt_file_name is not None:
            params.append(flt_file_name)
            
        return pd.read_sql(query, self.engine, params=tuple(params))
    
    def _extract_acars_table(self, start_date, end_date, callsign=None, std_utc=None):
        """Extract data from ACARS table"""
        # Extend date range for ACARS data (2 days before and after)
        start_date_extended = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
        end_date_extended = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
        
        std_utc_end = std_utc + pd.Timedelta(hours=12) if pd.notnull(std_utc) else None
        query = """
        SELECT ID,ADDRESS1,ADDRESS2,DATE,FLIGHT,FLIGHTSUFFIX,AIRCRAFT,ETA,DESTINATION,FOB,LOCATION1,LOCATION2,CODE,
               LATITUDE,LONGITUDE,ALTITUDE,GROUNDSPEED,TEMPERATURE,WINDDIRECTION,WINDSPEED,AIRSPEED,REPORTTIME,FUEL,TS
        FROM osusr_fam_tap_acars_plane
        WHERE REPORTTIME BETWEEN %s AND %s
        """ + (" AND FLIGHT = %s" if callsign is not None else "") + (" AND REPORTTIME BETWEEN %s AND %s" if std_utc is not None else "")
        
        params = [start_date_extended, end_date_extended]
        if callsign is not None:
            params.append(callsign)
        if std_utc is not None:
            params.append(std_utc)
            params.append(std_utc_end)
            
        return pd.read_sql(query, self.engine, params=tuple(params))
    
    def _extract_equipments_table(self):
        """Extract data from equipments table"""
        query = "SELECT ID, BODYTYPE, EQUIPTYPE, EQUIPTYPE2 FROM osusr_uuk_equiptype"
        return pd.read_sql(query, self.engine)
    
    def _extract_aircrafts_table(self):
        """Extract data from aircrafts table"""
        query = "SELECT ACREGISTRATION, EQUIPTYPEID FROM osusr_uuk_registrations"
        return pd.read_sql(query, self.engine)
    
    def _extract_stations_table(self, stationFrom=None, stationTo=None):
        """Extract data from stations table"""
        query = """
        SELECT STATION, TIMEDIFF_MINUTES, DAYS_NUM 
        FROM view_utc_time_diff_yearly""" + (" WHERE STATION IN (%s, %s)" if stationFrom is not None and stationTo is not None else "")

        params = []
        if stationFrom is not None:
            params.append(stationFrom)
        if stationTo is not None:
            params.append(stationTo)
        return pd.read_sql(query, self.engine, params=tuple(params))
    
    
    def _join_all_data(self, flights_df, flight_plans_df, waypoints_df, mel_df, 
                      acars_df, equipments_df, aircrafts_df, stations_df):
        """Join all data sources together"""
        def parse_dt(val):
            try:
                return pd.to_datetime(val, errors='coerce')
            except Exception:
                return pd.NaT
        
        # Start with flights as base
        combined_df = flights_df.copy()

        std_local = parse_dt(combined_df.get('STD').iloc[0])
        from_iata = combined_df.get('FROM_IATA').iloc[0]
        from_timediff_df = stations_df[(stations_df['STATION'] == from_iata) & (stations_df['DAYS_NUM'] == std_local.day_of_year)]
        from_timediff = from_timediff_df.get('timediff_minutes').iloc[0]
        std_utc = std_local - timedelta(minutes=int(from_timediff) if from_timediff is not None else 0)
        combined_df['FROM_TIMEDIFF'] = int(from_timediff) if from_timediff is not None else 0
        combined_df['STD_UTC'] = std_utc
        acars_callsign = combined_df.get('CALL_SIGN').iloc[0].replace('TAP', 'TP')
        combined_df['ACARS_CALLSIGN'] = acars_callsign
        sta_local = parse_dt(combined_df.get('STA').iloc[0])
        to_iata = combined_df.get('TO_IATA').iloc[0]
        to_timediff_df = stations_df[(stations_df['STATION'] == to_iata) & (stations_df['DAYS_NUM'] == sta_local.day_of_year)]
        to_timediff = to_timediff_df.get('timediff_minutes').iloc[0]
        combined_df['TO_TIMEDIFF'] = int(to_timediff) if to_timediff is not None else 0
        
        
        # Join flight plans on CALL_SIGN and STD
        if not flight_plans_df.empty:
            # Rename flight plan columns with fp_ prefix
            fp_rename_dict = {
                'FLP_FILE_NAME': 'fp_FLP_FILE_NAME',
                'CAPTAIN': 'fp_CAPTAIN',
                'AIRCRAFT_ICAO_TYPE': 'fp_AIRCRAFT_ICAO_TYPE',
                'AIRLINE_SPEC': 'fp_AIRLINE_SPEC',
                'PERFORMANCE_FACTOR': 'fp_PERFORMANCE_FACTOR',
                'ROUTE_NAME': 'fp_ROUTE_NAME',
                'ROUTE_OPTIMIZATION': 'fp_ROUTE_OPTIMIZATION',
                'CRUISE_CI': 'fp_CRUISE_CI',
                'CLIMB_PROC': 'fp_CLIMB_PROC',
                'CRUISE_PROC': 'fp_CRUISE_PROC',
                'DESCENT_PROC': 'fp_DESCENT_PROC',
                'GREAT_CIRC': 'fp_GREAT_CIRC',
                'ZERO_FUEL_WEIGHT': 'fp_ZERO_FUEL_WEIGHT'
            }
            flight_plans_renamed = flight_plans_df.rename(columns=fp_rename_dict)
            
            combined_df = combined_df.merge(
                flight_plans_renamed, 
                left_on=['CALL_SIGN', 'STD_UTC'], 
                right_on=['CALLSIGN', 'STD'], 
                how='left',
                suffixes=('', '_fp')
            )
        
        # Join aircraft equipment data
        if not aircrafts_df.empty and not equipments_df.empty:
            aircraft_equipment = aircrafts_df.merge(
                equipments_df, 
                left_on='EQUIPTYPEID', 
                right_on='ID', 
                how='left'
            )
            # Rename equipment columns with eq_ prefix
            eq_rename_dict = {
                'BODYTYPE': 'eq_BODYTYPE',
                'EQUIPTYPE': 'eq_EQUIPMENTTYPE',
                'EQUIPTYPE2': 'eq_EQUIPMENTTYPE2'
            }
            aircraft_equipment_renamed = aircraft_equipment.rename(columns=eq_rename_dict)
            
            combined_df = combined_df.merge(
                aircraft_equipment_renamed, 
                left_on='AC_REGISTRATION', 
                right_on='ACREGISTRATION', 
                how='left'
            )
        
        # Process waypoints data - create wp1_ to wp50_ features
        if not waypoints_df.empty:
            waypoint_features = self._process_waypoints_data(waypoints_df)
            combined_df = combined_df.merge(
                waypoint_features, 
                left_on='fp_FLP_FILE_NAME', 
                right_on='FLP_FILE_NAME', 
                how='left'
            )
        
        # Process ACARS data - create ac1_ to ac20_ features  
        if not acars_df.empty:
            acars_features = self._process_acars_data(acars_df)
            combined_df = combined_df.merge(
                acars_features, 
                left_on='ACARS_CALLSIGN', 
                right_on='FLIGHT', 
                how='left'
            )
        
        return combined_df
    
    def _process_waypoints_data(self, waypoints_df):
        """Process waypoints data to create wp1_ to wp50_ features"""
        waypoints_cols = ['SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE']
        
        # Create all columns at once to avoid fragmentation
        feature_data = {}
        feature_data['FLP_FILE_NAME'] = waypoints_df.get('FLP_FILE_NAME').iloc[0]
        for base_col in waypoints_cols:
            for i, (_, row) in enumerate(waypoints_df.iterrows(), 1):
                col_name = f'wp{i}_{base_col}'
                feature_data[col_name] = row[base_col]
        
        # Create DataFrame with all columns at once
        waypoint_features = pd.DataFrame([feature_data], index=[0])
                
        return waypoint_features
    
    def _process_acars_data(self, acars_df):
        """Process ACARS data to create acars1_ to acars20_ features"""
        acars_cols = ['FLIGHT''WINDDIRECTION', 'WINDSPEED']
        
        # Create all columns at once to avoid fragmentation
        feature_data = {}
        for base_col in acars_cols:
            for i, (_, row) in enumerate(acars_df.iterrows(), 1):
                col_name = f'acars{i}_{base_col}'
                feature_data[col_name] = row[base_col]
        
        # Create DataFrame with all columns at once
        acars_features = pd.DataFrame([feature_data], index=[0])
                
        return acars_features
    
    def _create_model_features(self, combined_df):
        """Create features in the format expected by the XGBoost model"""
        
        # Define all required columns as per model metadata requirements
        all_cols = [
            'OPERATOR', 'FLT_NR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA', 'STD', 'ETD', 'ATD', 'STA', 'ETA',
            'FROM_STAND', 'TO_STAND', 'AC_READY', 'TSAT', 'TOBT', 'CTOT', 'CALL_SIGN', 'SERV_TYP_COD', 'MVT',
            'fp_CAPTAIN', 'fp_AIRCRAFT_ICAO_TYPE', 'fp_AIRLINE_SPEC', 'fp_PERFORMANCE_FACTOR', 'fp_ROUTE_NAME',
            'fp_ROUTE_OPTIMIZATION', 'fp_CRUISE_CI', 'fp_CLIMB_PROC', 'fp_CRUISE_PROC', 'fp_DESCENT_PROC',
            'fp_GREAT_CIRC', 'fp_ZERO_FUEL_WEIGHT', 'eq_BODYTYPE', 'eq_EQUIPMENTTYPE', 'eq_EQUIPMENTTYPE2',
            'actual_taxi_out', 'actual_airborne', 'actual_taxi_in', 'AET', 'OFFBLOCK', 'ATA', 'ONBLOCK',
            'FROM_TERMINAL', 'TO_TERMINAL', 'FROM_GATE', 'CAPTAIN', 'AIRCRAFT_ICAO_TYPE','AIRLINE_SPEC', 
            'PERFORMANCE_FACTOR', 'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CRUISE_CI', 'CLIMB_PROC','CRUISE_PROC', 
            'DESCENT_PROC', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT', 'BODYTYPE', 'EQUIPTYPE', 'EQUIPTYPE2'
        ]
        
        # Ensure all columns are present
        for col in all_cols:
            if col not in combined_df.columns:
                combined_df[col] = None
        
        # Add waypoint columns (wp1_ to wp50_)
        waypoints_cols = ['SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE']
        for base in waypoints_cols:
            for i in range(1, 51):
                col = f'wp{i}_{base}'
                if col not in combined_df.columns:
                    combined_df[col] = None
        
        # Add ACARS columns (acars1_ to acars20_)
        acars_cols = ['WINDDIRECTION', 'WINDSPEED']
        for base in acars_cols:
            for i in range(1, 21):
                col = f'acars{i}_{base}'
                if col not in combined_df.columns:
                    combined_df[col] = None
        
        # Select only the columns needed for the model
        model_cols = all_cols + [f'wp{i}_{base}' for base in waypoints_cols for i in range(1, 51)] + \
                    [f'acars{i}_{base}' for base in acars_cols for i in range(1, 21)]
        
        features_df = combined_df[model_cols].copy()
        
        # Fill missing values with -1 as in the original preprocessing
        features_df = features_df.fillna(-1)
        
        return features_df


def get_flight_data_for_prediction(start_date=None, end_date=None, days_back=2, flight_id=None):
    """
    Convenience function to extract flight data for prediction
    
    Args:
        start_date: Start date for data extraction (YYYY-MM-DD format)
        end_date: End date for data extraction (YYYY-MM-DD format)
        days_back: Number of days back from today if dates not provided
        flight_id: Specific flight ID from OSUSR_UUK_FLT_INFO table
        
    Returns:
        pandas.DataFrame: Processed data ready for XGBoost model
    """
    extractor = DatabaseExtractor()
    return extractor.extract_flight_data(start_date, end_date, days_back, flight_id)


def get_flight_hist_aeteet(flight_id):
    """
    Convenience function to extract flight data for prediction
    
    Args:
        start_date: Start date for data extraction (YYYY-MM-DD format)
        end_date: End date for data extraction (YYYY-MM-DD format)
        days_back: Number of days back from today if dates not provided
        flight_id: Specific flight ID from OSUSR_UUK_FLT_INFO table
        
    Returns:
        pandas.DataFrame: Processed data ready for XGBoost model
    """
    extractor = DatabaseExtractor()
    return extractor.extract_flight_hist_aeteet(flight_id)
