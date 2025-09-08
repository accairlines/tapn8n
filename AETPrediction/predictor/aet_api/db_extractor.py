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

logger = logging.getLogger(__name__)

class DatabaseExtractor:
    """Extract and process data from the database for AET prediction"""
    
    def __init__(self):
        self.db_connection = connections['default']
        
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
            
            # Extract data from all tables
            flt_nr = flights_df['FLT_NR'].iloc[0] if not flights_df.empty else None
            flight_plans_df = self._extract_flight_plans_table(start_date, end_date, flt_nr=flt_nr)
            
            flt_file_name = flight_plans_df['FLP_FILE_NAME'].iloc[0] if not flight_plans_df.empty else None
            waypoints_df = self._extract_waypoints_table(start_date, end_date, flt_file_name=flt_file_name)
            mel_df = self._extract_mel_table(start_date, end_date, flt_file_name=flt_file_name)
            
            callsign = flights_df['CALL_SIGN'].iloc[0].replace('TAP', 'TP') if not flights_df.empty and 'CALL_SIGN' in flights_df.columns else None
            acars_df = self._extract_acars_table(start_date, end_date, callsign=callsign)
            equipments_df = self._extract_equipments_table()
            aircrafts_df = self._extract_aircrafts_table()
            stations_df = self._extract_stations_table()
            
            # Join all data together
            combined_df = self._join_all_data(
                flights_df, flight_plans_df, waypoints_df, mel_df, 
                acars_df, equipments_df, aircrafts_df, stations_df
            )
            
            # Create features for XGBoost model
            features_df = self._create_model_features(combined_df)
            
            logger.info(f"Successfully extracted {len(features_df)} records with {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting flight data: {str(e)}")
            raise
    
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
        result = pd.read_sql(query, self.db_connection, params=[flight_id])
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
        return pd.read_sql(query, self.db_connection, params=[start_date, end_date])
    
    def _extract_flight_plans_table(self, start_date, end_date, flt_nr=None):
        """Extract data from flight plans table"""
        query = """
        SELECT ID,FLP_FILE_NAME,PLAN_NBR,CATEGORY,COMP_AT,STD,DEP_DATE,CALLSIGN,CAPTAIN,FLT_NBR,DEPARTURE_AIRP,ARRIVAL_AIRP,TAIL,
               AIRCRAFT_ICAO_TYPE,AIRLINE_SPEC,AUTHOR_DISPACHER,TELEX_ADDR,DISPATCH_OFFICER,PERFORMANCE_FACTOR,AVERAGE_FUEL_FLOW,
               TAXI_FUEL_FLOW,HOLDING_FUEL_FLOW,ROUTE_NAME,ROUTE_OPTIMIZATION,CLIMB_PROC,CLIMB_CI,CRUISE_PROC,CRUISE_CI,DESCENT_PROC,
               DESCENT_CI,ROUTE_DESC,GROUND_DIST,AIR_DIST,GREAT_CIRC,TRIP_FUEL,TRIP_DURATION,CONTINGENCY_FUEL,CONTINGENCY_FUEL_DURATION,
               TAKE_OFF_FUEL,TAKE_OFF_DURATION,TAXI_FUEL,TAXI_FUEL_DURATION,BLOCK_FUEL,BLOCK_FUEL_DURATION,LANDING_FUEL,ARRIVAL_FUEL,
               DRY_OP_WEIGHT,LOAD_WEIGHT,ZERO_FUEL_WEIGHT,ZERO_FUEL_W_LIMIT,TAKEOFF_WEIGHT,TAKEOFF_W_OP_LIMIT,TAKEOFF_W_STRUCT_LIMIT,
               LANDING_WEIGHT,LANDING_W_OP_LIMIT,LANDING_W_STRUCT_LIMIT,TS,NULL AS TAXI_OUT_TIME,NULL AS TAXI_IN_TIME,NULL AS FLIGHT_TIME
        FROM osusr_fam_fp_arinc633 
        WHERE STR_TO_DATE(
            SUBSTRING_INDEX(SUBSTRING_INDEX(FLP_FILE_NAME, '.', 3), '.', -1),
            '%%d%%b%%Y'
        ) BETWEEN %s AND %s
        """ + (" AND FLT_NBR = %s" if flt_nr is not None else "")
        
        params = [start_date, end_date]
        if flt_nr is not None:
            params.append(flt_nr)
            
        return pd.read_sql(query, self.db_connection, params=params)
    
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
            
        return pd.read_sql(query, self.db_connection, params=params)
    
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
            
        return pd.read_sql(query, self.db_connection, params=params)
    
    def _extract_acars_table(self, start_date, end_date, callsign=None):
        """Extract data from ACARS table"""
        # Extend date range for ACARS data (2 days before and after)
        start_date_extended = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
        end_date_extended = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
        
        query = """
        SELECT ID,ADDRESS1,ADDRESS2,DATE,FLIGHT,FLIGHTSUFFIX,AIRCRAFT,ETA,DESTINATION,FOB,LOCATION1,LOCATION2,CODE,
               LATITUDE,LONGITUDE,ALTITUDE,GROUNDSPEED,TEMPERATURE,WINDDIRECTION,WINDSPEED,AIRSPEED,REPORTTIME,FUEL,TS
        FROM osusr_fam_tap_acars_plane
        WHERE REPORTTIME BETWEEN %s AND %s
        """ + (" AND FLIGHT = %s" if callsign is not None else "")
        
        params = [start_date_extended, end_date_extended]
        if callsign is not None:
            params.append(callsign)
            
        return pd.read_sql(query, self.db_connection, params=params)
    
    def _extract_equipments_table(self):
        """Extract data from equipments table"""
        query = "SELECT ID, BODYTYPE, EQUIPTYPE, EQUIPTYPE2 FROM equipments"
        return pd.read_sql(query, self.db_connection)
    
    def _extract_aircrafts_table(self):
        """Extract data from aircrafts table"""
        query = "SELECT ACREGISTRATION, EQUIPTYPEID FROM aircrafts"
        return pd.read_sql(query, self.db_connection)
    
    def _extract_stations_table(self):
        """Extract data from stations table"""
        query = "SELECT STATION, TIMEDIFF_MINUTES, DAY_NUM FROM stations_utc"
        return pd.read_sql(query, self.db_connection)
    
    def _join_all_data(self, flights_df, flight_plans_df, waypoints_df, mel_df, 
                      acars_df, equipments_df, aircrafts_df, stations_df):
        """Join all data sources together"""
        
        # Start with flights as base
        combined_df = flights_df.copy()
        
        # Join flight plans on CALL_SIGN and STD
        if not flight_plans_df.empty:
            combined_df = combined_df.merge(
                flight_plans_df, 
                left_on=['CALL_SIGN', 'STD'], 
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
            combined_df = combined_df.merge(
                aircraft_equipment[['ACREGISTRATION', 'BODYTYPE', 'EQUIPTYPE', 'EQUIPTYPE2']], 
                left_on='AC_REGISTRATION', 
                right_on='ACREGISTRATION', 
                how='left'
            )
        
        # Join stations data
        if not stations_df.empty:
            combined_df = combined_df.merge(
                stations_df, 
                left_on='FROM_IATA', 
                right_on='STATION', 
                how='left',
                suffixes=('', '_from_station')
            )
            combined_df = combined_df.merge(
                stations_df, 
                left_on='TO_IATA', 
                right_on='STATION', 
                how='left',
                suffixes=('', '_to_station')
            )
        
        # Process waypoints data - create wp1_ to wp50_ features
        if not waypoints_df.empty:
            waypoint_features = self._process_waypoints_data(waypoints_df)
            combined_df = combined_df.merge(
                waypoint_features, 
                left_on='CALL_SIGN', 
                right_on='CALL_SIGN', 
                how='left'
            )
        
        # Process ACARS data - create ac1_ to ac20_ features  
        if not acars_df.empty:
            acars_features = self._process_acars_data(acars_df)
            combined_df = combined_df.merge(
                acars_features, 
                left_on='CALL_SIGN', 
                right_on='CALL_SIGN', 
                how='left'
            )
        
        return combined_df
    
    def _process_waypoints_data(self, waypoints_df):
        """Process waypoints data to create wp1_ to wp50_ features"""
        waypoint_features = {}
        waypoints_cols = ['SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE']
        
        # Group by FLP_FILE_NAME and get up to 50 waypoints per flight
        for flp_file, group in waypoints_df.groupby('FLP_FILE_NAME'):
            # Sort by SEQUENCE to get ordered waypoints
            group = group.sort_values('SEQUENCE').head(50)
            
            for base_col in waypoints_cols:
                for i, (_, row) in enumerate(group.iterrows(), 1):
                    col_name = f'wp{i}_{base_col}'
                    if col_name not in waypoint_features:
                        waypoint_features[col_name] = {}
                    waypoint_features[col_name][flp_file] = row[base_col]
        
        # Convert to DataFrame
        waypoint_df = pd.DataFrame(waypoint_features).reset_index()
        waypoint_df.rename(columns={'index': 'FLP_FILE_NAME'}, inplace=True)
        
        # Get CALL_SIGN from flight plans to join properly
        # This is a simplified approach - you might need to adjust based on your data structure
        waypoint_df['CALL_SIGN'] = waypoint_df['FLP_FILE_NAME']  # Adjust this mapping as needed
        
        return waypoint_df
    
    def _process_acars_data(self, acars_df):
        """Process ACARS data to create ac1_ to ac20_ features"""
        acars_features = {}
        acars_cols = ['WINDDIRECTION', 'WINDSPEED']
        
        # Group by FLIGHT and get up to 20 ACARS messages per flight
        for flight, group in acars_df.groupby('FLIGHT'):
            # Sort by REPORTTIME to get chronological order
            group = group.sort_values('REPORTTIME').head(20)
            
            for base_col in acars_cols:
                for i, (_, row) in enumerate(group.iterrows(), 1):
                    col_name = f'ac{i}_{base_col}'
                    if col_name not in acars_features:
                        acars_features[col_name] = {}
                    acars_features[col_name][flight] = row[base_col]
        
        # Convert to DataFrame
        acars_df_processed = pd.DataFrame(acars_features).reset_index()
        acars_df_processed.rename(columns={'index': 'FLIGHT'}, inplace=True)
        acars_df_processed['CALL_SIGN'] = acars_df_processed['FLIGHT']  # Adjust this mapping as needed
        
        return acars_df_processed
    
    def _create_model_features(self, combined_df):
        """Create features in the format expected by the XGBoost model"""
        
        # Define all required columns as in trainer/preprocess.py
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
        
        # Add ACARS columns (ac1_ to ac20_)
        acars_cols = ['WINDDIRECTION', 'WINDSPEED']
        for base in acars_cols:
            for i in range(1, 21):
                col = f'ac{i}_{base}'
                if col not in combined_df.columns:
                    combined_df[col] = None
        
        # Select only the columns needed for the model
        model_cols = all_cols + [f'wp{i}_{base}' for base in waypoints_cols for i in range(1, 51)] + \
                    [f'ac{i}_{base}' for base in acars_cols for i in range(1, 21)]
        
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
