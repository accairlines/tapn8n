#!/usr/bin/env python3
"""
Test script to verify the updated field structure matches model metadata requirements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aet_api.db_extractor import DatabaseExtractor
import pandas as pd

def test_field_structure():
    """Test that the field structure matches the required model metadata"""
    
    # Create a sample DataFrame with the expected structure
    sample_data = {
        'OPERATOR': ['TAP'],
        'FLT_NR': ['1234'],
        'AC_REGISTRATION': ['CS-TNA'],
        'FROM_IATA': ['LIS'],
        'TO_IATA': ['MAD'],
        'STD': ['2024-01-01 10:00:00'],
        'ETD': ['2024-01-01 10:05:00'],
        'ATD': ['2024-01-01 10:10:00'],
        'STA': ['2024-01-01 11:30:00'],
        'ETA': ['2024-01-01 11:35:00'],
        'FROM_STAND': ['A1'],
        'TO_STAND': ['B2'],
        'AC_READY': ['2024-01-01 09:45:00'],
        'TSAT': ['2024-01-01 10:00:00'],
        'TOBT': ['2024-01-01 10:00:00'],
        'CTOT': ['2024-01-01 10:00:00'],
        'CALL_SIGN': ['TAP1234'],
        'SERV_TYP_COD': ['J'],
        'MVT': ['D'],
        'fp_CAPTAIN': ['John Doe'],
        'fp_AIRCRAFT_ICAO_TYPE': ['A320'],
        'fp_AIRLINE_SPEC': ['TAP'],
        'fp_PERFORMANCE_FACTOR': [1.0],
        'fp_ROUTE_NAME': ['LIS-MAD'],
        'fp_ROUTE_OPTIMIZATION': ['OPT'],
        'fp_CRUISE_CI': [0],
        'fp_CLIMB_PROC': ['STD'],
        'fp_CRUISE_PROC': ['STD'],
        'fp_DESCENT_PROC': ['STD'],
        'fp_GREAT_CIRC': [500],
        'fp_ZERO_FUEL_WEIGHT': [50000],
        'eq_BODYTYPE': ['NARROW'],
        'eq_EQUIPMENTTYPE': ['JET'],
        'eq_EQUIPMENTTYPE2': ['TURBO'],
        'actual_taxi_out': [15],
        'actual_airborne': [60],
        'actual_taxi_in': [10],
        'AET': [85],
        'OFFBLOCK': ['2024-01-01 10:10:00'],
        'ATA': ['2024-01-01 11:40:00'],
        'ONBLOCK': ['2024-01-01 11:45:00']
    }
    
    # Add waypoint fields
    for i in range(1, 51):
        sample_data[f'wp{i}_SEG_WIND_DIRECTION'] = [270]
        sample_data[f'wp{i}_SEG_WIND_SPEED'] = [15]
        sample_data[f'wp{i}_SEG_TEMPERATURE'] = [-20]
    
    # Add ACARS fields
    for i in range(1, 21):
        sample_data[f'acars{i}_WINDDIRECTION'] = [270]
        sample_data[f'acars{i}_WINDSPEED'] = [15]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Test the DatabaseExtractor's _create_model_features method
    extractor = DatabaseExtractor()
    
    try:
        # This should not raise an exception
        features_df = extractor._create_model_features(df)
        
        print("✅ Field structure test passed!")
        print(f"Generated {len(features_df.columns)} features")
        print(f"Sample columns: {list(features_df.columns)[:10]}...")
        
        # Check that all required fields are present
        required_fields = [
            'OPERATOR', 'FLT_NR', 'AC_REGISTRATION', 'FROM_IATA', 'TO_IATA', 'STD', 'ETD', 'ATD', 'STA', 'ETA',
            'FROM_STAND', 'TO_STAND', 'AC_READY', 'TSAT', 'TOBT', 'CTOT', 'CALL_SIGN', 'SERV_TYP_COD', 'MVT',
            'fp_CAPTAIN', 'fp_AIRCRAFT_ICAO_TYPE', 'fp_AIRLINE_SPEC', 'fp_PERFORMANCE_FACTOR', 'fp_ROUTE_NAME',
            'fp_ROUTE_OPTIMIZATION', 'fp_CRUISE_CI', 'fp_CLIMB_PROC', 'fp_CRUISE_PROC', 'fp_DESCENT_PROC',
            'fp_GREAT_CIRC', 'fp_ZERO_FUEL_WEIGHT', 'eq_BODYTYPE', 'eq_EQUIPMENTTYPE', 'eq_EQUIPMENTTYPE2',
            'actual_taxi_out', 'actual_airborne', 'actual_taxi_in', 'AET', 'OFFBLOCK', 'ATA', 'ONBLOCK'
        ]
        
        missing_fields = [field for field in required_fields if field not in features_df.columns]
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
        else:
            print("✅ All required fields are present")
            
        # Check waypoint fields
        waypoint_fields = [f'wp{i}_SEG_WIND_DIRECTION' for i in range(1, 51)]
        missing_waypoint_fields = [field for field in waypoint_fields if field not in features_df.columns]
        if missing_waypoint_fields:
            print(f"❌ Missing waypoint fields: {len(missing_waypoint_fields)}")
        else:
            print("✅ All waypoint fields are present")
            
        # Check ACARS fields
        acars_fields = [f'acars{i}_WINDDIRECTION' for i in range(1, 21)]
        missing_acars_fields = [field for field in acars_fields if field not in features_df.columns]
        if missing_acars_fields:
            print(f"❌ Missing ACARS fields: {len(missing_acars_fields)}")
        else:
            print("✅ All ACARS fields are present")
            
        return True
        
    except Exception as e:
        print(f"❌ Field structure test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_field_structure()
    sys.exit(0 if success else 1)
