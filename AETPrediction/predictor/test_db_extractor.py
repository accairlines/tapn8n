"""
Test script for database extractor
Run this to verify the database extraction works correctly
"""

import os
import sys
import django
from pathlib import Path

# Add the predictor directory to Python path
predictor_dir = Path(__file__).parent
sys.path.insert(0, str(predictor_dir))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aet_predictor.settings')
django.setup()

from aet_api.db_extractor import get_flight_data_for_prediction
import pandas as pd

def test_database_extraction():
    """Test the database extraction functionality"""
    try:
        print("Testing database extraction...")
        
        # Test 1: Extract data for the last 7 days
        print("\n=== Test 1: Date range extraction ===")
        df = get_flight_data_for_prediction(days_back=7)
        
        print(f"Successfully extracted {len(df)} records")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Test 2: Extract single flight (if we have data)
        if len(df) > 0:
            print("\n=== Test 2: Single flight extraction ===")
            # Get the first flight ID from the results
            first_flight_id = df['ID'].iloc[0] if 'ID' in df.columns else None
            
            if first_flight_id:
                print(f"Testing single flight extraction for ID: {first_flight_id}")
                single_df = get_flight_data_for_prediction(flight_id=first_flight_id)
                print(f"Single flight extraction: {len(single_df)} records")
                
                if len(single_df) > 0:
                    print("✓ Single flight extraction successful")
                else:
                    print("⚠ Single flight extraction returned no data")
            else:
                print("⚠ No flight ID found in data, skipping single flight test")
        else:
            print("\n=== Test 2: Single flight extraction ===")
            print("⚠ No data found for date range, skipping single flight test")
        
        # Check if we have the expected columns
        expected_base_cols = [
            'OFFBLOCK', 'MVT', 'ATA', 'ONBLOCK', 'ETA',
            'FROM_IATA', 'STD', 'CALL_SIGN', 'AC_REGISTRATION',
            'OPERATOR', 'FLT_NR', 'TO_IATA', 'ETD', 'ATD', 'STA', 'FROM_STAND', 'TO_STAND',
            'AC_READY', 'TSAT', 'TOBT', 'CTOT', 'SERV_TYP_COD',
            'FROM_TERMINAL', 'TO_TERMINAL', 'FROM_GATE', 'CAPTAIN', 'AIRCRAFT_ICAO_TYPE',
            'AIRLINE_SPEC', 'PERFORMANCE_FACTOR', 'ROUTE_NAME', 'ROUTE_OPTIMIZATION', 'CRUISE_CI', 'CLIMB_PROC',
            'CRUISE_PROC', 'DESCENT_PRO', 'GREAT_CIRC', 'ZERO_FUEL_WEIGHT',
            'BODYTYPE', 'EQUIPTYPE', 'EQUIPTYPE2'
        ]
        
        missing_cols = [col for col in expected_base_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing base columns: {missing_cols}")
        else:
            print("✓ All expected base columns are present")
        
        # Check waypoint columns
        waypoint_cols = [f'wp{i}_{base}' for base in ['SEG_WIND_DIRECTION', 'SEG_WIND_SPEED', 'SEG_TEMPERATURE'] for i in range(1, 51)]
        missing_wp_cols = [col for col in waypoint_cols if col not in df.columns]
        if missing_wp_cols:
            print(f"Warning: Missing waypoint columns: {len(missing_wp_cols)} out of {len(waypoint_cols)}")
        else:
            print("✓ All waypoint columns are present")
        
        # Check ACARS columns
        acars_cols = [f'ac{i}_{base}' for base in ['WINDDIRECTION', 'WINDSPEED'] for i in range(1, 21)]
        missing_acars_cols = [col for col in acars_cols if col not in df.columns]
        if missing_acars_cols:
            print(f"Warning: Missing ACARS columns: {len(missing_acars_cols)} out of {len(acars_cols)}")
        else:
            print("✓ All ACARS columns are present")
        
        # Show sample data
        print("\nSample data (first 3 rows, first 10 columns):")
        if len(df) > 0:
            print(df.iloc[:3, :10].to_string())
        else:
            print("No data to display")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        print(f"\nTotal missing values: {missing_count}")
        
        # Check data types
        print(f"\nData types summary:")
        print(df.dtypes.value_counts())
        
        print("\n✓ Database extraction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during database extraction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database_extraction()
    sys.exit(0 if success else 1)
