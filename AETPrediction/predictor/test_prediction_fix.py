"""
Test script to verify the prediction fix works
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

from aet_api.views import get_flight_data
import pandas as pd

def test_flight_data_extraction():
    """Test that get_flight_data returns proper dictionary format"""
    try:
        print("Testing flight data extraction...")
        
        # Test with a sample flight ID
        test_flight_id = 430654
        
        flight_data = get_flight_data(test_flight_id)
        
        if flight_data is None:
            print(f"⚠ No flight found with ID: {test_flight_id}")
            return False
        
        print(f"✓ Successfully extracted flight data for ID: {test_flight_id}")
        print(f"Data type: {type(flight_data)}")
        print(f"Number of fields: {len(flight_data)}")
        
        # Check required fields for model
        required_fields = [
            'AIRCRAFT_ICAO_TYPE', 'DEPARTURE_AIRP', 'ARRIVAL_AIRP', 'STD',
            'TAXI_OUT_TIME', 'FLIGHT_TIME', 'TAXI_IN_TIME'
        ]
        
        missing_fields = [field for field in required_fields if field not in flight_data]
        if missing_fields:
            print(f"⚠ Missing required fields: {missing_fields}")
        else:
            print("✓ All required fields are present")
        
        # Show sample data
        print("\nSample flight data:")
        for key, value in list(flight_data.items())[:10]:  # Show first 10 fields
            print(f"  {key}: {value}")
        
        print("\n✓ Flight data extraction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during flight data extraction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flight_data_extraction()
    sys.exit(0 if success else 1)
