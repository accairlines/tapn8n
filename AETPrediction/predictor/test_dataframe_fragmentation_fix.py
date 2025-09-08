"""
Test script to verify the DataFrame fragmentation fix works correctly
"""

import os
import sys
import django
from pathlib import Path
import pandas as pd
import warnings

# Add the predictor directory to Python path
predictor_dir = Path(__file__).parent
sys.path.insert(0, str(predictor_dir))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aet_predictor.settings')
django.setup()

from aet_api.db_extractor import DatabaseExtractor

def test_waypoints_data_processing():
    """Test that waypoints data processing doesn't cause fragmentation warnings"""
    print("Testing waypoints data processing...")
    
    # Create sample waypoints data
    sample_waypoints = pd.DataFrame({
        'SEG_WIND_DIRECTION': [270, 280, 290],
        'SEG_WIND_SPEED': [15, 20, 18],
        'SEG_TEMPERATURE': [-10, -12, -8]
    })
    
    extractor = DatabaseExtractor()
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        result = extractor._process_waypoints_data(sample_waypoints)
        
        # Check for fragmentation warnings
        fragmentation_warnings = [warning for warning in w 
                                if "fragmented" in str(warning.message).lower()]
        
        if fragmentation_warnings:
            print(f"⚠ Found {len(fragmentation_warnings)} fragmentation warning(s)")
            for warning in fragmentation_warnings:
                print(f"  - {warning.message}")
            return False
        else:
            print("✓ No fragmentation warnings detected")
    
    # Verify the result structure
    expected_cols = ['wp1_SEG_WIND_DIRECTION', 'wp1_SEG_WIND_SPEED', 'wp1_SEG_TEMPERATURE',
                    'wp2_SEG_WIND_DIRECTION', 'wp2_SEG_WIND_SPEED', 'wp2_SEG_TEMPERATURE',
                    'wp3_SEG_WIND_DIRECTION', 'wp3_SEG_WIND_SPEED', 'wp3_SEG_TEMPERATURE']
    
    missing_cols = [col for col in expected_cols if col not in result.columns]
    if missing_cols:
        print(f"⚠ Missing expected columns: {missing_cols}")
        return False
    else:
        print("✓ All expected columns are present")
    
    print(f"✓ Result shape: {result.shape}")
    print(f"✓ Result columns: {len(result.columns)}")
    
    return True

def test_acars_data_processing():
    """Test that ACARS data processing doesn't cause fragmentation warnings"""
    print("\nTesting ACARS data processing...")
    
    # Create sample ACARS data
    sample_acars = pd.DataFrame({
        'WINDDIRECTION': [180, 190, 200],
        'WINDSPEED': [25, 30, 28]
    })
    
    extractor = DatabaseExtractor()
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        result = extractor._process_acars_data(sample_acars)
        
        # Check for fragmentation warnings
        fragmentation_warnings = [warning for warning in w 
                                if "fragmented" in str(warning.message).lower()]
        
        if fragmentation_warnings:
            print(f"⚠ Found {len(fragmentation_warnings)} fragmentation warning(s)")
            for warning in fragmentation_warnings:
                print(f"  - {warning.message}")
            return False
        else:
            print("✓ No fragmentation warnings detected")
    
    # Verify the result structure
    expected_cols = ['acars1_WINDDIRECTION', 'acars1_WINDSPEED',
                    'acars2_WINDDIRECTION', 'acars2_WINDSPEED',
                    'acars3_WINDDIRECTION', 'acars3_WINDSPEED']
    
    missing_cols = [col for col in expected_cols if col not in result.columns]
    if missing_cols:
        print(f"⚠ Missing expected columns: {missing_cols}")
        return False
    else:
        print("✓ All expected columns are present")
    
    print(f"✓ Result shape: {result.shape}")
    print(f"✓ Result columns: {len(result.columns)}")
    
    return True

def test_performance_comparison():
    """Test performance improvement by timing the operations"""
    print("\nTesting performance improvement...")
    
    import time
    
    # Create larger sample data to see performance difference
    large_waypoints = pd.DataFrame({
        'SEG_WIND_DIRECTION': list(range(50)),
        'SEG_WIND_SPEED': list(range(50)),
        'SEG_TEMPERATURE': list(range(50))
    })
    
    extractor = DatabaseExtractor()
    
    # Time the operation
    start_time = time.time()
    result = extractor._process_waypoints_data(large_waypoints)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"✓ Processing time for 50 waypoints: {processing_time:.4f} seconds")
    print(f"✓ Result shape: {result.shape}")
    
    return True

if __name__ == "__main__":
    print("Testing DataFrame fragmentation fix...\n")
    
    success1 = test_waypoints_data_processing()
    success2 = test_acars_data_processing()
    success3 = test_performance_comparison()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed! DataFrame fragmentation fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
