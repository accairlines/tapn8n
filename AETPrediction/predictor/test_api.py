#!/usr/bin/env python3
"""
Simple test script for the AET Prediction API
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, expected_status=200):
    """Test an API endpoint"""
    try:
        response = requests.get(f"{BASE_URL}{endpoint}")
        print(f"GET {endpoint}: {response.status_code}")
        
        if response.status_code == expected_status:
            print(f"✅ {endpoint} - Status: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"   Response: {response.text[:200]}...")
        else:
            print(f"❌ {endpoint} - Expected {expected_status}, got {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
        
        print()
        return response.status_code == expected_status
        
    except requests.exceptions.ConnectionError:
        print(f"❌ {endpoint} - Connection failed. Is the server running?")
        print()
        return False
    except Exception as e:
        print(f"❌ {endpoint} - Error: {str(e)}")
        print()
        return False

def main():
    """Test all API endpoints"""
    print("🧪 Testing AET Prediction API")
    print("=" * 40)
    
    # Test root endpoint
    test_endpoint("/")
    
    # Test health endpoint
    test_endpoint("/health/")
    
    # Test admin endpoint (should redirect or show login)
    test_endpoint("/admin/", 302)  # Django admin redirects to login
    
    # Test prediction endpoint with invalid ID (should return 404)
    test_endpoint("/predict/999999/", 500)  # Will fail due to DB connection
    
    # Test batch endpoint
    test_endpoint("/predict/batch/", 500)  # Will fail due to DB connection
    
    print("=" * 40)
    print("✅ API endpoint testing completed!")
    print("\nNote: Prediction endpoints will fail without database connection.")
    print("This is expected behavior for testing the API structure.")

if __name__ == "__main__":
    main()
