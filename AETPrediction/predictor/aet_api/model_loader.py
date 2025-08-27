import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Set base path from environment or default
DATA_PATH = os.environ.get('AET_DATA_PATH')
LOG_PATH = os.environ.get('AET_LOG_PATH')
MODEL_PATH = os.environ.get('AET_MODEL_PATH')

class ModelLoader:
    """Handles model loading and prediction"""
    
    def __init__(self):
        self.model_path = MODEL_PATH
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            logging.info(f"Model loaded successfully from {self.model_path}")
            logging.info(f"Model trained on: {self.model_data['training_date']}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    
    def reload_model(self):
        """Reload model (called after retraining)"""
        self.load_model()
    
    def predict(self, flight_data):
        """Make predictions for a flight"""
        if not self.model_data:
            raise ValueError("Model not loaded")
        
        # Prepare features
        features = self._prepare_features(flight_data)
        
        # Scale features
        features_scaled = self.model_data['scaler'].transform(features)
        
        # Make predictions
        predictions = {}
        for target, model in self.model_data['models'].items():
            pred = model.predict(features_scaled)[0]
            # Ensure non-negative predictions
            pred = max(0, pred)
            
            # Map model target names to API response names
            if target == 'actual_taxi_out':
                predictions['taxi_out'] = pred
            elif target == 'actual_airborne':
                predictions['airborne'] = pred
            elif target == 'actual_taxi_in':
                predictions['taxi_in'] = pred
        
        return predictions
    
    def _prepare_features(self, flight_data):
        """Prepare feature vector from flight data"""
        features = pd.DataFrame([flight_data])
        
        # Extract basic features
        feature_dict = {
            'aircraft_type_code': self._encode_aircraft_type(flight_data.get('AIRCRAFT_ICAO_TYPE', 'A320')),
            'dep_airport_code': self._encode_airport(flight_data.get('DEPARTURE_AIRP', 'UNK')),
            'arr_airport_code': self._encode_airport(flight_data.get('ARRIVAL_AIRP', 'UNK')),
            'planned_taxi_out': flight_data.get('TAXI_OUT_TIME', 15),
            'planned_flight_time': flight_data.get('FLIGHT_TIME', 60),
            'planned_taxi_in': flight_data.get('TAXI_IN_TIME', 10),
            'hour_of_day': pd.to_datetime(flight_data.get('STD')).hour,
            'day_of_week': pd.to_datetime(flight_data.get('STD')).dayofweek,
            'is_weekend': int(pd.to_datetime(flight_data.get('STD')).dayofweek >= 5),
            'route_distance': flight_data.get('route_distance', 500),  # Default if not calculated
            'max_flight_time': flight_data.get('max_flight_time', 90),
            'mel_count': flight_data.get('mel_count', 0),
            'avg_wind_speed': flight_data.get('avg_wind_speed', 10),
            'max_wind_speed': flight_data.get('max_wind_speed', 20),
            'avg_temperature': flight_data.get('avg_temperature', 15),
            'max_altitude': flight_data.get('max_altitude', 35000)
        }
        
        # Create DataFrame with features in correct order
        feature_names = self.model_data['feature_names']
        features_df = pd.DataFrame([feature_dict])[feature_names]
        
        return features_df
    
    def _encode_aircraft_type(self, aircraft_type):
        """Simple encoding for aircraft types"""
        # In production, use the same encoding as training
        common_types = {
            'A320': 0, 'A321': 1, 'A319': 2, 'B737': 3, 'B738': 4,
            'A330': 5, 'B777': 6, 'A350': 7, 'B787': 8
        }
        return common_types.get(aircraft_type, 99)  # 99 for unknown
    
    def _encode_airport(self, airport_code):
        """Simple encoding for airports"""
        # In production, use the same encoding as training
        # This is simplified - you'd want to match training encoding
        return hash(airport_code) % 1000
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model_data is not None
    
    def get_model_info(self):
        """Get model information"""
        if not self.model_data:
            return None
        
        return {
            'training_date': self.model_data.get('training_date'),
            'feature_count': len(self.model_data.get('feature_names', [])),
            'targets': list(self.model_data.get('models', {}).keys())
        } 