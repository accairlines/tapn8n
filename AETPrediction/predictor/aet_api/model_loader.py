import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .preprocess import preprocess_flight_data

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles model loading and prediction"""
    
    def __init__(self, model_path='/app/models/model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model trained on: {self.model_data['training_date']}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def reload_model(self):
        """Reload model (called after retraining)"""
        self.load_model()
    
    def predict(self, flight_data):
        """Make predictions for a flight"""
        if not self.model_data:
            raise ValueError("Model not loaded")
        
        # Prepare features
        features = preprocess_flight_data(flight_data)
        
        logger.debug(f"Features: {str(features)}")
        # Scale features
        features_scaled = self.model_data['scaler'].transform(features)
        
        # Make predictions
        predictions = {}
        for target, model in self.model_data['models'].items():
            pred = model.predict(features_scaled)[0]
            # Map model target names to API response names
            predictions['delta'] = pred
        
        return predictions
    