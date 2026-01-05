import pickle
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import logging
from .preprocess import preprocess_flight_data, preprocess_flight_data_for_ft_transformer
from .ft_transformer import FTTransformer

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles model loading and prediction for both XGBoost and FT-Transformer"""
    
    def __init__(self, model_path='/app/models/model.pkl'):
        self.model_path = model_path
        self.model_data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained models from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model trained on: {self.model_data['training_date']}")
            
            # Load FT-Transformer models if they exist
            if 'ft_transformer_models' in self.model_data:
                logger.info("Found FT-Transformer models in saved data, loading...")
                self._load_ft_transformer_models()
                if hasattr(self, 'ft_transformer_trainers') and self.ft_transformer_trainers:
                    logger.info(f"Successfully loaded {len(self.ft_transformer_trainers)} FT-Transformer model(s)")
                else:
                    logger.warning("FT-Transformer models found in data but failed to load")
            else:
                logger.info("No FT-Transformer models found in saved data. Available keys: " + 
                           str(list(self.model_data.keys())))
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_ft_transformer_models(self):
        """Load FT-Transformer models from saved state dicts"""
        self.ft_transformer_trainers = {}
        
        try:
            if 'ft_transformer_models' not in self.model_data:
                logger.warning("No 'ft_transformer_models' key found in model data")
                return
            
            ft_models = self.model_data['ft_transformer_models']
            if not ft_models:
                logger.warning("ft_transformer_models dictionary is empty")
                return
            
            for target, ft_data in ft_models.items():
                if ft_data is None:
                    logger.warning(f"FT-Transformer data for target '{target}' is None, skipping")
                    continue
                
                try:
                    config = ft_data['model_config']
                    # Create model instance
                    model = FTTransformer(
                        num_numerical=config['num_numerical'],
                        num_categories=config['num_categories'],
                        d_token=config['d_token'],
                        n_layers=config['n_layers'],
                        n_heads=config['n_heads'],
                        d_ff=config['d_ff'],
                        dropout=config['dropout']
                    ).to(self.device)
                    
                    # Load state dict
                    model.load_state_dict(ft_data['model_state_dict'])
                    model.eval()
                    
                    # Create trainer-like object for prediction
                    class FTModelWrapper:
                        def __init__(self, model, scaler, device):
                            self.model = model
                            self.numerical_scaler = scaler
                            self.device = device
                        
                        def predict(self, X_numerical, X_categorical):
                            self.model.eval()
                            
                            # Convert to numpy if needed
                            if isinstance(X_numerical, pd.DataFrame):
                                X_numerical = X_numerical.values
                            if isinstance(X_categorical, pd.DataFrame):
                                X_categorical = X_categorical.values
                            
                            # Scale numerical features
                            if X_numerical is not None and len(X_numerical) > 0:
                                X_numerical = self.numerical_scaler.transform(X_numerical)
                            
                            # Convert to tensors
                            if X_numerical is not None and len(X_numerical) > 0:
                                X_num_tensor = torch.FloatTensor(X_numerical).to(self.device)
                            else:
                                X_num_tensor = None
                            
                            if X_categorical is not None and len(X_categorical) > 0:
                                X_cat_tensor = torch.LongTensor(X_categorical).to(self.device)
                            else:
                                X_cat_tensor = None
                            
                            with torch.no_grad():
                                predictions = self.model(X_num_tensor, X_cat_tensor)
                            
                            return predictions.cpu().numpy().flatten()
                    
                    self.ft_transformer_trainers[target] = FTModelWrapper(
                        model, ft_data['numerical_scaler'], self.device
                    )
                    logger.info(f"FT-Transformer model loaded successfully for target: {target}")
                except Exception as e:
                    logger.error(f"Failed to load FT-Transformer model for target '{target}': {str(e)}")
                    logger.error(f"Error details: {type(e).__name__}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
        except Exception as e:
            logger.error(f"Error loading FT-Transformer models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def reload_model(self):
        """Reload model (called after retraining)"""
        self.load_model()
    
    def predict(self, flight_data, model_type='ensemble'):
        """
        Make predictions for a flight
        
        Args:
            flight_data: Flight data dictionary
            model_type: 'xgb', 'ft_transformer', or 'ensemble' (default: 'ensemble')
        
        Returns:
            predictions: Dictionary with 'delta' key
        """
        if not self.model_data:
            raise ValueError("Model not loaded")
        
        predictions = {}
        
        if model_type in ['xgb', 'ensemble']:
            # Prepare features for XGBoost
            features = preprocess_flight_data(flight_data)
            
            # Scale features
            features_scaled = self.model_data['scaler'].transform(features)
            
            # Make XGBoost predictions
            xgb_pred = {}
            for target, model in self.model_data['models'].items():
                pred = model.predict(features_scaled)[0]
                xgb_pred['delta'] = pred
            
            if model_type == 'xgb':
                return xgb_pred
            predictions['xgb'] = xgb_pred['delta']
        
        if model_type in ['ft_transformer', 'ensemble']:
            # Check if FT-Transformer models are available
            if not hasattr(self, 'ft_transformer_trainers') or 'delta' not in self.ft_transformer_trainers:
                error_msg = "FT-Transformer model not available. "
                if not hasattr(self, 'ft_transformer_trainers'):
                    error_msg += "Model trainers not initialized. "
                elif 'delta' not in self.ft_transformer_trainers:
                    error_msg += f"Model for 'delta' target not found. Available targets: {list(getattr(self, 'ft_transformer_trainers', {}).keys())}. "
                error_msg += "Please train the model first using /train_model/ endpoint."
                
                if model_type == 'ft_transformer':
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                # If ensemble and FT-Transformer not available, use only XGBoost
                logger.warning(f"FT-Transformer not available, using only XGBoost. {error_msg}")
                return predictions if predictions else xgb_pred
            
            # Prepare features for FT-Transformer
            categorical_df, numerical_df = preprocess_flight_data_for_ft_transformer(flight_data)
            
            # Make FT-Transformer predictions
            ft_trainer = self.ft_transformer_trainers['delta']
            ft_pred = ft_trainer.predict(
                numerical_df if not numerical_df.empty else None,
                categorical_df if not categorical_df.empty else None
            )
            
            if model_type == 'ft_transformer':
                return {'delta': ft_pred[0] if len(ft_pred) > 0 else 0.0}
            predictions['ft_transformer'] = ft_pred[0] if len(ft_pred) > 0 else 0.0
        
        # Ensemble: average predictions
        if model_type == 'ensemble':
            if 'xgb' in predictions and 'ft_transformer' in predictions:
                predictions['delta'] = (predictions['xgb'] + predictions['ft_transformer']) / 2.0
            elif 'xgb' in predictions:
                predictions['delta'] = predictions['xgb']
            elif 'ft_transformer' in predictions:
                predictions['delta'] = predictions['ft_transformer']
        
        # Return in expected format (always return 'delta' key)
        return {'delta': predictions.get('delta', predictions.get('xgb', predictions.get('ft_transformer', 0.0)))}
    