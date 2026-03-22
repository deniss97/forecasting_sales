"""
Demand Forecasting Model Module

This module provides the demand forecasting model using CatBoost.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Optional
import joblib


class DemandForecaster:
    """
    A class for demand forecasting using CatBoost.
    
    Attributes:
        model: CatBoost regression model
        feature_names: List of feature names
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the DemandForecaster.
        
        Args:
            **kwargs: Additional arguments for CatBoostRegressor
        """
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 8,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'early_stopping_rounds': 50,
            'verbose': 100,
            'random_seed': 42
        }
        default_params.update(kwargs)
        
        self.model = CatBoostRegressor(**default_params)
        self.feature_names = []
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feature_names: List[str] = None) -> 'DemandForecaster':
        """
        Fit the model on training data.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation target
            feature_names: List of feature names
            
        Returns:
            Fitted DemandForecaster instance
        """
        self.feature_names = feature_names or []
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        self.model.fit(X, y, eval_set=eval_set)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        return {
            'rmse': rmse,
            'rmse_quantity': rmse
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances.
        
        Returns:
            Array of feature importances
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.feature_importances_
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importances as a DataFrame.
        
        Returns:
            DataFrame with feature names and importances
        """
        importance = self.get_feature_importance()
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False)
    
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        self.model.save_model(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> 'DemandForecaster':
        """
        Load a model from a file.
        
        Args:
            path: Path to the model file
            
        Returns:
            DemandForecaster instance with loaded model
        """
        self.model.load_model(path)
        self._fitted = True
        print(f"Model loaded from {path}")
        return self
    
    def get_model(self):
        """
        Get the underlying CatBoost model.
        
        Returns:
            CatBoostRegressor instance
        """
        return self.model
