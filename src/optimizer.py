"""
Revenue Optimization Module

This module provides the revenue optimization engine for price optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class RevenueOptimizer:
    """
    A class for revenue optimization through price adjustments.
    
    Attributes:
        demand_model: Trained demand forecasting model
        label_encoders: Dictionary of label encoders
        scaler: Scaler for numerical features
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        pricing_strategy: DataFrame with pricing strategy
    """
    
    def __init__(self, demand_model, label_encoders: Dict, scaler,
                 numerical_cols: List[str], categorical_cols: List[str],
                 pricing_strategy: pd.DataFrame = None):
        """
        Initialize the RevenueOptimizer.
        
        Args:
            demand_model: Trained demand forecasting model
            label_encoders: Dictionary of label encoders
            scaler: Scaler for numerical features
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            pricing_strategy: Optional pricing strategy DataFrame
        """
        self.demand_model = demand_model
        self.label_encoders = label_encoders
        self.scaler = scaler
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.pricing_strategy = pricing_strategy
    
    def preprocess_features(self, X_sample: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features using label encoding and scaling.
        
        Args:
            X_sample: Sample DataFrame to preprocess
            
        Returns:
            Preprocessed feature array
        """
        X_scenario = X_sample.copy()
        
        # Handle categorical features
        for col in self.categorical_cols:
            if col not in X_scenario.columns:
                X_scenario[col] = 'missing'
            
            X_scenario[col] = X_scenario[col].astype(str)
            le = self.label_encoders[col]
            known_categories = set(le.classes_)
            
            X_scenario[col] = X_scenario[col].apply(
                lambda x: x if x in known_categories 
                else 'missing' if 'missing' in known_categories 
                else le.classes_[0]
            )
            X_scenario[col] = le.transform(X_scenario[col])
        
        # Handle numerical features
        for col in self.numerical_cols:
            if col not in X_scenario.columns:
                X_scenario[col] = 0.0
        
        # Scale numerical features
        X_numerical = X_scenario[self.numerical_cols]
        X_numerical_scaled = self.scaler.transform(X_numerical)
        
        # Combine features
        X_transformed = np.hstack([
            X_numerical_scaled,
            X_scenario[self.categorical_cols].values
        ])
        
        return X_transformed
    
    def predict_demand(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict demand using the trained model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predicted demand values
        """
        X_transformed = self.preprocess_features(X)
        return self.demand_model.predict(X_transformed)
    
    def calculate_revenue(self, quantities: np.ndarray, 
                          prices: np.ndarray) -> float:
        """
        Calculate total revenue.
        
        Args:
            quantities: Array of quantities
            prices: Array of prices
            
        Returns:
            Total revenue
        """
        return np.sum(quantities * prices)
    
    def optimize_price_single_item(
        self, X_sample: pd.DataFrame, base_price: float,
        item_id: str = None, store_id: int = None,
        price_change_range: np.ndarray = None,
        verbose: bool = False
    ) -> Dict:
        """
        Find optimal price for a single item that maximizes revenue.
        
        Args:
            X_sample: Sample features DataFrame
            base_price: Current base price
            item_id: Optional item ID
            store_id: Optional store ID
            price_change_range: Array of price changes to test
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with optimization results
        """
        if price_change_range is None:
            price_change_range = np.arange(-0.3, 0.35, 0.05)
        
        revenues = []
        quantities = []
        prices_tested = []
        current_revenue = None
        current_quantity = None
        
        if verbose:
            print("\n--- Price Optimization Analysis ---")
            print(f"Base price: {base_price:.2f}")
            print(f"Testing {len(price_change_range)} price scenarios:")
            print(f"{'Change':<8} {'New Price':<10} {'Predicted Demand':<15} {'Revenue':<10}")
            print("-" * 45)
        
        for price_change in price_change_range:
            X_scenario = X_sample.copy()
            new_price = base_price * (1 + price_change)
            
            predicted_quantities = self.predict_demand(X_scenario)
            predicted_quantity = np.mean(predicted_quantities)
            
            predicted_revenues = predicted_quantities * new_price
            total_revenue = np.sum(predicted_revenues)
            
            revenues.append(total_revenue)
            quantities.append(predicted_quantity)
            prices_tested.append(new_price)
            
            if verbose:
                print(f"{price_change:>7.0%}  {new_price:>9.2f}  "
                      f"{predicted_quantity:>14.2f}  {total_revenue:>9.2f}")
            
            if abs(price_change) < 1e-10:
                current_revenue = total_revenue
                current_quantity = predicted_quantity
        
        optimal_idx = np.argmax(revenues)
        optimal_price_change = price_change_range[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        optimal_quantity = quantities[optimal_idx]
        optimal_price = prices_tested[optimal_idx]
        
        if verbose:
            print("-" * 45)
            print(f"OPTIMAL: {optimal_price_change:>7.0%} -> "
                  f"{optimal_price:.2f} rub, revenue = {optimal_revenue:.2f}")
            if current_revenue is not None:
                improvement = (optimal_revenue - current_revenue) / current_revenue * 100
                print(f"Improvement: {improvement:.2f}% "
                      f"(from {current_revenue:.2f} to {optimal_revenue:.2f})")
        
        revenue_improvement = None
        if current_revenue is not None and current_revenue > 0:
            revenue_improvement = (optimal_revenue - current_revenue) / current_revenue * 100
        elif current_revenue is None:
            X_baseline = X_sample.copy()
            baseline_quantities = self.predict_demand(X_baseline)
            current_revenue = np.sum(baseline_quantities * base_price)
            if current_revenue > 0:
                revenue_improvement = (optimal_revenue - current_revenue) / current_revenue * 100
        
        return {
            'item_id': item_id,
            'store_id': store_id,
            'base_price': base_price,
            'optimal_price_change': optimal_price_change,
            'optimal_new_price': base_price * (1 + optimal_price_change),
            'optimal_revenue': optimal_revenue,
            'optimal_quantity': optimal_quantity,
            'current_revenue': current_revenue,
            'current_quantity': current_quantity,
            'revenue_improvement': revenue_improvement
        }
    
    def optimize_price_with_elasticity(
        self, X_sample: pd.DataFrame, base_price: float,
        item_id: str = None, store_id: int = None
    ) -> Dict:
        """
        Optimize price using elasticity-based strategy if available.
        
        Args:
            X_sample: Sample features DataFrame
            base_price: Current base price
            item_id: Optional item ID
            store_id: Optional store ID
            
        Returns:
            Dictionary with optimization results
        """
        if self.pricing_strategy is not None and item_id is not None:
            item_strategy = self.pricing_strategy[
                (self.pricing_strategy['item_id'] == item_id) &
                (self.pricing_strategy['store_id'] == store_id)
            ]
            
            if len(item_strategy) > 0:
                recommended_change = item_strategy['recommended_price_change'].iloc[0]
                new_price = base_price * (1 + recommended_change)
                
                predicted_quantity = np.mean(self.predict_demand(X_sample))
                projected_revenue = predicted_quantity * new_price
                
                baseline_quantities = self.predict_demand(X_sample)
                current_revenue = np.sum(baseline_quantities * base_price)
                revenue_improvement = None
                if current_revenue > 0:
                    revenue_improvement = (projected_revenue - current_revenue) / current_revenue * 100
                
                return {
                    'item_id': item_id,
                    'store_id': store_id,
                    'base_price': base_price,
                    'optimal_price_change': recommended_change,
                    'optimal_new_price': new_price,
                    'optimal_revenue': projected_revenue,
                    'optimal_quantity': predicted_quantity,
                    'current_revenue': current_revenue,
                    'revenue_improvement': revenue_improvement,
                    'method': 'elasticity_based'
                }
        
        return self.optimize_price_single_item(
            X_sample, base_price, item_id, store_id
        )
    
    def optimize_portfolio(
        self, X_samples: pd.DataFrame, base_prices: np.ndarray,
        item_ids: List[str] = None, store_ids: List[int] = None
    ) -> pd.DataFrame:
        """
        Optimize prices for a portfolio of items.
        
        Args:
            X_samples: DataFrame with sample features
            base_prices: Array of base prices
            item_ids: Optional list of item IDs
            store_ids: Optional list of store IDs
            
        Returns:
            DataFrame with optimization results
        """
        results = []
        
        for i in range(len(X_samples)):
            X_sample = X_samples.iloc[i:i+1]
            base_price = base_prices[i]
            item_id = item_ids[i] if item_ids is not None else None
            store_id = store_ids[i] if store_ids is not None else None
            
            result = self.optimize_price_with_elasticity(
                X_sample, base_price, item_id, store_id
            )
            results.append(result)
        
        return pd.DataFrame(results)
