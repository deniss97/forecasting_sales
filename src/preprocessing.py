"""
Data Preprocessing Module for Revenue Optimization System

This module provides utilities for feature engineering and data preprocessing.
"""

import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional


class DataPreprocessor:
    """
    A class for preprocessing data for demand forecasting.
    
    Attributes:
        label_encoders (Dict): Dictionary of label encoders for categorical features
        scaler (StandardScaler): Scaler for numerical features
        numerical_cols (List): List of numerical column names
        categorical_cols (List): List of categorical column names
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_cols = []
        self.categorical_cols = []
        self._fitted = False
    
    def add_date_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Add date-based features to the dataframe.
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with added date features
        """
        df = df.copy()
        df["day"] = df[date_col].dt.day
        df["month"] = df[date_col].dt.month
        df["year"] = df[date_col].dt.year
        df["dayofweek"] = df[date_col].dt.dayofweek
        df["week"] = df[date_col].dt.isocalendar().week.astype(int)
        return df
    
    def add_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclic features for temporal data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added cyclic features
        """
        df = df.copy()
        
        # Day of month cyclic features
        df["day_sin"] = np.sin(2 * np.pi * (df["day"] - 1) / 31)
        df["day_cos"] = np.cos(2 * np.pi * (df["day"] - 1) / 31)
        
        # Day of week cyclic features
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 6)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 6)
        
        # Week of year cyclic features
        df["week_sin"] = np.sin(2 * np.pi * (df["week"] - 1) / 52)
        df["week_cos"] = np.cos(2 * np.pi * (df["week"] - 1) / 52)
        
        # Month cyclic features
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
        
        return df
    
    def add_holiday_features(self, df: pd.DataFrame, date_col: str = "date", 
                             years: List[int] = None, country: str = "RU") -> pd.DataFrame:
        """
        Add holiday and weekend features.
        
        Args:
            df: Input DataFrame
            date_col: Name of the date column
            years: List of years to consider for holidays
            country: Country code for holidays
            
        Returns:
            DataFrame with added holiday features
        """
        df = df.copy()
        
        if years is None:
            years = [2022, 2023, 2024, 2025]
        
        # Weekend feature
        df["is_weekend"] = df["dayofweek"].isin([4, 5, 6])
        
        # Sunday feature
        df["is_sunday"] = df["dayofweek"].eq(6)
        
        # Holiday feature
        country_holidays = holidays.CountryHoliday(country, years=years)
        df["holidays"] = df[date_col].isin(country_holidays.keys())
        
        return df
    
    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal features based on month.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added seasonal features
        """
        df = df.copy()
        
        def get_season(month):
            if month in [4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            elif month in [9, 10]:
                return 3  # Autumn
            else:
                return 4  # Winter
        
        df["season"] = df["month"].apply(get_season)
        return df
    
    def add_lag_features(self, df: pd.DataFrame, 
                         group_cols: List[str] = None,
                         price_col: str = "price_base",
                         lags: List[int] = None) -> pd.DataFrame:
        """
        Add lag features for price sensitivity analysis.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by for lag calculation
            price_col: Column name for price
            lags: List of lag periods
            
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
        
        if group_cols is None:
            group_cols = ["item_id", "store_id"]
        
        if lags is None:
            lags = [1, 7, 30]
        
        # Sort for proper lag calculation
        df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)
        
        for lag in lags:
            df[f"price_lag_{lag}"] = df.groupby(group_cols)[price_col].shift(lag)
            df[f"price_change_{lag}"] = (df[price_col] - df[f"price_lag_{lag}"]) / df[f"price_lag_{lag}"]
        
        # Fill NaN values
        lag_cols = [col for col in df.columns if "lag" in col or "price_change" in col]
        df[lag_cols] = df[lag_cols].fillna(0)
        
        return df
    
    def optimize_dtypes(self, df: pd.DataFrame, name: str = "") -> pd.DataFrame:
        """
        Optimize data types to reduce memory usage.
        
        Args:
            df: Input DataFrame
            name: Optional name for logging
            
        Returns:
            DataFrame with optimized data types
        """
        bytes_in_one_mb = 1048576
        print(f"{name}: from {round(df.memory_usage().sum() / bytes_in_one_mb, 2)}MB", end="")
        
        # Convert float64 to float32
        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype("float32")
        
        # Convert int64 to smaller int types
        int_cols = df.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            uniq = df[col].dropna().unique()
            if len(uniq) <= 2 and set(uniq).issubset({0, 1}):
                df[col] = df[col].astype("int8")
            else:
                df[col] = df[col].astype("int32")
        
        print(f" reduced to {round(df.memory_usage().sum() / bytes_in_one_mb, 2)}MB")
        return df
    
    def fit(self, df: pd.DataFrame, 
            numerical_cols: List[str] = None,
            categorical_cols: List[str] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            
        Returns:
            Fitted DataPreprocessor instance
        """
        if numerical_cols is not None:
            self.numerical_cols = numerical_cols
        
        if categorical_cols is not None:
            self.categorical_cols = categorical_cols
        
        # Fit scaler on numerical columns
        if self.numerical_cols:
            self.scaler.fit(df[self.numerical_cols])
        
        # Fit label encoders on categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Handle unseen categories
                df_col = df[col].astype(str)
                le.fit(df_col)
                self.label_encoders[col] = le
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed feature array
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Handle categorical features
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = 'missing'
            
            df[col] = df[col].astype(str)
            le = self.label_encoders[col]
            known_categories = set(le.classes_)
            
            # Map unseen categories
            df[col] = df[col].apply(
                lambda x: x if x in known_categories else 'missing' if 'missing' in known_categories else le.classes_[0]
            )
            df[col] = le.transform(df[col])
        
        # Handle numerical features
        for col in self.numerical_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Scale numerical features
        X_numerical = df[self.numerical_cols]
        X_numerical_scaled = self.scaler.transform(X_numerical)
        
        # Combine features
        X_transformed = np.hstack([
            X_numerical_scaled,
            df[self.categorical_cols].values
        ])
        
        return X_transformed
    
    def fit_transform(self, df: pd.DataFrame,
                      numerical_cols: List[str] = None,
                      categorical_cols: List[str] = None) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            df: DataFrame to fit and transform
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            
        Returns:
            Transformed feature array
        """
        self.fit(df, numerical_cols, categorical_cols)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features after transformation.
        
        Returns:
            List of feature names
        """
        return self.numerical_cols + self.categorical_cols
