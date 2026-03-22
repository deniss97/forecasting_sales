#!/usr/bin/env python3
"""
Demand Forecasting with Price Sensitivity - Part 1
Data Loading, Preprocessing, and Feature Engineering

This script loads data, performs preprocessing, and engineers features
for demand forecasting with price sensitivity analysis.
"""

import numpy as np
import pandas as pd
import os
import warnings
import holidays

warnings.simplefilter(action='ignore', category=FutureWarning)

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'notebooks')

print(f"Project directory: {PROJECT_DIR}")
print(f"Data directory: {DATA_DIR}")

# Data files configuration
files = {
    "sales": os.path.join(DATA_DIR, "sales.csv"),
    "online": os.path.join(DATA_DIR, "online.csv"),
    "markdowns": os.path.join(DATA_DIR, "markdowns.csv"),
    "price_history": os.path.join(DATA_DIR, "price_history.csv"),
    "discounts_history": os.path.join(DATA_DIR, "discounts_history.csv"),
    "actual_matrix": os.path.join(DATA_DIR, "actual_matrix.csv"),
    "catalog": os.path.join(DATA_DIR, "catalog.csv"),
    "stores": os.path.join(DATA_DIR, "stores.csv"),
    "test": os.path.join(DATA_DIR, "test.csv"),
    "sample_submission": os.path.join(DATA_DIR, "sample_submission.csv"),
}


def read_csv(name, **kwargs):
    """Load CSV file by name."""
    if name not in files:
        raise ValueError(f"Unknown file: {name}")
    fp = files[name]
    if not os.path.exists(fp):
        print(f"Warning: {fp} not found, skipping...")
        return None
    
    # Handle index column for specific files
    if 'index_col' in kwargs and kwargs['index_col'] == 0:
        # Check if first column looks like an index
        df = pd.read_csv(fp, index_col=0, **{k: v for k, v in kwargs.items() if k != 'index_col'})
    else:
        df = pd.read_csv(fp, **kwargs)
    return df


def optimizing_dtypes(df, nameCSV):
    """Optimize data types to reduce memory usage."""
    bytesInOneMB = 1048576
    print(f"{nameCSV}: from {round(df.memory_usage().sum()/bytesInOneMB, 2)}MB", end="")

    # Convert float64 to float32
    col_float64 = df.select_dtypes(include=["float64"]).columns.tolist()
    for col in col_float64:
        df[col] = df[col].astype("float32")

    # Convert int64 to smaller int types
    col_int64 = df.select_dtypes(include=["int64"]).columns.tolist()
    for col in col_int64:
        uniq = df[col].dropna().unique()
        if len(uniq) <= 2 and set(uniq).issubset({0, 1}):
            df[col] = df[col].astype("int8")
        else:
            df[col] = df[col].astype("int32")

    print(f" reduced to {round(df.memory_usage().sum()/bytesInOneMB, 2)}MB")
    return df


def date_features(dataframe, date_col="date"):
    """Add date-based features."""
    dataframe = dataframe.copy()
    dataframe["day"] = dataframe[date_col].dt.day
    dataframe["month"] = dataframe[date_col].dt.month
    dataframe["year"] = dataframe[date_col].dt.year
    dataframe["dayofweek"] = dataframe[date_col].dt.dayofweek
    dataframe["week"] = dataframe[date_col].dt.isocalendar().week.astype(int)
    return dataframe


def transform2cyclic(dataframe):
    """Add cyclic features for temporal data."""
    dataframe = dataframe.copy()
    
    # Day of month cyclic features
    dataframe["day_sin"] = np.sin(2 * np.pi * (dataframe["day"] - 1) / 31)
    dataframe["day_cos"] = np.cos(2 * np.pi * (dataframe["day"] - 1) / 31)
    
    # Day of week cyclic features
    dataframe["dayofweek_sin"] = np.sin(2 * np.pi * dataframe["dayofweek"] / 6)
    dataframe["dayofweek_cos"] = np.cos(2 * np.pi * dataframe["dayofweek"] / 6)
    
    # Week of year cyclic features
    dataframe["week_sin"] = np.sin(2 * np.pi * (dataframe["week"] - 1) / 52)
    dataframe["week_cos"] = np.cos(2 * np.pi * (dataframe["week"] - 1) / 52)
    
    # Month cyclic features
    dataframe["month_sin"] = np.sin(2 * np.pi * (dataframe["month"] - 1) / 12)
    dataframe["month_cos"] = np.cos(2 * np.pi * (dataframe["month"] - 1) / 12)
    
    return dataframe


def get_weekends(dataframe):
    """Add weekend feature."""
    dataframe = dataframe.copy()
    dataframe["is_weekend"] = dataframe["dayofweek"].isin([4, 5, 6])
    return dataframe


def get_sundays(dataframe):
    """Add Sunday feature."""
    dataframe = dataframe.copy()
    dataframe["is_sunday"] = dataframe["dayofweek"].eq(6)
    return dataframe


def get_holidays(dataframe, date_col="date", years=None, country="RU"):
    """Add holiday features."""
    dataframe = dataframe.copy()
    if years is None:
        years = [2022, 2023, 2024]
    
    country_holidays = holidays.CountryHoliday(country, years=years)
    dataframe["holidays"] = dataframe[date_col].isin(country_holidays.keys())
    return dataframe


def get_seasons(dataframe):
    """Add seasonal features based on month."""
    dataframe = dataframe.copy()
    
    def get_season(month):
        if month in [4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        elif month in [9, 10]:
            return 3  # Autumn
        else:
            return 4  # Winter
    
    dataframe["season"] = dataframe["month"].apply(get_season)
    return dataframe


def add_lag_features(df, group_cols=None, price_col="price_base", lags=None):
    """Add lag features for price sensitivity analysis."""
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


def main():
    """Main function to run Part 1 preprocessing."""
    print("=" * 60)
    print("PART 1: Data Loading and Preprocessing")
    print("=" * 60)
    
    # Load available data files
    print("\nLoading data files...")
    df_sales = read_csv("sales", index_col=0)
    df_online = read_csv("online", index_col=0)
    df_markdowns = read_csv("markdowns", index_col=0)
    df_price_history = read_csv("price_history", index_col=0)
    df_discounts_history = read_csv("discounts_history", index_col=0)
    df_actual_matrix = read_csv("actual_matrix", index_col=0)
    df_catalog = read_csv("catalog", index_col=0)
    df_stores = read_csv("stores", index_col=0)
    df_test = read_csv("test", sep=";", index_col="row_id")
    df_sample_submission = read_csv("sample_submission", index_col=0)
    
    # Check which files were loaded
    loaded_files = {
        "sales": df_sales,
        "online": df_online,
        "markdowns": df_markdowns,
        "price_history": df_price_history,
        "discounts_history": df_discounts_history,
        "actual_matrix": df_actual_matrix,
        "catalog": df_catalog,
        "stores": df_stores,
        "test": df_test,
        "sample_submission": df_sample_submission,
    }
    
    for name, df in loaded_files.items():
        if df is not None:
            print(f"  ✓ {name}: {df.shape}")
        else:
            print(f"  ✗ {name}: Not found")
    
    # Optimize dtypes
    print("\nOptimizing data types...")
    for name, df in loaded_files.items():
        if df is not None:
            loaded_files[name] = optimizing_dtypes(df, f"{name}.csv")
    
    # Check if we have the essential files
    if df_online is None:
        print("\nError: online.csv is required but not found!")
        return None, None
    
    # Clean online data
    print("\nCleaning data...")
    df_online["date"] = pd.to_datetime(df_online["date"])
    df_online["price_base"] = df_online["sum_total"] / df_online["quantity"]
    df_online = df_online.sort_values(["date", "item_id", "store_id"])
    
    mask = (
        (df_online["quantity"] <= 0) | 
        (df_online["price_base"] <= 0) | 
        (df_online["sum_total"] <= 0) | 
        ~np.isfinite(df_online["price_base"])
    )
    df_online = df_online.drop(df_online[mask].index, axis=0)
    
    # Use online data as main dataset if sales is not available
    if df_sales is not None:
        df_sales["date"] = pd.to_datetime(df_sales["date"])
        df_sales["price_base"] = df_sales["sum_total"] / df_sales["quantity"]
        df_sales = df_sales.sort_values(["date", "item_id", "store_id"])
        
        mask = (
            (df_sales["quantity"] <= 0) | 
            (df_sales["price_base"] <= 0) | 
            (df_sales["sum_total"] <= 0) | 
            ~np.isfinite(df_sales["price_base"])
        )
        df_sales = df_sales.drop(df_sales[mask].index, axis=0)
        
        # Merge sales and online
        print("Merging sales and online data...")
        df_online_renamed = df_online.rename(columns={
            "price_base": "price_base_online", 
            "sum_total": "sum_total_online"
        })
        df_online_renamed["online"] = True
        
        df = pd.merge(df_sales, df_online_renamed, on=["date", "item_id", "store_id"], how="outer", suffixes=("_x", "_y"))
        df["quantity"] = df[["quantity_x", "quantity_y"]].sum(axis=1)
        df = df[df_sales.columns.to_list()]
        df = df.fillna(0)
    else:
        print("Using online data as main dataset...")
        df = df_online.copy()
    
    # Add date features
    print("\nAdding features...")
    df = date_features(df)
    df = transform2cyclic(df)
    df = get_weekends(df)
    df = get_sundays(df)
    df = get_holidays(df)
    df = get_seasons(df)
    
    # Add lag features
    print("Adding lag features...")
    df = add_lag_features(df)
    
    # Process test data
    if df_test is not None:
        df_test["date"] = pd.to_datetime(df_test["date"])
        df_test = date_features(df_test)
        df_test = transform2cyclic(df_test)
        df_test = get_weekends(df_test)
        df_test = get_sundays(df_test)
        df_test = get_holidays(df_test)
        df_test = get_seasons(df_test)
    
    # Save processed data
    print("\nSaving processed data...")
    df.to_pickle(os.path.join(OUTPUT_DIR, 'df_processed_part1.pkl'))
    
    if df_test is not None:
        df_test.to_pickle(os.path.join(OUTPUT_DIR, 'df_test_processed_part1.pkl'))
        print(f"  ✓ Saved df_processed_part1.pkl and df_test_processed_part1.pkl")
    else:
        print(f"  ✓ Saved df_processed_part1.pkl (no test data)")
    
    print(f"\n✅ Part 1 completed!")
    print(f"  - Main data shape: {df.shape}")
    print(f"  - Saved to: {OUTPUT_DIR}")
    
    return df, df_test


if __name__ == "__main__":
    df, df_test = main()
