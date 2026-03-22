#!/usr/bin/env python3
"""
Demand Forecasting with Price Sensitivity - Part 2
Model Training

This script trains a CatBoost model for demand forecasting.
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
import logging
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, 'notebooks')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging with verbose option
verbose = '--verbose' in sys.argv or '-v' in sys.argv

log_file = os.path.join(LOGS_DIR, f'part2_model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
log_level = logging.DEBUG if verbose else logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Project directory: {PROJECT_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"Log file: {log_file}")
logger.info(f"Verbose mode: {verbose}")


def main():
    """Main function to run Part 2 model training."""
    logger.info("=" * 60)
    logger.info("PART 2: Model Training")
    logger.info("=" * 60)
    
    # Load preprocessed data from Part 1
    logger.info("Loading preprocessed data from Part 1...")
    df_path = os.path.join(NOTEBOOKS_DIR, 'df_processed_part1.pkl')
    df_test_path = os.path.join(NOTEBOOKS_DIR, 'df_test_processed_part1.pkl')
    
    if not os.path.exists(df_path):
        logger.error(f"Error: {df_path} not found!")
        logger.error("Please run Part 1 (part1_data_preprocessing.py) first.")
        return None
    
    logger.info(f"Reading {df_path}...")
    df = pd.read_pickle(df_path)
    df_test = pd.read_pickle(df_test_path) if os.path.exists(df_test_path) else None
    
    logger.info(f"Main data loaded: {df.shape}")
    if df_test is not None:
        logger.info(f"Test data loaded: {df_test.shape}")
    
    # Prepare features for modeling
    logger.info("Preparing features for modeling...")
    
    # Save price_base values before dropping
    price_base_values = df["price_base"].copy()
    
    # Columns to drop
    cols_drop = ["date", "day", "dayofweek", "week", "month"]
    
    # Drop sum_total if exists
    if "sum_total" in df.columns:
        cols_drop.append("sum_total")
    
    df_model = df.drop(columns=cols_drop)
    
    # Features and targets
    X = df_model.drop("quantity", axis=1)
    y_quantity = df_model["quantity"]
    
    # Revenue target
    if "sum_total" in df_model.columns:
        y_revenue = df_model["sum_total"]
        logger.info("Using sum_total as revenue target")
    else:
        y_revenue = df["quantity"] * price_base_values
        logger.info("Recreated revenue target from quantity * price_base")
    
    # Categorical columns to string
    categorical_cols = X.select_dtypes("object").columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Quantity target shape: {y_quantity.shape}")
    logger.info(f"Revenue target shape: {y_revenue.shape}")
    
    # Split data
    logger.info("Splitting data for training and validation...")
    seed_value = 42
    idx = np.arange(len(y_quantity))
    idx_train, idx_val = train_test_split(idx, train_size=0.9, random_state=seed_value)
    
    X_train = X.iloc[idx_train]
    X_val = X.iloc[idx_val]
    y_train_qty = y_quantity.iloc[idx_train]
    y_val_qty = y_quantity.iloc[idx_val]
    y_val_rev = y_revenue.iloc[idx_val]
    price_base_val = price_base_values.iloc[idx_val].values
    
    logger.info(f"Train size: {len(idx_train):,}")
    logger.info(f"Validation size: {len(idx_val):,}")
    
    # Handle numerical columns
    logger.info("Handling numerical columns...")
    num_cols = X.select_dtypes([np.int32, np.int64, np.float32, np.float64]).columns
    
    for col in num_cols:
        inf_cnt = int(np.isinf(X_train[col]).sum())
        nan_cnt = int(np.isnan(X_train[col]).sum())
        
        if inf_cnt > 0 or nan_cnt > 0:
            for df_data in [X_train, X_val]:
                df_data[col] = df_data[col].replace([np.inf, -np.inf], np.nan)
            
            med = float(X_train[col].median())
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)
            logger.info(f"  {col}: {inf_cnt} inf, {nan_cnt} NaN -> median {med:.4f}")
    
    # Handle categorical columns
    logger.info(f"Encoding categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    for col in categorical_cols:
        for df_data in [X_train, X_val]:
            df_data[col] = df_data[col].astype(str).replace('nan', 'missing').fillna('missing')
        logger.info(f"  {col}: {X_train[col].nunique()} uniques")
    
    # Label encoding
    logger.info("Applying Label Encoding...")
    label_encoders = {}
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_val[col]]))
        X_train_enc[col] = le.transform(X_train[col])
        X_val_enc[col] = le.transform(X_val[col])
        label_encoders[col] = le
        logger.debug(f"  {col}: {len(le.classes_)} categories encoded")
    
    # Scale numerical features
    logger.info("Scaling numerical features...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_enc[num_cols])
    X_val_num = scaler.transform(X_val_enc[num_cols])
    
    # Combine features
    X_train_transformed = np.hstack([X_train_num, X_train_enc[categorical_cols].values])
    X_val_transformed = np.hstack([X_val_num, X_val_enc[categorical_cols].values])
    
    logger.info(f"Final shapes: train {X_train_transformed.shape}, val {X_val_transformed.shape}")
    logger.info(f"Memory: train {X_train_transformed.nbytes/1024**2:.1f}MB")
    
    # Train CatBoost model
    logger.info("Training CatBoost model...")
    from catboost import CatBoostRegressor
    
    model_path = os.path.join(MODELS_DIR, "demand_forecast_model.cbm")
    catboost_verbose = 100 if verbose else False
    model_trained = False
    
    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        model = CatBoostRegressor()
        model.load_model(model_path)
        logger.info("Model loaded")
    else:
        logger.info("Training new model...")
        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.3,
            depth=10,
            l2_leaf_reg=3,
            early_stopping_rounds=50,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_state=seed_value,
            verbose=catboost_verbose
        )
        
        model.fit(
            X_train_transformed, y_train_qty,
            eval_set=[(X_val_transformed, y_val_qty)],
            verbose=catboost_verbose
        )
        model.save_model(model_path)
        logger.info(f"Model trained and saved to {model_path}")
        model_trained = True
    
    # Evaluate model
    logger.info("Evaluating model...")
    pred_val_qty = model.predict(X_val_transformed)
    
    from sklearn.metrics import mean_squared_error as RMSE
    rmse_qty = float(np.sqrt(RMSE(y_val_qty, pred_val_qty)))
    
    logger.info(f"Quantity RMSE: {rmse_qty:.4f}")
    logger.info(f"Predicted quantity - mean: {np.mean(pred_val_qty):.2f}, std: {np.std(pred_val_qty):.2f}")
    logger.info(f"True quantity - mean: {np.mean(y_val_qty):.2f}, std: {np.std(y_val_qty):.2f}")
    
    # Feature importance (only if model was just trained)
    if model_trained:
        logger.info("Computing feature importance...")
        importance = model.get_feature_importance()
        feature_names = list(num_cols) + categorical_cols
        
        imp_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 Feature Importance:")
        for _, row in imp_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}%")
    
    # Save preprocessing components
    logger.info("Saving preprocessing components...")
    
    X_val_reordered = pd.concat([X_val[num_cols], X_val[categorical_cols]], axis=1)
    
    preprocessing_components = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "numerical_cols": list(num_cols),
        "categorical_cols": categorical_cols,
        "price_base_values": price_base_values,
    }
    
    preprocessed_val_data = {
        "X_val_transformed": X_val_transformed,
        "y_val_qty": y_val_qty,
        "y_val_rev": y_val_rev,
        "price_base_val": price_base_val,
        "idx_val": idx_val,
        "X_val_features": X_val_reordered,
    }
    
    with open(os.path.join(MODELS_DIR, "preprocessing_components.pkl"), "wb") as f:
        pickle.dump(preprocessing_components, f)
    
    with open(os.path.join(MODELS_DIR, "preprocessed_val_data.pkl"), "wb") as f:
        pickle.dump(preprocessed_val_data, f)
    
    logger.info("Saved preprocessing_components.pkl")
    logger.info("Saved preprocessed_val_data.pkl")
    
    logger.info("=" * 60)
    logger.info("PART 2 COMPLETED!")
    logger.info("=" * 60)
    
    return model


if __name__ == "__main__":
    model = main()
