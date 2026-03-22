"""
Test Pipeline for Revenue Optimization System

This script tests the complete pipeline from data loading to price optimization.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 60)
print("REVENUE OPTIMIZATION SYSTEM - PIPELINE TEST")
print("=" * 60)

# Test 1: Data Loading
print("\n[TEST 1] Data Loading...")
try:
    from src.data_loader import DataLoader
    
    loader = DataLoader(data_dir=".")
    available_files = loader.list_available_files()
    print(f"  Available files: {available_files}")
    
    # Load key files
    df_catalog = loader.load_csv("catalog")
    df_actual = loader.load_csv("actual_matrix")
    df_online = loader.load_csv("online")
    df_discounts = loader.load_csv("discounts_history")
    df_markdowns = loader.load_csv("markdowns")
    
    print(f"  ✓ catalog: {df_catalog.shape}")
    print(f"  ✓ actual_matrix: {df_actual.shape}")
    print(f"  ✓ online: {df_online.shape}")
    print(f"  ✓ discounts_history: {df_discounts.shape}")
    print(f"  ✓ markdowns: {df_markdowns.shape}")
    print("  ✅ Data Loading: PASSED")
except Exception as e:
    print(f"  ❌ Data Loading: FAILED - {e}")
    sys.exit(1)

# Test 2: Data Preprocessing
print("\n[TEST 2] Data Preprocessing...")
try:
    from src.preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'item_id': np.random.choice(['item_001', 'item_002', 'item_003'], 100),
        'store_id': np.random.choice([1, 2, 3], 100),
        'price_base': np.random.uniform(10, 100, 100),
        'quantity': np.random.randint(1, 50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    # Add date features
    sample_data = preprocessor.add_date_features(sample_data)
    print(f"  ✓ Date features added: {list(sample_data.columns)}")
    
    # Add cyclic features
    sample_data = preprocessor.add_cyclic_features(sample_data)
    print(f"  ✓ Cyclic features added")
    
    # Add holiday features
    sample_data = preprocessor.add_holiday_features(sample_data)
    print(f"  ✓ Holiday features added")
    
    # Add seasonal features
    sample_data = preprocessor.add_seasonal_features(sample_data)
    print(f"  ✓ Seasonal features added")
    
    # Add lag features
    sample_data = preprocessor.add_lag_features(sample_data)
    print(f"  ✓ Lag features added")
    
    # Fit and transform
    numerical_cols = ['price_base', 'day', 'month', 'week', 'dayofweek', 
                      'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos',
                      'week_sin', 'week_cos', 'month_sin', 'month_cos']
    categorical_cols = ['item_id', 'store_id', 'category', 'region', 'season']
    
    X_transformed = preprocessor.fit_transform(sample_data, numerical_cols, categorical_cols)
    print(f"  ✓ Transformed shape: {X_transformed.shape}")
    print("  ✅ Data Preprocessing: PASSED")
except Exception as e:
    print(f"  ❌ Data Preprocessing: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model Training
print("\n[TEST 3] Model Training...")
try:
    from src.model import DemandForecaster
    
    # Create training data
    np.random.seed(42)
    n_samples = 1000
    
    X_train = np.random.randn(n_samples, len(numerical_cols) + len(categorical_cols))
    y_train = np.random.randint(1, 100, n_samples)
    
    X_val = np.random.randn(200, len(numerical_cols) + len(categorical_cols))
    y_val = np.random.randint(1, 100, 200)
    
    feature_names = numerical_cols + categorical_cols
    
    # Train model
    model = DemandForecaster(iterations=100, learning_rate=0.1, depth=6, verbose=False)
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
    print(f"  ✓ Model trained")
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    print(f"  ✓ RMSE: {metrics['rmse']:.4f}")
    
    # Get feature importance
    importance_df = model.get_feature_importance_df()
    print(f"  ✓ Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.2f}%)")
    
    # Save and load model
    model_path = "models/test_model.cbm"
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)
    
    # Load model
    new_model = DemandForecaster()
    new_model.load_model(model_path)
    print(f"  ✓ Model saved and loaded")
    
    print("  ✅ Model Training: PASSED")
except Exception as e:
    print(f"  ❌ Model Training: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Revenue Optimization
print("\n[TEST 4] Revenue Optimization...")
try:
    from src.optimizer import RevenueOptimizer
    
    # Create sample data for optimization
    X_sample = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-15')] * 10,
        'item_id': ['item_001'] * 10,
        'store_id': [1] * 10,
        'price_base': [50.0] * 10,
        'day': [15] * 10,
        'month': [1] * 10,
        'week': [3] * 10,
        'dayofweek': [0] * 10,
        'day_sin': [np.sin(2 * np.pi * 14 / 31)] * 10,
        'day_cos': [np.cos(2 * np.pi * 14 / 31)] * 10,
        'dayofweek_sin': [0] * 10,
        'dayofweek_cos': [1] * 10,
        'week_sin': [np.sin(2 * np.pi * 2 / 52)] * 10,
        'week_cos': [np.cos(2 * np.pi * 2 / 52)] * 10,
        'month_sin': [0] * 10,
        'month_cos': [1] * 10,
        'is_weekend': [False] * 10,
        'is_sunday': [False] * 10,
        'holidays': [False] * 10,
        'season': [4] * 10,
        'category': ['A'] * 10,
        'region': ['North'] * 10,
        'price_lag_1': [50.0] * 10,
        'price_lag_7': [48.0] * 10,
        'price_lag_30': [45.0] * 10,
        'price_change_1': [0] * 10,
        'price_change_7': [0.04] * 10,
        'price_change_30': [0.11] * 10,
    })
    
    # Create optimizer
    optimizer = RevenueOptimizer(
        demand_model=model,
        label_encoders=preprocessor.label_encoders,
        scaler=preprocessor.scaler,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols
    )
    
    # Test single item optimization
    result = optimizer.optimize_price_single_item(
        X_sample=X_sample.iloc[:1],
        base_price=50.0,
        item_id="item_001",
        store_id=1,
        price_change_range=np.arange(-0.2, 0.25, 0.05),
        verbose=False
    )
    
    print(f"  ✓ Optimal price change: {result['optimal_price_change']:.0%}")
    print(f"  ✓ New price: {result['optimal_new_price']:.2f}")
    print(f"  ✓ Optimal revenue: {result['optimal_revenue']:.2f}")
    
    if result['revenue_improvement']:
        print(f"  ✓ Revenue improvement: {result['revenue_improvement']:.2f}%")
    
    print("  ✅ Revenue Optimization: PASSED")
except Exception as e:
    print(f"  ❌ Revenue Optimization: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Integration Test
print("\n[TEST 5] Integration Test...")
try:
    # Test complete workflow with actual data structure
    df_online['date'] = pd.to_datetime(df_online['date'])
    df_online = df_online.head(1000)  # Use sample for speed
    
    # Calculate price_base if not exists
    if 'price_base' not in df_online.columns:
        df_online['price_base'] = df_online['sum_total'] / df_online['quantity']
    
    # Filter valid data
    mask = (df_online['quantity'] > 0) & (df_online['price_base'] > 0)
    df_online = df_online[mask].copy()
    
    # Add features
    df_online = preprocessor.add_date_features(df_online)
    df_online = preprocessor.add_cyclic_features(df_online)
    df_online = preprocessor.add_holiday_features(df_online)
    df_online = preprocessor.add_seasonal_features(df_online)
    
    # Add missing columns for integration test
    if 'category' not in df_online.columns:
        df_online['category'] = 'A'
    if 'region' not in df_online.columns:
        df_online['region'] = 'North'
    
    # Prepare features - only use columns that exist
    available_num_cols = [col for col in numerical_cols if col in df_online.columns]
    available_cat_cols = [col for col in categorical_cols if col in df_online.columns]
    
    X = df_online[available_num_cols + available_cat_cols].copy()
    
    # Handle missing columns
    for col in available_num_cols:
        if col not in X.columns:
            X[col] = 0.0
    for col in available_cat_cols:
        if col not in X.columns:
            X[col] = 'unknown'
    
    # Transform
    X_transformed = preprocessor.transform(X)
    
    # Predict
    predictions = model.predict(X_transformed)
    
    print(f"  ✓ Predictions shape: {predictions.shape}")
    print(f"  ✓ Mean predicted quantity: {np.mean(predictions):.2f}")
    print(f"  ✓ Std predicted quantity: {np.std(predictions):.2f}")
    
    # Calculate revenue
    prices = df_online['price_base'].values[:len(predictions)]
    revenue = predictions * prices
    print(f"  ✓ Total predicted revenue: {np.sum(revenue):,.2f}")
    
    print("  ✅ Integration Test: PASSED")
except Exception as e:
    print(f"  ❌ Integration Test: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✅")
print("=" * 60)
print("\nThe Revenue Optimization System is working correctly.")
print("You can now run the notebooks in order:")
print("  1. notebooks/demand_forecasting_with_price_sensitivity_1_new.ipynb")
print("  2. notebooks/demand_forecasting_with_price_sensitivity_2_new.ipynb")
print("  3. notebooks/demand_forecasting_with_price_sensitivity_3_complete.ipynb")
print("  4. notebooks/price_optimization_analysis.ipynb")
print("  5. notebooks/revenue_optimization_implementation_final.ipynb")
