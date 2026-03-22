#!/usr/bin/env python3
"""
Demand Forecasting with Price Sensitivity - Part 3
Model Evaluation and Multi-Scenario Revenue Prediction

This script evaluates the trained model and performs revenue optimization analysis.
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, 'notebooks')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging with verbose option
verbose = '--verbose' in sys.argv or '-v' in sys.argv
log_file = os.path.join(LOGS_DIR, f'part3_model_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
logger.info(f"Results directory: {RESULTS_DIR}")
logger.info(f"Verbose mode: {verbose}")


def predict_demand_under_price_scenarios(
    model, X_sample, base_prices, label_encoders, 
    num_cols, cat_cols, price_changes=None
):
    """Predict demand for different price change scenarios."""
    
    if price_changes is None:
        price_changes = [-0.1, -0.05, 0, 0.05, 0.1]
    
    results = []
    
    for price_change in price_changes:
        X_scenario = X_sample.copy()
        new_prices = base_prices * (1 + price_change)
        
        # Handle categorical features with LabelEncoder
        for col in cat_cols:
            if col in label_encoders:
                def safe_transform(val, encoder):
                    try:
                        str_val = str(val)
                        if str_val in encoder.classes_:
                            return encoder.transform([str_val])[0]
                        else:
                            return 0
                    except:
                        return 0
                
                X_scenario[col] = X_scenario[col].apply(
                    lambda x: safe_transform(x, label_encoders[col])
                )
        
        # Reorder columns
        X_scenario = pd.concat([X_scenario[num_cols], X_scenario[cat_cols]], axis=1)
        
        # Transform and predict
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scenario_num = scaler.fit_transform(X_scenario[num_cols])
        X_scenario_transformed = np.hstack([
            X_scenario_num,
            X_scenario[cat_cols].values
        ])
        
        predicted_quantities = model.predict(X_scenario_transformed)
        predicted_revenues = predicted_quantities * new_prices
        
        scenario_results = {
            'price_change': price_change,
            'predicted_quantity': np.mean(predicted_quantities),
            'predicted_revenue': np.mean(predicted_revenues),
            'total_predicted_revenue': np.sum(predicted_revenues)
        }
        results.append(scenario_results)
    
    return pd.DataFrame(results)


def find_optimal_price_for_sample(
    model, X_sample, base_prices, label_encoders,
    num_cols, cat_cols, price_change_range=None
):
    """Find the price change that maximizes revenue for a sample."""
    
    if price_change_range is None:
        price_change_range = np.arange(-0.2, 0.25, 0.05)
    
    revenues = []
    quantities = []
    
    for price_change in price_change_range:
        X_scenario = X_sample.copy()
        new_prices = base_prices * (1 + price_change)
        
        # Handle categorical features
        for col in cat_cols:
            if col in label_encoders:
                def safe_transform(val, encoder):
                    try:
                        str_val = str(val)
                        if str_val in encoder.classes_:
                            return encoder.transform([str_val])[0]
                        else:
                            return 0
                    except:
                        return 0
                
                X_scenario[col] = X_scenario[col].apply(
                    lambda x: safe_transform(x, label_encoders[col])
                )
        
        # Reorder columns
        X_scenario = pd.concat([X_scenario[num_cols], X_scenario[cat_cols]], axis=1)
        
        # Transform and predict
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scenario_num = scaler.fit_transform(X_scenario[num_cols])
        X_scenario_transformed = np.hstack([
            X_scenario_num,
            X_scenario[cat_cols].values
        ])
        
        predicted_quantities = model.predict(X_scenario_transformed)
        predicted_revenues = predicted_quantities * new_prices
        total_revenue = np.sum(predicted_revenues)
        
        revenues.append(total_revenue)
        quantities.append(np.mean(predicted_quantities))
    
    # Find optimal price change
    optimal_idx = np.argmax(revenues)
    optimal_price_change = price_change_range[optimal_idx]
    optimal_revenue = revenues[optimal_idx]
    
    current_revenue_idx = np.where(price_change_range == 0.0)[0]
    if len(current_revenue_idx) > 0:
        current_revenue = revenues[current_revenue_idx[0]]
        revenue_improvement = (optimal_revenue - current_revenue) / current_revenue * 100
    else:
        current_revenue = None
        revenue_improvement = None
    
    return {
        'optimal_price_change': optimal_price_change,
        'optimal_revenue': optimal_revenue,
        'current_revenue': current_revenue,
        'revenue_improvement': revenue_improvement
    }


def main():
    """Main function to run Part 3 model evaluation."""
    logger.info("=" * 60)
    logger.info("PART 3: Model Evaluation and Revenue Optimization")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model and preprocessing components...")
    from catboost import CatBoostRegressor
    
    model_path = os.path.join(MODELS_DIR, "demand_forecast_model.cbm")
    if not os.path.exists(model_path):
        logger.error(f"Error: {model_path} not found!")
        logger.error("Please run Part 2 (part2_model_training.py) first.")
        return None
    
    model = CatBoostRegressor()
    model.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Load preprocessing components
    with open(os.path.join(MODELS_DIR, "preprocessing_components.pkl"), "rb") as f:
        preprocessing_components = pickle.load(f)
    
    label_encoders = preprocessing_components["label_encoders"]
    scaler = preprocessing_components["scaler"]
    num_cols = preprocessing_components["numerical_cols"]
    cat_cols = preprocessing_components["categorical_cols"]
    price_base_values = preprocessing_components["price_base_values"]
    
    logger.info("Preprocessing components loaded")
    
    # Load validation data
    with open(os.path.join(MODELS_DIR, "preprocessed_val_data.pkl"), "rb") as f:
        preprocessed_val_data = pickle.load(f)
    
    X_val_transformed = preprocessed_val_data['X_val_transformed']
    y_val_qty = preprocessed_val_data['y_val_qty']
    y_val_rev = preprocessed_val_data['y_val_rev']
    price_base_val = preprocessed_val_data['price_base_val']
    idx_val = preprocessed_val_data['idx_val']
    X_val = preprocessed_val_data['X_val_features']
    
    logger.info(f"Validation data loaded: {X_val_transformed.shape}")
    
    # Model Evaluation
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    pred_val_qty = model.predict(X_val_transformed)
    
    from sklearn.metrics import mean_squared_error as RMSE
    rmse_qty = float(np.sqrt(RMSE(y_val_qty, pred_val_qty)))
    
    logger.info(f"Quantity RMSE: {rmse_qty:.4f}")
    logger.info(f"Predicted quantity - mean: {np.mean(pred_val_qty):.2f}, std: {np.std(pred_val_qty):.2f}")
    logger.info(f"True quantity - mean: {np.mean(y_val_qty):.2f}, std: {np.std(y_val_qty):.2f}")
    
    # Revenue calculation
    revenue_true_val = y_val_rev.values
    revenue_pred_val = pred_val_qty * price_base_val
    
    rmse_revenue = float(np.sqrt(RMSE(revenue_true_val, revenue_pred_val)))
    
    logger.info(f"Revenue RMSE: {rmse_revenue:.2f}")
    logger.info(f"True Revenue: {revenue_true_val.sum():,.0f}")
    logger.info(f"Pred Revenue: {revenue_pred_val.sum():,.0f}")
    logger.info(f"Revenue Bias: {revenue_pred_val.sum()/revenue_true_val.sum()*100:.1f}%")
    
    # Feature Importance
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)
    
    importance = model.get_feature_importance()
    feature_names = list(num_cols) + cat_cols
    
    imp_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 15 Feature Importance:")
    for _, row in imp_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}%")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top15 = imp_df.head(15)
    plt.barh(range(len(top15)), top15['importance'])
    plt.yticks(range(len(top15)), top15['feature'], fontsize=10)
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance (CatBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
    logger.info(f"Feature importance plot saved to {RESULTS_DIR}/feature_importance.png")
    
    # Revenue Optimization Analysis
    logger.info("=" * 60)
    logger.info("REVENUE OPTIMIZATION ANALYSIS")
    logger.info("=" * 60)
    
    # Sample from validation data
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_val), size=min(1000, len(X_val)), replace=False)
    X_sample = X_val.iloc[sample_indices]
    sample_prices = price_base_val[sample_indices]
    
    logger.info(f"Sample size for price scenario analysis: {len(sample_indices)}")
    logger.info(f"Sample price stats: mean={np.mean(sample_prices):.2f}, std={np.std(sample_prices):.2f}")
    
    # Price scenario analysis
    logger.info("Price Scenario Analysis:")
    scenario_results = predict_demand_under_price_scenarios(
        model, X_sample, sample_prices, label_encoders, num_cols, cat_cols
    )
    for _, row in scenario_results.iterrows():
        logger.info(f"  Price change: {row['price_change']:+.1%} -> Qty: {row['predicted_quantity']:.2f}, Revenue: {row['predicted_revenue']:.2f}")
    
    # Find optimal price
    logger.info("Finding optimal price...")
    optimal_result = find_optimal_price_for_sample(
        model, X_sample, sample_prices, label_encoders, num_cols, cat_cols
    )
    
    logger.info(f"Optimal price change: {optimal_result['optimal_price_change']:.2%}")
    logger.info(f"Optimal revenue: {optimal_result['optimal_revenue']:,.2f}")
    if optimal_result['current_revenue']:
        logger.info(f"Current revenue: {optimal_result['current_revenue']:,.2f}")
        logger.info(f"Revenue improvement: {optimal_result['revenue_improvement']:.2f}%")
    
    # Save results
    logger.info("=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    
    scenario_results.to_csv(os.path.join(RESULTS_DIR, 'price_scenario_analysis.csv'), index=False)
    logger.info(f"Price scenario analysis saved to {RESULTS_DIR}/price_scenario_analysis.csv")
    
    with open(os.path.join(RESULTS_DIR, 'optimal_price_result.pkl'), 'wb') as f:
        pickle.dump(optimal_result, f)
    logger.info(f"Optimal price result saved to {RESULTS_DIR}/optimal_price_result.pkl")
    
    logger.info("=" * 60)
    logger.info("PART 3 COMPLETED!")
    logger.info("=" * 60)
    
    return {
        'rmse_qty': rmse_qty,
        'rmse_revenue': rmse_revenue,
        'optimal_result': optimal_result,
        'feature_importance': imp_df
    }


if __name__ == "__main__":
    results = main()
