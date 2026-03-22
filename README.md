# Revenue Optimization System with Demand Forecasting

A comprehensive machine learning system for demand forecasting and price optimization to maximize revenue in retail sales.

## 📋 Project Overview

This project implements a complete revenue optimization pipeline that:

1. **Demand Forecasting** - Predicts product demand using CatBoost with price sensitivity features
2. **Price Elasticity Analysis** - Analyzes how price changes affect demand for different products
3. **Revenue Optimization** - Finds optimal prices that maximize total revenue
4. **Portfolio Optimization** - Provides price recommendations for multiple products simultaneously

## 📁 Project Structure

```
forecasting_sales/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
├── test_pipeline.py                       # Pipeline test script
│
├── data/                                  # Data directory (raw data files)
│   ├── sales.csv                          # Main sales transactions data
│   ├── actual_matrix.csv                  # Actual sales matrix for validation
│   ├── catalog.csv                        # Product catalog with categories
│   ├── discounts_history.csv              # Historical discount data
│   ├── markdowns.csv                      # Markdown events and price reductions
│   ├── online.csv                         # Online sales data
│   ├── price_history.csv                  # Historical price data
│   ├── test.csv                           # Test dataset
│   └── pricing_strategy.csv               # Generated pricing strategy output
│
├── notebooks/                             # Jupyter notebooks (exploratory analysis)
│   ├── demand_forecasting_with_price_sensitivity_1_new.ipynb    # Part 1: Data preprocessing
│   ├── demand_forecasting_with_price_sensitivity_2_new.ipynb    # Part 2: Model training
│   ├── demand_forecasting_with_price_sensitivity_3_complete.ipynb # Part 3: Evaluation
│   ├── price_optimization_analysis.ipynb  # Price elasticity analysis
│   └── revenue_optimization_implementation_final.ipynb # Revenue optimization
│
├── scripts/                               # Python scripts (production pipeline)
│   ├── part1_data_preprocessing.py        # Part 1: Data preprocessing & feature engineering
│   ├── part2_model_training.py            # Part 2: CatBoost model training
│   └── part3_model_evaluation.py          # Part 3: Model evaluation & optimization
│
├── models/                                # Trained models (generated)
│   ├── demand_forecast_model.cbm          # Trained CatBoost model (~31MB)
│   ├── preprocessing_components.pkl       # Label encoders, scaler, column info
│   └── preprocessed_val_data.pkl          # Validation data for evaluation
│
├── src/                                   # Python source code (modular components)
│   ├── __init__.py                        # Package initialization
│   ├── data_loader.py                     # Data loading utilities
│   ├── preprocessing.py                   # Feature engineering functions
│   ├── model.py                           # Model training & prediction
│   └── optimizer.py                       # Revenue optimization engine
│
├── logs/                                  # Log files (generated during training)
│   └── part2_model_training_YYYYMMDD_HHMMSS.log  # Training logs with timestamps
│
├── results/                               # Output results (generated)
│   ├── feature_importance.png             # Feature importance visualization
│   ├── price_scenario_analysis.csv        # Price scenario analysis results
│   └── optimal_price_result.pkl           # Optimal price calculation results
│
└── catboost_info/                         # CatBoost training artifacts (generated)
    ├── catboost_training.json             # Training metrics
    ├── learn_error.tsv                    # Learning error history
    └── test_error.tsv                     # Test error history
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Conda (recommended for environment management)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd forecasting_sales
```

2. Create and activate conda environment:
```bash
conda create -n forecasting_sales python=3.10 -y
conda activate forecasting_sales
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the test pipeline to verify installation:
```bash
python test_pipeline.py
```

5. Run the complete pipeline using Python scripts (recommended for production):
```bash
# Part 1: Data preprocessing & feature engineering
python scripts/part1_data_preprocessing.py

# Part 2: Model training (use --verbose for detailed logs)
python scripts/part2_model_training.py --verbose

# Part 3: Model evaluation and revenue optimization
python scripts/part3_model_evaluation.py
```

Or run the notebooks in order (recommended for exploration):
1. Start with `notebooks/demand_forecasting_with_price_sensitivity_1_new.ipynb`
2. Continue with `notebooks/demand_forecasting_with_price_sensitivity_2_new.ipynb`
3. Complete with `notebooks/demand_forecasting_with_price_sensitivity_3_complete.ipynb`
4. Run `notebooks/price_optimization_analysis.ipynb` for elasticity analysis
5. Finish with `notebooks/revenue_optimization_implementation_final.ipynb`

## 📊 Data Files

| File | Description |
|------|-------------|
| `sales.csv` | Main sales transactions with quantity and price data |
| `actual_matrix.csv` | Actual sales matrix for validation |
| `catalog.csv` | Product catalog with categories and attributes |
| `discounts_history.csv` | Historical discount and promotion data |
| `markdowns.csv` | Markdown events and price reductions |
| `online.csv` | Online sales data |
| `price_history.csv` | Historical price data for items |
| `test.csv` | Test dataset for predictions |

## 🔧 Notebook Pipeline

### Part 1: Data Preprocessing & Feature Engineering
- Data loading and cleaning
- Feature engineering (date features, cyclic features, holidays)
- Price history integration
- Lag features for price sensitivity

### Part 2: Model Training
- CatBoost model training for demand forecasting
- Preprocessing pipeline creation
- Model evaluation and validation

### Part 3: Model Evaluation & Scenario Analysis
- Multi-scenario revenue prediction
- Price change impact analysis
- Feature importance analysis

### Price Optimization Analysis
- Price elasticity calculation
- Elasticity-based categorization
- Pricing strategy recommendations

### Revenue Optimization (Final)
- Complete revenue optimization engine
- Single-item and portfolio optimization
- Implementation recommendations

## 🎯 Key Features

### Demand Forecasting Model
- **Algorithm**: CatBoost Regressor
- **Features**: 50+ features including price, seasonality, holidays, promotions
- **Metrics**: RMSE for quantity and revenue prediction
- **Performance**: Quantity RMSE ~10.23, Revenue Bias ~100.2%

### Price Elasticity Analysis
- Calculates elasticity from historical discount data
- Categorizes products as elastic, inelastic, or abnormal
- Provides data-driven pricing recommendations

### Revenue Optimization Engine
- Tests multiple price scenarios (-30% to +35%)
- Finds optimal price that maximizes revenue
- Supports portfolio-level optimization

## 📈 Usage Example

```python
from revenue_optimizer import RevenueOptimizer

# Initialize optimizer
optimizer = RevenueOptimizer(
    demand_model,
    label_encoders,
    scaler,
    numerical_cols,
    categorical_cols,
    pricing_strategy
)

# Optimize price for a single item
result = optimizer.optimize_price_single_item(
    X_sample,
    base_price=100.0,
    item_id=1001,
    store_id=1,
    verbose=True
)

print(f"Optimal price change: {result['optimal_price_change']:.2%}")
print(f"New recommended price: ${result['optimal_new_price']:.2f}")
print(f"Expected revenue improvement: {result['revenue_improvement']:.2f}%")
```

## 📋 Output Files

### Generated in `models/` directory:

| File | Description |
|------|-------------|
| `demand_forecast_model.cbm` | Trained CatBoost model (~31MB) |
| `preprocessing_components.pkl` | Label encoders, scaler, column information |
| `preprocessed_val_data.pkl` | Preprocessed validation data for evaluation |

### Generated in `logs/` directory:

| File | Description |
|------|-------------|
| `part2_model_training_YYYYMMDD_HHMMSS.log` | Training logs with timestamps |
| `part2_live.log` | Live training output (when running with --verbose) |

### Generated in `results/` directory:

| File | Description |
|------|-------------|
| `feature_importance.png` | Feature importance visualization |
| `price_scenario_analysis.csv` | Price scenario analysis results |
| `optimal_price_result.pkl` | Optimal price calculation results |

### Generated in `catboost_info/` directory:

| File | Description |
|------|-------------|
| `catboost_training.json` | Training metrics in JSON format |
| `learn_error.tsv` | Learning error history |
| `test_error.tsv` | Test error history |

## 🔬 Model Architecture

### Feature Categories
- **Temporal**: day, month, season, dayofweek, cyclic features (sin/cos)
- **Price**: base price, price history, discount percentage, lag features
- **Product**: department, class, weight, volume
- **Store**: store location attributes
- **Events**: holidays, weekends, promotions

### Preprocessing Pipeline
1. Label Encoding for categorical features
2. Standard Scaling for numerical features
3. Handling of unseen categories
4. Missing value imputation with median

### Model Performance (on validation set)
- **Quantity RMSE**: 10.23
- **Revenue RMSE**: 1911.94
- **Revenue Bias**: 100.2% (predictions slightly overestimate actual revenue)

### Top Feature Importances
1. `item_id` (32.53%) - Product identifier
2. `price_lag_1` (17.28%) - 1-day price lag
3. `price_change_1` (13.84%) - 1-day price change
4. `store_id` (13.48%) - Store identifier
5. `price_base` (6.11%) - Base price

## 📝 Implementation Recommendations

1. **Start with A/B testing** on a small subset of items
2. **Implement dynamic pricing** based on real-time demand signals
3. **Monitor competitor pricing** and adjust strategy accordingly
4. **Regularly retrain models** with new data (weekly recommended)
5. **Set up alerts** for significant price changes
6. **Implement rollback mechanisms** for pricing experiments

## 🛠️ Production Deployment

For production deployment:

1. Integrate with existing pricing systems
2. Set up automated model retraining pipelines
3. Implement real-time feature engineering
4. Create monitoring dashboards
5. Establish governance framework for pricing decisions
6. Build API endpoints for real-time recommendations

## 📄 License

This project is for research and educational purposes.

## 👥 Authors

Revenue Optimization System - Demand Forecasting with Price Sensitivity Analysis

## 📞 Support

For questions or issues, please refer to the notebook documentation or contact the project maintainers.
