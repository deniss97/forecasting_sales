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
├── data/                                  # Data directory
│   ├── actual_matrix.csv                  # Actual sales matrix
│   ├── catalog.csv                        # Product catalog
│   ├── discounts_history.csv              # Historical discount data
│   ├── markdowns.csv                      # Markdown events
│   ├── online.csv                         # Online sales data
│   ├── sales.csv                          # Historical sales data (if available)
│   ├── stores.csv                         # Store information (if available)
│   ├── price_history.csv                  # Historical prices (if available)
│   └── test.csv                           # Test dataset (if available)
├── notebooks/                             # Jupyter notebooks
│   ├── demand_forecasting_with_price_sensitivity_1_new.ipynb    # Part 1: Data preprocessing
│   ├── demand_forecasting_with_price_sensitivity_2_new.ipynb    # Part 2: Model training
│   ├── demand_forecasting_with_price_sensitivity_3_complete.ipynb # Part 3: Evaluation
│   ├── price_optimization_analysis.ipynb  # Price elasticity analysis
│   └── revenue_optimization_implementation_final.ipynb # Revenue optimization
├── models/                                # Trained models (generated)
│   ├── demand_forecast_model.cbm          # Trained CatBoost model
│   ├── preprocessing_components.pkl       # Preprocessing pipeline
│   └── revenue_optimizer.pkl              # Revenue optimizer instance
├── src/                                   # Python source code
│   ├── __init__.py                        # Package initialization
│   ├── data_loader.py                     # Data loading utilities
│   ├── preprocessing.py                   # Feature engineering
│   ├── model.py                           # Model training & prediction
│   └── optimizer.py                       # Revenue optimization engine
└── results/                               # Output results (generated)
    ├── pricing_strategy.csv               # Optimal pricing recommendations
    └── revenue_optimization_results.csv   # Optimization results
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd forecasting_sales
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebooks in order:
   - Start with `demand_forecasting_with_price_sensitivity_1_new.ipynb`
   - Continue with `demand_forecasting_with_price_sensitivity_2_new.ipynb`
   - Complete with `demand_forecasting_with_price_sensitivity_3_complete.ipynb`
   - Run `price_optimization_analysis.ipynb` for elasticity analysis
   - Finish with `revenue_optimization_implementation_final.ipynb`

## 📊 Data Files

| File | Description |
|------|-------------|
| `sales.csv` | Historical offline sales data |
| `online.csv` | Historical online sales data |
| `catalog.csv` | Product catalog with categories |
| `stores.csv` | Store locations and attributes |
| `discounts_history.csv` | Historical discount and promotion data |
| `price_history.csv` | Historical price changes |
| `markdowns.csv` | Markdown events |
| `test.csv` | Test dataset for predictions |
| `actual_matrix.csv` | Actual sales matrix for validation |

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

After running the complete pipeline:

| File | Description |
|------|-------------|
| `demand_forecast_model.cbm` | Trained CatBoost model |
| `preprocessing_components.pkl` | Label encoders, scaler, column info |
| `pricing_strategy.csv` | Elasticity-based pricing recommendations |
| `revenue_optimization_results.csv` | Final optimization results |
| `revenue_optimizer.pkl` | Serialized optimizer instance |

## 🔬 Model Architecture

### Feature Categories
- **Temporal**: day, month, season, dayofweek, cyclic features
- **Price**: base price, price history, discount percentage, lag features
- **Product**: department, class, weight, volume
- **Store**: store location attributes
- **Events**: holidays, weekends, promotions

### Preprocessing Pipeline
1. Label Encoding for categorical features
2. Standard Scaling for numerical features
3. Handling of unseen categories

## 📝 Implementation Recommendations

1. **Start with A/B testing** on a small subset of items
2. **Implement dynamic pricing** based on real-time demand signals
3. **Monitor competitor pricing** and adjust strategy accordingly
4. **Regularly retrain models** with new data
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
