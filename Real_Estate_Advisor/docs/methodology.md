# Methodology

## Objective

The advisor helps investors answer two questions:

1. Is this property likely to be a good investment?
2. What could the estimated price be after five years?

## Data Handling

The raw dataset is loaded from `data/india_housing_prices.csv`. The cleaning pipeline:

- standardizes column names and data types
- removes duplicate rows
- recomputes `Price_per_SqFt` from price and size
- derives property age using 2026 as the current project year
- normalizes yes/no fields
- fills missing numeric values with robust medians
- fills missing categorical values with explicit fallback labels

## Feature Engineering

The engineered features convert raw property records into investor-friendly signals:

- amenity flags for gym, pool, garden, clubhouse, and playground
- amenity count and amenity score
- transport score from low, medium, and high access
- furnished score
- parking and security flags
- infrastructure score using schools, hospitals, transport, parking, and security
- newness score based on property age
- relative price versus city median
- value score for underpriced opportunities
- growth score for long-term appreciation potential

## Target Creation

The dataset does not provide actual future resale outcomes. To support supervised modeling, the project creates domain targets.

`Good_Investment` is true when the investment score passes the threshold. The score blends value, growth, infrastructure, amenities, parking, security, and construction status.

`Estimated_Price_After_5Y` is calculated from current price and a capped annual appreciation rate. The rate is higher for stronger transport, infrastructure, amenities, newer properties, and better value versus the city median.

## Model Training

The advisor uses a NumPy ridge ensemble:

- classifier for good investment probability
- score regressor for investment score
- price regressor for five year value

This approach avoids fragile environment issues while still providing a real train and evaluate workflow.

## Evaluation

The training pipeline uses an 80/20 split and reports:

- accuracy
- precision
- recall
- F1 score
- RMSE
- MAE
- R2

## Deployment

The Streamlit app loads `artifacts/advisor_model.pkl`, accepts property details, returns investment probability and forecast value, and shows visual market insights.
