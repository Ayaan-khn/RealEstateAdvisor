# Real Estate Investment Advisor Project Report

## Business Problem

Real estate investors need a fast way to compare properties across cities, property types, amenities, infrastructure, and price levels. Manual analysis is slow and inconsistent, especially when each listing has many mixed numeric and categorical fields.

## Solution

This project builds a Streamlit-based investment advisor that combines data cleaning, feature engineering, EDA, classification, regression, and model reporting. The app gives an investor a clear recommendation and a five year value forecast from property details.

## Business Use Cases

- Identify high-potential properties in developing locations
- Compare listings by price per sqft, infrastructure, and amenity strength
- Support buyers with data-backed investment recommendations
- Help real estate platforms add automated advisory intelligence

## Deliverables

- `data/cleaned_data.csv`
- `src/features.py`
- `src/modeling.py`
- `src/train.py`
- `src/eda.py`
- `app.py`
- `reports/eda_report.md`
- `reports/model_card.md`
- `artifacts/advisor_model.pkl`

## Findings Summary

The generated EDA report covers the required 20 analysis questions. The Streamlit dashboard exposes the same insights interactively through market filters, charts, a prediction form, model metrics, and a data explorer.

## Limitations

The dataset is cross-sectional and does not contain actual resale history. The five year forecast is therefore a modeled estimate from domain signals, not a guarantee of return.

## Future Improvements

- Add real transaction history by year
- Add geospatial latitude and longitude for heatmaps
- Train tree-based models such as XGBoost after dependency installation
- Add Streamlit Cloud deployment secrets and CI checks
