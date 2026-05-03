# Real Estate Investment Advisor

A complete machine learning project for evaluating residential property investment potential in India. The project cleans the provided housing dataset, engineers domain features, creates investment targets, trains a classification and regression advisor, produces EDA outputs, and serves the results in a Streamlit dashboard.

## What This Project Delivers

- Cleaned and feature-engineered dataset at `data/cleaned_data.csv`
- Good investment classifier
- Five year property value forecaster
- EDA report with the 20 required analysis questions
- Streamlit dashboard for investor-facing predictions and insights
- MLflow tracking when MLflow is installed, with an offline run summary otherwise
- Project documentation and model card

## Project Structure

```text
Real_Estate_Advisor/
  app.py                         Streamlit dashboard
  data/
    india_housing_prices.csv     Raw dataset
    cleaned_data.csv             Generated cleaned dataset
  src/
    features.py                  Cleaning, feature engineering, target creation
    modeling.py                  NumPy model training and prediction logic
    train.py                     End-to-end training pipeline
    eda.py                       EDA report and chart generation
  reports/
    eda_report.md                Generated EDA findings
    model_card.md                Generated model card
    model_metrics.json           Model metrics
  artifacts/
    advisor_model.pkl            Trained advisor model
```

## Setup

Use the existing virtual environment from the parent project when available.

```powershell
C:\Users\mikez\PycharmProjects\RealEstateAdvisor\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training

```powershell
python -m src.train
```

This creates:

- `data/cleaned_data.csv`
- `artifacts/advisor_model.pkl`
- `reports/model_metrics.json`
- `reports/model_card.md`
- `reports/eda_report.md`

## Run EDA Only

```powershell
python -m src.eda
```

## Run Streamlit App

```powershell
streamlit run app.py
```

If the model artifact is missing, the app trains the advisor automatically on first launch.

## Modeling Notes

The source dataset does not include historical resale prices or actual investor returns. To make the project usable, the pipeline creates domain-driven targets:

- `Good_Investment`: based on value versus city median, infrastructure, transport, amenities, age, parking, security, and construction status.
- `Estimated_Price_After_5Y`: based on current price and a capped appreciation model driven by the same investment signals.

The model is a NumPy ridge ensemble that trains a classifier, investment score predictor, and five year price regressor. It runs without scikit-learn, which makes the project reliable in restricted internship evaluation environments. The requirements file still includes scikit-learn and MLflow so the environment matches common capstone expectations.
