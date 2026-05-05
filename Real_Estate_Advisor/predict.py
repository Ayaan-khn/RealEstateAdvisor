import joblib
import pandas as pd
from pathlib import Path

from utils.preprocessing import add_price_benchmark, feature_engineering, handle_missing


MODEL_BUNDLE_PATH = "models/model_bundle.pkl"
model_bundle = None
clf = None
reg = None
benchmark_lookups = None
classification_features = None
regression_features = None


def _load_models():
    global model_bundle, clf, reg, benchmark_lookups, classification_features, regression_features

    bundle_path = Path(MODEL_BUNDLE_PATH)
    if bundle_path.exists():
        model_bundle = joblib.load(bundle_path)
        clf = model_bundle["classification_model"]
        reg = model_bundle["regression_model"]
        benchmark_lookups = model_bundle["benchmark_lookups"]
        classification_features = model_bundle["classification_features"]
        regression_features = model_bundle["regression_features"]
        return

    # Fallback for legacy artifacts
    clf = joblib.load("models/classification_model.pkl")
    reg = joblib.load("models/regression_model.pkl")


_load_models()

# -------------------------
# Predict Function
# -------------------------
def predict(input_data: dict):
    df = pd.DataFrame([input_data])

    df = handle_missing(df)
    df = feature_engineering(df)
    if model_bundle is not None:
        df = add_price_benchmark(df, benchmark_lookups)
        df_clf = df[classification_features]
        df_reg = df[regression_features]
        pred_class = int(clf.predict(df_clf)[0])
        pred_price = float(reg.predict(df_reg)[0])
        benchmark_price = float(df["Benchmark_Price_Lakhs"].iloc[0])
    else:
        legacy_df = df.copy()
        legacy_df["ID"] = 0
        model_columns = clf.feature_names_in_
        legacy_df = legacy_df.reindex(columns=model_columns, fill_value=0)
        pred_class = int(clf.predict(legacy_df)[0])
        pred_price = float(reg.predict(legacy_df)[0])
        benchmark_price = pred_price

    current_price = float(df["Price_in_Lakhs"].iloc[0])
    upside = pred_price - current_price
    upside_pct = (upside / current_price * 100.0) if current_price else 0.0

    return {
        "Good_Investment": "Yes" if pred_class == 1 else "No",
        "Estimated_Market_Price": round(pred_price, 2),
        "Current_Price": round(current_price, 2),
        "Benchmark_Price": round(benchmark_price, 2),
        "Expected_Upside_Lakhs": round(upside, 2),
        "Expected_Upside_Percent": round(upside_pct, 2),
    }

# test
if __name__ == "__main__":
    sample = {
        "State": "Maharashtra",
        "City": "Mumbai",
        "Locality": "Andheri",
        "Property_Type": "Apartment",
        "BHK": 2,
        "Size_in_SqFt": 900,
        "Price_in_Lakhs": 120,
        "Year_Built": 2015,
        "Furnished_Status": "Semi-furnished",
        "Floor_No": 5,
        "Total_Floors": 10,
        "Nearby_Schools": 3,
        "Nearby_Hospitals": 2,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": "Yes",
        "Security": "Yes",
        "Amenities": "Gym",
        "Facing": "North",
        "Owner_Type": "Owner",
        "Availability_Status": "Ready_to_Move"
    }

    print(predict(sample))