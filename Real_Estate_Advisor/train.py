import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from utils.preprocessing import (
    add_price_benchmark,
    build_price_benchmark_lookups,
    feature_engineering,
    handle_missing,
    load_data,
)

DATA_PATH = "data/raw/india_housing_prices.csv"
MODEL_PATH = "models/model_bundle.pkl"
RANDOM_STATE = 42


def build_preprocessor(df, categorical_cols):
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def train():
    df = load_data(DATA_PATH)
    df = handle_missing(df)
    df = feature_engineering(df)

    benchmark_lookups = build_price_benchmark_lookups(df)
    df = add_price_benchmark(df, benchmark_lookups)

    # Label: property is attractive if asking price is at least 3% below benchmark.
    df["Good_Investment"] = (df["Price_vs_Benchmark"] <= 0.97).astype(int)

    regression_features = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Year_Built",
        "Furnished_Status",
        "Floor_No",
        "Total_Floors",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Facing",
        "Owner_Type",
        "Availability_Status",
        "Age_of_Property",
        "Floor_Ratio",
        "Area_per_BHK",
        "Social_Infra_Index",
        "Connectivity_Score",
    ]
    classification_features = regression_features + [
        "Price_in_Lakhs",
        "Benchmark_Price_Lakhs",
        "Price_vs_Benchmark",
        "Price_Gap_Lakhs",
    ]

    categorical_cols = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "Furnished_Status",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Facing",
        "Owner_Type",
        "Availability_Status",
    ]

    X_reg = df[regression_features]
    X_clf = df[classification_features]
    y_reg = df["Price_in_Lakhs"]
    y_clf = df["Good_Investment"]

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )
    X_reg_train, X_reg_test = X_reg.loc[train_idx], X_reg.loc[test_idx]
    X_clf_train, X_clf_test = X_clf.loc[train_idx], X_clf.loc[test_idx]
    y_reg_train, y_reg_test = y_reg.loc[train_idx], y_reg.loc[test_idx]
    y_clf_train, y_clf_test = y_clf.loc[train_idx], y_clf.loc[test_idx]

    reg_preprocessor = build_preprocessor(X_reg, categorical_cols)
    clf_preprocessor = build_preprocessor(X_clf, categorical_cols)

    reg_model = Pipeline(
        steps=[
            ("preprocessor", reg_preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    max_iter=250,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                ),
            ),
        ]
    )
    clf_model = Pipeline(
        steps=[
            ("preprocessor", clf_preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                ),
            ),
        ]
    )

    reg_model.fit(X_reg_train, y_reg_train)
    clf_model.fit(X_clf_train, y_clf_train)

    reg_preds = reg_model.predict(X_reg_test)
    clf_preds = clf_model.predict(X_clf_test)

    metrics = {
        "accuracy": float(accuracy_score(y_clf_test, clf_preds)),
        "f1_score": float(f1_score(y_clf_test, clf_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_reg_test, reg_preds))),
        "mae": float(mean_absolute_error(y_reg_test, reg_preds)),
        "r2": float(r2_score(y_reg_test, reg_preds)),
    }

    bundle = {
        "regression_model": reg_model,
        "classification_model": clf_model,
        "regression_features": regression_features,
        "classification_features": classification_features,
        "categorical_columns": categorical_cols,
        "benchmark_lookups": benchmark_lookups,
        "metrics": metrics,
        "ui_options": {
            "State": sorted(df["State"].dropna().unique().tolist()),
            "City": sorted(df["City"].dropna().unique().tolist()),
            "Property_Type": sorted(df["Property_Type"].dropna().unique().tolist()),
            "Furnished_Status": sorted(df["Furnished_Status"].dropna().unique().tolist()),
            "Public_Transport_Accessibility": sorted(
                df["Public_Transport_Accessibility"].dropna().unique().tolist()
            ),
            "Parking_Space": sorted(df["Parking_Space"].dropna().unique().tolist()),
            "Security": sorted(df["Security"].dropna().unique().tolist()),
            "Facing": sorted(df["Facing"].dropna().unique().tolist()),
            "Owner_Type": sorted(df["Owner_Type"].dropna().unique().tolist()),
            "Availability_Status": sorted(df["Availability_Status"].dropna().unique().tolist()),
        },
    }

    joblib.dump(bundle, MODEL_PATH)

    print("Training complete.")
    print(f"Classification Accuracy: {metrics['accuracy']:.4f}")
    print(f"Classification F1: {metrics['f1_score']:.4f}")
    print(f"Regression RMSE: {metrics['rmse']:.2f}")
    print(f"Regression MAE: {metrics['mae']:.2f}")
    print(f"Regression R2: {metrics['r2']:.4f}")
    print(f"Saved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    train()