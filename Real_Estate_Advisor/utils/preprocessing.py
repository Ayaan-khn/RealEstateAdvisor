from __future__ import annotations

from datetime import datetime

import pandas as pd


DEFAULT_REFERENCE_YEAR = datetime.now().year


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with type-aware defaults."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return (numerator / denominator).fillna(0.0)


def _to_numeric_signal(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1.0,
                "no": 0.0,
                "high": 3.0,
                "medium": 2.0,
                "low": 1.0,
            }
        )
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(mapped).fillna(0.0)


def feature_engineering(df: pd.DataFrame, reference_year: int = DEFAULT_REFERENCE_YEAR) -> pd.DataFrame:
    """Create stable, inference-safe engineered features."""
    df = df.copy()

    if "Year_Built" in df.columns:
        age = reference_year - df["Year_Built"].astype(float)
        df["Age_of_Property"] = age.clip(lower=0)

    if {"Floor_No", "Total_Floors"}.issubset(df.columns):
        df["Floor_Ratio"] = _safe_divide(
            df["Floor_No"].astype(float), df["Total_Floors"].astype(float)
        )

    if {"Size_in_SqFt", "BHK"}.issubset(df.columns):
        df["Area_per_BHK"] = _safe_divide(
            df["Size_in_SqFt"].astype(float), df["BHK"].astype(float)
        )

    if {"Nearby_Schools", "Nearby_Hospitals"}.issubset(df.columns):
        df["Social_Infra_Index"] = (
            df["Nearby_Schools"].astype(float) + df["Nearby_Hospitals"].astype(float)
        )

    if {"Public_Transport_Accessibility", "Parking_Space"}.issubset(df.columns):
        df["Connectivity_Score"] = (
            _to_numeric_signal(df["Public_Transport_Accessibility"])
            + _to_numeric_signal(df["Parking_Space"])
        )

    if {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(df.columns):
        df["Price_per_SqFt"] = _safe_divide(
            df["Price_in_Lakhs"].astype(float) * 100000.0, df["Size_in_SqFt"].astype(float)
        )

    return df


def _build_lookup(df: pd.DataFrame, keys: list[str]) -> dict[tuple, float]:
    grouped = (
        df.groupby(keys, dropna=False)["Price_in_Lakhs"]
        .median()
        .reset_index()
    )
    lookup: dict[tuple, float] = {}
    for row in grouped.itertuples(index=False):
        key = tuple(getattr(row, col) for col in keys)
        lookup[key] = float(row.Price_in_Lakhs)
    return lookup


def build_price_benchmark_lookups(df: pd.DataFrame) -> dict[str, object]:
    """Build hierarchical price benchmark tables for training/inference."""
    return {
        "city_type_bhk": _build_lookup(df, ["City", "Property_Type", "BHK"]),
        "city_type": _build_lookup(df, ["City", "Property_Type"]),
        "city_only": _build_lookup(df, ["City"]),
        "global_median": float(df["Price_in_Lakhs"].median()),
    }


def _lookup_price(row: pd.Series, lookups: dict[str, object]) -> float:
    key1 = (row["City"], row["Property_Type"], row["BHK"])
    if key1 in lookups["city_type_bhk"]:
        return lookups["city_type_bhk"][key1]

    key2 = (row["City"], row["Property_Type"])
    if key2 in lookups["city_type"]:
        return lookups["city_type"][key2]

    key3 = (row["City"],)
    if key3 in lookups["city_only"]:
        return lookups["city_only"][key3]

    return lookups["global_median"]


def add_price_benchmark(df: pd.DataFrame, lookups: dict[str, object]) -> pd.DataFrame:
    """Attach benchmark price and relative valuation features."""
    df = df.copy()
    df["Benchmark_Price_Lakhs"] = df.apply(lambda row: _lookup_price(row, lookups), axis=1)

    if "Price_in_Lakhs" in df.columns:
        df["Price_vs_Benchmark"] = _safe_divide(
            df["Price_in_Lakhs"].astype(float), df["Benchmark_Price_Lakhs"].astype(float)
        )
        df["Price_Gap_Lakhs"] = (
            df["Benchmark_Price_Lakhs"].astype(float) - df["Price_in_Lakhs"].astype(float)
        )

    return df