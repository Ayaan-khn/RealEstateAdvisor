from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CURRENT_YEAR, INVESTMENT_SCORE_THRESHOLD, RAW_DATA_PATH


REQUIRED_COLUMNS = [
    "ID",
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]

CATEGORICAL_COLUMNS = [
    "State",
    "City",
    "Property_Type",
    "Furnished_Status",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]

MODEL_NUMERIC_COLUMNS = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "amenity_count",
    "has_gym",
    "has_pool",
    "has_garden",
    "has_clubhouse",
    "has_playground",
    "transport_score",
    "furnished_score",
    "parking_flag",
    "security_flag",
    "floor_ratio",
    "infra_score",
    "amenity_score",
    "newness_score",
    "relative_price_to_city",
    "value_score",
    "growth_score",
]

AMENITY_KEYWORDS = ["Gym", "Pool", "Garden", "Clubhouse", "Playground"]

TRANSPORT_SCORE = {"Low": 0.25, "Medium": 0.60, "High": 1.00}
FURNISHED_SCORE = {"Unfurnished": 0.25, "Semi-furnished": 0.65, "Furnished": 1.00}
YES_NO_SCORE = {"No": 0.0, "Yes": 1.0}


@dataclass
class MarketReference:
    global_price_per_sqft: float
    city_price_median: dict[str, float]
    city_growth_score: dict[str, float]
    category_modes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_price_per_sqft": self.global_price_per_sqft,
            "city_price_median": self.city_price_median,
            "city_growth_score": self.city_growth_score,
            "category_modes": self.category_modes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarketReference":
        return cls(
            global_price_per_sqft=float(payload["global_price_per_sqft"]),
            city_price_median={str(k): float(v) for k, v in payload["city_price_median"].items()},
            city_growth_score={str(k): float(v) for k, v in payload["city_growth_score"].items()},
            category_modes={str(k): str(v) for k, v in payload["category_modes"].items()},
        )


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_yes_no(value: object) -> str:
    text = str(value).strip().title()
    return "Yes" if text in {"Yes", "Y", "True", "1"} else "No"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(col).strip() for col in data.columns]

    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    data = data[REQUIRED_COLUMNS].drop_duplicates().reset_index(drop=True)

    numeric_columns = [
        "ID",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS + ["Locality", "Amenities"]:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip()

    data["Size_in_SqFt"] = data["Size_in_SqFt"].fillna(data["Size_in_SqFt"].median()).clip(lower=1)
    data["Price_in_Lakhs"] = data["Price_in_Lakhs"].fillna(data["Price_in_Lakhs"].median()).clip(lower=1)
    data["BHK"] = data["BHK"].fillna(data["BHK"].median()).clip(lower=1)
    data["Year_Built"] = data["Year_Built"].fillna(CURRENT_YEAR - data["Age_of_Property"].median())
    data["Age_of_Property"] = (CURRENT_YEAR - data["Year_Built"]).clip(lower=0, upper=80)
    data["Floor_No"] = data["Floor_No"].fillna(0).clip(lower=0)
    data["Total_Floors"] = data["Total_Floors"].fillna(data["Total_Floors"].median()).clip(lower=1)
    data["Nearby_Schools"] = data["Nearby_Schools"].fillna(0).clip(lower=0)
    data["Nearby_Hospitals"] = data["Nearby_Hospitals"].fillna(0).clip(lower=0)
    data["Price_per_SqFt"] = data["Price_in_Lakhs"] / data["Size_in_SqFt"].replace(0, np.nan)
    data["Price_per_SqFt"] = data["Price_per_SqFt"].replace([np.inf, -np.inf], np.nan).fillna(
        data["Price_per_SqFt"].median()
    )

    data["Parking_Space"] = data["Parking_Space"].map(normalize_yes_no)
    data["Security"] = data["Security"].map(normalize_yes_no)

    return data


def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    amenities = data["Amenities"].fillna("").astype(str)
    amenity_lower = amenities.str.lower()

    for keyword in AMENITY_KEYWORDS:
        data[f"has_{keyword.lower()}"] = amenity_lower.str.contains(keyword.lower(), regex=False).astype(int)

    amenity_flags = [f"has_{keyword.lower()}" for keyword in AMENITY_KEYWORDS]
    data["amenity_count"] = data[amenity_flags].sum(axis=1)
    data["amenity_score"] = (data["amenity_count"] / len(AMENITY_KEYWORDS)).clip(0, 1)
    data["transport_score"] = data["Public_Transport_Accessibility"].map(TRANSPORT_SCORE).fillna(0.45)
    data["furnished_score"] = data["Furnished_Status"].map(FURNISHED_SCORE).fillna(0.45)
    data["parking_flag"] = data["Parking_Space"].map(YES_NO_SCORE).fillna(0)
    data["security_flag"] = data["Security"].map(YES_NO_SCORE).fillna(0)
    data["floor_ratio"] = (data["Floor_No"] / data["Total_Floors"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).clip(0, 1)

    schools_score = (data["Nearby_Schools"] / 10).clip(0, 1)
    hospital_score = (data["Nearby_Hospitals"] / 10).clip(0, 1)
    data["infra_score"] = (
        0.30 * schools_score
        + 0.30 * hospital_score
        + 0.25 * data["transport_score"]
        + 0.075 * data["parking_flag"]
        + 0.075 * data["security_flag"]
    ).clip(0, 1)
    data["newness_score"] = (1 - data["Age_of_Property"].clip(0, 60) / 60).clip(0, 1)
    data["under_construction_flag"] = (
        data["Availability_Status"].str.lower().eq("under_construction").astype(float)
    )
    return data


def build_market_reference(df: pd.DataFrame) -> MarketReference:
    data = _add_base_features(clean_dataset(df))
    city_price = data.groupby("City")["Price_per_SqFt"].median()
    city_growth = (
        data.assign(
            base_growth=(
                0.36 * data["transport_score"]
                + 0.26 * data["infra_score"]
                + 0.20 * data["amenity_score"]
                + 0.18 * data["newness_score"]
            ).clip(0, 1)
        )
        .groupby("City")["base_growth"]
        .mean()
    )
    modes = {}
    for column in CATEGORICAL_COLUMNS + ["Locality"]:
        mode = data[column].mode(dropna=True)
        modes[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"

    return MarketReference(
        global_price_per_sqft=float(data["Price_per_SqFt"].median()),
        city_price_median={str(k): float(v) for k, v in city_price.items()},
        city_growth_score={str(k): float(v) for k, v in city_growth.items()},
        category_modes=modes,
    )


def engineer_features(df: pd.DataFrame, reference: MarketReference | None = None) -> pd.DataFrame:
    data = _add_base_features(clean_dataset(df))
    ref = reference or build_market_reference(data)

    city_median = data["City"].map(ref.city_price_median).fillna(ref.global_price_per_sqft)
    city_growth = data["City"].map(ref.city_growth_score).fillna(np.mean(list(ref.city_growth_score.values())))

    data["city_price_median"] = city_median
    data["city_growth_score"] = city_growth
    data["relative_price_to_city"] = (data["Price_per_SqFt"] / city_median.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(1.0)
    data["value_score"] = ((1.35 - data["relative_price_to_city"]) / 0.70).clip(0, 1)
    data["growth_score"] = (
        0.34 * data["city_growth_score"]
        + 0.24 * data["infra_score"]
        + 0.17 * data["transport_score"]
        + 0.13 * data["amenity_score"]
        + 0.12 * data["newness_score"]
    ).clip(0, 1)

    data["Investment_Score"] = (
        100
        * (
            0.28 * data["growth_score"]
            + 0.22 * data["value_score"]
            + 0.18 * data["infra_score"]
            + 0.12 * data["amenity_score"]
            + 0.08 * data["parking_flag"]
            + 0.07 * data["security_flag"]
            + 0.05 * data["under_construction_flag"]
        )
    ).clip(0, 100)
    data["Good_Investment"] = data["Investment_Score"] >= INVESTMENT_SCORE_THRESHOLD

    price_penalty = (data["relative_price_to_city"] - 1).clip(0, 1)
    data["Annual_Appreciation_Rate"] = (
        0.035
        + 0.040 * data["growth_score"]
        + 0.020 * data["value_score"]
        + 0.015 * data["amenity_score"]
        + 0.012 * data["newness_score"]
        + 0.008 * data["under_construction_flag"]
        - 0.010 * price_penalty
    ).clip(0.03, 0.12)
    data["Estimated_Price_After_5Y"] = data["Price_in_Lakhs"] * (1 + data["Annual_Appreciation_Rate"]) ** 5

    return data
