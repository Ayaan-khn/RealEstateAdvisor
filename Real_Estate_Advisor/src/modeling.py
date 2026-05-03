from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import (
    CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    MarketReference,
    build_market_reference,
    clean_dataset,
    engineer_features,
)


@dataclass
class TabularEncoder:
    numeric_columns: list[str] = field(default_factory=lambda: MODEL_NUMERIC_COLUMNS.copy())
    categorical_columns: list[str] = field(default_factory=lambda: CATEGORICAL_COLUMNS.copy())
    dummy_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        frame = df.copy()
        self.medians = {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in self.numeric_columns
        }
        self.modes = {}
        for column in self.categorical_columns:
            mode = frame[column].astype(str).mode(dropna=True)
            self.modes[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"

        numeric = self._numeric_frame(frame)
        dummies = self._dummy_frame(frame, fit=True)
        features = pd.concat([numeric, dummies], axis=1)
        self.feature_columns = features.columns.tolist()
        self.means = features.mean().astype(float).to_dict()
        std = features.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.stds = std.astype(float).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        frame = df.copy()
        numeric = self._numeric_frame(frame)
        dummies = self._dummy_frame(frame, fit=False)
        features = pd.concat([numeric, dummies], axis=1)
        features = features.reindex(columns=self.feature_columns, fill_value=0.0)

        for column in self.feature_columns:
            features[column] = (features[column] - self.means[column]) / self.stds[column]

        matrix = features.to_numpy(dtype=float)
        intercept = np.ones((matrix.shape[0], 1), dtype=float)
        return np.hstack([intercept, matrix])

    def _numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = pd.DataFrame(index=df.index)
        for column in self.numeric_columns:
            fallback = self.medians.get(column, 0.0)
            numeric[column] = pd.to_numeric(df.get(column, fallback), errors="coerce").fillna(fallback)
        return numeric.astype(float)

    def _dummy_frame(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat = pd.DataFrame(index=df.index)
        for column in self.categorical_columns:
            fallback = self.modes.get(column, "Unknown")
            cat[column] = df.get(column, fallback)
            cat[column] = cat[column].fillna(fallback).astype(str)
        dummies = pd.get_dummies(cat, columns=self.categorical_columns, dtype=float)
        if fit:
            self.dummy_columns = dummies.columns.tolist()
        return dummies.reindex(columns=self.dummy_columns, fill_value=0.0)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return 0.0
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "classification_accuracy": accuracy,
        "classification_precision": float(precision),
        "classification_recall": float(recall),
        "classification_f1": float(f1),
    }


@dataclass
class RealEstateAdvisorModel:
    encoder: TabularEncoder = field(default_factory=TabularEncoder)
    market_reference: MarketReference | None = None
    classifier_coef: np.ndarray | None = None
    score_coef: np.ndarray | None = None
    regressor_coef: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    trained_rows: int = 0

    def fit(self, raw_df: pd.DataFrame, max_rows: int = 120_000, random_state: int = 42) -> "RealEstateAdvisorModel":
        clean = clean_dataset(raw_df)
        self.market_reference = build_market_reference(clean)
        data = engineer_features(clean, self.market_reference)

        if len(data) > max_rows:
            data = data.sample(max_rows, random_state=random_state).reset_index(drop=True)

        rng = np.random.default_rng(random_state)
        order = rng.permutation(len(data))
        split = int(len(data) * 0.80)
        train = data.iloc[order[:split]].reset_index(drop=True)
        test = data.iloc[order[split:]].reset_index(drop=True)

        self.encoder.fit(train)
        X_train = self.encoder.transform(train)
        X_test = self.encoder.transform(test)

        y_class = train["Good_Investment"].astype(float).to_numpy()
        y_score = train["Investment_Score"].astype(float).to_numpy()
        y_reg = train["Estimated_Price_After_5Y"].astype(float).to_numpy()

        self.classifier_coef = _fit_ridge(X_train, y_class, alpha=3.0)
        self.score_coef = _fit_ridge(X_train, y_score, alpha=3.0)
        self.regressor_coef = _fit_ridge(X_train, y_reg, alpha=8.0)
        self.trained_rows = int(len(train))

        self.metrics = self.evaluate(test, X_test)
        return self

    def evaluate(self, test: pd.DataFrame, X_test: np.ndarray | None = None) -> dict[str, float]:
        if X_test is None:
            X_test = self.encoder.transform(test)

        probability = self._predict_probability(X_test, test)
        class_pred = (probability >= 0.50).astype(int)
        class_true = test["Good_Investment"].astype(int).to_numpy()

        reg_pred = self._predict_price(X_test, test)
        reg_true = test["Estimated_Price_After_5Y"].astype(float).to_numpy()

        metrics = _classification_metrics(class_true, class_pred)
        metrics.update(
            {
                "regression_rmse": _rmse(reg_true, reg_pred),
                "regression_mae": _mae(reg_true, reg_pred),
                "regression_r2": _r2(reg_true, reg_pred),
                "test_rows": float(len(test)),
                "train_rows": float(self.trained_rows),
            }
        )
        return metrics

    def predict(self, raw_records: pd.DataFrame) -> pd.DataFrame:
        if self.market_reference is None:
            raise RuntimeError("Model has not been trained.")
        data = engineer_features(raw_records, self.market_reference)
        X = self.encoder.transform(data)

        probability = self._predict_probability(X, data)
        investment_score = self._predict_score(X, data)
        forecast = self._predict_price(X, data)
        confidence = np.abs(probability - 0.5) * 2

        result = data.copy()
        result["Investment_Probability"] = probability
        result["Model_Confidence"] = confidence.clip(0, 1)
        result["Investment_Score_Prediction"] = investment_score
        result["Good_Investment_Prediction"] = probability >= 0.50
        result["Predicted_5Y_Price_Lakhs"] = forecast
        result["Drivers"] = [self._drivers(row) for _, row in result.iterrows()]
        return result

    def _predict_probability(self, X: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        if self.classifier_coef is None:
            raise RuntimeError("Classifier coefficients are missing.")
        linear = X @ self.classifier_coef
        rule = data["Investment_Score"].to_numpy(dtype=float) / 100
        probability = 0.65 * linear + 0.35 * rule
        return np.clip(probability, 0.03, 0.97)

    def _predict_score(self, X: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        if self.score_coef is None:
            raise RuntimeError("Score coefficients are missing.")
        linear = X @ self.score_coef
        rule = data["Investment_Score"].to_numpy(dtype=float)
        return np.clip(0.65 * linear + 0.35 * rule, 0, 100)

    def _predict_price(self, X: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        if self.regressor_coef is None:
            raise RuntimeError("Regressor coefficients are missing.")
        linear = X @ self.regressor_coef
        rule = data["Estimated_Price_After_5Y"].to_numpy(dtype=float)
        current = data["Price_in_Lakhs"].to_numpy(dtype=float)
        forecast = 0.70 * linear + 0.30 * rule
        lower = current * (1.03**5)
        upper = current * (1.90)
        return np.clip(forecast, lower, upper)

    def _drivers(self, row: pd.Series) -> list[str]:
        drivers: list[str] = []
        if row["value_score"] >= 0.65:
            drivers.append("Priced attractively versus the city median")
        elif row["relative_price_to_city"] > 1.15:
            drivers.append("Price per sqft is high versus the city median")

        if row["transport_score"] >= 0.90:
            drivers.append("Strong public transport accessibility")
        elif row["transport_score"] <= 0.30:
            drivers.append("Weak public transport access")

        if row["infra_score"] >= 0.70:
            drivers.append("Healthy school, hospital, parking, and security mix")
        if row["amenity_score"] >= 0.70:
            drivers.append("Amenity package supports resale demand")
        if row["newness_score"] >= 0.70:
            drivers.append("Newer property age improves long term appeal")
        if row["under_construction_flag"] == 1:
            drivers.append("Under construction status adds appreciation potential")

        if not drivers:
            drivers.append("Balanced profile with no extreme positive or negative signal")
        return drivers

    def feature_importance(self) -> pd.DataFrame:
        if self.score_coef is None or not self.encoder.feature_columns:
            return pd.DataFrame(columns=["feature", "importance"])
        weights = np.abs(self.score_coef[1:])
        total = weights.sum()
        importance = weights / total if total else weights
        return (
            pd.DataFrame({"feature": self.encoder.feature_columns, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def model_card(self) -> dict[str, Any]:
        return {
            "model_type": "NumPy ridge ensemble for classification, score prediction, and 5 year price regression",
            "trained_rows": self.trained_rows,
            "metrics": self.metrics,
            "feature_count": len(self.encoder.feature_columns),
            "categorical_features": self.encoder.categorical_columns,
            "numeric_features": self.encoder.numeric_columns,
        }

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path | str) -> "RealEstateAdvisorModel":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError("Loaded artifact is not a RealEstateAdvisorModel.")
        return loaded
