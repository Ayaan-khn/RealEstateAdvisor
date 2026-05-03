from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RAW_DATA_PATH = DATA_DIR / "india_housing_prices.csv"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_data.csv"
ADVISOR_MODEL_PATH = ARTIFACT_DIR / "advisor_model.pkl"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
EDA_REPORT_PATH = REPORT_DIR / "eda_report.md"

CURRENT_YEAR = 2026
INVESTMENT_SCORE_THRESHOLD = 60.0
RANDOM_STATE = 42
