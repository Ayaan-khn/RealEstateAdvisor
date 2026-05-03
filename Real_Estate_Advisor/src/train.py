from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    ADVISOR_MODEL_PATH,
    CLEAN_DATA_PATH,
    EDA_REPORT_PATH,
    METRICS_PATH,
    MLRUNS_DIR,
    RAW_DATA_PATH,
    REPORT_DIR,
)
from src.features import clean_dataset, engineer_features, load_raw_data
from src.modeling import RealEstateAdvisorModel
from src.utils import ensure_dir, write_json


def _write_model_card(model: RealEstateAdvisorModel) -> None:
    ensure_dir(REPORT_DIR)
    metrics = model.metrics
    lines = [
        "# Real Estate Advisor Model Card",
        "",
        "## Purpose",
        "Predict whether a property is a good investment and estimate its price after five years.",
        "",
        "## Target Design",
        "The source data does not include historical resale outcomes. The project creates domain targets from value, infrastructure, amenities, transport access, property age, and construction status.",
        "",
        "## Model",
        model.model_card()["model_type"],
        "",
        "## Evaluation",
        f"- Classification accuracy: {metrics.get('classification_accuracy', 0):.4f}",
        f"- Precision: {metrics.get('classification_precision', 0):.4f}",
        f"- Recall: {metrics.get('classification_recall', 0):.4f}",
        f"- F1 score: {metrics.get('classification_f1', 0):.4f}",
        f"- Regression RMSE: {metrics.get('regression_rmse', 0):.4f} lakhs",
        f"- Regression MAE: {metrics.get('regression_mae', 0):.4f} lakhs",
        f"- Regression R2: {metrics.get('regression_r2', 0):.4f}",
        "",
        "## Responsible Use",
        "The advisor is a decision support tool. It should be combined with legal diligence, site inspection, loan terms, and local market knowledge before investing.",
        "",
    ]
    (REPORT_DIR / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def _log_mlflow_if_available(model: RealEstateAdvisorModel) -> None:
    ensure_dir(MLRUNS_DIR)
    try:
        import mlflow  # type: ignore
    except Exception as exc:
        status = {
            "status": "offline",
            "reason": f"MLflow is not installed: {exc}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": model.metrics,
        }
        write_json(MLRUNS_DIR / "offline_run.json", status)
        return

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("real_estate_investment_advisor")
    with mlflow.start_run(run_name="numpy_ridge_advisor"):
        mlflow.log_params(
            {
                "model_family": "numpy_ridge_ensemble",
                "trained_rows": model.trained_rows,
                "feature_count": len(model.encoder.feature_columns),
            }
        )
        for key, value in model.metrics.items():
            mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(ADVISOR_MODEL_PATH))
        mlflow.log_artifact(str(REPORT_DIR / "model_card.md"))


def train_pipeline() -> RealEstateAdvisorModel:
    raw = load_raw_data(RAW_DATA_PATH)
    cleaned = clean_dataset(raw)
    engineered = engineer_features(cleaned)

    ensure_dir(CLEAN_DATA_PATH.parent)
    engineered.to_csv(CLEAN_DATA_PATH, index=False)

    model = RealEstateAdvisorModel().fit(cleaned)
    model.save(ADVISOR_MODEL_PATH)

    write_json(METRICS_PATH, model.model_card())
    _write_model_card(model)
    _log_mlflow_if_available(model)

    if not EDA_REPORT_PATH.exists():
        from src.eda import generate_eda_report

        generate_eda_report()

    return model


if __name__ == "__main__":
    trained_model = train_pipeline()
    print("Training complete")
    print(f"Model: {ADVISOR_MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(trained_model.metrics)
