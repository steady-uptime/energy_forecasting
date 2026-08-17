# src/core/config/factories/model_search_factory.py
from pathlib import Path
from typing import Dict, Any, List

from src.core.config.schemas import ModelDefinition, ModelSearchConfig

ALLOWED_MODEL_KINDS = {
    "random_forest_regressor",
    "random_forest_classifier",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "elastic_net",
}

def build_model_search_config(raw: Dict[str, Any]) -> ModelSearchConfig:
    models = []

    for m in raw.get("models", []):
        kind = m["model_kind"]

        if kind not in ALLOWED_MODEL_KINDS:
            raise ValueError(f"Unsupported model_kind: {kind}")

        models.append(
            ModelDefinition(
                name=m["name"],
                model_kind=kind,
                params=m["params"],
            )
        )

    return ModelSearchConfig(
        strategy=raw["strategy"],
        scoring=raw["scoring"],
        max_trials=raw["max_trials"],
        models=models,
    )
