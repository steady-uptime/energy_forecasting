# src/core/config/factories/model_search_factory.py
from pathlib import Path
from typing import Dict, Any, List

from src.core.config.schemas import ModelDefinition, ModelSearchConfig


def build_model_search_config(raw: Dict[str, Any]) -> ModelSearchConfig:
    """
    Factory for constructing ModelSearchConfig from raw YAML dict.
    Ensures all model definitions are converted into typed ModelDefinition objects.
    """
    models = [
        ModelDefinition(
            name=m["name"],
            model_kind=m["model_kind"],
            params=m["params"],
        )
        for m in raw.get("models", [])
    ]

    return ModelSearchConfig(
        strategy=raw["strategy"],
        scoring=raw["scoring"],
        max_trials=raw["max_trials"],
        models=models,
    )
