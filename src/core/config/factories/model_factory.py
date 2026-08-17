# src/core/config/factories/model_factory.py
from ..schemas import ModelConfig, HPOConfig

# Add allowed model kinds here
ALLOWED_MODEL_KINDS = {
    "random_forest_regressor",
    "random_forest_classifier",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "elastic_net",
}

def build_hpo_config(raw: dict) -> HPOConfig:
    return HPOConfig(**raw)

def build_model_config(raw: dict) -> ModelConfig:
    model_kind = raw["model_kind"]

    if model_kind not in ALLOWED_MODEL_KINDS:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    return ModelConfig(
        name=raw["name"],
        model_kind=model_kind,
        dry_run=raw["dry_run"],
        params=raw["params"],
        hpo=build_hpo_config(raw["hpo"]),
    )
