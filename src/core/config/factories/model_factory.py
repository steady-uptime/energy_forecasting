# src/core/config/factories/model_factory.py
from ..schemas import ModelConfig, HPOConfig

def build_hpo_config(raw: dict) -> HPOConfig:
    # Parameters is a Dict[str, Any], so we can unpack it directly
    return HPOConfig(**raw)

def build_model_config(raw: dict) -> ModelConfig:
    """
    Constructs ModelConfig by delegating HPO construction.
    """
    return ModelConfig(
        name=raw["name"],
        model_kind=raw["model_kind"],
        dry_run=raw["dry_run"],
        params=raw["params"],
        hpo=build_hpo_config(raw["hpo"])
    )
