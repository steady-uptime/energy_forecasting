# src/core/config/factories/env_factory.py
from ..schemas import EnvConfig, ComputeConfig

def build_compute_config(raw: dict) -> ComputeConfig:
    return ComputeConfig(**raw)

def build_env_config(raw: dict) -> EnvConfig:
    """
    Handles nested ComputeConfig construction via sub-factory.
    """
    return EnvConfig(
        mode=raw["mode"],
        compute=build_compute_config(raw["compute"]),
        env_mapping=raw["env_mapping"]
    )