# src/core/config/factories/training_factory.py
from ..schemas import TrainingConfig

def build_training_config(raw: dict) -> TrainingConfig:
    """
    Standard primitive unpacking.
    """
    return TrainingConfig(**raw)
