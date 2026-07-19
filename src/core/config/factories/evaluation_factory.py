# src/core/config/factories/evaluation_factory.py
from ..schemas import EvaluationConfig
def build_evaluation_config(raw: dict) -> EvaluationConfig:
    return EvaluationConfig(**raw)