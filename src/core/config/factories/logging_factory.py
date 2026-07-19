# src/core/config/factories/logging_factory.py
from ..schemas import LoggingConfig
def build_logging_config(raw: dict) -> LoggingConfig:
    return LoggingConfig(**raw)
