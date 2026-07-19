# src/core/config_loader.py
from pathlib import Path
from typing import Optional
from loguru import logger

from src.core.config.raw_loader import RawLoader
from src.core.config.factories.app_factory import build_app_config
from src.core.config.schemas import AppConfig

class ConfigLoader:
    """
    Minimal configuration orchestrator.
    Flow: RawLoader (Input) -> Factories (Transformation) -> AppConfig (Contract)
    """

    _config: Optional[AppConfig] = None
    _project_root: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> AppConfig:
        """
        Load configuration once, using RawLoader + factories.
        """
        if cls._config is not None:
            return cls._config

        logger.info("System: Loading configuration using factory architecture.")

        # Resolve project root once
        if cls._project_root is None:
            # Path calculation: from core/config.py -> core -> src -> project_root
            cls._project_root = Path(__file__).resolve().parents[2]

        # 1. Load raw YAML → dict (The Boundary)
        raw = RawLoader.load(path=path, project_root=cls._project_root)

        # 2. Build typed AppConfig using factories (The Transformation)
        # This is where the raw dict is destroyed and turned into types.
        cls._config = build_app_config(raw)

        # 3. Validate top-level config (The Contract Check)
        cls._config.validate()

        logger.info("System: Configuration loaded and validated successfully.")
        return cls._config

    @classmethod
    def get(cls) -> AppConfig:
        """
        Retrieve cached configuration.
        """
        if cls._config is None:
            return cls.load()
        return cls._config


# Convenience global - Modules will import this 'config' object
config = ConfigLoader.get()
