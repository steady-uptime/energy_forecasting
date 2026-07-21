# src/core/config/factories/app_factory.py
from ..schemas import AppConfig, DebugConfig
from pathlib import Path
from uuid import UUID
import os

from .env_factory import build_env_config
from .data_factory import build_data_config, build_features_config
from .paths_factory import build_paths_config
from .model_factory import build_model_config
from .model_search_factory import build_model_search_config
from .training_factory import build_training_config
from .artifacts_factory import build_artifact_config
from .logging_factory import build_logging_config
from .monitoring_factory import build_monitoring_config
from .evaluation_factory import build_evaluation_config

def build_app_config(raw: dict) -> AppConfig:
    """
    The central orchestrator for configuration construction.
    Every nested object is constructed via its specific factory.
    """
    # --- Resolve Run Context Inline ---
    # Logic: YAML -> Environment Variable -> Generate Unique UUID
    raw_run_id = raw.get("run_id")
    env_run_id = os.getenv("RUN_ID")
    run_id_str = raw_run_id or env_run_id or str(UUID(bytes=os.urandom(16)))
    
    # Resolve project_root: YAML -> Default calculation
    raw_root = raw.get("project_root")
    project_root_path = Path(raw_root) if raw_root else Path(__file__).resolve().parents[2]

    # --- Build the AppConfig Contract ---
    return AppConfig(
        project_name=raw["project_name"],
        version=raw["version"],
        run_id=UUID(run_id_str),
        project_root=project_root_path,
        env=build_env_config(raw["env"]),
        data=build_data_config(raw["data"]),
        # Pass the resolved project_root_path to the paths factory
        paths=build_paths_config(raw["paths"], project_root_path),
        features=build_features_config(raw["features"]),
        model=build_model_config(raw["model"]),
        model_search=build_model_search_config(raw["model_search"]),
        training=build_training_config(raw["training"]),
        artifacts=build_artifact_config(raw["artifacts"]),
        logging=build_logging_config(raw["logging"]),
        monitoring=build_monitoring_config(raw["monitoring"]),
        evaluation=build_evaluation_config(raw["evaluation"]),
        debug=DebugConfig(**raw.get("debug", {})),
    )
