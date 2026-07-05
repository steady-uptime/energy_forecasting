# /src/core/config_loader.py
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

# --- Configuration Schemas (Contracts) ---

@dataclass(frozen=True)
class EnvConfig:
    mode: str
    compute: Dict[str, Any]

@dataclass(frozen=True)
class DataConfig:
    target_column: str
    feature_column_pattern: str
    time_features: List[str]
    split_config: Dict[str, Any]
    preprocessing: Dict[str, Any]

    raw_schema: List[Dict[str, str]]
    preprocessed_schema: List[Dict[str, str]]
    engineered_schema: List[Dict[str, str]]

    csv_separator: str = ","
    timestamp_precision: str = "ns"
    raw_columns: Dict[str, str] = None

    def validate(self):
        if not self.target_column:
            raise ValueError("DataConfig: target_column must not be empty.")
        if not self.feature_column_pattern:
            raise ValueError("DataConfig: feature_column_pattern must be defined.")
        if self.raw_columns is None:
            raise ValueError("DataConfig: raw_columns must be defined in data.yaml.")
        
        # Optional: Add validation for schemas to ensure they aren't empty
        if not self.raw_schema:
            raise ValueError("DataConfig: raw_schema must be defined.")
        if not self.preprocessed_schema:
            raise ValueError("DataConfig: preprocessed_schema must be defined.")
        if not self.engineered_schema:
            raise ValueError("DataConfig: engineered_schema must be defined.")

@dataclass(frozen=True)
class FeaturesConfig:
    total_load: Dict[str, Any]
    time_features: Dict[str, bool]

@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str
    params: Dict[str, Any]

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int

    def validate(self):
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"TrainingConfig: Invalid learning_rate ({self.learning_rate}).")
        if self.batch_size <= 0:
            raise ValueError(f"TrainingConfig: Invalid batch_size ({self.batch_size}).")

@dataclass(frozen=True)
class PathsConfig:
    raw_data: str
    processed_data: str
    interim_data: str
    logs: str
    project_root: Path

@dataclass(frozen=True)
class ArtifactConfig:
    input_file: str
    output_file: str
    base_path: str
    models_path: str
    metadata_path: str
    reports_path: str

@dataclass(frozen=True)
class LoggingConfig:
    file_path: str
    level: str
    rotation_mb: str
    retention_days: str
    format: str = "{timestamp} | {level} | {message} | {module}"

@dataclass(frozen=True)
class AppConfig:
    project_name: str
    version: str
    env: EnvConfig
    data: DataConfig
    paths: PathsConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactConfig
    logging: LoggingConfig
    project_root: str

    def validate(self):
        self.data.validate()
        self.training.validate()

# --- Singleton Loader ---

class ConfigLoader:
    """
    Singleton class that aggregates multiple YAML configuration files 
    into a single, type-safe AppConfig object.
    """
    _instance: Optional["ConfigLoader"] = None
    _config: Optional[AppConfig] = None
    project_root: Optional[Path] = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._initialize_config()
        return cls._instance

    @classmethod
    def _initialize_config(cls) -> None:
        # Law #1 & #3: Dynamic root calculation
        if cls.project_root is None:
            cls.project_root = Path(__file__).resolve().parent.parent.parent
        
        logger.info(f"System: Initializing Configuration from {cls.project_root}/configs")

        try:
            # 1. Load all yaml files in the configs directory
            config_dir = cls.project_root / "configs"
            combined_raw_data = {}

            for config_file in config_dir.glob("*.yaml"):
                # Skip the env_mapping logic if it's in config.yaml, 
                # but we need to handle it specially.
                with open(config_file, "r") as f:
                    file_content = yaml.safe_load(f)
                    if file_content:
                        # Merge content into a master dictionary
                        # We use a flat merge here, but you can adjust logic 
                        # if keys overlap significantly.
                        combined_raw_data.update(file_content)

            # 2. Inject Project Root
            combined_raw_data["project_root"] = (cls.project_root)

            # 3. Apply Environment Overrides
            if "env_mapping" in combined_raw_data:
                combined_raw_data = cls._apply_env_overrides(combined_raw_data)

            # 4. Map to Dataclasses
            # Note: Because your YAMLs have nested keys (like 'data:', 'model:'),
            # we extract those specifically to pass into the dataclasses.
            
            cls._config = AppConfig(
                project_name=combined_raw_data["project_name"],
                version=combined_raw_data["version"],
                env=EnvConfig(**combined_raw_data["env"]),
                data=DataConfig(**combined_raw_data["data"]),
                paths=PathsConfig(
                    **combined_raw_data["paths"], 
                    project_root=cls.project_root
                ),
                features=FeaturesConfig(**combined_raw_data["features"]),
                model=ModelConfig(**combined_raw_data["model"]),
                training=TrainingConfig(**combined_raw_data["training"]),
                artifacts=ArtifactConfig(**combined_raw_data["artifacts"]),
                logging=LoggingConfig(**combined_raw_data["logging"]),
                project_root=combined_raw_data["project_root"]
            )

            # 5. Contract Validation
            cls._config.validate()
            logger.info("System: Configuration aggregated and validated successfully.")

        except Exception as e:
            logger.error(f"System: Configuration Initialization Failed: {e}")
            raise

    @staticmethod
    def _apply_env_overrides(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        env_mapping = raw_data.get("env_mapping", {})
        for env_var, mapping_values in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                section, key = mapping_values 
                if section in raw_data and key in raw_data[section]:
                    target_type = type(raw_data[section][key])
                    if target_type == bool:
                        overridden_value = env_value.lower() in ("true", "1", "yes")
                    elif target_type in (int, float):
                        overridden_value = target_type(env_value)
                    else:
                        overridden_value = env_value
                    
                    raw_data[section][key] = overridden_value
                    logger.info(f"System: Overriding {section}.{key} with EnvVar {env_var}")
        return raw_data

    @classmethod
    def get_config(cls) -> AppConfig:
        if cls._config is None:
            ConfigLoader() 
        return cls._config

# Global instance
config = ConfigLoader().get_config()
