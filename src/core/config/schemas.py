# src/core/config/schemas.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import UUID

# --- Sub-Configurations ---

@dataclass(frozen=True)
class DebugConfig:
    enable_debug_mode: bool = False
    raw_data_path: Path = Path("")
    description: str = ""

@dataclass(frozen=True)
class ComputeConfig:
    num_workers: int
    gpu_id: int
    memory_limit_gb: int

@dataclass(frozen=True)
class EnvConfig:
    mode: str
    compute: ComputeConfig
    env_mapping: Dict[str, List[str]]

@dataclass(frozen=True)
class PathsConfig:
    raw_data: Path
    processed_data: Path
    interim_data: Path
    logs: Path

@dataclass(frozen=True)
class SplitConfig:
    method: str
    train_end: str
    val_end: str
    test_end: str

@dataclass(frozen=True)
class FrequencyConversionConfig:
    enabled: bool
    factor: float

@dataclass(frozen=True)
class ImputationConfig:
    strategy: str

@dataclass(frozen=True)
class PreprocessingConfig:
    frequency_conversion: FrequencyConversionConfig
    dst_strategy: str
    convert_to_numeric: bool
    imputation: ImputationConfig

@dataclass(frozen=True)
class SchemaItem:
    column: Optional[str] = None
    column_pattern: Optional[str] = None
    type: str = ""

@dataclass(frozen=True)
class AggregateFeatureConfig:
    method: str
    pattern: str

@dataclass(frozen=True)
class TimeFeatureFlags:
    hour_of_day: bool
    day_of_week: bool
    is_weekend: bool

@dataclass(frozen=True)
class FeaturesConfig:
    aggregate: AggregateFeatureConfig
    time: TimeFeatureFlags

@dataclass(frozen=True)
class DataConfig:
    csv_separator: str
    timestamp_precision: str
    target_column: str
    feature_column_pattern: str
    time_features: List[str]
    split_config: SplitConfig
    preprocessing: PreprocessingConfig
    raw_schema: List[SchemaItem]
    preprocessed_schema: List[SchemaItem]
    engineered_schema: List[SchemaItem]

@dataclass(frozen=True)
class HPOConfig:
    strategy: str
    parameters: Dict[str, Any]

@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_kind: str
    dry_run: bool
    params: Dict[str, Any]
    hpo: HPOConfig

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int

@dataclass(frozen=True)
class ArtifactsConfig:
    base_path: Path
    models_path: Path
    metadata_path: Path
    reports_path: Path
    monitoring_path: Path
    input_file: Path
    output_file: Path

@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    rotation_mb: str
    retention_days: int
    file_path: Path

@dataclass(frozen=True)
class ReportingConfig:
    drift_report_format: str

@dataclass(frozen=True)
class MonitoringConfig:
    primary_metric: str
    drift_threshold: float
    thresholds: Dict[str, float]
    reporting: ReportingConfig

@dataclass(frozen=True)
class EvaluationConfig:
    metrics: List[str]
    thresholds: Dict[str, float]
    report_format: str

# --- Root Configuration ---

@dataclass(frozen=True)
class AppConfig:
    project_name: str
    version: str
    run_id: UUID           # Top level attribute
    project_root: Path     # Top level attribute
    env: EnvConfig
    data: DataConfig
    paths: PathsConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    evaluation: EvaluationConfig
    debug: DebugConfig

    def validate(self) -> None:
        if self.training.learning_rate <= 0:
            raise ValueError(f"Training learning_rate must be positive. Got: {self.training.learning_rate}")
        if self.training.batch_size <= 0:
            raise ValueError(f"Training batch_size must be greater than 0. Got: {self.training.batch_size}")
        
        valid_models = ["random_forest_regressor", "xgboost_regressor", "gradient_boosting_regressor"]
        if self.model.model_kind not in valid_models:
            raise ValueError(f"model_kind must be one of {valid_models}. Got: {self.model.model_kind}")
        
        print("System: Configuration validation passed.")
