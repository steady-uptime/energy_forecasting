# src/core/config/factories/data_factory.py
from ..schemas import (
    DataConfig, SplitConfig, PreprocessingConfig, 
    FrequencyConversionConfig, ImputationConfig, 
    SchemaItem, AggregateFeatureConfig, TimeFeatureFlags,
    FeaturesConfig
)

def build_aggregate_feature_config(raw: dict) -> AggregateFeatureConfig:
    return AggregateFeatureConfig(**raw)

def build_time_feature_flags(raw: dict) -> TimeFeatureFlags:
    return TimeFeatureFlags(**raw)

def build_features_config(raw: dict) -> FeaturesConfig:
    return FeaturesConfig(
        aggregate=build_aggregate_feature_config(raw["total_load"]),
        time=build_time_feature_flags(raw["time_features"])
    )

def build_split_config(raw: dict) -> SplitConfig:
    return SplitConfig(**raw)

def build_frequency_conversion_config(raw: dict) -> FrequencyConversionConfig:
    return FrequencyConversionConfig(**raw)

def build_imputation_config(raw: dict) -> ImputationConfig:
    return ImputationConfig(**raw)

def build_preprocessing_config(raw: dict) -> PreprocessingConfig:
    """
    Handles nested preprocessing logic.
    """
    return PreprocessingConfig(
        frequency_conversion=build_frequency_conversion_config(raw["frequency_conversion"]),
        dst_strategy=raw["dst_strategy"],
        convert_to_numeric=raw["convert_to_numeric"],
        imputation=build_imputation_config(raw["imputation"])
    )

def build_data_config(raw: dict) -> DataConfig:
    """
    Handles complex nesting and list-of-dict to list-of-dataclass mapping.
    """
    # Transform lists of dicts into lists of SchemaItem objects
    raw_schema = [SchemaItem(**item) for item in raw["raw_schema"]]
    preprocessed_schema = [SchemaItem(**item) for item in raw["preprocessed_schema"]]
    engineered_schema = [SchemaItem(**item) for item in raw["engineered_schema"]]

    return DataConfig(
        csv_separator=raw["csv_separator"],
        timestamp_precision=raw["timestamp_precision"],
        target_column=raw["target_column"],
        feature_column_pattern=raw["feature_column_pattern"],
        time_features=raw["time_features"],
        split_config=build_split_config(raw["split_config"]),
        preprocessing=build_preprocessing_config(raw["preprocessing"]),
        raw_schema=raw_schema,
        preprocessed_schema=preprocessed_schema,
        engineered_schema=engineered_schema
    )
