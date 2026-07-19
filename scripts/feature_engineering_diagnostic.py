# scripts/feature_engineering_diagnostic.py
"""
Feature Engineering Diagnostic Script
Validates engineered features (total_load, time features) using synthetic debug data.
Ensures schema alignment, correct MT_* aggregation, and proper timestamp-based feature creation.
"""

from loguru import logger
from pathlib import Path
import pandas as pd

from core.config import ConfigLoader
from src.core.feature_engineering import FeatureEngineer
from src.infra.data_repository import DataRepository
from src.core.validator import DataValidator

def main():
    cfg = ConfigLoader.get_config()

    # Load preprocessed debug data
    raw_path = cfg.debug.raw_data_path
    logger.info(f"Debug mode enabled. Loading synthetic raw data from: {raw_path}")

    repo = DataRepository(
        project_root=cfg.project_root,
        data_config=cfg.data,
        run_id="debug_feature_engineering"
    )

    df_raw = repo.read_csv(raw_path, sep=cfg.data.csv_separator)

    # -------------------------------------------------
    # RAW SCHEMA VALIDATION
    # -------------------------------------------------
    DataValidator(cfg.data.raw_schema).validate(df_raw)

    # Apply preprocessing first
    from src.core.preprocessing_service import DataPreprocessor
    pre = DataPreprocessor(
        cfg.data.preprocessing,
        Path("debug_preprocessed"),
        "debug_feature_engineering"
    )
    df_pre = pre.clean_data(df_raw)

    logger.info("=== PREPROCESSED DATA SAMPLE ===")
    logger.info(df_pre.head()) 

    # -------------------------------------------------
    # PREPROCESSED SCHEMA VALIDATION
    # -------------------------------------------------
    DataValidator(cfg.data.preprocessed_schema).validate(df_pre)

    # Initialize FeatureEngineer
    fe = FeatureEngineer(
        rules=cfg.features,
        target_column=cfg.data.target_column,
        run_id="debug_feature_engineering"
    )

    # Apply feature engineering
    df_feat = fe.transform(df_pre)

    logger.info("=== ENGINEERED DATA SAMPLE ===")
    logger.info(df_feat.head())

    # -------------------------------------------------
    # ENGINEERED SCHEMA VALIDATION
    # -------------------------------------------------
    DataValidator(cfg.data.engineered_schema).validate(df_feat)
    
    # -------------------------------------------------
    # Validate total_load correctness
    # -------------------------------------------------
    mt_cols = df_pre.filter(regex=cfg.features.total_load["pattern"]).columns

    df_check = df_pre[mt_cols].sum(axis=1)
    df_engineered = df_feat["total_load"]

    logger.info("=== TOTAL LOAD VALIDATION ===")
    for i in range(5):
        logger.info(f"Row {i}: raw_sum={df_check.iloc[i]}, engineered={df_engineered.iloc[i]}")

    # -------------------------------------------------
    # Validate time features
    # -------------------------------------------------
    logger.info("=== TIME FEATURE VALIDATION ===")
    for i in range(5):
        ts = df_pre["timestamp"].iloc[i]
        logger.info(
            f"Row {i}: ts={ts}, "
            f"hour={df_feat['hour_of_day'].iloc[i]}, "
            f"dow={df_feat['day_of_week'].iloc[i]}, "
            f"is_weekend={df_feat['is_weekend'].iloc[i]}"
        )

    # -------------------------------------------------
    # Validate schema alignment
    # -------------------------------------------------
    logger.info("=== SCHEMA VALIDATION ===")
    logger.info(f"Final columns: {list(df_feat.columns)}")
    logger.info(f"Final dtypes:\n{df_feat.dtypes}")

    # Check for unexpected drops
    dropped = df_pre.columns.difference(df_feat.columns)
    if len(dropped) > 0:
        logger.warning(f"Dropped columns: {list(dropped)}")

    logger.info("Feature engineering diagnostics complete.")


if __name__ == "__main__":
    main()
