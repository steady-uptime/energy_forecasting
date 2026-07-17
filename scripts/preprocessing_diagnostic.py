"""
Preprocessing Diagnostic Script
Validates preprocessing transformations on synthetic test data defined in configs/debug.yaml.
This script isolates the preprocessing layer and ensures numeric normalization, timestamp parsing,
and schema alignment behave correctly before feature engineering is applied.
"""

from loguru import logger
from pathlib import Path
from src.core.config_loader import ConfigLoader
from src.infra.data_repository import DataRepository
from src.core.preprocessing_service import DataPreprocessor

def main():
    cfg = ConfigLoader.get_config()

    raw_path = cfg.debug.raw_data_path
    logger.info(f"Debug mode enabled. Loading synthetic raw data from: {raw_path}")

    # Initialize repository
    repo = DataRepository(
        project_root=cfg.project_root,
        data_config=cfg.data,
        run_id="debug_preprocessing"
    )

    # Load raw dataframe
    df_raw = repo.read_csv(raw_path, sep=cfg.data.csv_separator)

    logger.info("=== PREPROCESSING DIAGNOSTICS ===")
    logger.info("Raw head():")
    logger.info(df_raw.head())

    # Build rules dict directly from config
    rules = cfg.data.preprocessing
    
    # Initialize DataPreprocessor
    pre = DataPreprocessor(
        rules,
        Path("debug_preprocessed"),   # dummy path
        "debug_preprocessing"
    )
    
    # Apply preprocessing
    df_pre = pre.clean_data(df_raw)

    # TEMPORARY: Inspect raw vs preprocessed values for MT_* columns
    mt_cols = [c for c in df_raw.columns if c.startswith("MT_")]

    for col in mt_cols:
        print("\n==============================")
        print(f"Column: {col}")
        print("Raw values:")
        print(df_raw[col].head(10).to_string())

        print("\nPreprocessed values:")
        print(df_pre[col].head(10).to_string())
        print("==============================\n")


    logger.info("=== AFTER PREPROCESSING ===")
    logger.info(f"Preprocessed shape: {df_pre.shape}")
    logger.info("Preprocessed dtypes:")
    logger.info(df_pre.dtypes)
    logger.info("Preprocessed head():")
    logger.info(df_pre.head())

    # Validate MT_* numeric conversion
    mt_cols = [c for c in df_pre.columns if c.startswith("MT_")]
    logger.info(f"Detected {len(mt_cols)} MT_* columns: {mt_cols}")

    for col in mt_cols:
        raw_vals = df_raw[col].astype(str)
        pre_vals = df_pre[col]

        logger.info(f"--- Column {col} ---")
        logger.info(f"Raw sample: {raw_vals.iloc[0]}")
        logger.info(f"Preprocessed sample: {pre_vals.iloc[0]}")

        # Check for NaN
        if pre_vals.isna().any():
            logger.warning(f"{col}: Contains NaN values after preprocessing.")

        # Check for unexpected zeros
        if (pre_vals == 0).any() and not (raw_vals.str.contains("0").all()):
            logger.warning(f"{col}: Zero values appear unexpectedly after preprocessing.")

        # Check numeric conversion
        try:
            float(pre_vals.iloc[0])
        except Exception:
            logger.error(f"{col}: Failed to convert to float.")

    logger.info("Preprocessing diagnostics complete.")


if __name__ == "__main__":
    main()
