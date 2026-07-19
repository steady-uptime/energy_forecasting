# scripts/ingestion_diagnostic.py
"""
Ingestion Diagnostic Script
Validates raw ingestion using synthetic test data defined in configs/debug.yaml.
This script isolates the ingestion layer and ensures raw data is loaded correctly
before preprocessing or feature engineering are applied.
"""

from loguru import logger

from core.config import ConfigLoader
from src.infra.data_repository import DataRepository

def main():
    cfg = ConfigLoader.get_config()

    raw_path = cfg.debug.raw_data_path
    logger.info(f"Debug mode enabled. Loading synthetic raw data from: {raw_path}")

    repo = DataRepository(
        project_root=cfg.project_root,
        data_config=cfg.data,
        run_id="debug_ingestion"
    )

    df_raw = repo.read_csv(raw_path, sep=cfg.data.csv_separator)

    logger.info("=== RAW INGESTION DIAGNOSTICS ===")

    # Shape
    logger.info(f"Raw shape: {df_raw.shape}")

    # Dtypes
    logger.info("Raw dtypes:")
    logger.info(df_raw.dtypes)

    # First two rows
    logger.info("First two rows:")
    logger.info(df_raw.head(2))

    # Timestamp column
    timestamp_col = df_raw.columns[0]
    logger.info(f"Timestamp column detected: {timestamp_col}")
    logger.info(f"Timestamp dtype: {df_raw[timestamp_col].dtype}")

    # MT_* columns
    mt_cols = [c for c in df_raw.columns if c.startswith("MT_")]
    logger.info(f"Detected {len(mt_cols)} MT_* columns: {mt_cols}")

    # European decimal comma check
    logger.info("Checking for European decimal commas in MT_* columns...")
    for col in mt_cols:
        col_values = df_raw[col].astype(str)

        if col_values.str.contains(",").any():
            logger.info(f"{col}: European decimal comma detected.")
        elif (col_values.astype(float) == 0).all():
            logger.info(f"{col}: Column contains only zeros — no comma expected.")
        elif col_values.str.match(r"^\d+$").all():
            logger.info(f"{col}: Column contains only whole numbers — no comma expected.")
        else:
            logger.warning(f"{col}: No comma found, but column contains non-zero values.")

    logger.info("Ingestion diagnostics complete.")


if __name__ == "__main__":
    main()
