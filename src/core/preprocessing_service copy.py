# src/core/preprocessing_service.py
import pandas as pd
from pathlib import Path
from src.core.exceptions import PreprocessingError
from src.core.config.schemas import PreprocessingConfig
from loguru import logger

class DataPreprocessor:
    """
    Worker: Handles Data Sanitization.
    Responsibilities: Cleaning, Encoding, Numeric Conversion, Imputation.
    
    Invariant Compliance:
    - Receives injected PreprocessingConfig dataclass.
    - No dict/kwargs access for configuration.
    - Single responsibility: Data cleaning and transformation.
    """

    def __init__(self, config: PreprocessingConfig, processed_path: Path, run_id: str):
        # Dependency Injection: The config object is passed from the Orchestrator
        self.config = config
        self.processed_path = processed_path
        self.run_id = run_id

        logger.bind(
            module="DataPreprocessor",
            run_id=self.run_id,
            processed_path=str(processed_path)
        ).info("DataPreprocessor initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.bind(
            module="DataPreprocessor",
            run_id=self.run_id,
            rows=len(df),
            cols=len(df.columns)
        ).info("Starting data cleaning phase")

        try:
            df = df.copy()

            # -----------------------------
            # Drop Columns
            # -----------------------------
            actual_drops = [col for col in self.config.drop_columns if col in df.columns]
            df = df.drop(columns=actual_drops)

            logger.bind(
                module="DataPreprocessor",
                run_id=self.run_id,
                dropped_columns=actual_drops
            ).debug("Dropped columns")

            # -----------------------------
            # Timestamp normalization
            # -----------------------------
            # Assume first column is the timestamp, as in ingestion
            ts_col = df.columns[0]

            # Rename to canonical name expected by FeatureEngineer
            if ts_col != "timestamp":
                df = df.rename(columns={ts_col: "timestamp"})

            # Parse to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

            logger.bind(
                module="DataPreprocessor",
                run_id=self.run_id,
                timestamp_column="timestamp",
                dtype=str(df["timestamp"].dtype)
            ).debug("Normalized timestamp column")

            # -----------------------------
            # Convert MT_* columns to numeric
            # -----------------------------
            # New Access: config.convert_to_numeric
            if self.config.convert_to_numeric:
                mt_cols = df.filter(regex=r"^MT_\d{3}$").columns
            
                # Normalize decimal commas
                df[mt_cols] = df[mt_cols].replace({",": "."}, regex=True)
            
                logger.bind(
                    module="DataPreprocessor",
                    run_id=self.run_id,
                    columns=list(mt_cols)
                ).debug("Normalized decimal commas in MT_* columns")
            
                # Convert to numeric
                df[mt_cols] = df[mt_cols].apply(pd.to_numeric, errors="coerce")
            
                logger.bind(
                    module="DataPreprocessor",
                    run_id=self.run_id,
                    numeric_columns=list(mt_cols)
                ).debug("Converted MT_* columns to numeric")

            # -----------------------------
            # Frequency Conversion (kW → kWh)
            # -----------------------------
            # New Access: config.frequency_conversion (Nested Dataclass)
            freq_cfg = self.config.frequency_conversion
            if freq_cfg.enabled:
                factor = freq_cfg.factor
                mt_cols = df.filter(regex=r"^MT_\d{3}$").columns
                df[mt_cols] = df[mt_cols] * factor 

                logger.bind(
                    module="DataPreprocessor",
                    run_id=self.run_id,
                    factor=factor,
                    columns=list(mt_cols)
                ).debug("Applied frequency conversion")

            # -----------------------------
            # Encoding
            # -----------------------------
            for col, mapping in self.config.encoding.mappings.items():
                if col in df.columns:
                    df[col] = df[col].map(mapping)
                    logger.bind(
                        module="DataPreprocessor",
                        run_id=self.run_id,
                        column=col
                    ).debug("Applied encoding")

            # -----------------------------
            # Imputation
            # -----------------------------
            # New Access: config.imputation (Nested Dataclass)
            strategy = self.config.imputation.strategy

            if strategy == "forward_fill":
                df = df.ffill()
            elif strategy == "median":
                df = df.fillna(df.median(numeric_only=True))
            elif strategy == "mode":
                df = df.fillna(df.mode().iloc[0])

            logger.bind(
                module="DataPreprocessor",
                run_id=self.run_id,
                strategy=strategy
            ).debug("Applied imputation")

            logger.bind(
                module="DataPreprocessor",
                run_id=self.run_id,
                rows=len(df),
                cols=len(df.columns)
            ).info("Data cleaning completed")

            return df

        except Exception as e:
            logger.bind(
                module="DataPreprocessor",
                run_id=self.run_id,
                error=str(e)
            ).error("Preprocessing failure")

            raise PreprocessingError(
                "Failed during preprocessing",
                context={
                    "config": self.config,
                    "processed_path": str(self.processed_path)
                }
            ) from e

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        try:
            self.processed_path.mkdir(parents=True, exist_ok=True)
            save_path = self.processed_path / filename
            df.to_csv(save_path, index=False)

            logger.bind(
                module="DataPreprocessor",
                run_id=self.run_id,
                save_path=str(save_path)
            ).info("Successfully saved sanitized data")

        except Exception as e:
            raise PreprocessingError(
                "Failed to save processed data",
                context={"save_path": str(save_path)}
            ) from e
