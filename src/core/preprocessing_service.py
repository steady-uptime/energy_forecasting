# src/core/preprocessing_service.py
import pandas as pd
from pathlib import Path
from src.core.exceptions import PreprocessingError
from src.core.config.schemas import PreprocessingConfig
from loguru import logger


class DataPreprocessor:
    """
    Worker: Handles Data Sanitization.
    Responsibilities: Cleaning, Numeric Conversion, Imputation.
    
    Invariant Compliance:
    - Receives injected PreprocessingConfig dataclass.
    - No dict/kwargs access for configuration.
    - Single responsibility: Data cleaning and transformation.
    """

    def __init__(self, rules: PreprocessingConfig, processed_path: Path, run_id: str):
        # Dependency Injection: typed PreprocessingConfig
        self.rules = rules
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

            # Timestamp contract enforcement only
            if "timestamp" not in df.columns:
                raise PreprocessingError("Timestamp column missing after ingestion")

            # -----------------------------
            # Convert MT_* columns to numeric
            # -----------------------------
            if self.rules.convert_to_numeric:
                mt_cols = df.filter(regex=r"^MT_\d{3}$").columns

                df[mt_cols] = df[mt_cols].replace({",": "."}, regex=True)

                logger.bind(
                    module="DataPreprocessor",
                    run_id=self.run_id,
                    columns=list(mt_cols)
                ).debug("Normalized decimal commas in MT_* columns")

                df[mt_cols] = df[mt_cols].apply(pd.to_numeric, errors="coerce")

                logger.bind(
                    module="DataPreprocessor",
                    run_id=self.run_id,
                    numeric_columns=list(mt_cols)
                ).debug("Converted MT_* columns to numeric")

            # -----------------------------
            # Frequency Conversion (kW → kWh)
            # -----------------------------
            freq_cfg = self.rules.frequency_conversion
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
            # Imputation
            # -----------------------------
            strategy = self.rules.imputation.strategy

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
                    "rules": self.rules,
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
