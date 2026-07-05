# src/core/ingestion_service.py
import pandas as pd
from src.core.config_loader import DataConfig
from src.infra.data_repository import DataRepository
from src.core.exceptions import IngestionError, DataValidationError
from loguru import logger

class IngestionService:
    def __init__(self, repo: DataRepository, data_cfg: DataConfig, run_id: str):
        self.repo = repo
        self.data_cfg = data_cfg
        self.run_id = run_id

    def load_raw_energy_data(self, filename: str) -> pd.DataFrame:
        logger.bind(
            module="IngestionService",
            run_id=self.run_id,
            filename=filename
        ).info("Starting raw energy data ingestion")

        try:
            # 1. Fetch data via Infrastructure
            df = self.repo.read_csv(filename, sep=self.data_cfg.csv_separator)

            # 2. Immutability: Work on a copy
            df = df.copy()

            # 3. Config-Driven Renaming
            target_col = "timestamp"
            source_col = self.data_cfg.raw_columns["timestamp"]

            if source_col in df.columns:
                df = df.rename(columns={source_col: target_col})
            else:
                logger.bind(
                    module="IngestionService",
                    run_id=self.run_id,
                    missing_column=source_col
                ).warning("Source timestamp column missing")

                raise DataValidationError(
                    f"Required timestamp column '{source_col}' missing",
                    context={"expected_column": source_col, "columns": list(df.columns)}
                )

            # 4. Domain Logic: Type Casting
            df[target_col] = pd.to_datetime(df[target_col])

            precision = self.data_cfg.timestamp_precision
            df[target_col] = df[target_col].astype(f"datetime64[{precision}]")

            logger.bind(
                module="IngestionService",
                run_id=self.run_id,
                rows=len(df),
                cols=len(df.columns),
                precision=precision
            ).info("Ingestion successful")

            return df

        except Exception as e:
            logger.bind(
                module="IngestionService",
                run_id=self.run_id,
                error=str(e)
            ).error("Ingestion failure")

            raise IngestionError(
                "Failed during ingestion service",
                context={
                    "filename": filename,
                    "csv_separator": self.data_cfg.csv_separator,
                    "timestamp_precision": self.data_cfg.timestamp_precision
                }
            ) from e
