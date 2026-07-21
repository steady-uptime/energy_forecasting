# src/core/ingestion_service.py
import pandas as pd
from src.core.config.schemas import DataConfig
from src.infra.data_repository import DataRepository
from src.core.exceptions import IngestionError, DataValidationError
from loguru import logger


class IngestionService:
    def __init__(self, repo: DataRepository, data_cfg: DataConfig, run_id: str):
        # Invariant: Explicitly typed DataConfig injection
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
            # 1. Fetch raw data
            df = self.repo.read_csv(filename, sep=self.data_cfg.csv_separator).copy()
            
            # 2. Identify the first column (timestamp column)
            first_col = df.columns[0]
            
            # 3. Validate that the first column can be parsed as a timestamp
            df[first_col] = pd.to_datetime(df[first_col], errors="raise")
            
            # 4. Apply precision contract
            precision = self.data_cfg.timestamp_precision
            df[first_col] = df[first_col].astype(f"datetime64[{precision}]")
            
            # 5. Rename it to 'timestamp' only after successful casting
            df = df.rename(columns={first_col: "timestamp"})
            
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
