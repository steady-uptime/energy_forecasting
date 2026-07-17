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
    
            # 3. Dynamic Timestamp Detection
            # The real LD2011_2014.txt always has the timestamp in the first column ("Unnamed: 0")
            timestamp_col = df.columns[0]
            target_col = "timestamp"
    
            logger.bind(
                module="IngestionService",
                run_id=self.run_id,
                detected_timestamp_col=timestamp_col
            ).info("Detected timestamp column")
    
            df = df.rename(columns={timestamp_col: target_col})
    
            # 4. Domain Logic: Type Casting
            df[target_col] = pd.to_datetime(df[target_col], errors="raise")
    
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
