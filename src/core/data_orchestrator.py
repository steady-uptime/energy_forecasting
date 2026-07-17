# src/core/data_orchestrator.py
import pandas as pd
from typing import Any, Dict
from pathlib import Path
from loguru import logger
from datetime import datetime, UTC

from src.core.exceptions import PipelineError
from src.core.validator import DataValidator
from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.run_metadata import PhaseMetadata

class DataOrchestrator:
    """
    Orchestrator for the Data Engineering lifecycle.
    
    Responsibilities:
    - Coordinate Ingestion, Preprocessing, and Feature Engineering.
    - Enforce Data Contracts at every transition gate.
    - Abstract data engineering complexity from the Model Training lifecycle.
    - Persist intermediate artifacts (e.g., sanitized data).
    """

    def __init__(
        self,
        repo: Any,
        ingestion: IngestionService,
        preprocessor: DataPreprocessor,
        engineer: FeatureEngineer,
        data_cfg: Any,
        artifacts_cfg: Any,
        run_id: str,
        project_root: Path
    ):
        self.repo = repo
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.engineer = engineer
        self.data_cfg = data_cfg
        self.artifacts_cfg = artifacts_cfg
        self.run_id = run_id
        self.project_root = project_root
        
        self.logger = logger.bind(run_id=self.run_id, module="DataOrchestrator")

    def run_pipeline(self, filename: str) -> pd.DataFrame:
        """
        Executes the full data engineering pipeline.
        
        Returns:
            pd.DataFrame: The final engineered feature matrix.
        """
        self.logger.info(f"Starting Data Orchestration for: {filename}")
        
        try:
            # Phase 1: Ingestion
            raw_df = self._execute_ingestion(filename)
            
            # Phase 2: Preprocessing
            processed_df = self._execute_preprocessing(raw_df)
            
            # Phase 3: Feature Engineering
            engineered_df = self._execute_feature_engineering(processed_df)
            
            self.logger.info("Data Orchestration completed successfully.")
            return engineered_df

        except Exception as e:
            self.logger.error(f"Data Orchestration failed: {str(e)}")
            raise PipelineError(f"Data Orchestration Failure: {str(e)}")

    def _execute_ingestion(self, filename: str) -> pd.DataFrame:
        self.logger.info("Phase: Ingestion")
        
        # Infrastructure access
        df = self.repo.read_csv(filename, sep=self.data_cfg.csv_separator)
        
        # Contract Gate
        DataValidator(self.data_cfg.raw_schema).validate(df)
        
        # Domain Logic
        df = self.ingestion.load_raw_energy_data(filename)
        return df

    def _execute_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase: Preprocessing")
        
        processed_df = self.preprocessor.clean_data(df)
        
        # Contract Gate
        DataValidator(self.data_cfg.preprocessed_schema).validate(processed_df)
        
        return processed_df

    def _execute_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase: Feature Engineering")
        
        engineered_df = self.engineer.transform(df)
        
        # Contract Gate
        DataValidator(self.data_cfg.engineered_schema).validate(engineered_df)
        
        # Artifact Persistence (Save to disk)
        self.preprocessor.save_processed_data(
            engineered_df,
            filename=self.artifacts_cfg.output_file
        )
        
        return engineered_df
