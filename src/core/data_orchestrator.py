# src/core/data_orchestrator.py

import pandas as pd
from pathlib import Path
from typing import Protocol
from loguru import logger

from src.core.exceptions import PipelineError
from src.core.validator import DataValidator
from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.config.schemas import DataConfig, ArtifactsConfig


class Repository(Protocol):
    def read_csv(self, path: Path, sep: str) -> pd.DataFrame:
        ...


class DataOrchestrator:
    """
    Orchestrates the full data engineering lifecycle:
    - Ingestion
    - Preprocessing
    - Feature Engineering
    """

    def __init__(
        self,
        repo: Repository,
        ingestion: IngestionService,
        preprocessor: DataPreprocessor,
        engineer: FeatureEngineer,
        data_cfg: DataConfig,
        artifacts_cfg: ArtifactsConfig,
        run_id: str,
        project_root: Path,
    ):
        self.repo = repo
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.engineer = engineer

        self.data_cfg = data_cfg
        self.artifacts_cfg = artifacts_cfg
        self.run_id = run_id
        self.project_root = project_root

        self.logger = logger.bind(
            run_id=self.run_id,
            module="DataOrchestrator"
        )

    # ---------------------------------------------------------
    # Unified entry point expected by TrainingOrchestrator
    # ---------------------------------------------------------
    def run(self, raw_file: Path) -> pd.DataFrame:
        return self.run_pipeline(str(raw_file))

    # ---------------------------------------------------------
    # Full pipeline
    # ---------------------------------------------------------
    def run_pipeline(self, filename: str) -> pd.DataFrame:
        file_path = Path(filename)
        self.logger.info(f"Starting Data Orchestration for: {file_path}")

        try:
            raw_df = self._execute_ingestion(file_path)
            processed_df = self._execute_preprocessing(raw_df)
            engineered_df = self._execute_feature_engineering(processed_df)

            self.logger.info("Data Orchestration completed successfully.")
            return engineered_df

        except Exception as e:
            self.logger.error(f"Data Orchestration failed: {str(e)}")
            raise PipelineError(f"Data Orchestration Failure: {str(e)}")

    # ---------------------------------------------------------
    # Phases
    # ---------------------------------------------------------
    def _execute_ingestion(self, file_path: Path) -> pd.DataFrame:
        self.logger.info("Phase: Ingestion")

        df = self.repo.read_csv(
            file_path,
            sep=self.data_cfg.csv_separator,
        )

        DataValidator(self.data_cfg.raw_schema).validate(df)

        df = self.ingestion.load_raw_energy_data(str(file_path))
        return df

    def _execute_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase: Preprocessing")

        processed_df = self.preprocessor.clean_data(df)

        DataValidator(self.data_cfg.preprocessed_schema).validate(processed_df)
        return processed_df

    def _execute_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase: Feature Engineering")

        engineered_df = self.engineer.transform(df)

        DataValidator(self.data_cfg.engineered_schema).validate(engineered_df)

        self.preprocessor.save_processed_data(
            engineered_df,
            filename=self.artifacts_cfg.output_file,
        )

        return engineered_df
