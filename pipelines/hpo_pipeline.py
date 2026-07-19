# pipelines/hpo_pipeline.py
from typing import List
from loguru import logger
from pathlib import Path

# Orchestrators
from src.core.hpo_orchestrator import HyperparameterOrchestrator
from src.core.comparison_orchestrator import ComparisonOrchestrator

# Services (Dependency Injection)
from core.config import AppConfig
from src.core.comparison_service import CandidateResult 
from src.infra.data_repository import DataRepository
from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.splitter_service import TimeSeriesSplitter
from src.infra.artifact_manager import ArtifactManager
from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator
from src.core.model_registry import ModelRegistry
from src.core.comparison_service import ModelComparisonService
from src.core.validator import DataValidator
from src.core.model_validator import ModelMetricValidator


class HPOTrainingPipeline:
    """
    Wires the Hyperparameter Orchestrator and Comparison Orchestrator 
    into a single production pipeline.
    """
    def __init__(self, config: AppConfig, run_id: str):
        # Separation of concerns: config is the blueprint, run_id is the context
        self.config = config
        self.run_id = run_id
        
        # Instantiate Services (The Dependency Injection Container)
        self.repo = DataRepository()
        self.ingestion = IngestionService()
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.splitter = TimeSeriesSplitter()
        self.artifact_manager = ArtifactManager()
        
        # Cleaned up redundant initialization
        self.model_factory = ModelWorkerFactory(dry_run=self.config.model.dry_run)
        
        self.evaluator = ModelEvaluator()
        self.registry = ModelRegistry()
        self.data_validator = DataValidator(config.data.raw_schema)
        self.metric_validator = ModelMetricValidator()
        
        # Services specific to Comparison
        self.comparison_service = ModelComparisonService()

    def run(self) -> None:
        logger.info(f"--- Starting HPO-to-Comparison Pipeline [{self.run_id}] ---")

        # 1. Initialize HPO Orchestrator
        # This orchestrator now handles Data Engineering internally (Optimization)
        hpo_orchestrator = HyperparameterOrchestrator(
            config=self.config,
            repo=self.repo,
            ingestion=self.ingestion,
            preprocessor=self.preprocessor,
            engineer=self.engineer,
            splitter=self.splitter,
            artifact_manager=self.artifact_manager,
            model_factory=self.model_factory,
            evaluator=self.evaluator,
            registry=self.registry,
            data_validator=self.data_validator
        )

        # 2. Run HPO Loop
        # Returns List[CandidateResult]
        candidates = hpo_orchestrator.run()

        # 3. Initialize Comparison Orchestrator
        comparison_orchestrator = ComparisonOrchestrator(
            config=self.config,
            worker_factory=self.model_factory,
            evaluator=self.evaluator,
            registry=self.registry,
            comparison_service=self.comparison_service,
            data_validator=self.data_validator,
            metric_validator=self.metric_validator
        )

        # 4. Promote Champion (Lightweight Path)
        # In a production HPO workflow, the 'best' result is our primary candidate.
        dummy_baseline = CandidateResult(
            model_id="previous_champion",
            metrics={"mse": 0.0}, 
            hyperparameters={},
            artifact_path="none"
        )

        comparison_orchestrator.run_comparison_from_results(
            run_id=self.run_id,
            baseline=dummy_baseline,
            candidates=candidates
        )

        logger.success(f"Full HPO-to-Comparison pipeline completed for {self.run_id}.")
