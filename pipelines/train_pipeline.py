# pipelines/train_pipeline.py
import sys
import uuid
from pathlib import Path
from src.core.config_loader import config
from src.infra.data_repository import DataRepository
from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.splitter_service import TimeSeriesSplitter
from src.infra.logger import setup_logger
from src.infra.artifact_manager import ArtifactManager
from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator
from src.core.train_orchestrator import TrainingOrchestrator
from src.core.model_registry import ModelRegistry
from src.core.exceptions import PipelineError
from src.core.data_orchestrator import DataOrchestrator
from loguru import logger

# Initialize Logging
run_id = uuid.uuid4().hex
setup_logger(config.logging, Path(config.project_root))

def main():
    try:
        logger.bind(run_id=run_id).info("System Bootstrapping...")
        project_root = Path(config.project_root)

        # 1. Infrastructure Layer
        # Resolving paths via config and project_root to maintain Portability
        repo = DataRepository(project_root, config.data, run_id=run_id)
        
        artifact_manager = ArtifactManager(
            artifact_cfg=config.artifacts,
            project_root=project_root,
            run_id=run_id
        )

        registry = ModelRegistry(
            registry_dir=project_root / config.artifacts.metadata_path / "registry"
        )

        # 2. Data Engineering Services (Sub-components of DataOrchestrator)
        ingestion = IngestionService(repo, config.data, run_id=run_id)
        
        preprocessor = DataPreprocessor(
            rules=config.data.preprocessing,
            processed_path=project_root / config.paths.processed_data,
            run_id=run_id
        )
        
        engineer = FeatureEngineer(
            rules=config.features,
            target_column=config.data.target_column,
            run_id=run_id
        )

        # 3. Data Orchestration Layer
        # This encapsulates the Ingestion + Preprocessing + Engineering flow
        data_orchestrator = DataOrchestrator(
            repo=repo,
            ingestion=ingestion,
            preprocessor=preprocessor,
            engineer=engineer,
            data_cfg=config.data,
            artifacts_cfg=config.artifacts,
            run_id=run_id,
            project_root=project_root
        )

        # 4. Model Engineering Services
        splitter = TimeSeriesSplitter(
            split_cfg=config.data.split_config,
            target_column=config.data.target_column,
            run_id=run_id
        )

        # 5. Main Training Orchestrator
        # Dependency Injection: Injecting the consolidated DataOrchestrator
        orchestrator = TrainingOrchestrator(
            config=config,
            repo=repo,
            data_orchestrator=data_orchestrator, 
            splitter=splitter,
            artifact_manager=artifact_manager,
            model_factory=ModelWorkerFactory,
            evaluator=ModelEvaluator,
            registry=registry,
            run_id=run_id
        )

        # Execute the lifecycle
        orchestrator.run()

    except PipelineError as e:
        logger.bind(run_id=run_id).error(f"Pipeline failure: {e.message}")
        sys.exit(1)
    except Exception as e:
        logger.bind(run_id=run_id).exception("Unexpected System Failure")
        sys.exit(1)

if __name__ == "__main__":
    main()
