# pipelines/monitor_pipeline.py

import sys
import uuid
from pathlib import Path

from loguru import logger

from src.core.config_loader import config
from src.infra.logger import setup_logger
from src.infra.data_repository import DataRepository
from src.infra.artifact_manager import ArtifactManager

from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.splitter_service import TimeSeriesSplitter
from src.core.evaluator import ModelEvaluator
from src.core.model_registry import ModelRegistry
from src.core.monitoring_service import MonitoringService

from src.core.exceptions import MonitoringError


run_id = uuid.uuid4().hex
setup_logger(config.logging, Path(config.project_root))


def main():
    logger.bind(run_id=run_id).info("System Bootstrapping (Monitoring Pipeline)...")

    try:
        project_root = Path(config.project_root)

        # -----------------------------------------------------
        # Infrastructure Layer
        # -----------------------------------------------------
        repo = DataRepository(
            project_root=project_root,
            data_config=config.data,
            run_id=run_id
        )

        artifact_manager = ArtifactManager(
            artifact_cfg=config.artifacts,
            project_root=config.paths.project_root,
            run_id=run_id
        )

        registry = ModelRegistry(
            registry_dir=Path(config.artifacts.metadata_path) / "registry"
        )

        # -----------------------------------------------------
        # Service Layer (same DI pattern as training)
        # -----------------------------------------------------
        ingestion_service = IngestionService(
            repo=repo,
            data_cfg=config.data,
            run_id=run_id
        )

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

        splitter = TimeSeriesSplitter(
            split_cfg=config.data.split_config,
            target_column=config.data.target_column,
            run_id=run_id
        )

        evaluator = ModelEvaluator

        monitoring_cfg = config.monitoring

        # -----------------------------------------------------
        # Monitoring Service (DI)
        # -----------------------------------------------------
        monitoring_service = MonitoringService(
            config=config,
            repo=repo,
            ingestion=ingestion_service,
            preprocessor=preprocessor,
            engineer=engineer,
            splitter=splitter,
            artifact_manager=artifact_manager,
            evaluator=evaluator,
            registry=registry,
            monitoring_cfg=monitoring_cfg,
            run_id=run_id
        )

        # -----------------------------------------------------
        # Execute Monitoring
        # -----------------------------------------------------
        drift_detected = monitoring_service.run()

        if drift_detected:
            logger.bind(run_id=run_id).warning("Drift detected — retrain trigger emitted")
        else:
            logger.bind(run_id=run_id).info("No drift detected")

    except MonitoringError as e:
        logger.bind(run_id=run_id).error(f"Monitoring failure: {e.message}")
        sys.exit(1)

    except Exception as e:
        logger.bind(run_id=run_id).exception("Unexpected Monitoring Pipeline Failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
