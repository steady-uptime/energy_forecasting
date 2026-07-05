# pipelines/train_pipeline.py
import sys
from pathlib import Path
from src.core.config_loader import config
from src.infra.data_repository import DataRepository
from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.splitter_service import TimeSeriesSplitter
from src.core.validator import DataValidator
from src.infra.logger import setup_logger
from src.infra.artifact_manager import ArtifactManager
from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator
from src.core.exceptions import PipelineError
from loguru import logger
import uuid

run_id = uuid.uuid4().hex

# Initialize logging with project_root from the Singleton Config
setup_logger(config.logging, Path(config.project_root))

def main():
    try:
        logger.bind(run_id=run_id).info("Configuration loaded successfully.")

        # --- Configuration Slices ---
        data_cfg = config.data
        artifacts_cfg = config.artifacts
        model_cfg = config.model
        pre_cfg = data_cfg.preprocessing

        # Schema Gatekeepers
        raw_schema = data_cfg.raw_schema
        pre_schema = data_cfg.preprocessed_schema
        eng_schema = data_cfg.engineered_schema

        logger.bind(run_id=run_id).info("Pipeline started")

        project_root = Path(config.project_root)
        repo = DataRepository(project_root, data_cfg, run_id=run_id) 
        ingestion = IngestionService(repo, data_cfg, run_id=run_id)

        logger.bind(run_id=run_id).info("Loading raw data")
        raw_file = str(Path(config.paths.raw_data) / artifacts_cfg.input_file)
        raw_data = ingestion.load_raw_energy_data(raw_file)

        # Gatekeeper: Verify Raw Data Integrity
        logger.bind(run_id=run_id).info("Validating Raw Data Contract...")
        DataValidator(raw_schema).validate(raw_data)

        logger.bind(run_id=run_id).info("Preprocessing started...")
        processed_path_obj = project_root / config.paths.processed_data
        preprocessor = DataPreprocessor(
            rules=pre_cfg,
            processed_path=processed_path_obj,
            run_id=run_id
        )
        sanitized_data = preprocessor.clean_data(raw_data)

        logger.bind(run_id=run_id).info("Validating Preprocessed Data Contract...")
        DataValidator(pre_schema).validate(sanitized_data)

        logger.bind(run_id=run_id).info("Feature Engineering started...")
        engineer = FeatureEngineer(
            rules=config.features,
            target_column=data_cfg.target_column,
            run_id=run_id
        )
        engineered_data = engineer.transform(sanitized_data)

        logger.bind(run_id=run_id).info("Validating Engineered Data Contract...")
        DataValidator(eng_schema).validate(engineered_data)

        logger.bind(run_id=run_id).info("Saving Processed data...")
        preprocessor.save_processed_data(
            engineered_data,
            filename=artifacts_cfg.output_file
        )

        logger.bind(run_id=run_id).info("Data Splitter - Time Series - started...")
        splitter = TimeSeriesSplitter(
            split_cfg=data_cfg.split_config,
            target_column=data_cfg.target_column,
            run_id=run_id
        )
        X_train, y_train, X_test, y_test, _ = splitter.split(engineered_data)

        # Dependency Injection & Factory Pattern:
        # Instantiate Infrastructure
        artifact_manager = ArtifactManager(
            artifact_cfg=config.artifacts, 
            project_root=config.paths.project_root,
            run_id=run_id
        )
        
        # --- Model Engineering Phase ---
        logger.bind(run_id=run_id).info("--- Starting Model Engineering Phase ---")
        worker = ModelWorkerFactory.get_worker(model_cfg, run_id=run_id)
        trained_model = worker.train(X_train, y_train)
        
        logger.bind(run_id=run_id).info("Saving Model")
        model_path = artifact_manager.save_model(trained_model, model_cfg.name)
        
        logger.bind(run_id=run_id).info("--- Starting Evaluation Phase ---")
        metrics = ModelEvaluator.evaluate(trained_model, X_test, y_test, run_id=run_id)
        
        logger.bind(run_id=run_id).info("Saving Model Metrics")
        artifact_manager.save_metrics(
            metrics=metrics, 
            model_name=model_cfg.name, 
            model_uri=str(model_path),
            hyperparameters=model_cfg.params,
            training_params=config.training
        )
        
        logger.bind(run_id=run_id).info("Pipeline executed successfully.")
        
    except PipelineError as e:
        logger.bind(run_id=run_id, context=e.context).error(f"Pipeline failure: {e.message}")
        sys.exit(1)
    except Exception as e:
        logger.bind(run_id=run_id, error=str(e)).critical("Unexpected System Failure")
        logger.bind(run_id=run_id).exception("Stack trace for debugging")
        sys.exit(1)

if __name__ == "__main__":
    main()
