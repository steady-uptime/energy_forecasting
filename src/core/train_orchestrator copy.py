# src/core/train_orchestrator.py
from typing import Any, Dict, Tuple
from pathlib import Path
from loguru import logger

from src.core.exceptions import PipelineError
from src.core.validator import DataValidator
from src.infra.data_repository import DataRepository
from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.splitter_service import TimeSeriesSplitter
from src.infra.artifact_manager import ArtifactManager
from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator
from src.core.model_registry import ModelRegistry

from src.core.run_metadata import PipelineRunMetadata, PhaseMetadata
from datetime import datetime, UTC
import json
from dataclasses import asdict
from pathlib import Path
import yaml

class TrainingOrchestrator:
    """
    Production-grade training orchestrator.

    Responsibilities:
    - Manage lifecycle phases (ingestion → preprocessing → features → split → model → eval → registry).
    - Enforce validation gates (raw, preprocessed, engineered).
    - Track state (artifacts) across the pipeline.
    - Coordinate services via dependency injection.
    - Emit artifacts (processed data, model, metrics).
    - Provide structured logging and fail-fast behavior.
    """

    def __init__(
        self,
        config,
        repo: DataRepository,
        ingestion: IngestionService,
        preprocessor: DataPreprocessor,
        engineer: FeatureEngineer,
        splitter: TimeSeriesSplitter,
        artifact_manager: ArtifactManager,
        model_factory: ModelWorkerFactory,
        evaluator: ModelEvaluator,
        registry: ModelRegistry,
        run_id: str,
    ):

        # Core config slices
        self.config = config
        self.data_cfg = config.data
        self.artifacts_cfg = config.artifacts
        self.model_cfg = config.model

        # Services (DI)
        self.repo = repo
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.engineer = engineer
        self.splitter = splitter
        self.artifact_manager = artifact_manager
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.registry = registry

        # Context
        self.run_id = run_id
        self.project_root = Path(config.project_root)

        # State registry
        self.artifacts: Dict[str, Any] = {}

    def _log(self):
        return logger.bind(run_id=self.run_id)

    # -------------------------
    # Phase 1: Ingestion
    # -------------------------
    def _phase_ingestion(self) -> Any:
        log = self._log()
        log.info("Loading raw data")

        raw_file = Path(self.config.paths.raw_data) / self.artifacts_cfg.input_file
        log.info(f"Resolved raw data file: {raw_file}")

        # 1. Load raw CSV directly from the repository
        raw_df = self.repo.read_csv(str(raw_file), sep=self.data_cfg.csv_separator)

        # 2. Validate raw schema BEFORE ingestion renaming
        log.info("Validating Raw Data Contract...")
        DataValidator(self.data_cfg.raw_schema).validate(raw_df)

        # 3. Now run ingestion (renaming + timestamp parsing)
        raw_data = self.ingestion.load_raw_energy_data(str(raw_file))
        self.artifacts["raw"] = raw_data

        return raw_data
    
    # -------------------------
    # Phase 2: Preprocessing
    # -------------------------
    def _phase_preprocessing(self, raw_data: Any) -> Any:
        log = self._log()
        log.info("Preprocessing started...")

        processed_path = self.project_root / self.config.paths.processed_data

        processed_data = self.preprocessor.clean_data(raw_data)

        log.info("Validating Preprocessed Data Contract...")
        DataValidator(self.data_cfg.preprocessed_schema).validate(processed_data)

        self.artifacts["sanitized"] = processed_data
        self.artifacts["processed_path"] = processed_path

        return processed_data

    # -------------------------
    # Phase 3: Feature Engineering
    # -------------------------
    def _phase_feature_engineering(self, processed_data: Any) -> Any:
        log = self._log()
        log.info("Feature Engineering started...")

        engineered_data = self.engineer.transform(processed_data)

        log.info("Validating Engineered Data Contract...")
        DataValidator(self.data_cfg.engineered_schema).validate(engineered_data)

        log.info("Saving Processed data...")
        self.preprocessor.save_processed_data(
            engineered_data,
            filename=self.artifacts_cfg.output_file,
        )

        self.artifacts["engineered"] = engineered_data

        return engineered_data

    # -------------------------
    # Phase 4: Splitting
    # -------------------------
    def _phase_splitting(self, engineered_data: Any) -> Tuple[Any, Any, Any, Any, Any]:
        log = self._log()
        log.info("Data Splitter - Time Series - started...")

        X_train, y_train, X_test, y_test, meta = self.splitter.split(engineered_data)

        self.artifacts["split"] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "meta": meta,
        }

        return X_train, y_train, X_test, y_test, meta

    # -------------------------
    # Phase 5: Modeling
    # -------------------------
    def _phase_modeling(self, X_train: Any, y_train: Any) -> Any:
        log = self._log()
        log.info("--- Starting Model Engineering Phase ---")

        worker = self.model_factory.get_worker(self.model_cfg, run_id=self.run_id)
        trained_model = worker.train(X_train, y_train)

        log.info("Saving Model")
        model_path = self.artifact_manager.save_model(trained_model, self.model_cfg.name)

        self.artifacts["model"] = trained_model
        self.artifacts["model_path"] = model_path

        return trained_model, model_path

    # -------------------------
    # Phase 6: Evaluation & Metrics
    # -------------------------
    def _phase_evaluation(self, trained_model: Any, X_test: Any, y_test: Any, model_path: Path) -> Any:
        log = self._log()
        log.info("--- Starting Evaluation Phase ---")

        metrics = self.evaluator.evaluate(trained_model, X_test, y_test, run_id=self.run_id)

        log.info("Saving Model Metrics")
        self.artifact_manager.save_metrics(
            metrics=metrics,
            model_name=self.model_cfg.name,
            model_uri=str(model_path),
            hyperparameters=self.model_cfg.params,
            training_params=self.config.training,
        )

        self.artifacts["metrics"] = metrics

        return metrics

    # -------------------------
    # Phase 7: Model Registry
    # -------------------------
    def _phase_registry(self, model_path, metrics):
        log = self._log()
        log.info("--- Registering Model ---")

        record = self.registry.register(
            model_name=self.model_cfg.name,
            model_path=str(model_path),
            metrics=metrics
        )

        self.artifacts["registry_record"] = record
        return record

    # -------------------------
    # Orchestrator Entry Point
    # -------------------------
    def run(self) -> None:
        log = self._log()
        log.info("Configuration loaded successfully.")
        log.info("Pipeline started")

        metadata = PipelineRunMetadata(
            run_id=self.run_id,
            pipeline_name="train_pipeline",
            status="RUNNING",
            started_at=datetime.now(UTC),
            tags={"env": "dev"}  # optional
        )

        try:
            # Phase 1: Ingestion
            raw_data = self._run_phase("ingestion", metadata, self._phase_ingestion)

            # Phase 2: Preprocessing
            processed_data = self._run_phase("preprocessing", metadata, self._phase_preprocessing, raw_data)

            # Phase 3: Feature Engineering
            engineered_data = self._run_phase("feature_engineering", metadata, self._phase_feature_engineering, processed_data)

            # Phase 4: Splitting
            X_train, y_train, X_test, y_test, _ = self._run_phase("splitting", metadata, self._phase_splitting, engineered_data)

            # Phase 5: Modeling
            trained_model, model_path = self._run_phase("modeling", metadata, self._phase_modeling, X_train, y_train)

            # Phase 6: Evaluation
            metrics = self._run_phase("evaluation", metadata, self._phase_evaluation, trained_model, X_test, y_test, model_path)

            # Phase 7: Model Registry
            record = self._run_phase("registry", metadata, self._phase_registry, model_path, metrics)
            metadata.registry_version = record["version"]

            metadata.status = "SUCCESS"
            log.info("Pipeline executed successfully.")


        except PipelineError:
            # Already wrapped, let caller handle
            metadata.status = "FAILED"
            raise
        except Exception as e:
            metadata.status = "FAILED"
            log.exception("Pipeline failure with unexpected error")
            raise PipelineError(f"Orchestration failed: {e}") from e

        finally:
            self._snapshot_config(metadata)

            metadata.ended_at = datetime.now(UTC)
            metadata.duration_ms = int(
                (metadata.ended_at - metadata.started_at).total_seconds() * 1000
            )
            self._persist_metadata(metadata)

    def _run_phase(self, name, metadata, fn, *args):
        phase_meta = PhaseMetadata(name=name, status="RUNNING", started_at=datetime.now(UTC))
        metadata.phases[name] = phase_meta

        try:
            result = fn(*args)
            phase_meta.status = "SUCCESS"

            # If the phase returns artifacts, attach them
            if isinstance(result, dict):
                phase_meta.artifact_paths.update(result)

            return result

        except Exception as exc:
            phase_meta.status = "FAILED"
            phase_meta.error_message = str(exc)
            raise

        finally:
            phase_meta.ended_at = datetime.now(UTC)
            phase_meta.duration_ms = int(
                (phase_meta.ended_at - phase_meta.started_at).total_seconds() * 1000
            )

    def _persist_metadata(self, metadata):
        metadata_dir = self.project_root / self.artifacts_cfg.metadata_path
        metadata_dir.mkdir(parents=True, exist_ok=True)

        path = metadata_dir / f"run_{metadata.run_id}.json"

        def _serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj.__dict__

        with path.open("w") as f:
            json.dump(metadata, f, default=_serialize, indent=2)

    def _snapshot_config(self, metadata: PipelineRunMetadata) -> None:
        """
        Persist a frozen copy of the merged configuration used for this run.
        """
        snapshot_dir = self.project_root / self.artifacts_cfg.metadata_path
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshot_dir / f"config_{metadata.run_id}.yaml"

        # Convert dataclass config → dict
        config_dict = asdict(self.config)

        # Recursively convert Path objects → strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            else:
                return obj

        config_dict = convert_paths(config_dict)

        with snapshot_path.open("w") as f:
            yaml.safe_dump(config_dict, f)

        metadata.config_snapshot_path = str(snapshot_path)