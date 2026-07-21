# src/core/train_orchestrator.py
from typing import Any, Dict, Tuple, Callable
from pathlib import Path
from loguru import logger
from datetime import datetime, UTC
from dataclasses import asdict
import json
import yaml

from src.core.exceptions import PipelineError
from src.infra.data_repository import DataRepository
from src.infra.artifact_manager import ArtifactManager
from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator
from src.core.model_registry import ModelRegistry
from src.core.data_orchestrator import DataOrchestrator
from src.core.splitter_service import TimeSeriesSplitter
from src.core.run_metadata import PipelineRunMetadata, PhaseMetadata
from src.core.config.schemas import AppConfig, SearchResults
from src.core.model_search_engine import ModelSearchEngine

class TrainingOrchestrator:
    """
    Production-grade training orchestrator.

    Phases:
    - Data Orchestration (DataOrchestrator)
    - Splitting (TimeSeriesSplitter)
    - Model Search / Modeling (ModelSearchEngine + ModelWorkerFactory)
    - Evaluation (ModelEvaluator)
    - Registry (ModelRegistry)
    """

    def __init__(
        self,
        config: AppConfig,
        repo: DataRepository,
        data_orchestrator: DataOrchestrator,
        splitter: TimeSeriesSplitter,
        artifact_manager: ArtifactManager,
        model_factory: ModelWorkerFactory,
        model_search: ModelSearchEngine,
        evaluator: ModelEvaluator,
        registry: ModelRegistry,
        run_id: str,
    ):
        self.config = config
        self.data_cfg = config.data
        self.artifacts_cfg = config.artifacts
        self.model_cfg = config.model
        self.eval_cfg = config.evaluation

        self.repo = repo
        self.data_orchestrator = data_orchestrator
        self.splitter = splitter
        self.artifact_manager = artifact_manager
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.registry = registry
        self.model_search = model_search

        self.run_id = run_id
        self.project_root = Path(config.project_root)

        self.artifacts: Dict[str, Any] = {}

    def _log(self):
        return logger.bind(run_id=self.run_id)

    # -------------------------
    # Phase-Specific Logic
    # -------------------------
    def _phase_model_search(self, engineered_data):
        logger.info("Starting model search")

        # 1. Split data
        X_train, y_train, X_val, y_val = self.splitter.split(engineered_data)

        # 2. Run search engine
        search_results = self.model_search.run(
            X_train, y_train, X_val, y_val, self.run_id
        )

        # 3. Extract champion
        champion = search_results.champion

        # 4. Persist metrics
        self.artifact_manager.save_metrics(
            metrics=champion.metrics,
            model_name=champion.definition.name,
            model_uri=str(champion.artifact_path),
            training_params=self.config.training,
            hyperparameters=champion.definition.params,
        )

        # 5. Return search results
        return search_results

    def _phase_registry(self, model_path: Path, metrics: Any) -> Dict[str, Any]:
        log = self._log()
        log.info("--- Registering Model ---")

        record = self.registry.register(
            model_name=self.model_cfg.name,
            model_path=str(model_path),
            metrics=metrics,
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
            tags={"env": "dev"},
        )

        try:
            # Phase 1: Data Orchestration
            raw_file = Path(self.config.paths.raw_data) / self.artifacts_cfg.input_file

            log.info(f"Initiating Data Orchestration for {raw_file}")
            engineered_data = self._run_phase(
                "data_orchestration",
                metadata,
                self.data_orchestrator.run,
                raw_file,
            )

            # Phase 2: Model Search
            search_results = self._run_phase(
                "model_search",
                metadata,
                self._phase_model_search,
                engineered_data,
            )

            champion = search_results.champion
            model_path = champion.artifact_path
            metrics = champion.metrics

            # Phase 3: Model Registry
            record = self._run_phase(
                "registry",
                metadata,
                self._phase_registry,
                model_path,
                metrics,
            )

            metadata.registry_version = record.get("version")
            metadata.status = "SUCCESS"
            log.info("Pipeline executed successfully.")

        except PipelineError:
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

    # -------------------------
    # Utility & Observability Methods
    # -------------------------
    def _run_phase(
        self,
        name: str,
        metadata: PipelineRunMetadata,
        fn: Callable[..., Any],
        *args: Any,
    ) -> Any:
        phase_meta = PhaseMetadata(
            name=name,
            status="RUNNING",
            started_at=datetime.now(UTC),
        )
        metadata.phases[name] = phase_meta

        try:
            result = fn(*args)
            phase_meta.status = "SUCCESS"
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

    def _persist_metadata(self, metadata: PipelineRunMetadata) -> None:
        metadata_dir = self.project_root / self.artifacts_cfg.metadata_path
        metadata_dir.mkdir(parents=True, exist_ok=True)

        path = metadata_dir / f"run_{metadata.run_id}.json"

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return obj

        with path.open("w") as f:
            json.dump(metadata, f, default=_serialize, indent=2)

    def _snapshot_config(self, metadata: PipelineRunMetadata) -> None:
        snapshot_dir = self.project_root / self.artifacts_cfg.metadata_path
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshot_dir / f"config_{metadata.run_id}.yaml"

        config_dict = asdict(self.config)

        def convert_values(obj: Any) -> Any:
            from uuid import UUID
            from pathlib import Path
            from datetime import datetime
        
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, UUID):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: convert_values(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_values(v) for v in obj]
            return obj

        config_dict = convert_values(asdict(self.config))

        with snapshot_path.open("w") as f:
            yaml.safe_dump(config_dict, f)

        metadata.config_snapshot_path = str(snapshot_path)
