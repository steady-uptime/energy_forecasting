# src/core/train_orchestrator.py
from typing import Any, Dict, Tuple, Callable
from pathlib import Path
from loguru import logger
from datetime import datetime, UTC
import json
from dataclasses import asdict

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
from src.core.config.schemas import AppConfig
from src.core.hpo_orchestrator import HPOOrchestrator


class TrainingOrchestrator:
    """
    Production-grade training orchestrator.

    Phases:
    - Data Orchestration (DataOrchestrator)
    - Splitting (TimeSeriesSplitter)
    - HPO / Modeling (HPOOrchestrator + ModelWorkerFactory)
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
        evaluator: ModelEvaluator,
        registry: ModelRegistry,
        run_id: str,
    ):
        self.config = config
        self.data_cfg = config.data
        self.artifacts_cfg = config.artifacts
        self.model_cfg = config.model
        self.hpo_cfg = config.model.hpo
        self.eval_cfg = config.evaluation

        self.repo = repo
        self.data_orchestrator = data_orchestrator
        self.splitter = splitter
        self.artifact_manager = artifact_manager
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.registry = registry

        self.run_id = run_id
        self.project_root = Path(config.project_root)

        self.artifacts: Dict[str, Any] = {}

    def _log(self):
        return logger.bind(run_id=self.run_id)

    # -------------------------
    # Phase-Specific Logic
    # -------------------------
    def _phase_modeling_hpo(self, engineered_data: Any) -> Tuple[Any, Path, Dict[str, Any]]:
        """
        Run HPO over the engineered data, return best model, path, and metrics.
        """
        log = self._log()
        log.info("--- Starting HPO / Model Engineering Phase ---")

        hpo = HPOOrchestrator(
            model_factory=self.model_factory,
            evaluator=self.evaluator,
            splitter=self.splitter,
            model_cfg=self.model_cfg,
            hpo_cfg=self.hpo_cfg,
            eval_cfg=self.eval_cfg,
            run_id=self.run_id,
        )

        hpo_result = hpo.run(engineered_data)
        best_trial = hpo_result.best_trial(metric_name=self.eval_cfg.primary_metric)

        # Persist best model metrics via artifact manager
        self.artifact_manager.save_metrics(
            metrics=best_trial.metrics,
            model_name=self.model_cfg.name,
            model_uri=str(best_trial.model_path),
            hyperparameters=best_trial.params,
            training_params=self.config.training,
        )

        self.artifacts["model_path"] = best_trial.model_path
        self.artifacts["metrics"] = best_trial.metrics
        self.artifacts["best_trial_id"] = best_trial.trial_id

        log.info(f"Best trial: {best_trial.trial_id} | Metrics={best_trial.metrics}")
        return best_trial, best_trial.model_path, best_trial.metrics

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

            # Phase 2: HPO / Modeling (includes splitting internally)
            best_trial, model_path, metrics = self._run_phase(
                "modeling_hpo",
                metadata,
                self._phase_modeling_hpo,
                engineered_data,
            )

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

        def convert_paths(obj: Any) -> Any:
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
