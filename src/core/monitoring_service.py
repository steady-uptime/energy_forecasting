# src/core/monitoring_service.py
# Compared to TrainingOrchestrator.run():
# No modeling phase (no training)
# No registry phase (no new champion registration)
# No model saving
# No metrics saving for training
# No train/test split for training (only live slice)
# No hyperparameters / training config handling
# Monitoring is:
# same data path (ingestion → preprocessing → features → split)
# different end (evaluate → drift → report)

from pathlib import Path
from datetime import datetime, UTC
from typing import Any, Dict

from loguru import logger

from src.core.exceptions import MonitoringError
from src.core.validator import DataValidator
from src.infra.data_repository import DataRepository
from src.core.ingestion_service import IngestionService
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.splitter_service import TimeSeriesSplitter
from src.infra.artifact_manager import ArtifactManager
from src.core.evaluator import ModelEvaluator
from src.core.model_registry import ModelRegistry


class MonitoringService:
    """
    Production-grade monitoring orchestrator.

    Responsibilities:
    - Load champion model and baseline metrics from registry.
    - Ingest, preprocess, and engineer features for live data using the same
      services and contracts as the training pipeline.
    - Apply time-series split to obtain the live evaluation slice.
    - Evaluate champion model on live data.
    - Detect drift by comparing live metrics to baseline metrics.
    - Emit monitoring reports and structured logs.
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
        evaluator: ModelEvaluator,
        registry: ModelRegistry,
        monitoring_cfg,
        run_id: str,
    ):
        # Core config slices
        self.config = config
        self.data_cfg = config.data
        self.monitoring_cfg = monitoring_cfg

        # Services (DI)
        self.repo = repo
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.engineer = engineer
        self.splitter = splitter
        self.artifact_manager = artifact_manager
        self.evaluator = evaluator
        self.registry = registry

        # Context
        self.run_id = run_id
        self.project_root = Path(config.project_root)

        # State (optional, for debugging/inspection)
        self.artifacts: Dict[str, Any] = {}

    def _log(self):
        return logger.bind(run_id=self.run_id, module="MonitoringService")

    # -------------------------
    # Phase 0: Champion Load
    # -------------------------
    def _load_champion(self):
        log = self._log()
        log.info("Loading champion model from registry")

        try:
            record = self.registry.load_champion()
            model_path = record["model_path"]
            baseline_metrics = record["metrics"]
            version = record.get("version")

            model = self.artifact_manager.load_model(model_path)

            log.info("Model artifact loaded")
            self.artifacts["champion_record"] = record
            self.artifacts["champion_model_path"] = model_path

            return model, baseline_metrics, version

        except Exception as e:
            log.error(f"Failed to load champion model: {e}")
            raise MonitoringError("Champion load failure") from e

    # -------------------------
    # Phase 1: Live Data Ingestion + Preprocessing
    # -------------------------
    def _load_live_data(self):
        log = self._log()
        log.info("Loading live data for monitoring")

        try:
            live_file = Path(self.monitoring_cfg.live_data_path)
            log.info(f"Resolved live data file: {live_file}")

            # Ingestion (same as training orchestrator)
            raw_live = self.ingestion.load_raw_energy_data(str(live_file))
            self.artifacts["raw_live"] = raw_live

            log.info("Validating Raw Live Data Contract...")
            DataValidator(self.data_cfg.raw_schema).validate(raw_live)

            # Preprocessing (same as training orchestrator)
            preprocessed_live = self.preprocessor.clean_data(raw_live)
            self.artifacts["preprocessed_live"] = preprocessed_live

            log.info("Validating Preprocessed Live Data Contract...")
            DataValidator(self.data_cfg.preprocessed_schema).validate(preprocessed_live)

            return preprocessed_live

        except MonitoringError:
            raise
        except Exception as e:
            log.error(f"Failed to load live data: {e}")
            raise MonitoringError("Live data load failure") from e

    # -------------------------
    # Phase 2: Feature Engineering
    # -------------------------
    def _apply_feature_engineering(self, preprocessed_live):
        log = self._log()
        log.info("Applying feature engineering to live data")

        try:
            engineered_live = self.engineer.transform(preprocessed_live)
            self.artifacts["engineered_live"] = engineered_live

            log.info("Validating Engineered Live Data Contract...")
            DataValidator(self.data_cfg.engineered_schema).validate(engineered_live)

            return engineered_live

        except Exception as e:
            log.error(f"Feature engineering failure: {e}")
            raise MonitoringError("Feature engineering failure") from e

    # -------------------------
    # Phase 3: Time-Series Split (Live Slice)
    # -------------------------
    def _apply_split(self, engineered_live):
        log = self._log()
        log.info("Applying time-series split for monitoring")

        try:
            # Reuse the same splitter as training.
            # Convention: splitter returns (X_train, y_train, X_test, y_test, meta)
            # For monitoring, we treat X_test/y_test as the "live" slice.
            X_train, y_train, X_test, y_test, meta = self.splitter.split(engineered_live)

            self.artifacts["split_live"] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_live": X_test,
                "y_live": y_test,
                "meta": meta,
            }

            log.info("Monitoring split complete")
            return X_test, y_test, meta

        except Exception as e:
            log.error(f"Split failure: {e}")
            raise MonitoringError("Split failure") from e

    # -------------------------
    # Phase 4: Evaluation
    # -------------------------
    def _evaluate_live(self, model, X_live, y_live):
        log = self._log()
        log.info("Starting live data evaluation")

        try:
            metrics = self.evaluator.evaluate(model, X_live, y_live, run_id=self.run_id)
            self.artifacts["live_metrics"] = metrics

            log.info("Evaluation complete")
            return metrics

        except Exception as e:
            log.error(f"Live evaluation failure: {e}")
            raise MonitoringError("Live evaluation failure") from e

    # -------------------------
    # Phase 5: Drift Detection
    # -------------------------
    def _detect_drift(self, baseline_metrics: Dict[str, float], live_metrics: Dict[str, float]) -> bool:
        log = self._log()
        log.info("Starting drift detection")

        try:
            # Simple example: compare RMSE or MAE thresholds.
            # You can replace this with a more sophisticated drift logic.
            metric_name = self.monitoring_cfg.primary_metric  # e.g., "rmse"
            threshold = self.monitoring_cfg.drift_threshold   # e.g., 0.10 (10%)

            baseline = baseline_metrics[metric_name]
            current = live_metrics[metric_name]

            if baseline == 0:
                log.warning("Baseline metric is zero; drift detection may be unreliable.")
                drift = False
            else:
                relative_change = (current - baseline) / baseline
                drift = relative_change > threshold

            log.info(
                f"Drift detection result: metric={metric_name}, "
                f"baseline={baseline}, current={current}, "
                f"threshold={threshold}, drift={drift}"
            )

            self.artifacts["drift_detected"] = drift
            return drift

        except Exception as e:
            log.error(f"Drift detection failure: {e}")
            raise MonitoringError("Drift detection failure") from e

    # -------------------------
    # Phase 6: Reporting
    # -------------------------
    def _emit_report(self, drift_detected: bool, baseline_metrics, live_metrics, model_version: str | None):
        log = self._log()
        log.info("Emitting monitoring report")

        try:
            report_dir = self.project_root / self.monitoring_cfg.report_path
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(UTC).isoformat()
            report_path = report_dir / f"monitoring_{self.run_id}.json"

            report = {
                "run_id": self.run_id,
                "timestamp": timestamp,
                "model_version": model_version,
                "drift_detected": drift_detected,
                "baseline_metrics": baseline_metrics,
                "live_metrics": live_metrics,
            }

            import json
            with report_path.open("w") as f:
                json.dump(report, f, indent=2)

            log.info(f"Monitoring report saved at {report_path}")
            self.artifacts["report_path"] = str(report_path)

        except Exception as e:
            log.error(f"Failed to emit monitoring report: {e}")
            raise MonitoringError("Report emission failure") from e

    # -------------------------
    # Orchestrator Entry Point
    # -------------------------
    def run(self) -> bool:
        """
        Execute the monitoring pipeline:

        - Load champion model and baseline metrics.
        - Ingest, preprocess, and engineer live data.
        - Apply time-series split to obtain live evaluation slice.
        - Evaluate champion on live data.
        - Detect drift.
        - Emit monitoring report.

        Returns:
            bool: True if drift detected, False otherwise.
        """
        log = self._log()
        log.info("Monitoring pipeline started")

        try:
            # Phase 0: Champion
            model, baseline_metrics, model_version = self._load_champion()

            # Phase 1: Live Data (ingestion + preprocessing)
            preprocessed_live = self._load_live_data()

            # Phase 2: Feature Engineering
            engineered_live = self._apply_feature_engineering(preprocessed_live)

            # Phase 3: Split (live slice)
            X_live, y_live, meta = self._apply_split(engineered_live)
            self.artifacts["split_meta"] = meta

            # Phase 4: Evaluation
            live_metrics = self._evaluate_live(model, X_live, y_live)

            # Phase 5: Drift Detection
            drift_detected = self._detect_drift(baseline_metrics, live_metrics)

            # Phase 6: Reporting
            self._emit_report(drift_detected, baseline_metrics, live_metrics, model_version)

            log.info(f"Monitoring complete. Drift detected: {drift_detected}")
            return drift_detected

        except MonitoringError:
            # Already wrapped with context
            log.error("Monitoring pipeline failed with a monitored error")
            raise
        except Exception as e:
            log.exception("Monitoring pipeline failure with unexpected error")
            raise MonitoringError(f"Monitoring pipeline failure: {e}") from e
