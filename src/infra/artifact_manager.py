# src/infra/artifact_manager.py
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import is_dataclass, asdict
from datetime import datetime
from loguru import logger
from src.core.config.schemas import ArtifactsConfig
from src.core.exceptions import ArtifactError

class ArtifactManager:
    """
    Infrastructure Layer: Handles the 'How' and 'Where' of storage.
    Decoupled from the ML Model logic and the Registry.
    """
    def __init__(self, artifact_cfg: ArtifactsConfig, project_root: Path, run_id: str):
        self.artifact_cfg = artifact_cfg
        self.project_root = project_root
        self.run_id = run_id

        # Resolve directories
        self.models_dir = Path(artifact_cfg.models_path)
        self.metadata_dir = Path(artifact_cfg.metadata_path)
        self.reports_dir = Path(artifact_cfg.reports_path)

        # Ensure directory existence
        for path in [self.models_dir, self.metadata_dir, self.reports_dir]:
            path.mkdir(parents=True, exist_ok=True)

        logger.bind(
            module="ArtifactManager",
            run_id=self.run_id,
            project_root=str(self.project_root)
        ).info("ArtifactManager initialized")

    def save_drift_report(
        self,
        drift_detected: bool,
        baseline: dict,
        live: dict,
        model_version: str,
        run_id: str
    ):
        """
        Persist a drift report for monitoring.
        """
        log = logger.bind(
            module="ArtifactManager",
            run_id=run_id,
            model_version=model_version
        )

        try:
            # -----------------------------------------------------
            # Resolve Path
            # -----------------------------------------------------
            drift_dir = Path(self.artifact_cfg.reports_path) / "drift"
            drift_dir.mkdir(parents=True, exist_ok=True)

            report_path = drift_dir / f"drift_report_{run_id}.json"

            # -----------------------------------------------------
            # Build Report Structure
            # -----------------------------------------------------
            report = {
                "run_id": run_id,
                "model_version": model_version,
                "drift_detected": drift_detected,
                "baseline_metrics": baseline,
                "live_metrics": live,
            }

            # -----------------------------------------------------
            # Write Report
            # -----------------------------------------------------
            with report_path.open("w") as f:
                json.dump(report, f, indent=2)

            log.info(f"Drift report saved: {report_path}")

            return report_path

        except Exception as e:
            log.error(f"Failed to save drift report: {e}")
            raise ArtifactError(
                "Failed to save drift report",
                context={"run_id": run_id, "model_version": model_version}
            ) from e

    def load_model(self, model_path: str):
        """
        Load a serialized model artifact from disk.
        """
        try:
            path = Path(model_path)

            if not path.exists():
                raise ArtifactError(
                    "Model file not found",
                    context={"model_path": model_path}
                )

            model = joblib.load(path)

            logger.bind(
                module="ArtifactManager",
                run_id=self.run_id,
                model_path=model_path
            ).info("Model artifact loaded")

            return model

        except Exception as e:
            logger.bind(
                module="ArtifactManager",
                run_id=self.run_id,
                error=str(e),
                model_path=model_path
            ).error("Model artifact load failure")

            raise ArtifactError(
                "Failed to load model artifact",
                context={"model_path": model_path}
            ) from e

    def save_model(self, model: Any, model_name: str) -> Path:
        """Serializes the model weights to the filesystem."""
        save_path = self.models_dir / f"{model_name}.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(model, save_path)

            logger.bind(
                module="ArtifactManager",
                run_id=self.run_id,
                model_name=model_name,
                save_path=str(save_path)
            ).info("Model artifact saved")

            return save_path

        except Exception as e:
            logger.bind(
                module="ArtifactManager",
                run_id=self.run_id,
                error=str(e)
            ).error("Model artifact save failure")

            raise ArtifactError(
                "Failed to save model artifact",
                context={"model_name": model_name, "save_path": str(save_path)}
            ) from e

    def save_metrics(
        self, 
        metrics: Dict[str, float],
        model_name: str,
        model_uri: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_params: Any = None
    ) -> Path:
        """Saves evaluation metrics and metadata as a JSON artifact."""
        save_path = self.metadata_dir / f"{model_name}_metrics.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_name": model_name,
            "metrics": metrics,
            "model_uri": model_uri,
            "hyperparameters": hyperparameters or {},
            "training_params": training_params or {},
            "timestamp": datetime.utcnow().isoformat(),
            "project_root": str(self.project_root),
            "run_id": self.run_id
        }

        serializable_metadata = self._make_serializable(metadata)

        try:
            with open(save_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=4)

            logger.bind(
                module="ArtifactManager",
                run_id=self.run_id,
                model_name=model_name,
                save_path=str(save_path)
            ).info("Metrics artifact saved")

            return save_path

        except Exception as e:
            logger.bind(
                module="ArtifactManager",
                run_id=self.run_id,
                error=str(e)
            ).error("Metrics artifact save failure")

            raise ArtifactError(
                "Failed to save metrics artifact",
                context={"model_name": model_name, "save_path": str(save_path)}
            ) from e

    def _make_serializable(self, data: Any) -> Any:
        if is_dataclass(data) and not isinstance(data, type):
            return asdict(data)
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        return data
