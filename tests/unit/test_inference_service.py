# tests/unit/test_inference_service.py

import pandas as pd
from pathlib import Path
from src.api.inference_service import InferenceService
from src.core.model_registry import ModelRegistry
from src.infra.artifact_manager import ArtifactManager
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from core.config import config
from tests.helpers.payloads import sample_payload

def test_inference_service_predict():
    project_root = Path(config.project_root)
    run_id = "test-run"

    registry = ModelRegistry(project_root / config.artifacts.metadata_path / "registry")
    artifact_manager = ArtifactManager(config.artifacts, project_root, run_id)

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

    service = InferenceService(
        registry=registry,
        preprocessor=preprocessor,
        engineer=engineer,
        artifact_manager=artifact_manager,
        run_id=run_id
    )

    service.load_champion()

    result = service.predict(sample_payload())

    assert "prediction" in result
    assert isinstance(result["prediction"], float)
