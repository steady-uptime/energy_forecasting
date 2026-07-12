# pipelines/batch_inference.py
"""
Batch Inference Pipeline
------------------------
Processes any .txt or .csv file in data/interim/,
runs preprocessing + feature engineering,
loads the champion model, and writes predictions.
"""

import pandas as pd
from pathlib import Path
from loguru import logger

from src.core.config_loader import config
from src.core.model_registry import ModelRegistry
from src.infra.artifact_manager import ArtifactManager
from src.core.preprocessing_service import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.api.inference_service import InferenceService


def load_interim_files(interim_dir: Path):
    files = list(interim_dir.glob("*.txt")) + list(interim_dir.glob("*.csv"))
    if not files:
        logger.warning(f"No interim files found in {interim_dir}")
        return []
    logger.info(f"Found {len(files)} interim file(s): {[f.name for f in files]}")
    return files


def normalize_header(df: pd.DataFrame):
    cols = [c.strip('"') for c in df.columns]
    if cols[0] == "" or cols[0].lower().startswith("unnamed"):
        cols[0] = "timestamp"
    df.columns = cols
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def run_batch_inference():
    project_root = Path(config.project_root)

    interim_dir = project_root / config.paths.interim_data
    predictions_dir = project_root / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    run_id = "batch-inference"
    logger.bind(run_id=run_id).info("Starting batch inference pipeline")

    files = load_interim_files(interim_dir)
    if not files:
        logger.info("No files to process. Exiting.")
        return

    # DI (same pattern as train_pipeline)
    registry = ModelRegistry(Path(config.artifacts.metadata_path) / "registry")
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

    for file in files:
        logger.bind(run_id=run_id).info(f"Processing file: {file.name}")

        df = pd.read_csv(file, sep=";", decimal=",")
        df = normalize_header(df)

        preds = service.predict_batch(df)

        output_path = predictions_dir / f"{file.stem}_predictions.csv"
        preds.to_csv(output_path, index=True)

        logger.bind(run_id=run_id).info(f"Predictions written to: {output_path}")

    logger.bind(run_id=run_id).info("Batch inference pipeline completed successfully")


if __name__ == "__main__":
    run_batch_inference()
