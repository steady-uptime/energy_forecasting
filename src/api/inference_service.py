import pandas as pd
from loguru import logger

class InferenceService:
    def __init__(self, registry, preprocessor, engineer, artifact_manager, run_id: str):
        self.registry = registry
        self.preprocessor = preprocessor
        self.engineer = engineer
        self.artifact_manager = artifact_manager
        self.run_id = run_id
        self.model = None

    def load_champion(self):
        record = self.registry.get_champion()
        model_path = record["model_path"]

        logger.bind(run_id=self.run_id).info(f"Loading champion model from: {model_path}")
        self.model = self.artifact_manager.load_model(model_path)

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        logger.bind(
            module="InferenceService",
            run_id=self.run_id,
            rows=len(df),
            cols=len(df.columns)
        ).info("Starting batch inference")

        # Preprocessing
        df_clean = self.preprocessor.clean_data(df)

        # Feature engineering
        engineered = self.engineer.transform(df_clean)

        # Drop target column (must match training)
        target = self.engineer.target_column
        X = engineered.drop(columns=[target], errors="ignore")

        # Drop timestamp (model was not trained with it)
        X = X.drop(columns=["timestamp"], errors="ignore")

        # Predict
        preds = self.model.predict(X)

        logger.bind(
            module="InferenceService",
            run_id=self.run_id,
            rows=len(preds)
        ).info("Batch inference completed")

        return pd.Series(preds, index=df.index, name="prediction")
