# src/core/evaluator.py
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from typing import Any
from loguru import logger
from src.core.exceptions import ModelTrainingError

class ModelEvaluator:
    @staticmethod
    def evaluate(model: Any, X_test, y_test, run_id: str):
        logger.bind(
            module="ModelEvaluator",
            run_id=run_id,
            X_shape=X_test.shape,
            y_shape=y_test.shape
        ).info("Starting model evaluation")

        try:
            predictions = model.predict(X_test)

            metrics = {
                "rmse": root_mean_squared_error(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions)
            }

            logger.bind(
                module="ModelEvaluator",
                run_id=run_id,
                metrics=metrics
            ).info("Evaluation complete")

            return metrics

        except Exception as e:
            logger.bind(
                module="ModelEvaluator",
                run_id=run_id,
                error=str(e)
            ).error("Model evaluation failure")

            raise ModelTrainingError(
                "Model evaluation failed",
                context={
                    "X_shape": X_test.shape,
                    "y_shape": y_test.shape,
                    "model_type": type(model).__name__
                }
            ) from e
