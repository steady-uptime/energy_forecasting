# src/core/evaluator.py
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from typing import Any, Dict
from loguru import logger

from src.core.exceptions import ModelTrainingError
from src.core.config.schemas import EvaluationConfig


class ModelEvaluator:
    """
    Handles model performance evaluation using injected EvaluationConfig.
    """

    def __init__(self, eval_cfg: EvaluationConfig):
        self.eval_cfg = eval_cfg
        self.metrics = eval_cfg.metrics
        self.thresholds = eval_cfg.thresholds
        self.report_format = eval_cfg.report_format

    def evaluate(self, model: Any, X_val, y_val) -> Dict[str, float]:
        logger.bind(
            module="ModelEvaluator",
            X_shape=X_val.shape,
            y_shape=y_val.shape
        ).info("Starting model evaluation")

        try:
            predictions = model.predict(X_val)

            results = {}

            # Compute metrics
            if "rmse" in self.metrics:
                results["rmse"] = root_mean_squared_error(y_val, predictions)

            if "mse" in self.metrics:
                results["mse"] = mean_squared_error(y_val, predictions)

            # Threshold checks
            for metric_name, threshold in self.thresholds.items():
                if metric_name in results and results[metric_name] > threshold:
                    logger.warning(
                        f"Metric {metric_name} ({results[metric_name]}) exceeded threshold ({threshold})"
                    )

            logger.bind(
                module="ModelEvaluator",
                metrics=results
            ).info(f"Evaluation complete. Format: {self.report_format}")

            return results

        except Exception as e:
            logger.bind(
                module="ModelEvaluator",
                error=str(e)
            ).error("Model evaluation failure")

            raise ModelTrainingError(
                "Model evaluation failed",
                context={
                    "X_shape": X_val.shape,
                    "y_shape": y_val.shape,
                    "model_type": type(model).__name__,
                },
            ) from e
