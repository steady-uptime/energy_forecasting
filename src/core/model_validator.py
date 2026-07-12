# src/core/model_validator.py
from typing import Dict
from loguru import logger
from src.core.exceptions import ModelTrainingError

class ModelMetricValidator:
    """
    Gatekeeper for model performance metrics. 
    Ensures that the metrics produced by the Evaluator are numerically 
    sound before they reach the Comparison Service or the Registry.
    """
    def __init__(self, required_metrics: list[str]):
        self.required_metrics = required_metrics

    def validate(self, metrics: Dict[str, float], model_id: str) -> bool:
        logger.info(f"Validating performance metrics for model: {model_id}")
        
        # 1. Check for missing required metrics
        for metric in self.required_metrics:
            if metric not in metrics:
                # Swapped ValidationError for ModelTrainingError
                raise ModelTrainingError(f"Model {model_id} is missing required metric: {metric}", context={"model_id": model_id, "missing_metric": metric})

        # 2. Check for NaN or Infinite values
        for metric, value in metrics.items():
            if not (float('-inf') < value < float('inf')):
                # Swapped ValidationError for ModelTrainingError
                raise ModelTrainingError(f"Model {model_id} has invalid value for {metric}: {value}", context={"model_id": model_id, "metric": metric, "value": value})

        # 3. Logic Check (e.g., RMSE cannot be negative)
        if "rmse" in metrics and metrics["rmse"] < 0:
            # Swapped ValidationError for ModelTrainingError
            raise ModelTrainingError(f"Model {model_id} produced a negative RMSE: {metrics['rmse']}", context={"model_id": model_id, "rmse": metrics["rmse"]})

        logger.success(f"Metrics for {model_id} passed quality validation.")
        return True
