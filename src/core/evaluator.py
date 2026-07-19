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
    @staticmethod
    def evaluate(
        model: Any, 
        X_test, 
        y_test, 
        config: EvaluationConfig, 
        run_id: str
    ) -> Dict[str, float]:
        logger.bind(
            module="ModelEvaluator",
            run_id=run_id,
            X_shape=X_test.shape,
            y_shape=y_test.shape
        ).info("Starting model evaluation")
        
        try:
            predictions = model.predict(X_test)
            
            # Use metrics defined in the injected config
            metrics = {}
            if "rmse" in config.metrics:
                metrics["rmse"] = root_mean_squared_error(y_test, predictions)
            if "mse" in config.metrics:
                metrics["mse"] = mean_squared_error(y_test, predictions)

            # Check thresholds from injected config
            for metric_name, threshold in config.thresholds.items():
                if metric_name in metrics and metrics[metric_name] > threshold:
                    logger.warning(
                        f"Metric {metric_name} ({metrics[metric_name]}) exceeded threshold ({threshold})"
                    )

            logger.bind(
                module="ModelEvaluator",
                run_id=run_id,
                metrics=metrics
            ).info(f"Evaluation complete. Format: {config.report_format}")
            
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
