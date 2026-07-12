# pipelines/comparison_pipeline.py
import sys
from pathlib import Path
from src.core.config_loader import ConfigLoader
from src.core.validator import DataValidator
from src.core.model_validator import ModelMetricValidator
from src.core.evaluator import ModelEvaluator
from src.core.modeling import ModelWorkerFactory
from src.core.model_registry import ModelRegistry
from src.core.comparison_orchestrator import ComparisonOrchestrator
from src.core.comparison_service import ModelComparisonService # Assuming location based on architecture
from src.infra.logger import logger
from src.core.exceptions import PipelineError

def main():
    try:
        logger.info("Initializing Comparison Pipeline...")
        
        # 1. Load Configuration (Singleton)
        # The ConfigLoader handles path resolution via the project root
        config = ConfigLoader.get_config()
        comp_config = config.get("comparison", {})
        data_config = config.get("data", {})

        # 2. Instantiate Dependencies (Manual Dependency Injection)
        # We instantiate the "leaf" services first
        data_validator = DataValidator()
        metric_validator = ModelMetricValidator()
        evaluator = ModelEvaluator()
        registry = ModelRegistry()
        worker_factory = ModelWorkerFactory()
        
        # The ComparisonService handles the logic of picking a winner
        comparison_service = ModelComparisonService()

        # 3. Wire Dependencies into the Orchestrator
        # The Orchestrator receives its dependencies via constructor injection
        orchestrator = ComparisonOrchestrator(
            data_validator=data_validator,
            metric_validator=metric_validator,
            evaluator=evaluator,
            worker_factory=worker_factory,
            registry=registry,
            comparison_service=comparison_service,
            config=comp_config,
            data_config=data_config
        )

        # 4. Execute the workflow
        # The bootstrap loader calls the orchestrator's execution method
        orchestrator.run()
        
        logger.info("Comparison pipeline completed successfully.")

    except PipelineError as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected system error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
