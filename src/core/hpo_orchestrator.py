# src/core/hpo_orchestrator.py
import itertools
import random
import copy
from typing import List, Dict, Any
from datetime import datetime, UTC
from loguru import logger
from dataclasses import dataclass

from src.core.exceptions import PipelineError
from src.core.train_orchestrator import TrainingOrchestrator
# Import your existing Result schema
from src.core.comparison_service import CandidateResult

@dataclass
class ModelTrialResult:
    """Internal helper to capture the raw output of a TrainingOrchestrator trial."""
    run_id: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    model_path: str
    status: str
    error_message: str = ""

class HyperparameterOrchestrator:
    """
    Orchestrates hyperparameter tuning by executing multiple TrainingOrchestrator trials.
    Produces a list of CandidateResult objects for the ComparisonOrchestrator.
    """

    def __init__(
        self,
        config: Any,
        repo: Any,
        ingestion: Any,
        preprocessor: Any,
        engineer: Any,
        splitter: Any,
        artifact_manager: Any,
        model_factory: Any,
        evaluator: Any,
        registry: Any,
    ):
        self.config = config
        self.repo = repo
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.engineer = engineer
        self.splitter = splitter
        self.artifact_manager = artifact_manager
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.registry = registry

        # HPO specific config extraction
        self.hpo_cfg = config.model.hpo
        self.strategy = self.hpo_cfg.get("strategy", "grid")
        self.search_space = self.hpo_cfg.get("parameters", {})

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generates a list of hyperparameter dictionaries based on strategy."""
        if self.strategy == "grid":
            keys = self.search_space.keys()
            values = self.search_space.values()
            return [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        elif self.strategy == "random":
            # Defaulting to 10 trials for random; could be made config-driven
            return [{k: random.choice(v) for k, v in self.search_space.items()} 
                    for _ in range(10)]
        
        raise ValueError(f"Unknown HPO strategy: {self.strategy}")

    def _prepare_trial_config(self, hpo_params: Dict[str, Any]) -> Any:
        """Deep copies the global config and injects trial-specific params."""
        new_config = copy.deepcopy(self.config)
        
        # Inject into the model.params block
        # This ensures TrainingOrchestrator sees the merged set of static + dynamic params
        if hasattr(new_config, 'model') and hasattr(new_config.model, 'params'):
            current_params = dict(new_config.model.params)
            current_params.update(hpo_params)
            new_config.model.params = current_params
            
        return new_config

    def run(self) -> List[CandidateResult]:
        """
        Executes the HPO loop.
        Returns: List[CandidateResult] -> Directly compatible with ComparisonOrchestrator.
        """
        logger.info(f"Initiating HPO Search | Strategy: {self.strategy}")
        combinations = self._generate_combinations()
        candidate_results: List[CandidateResult] = []

        for i, combo in enumerate(combinations):
            # Unique ID per trial
            trial_id = f"hpo_trial_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{i}"
            logger.info(f"Trial {i+1}/{len(combinations)} | Params: {combo}")

            try:
                # 1. Isolate Configuration
                trial_config = self._prepare_trial_config(combo)

                # 2. Instantiate and Run TrainingOrchestrator (The "Trial" Unit)
                trial_orchestrator = TrainingOrchestrator(
                    config=trial_config,
                    repo=self.repo,
                    ingestion=self.ingestion,
                    preprocessor=self.preprocessor,
                    engineer=self.engineer,
                    splitter=self.splitter,
                    artifact_manager=self.artifact_manager,
                    model_factory=self.model_factory,
                    evaluator=self.evaluator,
                    registry=self.registry,
                    run_id=trial_id
                )

                trial_orchestrator.run()

                # 3. Extract Metrics and Paths
                # We pull these from the orchestrator's artifact state
                metrics = trial_orchestrator.artifacts["metrics"]
                model_path = trial_orchestrator.artifacts["model_path"]

                # 4. Map to CandidateResult (The Contract)
                candidate_results.append(CandidateResult(
                    model_id=f"hpo_{i}",
                    metrics=metrics,
                    hyperparameters=combo,
                    artifact_path=str(model_path)
                ))
                logger.success(f"Trial {i+1} completed successfully.")

            except Exception as e:
                logger.error(f"Trial {i+1} failed: {str(e)}")
                # We do not raise; we continue to next combination to ensure batch completion
                continue

        logger.info(f"HPO Complete. Generated {len(candidate_results)} candidates.")
        return candidate_results
