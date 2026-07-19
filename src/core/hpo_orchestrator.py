# src/core/hpo_orchestrator.py
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List
from loguru import logger

from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator
from src.core.splitter_service import TimeSeriesSplitter
from src.core.config.schemas import ModelConfig, HPOConfig, EvaluationConfig  # adjust if split modules
from src.core.exceptions import PipelineError

@dataclass
class TrialResult:
    trial_id: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    model_path: Path
    status: str
    error: str | None = None


@dataclass
class HPOResult:
    trials: List[TrialResult]

    def best_trial(self, metric_name: str) -> TrialResult:
        successful = [t for t in self.trials if t.status == "success"]
        if not successful:
            raise PipelineError("No successful HPO trials to select from.")
        return min(successful, key=lambda t: t.metrics[metric_name])


class HPOOrchestrator:
    """
    Orchestrates hyperparameter tuning over a typed HPOConfig.
    Consumes engineered data, produces TrialResult list and best trial selection.
    """

    def __init__(
        self,
        model_factory: ModelWorkerFactory,
        evaluator: ModelEvaluator,
        splitter: TimeSeriesSplitter,
        model_cfg: ModelConfig,
        hpo_cfg: HPOConfig,
        eval_cfg: EvaluationConfig,
        run_id: str,
    ):
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.splitter = splitter
        self.model_cfg = model_cfg
        self.hpo_cfg = hpo_cfg
        self.eval_cfg = eval_cfg
        self.run_id = run_id

    def _generate_combinations(self) -> List[dict[str, Any]]:
        strategy = self.hpo_cfg.strategy
        space = self.hpo_cfg.parameters

        if strategy == "grid":
            import itertools

            keys = list(space.keys())
            values = list(space.values())
            return [dict(zip(keys, v)) for v in itertools.product(*values)]

        if strategy == "random":
            import random

            trials = self.hpo_cfg.max_trials
            return [
                {k: random.choice(v) for k, v in space.items()}
                for _ in range(trials)
            ]

        raise ValueError(f"Unsupported HPO strategy: {strategy}")

    def run(self, engineered_data: Any) -> HPOResult:
        logger.info(f"HPO: Starting search | Strategy={self.hpo_cfg.strategy}")

        X_train, y_train, X_val, y_val, _ = self.splitter.split(engineered_data)

        combinations = self._generate_combinations()
        trials: List[TrialResult] = []

        for i, params in enumerate(combinations):
            trial_id = f"hpo_{self.run_id}_{i:03d}"
            logger.info(f"HPO Trial {i+1}/{len(combinations)} | Params={params}")

            try:
                trial_model_cfg = replace(self.model_cfg, params=params)

                worker = self.model_factory.get_worker(
                    model_cfg=trial_model_cfg,
                    run_id=trial_id,
                )
                trained_model = worker.train(X_train, y_train)

                metrics = self.evaluator.evaluate(
                    trained_model,
                    X_val,
                    y_val,
                    run_id=trial_id,
                )

                model_path = worker.save_model(trained_model)

                trials.append(
                    TrialResult(
                        trial_id=trial_id,
                        params=params,
                        metrics=metrics,
                        model_path=model_path,
                        status="success",
                    )
                )
                logger.success(f"HPO Trial {trial_id} completed successfully.")

            except Exception as e:
                logger.error(f"HPO Trial {trial_id} failed: {e}")
                trials.append(
                    TrialResult(
                        trial_id=trial_id,
                        params=params,
                        metrics={},
                        model_path=Path(),
                        status="failed",
                        error=str(e),
                    )
                )

        logger.info(f"HPO: Completed {len(trials)} trials.")
        return HPOResult(trials=trials)
