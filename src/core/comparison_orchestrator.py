# src/core/comparison_orchestrator.py
from typing import List, Dict, Any, Optional
from loguru import logger
from src.core.comparison_service import ModelComparisonService, CandidateResult
from src.core.validator import DataValidator
from src.core.model_validator import ModelMetricValidator
from src.core.exceptions import ComparisonError
from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator

class ComparisonOrchestrator:
    """
    Orchestrates the lifecycle of a model comparison experiment.
    
    Supports two execution modes:
    1. Heavy: Full training/evaluation of a suite of models.
    2. Lightweight: Promotion of pre-trained candidates (e.g., from HPO).
    """
    def __init__(
        self,
        config: Dict[str, Any],
        worker_factory: ModelWorkerFactory,
        evaluator: ModelEvaluator,
        registry: Any,
        comparison_service: ModelComparisonService,
        data_validator: DataValidator,
        metric_validator: ModelMetricValidator
    ):
        self.config = config
        self.worker_factory = worker_factory
        self.evaluator = evaluator
        self.registry = registry
        self.comparison_service = comparison_service
        self.data_validator = data_validator
        self.metric_validator = metric_validator

    # ---------------------------------------------------------
    # Entry Point 1: Heavy (Full Lifecycle)
    # ---------------------------------------------------------
    def run(self, train_data: Any, test_data: Any) -> CandidateResult:
        """
        Standard execution: Validates data, trains candidates, and promotes champion.
        """
        run_id = self.config.get("run_id", "manual_run")
        logger.info(f"Starting Heavy Comparison Experiment: {run_id}")

        try:
            # --- GATE 1: Data Integrity ---
            logger.info("Step 1: Validating input data contracts...")
            self.data_validator.validate(train_data)
            self.data_validator.validate(test_data)

            # --- Step 2: Baseline Evaluation ---
            baseline_cfg = self.config['comparison_suite']['baseline_model']
            baseline_worker = self.worker_factory.get_worker(baseline_cfg, run_id)
            
            logger.info(f"Training baseline model: {baseline_cfg['type']}")
            baseline_worker.train(train_data, None) 
            
            baseline_metrics = self.evaluator.evaluate(
                baseline_worker.model, test_data, None, run_id
            )
            
            self.metric_validator.validate(baseline_metrics, "baseline")
            
            baseline_result = CandidateResult(
                model_id="baseline",
                metrics=baseline_metrics,
                hyperparameters=baseline_cfg['params'],
                artifact_path=baseline_worker.get_path()
            )

            # --- Step 3: Candidate Evaluation ---
            candidates: List[CandidateResult] = []
            candidate_configs = self.config['comparison_suite']['candidates']

            for cand_cfg in candidate_configs:
                cand_id = cand_cfg['id']
                logger.info(f"Evaluating candidate: {cand_id}")
                
                worker = self.worker_factory.get_worker(cand_cfg, run_id)
                worker.train(train_data, None)
                
                metrics = self.evaluator.evaluate(worker.model, test_data, None, run_id)
                
                # --- GATE 2: Metric Quality ---
                self.metric_validator.validate(metrics, cand_id)
                
                candidates.append(CandidateResult(
                    model_id=cand_id,
                    metrics=metrics,
                    hyperparameters=cand_cfg['params'],
                    artifact_path=worker.get_path()
                ))

            # --- Step 4: Promotion ---
            return self._promote_champion(run_id, baseline_result, candidates)

        except Exception as e:
            logger.error(f"Heavy Experiment {run_id} failed: {str(e)}")
            raise ComparisonError(f"Heavy orchestration failed: {str(e)}")

    # ---------------------------------------------------------
    # Entry Point 2: Lightweight (HPO Handoff)
    # ---------------------------------------------------------
    def run_comparison_from_results(
        self, 
        run_id: str, 
        baseline_result: CandidateResult, 
        candidates: List[CandidateResult]
    ) -> CandidateResult:
        """
        Lightweight execution: Accepts pre-trained results (from HPO) and promotes champion.
        """
        logger.info(f"Starting Lightweight Comparison Experiment: {run_id}")
        
        try:
            # Skip data validation as training is already complete
            return self._promote_champion(run_id, baseline_result, candidates)
        
        except Exception as e:
            logger.error(f"Lightweight Experiment {run_id} failed: {str(e)}")
            raise ComparisonError(f"Lightweight orchestration failed: {str(e)}")

    # ---------------------------------------------------------
    # Private Promotion Logic (Shared by both paths)
    # ---------------------------------------------------------
    def _promote_champion(
        self, 
        run_id: str, 
        baseline: CandidateResult, 
        candidates: List[CandidateResult]
    ) -> CandidateResult:
        """
        Shared logic for champion selection, reporting, and registry updates.
        """
        logger.info("Step 4: Selecting Champion and generating reports...")
        
        # Selection
        champion = self.comparison_service.select_champion(baseline, candidates)

        # Reporting
        self.comparison_service.save_report(
            run_id=run_id,
            baseline=baseline,
            candidates=candidates,
            winner=champion
        )

        # Registry
        self.registry.register_champion(
            model_id=champion.model_id,
            metrics=champion.metrics,
            artifact_path=champion.artifact_path
        )

        logger.success(f"Experiment {run_id} complete. Champion promoted: {champion.model_id}")
        return champion
