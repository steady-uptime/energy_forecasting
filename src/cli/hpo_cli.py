# src/cli/hpo_cli.py
import argparse
import sys
from pathlib import Path
from loguru import logger
from core.config import ConfigLoader # Your existing Singleton
from pipelines.hpo_pipeline import HPOTrainingPipeline

def main():
    parser = argparse.ArgumentParser(description="MLOps Hyperparameter Optimization Job")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--run_id", type=str, default="hpo_batch_001", help="Unique ID for this run")
    
    args = parser.parse_args()

    try:
        # 1. Load Configuration (Static)
        config = ConfigLoader.load(args.config)
        
        # 2. Bootstrap Pipeline (Injecting both Config and Context)
        # We pass run_id separately to preserve the immutability of 'config'
        pipeline = HPOTrainingPipeline(config, run_id=args.run_id)
        pipeline.run()

    except Exception as e:
        logger.critical(f"System Failure: {e}")
        sys.exit(1)