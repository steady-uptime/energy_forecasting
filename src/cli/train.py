# src/cli/train.py

# --- Entry Point ---
from pathlib import Path
from loguru import logger  
from src.core.config_loader import config  # This triggers ConfigLoader()
from src.infra.logger import setup_logger


def main():
    # 1. Initialize the production logger immediately after config load
    # We pass the project_root from the config object we just loaded
    setup_logger(config.logging, Path(config.project_root))
    
    logger.info("System: Production logger initialized. Starting training pipeline.")

if __name__ == "__main__":
    main()
