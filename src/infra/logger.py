# src/infra/logger.py
import sys
from pathlib import Path
from loguru import logger
from src.core.config.schemas import LoggingConfig

def setup_logger(config: LoggingConfig, project_root: Path) -> None:
    """
    Configures the loguru logger based on the LoggingConfig dataclass.
    
    Args:
        config: The LoggingConfig instance from the AppConfig.
        project_root: The project root Path (to ensure absolute path resolution).
    """
    # Remove the default loguru handler
    logger.remove()

    # 1. Path Resolution (Law #3: Portability)
    # We resolve the file path against the project_root to ensure 
    # the logs always go to the correct directory regardless of where the 
    # execution command was triggered from.
    log_file_path = (project_root / config.file_path).resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Standard Output Handler (Stream)
    # We use the level and format defined in the config.
    logger.add(
        sys.stdout,
        level=config.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )

    # 3. File Handler (Sink)
    # Loguru requires "rotation_mb MB" (string), but our YAML provides an int/str.
    # We perform this transformation here to keep the Config clean.
    if isinstance(config.rotation_mb, (int, float)):
        rotation_str = f"{config.rotation_mb} MB"
    else:
        rotation_str = config.rotation_mb

    try:
        logger.add(
            str(log_file_path),
            rotation=rotation_str,
            retention=config.retention_days,
            level=config.level,
            serialize=False,
            enqueue=True  # Production best practice: prevents I/O blocking
        )
        logger.debug(f"File sink successfully attached at {log_file_path}")
    except Exception as e:
        # This is a critical infrastructure failure. 
        # If we can't write to logs, we must know immediately.
        import logging
        logging.critical(f"Failed to initialize loguru file sink: {e}")
        raise e
