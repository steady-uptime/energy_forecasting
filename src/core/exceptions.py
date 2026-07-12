# src/core/exceptions.py
from datetime import datetime, timezone
from typing import Any, Optional

class PipelineError(Exception):
    """Base class for all pipeline-related errors with context."""
    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def __str__(self) -> str:
        return f"{self.message} | context={self.context} | ts={self.timestamp}"

# --- Data Engineering Exceptions ---
class DataValidationError(PipelineError):
    """Raised when data fails schema or contract validation."""
    pass

class IngestionError(PipelineError):
    """Raised when raw data cannot be loaded or parsed."""
    pass

class PreprocessingError(PipelineError):
    """Raised when data cleaning or transformation fails."""
    pass

class FeatureEngineeringError(PipelineError):
    """Raised when feature creation or transformation fails."""
    pass

class SplitError(PipelineError):
    """Raised when dataset splitting fails."""
    pass

# --- ML Lifecycle & Infrastructure Exceptions ---
class ConfigError(PipelineError):
    """Raised when configuration validation fails."""
    pass

class ModelTrainingError(PipelineError):
    """Raised when model training encounters an unrecoverable issue."""
    pass

class ArtifactError(PipelineError):
    """Raised when saving or loading artifacts fails."""
    pass

class RegistryError(PipelineError):
    """Raised when interacting with the Model Registry fails."""
    pass

class ComparisonError(PipelineError):
    """Raised when model comparison or promotion logic fails."""
    pass

class MonitoringError(PipelineError):
    """Raised when monitoring pipeline fails."""
    pass
