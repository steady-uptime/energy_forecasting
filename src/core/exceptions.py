# src/core/exceptions.py
from datetime import datetime, timezone

class PipelineError(Exception):
    """Base class for all pipeline-related errors with context."""
    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def __str__(self):
        return f"{self.message} | context={self.context} | ts={self.timestamp}"

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


class ModelTrainingError(PipelineError):
    """Raised when model training encounters an unrecoverable issue."""
    pass


class ArtifactError(PipelineError):
    """Raised when saving or loading artifacts fails."""
    pass
