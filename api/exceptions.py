"""Domain-specific exceptions for the brain tumor segmentation API."""


class APIError(Exception):
    """Base exception for all API errors."""

    def __init__(self, message: str, client_message: str | None = None):
        self.message = message
        self.client_message = client_message or message
        super().__init__(self.message)


class ValidationError(APIError):
    """Raised when input validation fails."""

    def __init__(self, message: str, client_message: str | None = None):
        super().__init__(message, client_message or "Validation failed")


class CheckpointError(APIError):
    """Raised when checkpoint file is not found or invalid."""

    def __init__(self, message: str, client_message: str | None = None):
        super().__init__(message, client_message or "Checkpoint error")


class InferenceError(APIError):
    """Raised when inference execution fails."""

    def __init__(self, message: str, client_message: str | None = None):
        super().__init__(message, client_message or "Inference failed")
