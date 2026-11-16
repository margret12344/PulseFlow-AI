class LLMError(Exception):
    """Base exception for LLM-related errors."""


class LLMConnectionError(LLMError):
    """Raised when the LLM provider cannot be reached."""


class LLMRequestError(LLMError):
    """Raised when an LLM request fails."""


class LLMResponseError(LLMError):
    """Raised when the LLM provider returns an invalid response."""