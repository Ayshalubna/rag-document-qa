"""Domain exceptions, mapped to HTTP responses in the API layer."""


class RagQaError(Exception):
    """Base class for all domain errors."""


class IndexNotReadyError(RagQaError):
    """Raised when a query arrives before any documents have been indexed."""


class UnsupportedFileTypeError(RagQaError):
    """Raised when an uploaded/ingested file type has no registered loader."""


class EmptyDocumentError(RagQaError):
    """Raised when a document yields no extractable text."""


class LLMUnavailableError(RagQaError):
    """Raised when the LLM backend cannot be reached or times out."""
