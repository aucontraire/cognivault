"""
CogniVault Exception Hierarchy

This module provides a comprehensive exception hierarchy for CogniVault,
designed for LangGraph compatibility with trace metadata and structured
error context for robust error handling and observability.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryPolicy(Enum):
    """Retry policy classification for exceptions."""

    NEVER = "never"  # Permanent failures (invalid API key, missing files)
    IMMEDIATE = "immediate"  # Retry immediately (temporary network glitch)
    BACKOFF = "backoff"  # Retry with exponential backoff (rate limits, quota)
    CIRCUIT_BREAKER = "circuit_breaker"  # Use circuit breaker pattern (API down)


class CogniVaultError(Exception):
    """
    Base exception class for all CogniVault errors.

    Provides structured error context, trace metadata, and LangGraph compatibility
    for robust error handling and observability in multi-agent workflows.

    Attributes
    ----------
    message : str
        Human-readable error message
    error_code : str
        Machine-readable error code for categorization
    severity : ErrorSeverity
        Error severity level
    retry_policy : RetryPolicy
        Retry classification for this error type
    context : Dict[str, Any]
        Additional error context and metadata
    step_id : Optional[str]
        Execution step identifier for trace tracking
    agent_id : Optional[str]
        Agent identifier that raised the error
    timestamp : datetime
        When the error occurred
    cause : Optional[Exception]
        Original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retry_policy: RetryPolicy = RetryPolicy.NEVER,
        context: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.retry_policy = retry_policy
        self.context = dict(context or {})  # Create a copy to avoid mutation
        self.step_id = step_id
        self.agent_id = agent_id
        self.timestamp = datetime.now(timezone.utc)
        self.cause = cause

        # Add trace metadata to context
        self.context.update(
            {
                "step_id": self.step_id,
                "agent_id": self.agent_id,
                "timestamp": self.timestamp.isoformat(),
                "error_code": self.error_code,
                "severity": self.severity.value,
                "retry_policy": self.retry_policy.value,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging and serialization.

        Returns
        -------
        Dict[str, Any]
            Structured error data with all metadata
        """
        return {
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "retry_policy": self.retry_policy.value,
            "context": self.context,
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
            "exception_type": self.__class__.__name__,
        }

    def is_retryable(self) -> bool:
        """
        Check if this exception should be retried.

        Returns
        -------
        bool
            True if the error is retryable, False otherwise
        """
        return self.retry_policy in [
            RetryPolicy.IMMEDIATE,
            RetryPolicy.BACKOFF,
            RetryPolicy.CIRCUIT_BREAKER,
        ]

    def should_use_circuit_breaker(self) -> bool:
        """
        Check if this exception should trigger circuit breaker logic.

        Returns
        -------
        bool
            True if circuit breaker should be used, False otherwise
        """
        return self.retry_policy == RetryPolicy.CIRCUIT_BREAKER

    def get_user_message(self) -> str:
        """
        Get user-friendly error message with actionable guidance.

        Returns
        -------
        str
            User-friendly error message
        """
        base_message = f"❌ {self.agent_id or 'System'} failed: {self.message}"

        if self.error_code in ["llm_quota_exceeded", "llm_auth_error"]:
            base_message += "\n💡 Tip: Check your API key and billing dashboard."
        elif self.error_code in ["agent_dependency_missing"]:
            base_message += (
                "\n💡 Tip: Ensure all required agents completed successfully."
            )
        elif self.error_code in ["config_invalid"]:
            base_message += "\n💡 Tip: Check your configuration file for errors."

        return base_message

    def __str__(self) -> str:
        """String representation with trace metadata."""
        base = f"{self.__class__.__name__}: {self.message}"
        if self.agent_id:
            base += f" (agent: {self.agent_id})"
        if self.step_id:
            base += f" (step: {self.step_id})"
        return base

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"severity={self.severity.value}, "
            f"retry_policy={self.retry_policy.value}, "
            f"agent_id='{self.agent_id}', "
            f"step_id='{self.step_id}'"
            f")"
        )


# Import all exception types for convenient access
from .agent_errors import (
    AgentExecutionError,
    AgentDependencyMissingError,
    AgentTimeoutError,
    AgentConfigurationError,
    AgentResourceError,
    AgentValidationError,
)
from .llm_errors import (
    LLMError,
    LLMQuotaError,
    LLMAuthError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMContextLimitError,
    LLMModelNotFoundError,
    LLMServerError,
)
from .config_errors import (
    ConfigurationError,
    ConfigValidationError,
    EnvironmentError,
    APIKeyMissingError,
    ConfigFileError,
    ModelConfigurationError,
)
from .orchestration_errors import (
    OrchestrationError,
    PipelineExecutionError,
    DependencyResolutionError,
    WorkflowTimeoutError,
    StateTransitionError,
)
from .io_errors import (
    IOError,
    FileOperationError,
    DiskSpaceError,
    PermissionError,
    MarkdownExportError,
    DirectoryCreationError,
)


__all__ = [
    # Base classes
    "CogniVaultError",
    "ErrorSeverity",
    "RetryPolicy",
    # Agent errors
    "AgentExecutionError",
    "AgentDependencyMissingError",
    "AgentTimeoutError",
    "AgentConfigurationError",
    "AgentResourceError",
    "AgentValidationError",
    # LLM errors
    "LLMError",
    "LLMQuotaError",
    "LLMAuthError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMContextLimitError",
    "LLMModelNotFoundError",
    "LLMServerError",
    # Configuration errors
    "ConfigurationError",
    "ConfigValidationError",
    "EnvironmentError",
    "APIKeyMissingError",
    "ConfigFileError",
    "ModelConfigurationError",
    # Orchestration errors
    "OrchestrationError",
    "PipelineExecutionError",
    "DependencyResolutionError",
    "WorkflowTimeoutError",
    "StateTransitionError",
    # IO errors
    "IOError",
    "FileOperationError",
    "DiskSpaceError",
    "PermissionError",
    "MarkdownExportError",
    "DirectoryCreationError",
]
