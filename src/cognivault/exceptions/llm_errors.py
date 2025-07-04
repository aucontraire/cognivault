"""
LLM-specific exceptions for CogniVault.

This module defines exceptions related to Language Model interactions,
API failures, and LLM provider-specific error conditions with intelligent
retry policies and circuit breaker support.
"""

from typing import Optional, Dict, Any
from . import CogniVaultError, ErrorSeverity, RetryPolicy


class LLMError(CogniVaultError):
    """
    Base exception for LLM-related failures.

    Represents errors that occur during interactions with Language Model
    providers, designed for intelligent retry and circuit breaker patterns.
    """

    def __init__(
        self,
        message: str,
        llm_provider: str,
        error_code: str = "llm_error",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        retry_policy: RetryPolicy = RetryPolicy.BACKOFF,
        context: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        cause: Optional[Exception] = None,
        api_error_code: Optional[str] = None,
        api_error_type: Optional[str] = None,
    ):
        context = context or {}
        context.update(
            {
                "llm_provider": llm_provider,
                "api_error_code": api_error_code,
                "api_error_type": api_error_type,
            }
        )

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            retry_policy=retry_policy,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.llm_provider = llm_provider
        self.api_error_code = api_error_code
        self.api_error_type = api_error_type


class LLMQuotaError(LLMError):
    """
    Exception raised when LLM API quota is exhausted.

    Represents quota/billing limit exceeded errors that require
    manual intervention (billing update) and should not be retried.
    """

    def __init__(
        self,
        llm_provider: str,
        quota_type: str = "tokens",
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = message or f"{llm_provider} API quota exceeded for {quota_type}"

        context = context or {}
        context.update({"quota_type": quota_type, "billing_check_required": True})

        super().__init__(
            message=message,
            llm_provider=llm_provider,
            error_code="llm_quota_exceeded",
            severity=ErrorSeverity.CRITICAL,
            retry_policy=RetryPolicy.NEVER,  # Quota issues need billing fix
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
            api_error_code="insufficient_quota",
        )
        self.quota_type = quota_type

    def get_user_message(self) -> str:
        """Get user-friendly error message with billing guidance."""
        return (
            f"❌ {self.agent_id or 'LLM'} failed: API quota exceeded ({self.quota_type})\n"
            f"💡 Tip: Check your {self.llm_provider} billing dashboard and add credits."
        )


class LLMAuthError(LLMError):
    """
    Exception raised when LLM API authentication fails.

    Represents authentication failures due to invalid API keys,
    expired tokens, or insufficient permissions.
    """

    def __init__(
        self,
        llm_provider: str,
        auth_issue: str = "invalid_api_key",
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = message or f"{llm_provider} authentication failed: {auth_issue}"

        context = context or {}
        context.update({"auth_issue": auth_issue, "api_key_check_required": True})

        super().__init__(
            message=message,
            llm_provider=llm_provider,
            error_code="llm_auth_error",
            severity=ErrorSeverity.CRITICAL,
            retry_policy=RetryPolicy.NEVER,  # Auth issues need manual fix
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
            api_error_code="invalid_api_key",
        )
        self.auth_issue = auth_issue

    def get_user_message(self) -> str:
        """Get user-friendly error message with API key guidance."""
        return (
            f"❌ {self.agent_id or 'LLM'} failed: Authentication error ({self.auth_issue})\n"
            f"💡 Tip: Check your {self.llm_provider} API key in .env file."
        )


class LLMRateLimitError(LLMError):
    """
    Exception raised when LLM API rate limits are hit.

    Represents temporary rate limiting that should be retried
    with exponential backoff and circuit breaker patterns.
    """

    def __init__(
        self,
        llm_provider: str,
        rate_limit_type: str = "requests_per_minute",
        retry_after_seconds: Optional[float] = None,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = message or f"{llm_provider} rate limit exceeded: {rate_limit_type}"

        context = context or {}
        context.update(
            {
                "rate_limit_type": rate_limit_type,
                "retry_after_seconds": retry_after_seconds,
                "temporary_failure": True,
            }
        )

        super().__init__(
            message=message,
            llm_provider=llm_provider,
            error_code="llm_rate_limit",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=RetryPolicy.BACKOFF,  # Rate limits should be retried with backoff
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
            api_error_code="rate_limit_exceeded",
        )
        self.rate_limit_type = rate_limit_type
        self.retry_after_seconds = retry_after_seconds

    def get_user_message(self) -> str:
        """Get user-friendly error message with rate limit guidance."""
        retry_msg = ""
        if self.retry_after_seconds:
            retry_msg = f" (retry in {self.retry_after_seconds}s)"

        return (
            f"❌ {self.agent_id or 'LLM'} failed: Rate limit exceeded{retry_msg}\n"
            f"💡 Tip: The system will retry automatically with backoff."
        )


class LLMTimeoutError(LLMError):
    """
    Exception raised when LLM API requests time out.

    Represents timeout failures that may be retried depending
    on the timeout duration and context.
    """

    def __init__(
        self,
        llm_provider: str,
        timeout_seconds: float,
        timeout_type: str = "request",
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message or f"{llm_provider} {timeout_type} timeout after {timeout_seconds}s"
        )

        context = context or {}
        context.update(
            {
                "timeout_seconds": timeout_seconds,
                "timeout_type": timeout_type,
                "network_issue_possible": True,
            }
        )

        super().__init__(
            message=message,
            llm_provider=llm_provider,
            error_code="llm_timeout",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=RetryPolicy.BACKOFF,  # Timeouts might be temporary
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
            api_error_code="timeout",
        )
        self.timeout_seconds = timeout_seconds
        self.timeout_type = timeout_type

    def get_user_message(self) -> str:
        """Get user-friendly error message with timeout guidance."""
        return (
            f"❌ {self.agent_id or 'LLM'} failed: Request timeout ({self.timeout_seconds}s)\n"
            f"💡 Tip: Check your network connection or try a simpler query."
        )


class LLMContextLimitError(LLMError):
    """
    Exception raised when LLM context/token limits are exceeded.

    Represents input context that's too large for the model,
    requiring input reduction or chunking strategies.
    """

    def __init__(
        self,
        llm_provider: str,
        model_name: str,
        token_count: int,
        max_tokens: int,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message
            or f"{model_name} context limit exceeded: {token_count}/{max_tokens} tokens"
        )

        context = context or {}
        context.update(
            {
                "model_name": model_name,
                "token_count": token_count,
                "max_tokens": max_tokens,
                "token_overflow": token_count - max_tokens,
            }
        )

        super().__init__(
            message=message,
            llm_provider=llm_provider,
            error_code="llm_context_limit",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # Context limits need input reduction
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
            api_error_code="context_length_exceeded",
        )
        self.model_name = model_name
        self.token_count = token_count
        self.max_tokens = max_tokens

    def get_user_message(self) -> str:
        """Get user-friendly error message with context limit guidance."""
        overflow = self.token_count - self.max_tokens
        return (
            f"❌ {self.agent_id or 'LLM'} failed: Input too large ({overflow} tokens over limit)\n"
            f"💡 Tip: Reduce query length or break into smaller parts."
        )


class LLMModelNotFoundError(LLMError):
    """
    Exception raised when the specified LLM model is not available.

    Represents model availability issues, deprecation, or
    incorrect model names.
    """

    def __init__(
        self,
        llm_provider: str,
        model_name: str,
        available_models: Optional[list] = None,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message or f"{llm_provider} model '{model_name}' not found or unavailable"
        )

        context = context or {}
        context.update(
            {
                "model_name": model_name,
                "available_models": available_models or [],
                "model_deprecated_possible": True,
            }
        )

        super().__init__(
            message=message,
            llm_provider=llm_provider,
            error_code="llm_model_not_found",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # Model issues need config fix
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
            api_error_code="model_not_found",
        )
        self.model_name = model_name
        self.available_models = available_models or []

    def get_user_message(self) -> str:
        """Get user-friendly error message with model guidance."""
        base_msg = (
            f"❌ {self.agent_id or 'LLM'} failed: Model '{self.model_name}' not available\n"
            f"💡 Tip: Check model name in configuration."
        )

        if self.available_models:
            models_str = ", ".join(self.available_models[:3])
            base_msg += f" Available models: {models_str}"
            if len(self.available_models) > 3:
                base_msg += "..."

        return base_msg


class LLMServerError(LLMError):
    """
    Exception raised when LLM provider has server-side issues.

    Represents 5xx errors from the LLM provider that should
    trigger circuit breaker patterns and aggressive retries.
    """

    def __init__(
        self,
        llm_provider: str,
        http_status: int,
        error_details: Optional[str] = None,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message
            or f"{llm_provider} server error (HTTP {http_status}): {error_details or 'Unknown'}"
        )

        context = context or {}
        context.update(
            {
                "http_status": http_status,
                "error_details": error_details,
                "server_side_issue": True,
            }
        )

        super().__init__(
            message=message,
            llm_provider=llm_provider,
            error_code="llm_server_error",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.CIRCUIT_BREAKER,  # Server errors should use circuit breaker
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
            api_error_code=f"http_{http_status}",
        )
        self.http_status = http_status
        self.error_details = error_details

    def get_user_message(self) -> str:
        """Get user-friendly error message with server error guidance."""
        return (
            f"❌ {self.agent_id or 'LLM'} failed: {self.llm_provider} server error (HTTP {self.http_status})\n"
            f"💡 Tip: This is a provider issue. The system will retry automatically."
        )
