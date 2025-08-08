"""
Tests for LLM-specific exception classes.

This module tests all LLM-related exceptions including quota errors,
authentication errors, rate limiting, context limits, and server errors
with their provider-specific error handling.
"""

import pytest
from typing import Any
from cognivault.exceptions import (
    CogniVaultError,
    ErrorSeverity,
    RetryPolicy,
    LLMError,
    LLMQuotaError,
    LLMAuthError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMContextLimitError,
    LLMModelNotFoundError,
    LLMServerError,
)


class TestLLMErrorBase:
    """Test base LLMError functionality."""

    def test_llm_error_creation(self) -> None:
        """Test basic LLMError creation."""
        error = LLMError(message="LLM call failed", llm_provider="openai")

        assert error.message == "LLM call failed"
        assert error.llm_provider == "openai"
        assert error.context["llm_provider"] == "openai"
        assert error.error_code == "llm_error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.BACKOFF
        assert isinstance(error, CogniVaultError)

    def test_llm_error_with_api_details(self) -> None:
        """Test LLMError with API error details."""
        error = LLMError(
            message="OpenAI API error",
            llm_provider="openai",
            error_code="openai_api_error",
            api_error_code="rate_limit_exceeded",
            api_error_type="RateLimitError",
            step_id="llm_step",
            agent_id="LLMAgent",
        )

        assert error.message == "OpenAI API error"
        assert error.error_code == "openai_api_error"
        assert error.llm_provider == "openai"
        assert error.api_error_code == "rate_limit_exceeded"
        assert error.api_error_type == "RateLimitError"
        assert error.context["llm_provider"] == "openai"
        assert error.context["api_error_code"] == "rate_limit_exceeded"
        assert error.context["api_error_type"] == "RateLimitError"
        assert error.step_id == "llm_step"
        assert error.agent_id == "LLMAgent"

    def test_llm_error_context_injection(self) -> None:
        """Test that LLM provider info is injected into context."""
        error = LLMError(
            message="Provider test",
            llm_provider="anthropic",
            context={"custom": "data"},
        )

        assert error.context["llm_provider"] == "anthropic"
        assert error.context["custom"] == "data"
        assert error.context["api_error_code"] is None
        assert error.context["api_error_type"] is None


class TestLLMQuotaError:
    """Test LLMQuotaError functionality."""

    def test_quota_error_creation(self) -> None:
        """Test basic quota error creation."""
        error = LLMQuotaError(llm_provider="openai", quota_type="api_credits")

        assert error.llm_provider == "openai"
        assert error.quota_type == "api_credits"
        assert error.error_code == "llm_quota_exceeded"
        assert error.retry_policy == RetryPolicy.NEVER  # Quota issues need manual fix
        assert error.severity == ErrorSeverity.CRITICAL

        # Check default message
        expected_msg = "openai API quota exceeded for api_credits"
        assert error.message == expected_msg

    def test_quota_error_with_custom_context(self) -> None:
        """Test quota error with additional context details."""
        context = {
            "quota_limit": 1000,
            "quota_used": 1000,
            "quota_reset_time": "2024-01-01T00:00:00Z",
        }
        error = LLMQuotaError(
            llm_provider="openai",
            quota_type="requests_per_day",
            step_id="quota_step",
            context=context,
        )

        assert error.quota_type == "requests_per_day"
        assert error.step_id == "quota_step"
        assert error.context["quota_limit"] == 1000
        assert error.context["quota_used"] == 1000
        assert error.context["quota_reset_time"] == "2024-01-01T00:00:00Z"
        assert error.context["quota_type"] == "requests_per_day"
        assert error.context["billing_check_required"] is True

    def test_quota_user_message(self) -> None:
        """Test user-friendly quota error message."""
        error = LLMQuotaError(
            llm_provider="openai",
            quota_type="tokens",
        )

        user_msg = error.get_user_message()
        assert "quota exceeded" in user_msg.lower()
        assert "billing dashboard" in user_msg.lower()
        assert "openai" in user_msg.lower()


class TestLLMAuthError:
    """Test LLMAuthError functionality."""

    def test_auth_error_creation(self) -> None:
        """Test basic authentication error creation."""
        error = LLMAuthError(llm_provider="openai", auth_issue="invalid_api_key")

        assert error.llm_provider == "openai"
        assert error.auth_issue == "invalid_api_key"
        assert error.error_code == "llm_auth_error"
        assert error.retry_policy == RetryPolicy.NEVER  # Auth issues need manual fix
        assert error.severity == ErrorSeverity.CRITICAL

        # Check default message
        expected_msg = "openai authentication failed: invalid_api_key"
        assert error.message == expected_msg

    def test_auth_error_with_details(self) -> None:
        """Test auth error with additional details."""
        context = {"auth_details": "Token expired on 2024-01-01"}
        error = LLMAuthError(
            llm_provider="anthropic",
            auth_issue="expired_token",
            step_id="auth_step",
            agent_id="AuthAgent",
            context=context,
        )

        assert error.auth_issue == "expired_token"
        assert error.step_id == "auth_step"
        assert error.agent_id == "AuthAgent"
        assert error.context["auth_details"] == "Token expired on 2024-01-01"
        assert error.context["auth_issue"] == "expired_token"
        assert error.context["api_key_check_required"] is True

    def test_auth_user_message(self) -> None:
        """Test user-friendly auth error message."""
        error = LLMAuthError(llm_provider="openai", auth_issue="invalid_api_key")

        user_msg = error.get_user_message()
        assert "authentication" in user_msg.lower()
        assert "api key" in user_msg.lower()


class TestLLMTimeoutError:
    """Test LLMTimeoutError functionality."""

    def test_timeout_error_creation(self) -> None:
        """Test basic timeout error creation."""
        error = LLMTimeoutError(
            llm_provider="openai", timeout_seconds=30.0, timeout_type="api_request"
        )

        assert error.llm_provider == "openai"
        assert error.timeout_seconds == 30.0
        assert error.timeout_type == "api_request"
        assert error.error_code == "llm_timeout"
        assert error.retry_policy == RetryPolicy.BACKOFF  # Timeouts might be temporary
        assert error.severity == ErrorSeverity.MEDIUM

        # Check default message
        expected_msg = "openai api_request timeout after 30.0s"
        assert error.message == expected_msg

    def test_timeout_error_with_details(self) -> None:
        """Test timeout error with additional details."""
        context = {"request_id": "req_123"}
        error = LLMTimeoutError(
            llm_provider="anthropic",
            timeout_seconds=45.0,
            timeout_type="streaming_response",
            step_id="timeout_step",
            context=context,
        )

        assert error.timeout_type == "streaming_response"
        assert error.step_id == "timeout_step"
        assert error.context["request_id"] == "req_123"
        assert error.context["timeout_seconds"] == 45.0
        assert error.context["timeout_type"] == "streaming_response"

    def test_timeout_user_message(self) -> None:
        """Test user-friendly timeout error message."""
        error = LLMTimeoutError(
            llm_provider="openai", timeout_seconds=30.0, timeout_type="api_request"
        )

        user_msg = error.get_user_message()
        assert "timeout" in user_msg.lower()
        assert "30.0" in user_msg


class TestLLMRateLimitError:
    """Test LLMRateLimitError functionality."""

    def test_rate_limit_error_creation(self) -> None:
        """Test basic rate limit error creation."""
        error = LLMRateLimitError(
            llm_provider="openai", rate_limit_type="requests_per_minute"
        )

        assert error.llm_provider == "openai"
        assert error.rate_limit_type == "requests_per_minute"
        assert error.error_code == "llm_rate_limit"
        assert error.retry_policy == RetryPolicy.BACKOFF  # Rate limits are temporary
        assert error.severity == ErrorSeverity.MEDIUM

        # Check default message
        expected_msg = "openai rate limit exceeded: requests_per_minute"
        assert error.message == expected_msg

    def test_rate_limit_error_with_retry_after(self) -> None:
        """Test rate limit error with retry-after information."""
        context = {"current_usage": 1000, "limit_value": 800}
        error = LLMRateLimitError(
            llm_provider="openai",
            rate_limit_type="tokens_per_minute",
            retry_after_seconds=60.0,
            step_id="rate_limit_step",
            context=context,
        )

        assert error.retry_after_seconds == 60.0
        assert error.step_id == "rate_limit_step"
        assert error.context["retry_after_seconds"] == 60.0
        assert error.context["current_usage"] == 1000
        assert error.context["limit_value"] == 800
        assert error.context["rate_limit_type"] == "tokens_per_minute"
        assert error.context["temporary_failure"] is True

    def test_rate_limit_user_message(self) -> None:
        """Test user-friendly rate limit error message."""
        error = LLMRateLimitError(
            llm_provider="openai",
            rate_limit_type="requests_per_minute",
            retry_after_seconds=120.0,
        )

        user_msg = error.get_user_message()
        assert "rate limit" in user_msg.lower()
        assert "120" in user_msg or "automatically" in user_msg


class TestLLMContextLimitError:
    """Test LLMContextLimitError functionality."""

    def test_context_limit_error_creation(self) -> None:
        """Test basic context limit error creation."""
        error = LLMContextLimitError(
            llm_provider="openai", model_name="gpt-4", token_count=8500, max_tokens=8192
        )

        assert error.llm_provider == "openai"
        assert error.model_name == "gpt-4"
        assert error.token_count == 8500
        assert error.max_tokens == 8192
        assert error.context["token_overflow"] == 308  # 8500 - 8192
        assert error.error_code == "llm_context_limit"
        assert (
            error.retry_policy == RetryPolicy.NEVER
        )  # Context limits need content reduction
        assert error.severity == ErrorSeverity.HIGH

        # Check default message
        expected_msg = "gpt-4 context limit exceeded: 8500/8192 tokens"
        assert error.message == expected_msg

    def test_context_limit_error_with_details(self) -> None:
        """Test context limit error with additional details."""
        context = {"prompt_tokens": 95000, "completion_tokens": 5000}
        error = LLMContextLimitError(
            llm_provider="anthropic",
            model_name="claude-2",
            token_count=100000,
            max_tokens=100000,
            step_id="context_step",
            context=context,
        )

        assert error.model_name == "claude-2"
        assert error.context["token_overflow"] == 0  # At limit, not over
        assert error.step_id == "context_step"
        assert error.context["prompt_tokens"] == 95000
        assert error.context["completion_tokens"] == 5000
        assert error.context["model_name"] == "claude-2"
        assert error.context["token_count"] == 100000
        assert error.context["max_tokens"] == 100000

    def test_context_limit_user_message(self) -> None:
        """Test user-friendly context limit error message."""
        error = LLMContextLimitError(
            llm_provider="openai",
            model_name="gpt-3.5-turbo",
            token_count=4200,
            max_tokens=4096,
        )

        user_msg = error.get_user_message()
        assert "input too large" in user_msg.lower()
        assert "104" in user_msg  # token overflow


class TestLLMModelNotFoundError:
    """Test LLMModelNotFoundError functionality."""

    def test_model_not_found_error_creation(self) -> None:
        """Test basic model not found error creation."""
        error = LLMModelNotFoundError(llm_provider="openai", model_name="gpt-5")

        assert error.llm_provider == "openai"
        assert error.model_name == "gpt-5"
        assert error.error_code == "llm_model_not_found"
        assert error.retry_policy == RetryPolicy.NEVER  # Model issues need config fix
        assert error.severity == ErrorSeverity.HIGH

        # Check default message
        expected_msg = "openai model 'gpt-5' not found or unavailable"
        assert error.message == expected_msg

    def test_model_not_found_with_suggestions(self) -> None:
        """Test model not found error with suggested alternatives."""
        available_models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]

        error = LLMModelNotFoundError(
            llm_provider="openai",
            model_name="invalid-model",
            available_models=available_models,
            step_id="model_step",
        )

        assert error.available_models == available_models
        assert error.step_id == "model_step"
        assert error.context["available_models"] == available_models
        assert error.context["model_name"] == "invalid-model"
        assert error.context["model_deprecated_possible"] is True

    def test_model_not_found_user_message(self) -> None:
        """Test user-friendly model not found error message."""
        error = LLMModelNotFoundError(
            llm_provider="openai",
            model_name="non-existent-model",
            available_models=["gpt-4", "gpt-3.5-turbo"],
        )

        user_msg = error.get_user_message()
        assert "model" in user_msg.lower()
        assert "non-existent-model" in user_msg
        assert "gpt-4" in user_msg


class TestLLMServerError:
    """Test LLMServerError functionality."""

    def test_server_error_creation(self) -> None:
        """Test basic server error creation."""
        error = LLMServerError(
            llm_provider="openai",
            http_status=503,
            error_details="Service temporarily unavailable",
        )

        assert error.llm_provider == "openai"
        assert error.http_status == 503
        assert error.error_details == "Service temporarily unavailable"
        assert error.error_code == "llm_server_error"
        assert (
            error.retry_policy == RetryPolicy.CIRCUIT_BREAKER
        )  # Server errors should use circuit breaker
        assert error.severity == ErrorSeverity.HIGH

        # Check default message
        expected_msg = "openai server error (HTTP 503): Service temporarily unavailable"
        assert error.message == expected_msg

    def test_server_error_with_details(self) -> None:
        """Test server error with additional details."""
        context = {"request_id": "req_456", "server_timestamp": "2024-01-01T12:00:00Z"}
        error = LLMServerError(
            llm_provider="anthropic",
            http_status=500,
            error_details="Internal server error",
            step_id="server_step",
            context=context,
        )

        assert error.step_id == "server_step"
        assert error.context["request_id"] == "req_456"
        assert error.context["server_timestamp"] == "2024-01-01T12:00:00Z"
        assert error.context["http_status"] == 500
        assert error.context["error_details"] == "Internal server error"
        assert error.context["server_side_issue"] is True

    def test_server_error_user_message(self) -> None:
        """Test user-friendly server error message."""
        error = LLMServerError(
            llm_provider="openai", http_status=502, error_details="Bad gateway"
        )

        user_msg = error.get_user_message()
        assert "server error" in user_msg.lower()
        assert "502" in user_msg
        assert "retry automatically" in user_msg.lower()


class TestLLMErrorInheritance:
    """Test proper inheritance hierarchy for LLM errors."""

    def test_all_llm_errors_inherit_from_llm_error(self) -> None:
        """Test that specialized LLM errors inherit from LLMError."""
        errors = [
            LLMQuotaError("openai", "credits"),
            LLMAuthError("openai", "invalid_key"),
            LLMTimeoutError("openai", 30.0, "request"),
            LLMRateLimitError("openai", "rpm"),
            LLMContextLimitError("openai", "gpt-4", 8500, 8192),
            LLMModelNotFoundError("openai", "gpt-5"),
            LLMServerError("openai", 503, "unavailable"),
        ]

        for error in errors:
            assert isinstance(error, LLMError)
            assert isinstance(error, CogniVaultError)
            assert hasattr(error, "llm_provider")

    def test_llm_error_inherits_from_base(self) -> None:
        """Test that LLMError inherits from CogniVaultError."""
        error = LLMError("Test", "provider")
        assert isinstance(error, CogniVaultError)

    def test_polymorphic_behavior(self) -> None:
        """Test polymorphic behavior of LLM errors."""

        def handle_llm_error(error: LLMError) -> dict:
            return {
                "provider": error.context["llm_provider"],
                "retryable": error.is_retryable(),
                "severity": error.severity.value,
                "type": error.__class__.__name__,
            }

        errors = [
            LLMError("Base", "openai"),
            LLMQuotaError("openai", "credits"),
            LLMRateLimitError("openai", "rpm"),
            LLMServerError("openai", 503, "error"),
        ]

        results = [handle_llm_error(err) for err in errors]

        assert len(results) == 4
        assert all(r["provider"] == "openai" for r in results)

        # Check specific retry behaviors
        assert results[0]["retryable"] is True  # BACKOFF
        assert results[1]["retryable"] is False  # NEVER (quota)
        assert results[2]["retryable"] is True  # BACKOFF (rate limit)
        assert results[3]["retryable"] is True  # CIRCUIT_BREAKER (server)


class TestLLMErrorIntegration:
    """Test integration aspects of LLM errors."""

    def test_provider_specific_error_handling(self) -> None:
        """Test that LLM errors handle provider-specific information."""
        openai_error = LLMRateLimitError(
            llm_provider="openai",
            rate_limit_type="requests_per_minute",
            retry_after_seconds=60.0,
            context={"model": "gpt-4", "organization": "org-123"},
        )

        anthropic_error = LLMContextLimitError(
            llm_provider="anthropic",
            model_name="claude-2",
            token_count=100000,
            max_tokens=100000,
            context={"conversation_id": "conv-456"},
        )

        # Verify provider-specific context preservation
        assert openai_error.context["llm_provider"] == "openai"
        assert openai_error.context["model"] == "gpt-4"
        assert openai_error.context["organization"] == "org-123"

        assert anthropic_error.context["llm_provider"] == "anthropic"
        assert anthropic_error.context["conversation_id"] == "conv-456"

    def test_api_error_mapping(self) -> None:
        """Test that API error codes are properly mapped."""
        error = LLMError(
            message="API mapping test",
            llm_provider="openai",
            api_error_code="insufficient_quota",
            api_error_type="QuotaExceededError",
            context={"original_status": 429},
        )

        assert error.context["api_error_code"] == "insufficient_quota"
        assert error.context["api_error_type"] == "QuotaExceededError"
        assert error.context["original_status"] == 429

    def test_exception_raising_and_catching(self) -> None:
        """Test that LLM errors can be properly raised and caught."""
        # Test specific exception catching
        with pytest.raises(LLMQuotaError) as exc_info:
            raise LLMQuotaError("openai", "credits")

        assert exc_info.value.llm_provider == "openai"
        assert exc_info.value.quota_type == "credits"

        # Test catching as base LLM error
        with pytest.raises(LLMError) as exc_info:
            raise LLMTimeoutError("anthropic", 30.0, "streaming")

        assert exc_info.value.context["llm_provider"] == "anthropic"

        # Test catching as CogniVaultError
        with pytest.raises(CogniVaultError) as exc_info:
            raise LLMModelNotFoundError("openai", "invalid-model")

        assert "invalid-model" in str(exc_info.value)
