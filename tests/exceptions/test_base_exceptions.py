"""
Tests for the base CogniVault exception classes.

This module tests the core CogniVaultError class and the foundational
exception hierarchy functionality including trace metadata, error context,
and retry policies.
"""

import pytest
from datetime import datetime
from cognivault.exceptions import (
    CogniVaultError,
    ErrorSeverity,
    RetryPolicy,
)


class TestErrorSeverityEnum:
    """Test ErrorSeverity enum functionality."""

    def test_severity_values(self):
        """Test that severity enum has correct values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_severity_comparison(self):
        """Test severity enum ordering if needed."""
        severities = [
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL,
        ]
        assert len(severities) == 4


class TestRetryPolicyEnum:
    """Test RetryPolicy enum functionality."""

    def test_retry_policy_values(self):
        """Test that retry policy enum has correct values."""
        assert RetryPolicy.NEVER.value == "never"
        assert RetryPolicy.IMMEDIATE.value == "immediate"
        assert RetryPolicy.BACKOFF.value == "backoff"
        assert RetryPolicy.CIRCUIT_BREAKER.value == "circuit_breaker"

    def test_retry_policy_classification(self):
        """Test retry policy semantic grouping."""
        retryable_policies = [
            RetryPolicy.IMMEDIATE,
            RetryPolicy.BACKOFF,
            RetryPolicy.CIRCUIT_BREAKER,
        ]
        non_retryable_policies = [RetryPolicy.NEVER]

        assert len(retryable_policies) == 3
        assert len(non_retryable_policies) == 1


class TestCogniVaultErrorBase:
    """Test the base CogniVaultError class functionality."""

    def test_minimal_error_creation(self):
        """Test basic error creation with only required parameters."""
        error = CogniVaultError(
            message="Test error",
            error_code="test_error",
        )

        assert error.message == "Test error"
        assert error.error_code == "test_error"
        assert error.severity == ErrorSeverity.MEDIUM  # default
        assert error.retry_policy == RetryPolicy.NEVER  # default
        assert isinstance(error.timestamp, datetime)
        assert error.step_id is None
        assert error.agent_id is None
        assert error.cause is None
        assert isinstance(error.context, dict)

    def test_error_with_all_parameters(self):
        """Test error creation with all optional parameters."""
        cause_exception = ValueError("Original error")
        context = {"key": "value", "count": 42}

        error = CogniVaultError(
            message="Complex error",
            error_code="complex_error",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.BACKOFF,
            context=context,
            step_id="step_123",
            agent_id="TestAgent",
            cause=cause_exception,
        )

        assert error.message == "Complex error"
        assert error.error_code == "complex_error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.BACKOFF
        assert error.step_id == "step_123"
        assert error.agent_id == "TestAgent"
        assert error.cause == cause_exception

        # Check original context preserved
        assert "key" in error.context
        assert error.context["key"] == "value"
        assert error.context["count"] == 42

        # Check trace metadata is added to context
        assert error.context["step_id"] == "step_123"
        assert error.context["agent_id"] == "TestAgent"
        assert error.context["error_code"] == "complex_error"
        assert error.context["severity"] == "high"
        assert error.context["retry_policy"] == "backoff"

    def test_error_context_isolation(self):
        """Test that context modifications don't affect original dict."""
        original_context = {"shared": "data"}

        error = CogniVaultError(
            message="Context test", error_code="context_test", context=original_context
        )

        # Modifying error context shouldn't affect original
        error.context["new_key"] = "new_value"
        assert "new_key" not in original_context
        assert "shared" in error.context
        assert original_context["shared"] == "data"

    def test_to_dict_serialization(self):
        """Test error serialization to dictionary."""
        error = CogniVaultError(
            message="Serialization test",
            error_code="serialization_test",
            severity=ErrorSeverity.CRITICAL,
            step_id="ser_step",
            agent_id="SerAgent",
        )

        data = error.to_dict()

        # Check all expected fields present
        expected_fields = [
            "message",
            "error_code",
            "severity",
            "retry_policy",
            "context",
            "step_id",
            "agent_id",
            "timestamp",
            "cause",
            "exception_type",
        ]
        for field in expected_fields:
            assert field in data

        # Check specific values
        assert data["message"] == "Serialization test"
        assert data["error_code"] == "serialization_test"
        assert data["severity"] == "critical"
        assert data["step_id"] == "ser_step"
        assert data["agent_id"] == "SerAgent"
        assert data["exception_type"] == "CogniVaultError"
        assert data["cause"] is None

    def test_to_dict_with_cause(self):
        """Test serialization with cause exception."""
        cause = RuntimeError("Original problem")
        error = CogniVaultError(
            message="Caused error", error_code="caused_error", cause=cause
        )

        data = error.to_dict()
        assert data["cause"] == "Original problem"

    def test_is_retryable_logic(self):
        """Test retry policy checking logic."""
        # Non-retryable error
        error1 = CogniVaultError(
            message="Never retry", error_code="never", retry_policy=RetryPolicy.NEVER
        )
        assert not error1.is_retryable()

        # Retryable errors
        retryable_policies = [
            RetryPolicy.IMMEDIATE,
            RetryPolicy.BACKOFF,
            RetryPolicy.CIRCUIT_BREAKER,
        ]

        for policy in retryable_policies:
            error = CogniVaultError(
                message=f"Retry with {policy.value}",
                error_code=f"retry_{policy.value}",
                retry_policy=policy,
            )
            assert error.is_retryable(), f"Policy {policy.value} should be retryable"

    def test_should_use_circuit_breaker_logic(self):
        """Test circuit breaker policy checking."""
        # Circuit breaker policy
        error1 = CogniVaultError(
            message="Circuit breaker",
            error_code="cb",
            retry_policy=RetryPolicy.CIRCUIT_BREAKER,
        )
        assert error1.should_use_circuit_breaker()

        # Non-circuit breaker policies
        non_cb_policies = [
            RetryPolicy.NEVER,
            RetryPolicy.IMMEDIATE,
            RetryPolicy.BACKOFF,
        ]

        for policy in non_cb_policies:
            error = CogniVaultError(
                message=f"No CB {policy.value}",
                error_code=f"no_cb_{policy.value}",
                retry_policy=policy,
            )
            assert not error.should_use_circuit_breaker(), (
                f"Policy {policy.value} should not use circuit breaker"
            )

    def test_get_user_message_default(self):
        """Test default user-friendly error message."""
        error = CogniVaultError(
            message="Generic error", error_code="generic_error", agent_id="GenericAgent"
        )

        user_msg = error.get_user_message()
        assert "❌ GenericAgent failed: Generic error" in user_msg

    def test_get_user_message_no_agent(self):
        """Test user message when no agent is specified."""
        error = CogniVaultError(message="System error", error_code="system_error")

        user_msg = error.get_user_message()
        assert "❌ System failed: System error" in user_msg

    def test_get_user_message_with_tips(self):
        """Test user messages with specific error code tips."""
        # Test LLM quota error tip
        error1 = CogniVaultError(
            message="API quota exceeded",
            error_code="llm_quota_exceeded",
            agent_id="LLMAgent",
        )
        user_msg1 = error1.get_user_message()
        assert "💡 Tip: Check your API key and billing dashboard." in user_msg1

        # Test dependency error tip
        error2 = CogniVaultError(
            message="Missing dependency",
            error_code="agent_dependency_missing",
            agent_id="DependentAgent",
        )
        user_msg2 = error2.get_user_message()
        assert "💡 Tip: Ensure all required agents completed successfully." in user_msg2

        # Test config error tip
        error3 = CogniVaultError(
            message="Invalid config",
            error_code="config_invalid",
            agent_id="ConfigAgent",
        )
        user_msg3 = error3.get_user_message()
        assert "💡 Tip: Check your configuration file for errors." in user_msg3

    def test_string_representation(self):
        """Test __str__ method output."""
        error = CogniVaultError(
            message="String test",
            error_code="string_test",
            agent_id="StringAgent",
            step_id="str_step",
        )

        str_repr = str(error)
        assert "CogniVaultError: String test" in str_repr
        assert "(agent: StringAgent)" in str_repr
        assert "(step: str_step)" in str_repr

    def test_string_representation_minimal(self):
        """Test __str__ method with minimal information."""
        error = CogniVaultError(message="Minimal test", error_code="minimal_test")

        str_repr = str(error)
        assert str_repr == "CogniVaultError: Minimal test"

    def test_repr_representation(self):
        """Test __repr__ method output."""
        error = CogniVaultError(
            message="Repr test",
            error_code="repr_test",
            severity=ErrorSeverity.LOW,
            agent_id="ReprAgent",
            step_id="repr_step",
        )

        repr_str = repr(error)
        assert "CogniVaultError(" in repr_str
        assert "message='Repr test'" in repr_str
        assert "error_code='repr_test'" in repr_str
        assert "severity=low" in repr_str
        assert "agent_id='ReprAgent'" in repr_str
        assert "step_id='repr_step'" in repr_str

    def test_exception_inheritance(self):
        """Test that CogniVaultError inherits from Exception properly."""
        error = CogniVaultError(
            message="Inheritance test", error_code="inheritance_test"
        )

        assert isinstance(error, Exception)
        assert isinstance(error, CogniVaultError)

        # Test exception raising
        with pytest.raises(CogniVaultError) as exc_info:
            raise error

        assert exc_info.value == error
        assert str(exc_info.value) == str(error)


class TestErrorContextIntegration:
    """Test error context integration with trace metadata."""

    def test_trace_metadata_injection(self):
        """Test that trace metadata is automatically injected into context."""
        error = CogniVaultError(
            message="Trace test",
            error_code="trace_test",
            step_id="trace_step_456",
            agent_id="TraceAgent",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.BACKOFF,
        )

        # Verify trace metadata in context
        assert error.context["step_id"] == "trace_step_456"
        assert error.context["agent_id"] == "TraceAgent"
        assert error.context["error_code"] == "trace_test"
        assert error.context["severity"] == "high"
        assert error.context["retry_policy"] == "backoff"
        assert "timestamp" in error.context

    def test_context_merge_with_trace_metadata(self):
        """Test that custom context merges with trace metadata."""
        custom_context = {
            "execution_attempt": 3,
            "input_size": 2048,
            "custom_field": "custom_value",
        }

        error = CogniVaultError(
            message="Context merge test",
            error_code="context_merge",
            context=custom_context,
            step_id="merge_step",
            agent_id="MergeAgent",
        )

        # Check custom context preserved
        assert error.context["execution_attempt"] == 3
        assert error.context["input_size"] == 2048
        assert error.context["custom_field"] == "custom_value"

        # Check trace metadata added
        assert error.context["step_id"] == "merge_step"
        assert error.context["agent_id"] == "MergeAgent"
        assert error.context["error_code"] == "context_merge"

    def test_timestamp_consistency(self):
        """Test that timestamp is consistent between error and context."""
        error = CogniVaultError(message="Timestamp test", error_code="timestamp_test")

        # Timestamp should be the same in both places
        assert error.timestamp.isoformat() == error.context["timestamp"]

    def test_error_chaining_preservation(self):
        """Test that error chaining is properly preserved."""
        original_error = ValueError("Original problem")

        wrapped_error = CogniVaultError(
            message="Wrapped error", error_code="wrapped_error", cause=original_error
        )

        assert wrapped_error.cause is original_error
        assert wrapped_error.__cause__ is None  # Python's built-in chaining not used

        # Test serialization preserves cause information
        data = wrapped_error.to_dict()
        assert "Original problem" in data["cause"]
