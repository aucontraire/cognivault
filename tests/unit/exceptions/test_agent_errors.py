"""
Tests for agent-specific exception classes.

This module tests all agent-related exceptions including execution errors,
timeout errors, configuration errors, and their integration with the
retry and circuit breaker systems.
"""

import pytest
from typing import Any, Dict
from cognivault.exceptions import (
    CogniVaultError,
    ErrorSeverity,
    RetryPolicy,
    AgentExecutionError,
    AgentDependencyMissingError,
    AgentTimeoutError,
    AgentConfigurationError,
    AgentResourceError,
    AgentValidationError,
)


class TestAgentExecutionError:
    """Test AgentExecutionError base functionality."""

    def test_agent_execution_error_creation(self) -> None:
        """Test basic AgentExecutionError creation."""
        error = AgentExecutionError(
            message="Agent failed to execute", agent_name="TestAgent"
        )

        assert error.message == "Agent failed to execute"
        assert error.agent_name == "TestAgent"
        assert error.agent_id == "TestAgent"  # Should be set from agent_name
        assert error.error_code == "agent_execution_failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.BACKOFF
        assert isinstance(error, CogniVaultError)

    def test_agent_execution_error_with_all_params(self) -> None:
        """Test AgentExecutionError with all parameters."""
        cause = RuntimeError("Underlying error")
        context = {"attempt": 2, "input_data": "test"}

        error = AgentExecutionError(
            message="Detailed agent failure",
            agent_name="DetailedAgent",
            error_code="detailed_failure",
            severity=ErrorSeverity.CRITICAL,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id="detailed_step",
            cause=cause,
        )

        assert error.message == "Detailed agent failure"
        assert error.agent_name == "DetailedAgent"
        assert error.error_code == "detailed_failure"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.retry_policy == RetryPolicy.NEVER
        assert error.step_id == "detailed_step"
        assert error.cause == cause
        assert error.context["attempt"] == 2
        assert error.context["agent_name"] == "DetailedAgent"

    def test_agent_context_injection(self) -> None:
        """Test that agent_name is automatically added to context."""
        error = AgentExecutionError(
            message="Context injection test", agent_name="ContextAgent"
        )

        assert "agent_name" in error.context
        assert error.context["agent_name"] == "ContextAgent"


class TestAgentDependencyMissingError:
    """Test AgentDependencyMissingError functionality."""

    def test_dependency_missing_error_creation(self) -> None:
        """Test basic dependency missing error creation."""
        missing_deps = ["RefinerAgent", "HistorianAgent"]

        error = AgentDependencyMissingError(
            agent_name="CriticAgent", missing_dependencies=missing_deps
        )

        assert error.agent_name == "CriticAgent"
        assert error.missing_dependencies == missing_deps
        assert error.error_code == "agent_dependency_missing"
        assert (
            error.retry_policy == RetryPolicy.NEVER
        )  # Can't retry until deps satisfied
        assert error.severity == ErrorSeverity.HIGH

        # Check default message construction
        expected_msg = (
            "Agent 'CriticAgent' requires output from: RefinerAgent, HistorianAgent"
        )
        assert error.message == expected_msg

    def test_dependency_missing_with_custom_message(self) -> None:
        """Test dependency error with custom message."""
        missing_deps = ["SingleDep"]
        custom_message = "Custom dependency failure message"

        error = AgentDependencyMissingError(
            agent_name="CustomAgent",
            missing_dependencies=missing_deps,
            message=custom_message,
            step_id="dep_step",
        )

        assert error.message == custom_message
        assert error.missing_dependencies == missing_deps
        assert error.step_id == "dep_step"

    def test_dependency_context_data(self) -> None:
        """Test that dependency information is added to context."""
        missing_deps = ["Dep1", "Dep2", "Dep3"]

        error = AgentDependencyMissingError(
            agent_name="DependentAgent", missing_dependencies=missing_deps
        )

        assert error.context["missing_dependencies"] == missing_deps
        assert error.context["dependency_count"] == 3
        assert error.context["agent_name"] == "DependentAgent"

    def test_dependency_user_message(self) -> None:
        """Test user-friendly message for dependency errors."""
        missing_deps = ["RequiredAgent1", "RequiredAgent2"]

        error = AgentDependencyMissingError(
            agent_name="NeedsAgent", missing_dependencies=missing_deps
        )

        user_msg = error.get_user_message()
        assert "💡 Missing dependencies: RequiredAgent1, RequiredAgent2" in user_msg


class TestAgentTimeoutError:
    """Test AgentTimeoutError functionality."""

    def test_timeout_error_creation(self) -> None:
        """Test basic timeout error creation."""
        error = AgentTimeoutError(agent_name="SlowAgent", timeout_seconds=30.0)

        assert error.agent_name == "SlowAgent"
        assert error.timeout_seconds == 30.0
        assert error.error_code == "agent_timeout"
        assert error.retry_policy == RetryPolicy.BACKOFF  # Timeouts might be temporary
        assert error.severity == ErrorSeverity.MEDIUM

        # Check default message
        expected_msg = "Agent 'SlowAgent' timed out after 30.0s"
        assert error.message == expected_msg

    def test_timeout_error_with_cause(self) -> None:
        """Test timeout error with cause parameter."""
        original_timeout = TimeoutError("Asyncio timeout")

        error = AgentTimeoutError(
            agent_name="CausedTimeoutAgent",
            timeout_seconds=45.0,
            step_id="timeout_step",
            cause=original_timeout,
        )

        assert error.agent_name == "CausedTimeoutAgent"
        assert error.timeout_seconds == 45.0
        assert error.step_id == "timeout_step"
        assert error.cause == original_timeout

    def test_timeout_error_with_custom_message(self) -> None:
        """Test timeout error with custom message."""
        custom_message = "Custom timeout message"

        error = AgentTimeoutError(
            agent_name="CustomTimeoutAgent",
            timeout_seconds=60.0,
            message=custom_message,
        )

        assert error.message == custom_message
        assert error.timeout_seconds == 60.0

    def test_timeout_context_data(self) -> None:
        """Test that timeout information is added to context."""
        error = AgentTimeoutError(
            agent_name="TimeoutAgent",
            timeout_seconds=25.5,
            context={"attempt": 3, "max_retries": 5},
        )

        assert error.context["timeout_seconds"] == 25.5
        assert error.context["timeout_type"] == "agent_execution"
        assert error.context["attempt"] == 3
        assert error.context["agent_name"] == "TimeoutAgent"

    def test_timeout_user_message(self) -> None:
        """Test user-friendly message for timeout errors."""
        error = AgentTimeoutError(agent_name="UserTimeoutAgent", timeout_seconds=20.0)

        user_msg = error.get_user_message()
        assert (
            "💡 Tip: Consider increasing timeout or simplifying the query." in user_msg
        )


class TestAgentConfigurationError:
    """Test AgentConfigurationError functionality."""

    def test_config_error_creation(self) -> None:
        """Test basic configuration error creation."""
        error = AgentConfigurationError(
            agent_name="ConfigAgent",
            config_issue="Missing required parameter 'api_key'",
        )

        assert error.agent_name == "ConfigAgent"
        assert error.config_issue == "Missing required parameter 'api_key'"
        assert error.error_code == "agent_config_invalid"
        assert error.retry_policy == RetryPolicy.NEVER  # Config issues need manual fix
        assert error.severity == ErrorSeverity.HIGH

        # Check default message
        expected_msg = "Agent 'ConfigAgent' configuration error: Missing required parameter 'api_key'"
        assert error.message == expected_msg

    def test_config_error_with_custom_message(self) -> None:
        """Test configuration error with custom message."""
        custom_message = "Custom configuration failure"

        error = AgentConfigurationError(
            agent_name="CustomConfigAgent",
            config_issue="Invalid format",
            message=custom_message,
            step_id="config_step",
        )

        assert error.message == custom_message
        assert error.config_issue == "Invalid format"
        assert error.step_id == "config_step"

    def test_config_context_data(self) -> None:
        """Test that configuration information is added to context."""
        error = AgentConfigurationError(
            agent_name="ContextConfigAgent",
            config_issue="Validation failed",
            context={"config_file": "agent.yaml", "line": 42},
        )

        assert error.context["config_issue"] == "Validation failed"
        assert error.context["config_type"] == "agent_configuration"
        assert error.context["config_file"] == "agent.yaml"
        assert error.context["agent_name"] == "ContextConfigAgent"

    def test_config_user_message(self) -> None:
        """Test user-friendly message for configuration errors."""
        error = AgentConfigurationError(
            agent_name="UserConfigAgent", config_issue="Invalid JSON format"
        )

        user_msg = error.get_user_message()
        assert "💡 Tip: Check agent configuration for: Invalid JSON format" in user_msg


class TestAgentErrorInheritance:
    """Test proper inheritance hierarchy for agent errors."""

    def test_all_agent_errors_inherit_from_agent_execution_error(self) -> None:
        """Test that specialized agent errors inherit from AgentExecutionError."""
        dependency_error = AgentDependencyMissingError("Agent", ["Dep"])
        timeout_error = AgentTimeoutError("Agent", 30.0)
        config_error = AgentConfigurationError("Agent", "Issue")

        # All should inherit from AgentExecutionError
        assert isinstance(dependency_error, AgentExecutionError)
        assert isinstance(timeout_error, AgentExecutionError)
        assert isinstance(config_error, AgentExecutionError)

        # All should inherit from CogniVaultError
        assert isinstance(dependency_error, CogniVaultError)
        assert isinstance(timeout_error, CogniVaultError)
        assert isinstance(config_error, CogniVaultError)

    def test_agent_execution_error_inherits_from_base(self) -> None:
        """Test that AgentExecutionError inherits from CogniVaultError."""
        error = AgentExecutionError("Test", "Agent")
        assert isinstance(error, CogniVaultError)

    def test_polymorphic_behavior(self) -> None:
        """Test polymorphic behavior of agent errors."""

        def handle_agent_error(error: AgentExecutionError) -> Dict[str, Any]:
            return {
                "agent": error.agent_name,
                "retryable": error.is_retryable(),
                "severity": error.severity.value,
                "type": error.__class__.__name__,
            }

        errors = [
            AgentExecutionError("Base error", "BaseAgent"),
            AgentTimeoutError("TimeoutAgent", 30.0),
            AgentConfigurationError("ConfigAgent", "Issue"),
        ]

        results = [handle_agent_error(err) for err in errors]

        assert len(results) == 3
        assert results[0]["type"] == "AgentExecutionError"
        assert results[1]["type"] == "AgentTimeoutError"
        assert results[2]["type"] == "AgentConfigurationError"

        # Check specific retry behaviors
        assert results[0]["retryable"] is True  # BACKOFF
        assert results[1]["retryable"] is True  # BACKOFF
        assert results[2]["retryable"] is False  # NEVER


class TestAgentErrorIntegration:
    """Test integration aspects of agent errors."""

    def test_error_context_with_step_metadata(self) -> None:
        """Test that agent errors work properly with step metadata."""
        error = AgentExecutionError(
            message="Integration test",
            agent_name="IntegrationAgent",
            step_id="integration_step_789",
            context={"execution_attempt": 2, "max_attempts": 3, "input_hash": "abc123"},
        )

        # Verify all metadata is properly integrated
        assert error.context["step_id"] == "integration_step_789"
        assert error.context["agent_id"] == "IntegrationAgent"
        assert error.context["agent_name"] == "IntegrationAgent"
        assert error.context["execution_attempt"] == 2
        assert error.context["input_hash"] == "abc123"

        # Verify serialization includes everything
        data = error.to_dict()
        assert data["step_id"] == "integration_step_789"
        assert data["agent_id"] == "IntegrationAgent"
        assert "execution_attempt" in data["context"]

    def test_error_chaining_scenarios(self) -> None:
        """Test various error chaining scenarios with agent errors."""
        # Original error
        original = ValueError("Invalid input format")

        # Agent error wrapping original
        agent_error = AgentExecutionError(
            message="Failed to process input",
            agent_name="ProcessorAgent",
            step_id="proc_step",
            cause=original,
        )

        # Timeout error wrapping agent error
        timeout_error = AgentTimeoutError(
            agent_name="WrapperAgent",
            timeout_seconds=30.0,
            step_id="wrapper_step",
            cause=agent_error,
        )

        # Verify chaining
        assert timeout_error.cause == agent_error
        assert agent_error.cause == original

        # Verify serialization handles nested causes
        timeout_data = timeout_error.to_dict()
        agent_data = agent_error.to_dict()

        assert "Failed to process input" in timeout_data["cause"]
        assert "Invalid input format" in agent_data["cause"]

    def test_exception_raising_and_catching(self) -> None:
        """Test that agent errors can be properly raised and caught."""
        # Test specific exception catching
        with pytest.raises(AgentTimeoutError) as exc_info:
            raise AgentTimeoutError("TestAgent", 30.0)

        assert exc_info.value.agent_name == "TestAgent"
        assert exc_info.value.timeout_seconds == 30.0

        # Test catching as base type
        with pytest.raises(AgentExecutionError) as agent_exc_info:
            raise AgentConfigurationError("ConfigAgent", "Missing param")

        assert agent_exc_info.value.agent_name == "ConfigAgent"

        # Test catching as CogniVaultError
        with pytest.raises(CogniVaultError) as base_exc_info:
            raise AgentDependencyMissingError("DepAgent", ["Dep1"])

        assert base_exc_info.value.agent_id == "DepAgent"


class TestAgentResourceError:
    """Test AgentResourceError functionality."""

    def test_resource_error_creation(self) -> None:
        """Test basic AgentResourceError creation."""
        error = AgentResourceError(
            agent_name="TestAgent",
            resource_type="disk_space",
            resource_details="insufficient space for temp files",
        )

        assert error.agent_name == "TestAgent"
        assert error.resource_type == "disk_space"
        assert error.resource_details == "insufficient space for temp files"
        assert error.error_code == "agent_resource_unavailable"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retry_policy == RetryPolicy.NEVER  # disk_space -> NEVER

    def test_resource_error_with_network_type(self) -> None:
        """Test AgentResourceError with network resource type."""
        error = AgentResourceError(
            agent_name="NetworkAgent",
            resource_type="network",
            resource_details="connection timeout",
        )

        assert error.retry_policy == RetryPolicy.BACKOFF  # network -> BACKOFF

    def test_resource_error_with_api_type(self) -> None:
        """Test AgentResourceError with api resource type."""
        error = AgentResourceError(
            agent_name="APIAgent",
            resource_type="api",
            resource_details="rate limit exceeded",
        )

        assert error.retry_policy == RetryPolicy.BACKOFF  # api -> BACKOFF

    def test_resource_error_with_custom_message(self) -> None:
        """Test AgentResourceError with custom message."""
        error = AgentResourceError(
            agent_name="CustomAgent",
            resource_type="database",
            resource_details="connection pool exhausted",
            message="Custom resource error",
            step_id="resource_step",
        )

        assert error.message == "Custom resource error"
        assert error.step_id == "resource_step"
        assert error.retry_policy == RetryPolicy.NEVER  # not network/api -> NEVER

    def test_resource_error_context_injection(self) -> None:
        """Test that resource information is added to context."""
        error = AgentResourceError(
            agent_name="ContextAgent",
            resource_type="memory",
            resource_details="heap overflow",
            context={"heap_size": "2GB"},
        )

        assert error.context["resource_type"] == "memory"
        assert error.context["resource_details"] == "heap overflow"
        assert error.context["heap_size"] == "2GB"

    def test_resource_error_user_message_disk_space(self) -> None:
        """Test user message for disk space resource errors."""
        error = AgentResourceError(
            agent_name="DiskAgent",
            resource_type="disk_space",
            resource_details="no space left",
        )

        user_msg = error.get_user_message()
        assert "💡 Tip: Free up disk space and try again." in user_msg

    def test_resource_error_user_message_network(self) -> None:
        """Test user message for network resource errors."""
        error = AgentResourceError(
            agent_name="NetAgent",
            resource_type="network",
            resource_details="connection failed",
        )

        user_msg = error.get_user_message()
        assert "💡 Tip: Check your internet connection." in user_msg

    def test_resource_error_user_message_generic(self) -> None:
        """Test user message for generic resource errors."""
        error = AgentResourceError(
            agent_name="GenericAgent",
            resource_type="cpu",
            resource_details="high load",
        )

        user_msg = error.get_user_message()
        assert "💡 Tip: Check cpu availability." in user_msg


class TestAgentValidationError:
    """Test AgentValidationError functionality."""

    def test_validation_error_creation(self) -> None:
        """Test basic AgentValidationError creation."""
        error = AgentValidationError(
            agent_name="ValidatorAgent",
            validation_type="input_schema",
            validation_details="missing required field 'query'",
        )

        assert error.agent_name == "ValidatorAgent"
        assert error.validation_type == "input_schema"
        assert error.validation_details == "missing required field 'query'"
        assert error.error_code == "agent_validation_failed"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retry_policy == RetryPolicy.NEVER

    def test_validation_error_with_custom_message(self) -> None:
        """Test AgentValidationError with custom message."""
        error = AgentValidationError(
            agent_name="CustomValidator",
            validation_type="output_format",
            validation_details="invalid JSON structure",
            message="Custom validation error",
            step_id="validation_step",
        )

        assert error.message == "Custom validation error"
        assert error.step_id == "validation_step"

    def test_validation_error_context_injection(self) -> None:
        """Test that validation information is added to context."""
        error = AgentValidationError(
            agent_name="ContextValidator",
            validation_type="parameter_range",
            validation_details="temperature must be between 0.0 and 2.0",
            context={"parameter": "temperature", "value": 3.5},
        )

        assert error.context["validation_type"] == "parameter_range"
        assert (
            error.context["validation_details"]
            == "temperature must be between 0.0 and 2.0"
        )
        assert error.context["parameter"] == "temperature"
        assert error.context["value"] == 3.5

    def test_validation_error_user_message(self) -> None:
        """Test user-friendly message for validation errors."""
        error = AgentValidationError(
            agent_name="MessageValidator",
            validation_type="email_format",
            validation_details="invalid email address",
        )

        user_msg = error.get_user_message()
        assert "💡 Tip: Check email_format format and try again." in user_msg
