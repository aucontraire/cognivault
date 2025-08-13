"""
Tests for configuration-specific exception classes.

This module tests configuration-related exceptions including invalid
configuration errors, missing configuration errors, and other
configuration validation failures.
"""

import pytest
from typing import Any, Dict
from cognivault.exceptions import (
    CogniVaultError,
    ErrorSeverity,
    RetryPolicy,
    ConfigurationError,
    ConfigValidationError,
    EnvironmentError,
    APIKeyMissingError,
    ConfigFileError,
    ModelConfigurationError,
)


class TestConfigurationError:
    """Test ConfigurationError functionality."""

    def test_configuration_error_creation(self) -> None:
        """Test basic ConfigurationError creation."""
        error = ConfigurationError(
            message="Missing required field 'api_key'",
            config_section="agent_config",
        )

        assert error.message == "Missing required field 'api_key'"
        assert error.config_section == "agent_config"
        assert error.error_code == "config_error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER  # Config issues need manual fix

    def test_configuration_error_with_all_params(self) -> None:
        """Test ConfigurationError with all parameters."""
        cause = ValueError("Invalid JSON format")
        context = {"config_file": "settings.yaml", "line_number": 25}

        error = ConfigurationError(
            message="Custom configuration failure",
            config_section="llm_config",
            error_code="custom_config_error",
            severity=ErrorSeverity.CRITICAL,
            retry_policy=RetryPolicy.BACKOFF,
            context=context,
            step_id="config_step",
            agent_id="ConfigAgent",
            cause=cause,
        )

        assert error.message == "Custom configuration failure"
        assert error.config_section == "llm_config"
        assert error.error_code == "custom_config_error"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.retry_policy == RetryPolicy.BACKOFF
        assert error.step_id == "config_step"
        assert error.agent_id == "ConfigAgent"
        assert error.cause == cause
        assert error.context["config_file"] == "settings.yaml"

    def test_configuration_context_injection(self) -> None:
        """Test that configuration information is added to context."""
        error = ConfigurationError(
            message="Connection string invalid",
            config_section="database_config",
            context={"section": "production", "attempt": 1},
        )

        assert error.context["config_section"] == "database_config"
        assert error.context["section"] == "production"

    def test_configuration_error_with_minimal_params(self) -> None:
        """Test ConfigurationError with minimal required parameters."""
        error = ConfigurationError(
            message="Basic validation failed",
        )

        assert error.message == "Basic validation failed"
        assert error.config_section is None

    def test_configuration_error_inheritance(self) -> None:
        """Test ConfigurationError inheritance hierarchy."""
        error = ConfigurationError(
            message="Test issue",
            config_section="test_config",
        )

        assert isinstance(error, CogniVaultError)
        assert isinstance(error, ConfigurationError)


class TestConfigValidationError:
    """Test ConfigValidationError functionality."""

    def test_validation_error_creation(self) -> None:
        """Test basic ConfigValidationError creation."""
        validation_errors = [
            "api_key is required",
            "model_name is invalid",
            "temperature out of range",
        ]

        error = ConfigValidationError(
            config_section="llm_validation",
            validation_errors=validation_errors,
        )

        assert error.config_section == "llm_validation"
        assert error.validation_errors == validation_errors
        assert error.error_code == "config_validation_failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER

        # Check default message construction
        expected_msg = "Configuration validation failed in 'llm_validation' (3 errors)"
        assert error.message == expected_msg

    def test_validation_error_with_all_params(self) -> None:
        """Test ConfigValidationError with all parameters."""
        validation_errors = ["timeout must be positive", "retries must be integer"]
        context = {"schema_version": "1.2", "validator": "jsonschema"}

        error = ConfigValidationError(
            config_section="agent_validation",
            validation_errors=validation_errors,
            message="Custom validation failure",
            step_id="validation_step",
            agent_id="ValidationAgent",
            context=context,
        )

        assert error.config_section == "agent_validation"
        assert error.validation_errors == validation_errors
        assert error.message == "Custom validation failure"
        assert error.step_id == "validation_step"
        assert error.agent_id == "ValidationAgent"
        assert error.context["schema_version"] == "1.2"

    def test_validation_context_injection(self) -> None:
        """Test that validation information is added to context."""
        validation_errors = ["field1 invalid", "field2 missing"]

        error = ConfigValidationError(
            config_section="validation_test",
            validation_errors=validation_errors,
            context={"validator_name": "custom_validator"},
        )

        assert error.context["validation_errors"] == validation_errors
        assert error.context["error_count"] == 2
        assert error.context["validator_name"] == "custom_validator"

    def test_validation_error_with_single_error(self) -> None:
        """Test validation error with single validation error."""
        error = ConfigValidationError(
            config_section="single_error_test",
            validation_errors=["temperature must be between 0.0 and 2.0"],
        )

        expected_msg = (
            "Configuration validation failed in 'single_error_test' (1 errors)"
        )
        assert error.message == expected_msg
        assert error.context["error_count"] == 1

    def test_validation_error_with_empty_errors(self) -> None:
        """Test validation error with empty validation errors list."""
        error = ConfigValidationError(
            config_section="empty_errors_test",
            validation_errors=[],
            message="General schema validation failed",
        )

        assert error.validation_errors == []
        assert error.context["error_count"] == 0
        assert error.message == "General schema validation failed"

    def test_validation_error_user_message(self) -> None:
        """Test user-friendly message for validation errors."""
        validation_errors = [
            "api_endpoint format invalid",
            "timeout too large",
            "retries negative",
        ]

        error = ConfigValidationError(
            config_section="user_validation_test",
            validation_errors=validation_errors,
        )

        user_msg = error.get_user_message()
        assert (
            "❌ Configuration validation failed in 'user_validation_test'" in user_msg
        )
        assert "1. api_endpoint format invalid" in user_msg
        assert "2. timeout too large" in user_msg
        assert "3. retries negative" in user_msg
        assert "💡 Tip: Fix configuration errors and restart." in user_msg

    def test_validation_error_user_message_truncation(self) -> None:
        """Test user message truncation for many validation errors."""
        validation_errors = [f"error_{i}" for i in range(5)]

        error = ConfigValidationError(
            config_section="truncation_test",
            validation_errors=validation_errors,
        )

        user_msg = error.get_user_message()
        assert "1. error_0" in user_msg
        assert "2. error_1" in user_msg
        assert "3. error_2" in user_msg
        assert "... and 2 more errors" in user_msg

    def test_validation_error_inheritance(self) -> None:
        """Test ConfigValidationError inheritance hierarchy."""
        error = ConfigValidationError(
            config_section="inheritance_test",
            validation_errors=["test_error"],
        )

        assert isinstance(error, ConfigurationError)
        assert isinstance(error, CogniVaultError)
        assert isinstance(error, ConfigValidationError)


class TestConfigurationErrorInheritance:
    """Test proper inheritance hierarchy for configuration errors."""

    def test_validation_error_inherits_from_configuration_error(self) -> None:
        """Test that ConfigValidationError inherits from ConfigurationError."""
        validation_error = ConfigValidationError("test", ["field"])

        assert isinstance(validation_error, ConfigurationError)
        assert isinstance(validation_error, CogniVaultError)

    def test_configuration_error_inherits_from_base(self) -> None:
        """Test that ConfigurationError inherits from CogniVaultError."""
        config_error = ConfigurationError("test issue")
        assert isinstance(config_error, CogniVaultError)

    def test_polymorphic_behavior(self) -> None:
        """Test polymorphic behavior of configuration errors."""

        def handle_config_error(error: ConfigurationError) -> Dict[str, Any]:
            return {
                "config_section": error.config_section,
                "retryable": error.is_retryable(),
                "severity": error.severity.value,
                "type": error.__class__.__name__,
                "has_validation_errors": hasattr(error, "validation_errors"),
            }

        errors = [
            ConfigurationError("basic issue"),
            ConfigValidationError("validation", ["field1"]),
        ]

        results = [handle_config_error(err) for err in errors]

        assert len(results) == 2
        assert results[0]["type"] == "ConfigurationError"
        assert results[1]["type"] == "ConfigValidationError"

        # Neither should be retryable (NEVER policy)
        assert results[0]["retryable"] is False
        assert results[1]["retryable"] is False

        # Only validation error should have validation_errors
        assert results[0]["has_validation_errors"] is False
        assert results[1]["has_validation_errors"] is True


class TestConfigurationErrorIntegration:
    """Test integration aspects of configuration errors."""

    def test_config_error_with_step_metadata(self) -> None:
        """Test configuration errors work properly with step metadata."""
        error = ConfigurationError(
            message="Invalid integration settings",
            config_section="integration_config",
            step_id="integration_step_789",
            agent_id="IntegrationAgent",
            context={
                "validation_stage": "pre_execution",
                "source": "user_provided",
            },
        )

        # Verify all metadata is properly integrated
        assert error.step_id == "integration_step_789"
        assert error.agent_id == "IntegrationAgent"
        assert error.config_section == "integration_config"
        assert error.context["validation_stage"] == "pre_execution"

        # Verify serialization includes everything
        data = error.to_dict()
        assert data["step_id"] == "integration_step_789"
        assert data["agent_id"] == "IntegrationAgent"
        assert "validation_stage" in data["context"]

    def test_config_error_chaining_scenarios(self) -> None:
        """Test various configuration error chaining scenarios."""
        # Original parsing error
        original = ValueError("Invalid YAML syntax")

        # Configuration error wrapping parsing error
        config_error = ConfigurationError(
            message="Failed to parse configuration file",
            config_section="yaml_config",
            step_id="config_step",
            cause=original,
        )

        # Validation error wrapping configuration error
        ConfigValidationError(
            config_section="schema_validation",
            validation_errors=["Could not validate due to parsing failure"],
            step_id="validation_step",
        )

        # Verify chaining
        assert config_error.cause == original

        # Verify serialization handles nested causes
        config_data = config_error.to_dict()
        assert "Invalid YAML syntax" in config_data["cause"]

    def test_exception_raising_and_catching(self) -> None:
        """Test that configuration errors can be properly raised and caught."""
        # Test specific exception catching
        with pytest.raises(ConfigValidationError) as exc_info:
            raise ConfigValidationError("test_type", ["test_field"])

        assert exc_info.value.config_section == "test_type"
        assert exc_info.value.validation_errors == ["test_field"]

        # Test catching as base type
        with pytest.raises(ConfigurationError) as config_exc_info:
            raise ConfigValidationError("validation", ["field"])

        assert hasattr(config_exc_info.value, "validation_errors")

        # Test catching as CogniVaultError
        with pytest.raises(CogniVaultError) as base_exc_info:
            raise ConfigurationError("config issue")

        assert base_exc_info.value.error_code == "config_error"

    def test_config_error_retry_semantics(self) -> None:
        """Test retry semantics for configuration errors."""
        # Configuration errors should not be retryable
        config_error = ConfigurationError("test issue")
        assert not config_error.is_retryable()
        assert config_error.retry_policy == RetryPolicy.NEVER

        # Validation errors should also not be retryable
        validation_error = ConfigValidationError("test", ["field"])
        assert not validation_error.is_retryable()
        assert validation_error.retry_policy == RetryPolicy.NEVER

        # Neither should use circuit breaker
        assert not config_error.should_use_circuit_breaker()
        assert not validation_error.should_use_circuit_breaker()

    def test_config_error_scenarios(self) -> None:
        """Test various configuration error scenarios."""
        scenarios = [
            {
                "message": "Invalid API key format",
                "config_section": "llm_provider_config",
            },
            {
                "message": "Circular dependency detected",
                "config_section": "agent_workflow_config",
            },
            {
                "message": "Unsupported export format",
                "config_section": "output_format_config",
            },
        ]

        for scenario in scenarios:
            error = ConfigurationError(
                message=scenario["message"], config_section=scenario["config_section"]
            )

            # Verify all scenarios create valid errors
            assert isinstance(error, ConfigurationError)
            assert error.retry_policy == RetryPolicy.NEVER
            assert error.severity == ErrorSeverity.HIGH

    def test_validation_error_scenarios(self) -> None:
        """Test validation errors with various scenarios."""
        error_scenarios = [
            (["single_field_error"], "Single field validation"),
            (["field1_error", "field2_error"], "Multiple field validation"),
            ([], "Schema-level validation"),
            ([f"field_{i}_error" for i in range(10)], "Many field validation"),
        ]

        for errors, description in error_scenarios:
            error = ConfigValidationError(
                config_section="field_test",
                validation_errors=errors,
            )

            assert error.validation_errors == errors
            assert error.context["error_count"] == len(errors)

    def test_config_error_user_message_variations(self) -> None:
        """Test user messages for various configuration error scenarios."""
        # Test basic configuration error (uses default user message from base class)
        config_error = ConfigurationError("Missing API key")
        user_msg = config_error.get_user_message()
        assert "❌" in user_msg
        assert "Missing API key" in user_msg

        # Test validation error with custom user message
        validation_error = ConfigValidationError(
            "schema", ["timeout invalid", "retries negative"]
        )
        user_msg = validation_error.get_user_message()
        assert "❌ Configuration validation failed in 'schema'" in user_msg
        assert "1. timeout invalid" in user_msg
        assert "2. retries negative" in user_msg
        assert "💡 Tip: Fix configuration errors and restart." in user_msg


class TestEnvironmentError:
    """Test EnvironmentError functionality."""

    def test_environment_error_creation(self) -> None:
        """Test basic EnvironmentError creation."""
        error = EnvironmentError(
            environment_issue="Missing required environment variables",
            required_vars=["OPENAI_API_KEY", "DATABASE_URL"],
        )

        assert error.environment_issue == "Missing required environment variables"
        assert error.required_vars == ["OPENAI_API_KEY", "DATABASE_URL"]
        assert error.config_section == "environment"
        assert error.error_code == "environment_invalid"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER

    def test_environment_error_with_custom_message(self) -> None:
        """Test EnvironmentError with custom message."""
        error = EnvironmentError(
            environment_issue="Invalid path configuration",
            message="Custom environment error message",
            step_id="env_step",
            agent_id="EnvAgent",
        )

        assert error.message == "Custom environment error message"
        assert error.step_id == "env_step"
        assert error.agent_id == "EnvAgent"

    def test_environment_error_without_required_vars(self) -> None:
        """Test EnvironmentError without required vars."""
        error = EnvironmentError(environment_issue="PATH variable not set correctly")

        assert error.required_vars == []
        assert (
            "Environment setup error: PATH variable not set correctly" in error.message
        )

    def test_environment_error_user_message_with_vars(self) -> None:
        """Test user message with required variables."""
        error = EnvironmentError(
            environment_issue="API keys missing",
            required_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        )

        user_msg = error.get_user_message()
        assert "❌ Environment error: API keys missing" in user_msg
        assert (
            "💡 Tip: Set required environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY"
            in user_msg
        )

    def test_environment_error_user_message_without_vars(self) -> None:
        """Test user message without specific variables."""
        error = EnvironmentError(environment_issue="General environment issue")

        user_msg = error.get_user_message()
        assert "❌ Environment error: General environment issue" in user_msg
        assert "💡 Tip: Check your .env file and environment setup." in user_msg


class TestAPIKeyMissingError:
    """Test APIKeyMissingError functionality."""

    def test_api_key_missing_error_creation(self) -> None:
        """Test basic APIKeyMissingError creation."""
        error = APIKeyMissingError(
            service_name="OpenAI",
            api_key_var="OPENAI_API_KEY",
        )

        assert error.service_name == "OpenAI"
        assert error.api_key_var == "OPENAI_API_KEY"
        assert error.config_section == "api_keys"
        assert error.error_code == "api_key_missing"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.retry_policy == RetryPolicy.NEVER
        assert error.context["security_sensitive"] is True

    def test_api_key_missing_error_with_custom_message(self) -> None:
        """Test APIKeyMissingError with custom message."""
        error = APIKeyMissingError(
            service_name="Anthropic",
            api_key_var="ANTHROPIC_API_KEY",
            message="Custom API key error",
            step_id="api_step",
            agent_id="APIAgent",
        )

        assert error.message == "Custom API key error"
        assert error.step_id == "api_step"
        assert error.agent_id == "APIAgent"

    def test_api_key_missing_error_user_message(self) -> None:
        """Test user-friendly message for API key errors."""
        error = APIKeyMissingError(
            service_name="OpenAI",
            api_key_var="OPENAI_API_KEY",
        )

        user_msg = error.get_user_message()
        assert user_msg == (
            "❌ API key missing for OpenAI\n"
            "💡 Tip: Set OPENAI_API_KEY in your .env file."
        )


class TestConfigFileError:
    """Test ConfigFileError functionality."""

    def test_config_file_error_creation(self) -> None:
        """Test basic ConfigFileError creation."""
        error = ConfigFileError(
            config_file_path="/path/to/config.yaml",
            file_issue="File not found",
        )

        assert error.config_file_path == "/path/to/config.yaml"
        assert error.file_issue == "File not found"
        assert error.config_section == "file_system"
        assert error.error_code == "config_file_error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER

    def test_config_file_error_with_cause(self) -> None:
        """Test ConfigFileError with cause exception."""
        cause = FileNotFoundError("No such file")
        error = ConfigFileError(
            config_file_path="config.json",
            file_issue="File access failed",
            message="Custom file error",
            step_id="file_step",
            agent_id="FileAgent",
            cause=cause,
        )

        assert error.message == "Custom file error"
        assert error.cause == cause
        assert error.step_id == "file_step"
        assert error.agent_id == "FileAgent"

    def test_config_file_error_user_message_not_found(self) -> None:
        """Test user message for file not found."""
        error = ConfigFileError(
            config_file_path="missing.yaml",
            file_issue="File not found",
        )

        user_msg = error.get_user_message()
        assert "❌ Configuration file not found: missing.yaml" in user_msg
        assert "💡 Tip: Create the configuration file or check the path." in user_msg

    def test_config_file_error_user_message_permission(self) -> None:
        """Test user message for permission issues."""
        error = ConfigFileError(
            config_file_path="readonly.yaml",
            file_issue="Permission denied",
        )

        user_msg = error.get_user_message()
        assert "❌ Cannot access configuration file: readonly.yaml" in user_msg
        assert "💡 Tip: Check file permissions." in user_msg

    def test_config_file_error_user_message_generic(self) -> None:
        """Test user message for generic file issues."""
        error = ConfigFileError(
            config_file_path="corrupt.yaml",
            file_issue="Invalid YAML syntax",
        )

        user_msg = error.get_user_message()
        assert "❌ Configuration file error: Invalid YAML syntax" in user_msg
        assert "💡 Tip: Check file format and syntax." in user_msg


class TestModelConfigurationError:
    """Test ModelConfigurationError functionality."""

    def test_model_configuration_error_creation(self) -> None:
        """Test basic ModelConfigurationError creation."""
        error = ModelConfigurationError(
            model_name="gpt-4",
            config_issue="Invalid temperature setting",
        )

        assert error.model_name == "gpt-4"
        assert error.config_issue == "Invalid temperature setting"
        assert error.config_section == "model_config"
        assert error.error_code == "model_config_invalid"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER

    def test_model_configuration_error_with_custom_message(self) -> None:
        """Test ModelConfigurationError with custom message."""
        error = ModelConfigurationError(
            model_name="claude-3",
            config_issue="Unsupported model version",
            message="Custom model error",
            step_id="model_step",
            agent_id="ModelAgent",
        )

        assert error.message == "Custom model error"
        assert error.step_id == "model_step"
        assert error.agent_id == "ModelAgent"

    def test_model_configuration_error_user_message(self) -> None:
        """Test user-friendly message for model configuration errors."""
        error = ModelConfigurationError(
            model_name="gpt-3.5-turbo",
            config_issue="Token limit exceeded",
        )

        user_msg = error.get_user_message()
        assert (
            "❌ Model configuration error for 'gpt-3.5-turbo': Token limit exceeded"
            in user_msg
        )
        assert "💡 Tip: Check model name and parameters in configuration." in user_msg
