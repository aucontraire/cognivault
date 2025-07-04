"""
Configuration-related exceptions for CogniVault.

This module defines exceptions related to configuration validation,
environment setup, and application startup failures.
"""

from typing import Optional, Dict, Any, List
from . import CogniVaultError, ErrorSeverity, RetryPolicy


class ConfigurationError(CogniVaultError):
    """
    Base exception for configuration-related failures.

    Represents errors in application configuration, environment setup,
    or configuration validation failures.
    """

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        error_code: str = "config_error",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        retry_policy: RetryPolicy = RetryPolicy.NEVER,
        context: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = context or {}
        if config_section:
            context["config_section"] = config_section

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
        self.config_section = config_section


class ConfigValidationError(ConfigurationError):
    """
    Exception raised when configuration validation fails.

    Represents validation failures for configuration values,
    missing required settings, or invalid configuration formats.
    """

    def __init__(
        self,
        config_section: str,
        validation_errors: List[str],
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_count = len(validation_errors)
        message = (
            message
            or f"Configuration validation failed in '{config_section}' ({error_count} errors)"
        )

        context = context or {}
        context.update(
            {"validation_errors": validation_errors, "error_count": error_count}
        )

        super().__init__(
            message=message,
            config_section=config_section,
            error_code="config_validation_failed",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
        )
        self.validation_errors = validation_errors

    def get_user_message(self) -> str:
        """Get user-friendly error message with validation details."""
        base_msg = f"❌ Configuration validation failed in '{self.config_section}'\n"
        for i, error in enumerate(self.validation_errors[:3], 1):
            base_msg += f"  {i}. {error}\n"

        if len(self.validation_errors) > 3:
            remaining = len(self.validation_errors) - 3
            base_msg += f"  ... and {remaining} more errors\n"

        base_msg += "💡 Tip: Fix configuration errors and restart."
        return base_msg


class EnvironmentError(ConfigurationError):
    """
    Exception raised when environment setup is invalid.

    Represents issues with environment variables, file paths,
    or system environment configuration.
    """

    def __init__(
        self,
        environment_issue: str,
        required_vars: Optional[List[str]] = None,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = message or f"Environment setup error: {environment_issue}"

        context = context or {}
        context.update(
            {
                "environment_issue": environment_issue,
                "required_vars": required_vars or [],
            }
        )

        super().__init__(
            message=message,
            config_section="environment",
            error_code="environment_invalid",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
        )
        self.environment_issue = environment_issue
        self.required_vars = required_vars or []

    def get_user_message(self) -> str:
        """Get user-friendly error message with environment guidance."""
        base_msg = f"❌ Environment error: {self.environment_issue}\n"

        if self.required_vars:
            vars_str = ", ".join(self.required_vars)
            base_msg += f"💡 Tip: Set required environment variables: {vars_str}"
        else:
            base_msg += "💡 Tip: Check your .env file and environment setup."

        return base_msg


class APIKeyMissingError(ConfigurationError):
    """
    Exception raised when required API keys are missing.

    Represents missing or invalid API key configuration
    for external services like OpenAI, Anthropic, etc.
    """

    def __init__(
        self,
        service_name: str,
        api_key_var: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = message or f"API key missing for {service_name} (set {api_key_var})"

        context = context or {}
        context.update(
            {
                "service_name": service_name,
                "api_key_var": api_key_var,
                "security_sensitive": True,
            }
        )

        super().__init__(
            message=message,
            config_section="api_keys",
            error_code="api_key_missing",
            severity=ErrorSeverity.CRITICAL,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
        )
        self.service_name = service_name
        self.api_key_var = api_key_var

    def get_user_message(self) -> str:
        """Get user-friendly error message with API key guidance."""
        return (
            f"❌ API key missing for {self.service_name}\n"
            f"💡 Tip: Set {self.api_key_var} in your .env file."
        )


class ConfigFileError(ConfigurationError):
    """
    Exception raised when configuration files are missing or invalid.

    Represents issues with configuration file loading, parsing,
    or file system access for configuration files.
    """

    def __init__(
        self,
        config_file_path: str,
        file_issue: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            message or f"Configuration file error in '{config_file_path}': {file_issue}"
        )

        context = context or {}
        context.update({"config_file_path": config_file_path, "file_issue": file_issue})

        super().__init__(
            message=message,
            config_section="file_system",
            error_code="config_file_error",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.config_file_path = config_file_path
        self.file_issue = file_issue

    def get_user_message(self) -> str:
        """Get user-friendly error message with file guidance."""
        if "not found" in self.file_issue.lower():
            return (
                f"❌ Configuration file not found: {self.config_file_path}\n"
                f"💡 Tip: Create the configuration file or check the path."
            )
        elif "permission" in self.file_issue.lower():
            return (
                f"❌ Cannot access configuration file: {self.config_file_path}\n"
                f"💡 Tip: Check file permissions."
            )
        else:
            return (
                f"❌ Configuration file error: {self.file_issue}\n"
                f"💡 Tip: Check file format and syntax."
            )


class ModelConfigurationError(ConfigurationError):
    """
    Exception raised when LLM model configuration is invalid.

    Represents issues with model selection, model parameters,
    or model-specific configuration problems.
    """

    def __init__(
        self,
        model_name: str,
        config_issue: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = (
            message or f"Model configuration error for '{model_name}': {config_issue}"
        )

        context = context or {}
        context.update({"model_name": model_name, "config_issue": config_issue})

        super().__init__(
            message=message,
            config_section="model_config",
            error_code="model_config_invalid",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
        )
        self.model_name = model_name
        self.config_issue = config_issue

    def get_user_message(self) -> str:
        """Get user-friendly error message with model configuration guidance."""
        return (
            f"❌ Model configuration error for '{self.model_name}': {self.config_issue}\n"
            f"💡 Tip: Check model name and parameters in configuration."
        )
