"""
Agent-specific exceptions for CogniVault.

This module defines exceptions related to agent execution, dependency management,
and agent-level failures with LangGraph node compatibility.
"""

from typing import Optional, Dict, Any, List
from . import CogniVaultError, ErrorSeverity, RetryPolicy


class AgentExecutionError(CogniVaultError):
    """
    Base exception for agent execution failures.

    Represents errors that occur during agent execution, designed to be
    compatible with LangGraph node error handling patterns.
    """

    def __init__(
        self,
        message: str,
        agent_name: str,
        error_code: str = "agent_execution_failed",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        retry_policy: RetryPolicy = RetryPolicy.BACKOFF,
        context: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        context = context or {}
        context["agent_name"] = agent_name

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            retry_policy=retry_policy,
            context=context,
            step_id=step_id,
            agent_id=agent_name,
            cause=cause,
        )
        self.agent_name = agent_name


class AgentDependencyMissingError(AgentExecutionError):
    """
    Exception raised when an agent's dependencies are not satisfied.

    This error occurs when an agent requires output from another agent
    that either failed or was not executed. Maps to LangGraph conditional
    edge logic for handling missing upstream nodes.
    """

    def __init__(
        self,
        agent_name: str,
        missing_dependencies: List[str],
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        deps_str = ", ".join(missing_dependencies)
        message = message or f"Agent '{agent_name}' requires output from: {deps_str}"

        context = context or {}
        context.update(
            {
                "missing_dependencies": missing_dependencies,
                "dependency_count": len(missing_dependencies),
            }
        )

        super().__init__(
            message=message,
            agent_name=agent_name,
            error_code="agent_dependency_missing",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # Can't retry until dependencies are satisfied
            context=context,
            step_id=step_id,
        )
        self.missing_dependencies = missing_dependencies

    def get_user_message(self) -> str:
        """Get user-friendly error message with dependency guidance."""
        base_message = super().get_user_message()
        deps_str = ", ".join(self.missing_dependencies)
        return f"{base_message}\n💡 Missing dependencies: {deps_str}"


class AgentTimeoutError(AgentExecutionError):
    """
    Exception raised when an agent execution times out.

    Represents timeout failures during agent execution, with configurable
    retry policies for different timeout scenarios.
    """

    def __init__(
        self,
        agent_name: str,
        timeout_seconds: float,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        message = message or f"Agent '{agent_name}' timed out after {timeout_seconds}s"

        context = context or {}
        context.update(
            {"timeout_seconds": timeout_seconds, "timeout_type": "agent_execution"}
        )

        super().__init__(
            message=message,
            agent_name=agent_name,
            error_code="agent_timeout",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=RetryPolicy.BACKOFF,  # Timeout might be temporary
            context=context,
            step_id=step_id,
            cause=cause,
        )
        self.timeout_seconds = timeout_seconds

    def get_user_message(self) -> str:
        """Get user-friendly error message with timeout guidance."""
        base_message = super().get_user_message()
        return f"{base_message}\n💡 Tip: Consider increasing timeout or simplifying the query."


class AgentConfigurationError(AgentExecutionError):
    """
    Exception raised when an agent has invalid configuration.

    Represents configuration-related failures that prevent agent execution,
    such as missing required parameters or invalid settings.
    """

    def __init__(
        self,
        agent_name: str,
        config_issue: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = message or f"Agent '{agent_name}' configuration error: {config_issue}"

        context = context or {}
        context.update(
            {"config_issue": config_issue, "config_type": "agent_configuration"}
        )

        super().__init__(
            message=message,
            agent_name=agent_name,
            error_code="agent_config_invalid",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # Config issues need manual fix
            context=context,
            step_id=step_id,
        )
        self.config_issue = config_issue

    def get_user_message(self) -> str:
        """Get user-friendly error message with configuration guidance."""
        base_message = super().get_user_message()
        return f"{base_message}\n💡 Tip: Check agent configuration for: {self.config_issue}"


class AgentResourceError(AgentExecutionError):
    """
    Exception raised when an agent cannot access required resources.

    Represents resource availability issues such as memory constraints,
    disk space, or external service dependencies.
    """

    def __init__(
        self,
        agent_name: str,
        resource_type: str,
        resource_details: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = (
            message
            or f"Agent '{agent_name}' cannot access {resource_type}: {resource_details}"
        )

        context = context or {}
        context.update(
            {"resource_type": resource_type, "resource_details": resource_details}
        )

        # Resource issues might be temporary (network) or permanent (disk space)
        retry_policy = (
            RetryPolicy.BACKOFF
            if resource_type in ["network", "api"]
            else RetryPolicy.NEVER
        )

        super().__init__(
            message=message,
            agent_name=agent_name,
            error_code="agent_resource_unavailable",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=retry_policy,
            context=context,
            step_id=step_id,
        )
        self.resource_type = resource_type
        self.resource_details = resource_details

    def get_user_message(self) -> str:
        """Get user-friendly error message with resource guidance."""
        base_message = super().get_user_message()
        if self.resource_type == "disk_space":
            return f"{base_message}\n💡 Tip: Free up disk space and try again."
        elif self.resource_type == "network":
            return f"{base_message}\n💡 Tip: Check your internet connection."
        else:
            return f"{base_message}\n💡 Tip: Check {self.resource_type} availability."


class AgentValidationError(AgentExecutionError):
    """
    Exception raised when agent input or output validation fails.

    Represents validation failures for agent inputs, outputs, or
    internal state validation during execution.
    """

    def __init__(
        self,
        agent_name: str,
        validation_type: str,
        validation_details: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = (
            message
            or f"Agent '{agent_name}' validation failed for {validation_type}: {validation_details}"
        )

        context = context or {}
        context.update(
            {
                "validation_type": validation_type,
                "validation_details": validation_details,
            }
        )

        super().__init__(
            message=message,
            agent_name=agent_name,
            error_code="agent_validation_failed",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=RetryPolicy.NEVER,  # Validation failures need input correction
            context=context,
            step_id=step_id,
        )
        self.validation_type = validation_type
        self.validation_details = validation_details

    def get_user_message(self) -> str:
        """Get user-friendly error message with validation guidance."""
        base_message = super().get_user_message()
        return f"{base_message}\n💡 Tip: Check {self.validation_type} format and try again."
