"""
Orchestration and pipeline-related exceptions for CogniVault.

This module defines exceptions related to agent orchestration,
pipeline execution, and workflow management with LangGraph DAG compatibility.
"""

from typing import Optional, Dict, Any, List
from . import CogniVaultError, ErrorSeverity, RetryPolicy


class OrchestrationError(CogniVaultError):
    """
    Base exception for orchestration and pipeline failures.

    Represents errors in agent orchestration, pipeline execution,
    and workflow management designed for LangGraph DAG compatibility.
    """

    def __init__(
        self,
        message: str,
        pipeline_stage: Optional[str] = None,
        error_code: str = "orchestration_error",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        retry_policy: RetryPolicy = RetryPolicy.BACKOFF,
        context: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = context or {}
        if pipeline_stage:
            context["pipeline_stage"] = pipeline_stage

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
        self.pipeline_stage = pipeline_stage


class PipelineExecutionError(OrchestrationError):
    """
    Exception raised when pipeline execution fails.

    Represents failures during multi-agent pipeline execution,
    including partial failures and agent dependency issues.
    """

    def __init__(
        self,
        failed_agents: List[str],
        successful_agents: List[str],
        pipeline_stage: str,
        failure_reason: str,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        failed_count = len(failed_agents)
        success_count = len(successful_agents)
        total_agents = failed_count + success_count

        message = message or (
            f"Pipeline execution failed at '{pipeline_stage}': "
            f"{failed_count}/{total_agents} agents failed ({failure_reason})"
        )

        context = context or {}
        context.update(
            {
                "failed_agents": failed_agents,
                "successful_agents": successful_agents,
                "failed_count": failed_count,
                "success_count": success_count,
                "total_agents": total_agents,
                "failure_reason": failure_reason,
                "partial_failure": success_count > 0,
            }
        )

        # Determine retry policy based on failure type
        retry_policy = (
            RetryPolicy.BACKOFF
            if "timeout" in failure_reason.lower()
            else RetryPolicy.NEVER
        )

        super().__init__(
            message=message,
            pipeline_stage=pipeline_stage,
            error_code="pipeline_execution_failed",
            severity=ErrorSeverity.HIGH,
            retry_policy=retry_policy,
            context=context,
            step_id=step_id,
            agent_id=f"pipeline_{pipeline_stage}",
            cause=cause,
        )
        self.failed_agents = failed_agents
        self.successful_agents = successful_agents
        self.failure_reason = failure_reason

    def get_user_message(self) -> str:
        """Get user-friendly error message with pipeline details."""
        failed_str = ", ".join(self.failed_agents[:3])
        if len(self.failed_agents) > 3:
            failed_str += f" (and {len(self.failed_agents) - 3} more)"

        base_msg = (
            f"❌ Pipeline failed at '{self.pipeline_stage}': {self.failure_reason}\n"
        )
        base_msg += f"Failed agents: {failed_str}\n"

        if self.successful_agents:
            success_str = ", ".join(self.successful_agents[:3])
            if len(self.successful_agents) > 3:
                success_str += f" (and {len(self.successful_agents) - 3} more)"
            base_msg += f"✅ Successful agents: {success_str}\n"

        base_msg += (
            "💡 Tip: Check individual agent logs for detailed error information."
        )
        return base_msg


class DependencyResolutionError(OrchestrationError):
    """
    Exception raised when agent dependencies cannot be resolved.

    Represents circular dependencies, missing dependencies,
    or invalid dependency configurations in the agent graph.
    """

    def __init__(
        self,
        dependency_issue: str,
        affected_agents: List[str],
        dependency_graph: Optional[Dict[str, List[str]]] = None,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = message or f"Dependency resolution failed: {dependency_issue}"

        context = context or {}
        context.update(
            {
                "dependency_issue": dependency_issue,
                "affected_agents": affected_agents,
                "dependency_graph": dependency_graph or {},
                "graph_analysis_required": True,
            }
        )

        super().__init__(
            message=message,
            pipeline_stage="dependency_resolution",
            error_code="dependency_resolution_failed",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # Dependency issues need manual fix
            context=context,
            step_id=step_id,
            agent_id="orchestrator",
        )
        self.dependency_issue = dependency_issue
        self.affected_agents = affected_agents
        self.dependency_graph = dependency_graph or {}

    def get_user_message(self) -> str:
        """Get user-friendly error message with dependency guidance."""
        agents_str = ", ".join(self.affected_agents[:3])
        if len(self.affected_agents) > 3:
            agents_str += f" (and {len(self.affected_agents) - 3} more)"

        return (
            f"❌ Dependency resolution failed: {self.dependency_issue}\n"
            f"Affected agents: {agents_str}\n"
            f"💡 Tip: Check agent dependency configuration for circular or missing dependencies."
        )


class WorkflowTimeoutError(OrchestrationError):
    """
    Exception raised when workflow execution times out.

    Represents timeout failures at the workflow level,
    including overall pipeline timeouts and stage timeouts.
    """

    def __init__(
        self,
        timeout_seconds: float,
        timeout_stage: str,
        completed_agents: List[str],
        pending_agents: List[str],
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = message or (
            f"Workflow timeout at '{timeout_stage}' after {timeout_seconds}s "
            f"({len(completed_agents)} completed, {len(pending_agents)} pending)"
        )

        context = context or {}
        context.update(
            {
                "timeout_seconds": timeout_seconds,
                "timeout_stage": timeout_stage,
                "completed_agents": completed_agents,
                "pending_agents": pending_agents,
                "partial_completion": len(completed_agents) > 0,
            }
        )

        super().__init__(
            message=message,
            pipeline_stage=timeout_stage,
            error_code="workflow_timeout",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.BACKOFF,  # Timeouts might be retryable
            context=context,
            step_id=step_id,
            agent_id="orchestrator",
        )
        self.timeout_seconds = timeout_seconds
        self.timeout_stage = timeout_stage
        self.completed_agents = completed_agents
        self.pending_agents = pending_agents

    def get_user_message(self) -> str:
        """Get user-friendly error message with timeout guidance."""
        base_msg = (
            f"❌ Workflow timeout at '{self.timeout_stage}' ({self.timeout_seconds}s)\n"
        )

        if self.completed_agents:
            completed_str = ", ".join(self.completed_agents[:3])
            if len(self.completed_agents) > 3:
                completed_str += f" (and {len(self.completed_agents) - 3} more)"
            base_msg += f"✅ Completed: {completed_str}\n"

        if self.pending_agents:
            pending_str = ", ".join(self.pending_agents[:3])
            if len(self.pending_agents) > 3:
                pending_str += f" (and {len(self.pending_agents) - 3} more)"
            base_msg += f"⏳ Pending: {pending_str}\n"

        base_msg += "💡 Tip: Consider increasing timeout or simplifying the workflow."
        return base_msg


class StateTransitionError(OrchestrationError):
    """
    Exception raised when context state transitions fail.

    Represents failures in context state management, snapshot
    operations, or state rollback operations critical for LangGraph compatibility.
    """

    def __init__(
        self,
        transition_type: str,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        state_details: Optional[str] = None,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = message or f"State transition failed: {transition_type}"
        if from_state and to_state:
            message += f" ({from_state} → {to_state})"

        context = context or {}
        context.update(
            {
                "transition_type": transition_type,
                "from_state": from_state,
                "to_state": to_state,
                "state_details": state_details,
                "rollback_required": transition_type
                in ["snapshot_failed", "rollback_failed"],
            }
        )

        super().__init__(
            message=message,
            pipeline_stage="state_management",
            error_code="state_transition_failed",
            severity=ErrorSeverity.HIGH,
            retry_policy=RetryPolicy.NEVER,  # State issues need investigation
            context=context,
            step_id=step_id,
            agent_id=agent_id,
            cause=cause,
        )
        self.transition_type = transition_type
        self.from_state = from_state
        self.to_state = to_state
        self.state_details = state_details

    def get_user_message(self) -> str:
        """Get user-friendly error message with state guidance."""
        base_msg = f"❌ State transition failed: {self.transition_type}"

        if self.from_state and self.to_state:
            base_msg += f" ({self.from_state} → {self.to_state})"

        if self.transition_type == "snapshot_failed":
            base_msg += "\n💡 Tip: Context snapshot failed. Check memory usage and context size."
        elif self.transition_type == "rollback_failed":
            base_msg += "\n💡 Tip: Context rollback failed. Manual intervention may be required."
        else:
            base_msg += "\n💡 Tip: State management error. Check system resources and try again."

        return base_msg


class CircuitBreakerError(OrchestrationError):
    """
    Exception raised when circuit breaker is open.

    Represents circuit breaker activation due to repeated failures,
    preventing further attempts until the circuit recovers.
    """

    def __init__(
        self,
        service_name: str,
        failure_count: int,
        failure_threshold: int,
        circuit_open_duration: float,
        message: Optional[str] = None,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = message or (
            f"Circuit breaker open for '{service_name}': "
            f"{failure_count}/{failure_threshold} failures, "
            f"retry in {circuit_open_duration}s"
        )

        context = context or {}
        context.update(
            {
                "service_name": service_name,
                "failure_count": failure_count,
                "failure_threshold": failure_threshold,
                "circuit_open_duration": circuit_open_duration,
                "circuit_breaker_active": True,
            }
        )

        super().__init__(
            message=message,
            pipeline_stage="circuit_breaker",
            error_code="circuit_breaker_open",
            severity=ErrorSeverity.MEDIUM,
            retry_policy=RetryPolicy.CIRCUIT_BREAKER,
            context=context,
            step_id=step_id,
            agent_id=agent_id,
        )
        self.service_name = service_name
        self.failure_count = failure_count
        self.failure_threshold = failure_threshold
        self.circuit_open_duration = circuit_open_duration

    def get_user_message(self) -> str:
        """Get user-friendly error message with circuit breaker guidance."""
        return (
            f"❌ Circuit breaker open for '{self.service_name}' "
            f"({self.failure_count} consecutive failures)\n"
            f"💡 Tip: Service temporarily unavailable. "
            f"Will retry automatically in {self.circuit_open_duration}s."
        )
