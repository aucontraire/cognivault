"""
Tests for orchestration-specific exception classes.

This module tests orchestration-related exceptions including pipeline execution,
dependency resolution, workflow timeouts, state transitions, and circuit breaker
errors for DAG-based execution flow.
"""

import pytest
from cognivault.exceptions import (
    CogniVaultError,
    ErrorSeverity,
    RetryPolicy,
    OrchestrationError,
    PipelineExecutionError,
    DependencyResolutionError,
    WorkflowTimeoutError,
    StateTransitionError,
)


class TestOrchestrationErrorBase:
    """Test base OrchestrationError functionality."""

    def test_orchestration_error_creation(self):
        """Test basic OrchestrationError creation."""
        error = OrchestrationError(
            message="Orchestration failed",
            pipeline_stage="execution",
        )

        assert error.message == "Orchestration failed"
        assert error.pipeline_stage == "execution"
        assert error.error_code == "orchestration_error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.BACKOFF
        assert error.context["pipeline_stage"] == "execution"
        assert isinstance(error, CogniVaultError)

    def test_orchestration_error_with_all_params(self):
        """Test OrchestrationError with all parameters."""
        cause_exception = ValueError("Original error")
        context = {"workflow_id": "wf_123", "node_count": 5}

        error = OrchestrationError(
            message="Complex orchestration error",
            pipeline_stage="validation",
            error_code="custom_orchestration_error",
            severity=ErrorSeverity.CRITICAL,
            retry_policy=RetryPolicy.NEVER,
            context=context,
            step_id="orch_step",
            agent_id="OrchestratorAgent",
            cause=cause_exception,
        )

        assert error.message == "Complex orchestration error"
        assert error.pipeline_stage == "validation"
        assert error.error_code == "custom_orchestration_error"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.retry_policy == RetryPolicy.NEVER
        assert error.step_id == "orch_step"
        assert error.agent_id == "OrchestratorAgent"
        assert error.cause == cause_exception
        assert error.context["workflow_id"] == "wf_123"
        assert error.context["pipeline_stage"] == "validation"


class TestPipelineExecutionError:
    """Test PipelineExecutionError functionality."""

    def test_pipeline_execution_error_creation(self):
        """Test basic PipelineExecutionError creation."""
        failed_agents = ["agent1", "agent2"]
        successful_agents = ["agent3", "agent4", "agent5"]

        error = PipelineExecutionError(
            failed_agents=failed_agents,
            successful_agents=successful_agents,
            pipeline_stage="processing",
            failure_reason="network timeout",
        )

        assert error.failed_agents == failed_agents
        assert error.successful_agents == successful_agents
        assert error.pipeline_stage == "processing"
        assert error.failure_reason == "network timeout"
        assert error.error_code == "pipeline_execution_failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.BACKOFF  # timeout is retryable
        assert error.agent_id == "pipeline_processing"

        # Check context details
        assert error.context["failed_agents"] == failed_agents
        assert error.context["successful_agents"] == successful_agents
        assert error.context["failed_count"] == 2
        assert error.context["success_count"] == 3
        assert error.context["total_agents"] == 5
        assert error.context["failure_reason"] == "network timeout"
        assert error.context["partial_failure"] is True

        # Check default message construction
        expected_msg = (
            "Pipeline execution failed at 'processing': "
            "2/5 agents failed (network timeout)"
        )
        assert error.message == expected_msg

    def test_pipeline_execution_error_complete_failure(self):
        """Test pipeline execution error with complete failure."""
        failed_agents = ["agent1", "agent2", "agent3"]
        successful_agents = []

        error = PipelineExecutionError(
            failed_agents=failed_agents,
            successful_agents=successful_agents,
            pipeline_stage="initialization",
            failure_reason="configuration error",
        )

        assert error.context["partial_failure"] is False
        assert error.retry_policy == RetryPolicy.NEVER  # non-timeout failure

    def test_pipeline_execution_user_message(self):
        """Test user-friendly pipeline execution error message."""
        error = PipelineExecutionError(
            failed_agents=["refiner", "critic"],
            successful_agents=["researcher", "planner"],
            pipeline_stage="analysis",
            failure_reason="API quota exceeded",
        )

        user_msg = error.get_user_message()
        assert "Pipeline failed at 'analysis'" in user_msg
        assert "API quota exceeded" in user_msg
        assert "Failed agents: refiner, critic" in user_msg
        assert "✅ Successful agents: researcher, planner" in user_msg
        assert "💡 Tip: Check individual agent logs" in user_msg


class TestDependencyResolutionError:
    """Test DependencyResolutionError functionality."""

    def test_dependency_resolution_error_creation(self):
        """Test basic DependencyResolutionError creation."""
        affected_agents = ["agent2", "agent3", "agent4"]
        dependency_graph = {
            "agent1": [],
            "agent2": ["agent1"],
            "agent3": ["agent2"],
            "agent4": ["agent2", "agent3"],
        }

        error = DependencyResolutionError(
            dependency_issue="circular dependency detected",
            affected_agents=affected_agents,
            dependency_graph=dependency_graph,
        )

        assert error.dependency_issue == "circular dependency detected"
        assert error.affected_agents == affected_agents
        assert error.dependency_graph == dependency_graph
        assert error.error_code == "dependency_resolution_failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER
        assert error.pipeline_stage == "dependency_resolution"
        assert error.agent_id == "orchestrator"

        # Check context details
        assert error.context["dependency_issue"] == "circular dependency detected"
        assert error.context["affected_agents"] == affected_agents
        assert error.context["dependency_graph"] == dependency_graph
        assert error.context["graph_analysis_required"] is True

        # Check default message construction
        expected_msg = "Dependency resolution failed: circular dependency detected"
        assert error.message == expected_msg

    def test_dependency_resolution_user_message(self):
        """Test user-friendly dependency resolution error message."""
        error = DependencyResolutionError(
            dependency_issue="missing required dependency",
            affected_agents=["agent1", "agent2", "agent3", "agent4", "agent5"],
        )

        user_msg = error.get_user_message()
        assert "Dependency resolution failed: missing required dependency" in user_msg
        assert "Affected agents: agent1, agent2, agent3 (and 2 more)" in user_msg
        assert "💡 Tip: Check agent dependency configuration" in user_msg


class TestWorkflowTimeoutError:
    """Test WorkflowTimeoutError functionality."""

    def test_workflow_timeout_error_creation(self):
        """Test basic WorkflowTimeoutError creation."""
        completed_agents = ["agent1", "agent2"]
        pending_agents = ["agent3", "agent4", "agent5"]

        error = WorkflowTimeoutError(
            timeout_seconds=300.0,
            timeout_stage="execution",
            completed_agents=completed_agents,
            pending_agents=pending_agents,
        )

        assert error.timeout_seconds == 300.0
        assert error.timeout_stage == "execution"
        assert error.completed_agents == completed_agents
        assert error.pending_agents == pending_agents
        assert error.error_code == "workflow_timeout"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.BACKOFF
        assert error.pipeline_stage == "execution"
        assert error.agent_id == "orchestrator"

        # Check context details
        assert error.context["timeout_seconds"] == 300.0
        assert error.context["timeout_stage"] == "execution"
        assert error.context["completed_agents"] == completed_agents
        assert error.context["pending_agents"] == pending_agents
        assert error.context["partial_completion"] is True

        # Check default message construction
        expected_msg = (
            "Workflow timeout at 'execution' after 300.0s (2 completed, 3 pending)"
        )
        assert error.message == expected_msg

    def test_workflow_timeout_user_message(self):
        """Test user-friendly workflow timeout error message."""
        error = WorkflowTimeoutError(
            timeout_seconds=120.0,
            timeout_stage="analysis",
            completed_agents=["researcher", "validator"],
            pending_agents=["refiner", "critic", "reviewer", "summarizer"],
        )

        user_msg = error.get_user_message()
        assert "Workflow timeout at 'analysis' (120.0s)" in user_msg
        assert "✅ Completed: researcher, validator" in user_msg
        assert "⏳ Pending: refiner, critic, reviewer (and 1 more)" in user_msg
        assert "💡 Tip: Consider increasing timeout" in user_msg


class TestStateTransitionError:
    """Test StateTransitionError functionality."""

    def test_state_transition_error_creation(self):
        """Test basic StateTransitionError creation."""
        error = StateTransitionError(
            transition_type="agent_to_agent",
            from_state="refiner_completed",
            to_state="critic_starting",
        )

        assert error.transition_type == "agent_to_agent"
        assert error.from_state == "refiner_completed"
        assert error.to_state == "critic_starting"
        assert error.error_code == "state_transition_failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.retry_policy == RetryPolicy.NEVER
        assert error.pipeline_stage == "state_management"

        # Check default message construction with states
        expected_msg = "State transition failed: agent_to_agent (refiner_completed → critic_starting)"
        assert error.message == expected_msg

    def test_state_transition_error_with_all_params(self):
        """Test StateTransitionError with all parameters."""
        cause = ValueError("Invalid state data")
        context = {"execution_id": "exec_123", "node_count": 5}

        error = StateTransitionError(
            transition_type="workflow_validation",
            from_state="planning",
            to_state="execution",
            state_details="Missing required validation data",
            message="Custom state transition failure",
            step_id="validation_step",
            agent_id="OrchestrationAgent",
            context=context,
            cause=cause,
        )

        assert error.transition_type == "workflow_validation"
        assert error.from_state == "planning"
        assert error.to_state == "execution"
        assert error.state_details == "Missing required validation data"
        assert error.message == "Custom state transition failure (planning → execution)"
        assert error.step_id == "validation_step"
        assert error.agent_id == "OrchestrationAgent"
        assert error.cause == cause
        assert error.context["execution_id"] == "exec_123"

    def test_state_transition_context_injection(self):
        """Test that state transition information is added to context."""
        error = StateTransitionError(
            transition_type="conditional_branch",
            from_state="decision_point",
            to_state="branch_a",
            state_details="Condition evaluation failed",
        )

        assert error.context["transition_type"] == "conditional_branch"
        assert error.context["from_state"] == "decision_point"
        assert error.context["to_state"] == "branch_a"
        assert error.context["state_details"] == "Condition evaluation failed"
        assert error.context["rollback_required"] is False

    def test_state_transition_with_minimal_params(self):
        """Test StateTransitionError with minimal required parameters."""
        error = StateTransitionError(transition_type="initialization_failed")

        assert error.transition_type == "initialization_failed"
        assert error.from_state is None
        assert error.to_state is None
        assert error.state_details is None

        # Check message construction without states
        expected_msg = "State transition failed: initialization_failed"
        assert error.message == expected_msg

    def test_state_transition_rollback_scenarios(self):
        """Test state transition errors that require rollback."""
        # Test snapshot_failed
        snapshot_error = StateTransitionError(
            transition_type="snapshot_failed",
            state_details="Memory exhausted during snapshot",
        )
        assert snapshot_error.context["rollback_required"] is True

        # Test rollback_failed
        rollback_error = StateTransitionError(
            transition_type="rollback_failed",
            state_details="Rollback data corrupted",
        )
        assert rollback_error.context["rollback_required"] is True

    def test_state_transition_user_message(self):
        """Test user-friendly message for state transition errors."""
        # Test snapshot failure
        snapshot_error = StateTransitionError(
            transition_type="snapshot_failed",
            state_details="Context too large",
        )
        user_msg = snapshot_error.get_user_message()
        assert "Context snapshot failed" in user_msg
        assert "💡 Tip: Context snapshot failed. Check memory usage" in user_msg

        # Test rollback failure
        rollback_error = StateTransitionError(
            transition_type="rollback_failed",
            from_state="corrupted",
            to_state="recovered",
        )
        user_msg = rollback_error.get_user_message()
        assert "(corrupted → recovered)" in user_msg
        assert "💡 Tip: Context rollback failed. Manual intervention" in user_msg

        # Test generic failure
        generic_error = StateTransitionError(
            transition_type="dependency_check",
            from_state="waiting",
            to_state="ready",
        )
        user_msg = generic_error.get_user_message()
        assert "💡 Tip: State management error. Check system resources" in user_msg

    def test_state_transition_inheritance(self):
        """Test StateTransitionError inheritance hierarchy."""
        error = StateTransitionError(
            transition_type="test_transition",
            from_state="start",
            to_state="end",
        )

        assert isinstance(error, OrchestrationError)
        assert isinstance(error, CogniVaultError)
        assert isinstance(error, StateTransitionError)

    def test_state_transition_serialization(self):
        """Test StateTransitionError serialization."""
        error = StateTransitionError(
            transition_type="serialization_test",
            from_state="state_a",
            to_state="state_b",
            state_details="Test details",
            step_id="ser_step",
            agent_id="SerAgent",
        )

        data = error.to_dict()

        # Check standard fields
        assert data["message"].startswith("State transition failed:")
        assert data["error_code"] == "state_transition_failed"
        assert data["severity"] == "high"
        assert data["step_id"] == "ser_step"
        assert data["agent_id"] == "SerAgent"

        # Check transition-specific context
        assert data["context"]["transition_type"] == "serialization_test"
        assert data["context"]["from_state"] == "state_a"
        assert data["context"]["to_state"] == "state_b"
        assert data["context"]["state_details"] == "Test details"


class TestOrchestrationErrorInheritance:
    """Test proper inheritance hierarchy for orchestration errors."""

    def test_all_orchestration_errors_inherit_properly(self):
        """Test that specialized orchestration errors inherit from OrchestrationError."""
        errors = [
            PipelineExecutionError(["agent1"], ["agent2"], "test", "timeout"),
            DependencyResolutionError("circular", ["agent1", "agent2"]),
            WorkflowTimeoutError(60.0, "test", ["agent1"], ["agent2"]),
            StateTransitionError("test", "from", "to"),
        ]

        for error in errors:
            assert isinstance(error, OrchestrationError)
            assert isinstance(error, CogniVaultError)

    def test_orchestration_error_inherits_from_base(self):
        """Test that OrchestrationError inherits from CogniVaultError."""
        error = OrchestrationError("Test", "test_stage")
        assert isinstance(error, CogniVaultError)

    def test_polymorphic_behavior(self):
        """Test polymorphic behavior of orchestration errors."""

        def handle_orchestration_error(error: OrchestrationError) -> dict:
            return {
                "stage": error.pipeline_stage,
                "retryable": error.is_retryable(),
                "severity": error.severity.value,
                "type": error.__class__.__name__,
            }

        errors = [
            OrchestrationError("Base", "test"),
            PipelineExecutionError(["a1"], ["a2"], "test", "timeout"),  # retryable
            DependencyResolutionError("circular", ["a1"]),  # not retryable
            WorkflowTimeoutError(60.0, "test", ["a1"], ["a2"]),  # retryable
            StateTransitionError("snapshot_failed"),  # not retryable
        ]

        results = [handle_orchestration_error(err) for err in errors]

        assert len(results) == 5
        assert results[0]["retryable"] is True  # BACKOFF
        assert results[1]["retryable"] is True  # BACKOFF (timeout)
        assert results[2]["retryable"] is False  # NEVER (dependency)
        assert results[3]["retryable"] is True  # BACKOFF (timeout)
        assert results[4]["retryable"] is False  # NEVER (state)


class TestOrchestrationErrorIntegration:
    """Test integration aspects of orchestration errors."""

    def test_orchestration_error_with_pipeline_metadata(self):
        """Test orchestration errors work with pipeline execution metadata."""
        error = PipelineExecutionError(
            failed_agents=["researcher", "critic"],
            successful_agents=["refiner", "validator"],
            pipeline_stage="content_analysis",
            failure_reason="LLM quota exceeded",
            step_id="pipeline_step_456",
            context={
                "pipeline_id": "pipe_789",
                "execution_mode": "parallel",
                "start_time": "2024-01-01T10:00:00Z",
            },
        )

        # Verify all metadata is properly integrated
        assert error.agent_id == "pipeline_content_analysis"
        assert error.step_id == "pipeline_step_456"
        assert error.context["pipeline_id"] == "pipe_789"
        assert error.context["execution_mode"] == "parallel"

        # Verify serialization includes everything
        data = error.to_dict()
        assert data["step_id"] == "pipeline_step_456"
        assert data["agent_id"] == "pipeline_content_analysis"
        assert "pipeline_id" in data["context"]

    def test_orchestration_error_chaining_scenarios(self):
        """Test various orchestration error chaining scenarios."""
        # Original timeout error
        original = TimeoutError("Agent execution timeout")

        # State transition error wrapping timeout
        state_error = StateTransitionError(
            transition_type="timeout_recovery",
            from_state="executing",
            to_state="failed",
            state_details="Agent timeout during execution",
            step_id="state_step",
            cause=original,
        )

        # Pipeline error wrapping state error
        pipeline_error = PipelineExecutionError(
            failed_agents=["timeout_agent"],
            successful_agents=["stable_agent"],
            pipeline_stage="recovery",
            failure_reason="state transition failed",
            step_id="pipeline_step",
            cause=state_error,
        )

        # Verify chaining
        assert pipeline_error.cause == state_error
        assert state_error.cause == original

        # Verify serialization handles nested causes
        pipeline_data = pipeline_error.to_dict()
        state_data = state_error.to_dict()

        assert "State transition failed" in pipeline_data["cause"]
        assert "Agent execution timeout" in state_data["cause"]

    def test_exception_raising_and_catching(self):
        """Test that orchestration errors can be properly raised and caught."""
        # Test specific exception catching
        with pytest.raises(StateTransitionError) as exc_info:
            raise StateTransitionError("test_transition", "from", "to")

        assert exc_info.value.transition_type == "test_transition"
        assert exc_info.value.from_state == "from"

        # Test catching as base orchestration error
        with pytest.raises(OrchestrationError) as exc_info:
            raise PipelineExecutionError(["failed"], ["success"], "test", "reason")

        assert exc_info.value.pipeline_stage == "test"

        # Test catching as CogniVaultError
        with pytest.raises(CogniVaultError) as exc_info:
            raise DependencyResolutionError("circular", ["agent1"])

        assert exc_info.value.error_code == "dependency_resolution_failed"

    def test_orchestration_error_retry_semantics(self):
        """Test retry semantics for orchestration operations."""
        # Timeout-based failures should be retryable
        timeout_error = PipelineExecutionError(
            failed_agents=["agent1"],
            successful_agents=["agent2"],
            pipeline_stage="execution",
            failure_reason="agent timeout detected",
        )
        assert timeout_error.is_retryable()
        assert timeout_error.retry_policy == RetryPolicy.BACKOFF

        # Configuration failures should not be retryable
        config_error = PipelineExecutionError(
            failed_agents=["agent1"],
            successful_agents=[],
            pipeline_stage="initialization",
            failure_reason="invalid configuration",
        )
        assert not config_error.is_retryable()
        assert config_error.retry_policy == RetryPolicy.NEVER

        # Dependency errors should never be retryable
        dep_error = DependencyResolutionError("circular dependency", ["a1", "a2"])
        assert not dep_error.is_retryable()
        assert dep_error.retry_policy == RetryPolicy.NEVER

        # Workflow timeouts should be retryable
        workflow_timeout = WorkflowTimeoutError(300.0, "analysis", ["a1"], ["a2"])
        assert workflow_timeout.is_retryable()
        assert workflow_timeout.retry_policy == RetryPolicy.BACKOFF
