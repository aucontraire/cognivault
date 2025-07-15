"""
Tests for LangGraph routing functionality.

This module tests the various routing functions and strategies
for conditional execution in LangGraph DAGs.
"""

import pytest

from cognivault.context import AgentContext
from cognivault.orchestration.routing import (
    ConditionalRouter,
    SuccessFailureRouter,
    OutputBasedRouter,
    FailureHandlingRouter,
    AgentDependencyRouter,
    PipelineStageRouter,
    always_continue_to,
    route_on_query_type,
    route_on_success_failure,
    route_with_failure_handling,
    route_with_dependencies,
    route_by_pipeline_stage,
)


@pytest.fixture
def sample_context():
    """Create a sample agent context for testing."""
    context = AgentContext(query="Test query for routing")
    context.add_agent_output("TestAgent", "Sample output content")
    return context


@pytest.fixture
def success_context():
    """Create a context representing successful execution."""
    context = AgentContext(query="Success test")
    context.execution_state["last_agent_success"] = True
    context.successful_agents.add("TestAgent")
    return context


@pytest.fixture
def failure_context():
    """Create a context representing failed execution."""
    context = AgentContext(query="Failure test")
    context.execution_state["last_agent_success"] = False
    context.failed_agents.add("TestAgent")
    return context


class TestConditionalRouter:
    """Test cases for ConditionalRouter."""

    def test_initialization(self):
        """Test conditional router initialization."""
        conditions = [(lambda ctx: True, "target1"), (lambda ctx: False, "target2")]
        router = ConditionalRouter(conditions, "default")

        assert router.conditions == conditions
        assert router.default == "default"

    def test_first_matching_condition(self, sample_context):
        """Test that first matching condition is used."""
        conditions = [
            (lambda ctx: True, "first_match"),
            (lambda ctx: True, "second_match"),  # This should not be reached
        ]
        router = ConditionalRouter(conditions, "default")

        result = router(sample_context)
        assert result == "first_match"

    def test_no_matching_condition(self, sample_context):
        """Test fallback to default when no conditions match."""
        conditions = [(lambda ctx: False, "target1"), (lambda ctx: False, "target2")]
        router = ConditionalRouter(conditions, "default")

        result = router(sample_context)
        assert result == "default"

    def test_condition_evaluation(self, sample_context):
        """Test condition evaluation with context."""

        def check_query_content(ctx):
            return "Test" in ctx.query

        conditions = [(check_query_content, "test_target")]
        router = ConditionalRouter(conditions, "default")

        result = router(sample_context)
        assert result == "test_target"

    def test_get_possible_targets(self):
        """Test getting all possible target nodes."""
        conditions = [
            (lambda ctx: True, "target1"),
            (lambda ctx: False, "target2"),
            (lambda ctx: False, "target1"),  # Duplicate
        ]
        router = ConditionalRouter(conditions, "default")

        targets = router.get_possible_targets()
        assert set(targets) == {"target1", "target2", "default"}


class TestSuccessFailureRouter:
    """Test cases for SuccessFailureRouter."""

    def test_initialization(self):
        """Test success/failure router initialization."""
        router = SuccessFailureRouter("success_node", "failure_node")

        assert router.success_target == "success_node"
        assert router.failure_target == "failure_node"

    def test_success_routing(self, success_context):
        """Test routing on successful execution."""
        router = SuccessFailureRouter("success_node", "failure_node")

        result = router(success_context)
        assert result == "success_node"

    def test_failure_routing(self, failure_context):
        """Test routing on failed execution."""
        router = SuccessFailureRouter("success_node", "failure_node")

        result = router(failure_context)
        assert result == "failure_node"

    def test_default_success_routing(self, sample_context):
        """Test default routing when success state is not set."""
        # Default should be success if last_agent_success is not set
        router = SuccessFailureRouter("success_node", "failure_node")

        result = router(sample_context)
        assert result == "success_node"

    def test_get_possible_targets(self):
        """Test getting possible targets."""
        router = SuccessFailureRouter("success_node", "failure_node")

        targets = router.get_possible_targets()
        assert targets == ["success_node", "failure_node"]


class TestOutputBasedRouter:
    """Test cases for OutputBasedRouter."""

    def test_initialization(self):
        """Test output-based router initialization."""
        patterns = {"error": "error_handler", "success": "success_handler"}
        router = OutputBasedRouter(patterns, "default")

        assert router.output_patterns == patterns
        assert router.default == "default"

    def test_pattern_matching(self, sample_context):
        """Test routing based on output pattern matching."""
        patterns = {"sample": "sample_handler", "test": "test_handler"}
        router = OutputBasedRouter(patterns, "default")

        result = router(sample_context)
        assert result == "sample_handler"  # "Sample" should match "sample"

    def test_case_insensitive_matching(self, sample_context):
        """Test case-insensitive pattern matching."""
        patterns = {"SAMPLE": "upper_handler"}
        router = OutputBasedRouter(patterns, "default")

        result = router(sample_context)
        assert result == "upper_handler"

    def test_no_pattern_match(self, sample_context):
        """Test fallback to default when no patterns match."""
        patterns = {"nonexistent": "handler"}
        router = OutputBasedRouter(patterns, "default")

        result = router(sample_context)
        assert result == "default"

    def test_empty_agent_outputs(self):
        """Test handling of empty agent outputs."""
        context = AgentContext(query="Empty test")
        patterns = {"test": "test_handler"}
        router = OutputBasedRouter(patterns, "default")

        result = router(context)
        assert result == "default"

    def test_get_possible_targets(self):
        """Test getting all possible targets."""
        patterns = {"pattern1": "target1", "pattern2": "target2", "pattern3": "target1"}
        router = OutputBasedRouter(patterns, "default")

        targets = router.get_possible_targets()
        assert set(targets) == {"target1", "target2", "default"}


class TestFailureHandlingRouter:
    """Test cases for FailureHandlingRouter."""

    def test_initialization(self):
        """Test failure handling router initialization."""
        router = FailureHandlingRouter("success", "failure", "retry", max_failures=3)

        assert router.success_target == "success"
        assert router.failure_target == "failure"
        assert router.retry_target == "retry"
        assert router.max_failures == 3
        assert router.failure_count == 0
        assert router.circuit_open is False

    def test_success_routing_resets_failures(self, success_context):
        """Test that success resets failure count."""
        router = FailureHandlingRouter("success", "failure")
        router.failure_count = 2  # Set some failures

        result = router(success_context)

        assert result == "success"
        assert router.failure_count == 0
        assert router.circuit_open is False

    def test_failure_increments_count(self, failure_context):
        """Test that failure increments failure count."""
        router = FailureHandlingRouter("success", "failure", max_failures=3)

        result = router(failure_context)

        assert router.failure_count == 1
        assert result in ["failure", "retry"]  # Depends on retry logic

    def test_circuit_breaker_activation(self, failure_context):
        """Test circuit breaker activation after max failures."""
        router = FailureHandlingRouter("success", "failure", max_failures=2)

        # Trigger failures up to max
        router(failure_context)  # failure 1
        router(failure_context)  # failure 2 - should open circuit

        assert router.circuit_open is True
        assert router.failure_count == 2

        # Next failure should go to failure target
        result = router(failure_context)
        assert result == "failure"

    def test_retry_logic(self, failure_context):
        """Test retry logic before circuit breaking."""
        router = FailureHandlingRouter("success", "failure", "retry", max_failures=3)
        failure_context.execution_state["retry_count"] = 0

        result = router(failure_context)

        # Should retry before going to failure
        assert result == "retry"
        assert failure_context.execution_state["retry_count"] == 1

    def test_max_retries_exceeded(self, failure_context):
        """Test behavior when max retries are exceeded."""
        router = FailureHandlingRouter("success", "failure", "retry", max_failures=2)
        failure_context.execution_state["retry_count"] = 2  # Already at max

        result = router(failure_context)

        assert result == "failure"

    def test_reset_failure_state(self):
        """Test resetting failure state."""
        router = FailureHandlingRouter("success", "failure")
        router.failure_count = 5
        router.circuit_open = True

        router.reset_failure_state()

        assert router.failure_count == 0
        assert router.circuit_open is False

    def test_get_possible_targets(self):
        """Test getting possible targets."""
        router = FailureHandlingRouter("success", "failure", "retry")

        targets = router.get_possible_targets()
        assert set(targets) == {"success", "failure", "retry"}

    def test_get_possible_targets_no_retry(self):
        """Test getting possible targets when retry is same as failure."""
        router = FailureHandlingRouter(
            "success", "failure"
        )  # retry defaults to failure

        targets = router.get_possible_targets()
        assert set(targets) == {"success", "failure"}


class TestAgentDependencyRouter:
    """Test cases for AgentDependencyRouter."""

    def test_initialization(self):
        """Test dependency router initialization."""
        deps = {"target": ["dep1", "dep2"]}
        router = AgentDependencyRouter(deps, "target", "wait", "error")

        assert router.dependency_map == deps
        assert router.success_target == "target"
        assert router.wait_target == "wait"
        assert router.failure_target == "error"

    def test_dependencies_satisfied(self):
        """Test routing when all dependencies are satisfied."""
        deps = {"target": ["dep1", "dep2"]}
        router = AgentDependencyRouter(deps, "target", "wait", "error")

        context = AgentContext(query="test")
        context.successful_agents.add("dep1")
        context.successful_agents.add("dep2")

        result = router(context)
        assert result == "target"

    def test_dependencies_pending(self):
        """Test routing when dependencies are pending."""
        deps = {"target": ["dep1", "dep2"]}
        router = AgentDependencyRouter(deps, "target", "wait", "error")

        context = AgentContext(query="test")
        context.successful_agents.add("dep1")
        # dep2 is neither successful nor failed (pending)

        result = router(context)
        assert result == "wait"

    def test_dependencies_failed(self):
        """Test routing when dependencies have failed."""
        deps = {"target": ["dep1", "dep2"]}
        router = AgentDependencyRouter(deps, "target", "wait", "error")

        context = AgentContext(query="test")
        context.successful_agents.add("dep1")
        context.failed_agents.add("dep2")  # dep2 failed

        result = router(context)
        assert result == "error"

    def test_no_dependencies(self):
        """Test routing when target has no dependencies."""
        deps = {"target": []}
        router = AgentDependencyRouter(deps, "target", "wait", "error")

        context = AgentContext(query="test")

        result = router(context)
        assert result == "target"

    def test_get_possible_targets(self):
        """Test getting possible targets."""
        deps = {"target": ["dep1"]}
        router = AgentDependencyRouter(deps, "target", "wait", "error")

        targets = router.get_possible_targets()
        assert targets == ["target", "wait", "error"]


class TestPipelineStageRouter:
    """Test cases for PipelineStageRouter."""

    def test_initialization(self):
        """Test pipeline stage router initialization."""
        stages = {"stage1": "target1", "stage2": "target2"}
        router = PipelineStageRouter(stages, "default")

        assert router.stage_map == stages
        assert router.default_target == "default"

    def test_stage_routing(self):
        """Test routing based on pipeline stage."""
        stages = {"preparation": "prep_node", "execution": "exec_node"}
        router = PipelineStageRouter(stages, "default")

        context = AgentContext(query="test")
        context.execution_state["pipeline_stage"] = "execution"

        result = router(context)
        assert result == "exec_node"

    def test_unknown_stage(self):
        """Test routing for unknown stage."""
        stages = {"known_stage": "known_target"}
        router = PipelineStageRouter(stages, "default")

        context = AgentContext(query="test")
        context.execution_state["pipeline_stage"] = "unknown_stage"

        result = router(context)
        assert result == "default"

    def test_no_stage_set(self):
        """Test routing when no stage is set (uses 'initial')."""
        stages = {"initial": "initial_target", "other": "other_target"}
        router = PipelineStageRouter(stages, "default")

        context = AgentContext(query="test")
        # No pipeline_stage set - should default to "initial"

        result = router(context)
        assert result == "initial_target"

    def test_get_possible_targets(self):
        """Test getting possible targets."""
        stages = {"stage1": "target1", "stage2": "target2", "stage3": "target1"}
        router = PipelineStageRouter(stages, "default")

        targets = router.get_possible_targets()
        assert set(targets) == {"target1", "target2", "default"}


class TestFactoryFunctions:
    """Test cases for routing factory functions."""

    def test_always_continue_to(self, sample_context):
        """Test always_continue_to factory function."""
        router = always_continue_to("fixed_target")

        result = router(sample_context)
        assert result == "fixed_target"

        targets = router.get_possible_targets()
        assert targets == ["fixed_target"]

    def test_route_on_query_type(self, sample_context):
        """Test route_on_query_type factory function."""
        patterns = {"test": "test_handler"}
        router = route_on_query_type(patterns, "default")

        assert isinstance(router, OutputBasedRouter)
        # The factory creates an OutputBasedRouter, but we need agent output for pattern matching
        # Since sample_context has "Sample output content", this won't match "test"
        result = router(sample_context)
        assert result == "default"

    def test_route_on_success_failure_factory(self, success_context):
        """Test route_on_success_failure factory function."""
        router = route_on_success_failure("success", "failure")

        assert isinstance(router, SuccessFailureRouter)
        result = router(success_context)
        assert result == "success"

    def test_route_with_failure_handling_factory(self, failure_context):
        """Test route_with_failure_handling factory function."""
        router = route_with_failure_handling(
            "success", "failure", "retry", max_failures=2
        )

        assert isinstance(router, FailureHandlingRouter)
        assert router.max_failures == 2
        assert router.retry_target == "retry"

    def test_route_with_dependencies_factory(self):
        """Test route_with_dependencies factory function."""
        deps = {"target": ["dep1", "dep2"]}
        router = route_with_dependencies(deps, "target")

        assert isinstance(router, AgentDependencyRouter)
        assert router.dependency_map == deps
        assert router.success_target == "target"
        assert router.wait_target == "wait"  # default
        assert router.failure_target == "error"  # default

    def test_route_by_pipeline_stage_factory(self):
        """Test route_by_pipeline_stage factory function."""
        stages = {"stage1": "target1"}
        router = route_by_pipeline_stage(stages, "custom_default")

        assert isinstance(router, PipelineStageRouter)
        assert router.stage_map == stages
        assert router.default_target == "custom_default"

    def test_route_by_pipeline_stage_default_params(self):
        """Test route_by_pipeline_stage with default parameters."""
        stages = {"stage1": "target1"}
        router = route_by_pipeline_stage(stages)

        assert router.default_target == "end"


class TestRoutingIntegration:
    """Integration tests for routing functionality."""

    def test_complex_routing_scenario(self):
        """Test complex routing scenario with multiple conditions."""
        # Create a complex routing scenario
        context = AgentContext(query="test complex routing")

        # Set up context state
        context.execution_state["last_agent_success"] = False
        context.execution_state["pipeline_stage"] = "recovery"
        context.failed_agents.add("primary_agent")
        context.add_agent_output("ErrorAgent", "connection timeout error")

        # Test failure handling router
        failure_router = FailureHandlingRouter(
            "success", "error", "retry", max_failures=2
        )
        failure_result = failure_router(context)
        assert failure_result in ["error", "retry"]

        # Test output-based router
        output_router = OutputBasedRouter({"timeout": "timeout_handler"}, "default")
        output_result = output_router(context)
        assert output_result == "timeout_handler"

        # Test pipeline stage router
        stage_router = PipelineStageRouter({"recovery": "recovery_node"}, "default")
        stage_result = stage_router(context)
        assert stage_result == "recovery_node"

    def test_router_chaining(self, sample_context):
        """Test chaining multiple routers together."""
        # First router - check for errors
        error_patterns = {"error": "error_handler", "fail": "fail_handler"}
        error_router = OutputBasedRouter(error_patterns, "continue")

        # Second router - check success/failure
        success_router = SuccessFailureRouter("success_handler", "failure_handler")

        # Apply first router
        first_result = error_router(sample_context)
        assert first_result == "continue"  # No error patterns in sample output

        # Apply second router if first router says continue
        if first_result == "continue":
            second_result = success_router(sample_context)
            assert second_result == "success_handler"  # Default success state

    def test_router_state_modification(self):
        """Test that routers can modify context state."""
        context = AgentContext(query="test")
        context.execution_state["retry_count"] = 0

        router = FailureHandlingRouter("success", "failure", "retry")
        context.execution_state["last_agent_success"] = False

        router(context)

        # Router should have incremented retry count
        assert context.execution_state["retry_count"] == 1
