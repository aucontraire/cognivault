"""
Routing utilities for LangGraph conditional execution.

This module provides conditional routing functions and utilities
for building dynamic graph execution paths.
"""

from typing import Callable, Dict, Any, List, Optional
from abc import ABC, abstractmethod

from cognivault.context import AgentContext


class RoutingFunction(ABC):
    """Abstract base for routing functions."""

    @abstractmethod
    def __call__(self, context: AgentContext) -> str:
        """
        Determine the next node based on context.

        Parameters
        ----------
        context : AgentContext
            Current execution context

        Returns
        -------
        str
            Name of the next node to execute
        """
        pass

    @abstractmethod
    def get_possible_targets(self) -> List[str]:
        """Get list of possible target nodes."""
        pass


class ConditionalRouter(RoutingFunction):
    """
    Router that uses conditions to determine next node.

    This router evaluates a series of conditions and returns
    the first matching target node.
    """

    def __init__(
        self, conditions: List[tuple[Callable[[AgentContext], bool], str]], default: str
    ):
        """
        Initialize the conditional router.

        Parameters
        ----------
        conditions : List[tuple[Callable, str]]
            List of (condition_function, target_node) pairs
        default : str
            Default node if no conditions match
        """
        self.conditions = conditions
        self.default = default

    def __call__(self, context: AgentContext) -> str:
        """Evaluate conditions and return target node."""
        for condition_func, target_node in self.conditions:
            if condition_func(context):
                return target_node
        return self.default

    def get_possible_targets(self) -> List[str]:
        """Get all possible target nodes."""
        targets = [target for _, target in self.conditions]
        targets.append(self.default)
        return list(set(targets))


class SuccessFailureRouter(RoutingFunction):
    """Router based on agent execution success/failure."""

    def __init__(self, success_target: str, failure_target: str):
        self.success_target = success_target
        self.failure_target = failure_target

    def __call__(self, context: AgentContext) -> str:
        """Route based on last agent execution success."""
        # Check if last agent execution was successful
        if context.execution_state.get("last_agent_success", True):
            return self.success_target
        return self.failure_target

    def get_possible_targets(self) -> List[str]:
        return [self.success_target, self.failure_target]


class OutputBasedRouter(RoutingFunction):
    """Router based on agent output content."""

    def __init__(self, output_patterns: Dict[str, str], default: str):
        """
        Initialize output-based router.

        Parameters
        ----------
        output_patterns : Dict[str, str]
            Mapping of output patterns to target nodes
        default : str
            Default target if no patterns match
        """
        self.output_patterns = output_patterns
        self.default = default

    def __call__(self, context: AgentContext) -> str:
        """Route based on agent output content."""
        # Get the last agent output
        if context.agent_outputs:
            last_agent = list(context.agent_outputs.keys())[-1]
            last_output = context.agent_outputs[last_agent]

            # Check patterns
            for pattern, target in self.output_patterns.items():
                if pattern.lower() in last_output.lower():
                    return target

        return self.default

    def get_possible_targets(self) -> List[str]:
        targets = list(self.output_patterns.values())
        targets.append(self.default)
        return list(set(targets))


# Predefined routing functions for common scenarios
def always_continue_to(target: str) -> RoutingFunction:
    """Create a router that always routes to the same target."""

    class AlwaysRouter(RoutingFunction):
        def __call__(self, context: AgentContext) -> str:
            return target

        def get_possible_targets(self) -> List[str]:
            return [target]

    return AlwaysRouter()


def route_on_query_type(patterns: Dict[str, str], default: str) -> RoutingFunction:
    """Create a router based on query content patterns."""
    return OutputBasedRouter(patterns, default)


def route_on_success_failure(
    success_target: str, failure_target: str
) -> RoutingFunction:
    """Create a router based on execution success/failure."""
    return SuccessFailureRouter(success_target, failure_target)


class FailureHandlingRouter(RoutingFunction):
    """Router with sophisticated failure handling and recovery strategies."""

    def __init__(
        self,
        success_target: str,
        failure_target: str,
        retry_target: Optional[str] = None,
        max_failures: int = 3,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize failure handling router.

        Parameters
        ----------
        success_target : str
            Target node for successful execution
        failure_target : str
            Target node for failures
        retry_target : str, optional
            Target node for retry attempts (if None, uses failure_target)
        max_failures : int
            Maximum failures before circuit breaking
        enable_circuit_breaker : bool
            Whether to enable circuit breaker pattern
        """
        self.success_target = success_target
        self.failure_target = failure_target
        self.retry_target = retry_target or failure_target
        self.max_failures = max_failures
        self.enable_circuit_breaker = enable_circuit_breaker

        # Failure tracking
        self.failure_count = 0
        self.circuit_open = False

    def __call__(self, context: AgentContext) -> str:
        """Route based on execution state and failure history."""
        # Check if last execution was successful
        last_agent_success = context.execution_state.get("last_agent_success", True)

        if last_agent_success:
            # Reset failure count on success
            self.failure_count = 0
            self.circuit_open = False
            return self.success_target

        # Handle failure
        self.failure_count += 1

        # Check circuit breaker
        if self.enable_circuit_breaker and self.failure_count >= self.max_failures:
            self.circuit_open = True
            return self.failure_target

        # Check if we should retry
        current_retry_count = context.execution_state.get("retry_count", 0)
        if current_retry_count < self.max_failures and not self.circuit_open:
            context.execution_state["retry_count"] = current_retry_count + 1
            return self.retry_target

        return self.failure_target

    def get_possible_targets(self) -> List[str]:
        targets = [self.success_target, self.failure_target]
        if self.retry_target and self.retry_target not in targets:
            targets.append(self.retry_target)
        return targets

    def reset_failure_state(self) -> None:
        """Reset failure tracking state."""
        self.failure_count = 0
        self.circuit_open = False


class AgentDependencyRouter(RoutingFunction):
    """Router that considers agent dependencies and execution state."""

    def __init__(
        self,
        dependency_map: Dict[str, List[str]],
        success_target: str,
        wait_target: str,
        failure_target: str,
    ):
        """
        Initialize dependency router.

        Parameters
        ----------
        dependency_map : Dict[str, List[str]]
            Mapping of target nodes to their required dependencies
        success_target : str
            Target when dependencies are satisfied
        wait_target : str
            Target to wait for dependencies
        failure_target : str
            Target when dependencies fail
        """
        self.dependency_map = dependency_map
        self.success_target = success_target
        self.wait_target = wait_target
        self.failure_target = failure_target

    def __call__(self, context: AgentContext) -> str:
        """Route based on dependency satisfaction."""
        # Check if dependencies for success_target are satisfied
        dependencies = self.dependency_map.get(self.success_target, [])

        for dependency in dependencies:
            # Check if dependency agent completed successfully
            if dependency not in context.successful_agents:
                # Check if dependency failed
                if dependency in context.failed_agents:
                    return self.failure_target
                # Dependency still pending/running
                return self.wait_target

        # All dependencies satisfied
        return self.success_target

    def get_possible_targets(self) -> List[str]:
        return [self.success_target, self.wait_target, self.failure_target]


class PipelineStageRouter(RoutingFunction):
    """Router for managing multi-stage pipeline execution."""

    def __init__(self, stage_map: Dict[str, str], default_target: str):
        """
        Initialize pipeline stage router.

        Parameters
        ----------
        stage_map : Dict[str, str]
            Mapping of pipeline stages to target nodes
        default_target : str
            Default target if stage not found
        """
        self.stage_map = stage_map
        self.default_target = default_target

    def __call__(self, context: AgentContext) -> str:
        """Route based on current pipeline stage."""
        current_stage = context.execution_state.get("pipeline_stage", "initial")
        return self.stage_map.get(current_stage, self.default_target)

    def get_possible_targets(self) -> List[str]:
        targets = list(self.stage_map.values())
        if self.default_target not in targets:
            targets.append(self.default_target)
        return targets


# Extended factory functions for failure handling


def route_with_failure_handling(
    success_target: str,
    failure_target: str,
    retry_target: Optional[str] = None,
    max_failures: int = 3,
) -> RoutingFunction:
    """Create a router with sophisticated failure handling."""
    return FailureHandlingRouter(
        success_target=success_target,
        failure_target=failure_target,
        retry_target=retry_target,
        max_failures=max_failures,
    )


def route_with_dependencies(
    target_dependencies: Dict[str, List[str]],
    success_target: str,
    wait_target: str = "wait",
    failure_target: str = "error",
) -> RoutingFunction:
    """Create a router that considers agent dependencies."""
    return AgentDependencyRouter(
        dependency_map=target_dependencies,
        success_target=success_target,
        wait_target=wait_target,
        failure_target=failure_target,
    )


def route_by_pipeline_stage(
    stage_routing: Dict[str, str], default_target: str = "end"
) -> RoutingFunction:
    """Create a router based on pipeline execution stages."""
    return PipelineStageRouter(stage_map=stage_routing, default_target=default_target)
