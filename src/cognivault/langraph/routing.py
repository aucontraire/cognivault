"""
Routing utilities for LangGraph conditional execution.

This module provides conditional routing functions and utilities
for building dynamic graph execution paths.
"""

from typing import Callable, Dict, Any, List
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
