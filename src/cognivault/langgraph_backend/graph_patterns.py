"""
Graph patterns for CogniVault LangGraph backend.

This module defines different graph structures and patterns that can be used
for agent execution. Each pattern defines how agents are connected and the
execution flow between them.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GraphEdge:
    """Represents an edge in the graph."""

    from_node: str
    to_node: str
    edge_type: str = "standard"  # standard, conditional, parallel


class GraphPattern(ABC):
    """
    Abstract base class for graph patterns.

    Graph patterns define the structure and execution flow for different
    agent combinations and use cases.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Pattern name identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable pattern description."""
        pass

    @abstractmethod
    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        """
        Get edge definitions for the given agents.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the graph

        Returns
        -------
        List[Dict[str, str]]
            List of edge dictionaries with 'from' and 'to' keys
        """
        pass

    @abstractmethod
    def get_entry_point(self, agents: List[str]) -> Optional[str]:
        """
        Get the entry point agent for the graph.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the graph

        Returns
        -------
        Optional[str]
            Name of the entry point agent, or None if no specific entry point
        """
        pass

    @abstractmethod
    def get_exit_points(self, agents: List[str]) -> List[str]:
        """
        Get the exit point agents for the graph.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the graph

        Returns
        -------
        List[str]
            List of exit point agent names
        """
        pass

    def validate_agents(self, agents: List[str]) -> bool:
        """
        Validate that the given agents are compatible with this pattern.

        Parameters
        ----------
        agents : List[str]
            List of agent names to validate

        Returns
        -------
        bool
            True if agents are compatible with this pattern
        """
        return True  # Default: accept any agents

    def get_parallel_groups(self, agents: List[str]) -> List[List[str]]:
        """
        Get groups of agents that can execute in parallel.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the graph

        Returns
        -------
        List[List[str]]
            List of agent groups that can execute in parallel
        """
        return []  # Default: no parallel groups


class StandardPattern(GraphPattern):
    """
    Standard 4-agent pattern: refiner → [critic, historian] → synthesis

    This is the default CogniVault pattern where:
    1. Refiner processes the initial query
    2. Critic and Historian execute in parallel after Refiner
    3. Synthesis integrates all outputs for final analysis
    """

    @property
    def name(self) -> str:
        return "standard"

    @property
    def description(self) -> str:
        return "Standard 4-agent pattern: refiner → [critic, historian] → synthesis"

    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        """Get standard pattern edges."""
        edges = []
        agents_lower = [agent.lower() for agent in agents]

        # Define the standard DAG structure
        if "refiner" in agents_lower:
            # Refiner to Critic and Historian (parallel)
            if "critic" in agents_lower:
                edges.append({"from": "refiner", "to": "critic"})
            if "historian" in agents_lower:
                edges.append({"from": "refiner", "to": "historian"})

            # If synthesis is present, both critic and historian feed into it
            if "synthesis" in agents_lower:
                if "critic" in agents_lower:
                    edges.append({"from": "critic", "to": "synthesis"})
                if "historian" in agents_lower:
                    edges.append({"from": "historian", "to": "synthesis"})

                # If no critic or historian, refiner connects directly to synthesis
                if not ("critic" in agents_lower or "historian" in agents_lower):
                    edges.append({"from": "refiner", "to": "synthesis"})

                # Synthesis is the final node
                edges.append({"from": "synthesis", "to": "END"})
            else:
                # If no synthesis, critic and historian are terminal nodes
                if "critic" in agents_lower:
                    edges.append({"from": "critic", "to": "END"})
                if "historian" in agents_lower:
                    edges.append({"from": "historian", "to": "END"})

        # Handle edge cases with missing refiner
        elif "critic" in agents_lower or "historian" in agents_lower:
            # If no refiner, critic and/or historian are entry points
            if "synthesis" in agents_lower:
                if "critic" in agents_lower:
                    edges.append({"from": "critic", "to": "synthesis"})
                if "historian" in agents_lower:
                    edges.append({"from": "historian", "to": "synthesis"})
                edges.append({"from": "synthesis", "to": "END"})
            else:
                # Terminal nodes
                if "critic" in agents_lower:
                    edges.append({"from": "critic", "to": "END"})
                if "historian" in agents_lower:
                    edges.append({"from": "historian", "to": "END"})

        # Handle synthesis-only case
        elif "synthesis" in agents_lower:
            edges.append({"from": "synthesis", "to": "END"})

        return edges

    def get_entry_point(self, agents: List[str]) -> Optional[str]:
        """Get entry point for standard pattern."""
        agents_lower = [agent.lower() for agent in agents]

        # Refiner is preferred entry point
        if "refiner" in agents_lower:
            return "refiner"

        # If no refiner, use first available agent
        if agents_lower:
            return agents_lower[0]

        return None

    def get_exit_points(self, agents: List[str]) -> List[str]:
        """Get exit points for standard pattern."""
        agents_lower = [agent.lower() for agent in agents]

        # Synthesis is preferred exit point
        if "synthesis" in agents_lower:
            return ["synthesis"]

        # Otherwise, critic and historian are exit points
        exit_points = []
        if "critic" in agents_lower:
            exit_points.append("critic")
        if "historian" in agents_lower:
            exit_points.append("historian")

        return exit_points if exit_points else agents_lower

    def get_parallel_groups(self, agents: List[str]) -> List[List[str]]:
        """Get parallel execution groups."""
        agents_lower = [agent.lower() for agent in agents]

        # Critic and Historian can execute in parallel after Refiner
        parallel_group = []
        if "critic" in agents_lower:
            parallel_group.append("critic")
        if "historian" in agents_lower:
            parallel_group.append("historian")

        return [parallel_group] if len(parallel_group) > 1 else []


class ParallelPattern(GraphPattern):
    """
    Parallel pattern: Maximum parallelization where dependencies allow.

    This pattern attempts to execute as many agents in parallel as possible,
    respecting only essential dependencies.
    """

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def description(self) -> str:
        return "Maximum parallelization pattern with minimal dependencies"

    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        """Get parallel pattern edges with minimal dependencies."""
        edges = []
        agents_lower = [agent.lower() for agent in agents]

        # In parallel pattern, most agents can run independently
        # Only synthesis depends on outputs from others
        if "synthesis" in agents_lower:
            # All other agents feed into synthesis
            for agent in agents_lower:
                if agent != "synthesis":
                    edges.append({"from": agent, "to": "synthesis"})
            edges.append({"from": "synthesis", "to": "END"})
        else:
            # All agents are terminal if no synthesis
            for agent in agents_lower:
                edges.append({"from": agent, "to": "END"})

        return edges

    def get_entry_point(self, agents: List[str]) -> Optional[str]:
        """No single entry point in parallel pattern."""
        # In parallel pattern, we don't set a single entry point
        # to allow maximum parallelization
        return None

    def get_exit_points(self, agents: List[str]) -> List[str]:
        """Get exit points for parallel pattern."""
        agents_lower = [agent.lower() for agent in agents]

        if "synthesis" in agents_lower:
            return ["synthesis"]

        # All agents are exit points if no synthesis
        return agents_lower

    def get_parallel_groups(self, agents: List[str]) -> List[List[str]]:
        """Get parallel execution groups."""
        agents_lower = [agent.lower() for agent in agents]

        # All non-synthesis agents can run in parallel
        parallel_agents = [agent for agent in agents_lower if agent != "synthesis"]
        return [parallel_agents] if len(parallel_agents) > 1 else []


class ConditionalPattern(GraphPattern):
    """
    Conditional pattern: Dynamic routing based on agent outputs.

    This pattern supports conditional execution where the next agent
    to execute depends on the output of previous agents.
    """

    @property
    def name(self) -> str:
        return "conditional"

    @property
    def description(self) -> str:
        return "Conditional routing pattern with dynamic execution flow"

    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        """Get conditional pattern edges."""
        # For now, implement as standard pattern
        # Future enhancement: add conditional routing logic
        standard_pattern = StandardPattern()
        return standard_pattern.get_edges(agents)

    def get_entry_point(self, agents: List[str]) -> Optional[str]:
        """Get entry point for conditional pattern."""
        agents_lower = [agent.lower() for agent in agents]
        return (
            "refiner"
            if "refiner" in agents_lower
            else (agents_lower[0] if agents_lower else None)
        )

    def get_exit_points(self, agents: List[str]) -> List[str]:
        """Get exit points for conditional pattern."""
        agents_lower = [agent.lower() for agent in agents]
        return ["synthesis"] if "synthesis" in agents_lower else agents_lower


class PatternRegistry:
    """
    Registry for managing available graph patterns.

    This class provides a centralized way to register, retrieve, and
    manage different graph patterns.
    """

    def __init__(self) -> None:
        """Initialize the pattern registry with default patterns."""
        self._patterns: Dict[str, GraphPattern] = {}

        # Register default patterns
        self.register_pattern(StandardPattern())
        self.register_pattern(ParallelPattern())
        self.register_pattern(ConditionalPattern())

    def register_pattern(self, pattern: GraphPattern) -> None:
        """
        Register a new graph pattern.

        Parameters
        ----------
        pattern : GraphPattern
            Pattern to register
        """
        self._patterns[pattern.name] = pattern

    def get_pattern(self, name: str) -> Optional[GraphPattern]:
        """
        Get a pattern by name.

        Parameters
        ----------
        name : str
            Pattern name

        Returns
        -------
        Optional[GraphPattern]
            Pattern instance or None if not found
        """
        return self._patterns.get(name)

    def get_pattern_names(self) -> List[str]:
        """
        Get all registered pattern names.

        Returns
        -------
        List[str]
            List of pattern names
        """
        return list(self._patterns.keys())

    def list_patterns(self) -> Dict[str, str]:
        """
        Get all patterns with their descriptions.

        Returns
        -------
        Dict[str, str]
            Mapping of pattern names to descriptions
        """
        return {name: pattern.description for name, pattern in self._patterns.items()}

    def remove_pattern(self, name: str) -> bool:
        """
        Remove a pattern from the registry.

        Parameters
        ----------
        name : str
            Pattern name to remove

        Returns
        -------
        bool
            True if pattern was removed, False if not found
        """
        if name in self._patterns:
            del self._patterns[name]
            return True
        return False
