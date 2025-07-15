"""
Agent Registry for dynamic agent registration and creation.

This module provides a centralized registry for managing agent types,
their dependencies, and creation logic. It enables dynamic agent loading
while maintaining type safety and proper dependency injection.
"""

from typing import Dict, Type, Optional, List, Literal
from cognivault.agents.base_agent import BaseAgent
from cognivault.llm.llm_interface import LLMInterface
from cognivault.exceptions import (
    DependencyResolutionError,
    FailurePropagationStrategy,
)
from cognivault.agents.metadata import AgentMetadata


class AgentRegistry:
    """
    Central registry for managing agent types and creation.

    This registry provides a clean abstraction for agent management,
    supporting both the current architecture and future dynamic loading
    capabilities (e.g., LangGraph integration).
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentMetadata] = {}
        self._register_core_agents()

    def register(
        self,
        name: str,
        agent_class: Type[BaseAgent],
        requires_llm: bool = False,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        is_critical: bool = True,
        failure_strategy: FailurePropagationStrategy = FailurePropagationStrategy.FAIL_FAST,
        fallback_agents: Optional[List[str]] = None,
        health_checks: Optional[List[str]] = None,
        # New multi-axis classification parameters
        cognitive_speed: Literal["fast", "slow", "adaptive"] = "adaptive",
        cognitive_depth: Literal["shallow", "deep", "variable"] = "variable",
        processing_pattern: Literal["atomic", "composite", "chain"] = "atomic",
        primary_capability: str = "",
        secondary_capabilities: Optional[List[str]] = None,
        pipeline_role: Literal[
            "entry", "intermediate", "terminal", "standalone"
        ] = "standalone",
        bounded_context: str = "reflection",
    ) -> None:
        """
        Register an agent type with the registry.

        Parameters
        ----------
        name : str
            Unique name for the agent
        agent_class : Type[BaseAgent]
            The agent class to register
        requires_llm : bool, optional
            Whether this agent requires an LLM interface
        description : str, optional
            Human-readable description of the agent
        dependencies : List[str], optional
            List of agent names this agent depends on
        is_critical : bool, optional
            Whether agent failure should stop the pipeline
        failure_strategy : FailurePropagationStrategy, optional
            How to handle failures from this agent
        fallback_agents : List[str], optional
            Alternative agents to try if this one fails
        health_checks : List[str], optional
            Health check functions to run before executing
        cognitive_speed : str, optional
            Agent cognitive speed: "fast", "slow", "adaptive"
        cognitive_depth : str, optional
            Agent cognitive depth: "shallow", "deep", "variable"
        processing_pattern : str, optional
            Processing pattern: "atomic", "composite", "chain"
        primary_capability : str, optional
            Primary capability (e.g., "critical_analysis", "translation")
        secondary_capabilities : List[str], optional
            Additional capabilities this agent provides
        pipeline_role : str, optional
            Role in pipeline: "entry", "intermediate", "terminal", "standalone"
        bounded_context : str, optional
            Bounded context: "reflection", "transformation", "retrieval"
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered")

        metadata = AgentMetadata(
            name=name,
            agent_class=agent_class,
            requires_llm=requires_llm,
            description=description,
            dependencies=dependencies or [],
            is_critical=is_critical,
            failure_strategy=failure_strategy,
            fallback_agents=fallback_agents or [],
            health_checks=health_checks or [],
            # Multi-axis classification
            cognitive_speed=cognitive_speed,
            cognitive_depth=cognitive_depth,
            processing_pattern=processing_pattern,
            primary_capability=primary_capability,
            secondary_capabilities=secondary_capabilities or [],
            pipeline_role=pipeline_role,
            bounded_context=bounded_context,
        )
        self._agents[name] = metadata

    def create_agent(
        self, name: str, llm: Optional[LLMInterface] = None, **kwargs
    ) -> BaseAgent:
        """
        Create an agent instance by name.

        Parameters
        ----------
        name : str
            Name of the agent to create
        llm : LLMInterface, optional
            LLM interface for agents that require it
        **kwargs
            Additional keyword arguments for agent construction

        Returns
        -------
        BaseAgent
            Configured agent instance

        Raises
        ------
        ValueError
            If agent name is not registered or required dependencies are missing
        """
        if name not in self._agents:
            raise ValueError(
                f"Unknown agent: '{name}'. Available agents: {list(self.get_available_agents())}"
            )

        metadata = self._agents[name]

        # Check LLM requirement
        if metadata.requires_llm and llm is None:
            raise ValueError(f"Agent '{name}' requires an LLM interface")

        # Create agent with appropriate parameters
        try:
            # For agents that support configurable names, try to pass the name
            import inspect

            constructor_params = inspect.signature(
                metadata.agent_class.__init__
            ).parameters

            if metadata.requires_llm:
                if "name" in constructor_params:
                    return metadata.agent_class(llm=llm, name=name, **kwargs)  # type: ignore
                else:
                    return metadata.agent_class(llm=llm, **kwargs)  # type: ignore
            else:
                if "name" in constructor_params:
                    return metadata.agent_class(name=name, **kwargs)
                else:
                    return metadata.agent_class(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create agent '{name}': {e}") from e

    def get_available_agents(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self._agents.keys())

    def get_metadata(self, name: str) -> AgentMetadata:
        """
        Get metadata for a specific agent.

        Parameters
        ----------
        name : str
            Name of the agent

        Returns
        -------
        AgentMetadata
            Agent metadata

        Raises
        ------
        ValueError
            If agent name is not registered
        """
        if name not in self._agents:
            raise ValueError(f"Unknown agent: '{name}'")
        return self._agents[name]

    def get_agent_info(self, name: str) -> AgentMetadata:
        """
        Get metadata for a specific agent.

        Parameters
        ----------
        name : str
            Name of the agent

        Returns
        -------
        AgentMetadata
            Agent metadata

        Raises
        ------
        ValueError
            If agent name is not registered
        """
        if name not in self._agents:
            raise ValueError(f"Unknown agent: '{name}'")
        return self._agents[name]

    def get_agents_requiring_llm(self) -> List[str]:
        """Get list of agent names that require an LLM interface."""
        return [
            name for name, metadata in self._agents.items() if metadata.requires_llm
        ]

    def validate_pipeline(self, agent_names: List[str]) -> bool:
        """
        Validate that a pipeline of agents can be executed.

        Performs comprehensive validation including dependency checking
        and circular dependency detection.

        Parameters
        ----------
        agent_names : List[str]
            List of agent names in execution order

        Returns
        -------
        bool
            True if pipeline is valid, False otherwise
        """
        # Check that all agents are registered
        for name in agent_names:
            if name not in self._agents:
                return False

        # Check dependency resolution
        try:
            self.resolve_dependencies(agent_names)
            return True
        except DependencyResolutionError:
            return False

    def resolve_dependencies(self, agent_names: List[str]) -> List[str]:
        """
        Resolve agent dependencies and return optimal execution order.

        Parameters
        ----------
        agent_names : List[str]
            List of agent names to resolve

        Returns
        -------
        List[str]
            Agent names in dependency-resolved execution order

        Raises
        ------
        DependencyResolutionError
            If dependencies cannot be resolved or circular dependencies exist
        """
        # Build dependency graph
        dependency_graph = {}
        for name in agent_names:
            if name in self._agents:
                deps = self._agents[name].dependencies
                dependency_graph[name] = deps.copy() if deps is not None else []
            else:
                dependency_graph[name] = []

        # Topological sort using Kahn's algorithm
        # Calculate in-degrees: if agent A depends on agent B, then A has incoming edge from B
        in_degree = {name: 0 for name in agent_names}
        for name in agent_names:
            for dep in dependency_graph[name]:
                if dep in in_degree:
                    in_degree[name] += 1

        queue = [name for name in agent_names if in_degree[name] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Update in-degree for agents that depend on current
            for neighbor in agent_names:
                if current in dependency_graph[neighbor]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Check for circular dependencies
        if len(result) != len(agent_names):
            remaining = [name for name in agent_names if name not in result]
            raise DependencyResolutionError(
                dependency_issue="Circular dependency detected",
                affected_agents=remaining,
                dependency_graph=dependency_graph,
            )

        return result

    def check_health(self, agent_name: str) -> bool:
        """
        Check if an agent passes its health checks.

        Parameters
        ----------
        agent_name : str
            Name of the agent to check

        Returns
        -------
        bool
            True if all health checks pass
        """
        if agent_name not in self._agents:
            return True  # Unknown agents pass health check by default

        metadata = self._agents[agent_name]

        # For now, basic health checks - can be extended with actual health check functions
        # health_checks = metadata.health_checks

        # Basic checks: LLM requirement validation
        if metadata.requires_llm:
            # Would check LLM connectivity here
            pass

        # All health checks pass
        return True

    def get_fallback_agents(self, agent_name: str) -> List[str]:
        """
        Get fallback agents for a given agent.

        Parameters
        ----------
        agent_name : str
            Name of the agent that failed

        Returns
        -------
        List[str]
            List of fallback agent names
        """
        if agent_name not in self._agents:
            return []
        fallback = self._agents[agent_name].fallback_agents
        return fallback.copy() if fallback is not None else []

    def get_failure_strategy(self, agent_name: str) -> FailurePropagationStrategy:
        """
        Get failure propagation strategy for an agent.

        Parameters
        ----------
        agent_name : str
            Name of the agent

        Returns
        -------
        FailurePropagationStrategy
            The failure strategy for this agent
        """
        if agent_name not in self._agents:
            return FailurePropagationStrategy.FAIL_FAST
        return self._agents[agent_name].failure_strategy

    def is_critical_agent(self, agent_name: str) -> bool:
        """
        Check if an agent is critical to the pipeline.

        Parameters
        ----------
        agent_name : str
            Name of the agent

        Returns
        -------
        bool
            True if the agent is critical
        """
        if agent_name not in self._agents:
            return True  # Unknown agents are considered critical
        return self._agents[agent_name].is_critical

    def get_agent_metadata(self, agent_name: str) -> Optional[AgentMetadata]:
        """
        Get the full metadata for an agent.

        Parameters
        ----------
        agent_name : str
            Name of the agent

        Returns
        -------
        AgentMetadata, optional
            Complete metadata for the agent, or None if not found
        """
        return self._agents.get(agent_name)

    def _register_core_agents(self) -> None:
        """Register the core agents that ship with CogniVault with conditional execution support."""
        # Import here to avoid circular imports
        from cognivault.agents.refiner.agent import RefinerAgent
        from cognivault.agents.critic.agent import CriticAgent
        from cognivault.agents.historian.agent import HistorianAgent
        from cognivault.agents.synthesis.agent import SynthesisAgent

        # Register core agents with failure propagation strategies and multi-axis classification
        self.register(
            name="refiner",
            agent_class=RefinerAgent,
            requires_llm=True,
            description="Refines and improves user queries for better processing",
            dependencies=[],
            is_critical=True,  # Refiner failure is critical
            failure_strategy=FailurePropagationStrategy.FAIL_FAST,
            fallback_agents=[],  # No fallback - query refinement is essential
            # Multi-axis classification
            cognitive_speed="slow",
            cognitive_depth="deep",
            processing_pattern="atomic",
            primary_capability="intent_clarification",
            secondary_capabilities=["prompt_structuring", "scope_definition"],
            pipeline_role="entry",
            bounded_context="reflection",
        )

        self.register(
            name="critic",
            agent_class=CriticAgent,
            requires_llm=True,
            description="Analyzes refined queries to identify assumptions, gaps, and biases",
            dependencies=["refiner"],  # Critic processes RefinerAgent output
            is_critical=False,  # Critic can be skipped if it fails
            failure_strategy=FailurePropagationStrategy.GRACEFUL_DEGRADATION,
            fallback_agents=[],  # No direct fallback, but can be skipped
            # Multi-axis classification
            cognitive_speed="slow",
            cognitive_depth="deep",
            processing_pattern="composite",
            primary_capability="critical_analysis",
            secondary_capabilities=["assumption_identification", "bias_detection"],
            pipeline_role="intermediate",
            bounded_context="reflection",
        )

        self.register(
            name="historian",
            agent_class=HistorianAgent,
            requires_llm=False,
            description="Retrieves relevant historical context and information",
            dependencies=[],
            is_critical=False,  # Historian is helpful but not essential
            failure_strategy=FailurePropagationStrategy.WARN_CONTINUE,
            fallback_agents=[],  # No fallback needed for mock historical data
            # Multi-axis classification
            cognitive_speed="adaptive",
            cognitive_depth="variable",
            processing_pattern="composite",
            primary_capability="context_retrieval",
            secondary_capabilities=["memory_search", "relevance_ranking"],
            pipeline_role="intermediate",
            bounded_context="retrieval",
        )

        self.register(
            name="synthesis",
            agent_class=SynthesisAgent,
            requires_llm=False,
            description="Synthesizes outputs from multiple agents into final response",
            dependencies=[],  # Synthesis can work with any combination of agents
            is_critical=True,  # Synthesis is needed for final output
            failure_strategy=FailurePropagationStrategy.CONDITIONAL_FALLBACK,
            fallback_agents=[],  # Could fallback to simple concatenation
            # Multi-axis classification
            cognitive_speed="slow",
            cognitive_depth="deep",
            processing_pattern="chain",
            primary_capability="multi_perspective_synthesis",
            secondary_capabilities=["conflict_resolution", "theme_identification"],
            pipeline_role="terminal",
            bounded_context="reflection",
        )


# Global registry instance
_global_registry = None


def get_agent_registry() -> AgentRegistry:
    """
    Get the global agent registry instance.

    Returns
    -------
    AgentRegistry
        Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(
    name: str,
    agent_class: Type[BaseAgent],
    requires_llm: bool = False,
    description: str = "",
    dependencies: Optional[List[str]] = None,
    is_critical: bool = True,
    failure_strategy: FailurePropagationStrategy = FailurePropagationStrategy.FAIL_FAST,
    fallback_agents: Optional[List[str]] = None,
    health_checks: Optional[List[str]] = None,
    # New multi-axis classification parameters
    cognitive_speed: Literal["fast", "slow", "adaptive"] = "adaptive",
    cognitive_depth: Literal["shallow", "deep", "variable"] = "variable",
    processing_pattern: Literal["atomic", "composite", "chain"] = "atomic",
    primary_capability: str = "",
    secondary_capabilities: Optional[List[str]] = None,
    pipeline_role: Literal[
        "entry", "intermediate", "terminal", "standalone"
    ] = "standalone",
    bounded_context: str = "reflection",
) -> None:
    """
    Register an agent with the global registry.

    This is a convenience function for registering custom agents.

    Parameters
    ----------
    name : str
        Unique name for the agent
    agent_class : Type[BaseAgent]
        The agent class to register
    requires_llm : bool, optional
        Whether this agent requires an LLM interface
    description : str, optional
        Human-readable description of the agent
    dependencies : List[str], optional
        List of agent names this agent depends on
    is_critical : bool, optional
        Whether agent failure should stop the pipeline
    failure_strategy : FailurePropagationStrategy, optional
        How to handle failures from this agent
    fallback_agents : List[str], optional
        Alternative agents to try if this one fails
    health_checks : List[str], optional
        Health check functions to run before executing
    """
    registry = get_agent_registry()
    registry.register(
        name,
        agent_class,
        requires_llm,
        description,
        dependencies,
        is_critical,
        failure_strategy,
        fallback_agents,
        health_checks,
        cognitive_speed,
        cognitive_depth,
        processing_pattern,
        primary_capability,
        secondary_capabilities,
        pipeline_role,
        bounded_context,
    )


def create_agent(name: str, llm: Optional[LLMInterface] = None, **kwargs) -> BaseAgent:
    """
    Create an agent using the global registry.

    Parameters
    ----------
    name : str
        Name of the agent to create
    llm : LLMInterface, optional
        LLM interface for agents that require it
    **kwargs
        Additional keyword arguments for agent construction

    Returns
    -------
    BaseAgent
        Configured agent instance
    """
    registry = get_agent_registry()
    return registry.create_agent(name, llm, **kwargs)


def get_agent_metadata(name: str) -> Optional[AgentMetadata]:
    """
    Get agent metadata using the global registry.

    Parameters
    ----------
    name : str
        Name of the agent

    Returns
    -------
    AgentMetadata, optional
        Complete metadata for the agent, or None if not found
    """
    registry = get_agent_registry()
    return registry.get_agent_metadata(name)
