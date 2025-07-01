"""
Agent Registry for dynamic agent registration and creation.

This module provides a centralized registry for managing agent types,
their dependencies, and creation logic. It enables dynamic agent loading
while maintaining type safety and proper dependency injection.
"""

from typing import Dict, Type, Optional, List
from dataclasses import dataclass
from cognivault.agents.base_agent import BaseAgent
from cognivault.llm.llm_interface import LLMInterface


@dataclass
class AgentMetadata:
    """Metadata for registered agents."""

    name: str
    agent_class: Type[BaseAgent]
    requires_llm: bool = False
    description: str = ""
    dependencies: Optional[List[str]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


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
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered")

        metadata = AgentMetadata(
            name=name,
            agent_class=agent_class,
            requires_llm=requires_llm,
            description=description,
            dependencies=dependencies or [],
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
            if metadata.requires_llm:
                return metadata.agent_class(llm=llm, **kwargs)  # type: ignore
            else:
                return metadata.agent_class(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create agent '{name}': {e}") from e

    def get_available_agents(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self._agents.keys())

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

        Currently, performs basic validation. Can be extended for
        dependency checking in future versions.

        Parameters
        ----------
        agent_names : List[str]
            List of agent names in execution order

        Returns
        -------
        bool
            True if pipeline is valid
        """
        # Check that all agents are registered
        for name in agent_names:
            if name not in self._agents:
                return False

        # Future: Add dependency resolution validation
        return True

    def _register_core_agents(self) -> None:
        """Register the core agents that ship with CogniVault."""
        # Import here to avoid circular imports
        from cognivault.agents.refiner.agent import RefinerAgent
        from cognivault.agents.critic.agent import CriticAgent
        from cognivault.agents.historian.agent import HistorianAgent
        from cognivault.agents.synthesis.agent import SynthesisAgent

        # Register core agents
        self.register(
            name="refiner",
            agent_class=RefinerAgent,
            requires_llm=True,
            description="Refines and improves user queries for better processing",
            dependencies=[],
        )

        self.register(
            name="critic",
            agent_class=CriticAgent,
            requires_llm=True,
            description="Analyzes refined queries to identify assumptions, gaps, and biases",
            dependencies=["refiner"],  # Critic typically processes RefinerAgent output
        )

        self.register(
            name="historian",
            agent_class=HistorianAgent,
            requires_llm=False,
            description="Retrieves relevant historical context and information",
            dependencies=[],
        )

        self.register(
            name="synthesis",
            agent_class=SynthesisAgent,
            requires_llm=False,
            description="Synthesizes outputs from multiple agents into final response",
            dependencies=[],  # Synthesis can work with any combination of agents
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
    """
    registry = get_agent_registry()
    registry.register(name, agent_class, requires_llm, description, dependencies)


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
