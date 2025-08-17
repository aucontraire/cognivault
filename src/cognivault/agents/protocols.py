"""
Agent constructor protocols for type-safe agent registration and creation.

This module defines protocols that specify the expected constructor signatures
for different types of agents, enabling static type checking while eliminating
runtime introspection. These protocols replace the need for `inspect.signature()`
calls and associated `# type: ignore[call-arg]` usage.
"""

from typing import Protocol, Optional, Union, Any, TYPE_CHECKING
from enum import Enum


class AgentConstructorPattern(Enum):
    """Enumeration of agent constructor patterns for type-safe instantiation."""

    LLM_REQUIRED = "llm_required"  # RefinerAgent, CriticAgent
    LLM_OPTIONAL = "llm_optional"  # HistorianAgent, SynthesisAgent
    STANDARD = "standard"  # BaseAgent with name parameter
    FLEXIBLE = "flexible"  # Catch-all for unusual patterns


if TYPE_CHECKING:
    from cognivault.llm.llm_interface import LLMInterface
    from cognivault.config.agent_configs import (
        RefinerConfig,
        CriticConfig,
        HistorianConfig,
        SynthesisConfig,
    )
    from cognivault.agents.base_agent import BaseAgent


class LLMRequiredAgentProtocol(Protocol):
    """
    Protocol for agents that require an LLM interface as the first parameter.

    This protocol covers agents like RefinerAgent, CriticAgent that have:
    __init__(self, llm: LLMInterface, config: Optional[ConfigType] = None)
    """

    def __init__(
        self, llm: "LLMInterface", config: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """Initialize agent with required LLM interface."""
        ...


class LLMOptionalAgentProtocol(Protocol):
    """
    Protocol for agents that accept an optional LLM interface.

    This protocol covers agents like HistorianAgent, SynthesisAgent that have:
    __init__(self, llm: Optional[Union[LLMInterface, str]] = "default", ...)
    """

    def __init__(
        self, llm: Optional[Union["LLMInterface", str]] = "default", **kwargs: Any
    ) -> None:
        """Initialize agent with optional LLM interface."""
        ...


class StandardAgentProtocol(Protocol):
    """
    Protocol for agents that follow the standard BaseAgent pattern.

    This protocol covers BaseAgent and potential future agents that have:
    __init__(self, name: str, **kwargs: Any)
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize agent with name and optional parameters."""
        ...


class FlexibleAgentProtocol(Protocol):
    """
    Protocol for agents with completely flexible constructor signatures.

    This is a fallback protocol for any agent that doesn't fit the other patterns.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize agent with flexible parameters."""
        ...


# Agent-specific protocols for more precise type checking
class RefinerAgentProtocol(Protocol):
    """Specific protocol for RefinerAgent constructor."""

    def __init__(
        self, llm: "LLMInterface", config: Optional["RefinerConfig"] = None
    ) -> None:
        """Initialize RefinerAgent with LLM and optional config."""
        ...


class CriticAgentProtocol(Protocol):
    """Specific protocol for CriticAgent constructor."""

    def __init__(
        self, llm: "LLMInterface", config: Optional["CriticConfig"] = None
    ) -> None:
        """Initialize CriticAgent with LLM and optional config."""
        ...


class HistorianAgentProtocol(Protocol):
    """Specific protocol for HistorianAgent constructor."""

    def __init__(
        self,
        llm: Optional[Union["LLMInterface", str]] = "default",
        search_type: str = "hybrid",
        config: Optional["HistorianConfig"] = None,
    ) -> None:
        """Initialize HistorianAgent with optional LLM and search config."""
        ...


class SynthesisAgentProtocol(Protocol):
    """Specific protocol for SynthesisAgent constructor."""

    def __init__(
        self,
        llm: Optional[Union["LLMInterface", str]] = "default",
        config: Optional["SynthesisConfig"] = None,
    ) -> None:
        """Initialize SynthesisAgent with optional LLM and config."""
        ...


# Protocol for agents that have LLM attributes (for type checking)
class AgentWithLLMProtocol(Protocol):
    """
    Protocol for agents that have an 'llm' attribute.

    This protocol enables type checking for tests and code that accesses
    the 'llm' attribute on agents without requiring casting to specific types.

    Covers RefinerAgent, CriticAgent, HistorianAgent, and SynthesisAgent.
    """

    llm: Optional["LLMInterface"]
    name: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize agent."""
        ...


# Type aliases for convenience
AgentConstructorProtocol = Union[
    LLMRequiredAgentProtocol,
    LLMOptionalAgentProtocol,
    StandardAgentProtocol,
    FlexibleAgentProtocol,
]

SpecificAgentProtocol = Union[
    RefinerAgentProtocol,
    CriticAgentProtocol,
    HistorianAgentProtocol,
    SynthesisAgentProtocol,
]
