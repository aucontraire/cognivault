"""
Comprehensive test coverage for AgentRegistry missing scenarios.

This file addresses the remaining 25% coverage gaps in agent registry functionality,
focusing on error handling, edge cases, and advanced features.
"""

import pytest
from unittest.mock import Mock
from typing import Any, Optional

from cognivault.agents.registry import (
    AgentRegistry,
    get_agent_registry,
    register_agent,
    get_agent_metadata,
)
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.metadata import AgentMetadata
from cognivault.llm.llm_interface import LLMInterface
from cognivault.context import AgentContext
from cognivault.exceptions import (
    DependencyResolutionError,
    FailurePropagationStrategy,
)


class MockAgentWithName(BaseAgent):
    """Mock agent that accepts name parameter in constructor."""

    def __init__(self, name: str = "MockWithName") -> None:
        super().__init__(name)

    async def run(self, context: AgentContext) -> AgentContext:
        return context


class MockAgentWithLLMAndName(BaseAgent):
    """Mock agent that accepts both LLM and name parameters."""

    def __init__(self, llm: LLMInterface, name: str = "MockLLMWithName") -> None:
        super().__init__(name)
        self.llm = llm

    async def run(self, context: AgentContext) -> AgentContext:
        return context


class MockAgentWithKwargs(BaseAgent):
    """Mock agent that accepts additional kwargs."""

    def __init__(self, custom_param: str = "default", **kwargs: Any) -> None:
        super().__init__("MockWithKwargs")
        self.custom_param = custom_param

    async def run(self, context: AgentContext) -> AgentContext:
        return context


class TestAgentRegistryAdvancedFeatures:
    """Test advanced registry features and edge cases."""

    def test_create_agent_with_name_parameter_llm_required(self) -> None:
        """Test creating agent with name parameter when LLM is required (line 163)."""
        registry = AgentRegistry()
        mock_llm: Mock = Mock()

        # Register agent that accepts both llm and name
        registry.register(
            "agent_with_name_llm", MockAgentWithLLMAndName, requires_llm=True
        )

        agent = registry.create_agent_with_llm("agent_with_name_llm", llm=mock_llm)

        assert agent is not None
        # The registry passes the registered name to the agent constructor
        assert agent.name == "agent_with_name_llm"
        assert agent.llm == mock_llm

    def test_create_agent_with_name_parameter_no_llm(self) -> None:
        """Test creating agent with name parameter when LLM not required (line 168)."""
        registry = AgentRegistry()

        # Register agent that accepts name parameter
        registry.register("agent_with_name", MockAgentWithName, requires_llm=False)

        agent = registry.create_agent("agent_with_name")

        assert agent is not None
        # The registry passes the registered name to the agent constructor
        assert agent.name == "agent_with_name"

    def test_get_metadata_unknown_agent(self) -> None:
        """Test get_metadata with unknown agent (line 199)."""
        registry = AgentRegistry()

        with pytest.raises(ValueError, match="Unknown agent: 'nonexistent'"):
            registry.get_metadata("nonexistent")

    def test_validate_pipeline_dependency_resolution_error(self) -> None:
        """Test validate_pipeline when dependency resolution fails (lines 256-257)."""
        registry = AgentRegistry()

        # Register agents with circular dependencies
        registry.register(
            "agent_a", MockAgentWithName, requires_llm=False, dependencies=["agent_b"]
        )
        registry.register(
            "agent_b", MockAgentWithName, requires_llm=False, dependencies=["agent_a"]
        )

        # Should fail validation due to circular dependency
        result = registry.validate_pipeline(["agent_a", "agent_b"])
        assert result is False

    def test_resolve_dependencies_unknown_agent_in_graph(self) -> None:
        """Test resolve_dependencies with unknown agent creates empty deps (line 285)."""
        registry = AgentRegistry()

        # Create dependency resolution with unknown agent
        # Unknown agents should be handled by giving them empty dependencies
        # and should succeed unless there are circular dependencies
        result = registry.resolve_dependencies(["unknown_agent", "another_unknown"])

        # Should return the agents in some order (they have no dependencies)
        assert set(result) == {"unknown_agent", "another_unknown"}

    def test_resolve_dependencies_circular_dependency_detection(self) -> None:
        """Test circular dependency detection in resolve_dependencies (lines 292, 306, 311-312)."""
        registry = AgentRegistry()

        # Register agents with circular dependencies
        registry.register(
            "circular_a",
            MockAgentWithName,
            requires_llm=False,
            dependencies=["circular_b"],
        )
        registry.register(
            "circular_b",
            MockAgentWithName,
            requires_llm=False,
            dependencies=["circular_a"],
        )

        # Should raise DependencyResolutionError for circular dependency
        with pytest.raises(DependencyResolutionError) as exc_info:
            registry.resolve_dependencies(["circular_a", "circular_b"])

        error = exc_info.value
        assert error.dependency_issue == "Circular dependency detected"
        assert (
            "circular_a" in error.affected_agents
            or "circular_b" in error.affected_agents
        )

    def test_resolve_dependencies_successful_ordering(self) -> None:
        """Test successful dependency resolution and ordering."""
        registry = AgentRegistry()

        # Register agents with linear dependencies: c -> b -> a
        registry.register(
            "dep_a", MockAgentWithName, requires_llm=False, dependencies=[]
        )
        registry.register(
            "dep_b", MockAgentWithName, requires_llm=False, dependencies=["dep_a"]
        )
        registry.register(
            "dep_c", MockAgentWithName, requires_llm=False, dependencies=["dep_b"]
        )

        result = registry.resolve_dependencies(["dep_c", "dep_b", "dep_a"])

        # Should order correctly: a, then b, then c
        assert result == ["dep_a", "dep_b", "dep_c"]

    def test_check_health_unknown_agent(self) -> None:
        """Test check_health with unknown agent returns True (lines 334-335)."""
        registry = AgentRegistry()

        # Unknown agents should pass health check by default
        result = registry.check_health("unknown_agent")
        assert result is True

    def test_check_health_agent_requires_llm(self) -> None:
        """Test check_health for agent that requires LLM (lines 343-348)."""
        registry = AgentRegistry()

        # Test health check for agent that requires LLM
        result = registry.check_health("refiner")  # refiner requires LLM
        assert result is True  # Should pass basic health check

    def test_get_fallback_agents_unknown_agent(self) -> None:
        """Test get_fallback_agents with unknown agent (lines 364-365)."""
        registry = AgentRegistry()

        # Unknown agents should return empty fallback list
        fallbacks = registry.get_fallback_agents("unknown_agent")
        assert fallbacks == []

    def test_get_fallback_agents_with_fallbacks(self) -> None:
        """Test get_fallback_agents with configured fallbacks (lines 366-367)."""
        registry = AgentRegistry()

        # Register agent with fallback agents
        registry.register(
            "primary_agent",
            MockAgentWithName,
            requires_llm=False,
            fallback_agents=["fallback1", "fallback2"],
        )

        fallbacks = registry.get_fallback_agents("primary_agent")
        assert fallbacks == ["fallback1", "fallback2"]

        # Verify it returns a copy (not the original list)
        fallbacks.append("modified")
        original_fallbacks = registry.get_fallback_agents("primary_agent")
        assert "modified" not in original_fallbacks

    def test_get_failure_strategy_unknown_agent(self) -> None:
        """Test get_failure_strategy with unknown agent (lines 383-384)."""
        registry = AgentRegistry()

        # Unknown agents should return FAIL_FAST strategy
        strategy = registry.get_failure_strategy("unknown_agent")
        assert strategy == FailurePropagationStrategy.FAIL_FAST

    def test_get_failure_strategy_configured_agent(self) -> None:
        """Test get_failure_strategy with configured agent (line 385)."""
        registry = AgentRegistry()

        # Register agent with specific failure strategy
        registry.register(
            "graceful_agent",
            MockAgentWithName,
            requires_llm=False,
            failure_strategy=FailurePropagationStrategy.GRACEFUL_DEGRADATION,
        )

        strategy = registry.get_failure_strategy("graceful_agent")
        assert strategy == FailurePropagationStrategy.GRACEFUL_DEGRADATION

    def test_is_critical_agent_unknown_agent(self) -> None:
        """Test is_critical_agent with unknown agent (lines 401-402)."""
        registry = AgentRegistry()

        # Unknown agents should be considered critical by default
        result = registry.is_critical_agent("unknown_agent")
        assert result is True

    def test_is_critical_agent_configured_agent(self) -> None:
        """Test is_critical_agent with configured agent (line 403)."""
        registry = AgentRegistry()

        # Register non-critical agent
        registry.register(
            "optional_agent", MockAgentWithName, requires_llm=False, is_critical=False
        )

        result = registry.is_critical_agent("optional_agent")
        assert result is False

    def test_get_agent_metadata_exists(self) -> None:
        """Test get_agent_metadata with existing agent (line 419)."""
        registry = AgentRegistry()

        metadata = registry.get_agent_metadata("refiner")
        assert metadata is not None
        assert metadata.name == "refiner"
        assert isinstance(metadata, AgentMetadata)

    def test_get_agent_metadata_not_exists(self) -> None:
        """Test get_agent_metadata with non-existent agent."""
        registry = AgentRegistry()

        metadata = registry.get_agent_metadata("nonexistent")
        assert metadata is None

    def test_create_agent_with_additional_kwargs(self) -> None:
        """Test creating agent with additional kwargs passed through."""
        registry = AgentRegistry()

        # Register agent that accepts additional kwargs
        registry.register("agent_with_kwargs", MockAgentWithKwargs, requires_llm=False)

        agent = registry.create_agent("agent_with_kwargs", custom_param="test_value")

        assert agent is not None
        assert hasattr(agent, "custom_param")
        assert getattr(agent, "custom_param") == "test_value"

    def test_agent_creation_constructor_inspection_edge_cases(self) -> None:
        """Test edge cases in constructor parameter inspection."""
        registry = AgentRegistry()

        # Create an agent class with complex constructor
        class ComplexAgent(BaseAgent):
            def __init__(
                self,
                llm: Optional[LLMInterface] = None,
                name: str = "Complex",
                **kwargs: Any,
            ) -> None:
                super().__init__(name)
                self.llm = llm
                self.extra_params = kwargs

            async def run(self, context: AgentContext) -> AgentContext:
                return context

        registry.register("complex_agent", ComplexAgent, requires_llm=True)

        mock_llm: Mock = Mock()
        agent = registry.create_agent_with_llm(
            "complex_agent", llm=mock_llm, extra_param="value"
        )

        assert agent is not None
        assert agent.llm == mock_llm
        assert hasattr(agent, "extra_params")
        assert getattr(agent, "extra_params") == {"extra_param": "value"}


class TestGlobalRegistryFunctions:
    """Test global registry convenience functions."""

    def test_get_agent_metadata_global_function(self) -> None:
        """Test get_agent_metadata global function (lines 630-631)."""
        # Test with existing agent
        metadata = get_agent_metadata("refiner")
        assert metadata is not None
        assert metadata.name == "refiner"

        # Test with non-existent agent
        metadata = get_agent_metadata("nonexistent")
        assert metadata is None

    def test_global_registry_persistence(self) -> None:
        """Test that global registry persists across function calls."""
        # Register agent using global function
        register_agent("persistent_agent", MockAgentWithName, requires_llm=False)

        # Verify it persists in subsequent calls
        registry1 = get_agent_registry()
        registry2 = get_agent_registry()

        assert "persistent_agent" in registry1.get_available_agents()
        assert "persistent_agent" in registry2.get_available_agents()
        assert registry1 is registry2  # Should be the same instance

    def test_register_agent_with_all_parameters(self) -> None:
        """Test register_agent global function with all parameters."""
        register_agent(
            name="full_featured_agent",
            agent_class=MockAgentWithName,
            requires_llm=False,
            description="Agent with all features",
            dependencies=["refiner"],
            is_critical=False,
            failure_strategy=FailurePropagationStrategy.WARN_CONTINUE,
            fallback_agents=["backup_agent"],
            health_checks=["basic_check"],
            cognitive_speed="slow",
            cognitive_depth="deep",
            processing_pattern="composite",
            primary_capability="advanced_processing",
            secondary_capabilities=["analysis", "synthesis"],
            pipeline_role="intermediate",
            bounded_context="transformation",
        )

        registry = get_agent_registry()
        metadata = registry.get_agent_metadata("full_featured_agent")

        assert metadata is not None
        assert metadata.description == "Agent with all features"
        assert metadata.dependencies == ["refiner"]
        assert metadata.is_critical is False
        assert metadata.failure_strategy == FailurePropagationStrategy.WARN_CONTINUE
        assert metadata.fallback_agents == ["backup_agent"]
        assert metadata.cognitive_speed == "slow"
        assert metadata.cognitive_depth == "deep"
        assert metadata.processing_pattern == "composite"
        assert metadata.primary_capability == "advanced_processing"
        assert metadata.secondary_capabilities == ["analysis", "synthesis"]
        assert metadata.pipeline_role == "intermediate"
        assert metadata.bounded_context == "transformation"


class TestRegistryErrorScenarios:
    """Test error handling and edge cases."""

    def test_create_agent_creation_exception_details(self) -> None:
        """Test detailed exception handling during agent creation."""
        registry = AgentRegistry()

        class ProblemAgent(BaseAgent):
            def __init__(self) -> None:
                raise RuntimeError("Constructor failed with specific message")

            async def run(self, context: AgentContext) -> AgentContext:
                return context

        registry.register("problem_agent", ProblemAgent, requires_llm=False)

        with pytest.raises(ValueError) as exc_info:
            registry.create_agent("problem_agent")

        error_message = str(exc_info.value)
        assert "Failed to create agent 'problem_agent'" in error_message
        assert "Constructor failed with specific message" in error_message

    def test_dependency_resolution_with_partial_unknown_agents(self) -> None:
        """Test dependency resolution when some agents are unknown."""
        registry = AgentRegistry()

        # Mix of known and unknown agents
        agent_list = ["refiner", "unknown_agent", "historian"]

        # Should handle unknown agents gracefully by treating them as having no dependencies
        # But this particular combination should still work since refiner and historian exist
        try:
            result = registry.resolve_dependencies(agent_list)
            # If it succeeds, check the order makes sense
            assert "refiner" in result
            assert "historian" in result
        except DependencyResolutionError:
            # This is also acceptable behavior for unknown agents
            pass

    def test_validate_pipeline_mixed_valid_invalid(self) -> None:
        """Test validate_pipeline with mix of valid and invalid agents."""
        registry = AgentRegistry()

        # Pipeline with one valid and one invalid agent
        mixed_pipeline = ["refiner", "completely_unknown_agent"]
        result = registry.validate_pipeline(mixed_pipeline)

        assert result is False

    def test_agent_registration_edge_cases(self) -> None:
        """Test edge cases in agent registration."""
        registry = AgentRegistry()

        # Register agent with minimal parameters
        registry.register("minimal_agent", MockAgentWithName)

        metadata = registry.get_agent_metadata("minimal_agent")
        assert metadata is not None
        assert metadata.name == "minimal_agent"
        assert metadata.agent_class == MockAgentWithName
        assert metadata.requires_llm is False  # Default value
        assert metadata.description == ""  # Default value
        assert metadata.dependencies == []  # Default value
        assert metadata.is_critical is True  # Default value

    def test_multi_axis_classification_defaults(self) -> None:
        """Test multi-axis classification with default values."""
        registry = AgentRegistry()

        registry.register("classified_agent", MockAgentWithName)

        metadata = registry.get_agent_metadata("classified_agent")
        assert metadata is not None
        assert metadata.cognitive_speed == "adaptive"
        assert metadata.cognitive_depth == "variable"
        assert metadata.processing_pattern == "atomic"
        assert metadata.pipeline_role == "standalone"
        assert metadata.bounded_context == "reflection"
        # primary_capability defaults to the agent name when not specified
        assert metadata.primary_capability == "classified_agent"
        assert metadata.secondary_capabilities == []
