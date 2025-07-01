"""Tests for the Agent Registry system."""

import pytest
from unittest.mock import MagicMock
from cognivault.agents.registry import (
    AgentRegistry,
    get_agent_registry,
    register_agent,
    create_agent,
)
from cognivault.agents.base_agent import BaseAgent
from cognivault.llm.llm_interface import LLMInterface, LLMResponse
from cognivault.context import AgentContext


class MockAgentWithLLM(BaseAgent):
    """Mock agent that requires LLM for testing."""

    def __init__(self, llm: LLMInterface):
        super().__init__("MockLLM")
        self.llm = llm

    async def run(self, context: AgentContext) -> AgentContext:
        return context


class MockAgentWithoutLLM(BaseAgent):
    """Mock agent that doesn't require LLM for testing."""

    def __init__(self):
        super().__init__("MockNoLLM")

    async def run(self, context: AgentContext) -> AgentContext:
        return context


def create_mock_llm() -> MagicMock:
    """Create a mock LLM interface for testing."""
    mock_llm = MagicMock()
    mock_response = LLMResponse(
        text="Mock response",
        tokens_used=10,
        model_name="test-model",
        finish_reason="stop",
    )
    mock_llm.generate.return_value = mock_response
    return mock_llm


class TestAgentRegistry:
    """Test suite for AgentRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes with core agents."""
        registry = AgentRegistry()

        # Check that core agents are registered
        available_agents = registry.get_available_agents()
        assert "refiner" in available_agents
        assert "critic" in available_agents
        assert "historian" in available_agents
        assert "synthesis" in available_agents

        # Should have exactly the core agents
        assert len(available_agents) == 4

    def test_register_new_agent(self):
        """Test registering a new agent."""
        registry = AgentRegistry()

        # Register a new agent
        registry.register(
            name="test_agent",
            agent_class=MockAgentWithoutLLM,
            requires_llm=False,
            description="Test agent for unit tests",
        )

        # Verify it was registered
        available_agents = registry.get_available_agents()
        assert "test_agent" in available_agents

        # Verify metadata
        metadata = registry.get_agent_info("test_agent")
        assert metadata.name == "test_agent"
        assert metadata.agent_class == MockAgentWithoutLLM
        assert metadata.requires_llm is False
        assert metadata.description == "Test agent for unit tests"

    def test_register_duplicate_agent_raises_error(self):
        """Test that registering duplicate agent name raises error."""
        registry = AgentRegistry()

        # Try to register agent with existing name
        with pytest.raises(ValueError, match="Agent 'refiner' is already registered"):
            registry.register("refiner", MockAgentWithoutLLM)

    def test_create_agent_without_llm(self):
        """Test creating agent that doesn't require LLM."""
        registry = AgentRegistry()

        # Create historian agent (doesn't require LLM)
        agent = registry.create_agent("historian")

        assert agent is not None
        assert agent.name == "Historian"

    def test_create_agent_with_llm(self):
        """Test creating agent that requires LLM."""
        registry = AgentRegistry()
        mock_llm = create_mock_llm()

        # Create refiner agent (requires LLM)
        agent = registry.create_agent("refiner", llm=mock_llm)

        assert agent is not None
        assert agent.name == "Refiner"
        assert hasattr(agent, "llm")
        assert agent.llm == mock_llm

    def test_create_agent_llm_required_but_not_provided(self):
        """Test error when LLM required but not provided."""
        registry = AgentRegistry()

        # Try to create refiner without LLM
        with pytest.raises(
            ValueError, match="Agent 'refiner' requires an LLM interface"
        ):
            registry.create_agent("refiner")

    def test_create_unknown_agent_raises_error(self):
        """Test error when trying to create unknown agent."""
        registry = AgentRegistry()

        with pytest.raises(ValueError, match="Unknown agent: 'nonexistent'"):
            registry.create_agent("nonexistent")

    def test_get_agents_requiring_llm(self):
        """Test getting list of agents that require LLM."""
        registry = AgentRegistry()

        llm_agents = registry.get_agents_requiring_llm()

        # Refiner and Critic require LLM
        assert "refiner" in llm_agents
        assert "critic" in llm_agents

        # Historian and Synthesis don't require LLM
        assert "historian" not in llm_agents
        assert "synthesis" not in llm_agents

    def test_validate_pipeline_valid_agents(self):
        """Test pipeline validation with valid agents."""
        registry = AgentRegistry()

        # Valid pipeline
        valid_pipeline = ["refiner", "critic", "historian"]
        assert registry.validate_pipeline(valid_pipeline) is True

    def test_validate_pipeline_invalid_agent(self):
        """Test pipeline validation with invalid agent."""
        registry = AgentRegistry()

        # Invalid pipeline (contains unknown agent)
        invalid_pipeline = ["refiner", "unknown_agent", "historian"]
        assert registry.validate_pipeline(invalid_pipeline) is False

    def test_get_agent_info_unknown_agent(self):
        """Test getting info for unknown agent raises error."""
        registry = AgentRegistry()

        with pytest.raises(ValueError, match="Unknown agent: 'unknown'"):
            registry.get_agent_info("unknown")

    def test_custom_agent_with_dependencies(self):
        """Test registering agent with dependencies."""
        registry = AgentRegistry()

        registry.register(
            name="custom_agent",
            agent_class=MockAgentWithoutLLM,
            requires_llm=False,
            dependencies=["refiner", "critic"],
        )

        metadata = registry.get_agent_info("custom_agent")
        assert metadata.dependencies == ["refiner", "critic"]

    def test_agent_creation_with_kwargs(self):
        """Test agent creation with additional kwargs."""
        registry = AgentRegistry()

        # Register mock agent that accepts additional parameters
        registry.register("mock_with_llm", MockAgentWithLLM, requires_llm=True)

        mock_llm = create_mock_llm()
        agent = registry.create_agent("mock_with_llm", llm=mock_llm)

        assert agent is not None
        assert agent.name == "MockLLM"
        assert agent.llm == mock_llm

    def test_agent_metadata_dependencies_post_init(self):
        """Test AgentMetadata.__post_init__ sets empty list when dependencies is None."""
        from cognivault.agents.registry import AgentMetadata

        # Create metadata without explicit dependencies (should be None initially)
        metadata = AgentMetadata(
            name="test",
            agent_class=MockAgentWithoutLLM,
            requires_llm=False,
            description="Test agent",
            # dependencies not specified, will be None
        )

        # __post_init__ should have converted None to empty list
        assert metadata.dependencies == []

    def test_agent_creation_failure_exception_handling(self):
        """Test exception handling when agent creation fails."""

        class FailingAgent(BaseAgent):
            """Mock agent that fails during initialization."""

            def __init__(self):
                raise RuntimeError("Initialization failed")

            async def run(self, context):
                return context

        registry = AgentRegistry()
        registry.register("failing_agent", FailingAgent, requires_llm=False)

        # Should catch the RuntimeError and re-raise as ValueError
        with pytest.raises(
            ValueError,
            match="Failed to create agent 'failing_agent': Initialization failed",
        ):
            registry.create_agent("failing_agent")


class TestGlobalRegistry:
    """Test suite for global registry functions."""

    def test_get_agent_registry_singleton(self):
        """Test that get_agent_registry returns singleton."""
        registry1 = get_agent_registry()
        registry2 = get_agent_registry()

        # Should be the same instance
        assert registry1 is registry2

    def test_register_agent_global_function(self):
        """Test global register_agent function."""
        # Register using global function
        register_agent(
            "global_test", MockAgentWithoutLLM, description="Global test agent"
        )

        # Verify it's in the global registry
        registry = get_agent_registry()
        assert "global_test" in registry.get_available_agents()

        metadata = registry.get_agent_info("global_test")
        assert metadata.description == "Global test agent"

    def test_create_agent_global_function(self):
        """Test global create_agent function."""
        mock_llm = create_mock_llm()

        # Create agent using global function
        agent = create_agent("refiner", llm=mock_llm)

        assert agent is not None
        assert agent.name == "Refiner"


class TestRegistryWithOrchestrator:
    """Test registry integration with orchestrator patterns."""

    def test_core_agents_metadata(self):
        """Test that core agents have proper metadata."""
        registry = AgentRegistry()

        # Test refiner metadata
        refiner_meta = registry.get_agent_info("refiner")
        assert refiner_meta.requires_llm is True
        assert "refines" in refiner_meta.description.lower()
        assert refiner_meta.dependencies == []

        # Test critic metadata
        critic_meta = registry.get_agent_info("critic")
        assert critic_meta.requires_llm is True
        assert "analyzes" in critic_meta.description.lower()
        assert "refiner" in critic_meta.dependencies

        # Test historian metadata
        historian_meta = registry.get_agent_info("historian")
        assert historian_meta.requires_llm is False
        assert "historical" in historian_meta.description.lower()

        # Test synthesis metadata
        synthesis_meta = registry.get_agent_info("synthesis")
        assert synthesis_meta.requires_llm is False
        assert "synthesizes" in synthesis_meta.description.lower()

    def test_all_core_agents_can_be_created(self):
        """Test that all core agents can be successfully created."""
        registry = AgentRegistry()
        mock_llm = create_mock_llm()

        # Test each core agent
        for agent_name in ["refiner", "critic", "historian", "synthesis"]:
            agent = registry.create_agent(agent_name, llm=mock_llm)
            assert agent is not None
            # Verify agent name matches expected format
            expected_name = agent_name.capitalize()
            assert agent.name == expected_name
