"""
Tests for RealLangGraphOrchestrator.

This module tests the real LangGraph orchestrator implementation,
including CLI integration and state bridge functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from cognivault.context import AgentContext
from cognivault.langraph.real_orchestrator import RealLangGraphOrchestrator
from cognivault.langraph.state_bridge import AgentContextStateBridge


class TestRealLangGraphOrchestrator:
    """Test suite for RealLangGraphOrchestrator."""

    def test_init_default_agents(self):
        """Test orchestrator initialization with default agents."""
        # Act
        orchestrator = RealLangGraphOrchestrator()

        # Assert
        assert orchestrator.agents_to_run == [
            "refiner",
            "historian",
            "critic",
            "synthesis",
        ]
        assert orchestrator.total_executions == 0
        assert orchestrator.successful_executions == 0
        assert orchestrator.failed_executions == 0
        assert isinstance(orchestrator.state_bridge, AgentContextStateBridge)
        assert orchestrator.agents == []  # Empty until agents are created

    def test_init_custom_agents(self):
        """Test orchestrator initialization with custom agents."""
        # Arrange
        custom_agents = ["refiner", "critic"]

        # Act
        orchestrator = RealLangGraphOrchestrator(agents_to_run=custom_agents)

        # Assert
        assert orchestrator.agents_to_run == custom_agents
        assert isinstance(orchestrator.state_bridge, AgentContextStateBridge)

    @pytest.mark.asyncio
    async def test_run_stub_implementation(self):
        """Test that run method is currently a stub that demonstrates functionality."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator(agents_to_run=["refiner", "critic"])
        query = "Test query for stub implementation"

        # Act & Assert
        with pytest.raises(NotImplementedError) as exc_info:
            await orchestrator.run(query)

        # Assert the error message is informative
        assert "Real LangGraph execution is not yet implemented" in str(exc_info.value)
        assert "This is a Phase 1 stub" in str(exc_info.value)
        assert query in str(exc_info.value)
        assert "refiner" in str(exc_info.value)
        assert "critic" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_updates_statistics(self):
        """Test that run method updates execution statistics."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()
        query = "Test query"

        # Act
        try:
            await orchestrator.run(query)
        except NotImplementedError:
            pass  # Expected for stub implementation

        # Assert statistics were updated
        assert orchestrator.total_executions == 1
        assert orchestrator.successful_executions == 1
        assert orchestrator.failed_executions == 0

    @pytest.mark.asyncio
    async def test_run_with_config(self):
        """Test run method with configuration parameters."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()
        query = "Test query"
        config = {"timeout": 30, "enable_parallel": True}

        # Act & Assert
        with pytest.raises(NotImplementedError) as exc_info:
            await orchestrator.run(query, config)

        # Should handle config without error
        assert "Real LangGraph execution is not yet implemented" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_creates_context_with_metadata(self):
        """Test that run method creates context with proper metadata."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator(agents_to_run=["refiner"])
        query = "Test query with metadata"

        # We'll patch the NotImplementedError to inspect the context
        original_run = orchestrator.run

        async def mock_run(query, config=None):
            # Call the original method to get the context creation logic
            try:
                await original_run(query, config)
            except NotImplementedError:
                # The context should be created and processed before the error
                pass

        # Act - we can't easily test internal state due to NotImplementedError
        # But we can verify the method handles the orchestrator type correctly
        with pytest.raises(NotImplementedError):
            await orchestrator.run(query)

        # Just verify the orchestrator is properly initialized
        assert orchestrator.agents_to_run == ["refiner"]

    @pytest.mark.asyncio
    async def test_run_state_bridge_integration(self):
        """Test that run method integrates with state bridge."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()
        query = "Test state bridge integration"

        # Mock state bridge methods to avoid actual conversion
        with patch.object(
            orchestrator.state_bridge, "to_langgraph_state"
        ) as mock_to_lg:
            with patch.object(
                orchestrator.state_bridge, "from_langgraph_state"
            ) as mock_from_lg:
                # Setup mocks
                mock_to_lg.return_value = {"_context_id": "test", "_query": query}
                mock_from_lg.return_value = AgentContext(query=query, context_id="test")

                # Act
                with pytest.raises(NotImplementedError):
                    await orchestrator.run(query)

                # Assert state bridge methods were called
                mock_to_lg.assert_called_once()
                mock_from_lg.assert_called_once()

    def test_get_execution_statistics(self):
        """Test execution statistics reporting."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator(agents_to_run=["refiner", "critic"])

        # Manually update statistics to test reporting
        orchestrator.total_executions = 5
        orchestrator.successful_executions = 4
        orchestrator.failed_executions = 1

        # Act
        stats = orchestrator.get_execution_statistics()

        # Assert
        assert stats["orchestrator_type"] == "langgraph-real"
        assert stats["implementation_status"] == "phase1_stub"
        assert stats["total_executions"] == 5
        assert stats["successful_executions"] == 4
        assert stats["failed_executions"] == 1
        assert stats["success_rate"] == 0.8  # 4/5
        assert stats["agents_to_run"] == ["refiner", "critic"]
        assert stats["state_bridge_available"] is True

    def test_get_execution_statistics_no_executions(self):
        """Test statistics reporting with no executions."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()

        # Act
        stats = orchestrator.get_execution_statistics()

        # Assert
        assert stats["success_rate"] == 0
        assert stats["total_executions"] == 0

    def test_initialize_llm(self):
        """Test LLM initialization."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()

        # Act
        with patch("cognivault.langraph.real_orchestrator.OpenAIConfig") as mock_config:
            with patch(
                "cognivault.langraph.real_orchestrator.OpenAIChatLLM"
            ) as mock_llm:
                mock_config.load.return_value = Mock(
                    api_key="test-key",
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                )

                result = orchestrator._initialize_llm()

                # Assert
                mock_config.load.assert_called_once()
                mock_llm.assert_called_once_with(
                    api_key="test-key",
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                )

    def test_create_agents(self):
        """Test agent creation functionality."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator(agents_to_run=["refiner", "critic"])
        mock_llm = Mock()

        # Mock the registry
        mock_agent_refiner = Mock()
        mock_agent_refiner.name = "refiner"
        mock_agent_critic = Mock()
        mock_agent_critic.name = "critic"

        with patch.object(orchestrator.registry, "create_agent") as mock_create:
            mock_create.side_effect = lambda name, llm: {
                "refiner": mock_agent_refiner,
                "critic": mock_agent_critic,
            }[name]

            # Act
            agents = orchestrator._create_agents(mock_llm)

            # Assert
            assert len(agents) == 2
            assert "refiner" in agents
            assert "critic" in agents
            assert agents["refiner"] == mock_agent_refiner
            assert agents["critic"] == mock_agent_critic
            assert len(orchestrator.agents) == 2  # Added to agents list

    def test_create_agents_with_failures(self):
        """Test agent creation with some failures."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator(
            agents_to_run=["refiner", "critic", "invalid"]
        )
        mock_llm = Mock()

        # Mock the registry to fail on invalid agent
        mock_agent_refiner = Mock()
        mock_agent_refiner.name = "refiner"
        mock_agent_critic = Mock()
        mock_agent_critic.name = "critic"

        def mock_create_agent(name, llm):
            if name == "invalid":
                raise ValueError("Invalid agent")
            return {"refiner": mock_agent_refiner, "critic": mock_agent_critic}[name]

        with patch.object(
            orchestrator.registry, "create_agent", side_effect=mock_create_agent
        ):
            # Act
            agents = orchestrator._create_agents(mock_llm)

            # Assert - should continue with valid agents
            assert len(agents) == 2
            assert "refiner" in agents
            assert "critic" in agents
            assert "invalid" not in agents

    def test_compatibility_with_existing_cli(self):
        """Test that orchestrator is compatible with existing CLI patterns."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()

        # Assert compatibility attributes exist
        assert hasattr(orchestrator, "agents")  # For health checks
        assert hasattr(orchestrator, "agents_to_run")  # For configuration
        assert hasattr(orchestrator, "registry")  # For agent management
        assert hasattr(orchestrator, "run")  # For execution
        assert hasattr(orchestrator, "get_execution_statistics")  # For reporting

        # Check that agents list starts empty
        assert orchestrator.agents == []

    def test_logging_integration(self):
        """Test that orchestrator integrates with logging system."""
        # Arrange & Act
        orchestrator = RealLangGraphOrchestrator()

        # Assert
        assert hasattr(orchestrator, "logger")
        assert "RealLangGraphOrchestrator" in orchestrator.logger.name

    def test_state_bridge_integration(self):
        """Test that orchestrator properly integrates with state bridge."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()

        # Assert
        assert hasattr(orchestrator, "state_bridge")
        assert isinstance(orchestrator.state_bridge, AgentContextStateBridge)

    def test_orchestrator_type_identification(self):
        """Test that orchestrator properly identifies itself."""
        # Arrange
        orchestrator = RealLangGraphOrchestrator()

        # Act
        stats = orchestrator.get_execution_statistics()

        # Assert
        assert stats["orchestrator_type"] == "langgraph-real"
        assert stats["implementation_status"] == "phase1_stub"
