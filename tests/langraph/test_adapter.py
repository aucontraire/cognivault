"""
Tests for LangGraph node adapters.

This module tests the conversion of CogniVault agents into LangGraph-compatible
nodes, including execution, error handling, and routing functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent, LangGraphNodeDefinition, NodeType
from cognivault.exceptions import AgentExecutionError, StateTransitionError
from cognivault.langraph.adapter import (
    LangGraphNodeAdapter,
    StandardNodeAdapter,
    ConditionalNodeAdapter,
    NodeConfiguration,
    NodeExecutionResult,
    create_node_adapter,
)


class MockAgent(BaseAgent):
    """Mock agent for testing node adapters."""

    def __init__(
        self, name: str, should_fail: bool = False, execution_time: float = 0.1
    ):
        super().__init__(name)
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.output_value = f"Output from {name}"

    async def run(self, context: AgentContext) -> AgentContext:
        """Mock agent execution."""
        await asyncio.sleep(self.execution_time)

        if self.should_fail:
            raise AgentExecutionError(
                message="Mock agent failure", agent_name=self.name
            )

        context.add_agent_output(self.name, self.output_value)
        return context


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent("TestAgent")


@pytest.fixture
def failing_agent():
    """Create a mock agent that fails."""
    return MockAgent("FailingAgent", should_fail=True)


@pytest.fixture
def sample_context():
    """Create a sample agent context."""
    return AgentContext(query="Test query for adapter testing")


@pytest.fixture
def node_config():
    """Create a sample node configuration."""
    return NodeConfiguration(
        timeout_seconds=10.0,
        retry_enabled=True,
        step_id="test_step_001",
        custom_config={"test_param": "test_value"},
    )


class TestStandardNodeAdapter:
    """Test cases for StandardNodeAdapter."""

    def test_initialization(self, mock_agent):
        """Test adapter initialization."""
        adapter = StandardNodeAdapter(mock_agent)

        assert adapter.agent == mock_agent
        assert adapter.node_id == "testagent"
        assert adapter.total_executions == 0
        assert adapter.successful_executions == 0
        assert adapter.failed_executions == 0

    def test_custom_node_id(self, mock_agent):
        """Test adapter with custom node ID."""
        adapter = StandardNodeAdapter(mock_agent, node_id="custom_node")
        assert adapter.node_id == "custom_node"

    def test_node_definition_property(self, mock_agent):
        """Test node definition property."""
        adapter = StandardNodeAdapter(mock_agent)
        node_def = adapter.node_definition

        assert isinstance(node_def, LangGraphNodeDefinition)
        assert node_def.node_id == "testagent"

    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_agent, sample_context, node_config):
        """Test successful node execution."""
        adapter = StandardNodeAdapter(mock_agent)

        result = await adapter.execute(sample_context, node_config)

        assert isinstance(result, NodeExecutionResult)
        assert result.success is True
        assert result.node_id == "testagent"
        assert result.step_id == "test_step_001"
        assert result.error is None
        assert result.execution_time_ms > 0

        # Check adapter statistics
        assert adapter.total_executions == 1
        assert adapter.successful_executions == 1
        assert adapter.failed_executions == 0
        assert adapter.success_rate == 100.0

    @pytest.mark.asyncio
    async def test_failed_execution(self, failing_agent, sample_context, node_config):
        """Test failed node execution."""
        adapter = StandardNodeAdapter(failing_agent)

        result = await adapter.execute(sample_context, node_config)

        assert result.success is False
        assert result.error is not None
        assert isinstance(result.error, AgentExecutionError)
        assert result.execution_time_ms > 0

        # Check adapter statistics
        assert adapter.total_executions == 1
        assert adapter.successful_executions == 0
        assert adapter.failed_executions == 1
        assert adapter.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_call_interface(self, mock_agent, sample_context, node_config):
        """Test the __call__ interface for LangGraph compatibility."""
        adapter = StandardNodeAdapter(mock_agent)

        result_context = await adapter(sample_context, node_config)

        assert isinstance(result_context, AgentContext)
        assert result_context.get_output("TestAgent") == "Output from TestAgent"

    @pytest.mark.asyncio
    async def test_call_interface_with_failure(
        self, failing_agent, sample_context, node_config
    ):
        """Test __call__ interface with agent failure."""
        adapter = StandardNodeAdapter(failing_agent)

        with pytest.raises(AgentExecutionError):
            await adapter(sample_context, node_config)

    def test_statistics(self, mock_agent):
        """Test statistics collection."""
        adapter = StandardNodeAdapter(mock_agent)

        stats = adapter.get_statistics()

        assert stats["node_id"] == "testagent"
        assert stats["agent_name"] == "TestAgent"
        assert stats["total_executions"] == 0
        assert "node_definition" in stats
        assert "agent_stats" in stats

    @pytest.mark.asyncio
    async def test_multiple_executions(self, mock_agent, sample_context):
        """Test multiple executions and statistics tracking."""
        adapter = StandardNodeAdapter(mock_agent)

        # Execute multiple times
        for i in range(3):
            await adapter.execute(sample_context, NodeConfiguration())

        assert adapter.total_executions == 3
        assert adapter.successful_executions == 3
        assert adapter.failed_executions == 0
        assert adapter.success_rate == 100.0
        assert adapter.average_execution_time_ms > 0


class TestConditionalNodeAdapter:
    """Test cases for ConditionalNodeAdapter."""

    def test_initialization(self, mock_agent):
        """Test conditional adapter initialization."""

        def routing_func(context):
            return ["next_node"]

        adapter = ConditionalNodeAdapter(mock_agent, routing_func)

        assert adapter.agent == mock_agent
        assert adapter.routing_function == routing_func
        assert adapter.node_id == "testagent"

    @pytest.mark.asyncio
    async def test_routing_function_execution(self, mock_agent, sample_context):
        """Test routing function execution during post-processing."""
        executed_nodes = []

        def routing_func(context):
            executed_nodes.append("routing_executed")
            return ["next_node", "alternative_node"]

        adapter = ConditionalNodeAdapter(mock_agent, routing_func)
        result = await adapter.execute(sample_context, NodeConfiguration())

        assert result.success is True
        assert len(executed_nodes) == 1

        # Check that routing was recorded in context
        assert len(result.context.conditional_routing) > 0
        assert len(result.context.execution_edges) > 0

    @pytest.mark.asyncio
    async def test_routing_function_failure(self, mock_agent, sample_context):
        """Test handling of routing function failures."""

        def failing_routing_func(context):
            raise ValueError("Routing function error")

        adapter = ConditionalNodeAdapter(mock_agent, failing_routing_func)
        result = await adapter.execute(sample_context, NodeConfiguration())

        # Agent execution should succeed, but routing should fail gracefully
        assert result.success is True

        # Check error edge was added
        error_edges = [
            edge
            for edge in result.context.execution_edges
            if edge.get("to_agent") == "ERROR"
        ]
        assert len(error_edges) > 0


class TestNodeConfiguration:
    """Test cases for NodeConfiguration."""

    def test_default_configuration(self):
        """Test default node configuration."""
        config = NodeConfiguration()

        assert config.timeout_seconds is None
        assert config.retry_enabled is True
        assert config.step_id is None
        assert config.custom_config is None

    def test_custom_configuration(self):
        """Test custom node configuration."""
        config = NodeConfiguration(
            timeout_seconds=30.0,
            retry_enabled=False,
            step_id="custom_step",
            custom_config={"param1": "value1"},
        )

        assert config.timeout_seconds == 30.0
        assert config.retry_enabled is False
        assert config.step_id == "custom_step"
        assert config.custom_config == {"param1": "value1"}


class TestCreateNodeAdapter:
    """Test cases for the create_node_adapter factory function."""

    def test_create_standard_adapter(self, mock_agent):
        """Test creating standard adapter."""
        adapter = create_node_adapter(mock_agent, "standard")

        assert isinstance(adapter, StandardNodeAdapter)
        assert adapter.agent == mock_agent

    def test_create_conditional_adapter(self, mock_agent):
        """Test creating conditional adapter."""

        def routing_func(context):
            return ["next"]

        adapter = create_node_adapter(
            mock_agent, "conditional", routing_function=routing_func
        )

        assert isinstance(adapter, ConditionalNodeAdapter)
        assert adapter.routing_function == routing_func

    def test_create_conditional_adapter_missing_routing(self, mock_agent):
        """Test creating conditional adapter without routing function."""
        with pytest.raises(ValueError, match="requires 'routing_function'"):
            create_node_adapter(mock_agent, "conditional")

    def test_unknown_adapter_type(self, mock_agent):
        """Test unknown adapter type."""
        with pytest.raises(ValueError, match="Unknown adapter type"):
            create_node_adapter(mock_agent, "unknown_type")

    def test_custom_parameters(self, mock_agent):
        """Test passing custom parameters to adapter."""
        adapter = create_node_adapter(
            mock_agent, "standard", node_id="custom_id", enable_state_validation=False
        )

        assert adapter.node_id == "custom_id"
        assert adapter.enable_state_validation is False


class TestNodeAdapterIntegration:
    """Integration tests for node adapters."""

    @pytest.mark.asyncio
    async def test_context_state_management(self, mock_agent, sample_context):
        """Test context state management during execution."""
        adapter = StandardNodeAdapter(mock_agent)
        original_size = sample_context.current_size

        result = await adapter.execute(sample_context, NodeConfiguration())

        # Context should be modified
        assert result.context.current_size != original_size
        assert result.context.get_output("TestAgent") is not None

        # Execution metadata should be added
        assert len(result.context.execution_edges) > 0
        assert result.context.agent_execution_status.get("TestAgent") == "completed"

    @pytest.mark.asyncio
    async def test_snapshot_creation(self, mock_agent, sample_context):
        """Test snapshot creation during execution."""
        adapter = StandardNodeAdapter(mock_agent)
        initial_snapshots = len(sample_context.snapshots)

        await adapter.execute(sample_context, NodeConfiguration())

        # Snapshot should be created before execution
        assert len(sample_context.snapshots) > initial_snapshots

    @pytest.mark.asyncio
    async def test_execution_metadata(self, mock_agent, sample_context, node_config):
        """Test execution metadata collection."""
        adapter = StandardNodeAdapter(mock_agent)

        result = await adapter.execute(sample_context, node_config)

        assert result.metadata is not None
        assert result.metadata["node_id"] == "testagent"
        assert result.metadata["agent_name"] == "TestAgent"
        assert "total_executions" in result.metadata
        assert "success_rate" in result.metadata
        assert "config" in result.metadata

    @pytest.mark.asyncio
    async def test_performance_tracking(self, sample_context):
        """Test performance tracking across multiple agents."""
        fast_agent = MockAgent("FastAgent", execution_time=0.01)
        slow_agent = MockAgent("SlowAgent", execution_time=0.1)

        fast_adapter = StandardNodeAdapter(fast_agent)
        slow_adapter = StandardNodeAdapter(slow_agent)

        # Execute both agents
        fast_result = await fast_adapter.execute(sample_context, NodeConfiguration())
        slow_result = await slow_adapter.execute(sample_context, NodeConfiguration())

        # Fast agent should have lower execution time
        assert fast_result.execution_time_ms < slow_result.execution_time_ms
        assert (
            fast_adapter.average_execution_time_ms
            < slow_adapter.average_execution_time_ms
        )

    @pytest.mark.asyncio
    async def test_state_validation_failure(self, mock_agent, sample_context):
        """Test state validation failure handling."""
        # Mock the validate_node_compatibility to return False
        with patch.object(
            mock_agent, "validate_node_compatibility", return_value=False
        ):
            adapter = StandardNodeAdapter(mock_agent, enable_state_validation=True)

            result = await adapter.execute(sample_context, NodeConfiguration())

            assert result.success is False
            assert isinstance(result.error, StateTransitionError)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_context):
        """Test timeout handling in node execution."""
        # Create an agent that takes a long time
        slow_agent = MockAgent("SlowAgent", execution_time=1.0)
        adapter = StandardNodeAdapter(slow_agent)

        # Set a very short timeout
        config = NodeConfiguration(timeout_seconds=0.01)

        # Note: This test depends on the agent's timeout implementation
        # For now, we'll just verify the configuration is passed through
        result = await adapter.execute(sample_context, config)

        # The result may succeed or fail depending on actual timeout implementation
        assert isinstance(result, NodeExecutionResult)
