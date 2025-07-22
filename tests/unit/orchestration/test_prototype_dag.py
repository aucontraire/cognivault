"""
Tests for prototype DAG execution.

This module tests the LangGraph-compatible DAG execution system,
including the Refiner → Critic flow and error handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.exceptions import AgentExecutionError
from cognivault.orchestration.prototype_dag import (
    PrototypeDAGExecutor,
    DAGExecutionResult,
    run_prototype_demo,
)


class MockRefinerAgent(BaseAgent):
    """Mock Refiner agent for testing."""

    def __init__(
        self, should_fail: bool = False, output_content: str = "Refined query"
    ):
        super().__init__("Refiner")
        self.should_fail = should_fail
        self.output_content = output_content

    async def run(self, context: AgentContext) -> AgentContext:
        if self.should_fail:
            raise AgentExecutionError("Refiner failed", agent_name="Refiner")

        await asyncio.sleep(0.01)  # Simulate processing time
        context.add_agent_output("Refiner", self.output_content)
        return context


class MockCriticAgent(BaseAgent):
    """Mock Critic agent for testing."""

    def __init__(
        self, should_fail: bool = False, output_content: str = "Critical analysis"
    ):
        super().__init__("Critic")
        self.should_fail = should_fail
        self.output_content = output_content

    async def run(self, context: AgentContext) -> AgentContext:
        if self.should_fail:
            raise AgentExecutionError("Critic failed", agent_name="Critic")

        await asyncio.sleep(0.01)  # Simulate processing time
        context.add_agent_output("Critic", self.output_content)
        return context


@pytest.fixture
def mock_registry():
    """Mock agent registry for testing."""
    registry = Mock()

    def create_agent_side_effect(agent_name, **kwargs):
        if agent_name == "refiner":
            return MockRefinerAgent()
        elif agent_name == "critic":
            return MockCriticAgent()
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    registry.create_agent.side_effect = create_agent_side_effect
    return registry


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.api_key = "test_key"
    llm.model = "test_model"
    return llm


class TestPrototypeDAGExecutor:
    """Test cases for PrototypeDAGExecutor."""

    def test_initialization(self):
        """Test executor initialization."""
        executor = PrototypeDAGExecutor()

        assert executor.enable_parallel_execution is False
        assert executor.max_execution_time_seconds == 300.0
        assert executor.total_executions == 0
        assert executor.successful_executions == 0
        assert executor.failed_executions == 0

    def test_custom_initialization(self):
        """Test executor with custom parameters."""
        executor = PrototypeDAGExecutor(
            enable_parallel_execution=True, max_execution_time_seconds=120.0
        )

        assert executor.enable_parallel_execution is True
        assert executor.max_execution_time_seconds == 120.0

    @pytest.mark.asyncio
    @patch("cognivault.orchestration.prototype_dag.get_agent_registry")
    @patch("cognivault.orchestration.prototype_dag.OpenAIConfig")
    @patch("cognivault.orchestration.prototype_dag.OpenAIChatLLM")
    async def test_successful_dag_execution(
        self, mock_llm_class, mock_config, mock_get_registry, mock_registry
    ):
        """Test successful DAG execution."""
        # Setup mocks
        mock_get_registry.return_value = mock_registry
        mock_config.load.return_value = Mock(
            api_key="test", model="test", base_url="test"
        )
        mock_llm_class.return_value = Mock()

        executor = PrototypeDAGExecutor()

        result = await executor.execute_refiner_critic_dag(
            query="Test query for DAG execution", config={"enable_validation": True}
        )

        assert isinstance(result, DAGExecutionResult)
        assert result.success is True
        assert len(result.nodes_executed) >= 1  # At least refiner should execute
        assert result.total_execution_time_ms > 0
        assert len(result.errors) == 0

        # Check executor statistics
        assert executor.total_executions == 1
        assert executor.successful_executions == 1
        assert executor.failed_executions == 0

    @pytest.mark.asyncio
    @patch("cognivault.orchestration.prototype_dag.get_agent_registry")
    @patch("cognivault.orchestration.prototype_dag.OpenAIConfig")
    @patch("cognivault.orchestration.prototype_dag.OpenAIChatLLM")
    async def test_dag_execution_with_failure(
        self, mock_llm_class, mock_config, mock_get_registry
    ):
        """Test DAG execution with agent failure."""
        # Setup mocks to return failing agents
        failing_registry = Mock()

        def create_failing_agent(agent_name, **kwargs):
            if agent_name == "refiner":
                return MockRefinerAgent(should_fail=True)
            elif agent_name == "critic":
                return MockCriticAgent()
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

        failing_registry.create_agent.side_effect = create_failing_agent
        mock_get_registry.return_value = failing_registry
        mock_config.load.return_value = Mock(
            api_key="test", model="test", base_url="test"
        )
        mock_llm_class.return_value = Mock()

        executor = PrototypeDAGExecutor()

        result = await executor.execute_refiner_critic_dag(
            query="Test query with failure", config={"fail_fast": True}
        )

        assert isinstance(result, DAGExecutionResult)
        assert result.success is False
        assert len(result.errors) > 0
        assert len(result.nodes_executed) == 0  # Refiner should fail immediately

        # Check executor statistics
        assert executor.total_executions == 1
        assert executor.successful_executions == 0
        assert executor.failed_executions == 1

    @pytest.mark.asyncio
    @patch("cognivault.orchestration.prototype_dag.get_agent_registry")
    @patch("cognivault.orchestration.prototype_dag.OpenAIConfig")
    @patch("cognivault.orchestration.prototype_dag.OpenAIChatLLM")
    async def test_partial_execution_success(
        self, mock_llm_class, mock_config, mock_get_registry
    ):
        """Test DAG execution where refiner succeeds but critic fails."""
        # Setup mocks
        partial_registry = Mock()

        def create_partial_failing_agents(agent_name, **kwargs):
            if agent_name == "refiner":
                return MockRefinerAgent()  # Succeeds
            elif agent_name == "critic":
                return MockCriticAgent(should_fail=True)  # Fails
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

        partial_registry.create_agent.side_effect = create_partial_failing_agents
        mock_get_registry.return_value = partial_registry
        mock_config.load.return_value = Mock(
            api_key="test", model="test", base_url="test"
        )
        mock_llm_class.return_value = Mock()

        executor = PrototypeDAGExecutor()

        result = await executor.execute_refiner_critic_dag(
            query="Test query with partial failure",
            config={"fail_fast": False},  # Continue on failure
        )

        assert isinstance(result, DAGExecutionResult)
        # Should be considered failed due to critic failure
        assert result.success is False
        assert len(result.nodes_executed) >= 1  # Refiner should execute
        assert len(result.errors) > 0  # Should have critic error

    def test_get_executor_statistics(self):
        """Test executor statistics collection."""
        executor = PrototypeDAGExecutor()

        # Manually update statistics for testing
        executor.total_executions = 5
        executor.successful_executions = 3
        executor.failed_executions = 2

        stats = executor.get_executor_statistics()

        assert stats["total_executions"] == 5
        assert stats["successful_executions"] == 3
        assert stats["failed_executions"] == 2
        assert stats["success_rate"] == 0.6  # 3/5
        assert "configuration" in stats

    @pytest.mark.asyncio
    @patch("cognivault.orchestration.prototype_dag.get_agent_registry")
    @patch("cognivault.orchestration.prototype_dag.OpenAIConfig")
    @patch("cognivault.orchestration.prototype_dag.OpenAIChatLLM")
    async def test_execution_with_custom_config(
        self, mock_llm_class, mock_config, mock_get_registry, mock_registry
    ):
        """Test DAG execution with custom configuration."""
        mock_get_registry.return_value = mock_registry
        mock_config.load.return_value = Mock(
            api_key="test", model="test", base_url="test"
        )
        mock_llm_class.return_value = Mock()

        executor = PrototypeDAGExecutor()

        custom_config = {
            "enable_validation": False,
            "fail_fast": False,
            "retry_enabled": False,
            "node_timeout_seconds": 15.0,
            "node_configs": {
                "refiner": {"custom_param": "refiner_value"},
                "critic": {"custom_param": "critic_value"},
            },
        }

        result = await executor.execute_refiner_critic_dag(
            query="Test query with custom config", config=custom_config
        )

        assert isinstance(result, DAGExecutionResult)
        assert result.success is True

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        executor = PrototypeDAGExecutor()

        # Create a mock execution result
        mock_execution_result = DAGExecutionResult(
            final_context=AgentContext(query="test"),
            success=True,
            total_execution_time_ms=0,  # Will be set by method
            nodes_executed=["refiner", "critic"],
            edges_traversed=[("refiner", "critic")],
            errors=[],
            execution_path=[
                {
                    "node_id": "refiner",
                    "success": True,
                    "execution_time_ms": 100.0,
                    "iteration": 1,
                },
                {
                    "node_id": "critic",
                    "success": True,
                    "execution_time_ms": 150.0,
                    "iteration": 2,
                },
            ],
            performance_metrics={},
        )

        total_time_ms = 300.0
        metrics = executor._calculate_performance_metrics(
            mock_execution_result, total_time_ms
        )

        assert metrics["total_execution_time_ms"] == 300.0
        assert metrics["total_node_execution_time_ms"] == 250.0  # 100 + 150
        assert metrics["overhead_time_ms"] == 50.0  # 300 - 250
        assert metrics["nodes_executed"] == 2
        assert metrics["edges_traversed"] == 1
        assert metrics["errors_count"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["average_node_time_ms"] == 125.0  # 250 / 2
        assert metrics["execution_efficiency"] == pytest.approx(
            0.833, rel=1e-2
        )  # 250 / 300


class TestDAGExecutionResult:
    """Test cases for DAGExecutionResult."""

    def test_dag_execution_result_creation(self):
        """Test DAG execution result creation."""
        context = AgentContext(query="test")

        result = DAGExecutionResult(
            final_context=context,
            success=True,
            total_execution_time_ms=250.0,
            nodes_executed=["refiner", "critic"],
            edges_traversed=[("refiner", "critic")],
            errors=[],
            execution_path=[{"node_id": "refiner", "success": True}],
            performance_metrics={"efficiency": 0.8},
        )

        assert result.final_context == context
        assert result.success is True
        assert result.total_execution_time_ms == 250.0
        assert result.nodes_executed == ["refiner", "critic"]
        assert result.edges_traversed == [("refiner", "critic")]
        assert result.errors == []
        assert len(result.execution_path) == 1
        assert result.performance_metrics["efficiency"] == 0.8


class TestRunPrototypeDemo:
    """Test cases for the run_prototype_demo function."""

    @pytest.mark.asyncio
    @patch("cognivault.orchestration.prototype_dag.PrototypeDAGExecutor")
    async def test_run_prototype_demo(self, mock_executor_class):
        """Test the prototype demo function."""
        # Setup mock executor
        mock_executor = Mock()
        mock_result = DAGExecutionResult(
            final_context=AgentContext(query="test"),
            success=True,
            total_execution_time_ms=200.0,
            nodes_executed=["refiner", "critic"],
            edges_traversed=[("refiner", "critic")],
            errors=[],
            execution_path=[],
            performance_metrics={"efficiency": 0.9},
        )
        mock_executor.execute_refiner_critic_dag = AsyncMock(return_value=mock_result)
        mock_executor_class.return_value = mock_executor

        result = await run_prototype_demo("Test demo query")

        assert result == mock_result
        mock_executor.execute_refiner_critic_dag.assert_called_once()

        # Check that the correct configuration was passed
        call_args = mock_executor.execute_refiner_critic_dag.call_args
        assert call_args[0][0] == "Test demo query"  # First positional argument (query)
        config = call_args[0][1]  # Second positional argument (config)
        assert config["enable_validation"] is True
        assert config["fail_fast"] is True
        assert config["retry_enabled"] is True
        assert config["node_timeout_seconds"] == 30.0

    @pytest.mark.asyncio
    @patch("cognivault.orchestration.prototype_dag.PrototypeDAGExecutor")
    async def test_run_prototype_demo_default_query(self, mock_executor_class):
        """Test the prototype demo with default query."""
        mock_executor = Mock()
        mock_result = DAGExecutionResult(
            final_context=AgentContext(query="default"),
            success=True,
            total_execution_time_ms=150.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            errors=[],
            execution_path=[],
            performance_metrics={},
        )
        mock_executor.execute_refiner_critic_dag = AsyncMock(return_value=mock_result)
        mock_executor_class.return_value = mock_executor

        result = await run_prototype_demo()

        # Should use default query
        call_args = mock_executor.execute_refiner_critic_dag.call_args
        assert "renewable energy" in call_args[0][0].lower()

    @pytest.mark.asyncio
    @patch("cognivault.orchestration.prototype_dag.logger")
    @patch("cognivault.orchestration.prototype_dag.PrototypeDAGExecutor")
    async def test_run_prototype_demo_logging(self, mock_executor_class, mock_logger):
        """Test that the demo function logs execution results."""
        mock_executor = Mock()
        mock_result = DAGExecutionResult(
            final_context=AgentContext(query="test"),
            success=True,
            total_execution_time_ms=100.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            errors=[],
            execution_path=[],
            performance_metrics={"test": "value"},
        )
        mock_executor.execute_refiner_critic_dag = AsyncMock(return_value=mock_result)
        mock_executor_class.return_value = mock_executor

        await run_prototype_demo("Test logging")

        # Verify that info logs were called
        assert mock_logger.info.call_count >= 1

        # Check that the log calls contain expected information
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Demo execution completed" in log for log in log_calls)


class TestDAGIntegration:
    """Integration tests for the DAG execution system."""

    @pytest.mark.asyncio
    async def test_full_dag_execution_flow(self):
        """Test complete DAG execution flow with real components."""
        # This test uses actual components but with mocked external dependencies

        with (
            patch(
                "cognivault.orchestration.prototype_dag.get_agent_registry"
            ) as mock_get_registry,
            patch("cognivault.orchestration.prototype_dag.OpenAIConfig") as mock_config,
            patch(
                "cognivault.orchestration.prototype_dag.OpenAIChatLLM"
            ) as mock_llm_class,
        ):
            # Setup mocks
            registry = Mock()
            registry.create_agent.side_effect = lambda name, **kwargs: {
                "refiner": MockRefinerAgent(),
                "critic": MockCriticAgent(),
            }[name]

            mock_get_registry.return_value = registry
            mock_config.load.return_value = Mock(
                api_key="test", model="test", base_url="test"
            )
            mock_llm_class.return_value = Mock()

            executor = PrototypeDAGExecutor()

            result = await executor.execute_refiner_critic_dag(
                query="Integration test query",
                config={
                    "enable_validation": True,
                    "fail_fast": False,
                    "retry_enabled": True,
                },
            )

            # Verify complete execution
            assert result.success is True
            assert len(result.nodes_executed) >= 1
            assert result.final_context.get_output("Refiner") is not None

            # Verify execution path tracking
            assert len(result.execution_path) >= 1
            assert all("node_id" in step for step in result.execution_path)
            assert all("success" in step for step in result.execution_path)

            # Verify performance metrics
            assert result.performance_metrics["total_execution_time_ms"] > 0
            assert "nodes_executed" in result.performance_metrics
            assert "success_rate" in result.performance_metrics
