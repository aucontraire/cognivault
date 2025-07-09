"""
Tests for LangGraphOrchestrator class.

This module provides comprehensive test coverage for the LangGraph-based orchestrator
that handles DAG execution of CogniVault agents.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import time
from typing import Dict, Any, List

from cognivault.context import AgentContext
from cognivault.langraph.orchestrator import (
    LangGraphOrchestrator,
    LangGraphExecutionResult,
)
from cognivault.langraph.prototype_dag import DAGExecutionResult
from cognivault.agents.base_agent import BaseAgent
from cognivault.exceptions import PipelineExecutionError


class TestLangGraphOrchestrator:
    """Test suite for LangGraphOrchestrator."""

    def test_init_default_agents(self):
        """Test LangGraphOrchestrator initialization with default agents."""
        orchestrator = LangGraphOrchestrator()

        assert orchestrator.agents_to_run == [
            "refiner",
            "historian",
            "critic",
            "synthesis",
        ]
        assert orchestrator.registry is not None
        assert orchestrator.logger is not None
        assert orchestrator.agents == []
        assert orchestrator.total_executions == 0
        assert orchestrator.successful_executions == 0
        assert orchestrator.failed_executions == 0

    def test_init_custom_agents(self):
        """Test LangGraphOrchestrator initialization with custom agents."""
        custom_agents = ["refiner", "critic"]
        orchestrator = LangGraphOrchestrator(agents_to_run=custom_agents)

        assert orchestrator.agents_to_run == custom_agents
        assert orchestrator.registry is not None
        assert orchestrator.agents == []

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful execution of LangGraph orchestrator."""
        query = "test query"

        # Mock the DAG execution result
        mock_dag_result = DAGExecutionResult(
            final_context=AgentContext(query=query),
            success=True,
            total_execution_time_ms=1000.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            errors=[],
            execution_path=[],
            performance_metrics={"test": "metrics"},
        )

        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])

        with (
            patch.object(orchestrator, "_initialize_llm") as mock_init_llm,
            patch.object(orchestrator, "_create_agents") as mock_create_agents,
            patch.object(
                orchestrator,
                "_execute_langgraph_dag",
                return_value=LangGraphExecutionResult(
                    final_context=mock_dag_result.final_context,
                    success=True,
                    execution_time_ms=1000.0,
                    nodes_executed=["refiner"],
                    edges_traversed=[],
                    performance_metrics={"test": "metrics"},
                    dag_result=mock_dag_result,
                ),
            ) as mock_execute,
        ):
            mock_llm = Mock()
            mock_init_llm.return_value = mock_llm
            mock_create_agents.return_value = {"refiner": Mock()}

            result = await orchestrator.run(query)

            assert isinstance(result, AgentContext)
            assert result.query == query
            assert orchestrator.total_executions == 1
            assert orchestrator.successful_executions == 1
            assert orchestrator.failed_executions == 0

            mock_init_llm.assert_called_once()
            mock_create_agents.assert_called_once_with(mock_llm)
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_config(self):
        """Test LangGraph orchestrator run with custom configuration."""
        query = "test query with config"
        config = {
            "enable_parallel_execution": True,
            "max_execution_time_seconds": 600.0,
        }

        mock_dag_result = DAGExecutionResult(
            final_context=AgentContext(query=query),
            success=True,
            total_execution_time_ms=1500.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            errors=[],
            execution_path=[],
            performance_metrics={"test": "metrics"},
        )

        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])

        with (
            patch.object(orchestrator, "_initialize_llm") as mock_init_llm,
            patch.object(orchestrator, "_create_agents") as mock_create_agents,
            patch.object(
                orchestrator,
                "_execute_langgraph_dag",
                return_value=LangGraphExecutionResult(
                    final_context=mock_dag_result.final_context,
                    success=True,
                    execution_time_ms=1500.0,
                    nodes_executed=["refiner"],
                    edges_traversed=[],
                    performance_metrics={"test": "metrics"},
                    dag_result=mock_dag_result,
                ),
            ) as mock_execute,
        ):
            mock_llm = Mock()
            mock_init_llm.return_value = mock_llm
            mock_create_agents.return_value = {"refiner": Mock()}

            result = await orchestrator.run(query, config)

            assert isinstance(result, AgentContext)
            mock_execute.assert_called_once()
            # Verify config was passed to the execution method
            call_args = mock_execute.call_args
            assert call_args[0][2] == config  # config is the third argument

    @pytest.mark.asyncio
    async def test_run_execution_failure(self):
        """Test LangGraph orchestrator run with execution failure."""
        query = "test query failure"

        mock_dag_result = DAGExecutionResult(
            final_context=AgentContext(query=query),
            success=False,
            total_execution_time_ms=500.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            errors=[Exception("Test error")],
            execution_path=[],
            performance_metrics={"error": "test"},
        )

        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])

        with (
            patch.object(orchestrator, "_initialize_llm") as mock_init_llm,
            patch.object(orchestrator, "_create_agents") as mock_create_agents,
            patch.object(
                orchestrator,
                "_execute_langgraph_dag",
                return_value=LangGraphExecutionResult(
                    final_context=mock_dag_result.final_context,
                    success=False,
                    execution_time_ms=500.0,
                    nodes_executed=["refiner"],
                    edges_traversed=[],
                    performance_metrics={"error": "test"},
                    dag_result=mock_dag_result,
                ),
            ) as mock_execute,
        ):
            mock_llm = Mock()
            mock_init_llm.return_value = mock_llm
            mock_create_agents.return_value = {"refiner": Mock()}

            result = await orchestrator.run(query)

            assert isinstance(result, AgentContext)
            assert orchestrator.total_executions == 1
            assert orchestrator.successful_executions == 0
            assert orchestrator.failed_executions == 1

    @pytest.mark.asyncio
    async def test_run_exception_handling(self):
        """Test LangGraph orchestrator exception handling."""
        query = "test query exception"

        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])

        with patch.object(
            orchestrator, "_initialize_llm", side_effect=Exception("LLM init failed")
        ):
            result = await orchestrator.run(query)

            assert isinstance(result, AgentContext)
            assert result.query == query
            assert "langgraph_error" in result.execution_state
            assert "LLM init failed" in result.execution_state["langgraph_error"]
            assert "execution_time_ms" in result.execution_state
            assert orchestrator.total_executions == 1
            assert orchestrator.successful_executions == 0
            assert orchestrator.failed_executions == 1

    def test_create_agents_success(self):
        """Test successful agent creation."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner", "critic"])
        mock_llm = Mock()

        # Mock agents
        mock_refiner = Mock(spec=BaseAgent)
        mock_refiner.name = "Refiner"
        mock_critic = Mock(spec=BaseAgent)
        mock_critic.name = "Critic"

        with patch.object(orchestrator.registry, "create_agent") as mock_create:
            mock_create.side_effect = [mock_refiner, mock_critic]

            agents = orchestrator._create_agents(mock_llm)

            assert len(agents) == 2
            assert "refiner" in agents
            assert "critic" in agents
            assert agents["refiner"] == mock_refiner
            assert agents["critic"] == mock_critic
            assert len(orchestrator.agents) == 2
            assert orchestrator.agents[0] == mock_refiner
            assert orchestrator.agents[1] == mock_critic

            # Verify registry calls
            assert mock_create.call_count == 2
            mock_create.assert_any_call("refiner", llm=mock_llm)
            mock_create.assert_any_call("critic", llm=mock_llm)

    def test_create_agents_with_failures(self):
        """Test agent creation with some failures."""
        orchestrator = LangGraphOrchestrator(
            agents_to_run=["refiner", "critic", "historian"]
        )
        mock_llm = Mock()

        # Mock agents
        mock_refiner = Mock(spec=BaseAgent)
        mock_refiner.name = "Refiner"
        mock_historian = Mock(spec=BaseAgent)
        mock_historian.name = "Historian"

        def create_agent_side_effect(agent_name, **kwargs):
            if agent_name == "refiner":
                return mock_refiner
            elif agent_name == "critic":
                raise Exception("Failed to create critic")
            elif agent_name == "historian":
                return mock_historian

        with patch.object(
            orchestrator.registry, "create_agent", side_effect=create_agent_side_effect
        ):
            agents = orchestrator._create_agents(mock_llm)

            # Should only have successful agents
            assert len(agents) == 2
            assert "refiner" in agents
            assert "historian" in agents
            assert "critic" not in agents
            assert len(orchestrator.agents) == 2

    @pytest.mark.asyncio
    async def test_execute_langgraph_dag_refiner_critic(self):
        """Test LangGraph DAG execution for refiner-critic combination."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner", "critic"])
        context = AgentContext(query="test query")
        agents = {"refiner": Mock(), "critic": Mock()}
        config = {"test": "config"}

        mock_dag_result = DAGExecutionResult(
            final_context=context,
            success=True,
            total_execution_time_ms=1000.0,
            nodes_executed=["refiner", "critic"],
            edges_traversed=[("refiner", "critic")],
            errors=[],
            execution_path=[],
            performance_metrics={"nodes": 2},
        )

        with patch(
            "cognivault.langraph.orchestrator.PrototypeDAGExecutor"
        ) as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            mock_executor.execute_refiner_critic_dag = AsyncMock(
                return_value=mock_dag_result
            )

            result = await orchestrator._execute_langgraph_dag(context, agents, config)

            assert isinstance(result, LangGraphExecutionResult)
            assert result.success is True
            assert result.execution_time_ms == 1000.0
            assert result.nodes_executed == ["refiner", "critic"]
            assert result.edges_traversed == [("refiner", "critic")]
            assert result.performance_metrics == {"nodes": 2}
            assert result.dag_result == mock_dag_result

            # Verify executor was configured correctly
            mock_executor_class.assert_called_once_with(
                enable_parallel_execution=config.get(
                    "enable_parallel_execution", False
                ),
                max_execution_time_seconds=config.get(
                    "max_execution_time_seconds", 300.0
                ),
            )
            mock_executor.execute_refiner_critic_dag.assert_called_once_with(
                context.query, config
            )

    @pytest.mark.asyncio
    async def test_execute_langgraph_dag_general(self):
        """Test LangGraph DAG execution for general agent combinations."""
        orchestrator = LangGraphOrchestrator(
            agents_to_run=["refiner", "historian", "synthesis"]
        )
        context = AgentContext(query="test query")
        agents = {"refiner": Mock(), "historian": Mock(), "synthesis": Mock()}
        config = {}

        mock_dag_result = DAGExecutionResult(
            final_context=context,
            success=True,
            total_execution_time_ms=2000.0,
            nodes_executed=["refiner", "historian", "synthesis"],
            edges_traversed=[("refiner", "historian"), ("historian", "synthesis")],
            errors=[],
            execution_path=[],
            performance_metrics={"nodes": 3},
        )

        with (
            patch(
                "cognivault.langraph.orchestrator.PrototypeDAGExecutor"
            ) as mock_executor_class,
            patch.object(
                orchestrator, "_execute_general_dag", return_value=mock_dag_result
            ) as mock_general_dag,
        ):
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor

            result = await orchestrator._execute_langgraph_dag(context, agents, config)

            assert isinstance(result, LangGraphExecutionResult)
            assert result.success is True
            assert result.execution_time_ms == 2000.0
            assert result.nodes_executed == ["refiner", "historian", "synthesis"]

            # Verify general DAG execution was called
            mock_general_dag.assert_called_once_with(
                mock_executor, context, agents, config
            )

    @pytest.mark.asyncio
    async def test_execute_general_dag_success(self):
        """Test general DAG execution with successful agents."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner", "critic"])

        # Create mock agents
        mock_refiner = AsyncMock()
        mock_critic = AsyncMock()

        # Create mock contexts
        initial_context = AgentContext(query="test query")
        refiner_context = AgentContext(query="test query")
        refiner_context.agent_outputs["Refiner"] = "Refined output"
        final_context = AgentContext(query="test query")
        final_context.agent_outputs["Refiner"] = "Refined output"
        final_context.agent_outputs["Critic"] = "Critical analysis"

        mock_refiner.run.return_value = refiner_context
        mock_critic.run.return_value = final_context

        agents = {"refiner": mock_refiner, "critic": mock_critic}
        config = {}

        # Mock the executor
        mock_executor = Mock()

        result = await orchestrator._execute_general_dag(
            mock_executor, initial_context, agents, config
        )

        assert isinstance(result, DAGExecutionResult)
        assert result.success is True
        assert result.nodes_executed == ["refiner", "critic"]
        assert result.edges_traversed == [("refiner", "critic")]
        assert len(result.errors) == 0
        assert result.final_context == final_context

        # Verify agents were called in order
        mock_refiner.run.assert_called_once_with(initial_context)
        mock_critic.run.assert_called_once_with(refiner_context)

    @pytest.mark.asyncio
    async def test_execute_general_dag_with_agent_failure(self):
        """Test general DAG execution with agent failure."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner", "critic"])

        # Create mock agents
        mock_refiner = AsyncMock()
        mock_critic = AsyncMock()

        # Refiner succeeds, critic fails
        refiner_context = AgentContext(query="test query")
        refiner_context.agent_outputs["Refiner"] = "Refined output"
        mock_refiner.run.return_value = refiner_context
        mock_critic.run.side_effect = Exception("Critic failed")

        agents = {"refiner": mock_refiner, "critic": mock_critic}
        config = {}
        initial_context = AgentContext(query="test query")

        # Mock the executor
        mock_executor = Mock()

        result = await orchestrator._execute_general_dag(
            mock_executor, initial_context, agents, config
        )

        assert isinstance(result, DAGExecutionResult)
        assert result.success is False  # Should fail due to errors
        assert result.nodes_executed == ["refiner"]  # Only refiner succeeded
        assert len(result.errors) == 1
        assert "Critic failed" in str(result.errors[0])

        # Verify refiner was called but critic failed
        mock_refiner.run.assert_called_once()
        mock_critic.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_general_dag_exception(self):
        """Test general DAG execution with unexpected exception."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])

        initial_context = AgentContext(query="test query")

        # Create an async mock agent that will raise an exception
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("Agent execution failed")
        agents = {"refiner": mock_agent}
        config = {}

        # Mock the executor
        mock_executor = Mock()

        result = await orchestrator._execute_general_dag(
            mock_executor, initial_context, agents, config
        )

        assert isinstance(result, DAGExecutionResult)
        assert result.success is False
        assert len(result.errors) == 1
        assert "Agent execution failed" in str(result.errors[0])
        assert result.nodes_executed == []  # No nodes executed due to failure

    @pytest.mark.asyncio
    async def test_execute_general_dag_system_exception(self):
        """Test general DAG execution with system-level exception."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])

        initial_context = AgentContext(query="test query")
        agents = {"refiner": AsyncMock()}
        config = {}

        # Mock the executor
        mock_executor = Mock()

        # Force a system exception by making the iteration over agents_to_run fail
        # We'll override the agents_to_run property to be a mock that raises an exception when iterated
        class FailingList:
            def __iter__(self):
                raise Exception("System error during iteration")

        orchestrator.agents_to_run = FailingList()

        result = await orchestrator._execute_general_dag(
            mock_executor, initial_context, agents, config
        )

        assert isinstance(result, DAGExecutionResult)
        assert result.success is False
        assert len(result.errors) == 1
        assert "System error during iteration" in str(result.errors[0])
        assert (
            result.final_context == initial_context
        )  # Should return original context on system failure

    def test_initialize_llm(self):
        """Test LLM initialization."""
        orchestrator = LangGraphOrchestrator()

        with (
            patch("cognivault.langraph.orchestrator.OpenAIConfig") as mock_config_class,
            patch("cognivault.langraph.orchestrator.OpenAIChatLLM") as mock_llm_class,
        ):
            # Mock config
            mock_config = Mock()
            mock_config.api_key = "test-key"
            mock_config.model = "gpt-4"
            mock_config.base_url = "https://api.openai.com/v1"
            mock_config_class.load.return_value = mock_config

            # Mock LLM
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            result = orchestrator._initialize_llm()

            assert result == mock_llm
            mock_config_class.load.assert_called_once()
            mock_llm_class.assert_called_once_with(
                api_key="test-key", model="gpt-4", base_url="https://api.openai.com/v1"
            )

    def test_get_execution_statistics_no_executions(self):
        """Test execution statistics with no executions."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner", "critic"])

        stats = orchestrator.get_execution_statistics()

        assert stats["orchestrator_type"] == "langgraph"
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 0
        assert stats["success_rate"] == 0
        assert stats["agents_to_run"] == ["refiner", "critic"]

    def test_get_execution_statistics_with_executions(self):
        """Test execution statistics with some executions."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])

        # Simulate some executions
        orchestrator.total_executions = 10
        orchestrator.successful_executions = 8
        orchestrator.failed_executions = 2

        stats = orchestrator.get_execution_statistics()

        assert stats["orchestrator_type"] == "langgraph"
        assert stats["total_executions"] == 10
        assert stats["successful_executions"] == 8
        assert stats["failed_executions"] == 2
        assert stats["success_rate"] == 0.8
        assert stats["agents_to_run"] == ["refiner"]

    def test_agents_property_type_annotation(self):
        """Test that agents property has correct type annotation."""
        orchestrator = LangGraphOrchestrator()

        # Agents should be an empty list initially
        assert isinstance(orchestrator.agents, list)
        assert len(orchestrator.agents) == 0

        # Should be able to add BaseAgent instances
        mock_agent = Mock(spec=BaseAgent)
        orchestrator.agents.append(mock_agent)
        assert len(orchestrator.agents) == 1
        assert orchestrator.agents[0] == mock_agent

    def test_langraph_execution_result_dataclass(self):
        """Test LangGraphExecutionResult dataclass."""
        context = AgentContext(query="test")
        dag_result = DAGExecutionResult(
            final_context=context,
            success=True,
            total_execution_time_ms=1000.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            errors=[],
            execution_path=[],
            performance_metrics={},
        )

        result = LangGraphExecutionResult(
            final_context=context,
            success=True,
            execution_time_ms=1000.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            performance_metrics={"test": "metric"},
            dag_result=dag_result,
        )

        assert result.final_context == context
        assert result.success is True
        assert result.execution_time_ms == 1000.0
        assert result.nodes_executed == ["refiner"]
        assert result.edges_traversed == []
        assert result.performance_metrics == {"test": "metric"}
        assert result.dag_result == dag_result

    def test_langraph_execution_result_optional_dag_result(self):
        """Test LangGraphExecutionResult with optional dag_result."""
        context = AgentContext(query="test")

        result = LangGraphExecutionResult(
            final_context=context,
            success=False,
            execution_time_ms=500.0,
            nodes_executed=[],
            edges_traversed=[],
            performance_metrics={"error": "test"},
            # dag_result is optional and not provided
        )

        assert result.final_context == context
        assert result.success is False
        assert result.execution_time_ms == 500.0
        assert result.nodes_executed == []
        assert result.edges_traversed == []
        assert result.performance_metrics == {"error": "test"}
        assert result.dag_result is None

    @pytest.mark.asyncio
    async def test_integration_execution_tracking(self):
        """Test integration of execution tracking throughout the orchestrator."""
        orchestrator = LangGraphOrchestrator(agents_to_run=["refiner"])
        query = "integration test query"

        # Mock successful execution
        mock_dag_result = DAGExecutionResult(
            final_context=AgentContext(query=query),
            success=True,
            total_execution_time_ms=1234.0,
            nodes_executed=["refiner"],
            edges_traversed=[],
            errors=[],
            execution_path=[],
            performance_metrics={"integration": "test"},
        )

        with (
            patch.object(orchestrator, "_initialize_llm") as mock_init_llm,
            patch.object(orchestrator, "_create_agents") as mock_create_agents,
            patch.object(
                orchestrator,
                "_execute_langgraph_dag",
                return_value=LangGraphExecutionResult(
                    final_context=mock_dag_result.final_context,
                    success=True,
                    execution_time_ms=1234.0,
                    nodes_executed=["refiner"],
                    edges_traversed=[],
                    performance_metrics={"integration": "test"},
                    dag_result=mock_dag_result,
                ),
            ),
        ):
            mock_llm = Mock()
            mock_init_llm.return_value = mock_llm
            mock_create_agents.return_value = {"refiner": Mock()}

            # Execute multiple times to test tracking
            await orchestrator.run(query)
            await orchestrator.run(query)

            stats = orchestrator.get_execution_statistics()

            assert stats["total_executions"] == 2
            assert stats["successful_executions"] == 2
            assert stats["failed_executions"] == 0
            assert stats["success_rate"] == 1.0
