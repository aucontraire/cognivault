"""
Additional CLI tests to improve coverage.

Focuses on untested code paths and edge cases.
"""

import pytest
from typing import Any
import os
import tempfile
import json
from unittest.mock import AsyncMock, Mock, patch
from rich.console import Console

from cognivault.cli.main_commands import (
    create_llm_instance,
    _validate_langgraph_runtime,
    _run_with_api,
    _run_health_check,
    _run_dry_run,
    _display_standard_output,
    _display_detailed_trace,
    _export_trace_data,
    _run_rollback_mode,
)
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import AgentContextFactory
from tests.factories.api_model_factories import APIModelFactory


class TestCLIUtilityFunctions:
    """Test CLI utility functions."""

    def test_create_llm_instance(self) -> None:
        """Test LLM instance creation."""
        with (
            patch("cognivault.cli.main_commands.OpenAIConfig") as mock_config_class,
            patch("cognivault.cli.main_commands.OpenAIChatLLM") as mock_llm_class,
        ):
            mock_config: Mock = Mock()
            mock_config.api_key = "test-key"
            mock_config.model = "gpt-4"
            mock_config.base_url = "https://api.openai.com"
            mock_config_class.load.return_value = mock_config

            mock_llm: Mock = Mock()
            mock_llm_class.return_value = mock_llm

            result = create_llm_instance()

            assert result is mock_llm
            mock_config_class.load.assert_called_once()
            mock_llm_class.assert_called_once_with(
                api_key="test-key", model="gpt-4", base_url="https://api.openai.com"
            )

    def test_validate_langgraph_runtime_success(self) -> None:
        """Test successful LangGraph runtime validation."""
        # Create mock modules
        mock_langgraph: Mock = Mock()
        mock_langgraph.__version__ = "0.6.4"

        mock_state_graph: Mock = Mock()
        mock_graph_instance: Mock = Mock()
        mock_state_graph.return_value = mock_graph_instance
        mock_app: Mock = Mock()
        mock_graph_instance.compile.return_value = mock_app

        mock_langgraph_graph: Mock = Mock()
        mock_langgraph_graph.StateGraph = mock_state_graph
        mock_langgraph_graph.END = "END"

        mock_langgraph_checkpoint: Mock = Mock()
        mock_langgraph_checkpoint.MemorySaver = Mock()

        # Mock the modules in sys.modules before import
        modules = {
            "langgraph": mock_langgraph,
            "langgraph.graph": mock_langgraph_graph,
            "langgraph.checkpoint": Mock(),
            "langgraph.checkpoint.memory": mock_langgraph_checkpoint,
        }

        with patch.dict("sys.modules", modules):
            # Should not raise an exception
            _validate_langgraph_runtime()

            # Verify the graph was created and configured correctly
            mock_state_graph.assert_called_once()
            mock_graph_instance.add_node.assert_called_once()
            mock_graph_instance.add_edge.assert_called_once_with("test", "END")
            mock_graph_instance.set_entry_point.assert_called_once_with("test")
            mock_graph_instance.compile.assert_called_once()

    def test_validate_langgraph_runtime_wrong_version(self) -> None:
        """Test LangGraph validation with wrong version."""
        # Mock langgraph module with incompatible version
        mock_langgraph: Mock = Mock()
        mock_langgraph.__version__ = "0.4.0"

        with patch.dict("sys.modules", {"langgraph": mock_langgraph}):
            with pytest.raises(
                RuntimeError, match="LangGraph version 0.4.0 may not be compatible"
            ):
                _validate_langgraph_runtime()

    def test_validate_langgraph_runtime_import_error(self) -> None:
        """Test LangGraph validation with import error."""

        # Create a mock that raises ImportError when langgraph is imported
        def import_side_effect(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langgraph":
                raise ImportError("No module named 'langgraph'")
            return Mock()  # For other imports

        with patch("builtins.__import__", side_effect=import_side_effect):
            with pytest.raises(ImportError, match="LangGraph import failed"):
                _validate_langgraph_runtime()

    def test_validate_langgraph_runtime_compilation_error(self) -> None:
        """Test LangGraph validation with compilation error."""
        # Create mock modules
        mock_langgraph: Mock = Mock()
        mock_langgraph.__version__ = "0.6.4"

        mock_state_graph: Mock = Mock()
        mock_graph_instance: Mock = Mock()
        mock_state_graph.return_value = mock_graph_instance
        mock_graph_instance.compile.side_effect = Exception("Compilation failed")

        mock_langgraph_graph: Mock = Mock()
        mock_langgraph_graph.StateGraph = mock_state_graph
        mock_langgraph_graph.END = "END"

        mock_langgraph_checkpoint: Mock = Mock()
        mock_langgraph_checkpoint.MemorySaver = Mock()

        # Mock the modules in sys.modules
        modules = {
            "langgraph": mock_langgraph,
            "langgraph.graph": mock_langgraph_graph,
            "langgraph.checkpoint": Mock(),
            "langgraph.checkpoint.memory": mock_langgraph_checkpoint,
        }

        with patch.dict("sys.modules", modules):
            with pytest.raises(
                RuntimeError, match="LangGraph runtime validation failed"
            ):
                _validate_langgraph_runtime()


class TestCLIRunModes:
    """Test different CLI run modes."""

    @pytest.mark.asyncio
    async def test_run_with_api_success(self) -> None:
        """Test successful API execution."""
        console = Console()

        with (
            patch("cognivault.cli.main_commands.initialize_api") as mock_init_api,
            patch("cognivault.cli.main_commands.shutdown_api") as mock_shutdown_api,
            patch("cognivault.cli.main_commands.get_api_mode", return_value="mock"),
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            # Mock API
            mock_api = AsyncMock()
            mock_init_api.return_value = mock_api

            # Mock workflow response
            mock_response = APIModelFactory.create_valid_workflow_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="completed",
                agent_outputs={
                    "refiner": "Refined output",
                    "critic": "Critical analysis",
                },
                execution_time_seconds=2.5,
                correlation_id="test-corr",
            )
            mock_api.execute_workflow.return_value = mock_response

            result = await _run_with_api(
                query="Test query",
                agents_to_run=["refiner", "critic"],
                console=console,
                trace=True,
                execution_mode="langgraph-real",
                api_mode="mock",
            )

            assert isinstance(result, AgentContext)
            assert result.query == "Test query"
            assert "refiner" in result.agent_outputs
            assert "critic" in result.agent_outputs
            assert hasattr(result, "metadata")
            assert "workflow_id" in result.metadata  # UUID generated dynamically
            assert result.metadata["api_mode"] == "mock"

            mock_init_api.assert_called_once()
            mock_shutdown_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_api_failure(self) -> None:
        """Test API execution failure."""
        console = Console()

        with (
            patch("cognivault.cli.main_commands.initialize_api") as mock_init_api,
            patch("cognivault.cli.main_commands.shutdown_api") as mock_shutdown_api,
            patch("cognivault.cli.main_commands.get_api_mode", return_value="mock"),
        ):
            # Mock API that raises an exception
            mock_api = AsyncMock()
            mock_init_api.return_value = mock_api
            mock_api.execute_workflow.side_effect = Exception("API execution failed")

            result = await _run_with_api(
                query="Test query",
                agents_to_run=["refiner"],
                console=console,
                trace=False,
                execution_mode="langgraph-real",
                api_mode=None,
            )

            assert isinstance(result, AgentContext)
            assert result.query == "Test query"
            assert "error" in result.agent_outputs
            assert "API execution failed" in result.agent_outputs["error"]
            assert result.metadata["error"] == "API execution failed"

            mock_shutdown_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_api_failed_workflow(self) -> None:
        """Test API execution with failed workflow response."""
        console = Console()

        with (
            patch("cognivault.cli.main_commands.initialize_api") as mock_init_api,
            patch("cognivault.cli.main_commands.shutdown_api") as mock_shutdown_api,
            patch("cognivault.cli.main_commands.get_api_mode", return_value="mock"),
        ):
            mock_api = AsyncMock()
            mock_init_api.return_value = mock_api

            # Mock failed workflow response
            mock_response = APIModelFactory.create_valid_workflow_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440001",
                status="failed",
                agent_outputs={},
                execution_time_seconds=1.0,
                error_message="Workflow execution failed",
            )
            mock_api.execute_workflow.return_value = mock_response

            result = await _run_with_api(
                query="Test query",
                agents_to_run=["refiner"],
                console=console,
                trace=False,
                execution_mode="langgraph-real",
                api_mode=None,
            )

            assert result.metadata["api_status"] == "failed"

    @pytest.mark.asyncio
    async def test_run_health_check(self) -> None:
        """Test health check mode."""
        console = Console()

        # Mock orchestrator with registry
        mock_orchestrator: Mock = Mock()
        mock_registry: Mock = Mock()
        mock_orchestrator.registry = mock_registry

        # Mock successful health checks
        mock_registry.check_health.return_value = True

        agents_to_run = ["refiner", "critic"]

        with patch.object(console, "print") as mock_print:
            await _run_health_check(mock_orchestrator, console, agents_to_run)

            # Should print health table and success message
            assert mock_print.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_health_check_failures(self) -> None:
        """Test health check mode with failures."""
        console = Console()

        mock_orchestrator: Mock = Mock()
        mock_registry: Mock = Mock()
        mock_orchestrator.registry = mock_registry

        # Mock failed health check
        mock_registry.check_health.side_effect = [
            True,
            False,
        ]  # First succeeds, second fails

        agents_to_run = ["refiner", "critic"]

        with patch.object(console, "print") as mock_print:
            await _run_health_check(mock_orchestrator, console, agents_to_run)

            # Should print health table and failure message
            assert mock_print.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_health_check_exception(self) -> None:
        """Test health check mode with exception."""
        console = Console()

        mock_orchestrator: Mock = Mock()
        mock_registry: Mock = Mock()
        mock_orchestrator.registry = mock_registry

        # Mock exception during health check
        mock_registry.check_health.side_effect = Exception("Health check error")

        agents_to_run = ["refiner"]

        with patch.object(console, "print") as mock_print:
            await _run_health_check(mock_orchestrator, console, agents_to_run)

            # Should handle exception and show error
            assert mock_print.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_health_check_default_agents(self) -> None:
        """Test health check mode with default agents."""
        console = Console()

        mock_orchestrator: Mock = Mock()
        mock_registry: Mock = Mock()
        mock_orchestrator.registry = mock_registry
        mock_registry.check_health.return_value = True

        agents_to_run = None  # Use default agents

        await _run_health_check(mock_orchestrator, console, agents_to_run)

        # Should check default agents
        expected_calls = 4  # refiner, critic, historian, synthesis
        assert mock_registry.check_health.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_run_dry_run(self) -> None:
        """Test dry run mode."""
        console = Console()
        query = "Test query for dry run"
        agents_to_run = ["refiner", "critic"]

        # Mock orchestrator with agents
        mock_agent1: Mock = Mock()
        mock_agent1.name = "refiner"
        mock_agent2: Mock = Mock()
        mock_agent2.name = "critic"

        mock_orchestrator: Mock = Mock()
        mock_orchestrator.agents = [mock_agent1, mock_agent2]

        with (
            patch(
                "cognivault.cli.main_commands._run_health_check"
            ) as mock_health_check,
            patch.object(console, "print") as mock_print,
        ):
            await _run_dry_run(mock_orchestrator, console, query, agents_to_run)

            # Should call health check
            mock_health_check.assert_called_once()

            # Should print configuration and validation info
            assert mock_print.call_count >= 3


class TestCLIDisplayFunctions:
    """Test CLI display functions."""

    def test_display_standard_output(self) -> None:
        """Test standard output display."""
        console = Console()

        context = AgentContextFactory.basic(
            query="Test query",
            agent_outputs={
                "Refiner": "Refined analysis of the test query",
                "Critic": "Critical evaluation of the analysis",
                "Synthesis": "Synthesized conclusions",
            },
        )
        context.successful_agents = {"Refiner", "Critic", "Synthesis"}
        context.failed_agents = set()

        execution_time = 5.2

        with patch.object(console, "print") as mock_print:
            _display_standard_output(console, context, execution_time)

            # Should print performance summary and agent outputs
            assert mock_print.call_count >= 5  # Performance + 3 agents

    def test_display_standard_output_with_failures(self) -> None:
        """Test standard output display with failed agents."""
        console = Console()

        context = AgentContextFactory.basic(
            query="Test query",
            agent_outputs={
                "Refiner": "Refined analysis",
                "Critic": "Critical evaluation",
            },
        )
        context.successful_agents = {"Refiner", "Critic"}
        context.failed_agents = {"Historian"}

        execution_time = 3.1

        with patch.object(console, "print") as mock_print:
            _display_standard_output(console, context, execution_time)

            # Should print failure info
            assert mock_print.call_count >= 4

    def test_display_detailed_trace(self) -> None:
        """Test detailed trace display."""
        console = Console()

        context = AgentContextFactory.basic(
            query="Test query",
            agent_outputs={"Refiner": "Refined output", "Critic": "Critical analysis"},
        )
        context.context_id = "test-context-123"
        context.successful_agents = {"Refiner", "Critic"}
        context.failed_agents = set()
        context.current_size = 1024
        context.agent_execution_status = {"Refiner": "completed", "Critic": "completed"}
        context.execution_edges = [
            {"from_agent": "START", "to_agent": "Refiner", "edge_type": "normal"},
            {"from_agent": "Refiner", "to_agent": "Critic", "edge_type": "normal"},
        ]
        context.conditional_routing = {
            "refiner_output": "Chose critic path based on content analysis"
        }

        execution_time = 4.7

        with patch.object(console, "print") as mock_print:
            _display_detailed_trace(console, context, execution_time)

            # Should print comprehensive trace info
            assert mock_print.call_count >= 8

    def test_display_detailed_trace_minimal(self) -> None:
        """Test detailed trace display with minimal context."""
        console = Console()

        context = AgentContextFactory.basic(
            query="Minimal test", agent_outputs={"Refiner": "Basic output"}
        )
        context.context_id = "minimal-123"
        context.successful_agents = {"Refiner"}
        context.failed_agents = set()
        context.current_size = 512

        execution_time = 1.0

        with patch.object(console, "print") as mock_print:
            _display_detailed_trace(console, context, execution_time)

            # Should still print basic trace info
            assert mock_print.call_count >= 3

    def test_export_trace_data(self) -> None:
        """Test trace data export."""
        context = AgentContextFactory.basic(
            query="Export test query", agent_outputs={"refiner": "Exported output"}
        )
        context.context_id = "export-123"
        context.successful_agents = {"refiner"}
        context.failed_agents = set()
        context.agent_execution_status = {"refiner": "completed"}
        context.execution_edges = []
        context.conditional_routing = {}
        context.path_metadata = {}
        context.agent_trace = {}
        context.current_size = 256
        context.execution_state = {}

        execution_time = 2.3

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            _export_trace_data(context, export_path, execution_time)

            # Verify file was created and has correct content
            assert os.path.exists(export_path)

            with open(export_path, "r") as f:
                trace_data = json.load(f)

            assert trace_data["pipeline_id"] == "export-123"
            assert trace_data["execution_time_seconds"] == 2.3
            assert trace_data["query"] == "Export test query"
            assert "refiner" in trace_data["agent_outputs"]
            assert "timestamp" in trace_data

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestCLIRollbackMode:
    """Test CLI rollback mode functionality."""

    @pytest.mark.asyncio
    async def test_run_rollback_mode_success(self) -> None:
        """Test successful rollback mode."""
        console = Console()
        thread_id = "test-thread-123"

        # Mock orchestrator with rollback capability
        mock_orchestrator: Mock = Mock()
        mock_orchestrator.rollback_to_checkpoint = AsyncMock()

        # Mock restored context
        restored_context = AgentContextFactory.basic(
            query="Restored query",
            agent_outputs={
                "refiner": "Restored refiner output",
                "critic": "Restored critic output",
            },
        )
        mock_orchestrator.rollback_to_checkpoint.return_value = restored_context

        # Mock checkpoint history
        mock_orchestrator.get_checkpoint_history = Mock()
        mock_orchestrator.get_checkpoint_history.return_value = [
            {
                "agent_step": "critic_completed",
                "timestamp": "2024-01-01T10:00:00Z",
                "state_size_bytes": 1024,
            }
        ]

        with patch.object(console, "print") as mock_print:
            await _run_rollback_mode(mock_orchestrator, console, thread_id)

            mock_orchestrator.rollback_to_checkpoint.assert_called_once_with(
                thread_id=thread_id
            )
            # Should print success and restoration info
            assert mock_print.call_count >= 5

    @pytest.mark.asyncio
    async def test_run_rollback_mode_no_checkpoint(self) -> None:
        """Test rollback mode when no checkpoint exists."""
        console = Console()
        thread_id = "no-checkpoint-thread"

        mock_orchestrator: Mock = Mock()
        mock_orchestrator.rollback_to_checkpoint = AsyncMock()
        mock_orchestrator.rollback_to_checkpoint.return_value = None  # No checkpoint

        with patch.object(console, "print") as mock_print:
            await _run_rollback_mode(mock_orchestrator, console, thread_id)

            # Should print warning about no checkpoint
            assert mock_print.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_rollback_mode_no_thread_id(self) -> None:
        """Test rollback mode without thread ID."""
        console = Console()
        thread_id = "no-thread-id"

        mock_orchestrator: Mock = Mock()
        mock_orchestrator.rollback_to_checkpoint = AsyncMock()
        mock_orchestrator.rollback_to_checkpoint.return_value = None

        with patch.object(console, "print") as mock_print:
            await _run_rollback_mode(mock_orchestrator, console, thread_id)

            # Should print guidance about thread ID
            assert mock_print.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_rollback_mode_not_supported(self) -> None:
        """Test rollback mode with orchestrator that doesn't support rollback."""
        console = Console()
        thread_id = "test-thread"

        # Mock orchestrator without rollback capability
        mock_orchestrator: Mock = Mock()
        # Don't add rollback_to_checkpoint attribute

        with patch.object(console, "print") as mock_print:
            await _run_rollback_mode(mock_orchestrator, console, thread_id)

            # Should print error about not supported
            assert mock_print.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_rollback_mode_exception(self) -> None:
        """Test rollback mode with exception."""
        console = Console()
        thread_id = "error-thread"

        mock_orchestrator: Mock = Mock()
        mock_orchestrator.rollback_to_checkpoint = AsyncMock()
        mock_orchestrator.rollback_to_checkpoint.side_effect = Exception(
            "Rollback failed"
        )

        with patch.object(console, "print") as mock_print:
            await _run_rollback_mode(mock_orchestrator, console, thread_id)

            # Should print error message
            assert mock_print.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_rollback_mode_empty_outputs(self) -> None:
        """Test rollback mode with context that has no agent outputs."""
        console = Console()
        thread_id = "empty-thread"

        mock_orchestrator: Mock = Mock()
        mock_orchestrator.rollback_to_checkpoint = AsyncMock()

        # Mock restored context with no outputs
        restored_context = AgentContextFactory.basic(
            query="Restored query", agent_outputs={}
        )
        mock_orchestrator.rollback_to_checkpoint.return_value = restored_context

        with patch.object(console, "print") as mock_print:
            await _run_rollback_mode(mock_orchestrator, console, thread_id)

            # Should print message about no outputs
            assert mock_print.call_count >= 3

    @pytest.mark.asyncio
    async def test_run_rollback_mode_long_outputs(self) -> None:
        """Test rollback mode with long agent outputs (truncation)."""
        console = Console()
        thread_id = "long-thread"

        mock_orchestrator: Mock = Mock()
        mock_orchestrator.rollback_to_checkpoint = AsyncMock()

        # Mock restored context with long output
        long_output = "x" * 300  # Longer than 200 char limit
        restored_context = AgentContextFactory.basic(
            query="Restored query", agent_outputs={"refiner": long_output}
        )
        mock_orchestrator.rollback_to_checkpoint.return_value = restored_context

        with patch.object(console, "print") as mock_print:
            await _run_rollback_mode(mock_orchestrator, console, thread_id)

            # Should truncate long outputs
            assert mock_print.call_count >= 4
