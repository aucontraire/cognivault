"""
Comprehensive tests for DAG Explorer module.

Tests the InteractiveDAGExplorer class with various graph patterns,
exploration modes, and analysis capabilities.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from rich.console import Console
from pathlib import Path

from cognivault.diagnostics.dag_explorer import (
    InteractiveDAGExplorer,
    ExplorationMode,
    DAGNode,
    DAGExecution,
)
from cognivault.context import AgentContext
from cognivault.langgraph_backend.build_graph import GraphConfig


class TestInteractiveDAGExplorer:
    """Test suite for InteractiveDAGExplorer class."""

    @pytest.fixture
    def explorer(self):
        """Create an InteractiveDAGExplorer instance for testing."""
        return InteractiveDAGExplorer()

    @pytest.fixture
    def mock_graph_factory(self):
        """Mock GraphFactory for testing."""
        factory = Mock()
        factory.create_graph.return_value = Mock()
        return factory

    @pytest.fixture
    def sample_dag_structure(self):
        """Sample DAG structure for testing."""
        return {
            "pattern_name": "test_pattern",
            "total_nodes": 5,
            "total_edges": 4,
            "entry_points": ["start"],
            "exit_points": ["end"],
            "node_types": {"agent": 3, "conditional": 1, "end": 1},
            "dependency_graph": {"start": ["node1"], "node1": ["end"]},
            "critical_path": ["start", "node1", "end"],
            "complexity_metrics": {"branching_factor": 1.2, "depth": 3},
        }

    @pytest.fixture
    def sample_dag_execution(self):
        """Sample DAG execution for testing."""
        return DAGExecution(
            execution_id="test_exec_123",
            nodes_executed=["start", "refiner", "critic", "end"],
            execution_path=[
                ("start", "refiner"),
                ("refiner", "critic"),
                ("critic", "end"),
            ],
            timing_data={"refiner": 1.0, "critic": 1.2},
            conditional_decisions={},
            total_duration=2.5,
            success=True,
        )

    def test_initialization(self, explorer):
        """Test InteractiveDAGExplorer initialization."""
        assert explorer.console is not None
        assert isinstance(explorer.console, Console)
        assert explorer.current_graph is None
        assert explorer.current_nodes == {}
        assert explorer.graph_factory is not None

    def test_create_app(self, explorer):
        """Test CLI app creation."""
        app = explorer.create_app()
        assert app is not None
        assert app.info.name == "dag-explorer"

    @patch("cognivault.diagnostics.dag_explorer.GraphConfig")
    def test_explore_dag_console_output(
        self, mock_config, explorer, mock_graph_factory
    ):
        """Test DAG structure exploration with console output."""
        explorer.graph_factory = mock_graph_factory

        with patch.object(explorer, "_analyze_graph_structure"):
            with patch.object(explorer, "_display_structure_console"):
                # Test console output
                explorer.explore_dag(
                    pattern="standard",
                    agents="refiner,critic",
                    output="console",
                    show_details=True,
                )

                mock_config.assert_called_once()
                mock_graph_factory.create_graph.assert_called_once()

    @patch("cognivault.diagnostics.dag_explorer.GraphConfig")
    def test_explore_structure_json_output(
        self, mock_config, explorer, mock_graph_factory, capsys
    ):
        """Test DAG structure exploration with JSON output."""
        explorer.graph_factory = mock_graph_factory

        with patch.object(explorer, "_analyze_graph_structure"):
            explorer.explore_dag(
                pattern="conditional",
                agents="refiner,critic,synthesis",
                output="json",
                show_details=False,
            )

            captured = capsys.readouterr()
            # Should contain JSON structure
            assert '"pattern": "conditional"' in captured.out
            assert '"agents":' in captured.out

    @patch("cognivault.diagnostics.dag_explorer.GraphConfig")
    def test_explore_structure_dot_output(
        self, mock_config, explorer, mock_graph_factory, capsys
    ):
        """Test DAG structure exploration with DOT output."""
        explorer.graph_factory = mock_graph_factory

        with patch.object(explorer, "_analyze_graph_structure"):
            explorer.explore_dag(
                pattern="sequential", agents="refiner", output="dot", show_details=False
            )

            captured = capsys.readouterr()
            # Should contain DOT format
            assert "digraph DAG {" in captured.out
            assert "}" in captured.out

    def test_explore_structure_error_handling(self, explorer):
        """Test error handling during structure exploration."""
        # Mock the graph factory to raise an exception
        with patch.object(
            explorer.graph_factory,
            "create_graph",
            side_effect=Exception("Graph creation failed"),
        ):
            # typer.Exit raises click.exceptions.Exit, not SystemExit
            from click.exceptions import Exit

            with pytest.raises(Exit):
                explorer.explore_dag(
                    pattern="invalid", agents="nonexistent", output="console"
                )

    @patch("cognivault.diagnostics.dag_explorer.GraphConfig")
    def test_analyze_structure(self, mock_config, explorer, mock_graph_factory):
        """Test DAG structural analysis."""
        explorer.graph_factory = mock_graph_factory

        with patch.object(explorer, "_analyze_graph_structure"):
            with patch.object(
                explorer, "_perform_structural_analysis"
            ) as mock_analysis:
                with patch.object(explorer, "_display_structural_analysis"):
                    mock_analysis.return_value = {"complexity": "medium"}

                    explorer.analyze_structure(
                        pattern="standard", agents="refiner,critic", depth="detailed"
                    )

                    mock_analysis.assert_called_once_with("detailed")

    @patch("asyncio.run")
    @patch("cognivault.diagnostics.dag_explorer.RealLangGraphOrchestrator")
    def test_trace_execution_without_live(
        self, mock_orchestrator, mock_asyncio, explorer
    ):
        """Test DAG execution tracing without live monitoring."""
        mock_orchestrator_instance = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance

        # Mock the async execution
        mock_execution = Mock()
        mock_asyncio.return_value = mock_execution

        with patch.object(explorer, "_display_execution_trace"):
            explorer.trace_execution(
                query="Test query",
                agents="refiner,critic",
                pattern="standard",
                live_trace=False,
            )

            mock_orchestrator.assert_called_once()
            mock_asyncio.assert_called()

    @patch("asyncio.run")
    @patch("cognivault.diagnostics.dag_explorer.RealLangGraphOrchestrator")
    def test_trace_execution_with_live(
        self, mock_orchestrator, mock_asyncio, explorer, capsys
    ):
        """Test DAG execution tracing with live monitoring."""
        mock_orchestrator_instance = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance

        # Mock the async execution
        mock_execution = Mock()
        mock_asyncio.return_value = mock_execution

        with patch.object(explorer, "_display_execution_trace"):
            explorer.trace_execution(
                query="Test query with live trace",
                agents="refiner,critic,synthesis",
                pattern="conditional",
                live_trace=True,
            )

            captured = capsys.readouterr()
            assert "Live tracing would be implemented here" in captured.out

    def test_analyze_performance_basic(self, explorer):
        """Test basic performance analysis."""
        with patch.object(explorer, "_analyze_graph_structure"):
            with patch.object(explorer, "_run_performance_analysis") as mock_benchmark:
                with patch.object(explorer, "_display_performance_analysis"):
                    mock_benchmark.return_value = {"avg_duration": 1.5}

                    explorer.analyze_performance(
                        agents="refiner,critic",
                        pattern="standard",
                        runs=3,
                        queries_file=None,
                    )

                    mock_benchmark.assert_called_once()

    def test_explore_patterns_specific(self, explorer, capsys):
        """Test specific pattern exploration."""
        explorer.explore_patterns(pattern="conditional", compare=False, validate=False)

        captured = capsys.readouterr()
        assert "Exploring pattern: conditional" in captured.out

    def test_explore_patterns_compare(self, explorer, capsys):
        """Test pattern comparison."""
        explorer.explore_patterns(pattern=None, compare=True, validate=False)

        captured = capsys.readouterr()
        assert "Pattern Comparison" in captured.out
        assert "standard" in captured.out
        assert "conditional" in captured.out

    def test_explore_patterns_validate(self, explorer, capsys):
        """Test pattern validation."""
        explorer.explore_patterns(pattern=None, compare=False, validate=True)

        captured = capsys.readouterr()
        assert "Pattern Validation Results" in captured.out
        assert "All patterns passed validation" in captured.out

    def test_explore_patterns_list_all(self, explorer):
        """Test listing all available patterns."""
        with patch.object(explorer, "_list_available_patterns") as mock_list:
            explorer.explore_patterns(pattern=None, compare=False, validate=False)

            mock_list.assert_called_once()

    def test_interactive_mode_basic(self, explorer):
        """Test interactive mode initialization."""
        with patch.object(explorer, "_start_interactive_session") as mock_session:
            explorer.interactive_mode(agents="refiner,critic", pattern="standard")

            mock_session.assert_called_once()

    def test_analyze_graph_structure(self, explorer, sample_dag_structure):
        """Test graph structure analysis."""
        # Mock current graph
        explorer.current_graph = Mock()

        # Test the method (it's a placeholder implementation)
        explorer._analyze_graph_structure()

        # Should not raise any exceptions
        assert True

    def test_display_structure_console(self, explorer, sample_dag_structure):
        """Test console display of structure."""
        explorer.current_structure = sample_dag_structure

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_structure_console(show_details=True)

            # Should print structure information
            mock_print.assert_called()

    def test_display_structure_console_no_structure(self, explorer):
        """Test console display when no structure is available."""
        explorer.current_structure = None

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_structure_console(show_details=True)

            # Should handle None structure gracefully
            mock_print.assert_called()

    def test_perform_structural_analysis(self, explorer):
        """Test structural analysis method."""
        result = explorer._perform_structural_analysis("basic")

        assert isinstance(result, dict)
        assert "complexity_score" in result
        assert "parallel_branches" in result

    def test_display_structural_analysis(self, explorer):
        """Test display of structural analysis results."""
        analysis = {
            "node_count": 3,
            "edge_count": 2,
            "parallel_branches": 2,
            "complexity_score": 3.5,
            "critical_path": ["start", "middle", "end"],
        }

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_structural_analysis(analysis)

            mock_print.assert_called()

    def test_execute_and_trace_async(self, explorer):
        """Test async execution and tracing."""
        import asyncio

        async def test_execution():
            mock_orchestrator = Mock()
            mock_orchestrator.run = AsyncMock(return_value=AgentContext(query="test"))

            result = await explorer._execute_and_trace(mock_orchestrator, "test query")

            assert isinstance(result, DAGExecution)
            assert result.execution_id.startswith("exec_")
            assert result.success is True

        # Run the async test
        asyncio.run(test_execution())

    def test_display_execution_trace(self, explorer, sample_dag_execution):
        """Test display of execution trace."""
        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_execution_trace(sample_dag_execution)

            mock_print.assert_called()

    def test_run_performance_benchmark(self, explorer):
        """Test performance benchmarking."""
        result = explorer._run_performance_analysis(
            agents=["refiner", "critic"],
            pattern="standard",
            queries=["test query"],
            runs=3,
        )

        assert isinstance(result, dict)
        assert "avg_execution_time" in result
        assert "memory_usage" in result

    def test_display_performance_analysis(self, explorer):
        """Test display of performance analysis."""
        analysis = {
            "avg_execution_time": 1.5,
            "min_execution_time": 1.0,
            "max_execution_time": 2.0,
            "success_rate": 0.95,
            "throughput": 40.0,
            "memory_usage": 256.0,
        }

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_performance_analysis(analysis)

            mock_print.assert_called()

    def test_list_available_patterns(self, explorer):
        """Test listing available patterns."""
        with patch.object(explorer.console, "print") as mock_print:
            explorer._list_available_patterns()

            mock_print.assert_called()

    def test_start_interactive_session(self, explorer):
        """Test interactive session startup."""
        with patch("rich.prompt.Prompt.ask", side_effect=["help", "exit"]):
            with patch.object(explorer, "_show_interactive_help") as mock_help:
                explorer._start_interactive_session(["refiner", "critic"], "standard")

                # Should call help when help command is entered
                mock_help.assert_called_once()

    def test_show_interactive_help(self, explorer):
        """Test interactive help display."""
        with patch.object(explorer.console, "print") as mock_print:
            explorer._show_interactive_help()

            mock_print.assert_called()


class TestDAGNode:
    """Test suite for DAGNode dataclass."""

    def test_dag_node_creation(self):
        """Test DAG node creation."""
        node = DAGNode(
            name="test_node",
            type="agent",
            agent_class="refiner",
            dependencies=["start"],
            execution_time=1.5,
        )

        assert node.name == "test_node"
        assert node.type == "agent"
        assert node.agent_class == "refiner"
        assert len(node.dependencies) == 1
        assert node.execution_time == 1.5


class TestDAGExecution:
    """Test suite for DAGExecution dataclass."""

    def test_dag_execution_creation(self):
        """Test DAG execution creation."""
        execution = DAGExecution(
            execution_id="test_123",
            nodes_executed=["start", "end"],
            execution_path=[("start", "end")],
            timing_data={"start": 0.5},
            conditional_decisions={},
            total_duration=1.0,
            success=True,
        )

        assert execution.execution_id == "test_123"
        assert execution.success is True
        assert execution.total_duration == 1.0
        assert len(execution.nodes_executed) == 2

    def test_dag_execution_with_error(self):
        """Test DAG execution with error details."""
        execution = DAGExecution(
            execution_id="error_123",
            nodes_executed=["start"],
            execution_path=[],
            timing_data={},
            conditional_decisions={},
            total_duration=0.5,
            success=False,
            error_node="start",
        )

        assert execution.success is False
        assert execution.error_node == "start"


class TestExplorationMode:
    """Test suite for ExplorationMode enum."""

    def test_exploration_modes(self):
        """Test exploration mode enumeration."""
        assert ExplorationMode.INTERACTIVE.value == "interactive"
        assert ExplorationMode.STRUCTURE.value == "structure"
        assert ExplorationMode.EXECUTION.value == "execution"
        assert ExplorationMode.PERFORMANCE.value == "performance"
        assert ExplorationMode.PATTERNS.value == "patterns"


# Integration tests
class TestDAGExplorerIntegration:
    """Integration tests for DAG Explorer."""

    @pytest.fixture
    def explorer(self):
        """Create explorer for integration tests."""
        return InteractiveDAGExplorer()

    def test_full_exploration_workflow(self, explorer):
        """Test complete exploration workflow."""
        with patch.object(explorer, "graph_factory") as mock_factory:
            mock_factory.create_graph.return_value = Mock()

            with patch.object(explorer, "_analyze_graph_structure"):
                with patch.object(explorer, "_display_structure_console"):
                    # Should not raise exceptions
                    explorer.explore_dag(
                        pattern="standard", agents="refiner,critic", output="console"
                    )

    def test_cli_app_integration(self, explorer):
        """Test CLI app creation and basic functionality."""
        app = explorer.create_app()

        # Test that app was created successfully
        assert app is not None
        assert app.info.name == "dag-explorer"


# Performance tests
class TestDAGExplorerPerformance:
    """Performance tests for DAG Explorer."""

    @pytest.fixture
    def explorer(self):
        """Create explorer for performance tests."""
        return InteractiveDAGExplorer()

    def test_structure_analysis_performance(self, explorer):
        """Test performance of structure analysis."""
        import time

        start_time = time.time()
        result = explorer._perform_structural_analysis("basic")
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 1.0
        assert isinstance(result, dict)

    def test_benchmark_performance(self, explorer):
        """Test performance benchmarking functionality."""
        import time

        start_time = time.time()
        result = explorer._run_performance_analysis(
            agents=["refiner"], pattern="standard", queries=["test query"], runs=10
        )
        end_time = time.time()

        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0
        assert isinstance(result, dict)
        assert "avg_execution_time" in result
        assert isinstance(result, dict)
