"""
Minimal tests for DAG Explorer module to achieve coverage.

Tests the core InteractiveDAGExplorer functionality.
"""

import pytest
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from rich.console import Console

from cognivault.diagnostics.dag_explorer import (
    InteractiveDAGExplorer,
    ExplorationMode,
    NodeInfo,
    ExplorerState,
)


class TestDAGExplorer:
    """Minimal test suite for DAG Explorer functionality."""

    @pytest.fixture
    def explorer(self) -> Any:
        """Create an InteractiveDAGExplorer instance for testing."""
        return InteractiveDAGExplorer()

    def test_initialization(self, explorer: Any) -> None:
        """Test DAG explorer initialization."""
        assert explorer.console is not None
        assert isinstance(explorer.console, Console)
        assert explorer.current_graph is None
        assert explorer.current_nodes == {}
        assert explorer.execution_history == []
        assert explorer.graph_factory is not None

    def test_create_app(self, explorer: Any) -> None:
        """Test CLI app creation."""
        app = explorer.create_app()
        assert app is not None
        assert app.info.name == "dag-explorer"

    def test_dag_node_creation(self) -> None:
        """Test DAG node creation."""
        node = NodeInfo(
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

    def test_dag_execution_creation(self) -> None:
        """Test DAG execution creation."""
        execution = ExplorerState(
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

    def test_exploration_modes(self) -> None:
        """Test exploration mode enumeration."""
        assert ExplorationMode.INTERACTIVE.value == "interactive"
        assert ExplorationMode.STRUCTURE.value == "structure"
        assert ExplorationMode.EXECUTION.value == "execution"
        assert ExplorationMode.PERFORMANCE.value == "performance"
        assert ExplorationMode.PATTERNS.value == "patterns"

    @patch("cognivault.diagnostics.dag_explorer.GraphConfig")
    def test_explore_dag_basic(self, mock_config: Mock, explorer: Any) -> None:
        """Test basic DAG exploration."""
        explorer.graph_factory = Mock()
        explorer.graph_factory.create_graph.return_value = Mock()

        with patch.object(explorer, "_analyze_graph_structure"):
            with patch.object(explorer, "_display_structure_console"):
                # Should not raise exceptions
                explorer.explore_dag(
                    pattern="standard",
                    agents="refiner,critic",
                    output="console",
                    show_details=True,
                )

    def test_analyze_graph_structure(self, explorer: Any) -> None:
        """Test graph structure analysis."""
        explorer.current_graph = Mock()

        # Should not raise exceptions
        explorer._analyze_graph_structure()

    def test_display_structure_console(self, explorer: Any) -> None:
        """Test console display of structure."""
        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_structure_console(show_details=True)

            # Should call print (even if nothing to display)
            mock_print.assert_called()

    def test_perform_structural_analysis(self, explorer: Any) -> None:
        """Test structural analysis method."""
        result = explorer._perform_structural_analysis(depth=2)

        assert isinstance(result, dict)
        assert "complexity_score" in result
        assert "parallel_branches" in result

    def test_display_structural_analysis(self, explorer: Any) -> None:
        """Test display of structural analysis results."""
        analysis = {
            "node_count": 3,
            "edge_count": 2,
            "complexity_score": 3.5,
            "parallel_branches": 2,
            "critical_path": ["start", "middle", "end"],
            "recommendations": ["Optimize path"],
        }

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_structural_analysis(analysis)

            mock_print.assert_called()

    def test_list_available_patterns(self, explorer: Any) -> None:
        """Test listing available patterns."""
        with patch.object(explorer.console, "print") as mock_print:
            explorer._list_available_patterns()

            mock_print.assert_called()

    def test_load_test_queries(self, explorer: Any) -> None:
        """Test loading test queries."""
        queries = explorer._load_test_queries(None)

        assert isinstance(queries, list)
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)

    def test_run_performance_analysis(self, explorer: Any) -> None:
        """Test performance analysis."""
        queries = ["test query"]
        agents = ["refiner"]

        with patch("asyncio.run"):
            with patch("cognivault.diagnostics.dag_explorer.LangGraphOrchestrator"):
                result = explorer._run_performance_analysis(
                    agents=agents, pattern="standard", queries=queries, runs=3
                )

                assert isinstance(result, dict)
                assert "avg_execution_time" in result
                assert "memory_usage" in result

    def test_display_performance_analysis(self, explorer: Any) -> None:
        """Test display of performance analysis."""
        data = {
            "avg_execution_time": 1.5,
            "min_execution_time": 1.0,
            "max_execution_time": 2.0,
            "success_rate": 0.95,
            "throughput": 40.0,
            "memory_usage": 256.0,
        }

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_performance_analysis(data)

            mock_print.assert_called()

    def test_validate_dag_structure(self, explorer: Any) -> None:
        """Test DAG structure validation."""
        with patch.object(explorer, "_analyze_graph_structure"):
            result = explorer._validate_dag_structure(
                ["refiner", "critic"], "standard", True
            )

            assert isinstance(result, dict)
            assert "is_valid" in result
            assert "errors" in result

    def test_display_validation_results(self, explorer: Any) -> None:
        """Test display of validation results."""
        results = {"is_valid": True, "issues": [], "warnings": ["Minor warning"]}

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_validation_results(results)

            mock_print.assert_called()

    def test_run_benchmark_suite(self, explorer: Any) -> None:
        """Test benchmark suite execution."""
        with patch("asyncio.run"):
            with patch("cognivault.diagnostics.dag_explorer.LangGraphOrchestrator"):
                result = explorer._run_benchmark_suite(["refiner"], ["standard"], 5, 2)

                assert isinstance(result, dict)
                assert "patterns_tested" in result

    def test_display_benchmark_results(self, explorer: Any) -> None:
        """Test display of benchmark results."""
        results = {
            "patterns_tested": 2,
            "total_queries": 10,
            "results": {
                "standard": {"avg_time": 1.2, "success_rate": 0.95},
                "conditional": {"avg_time": 1.4, "success_rate": 0.98},
            },
        }

        with patch.object(explorer.console, "print") as mock_print:
            explorer._display_benchmark_results(results)

            mock_print.assert_called()
