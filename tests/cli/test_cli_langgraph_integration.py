"""
Additional tests for CLI LangGraph integration features.

This module tests additional LangGraph-related CLI features including:
- --visualize-dag flag functionality
- langgraph execution mode
- DAG visualization integration
- Error handling for visualization
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from typer.testing import CliRunner

from cognivault.cli import app
from cognivault.context import AgentContext


class TestCLILangGraphVisualization:
    """Test suite for CLI DAG visualization features."""

    def test_visualize_dag_flag_available(self):
        """Test that --visualize-dag flag is available in help."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(app, ["main", "--help"])

        # Assert
        assert result.exit_code == 0
        assert "--visualize-dag" in result.output
        assert "Visualize the DAG structure" in result.output
        assert "stdout" in result.output
        assert "filepath" in result.output

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualize_dag_stdout_mode(self, mock_visualize_dag):
        """Test DAG visualization to stdout."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(app, ["main", "test query", "--visualize-dag", "stdout"])

        # Assert
        mock_visualize_dag.assert_called_once_with(
            agents=None,
            output="stdout",
            version="Phase 2.1",
            show_state_flow=True,
            show_details=True,
        )

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualize_dag_file_mode(self, mock_visualize_dag):
        """Test DAG visualization to file."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(app, ["main", "test query", "--visualize-dag", "dag.md"])

        # Assert
        mock_visualize_dag.assert_called_once_with(
            agents=None,
            output="dag.md",
            version="Phase 2.1",
            show_state_flow=True,
            show_details=True,
        )

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualize_dag_with_specific_agents(self, mock_visualize_dag):
        """Test DAG visualization with specific agents."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--agents",
                "refiner,critic",
                "--visualize-dag",
                "stdout",
            ],
        )

        # Assert
        mock_visualize_dag.assert_called_once_with(
            agents=["refiner", "critic"],
            output="stdout",
            version="Phase 2.1",
            show_state_flow=True,
            show_details=True,
        )

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualize_dag_includes_supported_agents(self, mock_visualize_dag):
        """Test DAG visualization includes all supported agents including historian."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--agents",
                "refiner,historian,critic",
                "--visualize-dag",
                "stdout",
            ],
        )

        # Assert
        # historian is now supported in Phase 2.1
        mock_visualize_dag.assert_called_once_with(
            agents=["refiner", "historian", "critic"],
            output="stdout",
            version="Phase 2.1",
            show_state_flow=True,
            show_details=True,
        )

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualize_dag_filters_unsupported_agents(self, mock_visualize_dag):
        """Test DAG visualization filters out truly unsupported agents."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--agents",
                "refiner,historian,critic,unsupported_agent",
                "--visualize-dag",
                "stdout",
            ],
        )

        # Assert
        # unsupported_agent should be filtered out
        mock_visualize_dag.assert_called_once_with(
            agents=["refiner", "historian", "critic"],
            output="stdout",
            version="Phase 2.1",
            show_state_flow=True,
            show_details=True,
        )

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualize_dag_only_mode(self, mock_visualize_dag):
        """Test DAG visualization only mode (no query execution)."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(
            app,
            ["main", "", "--visualize-dag", "stdout"],  # Empty query
        )

        # Assert
        mock_visualize_dag.assert_called_once()
        # Should not continue to execution since query is empty

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualize_dag_error_handling(self, mock_visualize_dag):
        """Test DAG visualization error handling."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.side_effect = Exception("Visualization failed")

        # Act
        result = runner.invoke(app, ["main", "test query", "--visualize-dag", "stdout"])

        # Assert
        assert "DAG visualization failed" in result.output
        assert "Visualization failed" in result.output

    def test_visualize_dag_with_execution_modes(self):
        """Test that DAG visualization works with different execution modes."""
        # Arrange
        runner = CliRunner()

        # Test with different execution modes
        modes = ["legacy", "langgraph", "langgraph-real"]

        for mode in modes:
            with patch("cognivault.cli.cli_visualize_dag") as mock_visualize_dag:
                mock_visualize_dag.return_value = None

                # Act
                result = runner.invoke(
                    app,
                    [
                        "main",
                        "",  # Empty query to skip execution
                        "--execution-mode",
                        mode,
                        "--visualize-dag",
                        "stdout",
                    ],
                )

                # Assert
                mock_visualize_dag.assert_called_once()


class TestCLILangGraphMode:
    """Test suite for CLI langgraph execution mode."""

    def test_langgraph_mode_available(self):
        """Test that langgraph mode is available in help."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(app, ["main", "--help"])

        # Assert
        assert result.exit_code == 0
        assert "langgraph" in result.output
        assert "LangGraph integration" in result.output

    @pytest.mark.asyncio
    async def test_cli_creates_langgraph_orchestrator(self):
        """Test that CLI creates LangGraphOrchestrator for langgraph mode."""
        # Arrange
        from cognivault.cli import run

        # Mock the orchestrator
        with patch("cognivault.cli.LangGraphOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator.run = AsyncMock()
            mock_orchestrator.get_execution_statistics = Mock(return_value={})
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock other dependencies
            with patch("cognivault.cli.create_llm_instance") as mock_create_llm:
                mock_create_llm.return_value = Mock()

                # Act
                try:
                    await run("test query", execution_mode="langgraph")
                except Exception:
                    pass  # We expect some errors due to mocking

                # Assert
                mock_orchestrator_class.assert_called_once_with(agents_to_run=None)

    @pytest.mark.asyncio
    async def test_cli_passes_agents_to_langgraph_orchestrator(self):
        """Test that CLI passes agents parameter to LangGraphOrchestrator."""
        # Arrange
        from cognivault.cli import run

        # Mock the orchestrator
        with patch("cognivault.cli.LangGraphOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator.run = AsyncMock()
            mock_orchestrator.get_execution_statistics = Mock(return_value={})
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock other dependencies
            with patch("cognivault.cli.create_llm_instance") as mock_create_llm:
                mock_create_llm.return_value = Mock()

                # Act
                try:
                    await run(
                        "test query",
                        agents="refiner,critic",
                        execution_mode="langgraph",
                    )
                except Exception:
                    pass  # We expect some errors due to mocking

                # Assert
                mock_orchestrator_class.assert_called_once_with(
                    agents_to_run=["refiner", "critic"]
                )

    def test_langgraph_mode_works_with_compare_modes(self):
        """Test that langgraph mode works with compare modes."""
        # Arrange
        runner = CliRunner()

        with patch("cognivault.cli._run_comparison_mode") as mock_comparison:
            mock_comparison.return_value = None

            # Act
            result = runner.invoke(
                app,
                [
                    "main",
                    "test query",
                    "--execution-mode",
                    "langgraph",
                    "--compare-modes",
                ],
            )

            # Assert - should not fail due to mode validation
            assert (
                "Must be 'legacy', 'langgraph', or 'langgraph-real'"
                not in result.output
            )

    def test_langgraph_mode_with_benchmark_runs(self):
        """Test that langgraph mode works with benchmark runs."""
        # Arrange
        runner = CliRunner()

        with patch("cognivault.cli._run_comparison_mode") as mock_comparison:
            mock_comparison.return_value = None

            # Act
            result = runner.invoke(
                app,
                [
                    "main",
                    "test query",
                    "--execution-mode",
                    "langgraph",
                    "--compare-modes",
                    "--benchmark-runs",
                    "3",
                ],
            )

            # Assert - should not fail due to mode validation
            assert (
                "Must be 'legacy', 'langgraph', or 'langgraph-real'"
                not in result.output
            )


class TestCLIHealthAndDryRun:
    """Test suite for CLI health check and dry run modes."""

    @pytest.mark.asyncio
    async def test_health_check_with_langgraph_mode(self):
        """Test health check with langgraph mode."""
        # Arrange
        from cognivault.cli import run

        # Mock the orchestrator
        with patch("cognivault.cli.LangGraphOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock health check function
            with patch("cognivault.cli._run_health_check") as mock_health_check:
                mock_health_check.return_value = None

                # Act
                try:
                    await run(
                        "test query", execution_mode="langgraph", health_check=True
                    )
                except Exception:
                    pass  # We expect some errors due to mocking

                # Assert
                mock_orchestrator_class.assert_called_once_with(agents_to_run=None)
                mock_health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_dry_run_with_langgraph_mode(self):
        """Test dry run with langgraph mode."""
        # Arrange
        from cognivault.cli import run

        # Mock the orchestrator
        with patch("cognivault.cli.LangGraphOrchestrator") as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock dry run function
            with patch("cognivault.cli._run_dry_run") as mock_dry_run:
                mock_dry_run.return_value = None

                # Act
                try:
                    await run("test query", execution_mode="langgraph", dry_run=True)
                except Exception:
                    pass  # We expect some errors due to mocking

                # Assert
                mock_orchestrator_class.assert_called_once_with(agents_to_run=None)
                mock_dry_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_with_langgraph_real_mode(self):
        """Test health check with langgraph-real mode."""
        # Arrange
        from cognivault.cli import run

        # Mock the orchestrator
        with patch(
            "cognivault.cli.RealLangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock health check function
            with patch("cognivault.cli._run_health_check") as mock_health_check:
                mock_health_check.return_value = None

                # Mock LangGraph validation
                with patch("cognivault.cli._validate_langgraph_runtime"):
                    # Act
                    try:
                        await run(
                            "test query",
                            execution_mode="langgraph-real",
                            health_check=True,
                        )
                    except Exception:
                        pass  # We expect some errors due to mocking

                    # Assert
                    mock_orchestrator_class.assert_called_once_with(agents_to_run=None)
                    mock_health_check.assert_called_once()


class TestCLILangGraphValidation:
    """Test suite for LangGraph runtime validation."""

    def test_langgraph_runtime_validation_import_error(self):
        """Test handling of LangGraph import errors."""
        # Arrange
        runner = CliRunner()

        with patch("cognivault.cli._validate_langgraph_runtime") as mock_validate:
            mock_validate.side_effect = ImportError("LangGraph not installed")

            # Act
            result = runner.invoke(
                app, ["main", "test query", "--execution-mode", "langgraph-real"]
            )

            # Assert
            assert result.exit_code == 1
            assert "LangGraph is not installed" in result.output
            assert "pip install langgraph==0.5.1" in result.output

    def test_langgraph_runtime_validation_runtime_error(self):
        """Test handling of LangGraph runtime errors."""
        # Arrange
        runner = CliRunner()

        with patch("cognivault.cli._validate_langgraph_runtime") as mock_validate:
            mock_validate.side_effect = RuntimeError("LangGraph runtime error")

            # Act
            result = runner.invoke(
                app, ["main", "test query", "--execution-mode", "langgraph-real"]
            )

            # Assert
            assert result.exit_code == 1
            assert "LangGraph runtime error" in result.output
            assert "cognivault diagnostics health" in result.output


class TestCLIIntegrationScenarios:
    """Test suite for complete CLI integration scenarios."""

    @patch("cognivault.cli.cli_visualize_dag")
    @patch("cognivault.cli.LangGraphOrchestrator")
    def test_complete_workflow_langgraph_with_visualization(
        self, mock_orchestrator_class, mock_visualize_dag
    ):
        """Test complete workflow with langgraph mode and visualization."""
        # Arrange
        runner = CliRunner()
        mock_orchestrator = Mock()
        mock_orchestrator.run = AsyncMock()
        mock_orchestrator.get_execution_statistics = Mock(return_value={})
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--execution-mode",
                "langgraph",
                "--visualize-dag",
                "stdout",
                "--agents",
                "refiner,critic",
            ],
        )

        # Assert
        mock_visualize_dag.assert_called_once_with(
            agents=["refiner", "critic"],
            output="stdout",
            version="Phase 2.1",
            show_state_flow=True,
            show_details=True,
        )
        mock_orchestrator_class.assert_called_once_with(
            agents_to_run=["refiner", "critic"]
        )

    @patch("cognivault.cli.cli_visualize_dag")
    def test_visualization_only_workflow(self, mock_visualize_dag):
        """Test workflow with only visualization (no execution)."""
        # Arrange
        runner = CliRunner()
        mock_visualize_dag.return_value = None

        # Act
        result = runner.invoke(
            app,
            ["main", "", "--visualize-dag", "dag.md"],  # Empty query
        )

        # Assert
        mock_visualize_dag.assert_called_once()
        # Should not create orchestrator since query is empty

    def test_error_handling_invalid_mode_combination(self):
        """Test error handling for invalid mode combinations."""
        # Arrange
        runner = CliRunner()

        # Act - invalid execution mode
        result = runner.invoke(
            app, ["main", "test query", "--execution-mode", "invalid-mode"]
        )

        # Assert
        assert result.exit_code != 0
        # Check that error message mentions the invalid mode
        assert "invalid-mode" in str(
            result.exception
        ) or "Invalid execution mode" in str(result.exception)

    def test_backwards_compatibility_preserved(self):
        """Test that existing CLI functionality is preserved."""
        # Arrange
        runner = CliRunner()

        # Act - legacy mode should still work
        result = runner.invoke(
            app, ["main", "test query", "--execution-mode", "legacy"]
        )

        # Assert - should not fail due to mode validation
        assert "Must be 'legacy', 'langgraph', or 'langgraph-real'" not in result.output

    def test_all_modes_in_help_text(self):
        """Test that all execution modes are documented in help."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(app, ["main", "--help"])

        # Assert
        assert result.exit_code == 0
        help_text = result.output.lower()

        # All modes should be mentioned
        assert "legacy" in help_text
        assert "langgraph" in help_text
        assert "langgraph-real" in help_text

        # Key concepts should be mentioned
        assert "execution mode" in help_text
        assert "dag" in help_text

    def test_flag_combinations_work_correctly(self):
        """Test that various flag combinations work correctly."""
        # Arrange
        runner = CliRunner()

        # Test various combinations
        test_cases = [
            ["--execution-mode", "langgraph", "--log-level", "DEBUG"],
            ["--execution-mode", "langgraph", "--trace"],
            ["--execution-mode", "langgraph", "--agents", "refiner"],
            ["--visualize-dag", "stdout", "--execution-mode", "langgraph"],
        ]

        for args in test_cases:
            # Act
            result = runner.invoke(app, ["main", "test query"] + args)

            # Assert - should not fail due to argument parsing
            assert "error" not in result.output.lower() or result.exit_code != 2
