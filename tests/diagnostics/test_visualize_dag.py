"""
Tests for the DAG visualization utility.

This module tests the DAG visualization functionality including:
- DAGVisualizationConfig dataclass
- DAGVisualizer class methods
- Mermaid diagram generation
- File and stdout output modes
- CLI integration functions
- Edge case handling
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from cognivault.diagnostics.visualize_dag import (
    DAGVisualizationConfig,
    DAGVisualizer,
    create_dag_visualization,
    cli_visualize_dag,
    get_default_agents,
    validate_agents,
)


class TestDAGVisualizationConfig:
    """Test DAGVisualizationConfig dataclass."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = DAGVisualizationConfig()

        assert config.version == "Phase 2.2"
        assert config.show_state_flow is True
        assert config.show_node_details is True
        assert config.include_metadata is True

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = DAGVisualizationConfig(
            version="Phase 3.0",
            show_state_flow=False,
            show_node_details=False,
            include_metadata=False,
        )

        assert config.version == "Phase 3.0"
        assert config.show_state_flow is False
        assert config.show_node_details is False
        assert config.include_metadata is False

    def test_config_partial_overrides(self):
        """Test config allows partial overrides."""
        config = DAGVisualizationConfig(version="Custom Version")

        assert config.version == "Custom Version"
        assert config.show_state_flow is True  # Default
        assert config.show_node_details is True  # Default
        assert config.include_metadata is True  # Default


class TestDAGVisualizer:
    """Test DAGVisualizer class functionality."""

    def test_visualizer_initialization_default(self):
        """Test visualizer initialization with default config."""
        visualizer = DAGVisualizer()

        assert visualizer.config.version == "Phase 2.2"
        assert visualizer.config.show_state_flow is True
        assert visualizer.config.show_node_details is True
        assert visualizer.config.include_metadata is True

    def test_visualizer_initialization_custom_config(self):
        """Test visualizer initialization with custom config."""
        config = DAGVisualizationConfig(version="Test Version", show_state_flow=False)
        visualizer = DAGVisualizer(config)

        assert visualizer.config.version == "Test Version"
        assert visualizer.config.show_state_flow is False

    def test_get_node_label_known_agents(self):
        """Test node label generation for known agents."""
        visualizer = DAGVisualizer()

        assert (
            visualizer._get_node_label("refiner") == "🔍 Refiner<br/>Query Refinement"
        )
        assert visualizer._get_node_label("critic") == "⚖️ Critic<br/>Critical Analysis"
        assert (
            visualizer._get_node_label("synthesis")
            == "🔗 Synthesis<br/>Final Integration"
        )
        assert (
            visualizer._get_node_label("historian")
            == "📚 Historian<br/>Context Retrieval"
        )

    def test_get_node_label_case_insensitive(self):
        """Test node label generation is case insensitive."""
        visualizer = DAGVisualizer()

        assert (
            visualizer._get_node_label("REFINER") == "🔍 Refiner<br/>Query Refinement"
        )
        assert visualizer._get_node_label("Critic") == "⚖️ Critic<br/>Critical Analysis"
        assert (
            visualizer._get_node_label("SYNTHESIS")
            == "🔗 Synthesis<br/>Final Integration"
        )

    def test_get_node_label_unknown_agent(self):
        """Test node label generation for unknown agents."""
        visualizer = DAGVisualizer()

        assert visualizer._get_node_label("unknown") == "🤖 Unknown"
        assert visualizer._get_node_label("custom_agent") == "🤖 Custom_Agent"

    def test_get_node_style_known_agents(self):
        """Test node style generation for known agents."""
        visualizer = DAGVisualizer()

        assert visualizer._get_node_style("refiner") == "refiner-node"
        assert visualizer._get_node_style("critic") == "critic-node"
        assert visualizer._get_node_style("synthesis") == "synthesis-node"
        assert visualizer._get_node_style("historian") == "historian-node"

    def test_get_node_style_case_insensitive(self):
        """Test node style generation is case insensitive."""
        visualizer = DAGVisualizer()

        assert visualizer._get_node_style("REFINER") == "refiner-node"
        assert visualizer._get_node_style("Critic") == "critic-node"

    def test_get_node_style_unknown_agent(self):
        """Test node style generation for unknown agents."""
        visualizer = DAGVisualizer()

        assert visualizer._get_node_style("unknown") == "default-node"

    def test_generate_edges_simple_chain(self):
        """Test edge generation for simple chain dependency."""
        visualizer = DAGVisualizer()
        agents = ["refiner", "critic", "synthesis"]
        dependencies = {"critic": ["refiner"], "synthesis": ["refiner", "critic"]}

        edges = visualizer._generate_edges(agents, dependencies)

        # Should have START -> refiner (entry point)
        assert "START --> REFINER" in edges
        # Should have dependency edges
        assert "REFINER --> CRITIC" in edges
        assert "REFINER --> SYNTHESIS" in edges
        assert "CRITIC --> SYNTHESIS" in edges
        # Should have synthesis -> END (terminal)
        assert "SYNTHESIS --> END" in edges

    def test_generate_edges_no_dependencies(self):
        """Test edge generation when no dependencies exist."""
        visualizer = DAGVisualizer()
        agents = ["refiner", "critic"]
        dependencies = {}

        edges = visualizer._generate_edges(agents, dependencies)

        # All should be entry points
        assert "START --> REFINER" in edges
        assert "START --> CRITIC" in edges
        # All should be terminal
        assert "REFINER --> END" in edges
        assert "CRITIC --> END" in edges

    def test_generate_edges_complex_dag(self):
        """Test edge generation for complex DAG structure."""
        visualizer = DAGVisualizer()
        agents = ["refiner", "historian", "critic", "synthesis"]
        dependencies = {"critic": ["refiner", "historian"], "synthesis": ["critic"]}

        edges = visualizer._generate_edges(agents, dependencies)

        # Entry points (no dependencies)
        assert "START --> REFINER" in edges
        assert "START --> HISTORIAN" in edges
        # Dependencies
        assert "REFINER --> CRITIC" in edges
        assert "HISTORIAN --> CRITIC" in edges
        assert "CRITIC --> SYNTHESIS" in edges
        # Terminal node
        assert "SYNTHESIS --> END" in edges

    def test_generate_state_flow_annotations(self):
        """Test state flow annotation generation."""
        visualizer = DAGVisualizer()
        agents = ["refiner", "critic"]

        annotations = visualizer._generate_state_flow_annotations(agents)

        # Should contain header comments
        assert "%% State Flow Information:" in annotations
        assert "%% - Initial state contains query and metadata" in annotations
        # Should contain agent-specific annotations
        assert '%% - Refiner adds RefinerOutput to state["refiner"]' in annotations
        assert '%% - Critic adds CriticOutput to state["critic"]' in annotations

    def test_generate_node_styling(self):
        """Test node styling generation."""
        visualizer = DAGVisualizer()

        styling = visualizer._generate_node_styling()

        # Should contain all known node styles
        assert (
            "classDef refiner-node fill:#e1f5fe,stroke:#0277bd,stroke-width:2px"
            in styling
        )
        assert (
            "classDef critic-node fill:#fff3e0,stroke:#f57c00,stroke-width:2px"
            in styling
        )
        assert (
            "classDef synthesis-node fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px"
            in styling
        )
        assert (
            "classDef historian-node fill:#e8f5e8,stroke:#388e3c,stroke-width:2px"
            in styling
        )
        assert (
            "classDef default-node fill:#f5f5f5,stroke:#616161,stroke-width:2px"
            in styling
        )

    def test_generate_mermaid_diagram_basic(self):
        """Test basic mermaid diagram generation."""
        visualizer = DAGVisualizer()
        agents = ["refiner", "critic"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        # Should contain mermaid header
        assert "graph TD" in diagram
        # Should contain nodes (enhanced with error policy indicators)
        assert "REFINER[🔍 Refiner<br/>Query Refinement<br/>🔄]" in diagram
        assert "CRITIC[⚖️ Critic<br/>Critical Analysis<br/>🛡️]" in diagram
        # Should contain START and END
        assert "START([🚀 START])" in diagram
        assert "END([🏁 END])" in diagram

    def test_generate_mermaid_diagram_with_metadata(self):
        """Test mermaid diagram generation with metadata."""
        config = DAGVisualizationConfig(version="Test Version", include_metadata=True)
        visualizer = DAGVisualizer(config)
        agents = ["refiner"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        assert "%% DAG Version: Test Version" in diagram
        assert "%% Agents: refiner" in diagram
        assert "%% Generated:" in diagram

    def test_generate_mermaid_diagram_without_metadata(self):
        """Test mermaid diagram generation without metadata."""
        config = DAGVisualizationConfig(include_metadata=False)
        visualizer = DAGVisualizer(config)
        agents = ["refiner"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        assert "%% DAG Version:" not in diagram
        assert "%% Generated:" not in diagram

    def test_generate_mermaid_diagram_with_state_flow(self):
        """Test mermaid diagram generation with state flow."""
        config = DAGVisualizationConfig(show_state_flow=True)
        visualizer = DAGVisualizer(config)
        agents = ["refiner"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        assert "%% Enhanced State Flow Information (Phase 2.2):" in diagram
        assert '%% - Refiner adds RefinerOutput to state["refiner"]' in diagram

    def test_generate_mermaid_diagram_without_state_flow(self):
        """Test mermaid diagram generation without state flow."""
        config = DAGVisualizationConfig(show_state_flow=False)
        visualizer = DAGVisualizer(config)
        agents = ["refiner"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        assert "%% State Flow Information:" not in diagram

    def test_generate_mermaid_diagram_with_node_details(self):
        """Test mermaid diagram generation with node details."""
        config = DAGVisualizationConfig(show_node_details=True)
        visualizer = DAGVisualizer(config)
        agents = ["refiner"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        assert "classDef refiner-node" in diagram
        assert "class REFINER refiner-node" in diagram

    def test_generate_mermaid_diagram_without_node_details(self):
        """Test mermaid diagram generation without node details."""
        config = DAGVisualizationConfig(show_node_details=False)
        visualizer = DAGVisualizer(config)
        agents = ["refiner"]

        diagram = visualizer.generate_mermaid_diagram(agents)

        assert "classDef refiner-node" not in diagram

    def test_output_to_stdout(self):
        """Test outputting diagram to stdout."""
        visualizer = DAGVisualizer()
        diagram = "test diagram content"

        with patch("builtins.print") as mock_print:
            visualizer.output_to_stdout(diagram)
            mock_print.assert_called_once_with(diagram)

    def test_output_to_file_success(self):
        """Test successful file output."""
        visualizer = DAGVisualizer()
        diagram = "test diagram content"

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test.md")

            visualizer.output_to_file(diagram, file_path)

            # Verify file was created with correct content
            assert os.path.exists(file_path)
            with open(file_path, "r") as f:
                content = f.read()
            assert content == diagram

    def test_output_to_file_creates_directory(self):
        """Test file output creates directory if it doesn't exist."""
        visualizer = DAGVisualizer()
        diagram = "test diagram content"

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "subdir", "test.md")

            visualizer.output_to_file(diagram, file_path)

            # Verify directory was created and file exists
            assert os.path.exists(file_path)
            with open(file_path, "r") as f:
                content = f.read()
            assert content == diagram

    def test_output_to_file_no_directory(self):
        """Test file output with no directory component."""
        visualizer = DAGVisualizer()
        diagram = "test diagram content"

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                file_path = "test.md"

                visualizer.output_to_file(diagram, file_path)

                # Verify file was created
                assert os.path.exists(file_path)
                with open(file_path, "r") as f:
                    content = f.read()
                assert content == diagram
            finally:
                os.chdir(original_cwd)

    def test_output_to_file_error_handling(self):
        """Test file output error handling."""
        visualizer = DAGVisualizer()
        diagram = "test diagram content"

        # Use invalid path that should cause write error
        with pytest.raises(Exception):
            visualizer.output_to_file(diagram, "/root/invalid/path/test.md")

    def test_auto_detect_output_mode_stdout(self):
        """Test auto-detection of stdout output mode."""
        visualizer = DAGVisualizer()

        assert visualizer.auto_detect_output_mode("stdout") == "stdout"
        assert visualizer.auto_detect_output_mode("STDOUT") == "stdout"

    def test_auto_detect_output_mode_file(self):
        """Test auto-detection of file output mode."""
        visualizer = DAGVisualizer()

        assert visualizer.auto_detect_output_mode("diagram.md") == "file"
        assert visualizer.auto_detect_output_mode("path/to/file.mmd") == "file"
        assert visualizer.auto_detect_output_mode("file.txt") == "file"

    def test_auto_detect_output_mode_unknown(self):
        """Test auto-detection defaults to stdout for unknown specs."""
        visualizer = DAGVisualizer()

        assert visualizer.auto_detect_output_mode("unknown") == "stdout"
        assert visualizer.auto_detect_output_mode("") == "stdout"

    def test_visualize_dag_stdout(self):
        """Test complete DAG visualization to stdout."""
        visualizer = DAGVisualizer()
        agents = ["refiner"]

        with patch.object(visualizer, "output_to_stdout") as mock_stdout:
            visualizer.visualize_dag(agents, "stdout")
            mock_stdout.assert_called_once()

    def test_visualize_dag_file(self):
        """Test complete DAG visualization to file."""
        visualizer = DAGVisualizer()
        agents = ["refiner"]

        with patch.object(visualizer, "output_to_file") as mock_file:
            visualizer.visualize_dag(agents, "diagram.md")
            mock_file.assert_called_once_with(mock_file.call_args[0][0], "diagram.md")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_default_agents(self):
        """Test getting default agent list."""
        agents = get_default_agents()

        assert agents == ["refiner", "critic", "historian", "synthesis"]

    def test_validate_agents_all_supported(self):
        """Test agent validation with all supported agents."""
        assert validate_agents(["refiner", "critic", "synthesis"]) is True
        assert validate_agents(["historian", "refiner"]) is True
        assert validate_agents(["synthesis"]) is True

    def test_validate_agents_case_insensitive(self):
        """Test agent validation is case insensitive."""
        assert validate_agents(["REFINER", "Critic", "SYNTHESIS"]) is True

    def test_validate_agents_unsupported(self):
        """Test agent validation with unsupported agents."""
        assert validate_agents(["refiner", "unknown"]) is False
        assert validate_agents(["invalid_agent"]) is False

    def test_validate_agents_empty_list(self):
        """Test agent validation with empty list."""
        assert validate_agents([]) is True

    def test_create_dag_visualization_default(self):
        """Test create_dag_visualization with default config."""
        agents = ["refiner"]

        with patch(
            "cognivault.diagnostics.visualize_dag.DAGVisualizer"
        ) as mock_visualizer_class:
            mock_visualizer = MagicMock()
            mock_visualizer_class.return_value = mock_visualizer

            create_dag_visualization(agents)

            mock_visualizer_class.assert_called_once_with(None)
            mock_visualizer.visualize_dag.assert_called_once_with(agents, "stdout")

    def test_create_dag_visualization_custom_config(self):
        """Test create_dag_visualization with custom config."""
        agents = ["refiner"]
        config = DAGVisualizationConfig(version="Test")

        with patch(
            "cognivault.diagnostics.visualize_dag.DAGVisualizer"
        ) as mock_visualizer_class:
            mock_visualizer = MagicMock()
            mock_visualizer_class.return_value = mock_visualizer

            create_dag_visualization(agents, "file.md", config)

            mock_visualizer_class.assert_called_once_with(config)
            mock_visualizer.visualize_dag.assert_called_once_with(agents, "file.md")

    def test_cli_visualize_dag_default_agents(self):
        """Test CLI visualization with default agents."""
        with patch(
            "cognivault.diagnostics.visualize_dag.create_dag_visualization"
        ) as mock_create:
            cli_visualize_dag()

            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert args[0] == [
                "refiner",
                "critic",
                "historian",
                "synthesis",
            ]  # Default agents

    def test_cli_visualize_dag_custom_agents(self):
        """Test CLI visualization with custom agents."""
        agents = ["refiner", "historian"]

        with patch(
            "cognivault.diagnostics.visualize_dag.create_dag_visualization"
        ) as mock_create:
            cli_visualize_dag(agents=agents)

            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert args[0] == agents

    def test_cli_visualize_dag_custom_config(self):
        """Test CLI visualization with custom configuration."""
        with patch(
            "cognivault.diagnostics.visualize_dag.create_dag_visualization"
        ) as mock_create:
            cli_visualize_dag(
                output="file.md",
                version="Test Version",
                show_state_flow=False,
                show_details=False,
            )

            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert args[1] == "file.md"  # Output spec

            config = args[2]
            assert config.version == "Test Version"
            assert config.show_state_flow is False
            assert config.show_node_details is False

    def test_cli_visualize_dag_invalid_agents(self):
        """Test CLI visualization with invalid agents."""
        with pytest.raises(ValueError, match="Unsupported agents found"):
            cli_visualize_dag(agents=["invalid_agent"])


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_stdout(self):
        """Test complete workflow outputting to stdout."""
        agents = ["refiner", "critic", "synthesis"]

        with patch("builtins.print") as mock_print:
            create_dag_visualization(agents, "stdout")

            # Should have printed the diagram
            mock_print.assert_called_once()
            diagram = mock_print.call_args[0][0]

            # Verify diagram content
            assert "graph TD" in diagram
            assert "REFINER[🔍 Refiner<br/>Query Refinement<br/>🔄]" in diagram
            assert "CRITIC[⚖️ Critic<br/>Critical Analysis<br/>🛡️]" in diagram
            assert "SYNTHESIS[🔗 Synthesis<br/>Final Integration<br/>🔄]" in diagram

    def test_complete_workflow_file(self):
        """Test complete workflow outputting to file."""
        agents = ["refiner", "critic"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_diagram.md")

            create_dag_visualization(agents, file_path)

            # Verify file was created
            assert os.path.exists(file_path)

            # Verify content
            with open(file_path, "r") as f:
                content = f.read()

            assert "graph TD" in content
            assert "REFINER[🔍 Refiner<br/>Query Refinement<br/>🔄]" in content
            assert "CRITIC[⚖️ Critic<br/>Critical Analysis<br/>🛡️]" in content

    def test_complete_workflow_with_dependencies(self):
        """Test complete workflow with complex dependencies."""
        agents = ["refiner", "historian", "critic", "synthesis"]

        with patch("builtins.print") as mock_print:
            create_dag_visualization(agents, "stdout")

            diagram = mock_print.call_args[0][0]

            # Should contain all agents
            assert "REFINER[🔍 Refiner<br/>Query Refinement<br/>🔄]" in diagram
            assert "HISTORIAN[📚 Historian<br/>Context Retrieval<br/>🔌]" in diagram
            assert "CRITIC[⚖️ Critic<br/>Critical Analysis<br/>🛡️]" in diagram
            assert "SYNTHESIS[🔗 Synthesis<br/>Final Integration<br/>🔄]" in diagram

    def test_minimal_configuration(self):
        """Test with minimal configuration."""
        config = DAGVisualizationConfig(
            show_state_flow=False, show_node_details=False, include_metadata=False
        )

        with patch("builtins.print") as mock_print:
            create_dag_visualization(["refiner"], "stdout", config)

            diagram = mock_print.call_args[0][0]

            # Should contain basic structure
            assert "graph TD" in diagram
            assert "REFINER[🔍 Refiner<br/>Query Refinement<br/>🔄]" in diagram

            # Should not contain optional elements
            assert "%% State Flow Information:" not in diagram
            assert "classDef refiner-node" not in diagram
            assert "%% DAG Version:" not in diagram

    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Test with invalid file path
        with pytest.raises(Exception):
            create_dag_visualization(["refiner"], "/invalid/path/file.md")
