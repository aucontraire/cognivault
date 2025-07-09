"""
DAG visualization utility for CogniVault LangGraph orchestration.

This module provides tools to visualize the LangGraph StateGraph DAG structure
using mermaid diagrams. It supports both console output and file generation
with automatic format detection.

Features:
- Mermaid diagram generation for DAG structure
- Version annotations for tracking DAG evolution
- State flow visualization between nodes
- Support for stdout and file output modes
- Auto-detection of output format from file extensions
- Integration with CLI for easy usage
"""

import os
import sys
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from cognivault.langraph.node_wrappers import get_node_dependencies
from cognivault.observability import get_logger

logger = get_logger(__name__)


@dataclass
class DAGVisualizationConfig:
    """Configuration for DAG visualization."""

    version: str = "Phase 2.0"
    """Version annotation for the DAG."""

    show_state_flow: bool = True
    """Whether to show state flow information."""

    show_node_details: bool = True
    """Whether to show node details in the diagram."""

    include_metadata: bool = True
    """Whether to include metadata in the diagram."""


class DAGVisualizer:
    """
    DAG visualization utility for LangGraph StateGraph structures.

    This class generates mermaid diagrams that visualize the structure
    and flow of CogniVault's LangGraph-based agent orchestration.
    """

    def __init__(self, config: Optional[DAGVisualizationConfig] = None):
        """
        Initialize the DAG visualizer.

        Parameters
        ----------
        config : DAGVisualizationConfig, optional
            Configuration for visualization options
        """
        self.config = config or DAGVisualizationConfig()
        self.logger = get_logger(f"{__name__}.DAGVisualizer")

    def generate_mermaid_diagram(self, agents: List[str]) -> str:
        """
        Generate a mermaid diagram for the given agent list.

        Parameters
        ----------
        agents : List[str]
            List of agent names to include in the diagram

        Returns
        -------
        str
            Mermaid diagram as a string
        """
        self.logger.info(f"Generating mermaid diagram for agents: {agents}")

        # Get dependency information
        dependencies = get_node_dependencies()

        # Start building the diagram
        lines = []

        # Add version annotation
        if self.config.include_metadata:
            lines.append(f"%% DAG Version: {self.config.version}")
            lines.append(f"%% Generated: {datetime.now().isoformat()}")
            lines.append(f"%% Agents: {', '.join(agents)}")
            lines.append("")

        # Start the mermaid graph
        lines.append("graph TD")
        lines.append("")

        # Add START node
        lines.append("    START([🚀 START])")
        lines.append("")

        # Add agent nodes with styling
        for agent in agents:
            node_id = agent.upper()
            node_label = self._get_node_label(agent)
            node_style = self._get_node_style(agent)
            lines.append(f"    {node_id}[{node_label}]")

            # Add styling
            if node_style:
                lines.append(f"    class {node_id} {node_style}")

        lines.append("")

        # Add END node
        lines.append("    END([🏁 END])")
        lines.append("")

        # Add edges based on dependencies
        edges = self._generate_edges(agents, dependencies)
        for edge in edges:
            lines.append(f"    {edge}")

        lines.append("")

        # Add state flow annotations if enabled
        if self.config.show_state_flow:
            lines.extend(self._generate_state_flow_annotations(agents))

        # Add styling classes
        if self.config.show_node_details:
            lines.extend(self._generate_node_styling())

        return "\n".join(lines)

    def _get_node_label(self, agent: str) -> str:
        """
        Get the display label for a node.

        Parameters
        ----------
        agent : str
            Agent name

        Returns
        -------
        str
            Display label for the node
        """
        labels = {
            "refiner": "🔍 Refiner<br/>Query Refinement",
            "critic": "⚖️ Critic<br/>Critical Analysis",
            "synthesis": "🔗 Synthesis<br/>Final Integration",
            "historian": "📚 Historian<br/>Context Retrieval",
        }

        return labels.get(agent.lower(), f"🤖 {agent.title()}")

    def _get_node_style(self, agent: str) -> str:
        """
        Get the CSS class for node styling.

        Parameters
        ----------
        agent : str
            Agent name

        Returns
        -------
        str
            CSS class name
        """
        styles = {
            "refiner": "refiner-node",
            "critic": "critic-node",
            "synthesis": "synthesis-node",
            "historian": "historian-node",
        }

        return styles.get(agent.lower(), "default-node")

    def _generate_edges(
        self, agents: List[str], dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """
        Generate edges for the DAG based on dependencies.

        Parameters
        ----------
        agents : List[str]
            List of agents
        dependencies : Dict[str, List[str]]
            Dependency mapping

        Returns
        -------
        List[str]
            List of mermaid edge definitions
        """
        edges = []

        # Find entry points (nodes with no dependencies)
        entry_points = [agent for agent in agents if not dependencies.get(agent, [])]

        # Connect START to entry points
        for entry in entry_points:
            edges.append(f"START --> {entry.upper()}")

        # Connect agents based on dependencies
        for agent in agents:
            agent_deps = dependencies.get(agent, [])
            for dep in agent_deps:
                if dep in agents:
                    edges.append(f"{dep.upper()} --> {agent.upper()}")

        # Find terminal nodes (nodes that no other node depends on)
        terminal_nodes = []
        for agent in agents:
            is_terminal = True
            for other_agent in agents:
                if agent in dependencies.get(other_agent, []):
                    is_terminal = False
                    break
            if is_terminal:
                terminal_nodes.append(agent)

        # Connect terminal nodes to END
        for terminal in terminal_nodes:
            edges.append(f"{terminal.upper()} --> END")

        return edges

    def _generate_state_flow_annotations(self, agents: List[str]) -> List[str]:
        """
        Generate state flow annotations for the diagram.

        Parameters
        ----------
        agents : List[str]
            List of agents

        Returns
        -------
        List[str]
            List of annotation lines
        """
        annotations = [
            "%% State Flow Information:",
            "%% - Initial state contains query and metadata",
            "%% - Each agent adds its typed output to the state",
            "%% - Final state contains all agent outputs",
            "",
        ]

        for agent in agents:
            output_type = f"{agent.title()}Output"
            annotations.append(
                f'%% - {agent.title()} adds {output_type} to state["{agent}"]'
            )

        annotations.append("")
        return annotations

    def _generate_node_styling(self) -> List[str]:
        """
        Generate CSS styling for nodes.

        Returns
        -------
        List[str]
            List of styling definitions
        """
        return [
            "%% Node Styling",
            "classDef refiner-node fill:#e1f5fe,stroke:#0277bd,stroke-width:2px",
            "classDef critic-node fill:#fff3e0,stroke:#f57c00,stroke-width:2px",
            "classDef synthesis-node fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px",
            "classDef historian-node fill:#e8f5e8,stroke:#388e3c,stroke-width:2px",
            "classDef default-node fill:#f5f5f5,stroke:#616161,stroke-width:2px",
            "",
        ]

    def output_to_stdout(self, diagram: str) -> None:
        """
        Output the diagram to stdout.

        Parameters
        ----------
        diagram : str
            Mermaid diagram content
        """
        self.logger.info("Outputting DAG diagram to stdout")
        print(diagram)

    def output_to_file(self, diagram: str, file_path: str) -> None:
        """
        Output the diagram to a file.

        Parameters
        ----------
        diagram : str
            Mermaid diagram content
        file_path : str
            Path to output file
        """
        self.logger.info(f"Outputting DAG diagram to file: {file_path}")

        try:
            # Ensure directory exists (but handle case where file_path has no directory)
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create directory if there is one
                os.makedirs(dir_path, exist_ok=True)

            # Write to file
            with open(file_path, "w") as f:
                f.write(diagram)

            self.logger.info(f"DAG diagram saved to: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save DAG diagram to {file_path}: {e}")
            raise

    def auto_detect_output_mode(self, output_spec: str) -> str:
        """
        Auto-detect output mode from specification.

        Parameters
        ----------
        output_spec : str
            Output specification (stdout, file.md, etc.)

        Returns
        -------
        str
            Output mode: 'stdout' or 'file'
        """
        if output_spec.lower() == "stdout":
            return "stdout"
        elif "." in output_spec:
            # Has extension, treat as file
            return "file"
        else:
            # Default to stdout for unrecognized specs
            return "stdout"

    def visualize_dag(self, agents: List[str], output_spec: str = "stdout") -> None:
        """
        Visualize the DAG with the specified output.

        Parameters
        ----------
        agents : List[str]
            List of agent names
        output_spec : str, optional
            Output specification: 'stdout' or file path
        """
        self.logger.info(f"Visualizing DAG for agents: {agents}, output: {output_spec}")

        # Generate the diagram
        diagram = self.generate_mermaid_diagram(agents)

        # Determine output mode
        output_mode = self.auto_detect_output_mode(output_spec)

        # Output the diagram
        if output_mode == "stdout":
            self.output_to_stdout(diagram)
        else:
            self.output_to_file(diagram, output_spec)


def create_dag_visualization(
    agents: List[str],
    output_spec: str = "stdout",
    config: Optional[DAGVisualizationConfig] = None,
) -> None:
    """
    Create and output a DAG visualization.

    This is the main entry point for DAG visualization functionality.

    Parameters
    ----------
    agents : List[str]
        List of agent names to visualize
    output_spec : str, optional
        Output specification: 'stdout' or file path
    config : DAGVisualizationConfig, optional
        Configuration for visualization options
    """
    visualizer = DAGVisualizer(config)
    visualizer.visualize_dag(agents, output_spec)


def get_default_agents() -> List[str]:
    """
    Get the default agent list for Phase 2.0.

    Returns
    -------
    List[str]
        Default agent list
    """
    return ["refiner", "critic", "synthesis"]


def validate_agents(agents: List[str]) -> bool:
    """
    Validate that the agent list is supported.

    Parameters
    ----------
    agents : List[str]
        List of agent names

    Returns
    -------
    bool
        True if all agents are supported
    """
    supported_agents = {"refiner", "critic", "synthesis", "historian"}
    return all(agent.lower() in supported_agents for agent in agents)


# CLI integration functions
def cli_visualize_dag(
    agents: Optional[List[str]] = None,
    output: str = "stdout",
    version: str = "Phase 2.0",
    show_state_flow: bool = True,
    show_details: bool = True,
) -> None:
    """
    CLI interface for DAG visualization.

    Parameters
    ----------
    agents : List[str], optional
        List of agents to visualize. If None, uses default.
    output : str, optional
        Output specification: 'stdout' or file path
    version : str, optional
        Version annotation for the diagram
    show_state_flow : bool, optional
        Whether to show state flow information
    show_details : bool, optional
        Whether to show detailed node information
    """
    # Use default agents if not specified
    if agents is None:
        agents = get_default_agents()

    # Validate agents
    if not validate_agents(agents):
        raise ValueError(
            f"Unsupported agents found. Supported: refiner, critic, synthesis, historian"
        )

    # Create configuration
    config = DAGVisualizationConfig(
        version=version, show_state_flow=show_state_flow, show_node_details=show_details
    )

    # Create visualization
    create_dag_visualization(agents, output, config)


# Export main functions
__all__ = [
    "DAGVisualizer",
    "DAGVisualizationConfig",
    "create_dag_visualization",
    "cli_visualize_dag",
    "get_default_agents",
    "validate_agents",
]
