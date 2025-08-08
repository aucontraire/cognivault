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
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from cognivault.orchestration.node_wrappers import get_node_dependencies
from cognivault.orchestration.error_policies import (
    get_error_policy_manager,
    ErrorPolicyType,
)
from cognivault.observability import get_logger

logger = get_logger(__name__)


class DAGVisualizationConfig(BaseModel):
    """Configuration for DAG visualization."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    version: str = Field(
        default="Phase 2.2", description="Version annotation for the DAG"
    )
    show_state_flow: bool = Field(
        default=True, description="Whether to show state flow information"
    )
    show_node_details: bool = Field(
        default=True, description="Whether to show node details in the diagram"
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata in the diagram"
    )
    show_checkpoints: bool = Field(
        default=True, description="Whether to show checkpoint information for nodes"
    )
    show_error_policies: bool = Field(
        default=True, description="Whether to show error policy information"
    )
    show_fallback_routes: bool = Field(
        default=True, description="Whether to show fallback routes and error handling"
    )
    checkpoints_enabled: bool = Field(
        default=False,
        description="Whether checkpointing is enabled in the current session",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "version": self.version,
            "show_state_flow": self.show_state_flow,
            "show_node_details": self.show_node_details,
            "include_metadata": self.include_metadata,
            "show_checkpoints": self.show_checkpoints,
            "show_error_policies": self.show_error_policies,
            "show_fallback_routes": self.show_fallback_routes,
            "checkpoints_enabled": self.checkpoints_enabled,
        }


class DAGVisualizer:
    """
    DAG visualization utility for LangGraph StateGraph structures.

    This class generates mermaid diagrams that visualize the structure
    and flow of CogniVault's LangGraph-based agent orchestration.
    """

    def __init__(self, config: Optional[DAGVisualizationConfig] = None) -> None:
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
            lines.append(f"%% Checkpoints Enabled: {self.config.checkpoints_enabled}")
            lines.append("")

        # Start the mermaid graph
        lines.append("graph TD")
        lines.append("")

        # Add START node
        lines.append("    START([🚀 START])")
        lines.append("")

        # Add checkpoint node if enabled
        if self.config.checkpoints_enabled and self.config.show_checkpoints:
            lines.append("    INIT_CP{💾 Init<br/>Checkpoint}")
            lines.append("    class INIT_CP checkpoint-node")
            lines.append("")

        # Add agent nodes with styling
        for agent in agents:
            node_id = agent.upper()
            node_label = self._get_enhanced_node_label(agent)
            node_style = self._get_enhanced_node_style(agent)
            lines.append(f"    {node_id}[{node_label}]")

            # Add styling
            if node_style:
                lines.append(f"    class {node_id} {node_style}")

            # Add error handling nodes if enabled
            if self.config.show_fallback_routes:
                error_node_id = f"{node_id}_ERR"
                fallback_node_id = f"{node_id}_FB"
                lines.append(f"    {error_node_id}{{⚠️ {agent.title()}<br/>Error}}")
                lines.append(f"    {fallback_node_id}[🔄 {agent.title()}<br/>Fallback]")
                lines.append(f"    class {error_node_id} error-node")
                lines.append(f"    class {fallback_node_id} fallback-node")

            # Add checkpoint nodes if enabled
            if self.config.checkpoints_enabled and self.config.show_checkpoints:
                checkpoint_node_id = f"{node_id}_CP"
                lines.append(
                    f"    {checkpoint_node_id}{{💾 {agent.title()}<br/>Checkpoint}}"
                )
                lines.append(f"    class {checkpoint_node_id} checkpoint-node")

        lines.append("")

        # Add END node
        lines.append("    END([🏁 END])")
        lines.append("")

        # Add completion checkpoint if enabled
        if self.config.checkpoints_enabled and self.config.show_checkpoints:
            lines.append("    FINAL_CP{💾 Final<br/>Checkpoint}")
            lines.append("    class FINAL_CP checkpoint-node")
            lines.append("")

        # Add edges based on dependencies
        edges = self._generate_enhanced_edges(agents, dependencies)
        for edge in edges:
            lines.append(f"    {edge}")

        lines.append("")

        # Add state flow annotations if enabled
        if self.config.show_state_flow:
            lines.extend(self._generate_enhanced_state_flow_annotations(agents))

        # Add error policy annotations if enabled
        if self.config.show_error_policies:
            lines.extend(self._generate_error_policy_annotations(agents))

        # Add styling classes
        if self.config.show_node_details:
            lines.extend(self._generate_enhanced_node_styling())

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

    def _get_enhanced_node_label(self, agent: str) -> str:
        """
        Get enhanced display label for a node with checkpoint and error policy info.

        Parameters
        ----------
        agent : str
            Agent name

        Returns
        -------
        str
            Enhanced display label for the node
        """
        base_label = self._get_node_label(agent)

        if not (self.config.show_checkpoints or self.config.show_error_policies):
            return base_label

        # Get error policy information
        error_manager = get_error_policy_manager()
        policy = error_manager.get_policy(agent)

        # Add checkpoint indicator
        checkpoint_indicator = ""
        if self.config.checkpoints_enabled and self.config.show_checkpoints:
            checkpoint_indicator = "<br/>💾"

        # Add error policy indicator
        policy_indicator = ""
        if self.config.show_error_policies:
            if policy.policy_type == ErrorPolicyType.CIRCUIT_BREAKER:
                policy_indicator = "<br/>🔌"
            elif policy.policy_type == ErrorPolicyType.RETRY_WITH_BACKOFF:
                policy_indicator = "<br/>🔄"
            elif policy.policy_type == ErrorPolicyType.GRACEFUL_DEGRADATION:
                policy_indicator = "<br/>🛡️"
            elif policy.policy_type == ErrorPolicyType.FAIL_FAST:
                policy_indicator = "<br/>⚡"

        return f"{base_label}{checkpoint_indicator}{policy_indicator}"

    def _get_enhanced_node_style(self, agent: str) -> str:
        """
        Get enhanced CSS class for node styling with checkpoint and error policy info.

        Parameters
        ----------
        agent : str
            Agent name

        Returns
        -------
        str
            Enhanced CSS class name
        """
        base_style = self._get_node_style(agent)

        # Get error policy information
        error_manager = get_error_policy_manager()
        policy = error_manager.get_policy(agent)

        # Add policy-specific styling
        if policy.policy_type == ErrorPolicyType.CIRCUIT_BREAKER:
            return f"{base_style},circuit-breaker-node"
        elif policy.policy_type == ErrorPolicyType.RETRY_WITH_BACKOFF:
            return f"{base_style},retry-node"
        elif policy.policy_type == ErrorPolicyType.GRACEFUL_DEGRADATION:
            return f"{base_style},graceful-node"

        return base_style

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

    def _generate_enhanced_edges(
        self, agents: List[str], dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """
        Generate enhanced edges for the DAG with checkpoint and error handling routes.

        Parameters
        ----------
        agents : List[str]
            List of agents
        dependencies : Dict[str, List[str]]
            Dependency mapping

        Returns
        -------
        List[str]
            List of enhanced mermaid edge definitions
        """
        edges = []

        # Start with basic edges
        basic_edges = self._generate_edges(agents, dependencies)

        # Modify basic edges to include checkpoints and error handling
        for edge in basic_edges:
            if self.config.checkpoints_enabled and self.config.show_checkpoints:
                # Insert checkpoint nodes in the flow
                if "START -->" in edge:
                    # START -> INIT_CP -> agent
                    if "INIT_CP" not in edge:
                        agent = edge.split("-->")[1].strip()
                        edges.append("START --> INIT_CP")
                        edges.append(f"INIT_CP --> {agent}")
                    else:
                        edges.append(edge)
                elif "--> END" in edge:
                    # agent -> FINAL_CP -> END
                    agent = edge.split("-->")[0].strip()
                    edges.append(f"{agent} --> FINAL_CP")
                    edges.append("FINAL_CP --> END")
                elif "-->" in edge and not any(
                    special in edge for special in ["CP", "ERR", "FB"]
                ):
                    # agent1 -> agent1_CP -> agent2
                    parts = edge.split("-->")
                    if len(parts) == 2:
                        from_agent = parts[0].strip()
                        to_agent = parts[1].strip()
                        edges.append(f"{from_agent} --> {from_agent}_CP")
                        edges.append(f"{from_agent}_CP --> {to_agent}")
                    else:
                        edges.append(edge)
                else:
                    edges.append(edge)
            else:
                edges.append(edge)

        # Add error handling edges if enabled
        if self.config.show_fallback_routes:
            for agent in agents:
                agent_upper = agent.upper()
                error_node = f"{agent_upper}_ERR"
                fallback_node = f"{agent_upper}_FB"

                # Agent can fail -> Error node
                edges.append(f"{agent_upper} -.->|failure| {error_node}")

                # Error node -> Fallback
                edges.append(f"{error_node} --> {fallback_node}")

                # Fallback can continue to next agents or END
                # Find what this agent connects to normally
                for edge in basic_edges:
                    if f"{agent_upper} -->" in edge and not any(
                        special in edge for special in ["ERR", "FB", "CP"]
                    ):
                        target = edge.split("-->")[1].strip()
                        edges.append(f"{fallback_node} -.->|recovery| {target}")

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

    def _generate_enhanced_state_flow_annotations(self, agents: List[str]) -> List[str]:
        """
        Generate enhanced state flow annotations with checkpoint information.

        Parameters
        ----------
        agents : List[str]
            List of agents

        Returns
        -------
        List[str]
            List of enhanced annotation lines
        """
        annotations = [
            "%% Enhanced State Flow Information (Phase 2.2):",
            "%% - Initial state contains query and metadata",
            "%% - Each agent adds its typed output to the state",
            "%% - Final state contains all agent outputs",
        ]

        if self.config.checkpoints_enabled:
            annotations.extend(
                [
                    "%% - Checkpoints capture state at key points",
                    "%% - State can be rolled back to any checkpoint",
                    "%% - Thread ID scopes conversation persistence",
                ]
            )

        annotations.append("")

        for agent in agents:
            output_type = f"{agent.title()}Output"
            annotations.append(
                f'%% - {agent.title()} adds {output_type} to state["{agent}"]'
            )

        if self.config.checkpoints_enabled:
            annotations.append("")
            annotations.append("%% Checkpoint Flow:")
            annotations.append("%% 1. Initialization checkpoint before execution")
            annotations.append("%% 2. Agent checkpoints after successful execution")
            annotations.append("%% 3. Final checkpoint with complete state")

        annotations.append("")
        return annotations

    def _generate_error_policy_annotations(self, agents: List[str]) -> List[str]:
        """
        Generate error policy annotations for the diagram.

        Parameters
        ----------
        agents : List[str]
            List of agents

        Returns
        -------
        List[str]
            List of error policy annotation lines
        """
        annotations = [
            "%% Error Policy Information:",
            "%% Legend: 🔌=Circuit Breaker, 🔄=Retry, 🛡️=Graceful, ⚡=Fail Fast",
            "",
        ]

        error_manager = get_error_policy_manager()

        for agent in agents:
            policy = error_manager.get_policy(agent)
            policy_name = policy.policy_type.value.replace("_", " ").title()

            timeout_info = ""
            if policy.timeout_seconds:
                timeout_info = f" (timeout: {policy.timeout_seconds}s)"

            retry_info = ""
            if (
                policy.retry_config
                and policy.policy_type == ErrorPolicyType.RETRY_WITH_BACKOFF
            ):
                retry_info = f" (max attempts: {policy.retry_config.max_attempts})"

            cb_info = ""
            if (
                policy.circuit_breaker_config
                and policy.policy_type == ErrorPolicyType.CIRCUIT_BREAKER
            ):
                cb_info = f" (failure threshold: {policy.circuit_breaker_config.failure_threshold})"

            annotations.append(
                f"%% - {agent.title()}: {policy_name}{timeout_info}{retry_info}{cb_info}"
            )

        annotations.append("")

        if self.config.show_fallback_routes:
            annotations.extend(
                [
                    "%% Fallback Strategies:",
                    "%% - Error nodes catch failures",
                    "%% - Fallback nodes provide recovery paths",
                    "%% - Dotted lines show error/recovery routes",
                    "",
                ]
            )

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

    def _generate_enhanced_node_styling(self) -> List[str]:
        """
        Generate enhanced CSS styling for nodes with checkpoint and error policy styles.

        Returns
        -------
        List[str]
            List of enhanced styling definitions
        """
        styles = [
            "%% Enhanced Node Styling (Phase 2.2)",
            "classDef refiner-node fill:#e1f5fe,stroke:#0277bd,stroke-width:2px",
            "classDef critic-node fill:#fff3e0,stroke:#f57c00,stroke-width:2px",
            "classDef synthesis-node fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px",
            "classDef historian-node fill:#e8f5e8,stroke:#388e3c,stroke-width:2px",
            "classDef default-node fill:#f5f5f5,stroke:#616161,stroke-width:2px",
            "",
            "%% Checkpoint and Memory Styling",
            "classDef checkpoint-node fill:#e8eaf6,stroke:#3f51b5,stroke-width:3px,stroke-dasharray: 5 5",
            "",
            "%% Error Policy Styling",
            "classDef circuit-breaker-node stroke:#d32f2f,stroke-width:3px",
            "classDef retry-node stroke:#ff9800,stroke-width:3px",
            "classDef graceful-node stroke:#4caf50,stroke-width:3px",
            "",
            "%% Error Handling Styling",
            "classDef error-node fill:#ffebee,stroke:#f44336,stroke-width:2px",
            "classDef fallback-node fill:#fff8e1,stroke:#ffc107,stroke-width:2px",
            "",
        ]

        return styles

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
    Get the default agent list for Phase 2.2.

    Returns
    -------
    List[str]
        Default agent list including historian
    """
    return ["refiner", "critic", "historian", "synthesis"]


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
    version: str = "Phase 2.2",
    show_state_flow: bool = True,
    show_details: bool = True,
    show_checkpoints: bool = True,
    show_error_policies: bool = True,
    show_fallback_routes: bool = True,
    checkpoints_enabled: bool = False,
) -> None:
    """
    CLI interface for DAG visualization with Phase 2.2 features.

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
    show_checkpoints : bool, optional
        Whether to show checkpoint information for nodes
    show_error_policies : bool, optional
        Whether to show error policy information
    show_fallback_routes : bool, optional
        Whether to show fallback routes and error handling
    checkpoints_enabled : bool, optional
        Whether checkpointing is enabled in the current session
    """
    # Use default agents if not specified
    if agents is None:
        agents = get_default_agents()

    # Validate agents
    if not validate_agents(agents):
        raise ValueError(
            "Unsupported agents found. Supported: refiner, critic, synthesis, historian"
        )

    # Create configuration
    config = DAGVisualizationConfig(
        version=version,
        show_state_flow=show_state_flow,
        show_node_details=show_details,
        show_checkpoints=show_checkpoints,
        show_error_policies=show_error_policies,
        show_fallback_routes=show_fallback_routes,
        checkpoints_enabled=checkpoints_enabled,
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
