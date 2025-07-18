"""
Declarative workflow definition schema for CogniVault DAG workflows.

This module provides the core data models for defining sophisticated DAG workflows
with advanced node types, conditional routing, and ecosystem-ready metadata.
Supports workflow versioning, attribution, and plugin architecture foundation.
"""

import json
import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Union

# from cognivault.agents.metadata import AgentMetadata  # Commented out to avoid circular imports


class NodeCategory(str, Enum):
    """Node category taxonomy for clear distinction between node types."""

    ADVANCED = "advanced"  # CogniVault-native advanced nodes (DecisionNode, etc.)
    BASE = "base"  # Standard agent execution (existing pipeline agents)


class AdvancedNodeType(str, Enum):
    """Advanced node types for sophisticated workflow orchestration."""

    DECISION = "decision"  # Conditional routing decision points
    AGGREGATOR = "aggregator"  # Parallel output combination
    VALIDATOR = "validator"  # Quality assurance checkpoints
    TERMINATOR = "terminator"  # Early termination logic


class BaseNodeType(str, Enum):
    """Base node types for standard agent execution."""

    PROCESSOR = "processor"  # Refiner, Historian, Critic, Synthesis agents


@dataclass
class EdgeDefinition:
    """
    Definition of an edge between workflow nodes with sophisticated conditional routing.

    Supports metadata-aware routing, success/failure paths, and classification-based
    conditional logic for Airflow/Prefect-level expressiveness.
    """

    from_node: str
    to_node: str
    edge_type: str = "sequential"  # Edge type (sequential, conditional, parallel)
    condition: Optional[str] = None  # Conditional routing logic expression
    next_node_if: Optional[str] = None  # Success path routing
    failover_node: Optional[str] = None  # Failure path routing
    metadata: Dict[str, Any] = field(default_factory=dict)  # Edge metadata
    metadata_filters: Optional[Dict[str, Any]] = None  # Classification-based routing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_type": self.edge_type,
            "condition": self.condition,
            "next_node_if": self.next_node_if,
            "failover_node": self.failover_node,
            "metadata": self.metadata,
            "metadata_filters": self.metadata_filters,
        }


@dataclass
class FlowDefinition:
    """
    Sophisticated flow definition with conditional routing and metadata awareness.

    Supports advanced routing logic, success/failure paths, and metadata-driven
    conditional execution for production-grade workflow orchestration.
    """

    entry_point: str
    edges: List[EdgeDefinition]
    terminal_nodes: List[str] = field(default_factory=list)
    conditional_routing: Optional[Dict[str, Any]] = None  # Advanced routing logic

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "entry_point": self.entry_point,
            "edges": [edge.to_dict() for edge in self.edges],
            "terminal_nodes": self.terminal_nodes,
            "conditional_routing": self.conditional_routing,
        }


@dataclass
class NodeConfiguration:
    """
    Node configuration with clear taxonomy and classification filters.

    Provides explicit BASE vs ADVANCED categorization, type-safe node types,
    and metadata-aware routing support for the plugin architecture foundation.
    """

    node_id: str
    node_type: str  # Node type as string for flexibility
    category: str  # Category as string ("BASE" or "ADVANCED")
    execution_pattern: str = "processor"  # Execution pattern
    config: Dict[str, Any] = field(default_factory=dict)  # Type-specific configuration
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Multi-axis classification metadata
    classification_filters: Optional[Dict[str, Any]] = None  # Metadata-aware routing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "category": self.category,
            "execution_pattern": self.execution_pattern,
            "config": self.config,
            "metadata": self.metadata,
            "classification_filters": self.classification_filters,
        }


@dataclass
class WorkflowDefinition:
    """
    Ecosystem-ready workflow definition with versioning, attribution, and sharing contracts.

    This is the core schema for declarative DAG workflows, designed for the
    "Kubernetes of intelligent DAG workflows" ecosystem with plugin architecture
    foundation and reproducible workflow sharing capabilities.
    """

    name: str
    version: str
    workflow_id: str  # Unique identifier for ecosystem
    created_by: str  # Creator attribution
    created_at: datetime  # Creation timestamp
    nodes: List[NodeConfiguration]
    flow: FlowDefinition
    description: Optional[str] = None  # Human-readable description
    tags: List[str] = field(default_factory=list)  # Categorization
    workflow_schema_version: str = "1.0.0"  # Forward compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    @classmethod
    def create(
        cls,
        name: str,
        version: str,
        created_by: str,
        nodes: List[NodeConfiguration],
        flow: FlowDefinition,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "WorkflowDefinition":
        """
        Create a new workflow definition with auto-generated ID and timestamp.

        Parameters
        ----------
        name : str
            Human-readable workflow name
        version : str
            Workflow version (semantic versioning recommended)
        created_by : str
            Creator attribution
        nodes : List[NodeConfiguration]
            Workflow node configurations
        flow : FlowDefinition
            Workflow flow definition
        description : Optional[str]
            Human-readable description
        tags : Optional[List[str]]
            Categorization tags
        metadata : Optional[Dict[str, Any]]
            Additional metadata

        Returns
        -------
        WorkflowDefinition
            New workflow definition instance
        """
        return cls(
            name=name,
            version=version,
            workflow_id=str(uuid.uuid4()),
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
            nodes=nodes,
            flow=flow,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )

    def to_json_snapshot(self) -> Dict[str, Any]:
        """
        Serialization contract for ecosystem sharing and reproducibility.

        Returns
        -------
        Dict[str, Any]
            Complete serialized workflow definition
        """
        return {
            "name": self.name,
            "version": self.version,
            "workflow_id": self.workflow_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "tags": self.tags,
            "workflow_schema_version": self.workflow_schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "flow": self.flow.to_dict(),
            "metadata": self.metadata,
        }

    def export(self, format: str = "json") -> str:
        """
        Multi-format export for workflow sharing and storage.

        Parameters
        ----------
        format : str
            Export format ("json", "yaml", future: "mermaid")

        Returns
        -------
        str
            Exported workflow definition

        Raises
        ------
        ValueError
            If export format is not supported
        """
        snapshot = self.to_json_snapshot()

        if format.lower() == "json":
            return json.dumps(snapshot, indent=2, sort_keys=True)
        elif format.lower() == "yaml":
            return yaml.safe_dump(snapshot, default_flow_style=False, sort_keys=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def validated_by(self) -> str:
        """
        Validation attribution for ecosystem trust.

        Returns
        -------
        str
            Validation attribution string
        """
        return f"cognivault-v{self.workflow_schema_version}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        """Create workflow definition from dictionary data."""
        return cls.from_json_snapshot(data)

    @classmethod
    def from_json_snapshot(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        """
        Deserialize workflow definition from JSON snapshot.

        Parameters
        ----------
        data : Dict[str, Any]
            JSON snapshot data

        Returns
        -------
        WorkflowDefinition
            Deserialized workflow definition
        """
        # Parse nodes
        nodes = []
        for node_data in data["nodes"]:
            node = NodeConfiguration(
                node_id=node_data["node_id"],
                node_type=node_data["node_type"],
                category=node_data["category"],
                execution_pattern=node_data.get("execution_pattern", "processor"),
                config=node_data.get("config", {}),
                metadata=node_data.get("metadata", {}),
                classification_filters=node_data.get("classification_filters"),
            )
            nodes.append(node)

        # Parse flow
        edges = []
        for edge_data in data["flow"]["edges"]:
            edge = EdgeDefinition(
                from_node=edge_data["from_node"],
                to_node=edge_data["to_node"],
                edge_type=edge_data.get("edge_type", "sequential"),
                condition=edge_data.get("condition"),
                next_node_if=edge_data.get("next_node_if"),
                failover_node=edge_data.get("failover_node"),
                metadata=edge_data.get("metadata", {}),
                metadata_filters=edge_data.get("metadata_filters"),
            )
            edges.append(edge)

        flow = FlowDefinition(
            entry_point=data["flow"]["entry_point"],
            edges=edges,
            terminal_nodes=data["flow"].get("terminal_nodes", []),
            conditional_routing=data["flow"].get("conditional_routing"),
        )

        return cls(
            name=data["name"],
            version=data["version"],
            workflow_id=data["workflow_id"],
            created_by=data["created_by"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description"),
            tags=data.get("tags", []),
            workflow_schema_version=data.get("workflow_schema_version", "1.0.0"),
            nodes=nodes,
            flow=flow,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_yaml_file(cls, file_path: str) -> "WorkflowDefinition":
        """
        Load workflow definition from YAML file.

        Parameters
        ----------
        file_path : str
            Path to YAML workflow definition file

        Returns
        -------
        WorkflowDefinition
            Loaded workflow definition
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_json_snapshot(data)

    @classmethod
    def from_json_file(cls, file_path: str) -> "WorkflowDefinition":
        """
        Load workflow definition from JSON file.

        Parameters
        ----------
        file_path : str
            Path to JSON workflow definition file

        Returns
        -------
        WorkflowDefinition
            Loaded workflow definition
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_json_snapshot(data)

    def save_to_file(self, file_path: str, format: Optional[str] = None) -> None:
        """
        Save workflow definition to file.

        Parameters
        ----------
        file_path : str
            Output file path
        format : Optional[str]
            Export format (auto-detected from file extension if None)
        """
        if format is None:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                format = "yaml"
            elif file_path.endswith(".json"):
                format = "json"
            else:
                raise ValueError("Cannot auto-detect format from file extension")

        content = self.export(format)
        with open(file_path, "w") as f:
            f.write(content)


# Type aliases for convenience
WorkflowConfig = WorkflowDefinition
NodeConfig = NodeConfiguration
FlowConfig = FlowDefinition
EdgeConfig = EdgeDefinition
