"""
Base Advanced Node Infrastructure for CogniVault.

This module provides the foundation for advanced node types including
DECISION, AGGREGATOR, VALIDATOR, and TERMINATOR nodes. It defines
the base abstractions and execution context for all advanced nodes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from cognivault.agents.metadata import AgentMetadata, TaskClassification


@dataclass
class NodeExecutionContext:
    """
    Execution context for advanced nodes.

    This context carries all necessary information for node execution including
    correlation tracking, workflow identification, cognitive classification,
    and resource usage metrics.
    """

    # Required fields
    correlation_id: str
    workflow_id: str
    cognitive_classification: Dict[str, str]
    task_classification: TaskClassification

    # Optional fields with defaults
    execution_path: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    # Metadata for advanced routing
    previous_nodes: List[str] = field(default_factory=list)
    available_inputs: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values and validate context."""
        if self.resource_usage is None:
            self.resource_usage = {}

        # Initialize default resource tracking
        if "start_time" not in self.resource_usage:
            self.resource_usage["start_time"] = datetime.now(timezone.utc)

        # Validate confidence score range
        if self.confidence_score is not None:
            if not 0.0 <= self.confidence_score <= 1.0:
                raise ValueError(
                    f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}"
                )

    def add_to_execution_path(self, node_name: str) -> None:
        """Add a node to the execution path."""
        self.execution_path.append(node_name)
        self.previous_nodes.append(node_name)

    def update_resource_usage(self, metrics: Dict[str, Any]) -> None:
        """Update resource usage metrics."""
        self.resource_usage.update(metrics)

    def get_execution_time_ms(self) -> Optional[float]:
        """Calculate execution time in milliseconds if start_time is available."""
        if "start_time" in self.resource_usage and "end_time" in self.resource_usage:
            start = self.resource_usage["start_time"]
            end = self.resource_usage["end_time"]
            return (end - start).total_seconds() * 1000
        return None

    def has_input_from(self, node_name: str) -> bool:
        """Check if context has input from a specific node."""
        return node_name in self.available_inputs

    def get_input_from(self, node_name: str) -> Optional[Any]:
        """Get input from a specific node if available."""
        return self.available_inputs.get(node_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "correlation_id": self.correlation_id,
            "workflow_id": self.workflow_id,
            "cognitive_classification": self.cognitive_classification,
            "task_classification": self.task_classification.to_dict(),
            "execution_path": self.execution_path,
            "confidence_score": self.confidence_score,
            "resource_usage": self.resource_usage,
            "previous_nodes": self.previous_nodes,
            "available_inputs": self.available_inputs,
            "execution_metadata": self.execution_metadata,
        }


class BaseAdvancedNode(ABC):
    """
    Base class for all advanced node types.

    This abstract class defines the interface and common functionality for
    DECISION, AGGREGATOR, VALIDATOR, and TERMINATOR nodes. Each node type
    must implement the execute and can_handle methods.
    """

    def __init__(self, metadata: AgentMetadata, node_name: str):
        """
        Initialize the advanced node.

        Parameters
        ----------
        metadata : AgentMetadata
            The agent metadata containing multi-axis classification
        node_name : str
            Unique name for this node instance
        """
        self.metadata = metadata
        self.node_name = node_name
        self.execution_pattern = metadata.execution_pattern

        # Validate execution pattern
        valid_patterns = {
            "processor",
            "decision",
            "aggregator",
            "validator",
            "terminator",
        }
        if self.execution_pattern not in valid_patterns:
            raise ValueError(
                f"Invalid execution pattern '{self.execution_pattern}'. "
                f"Must be one of: {valid_patterns}"
            )

    @abstractmethod
    async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
        """
        Execute the node logic.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context containing all necessary information

        Returns
        -------
        Dict[str, Any]
            The execution result containing output data and metadata
        """
        pass

    @abstractmethod
    def can_handle(self, context: NodeExecutionContext) -> bool:
        """
        Check if this node can handle the given context.

        This method should evaluate the context against the node's
        requirements and capabilities to determine if execution is possible.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context to evaluate

        Returns
        -------
        bool
            True if the node can handle the context, False otherwise
        """
        pass

    def get_fallback_patterns(self) -> List[str]:
        """
        Get fallback execution patterns for this node.

        Returns a list of execution patterns that can be used as fallbacks
        if this node fails or cannot handle the context.

        Returns
        -------
        List[str]
            List of fallback execution pattern names
        """
        FALLBACK_PATTERNS = {
            "decision": ["processor", "terminator"],
            "aggregator": ["processor", "validator"],
            "validator": ["processor", "terminator"],
            "processor": ["terminator"],
            "terminator": [],
        }
        return FALLBACK_PATTERNS.get(self.execution_pattern, [])

    def get_node_info(self) -> Dict[str, Any]:
        """
        Get information about this node.

        Returns
        -------
        Dict[str, Any]
            Node information including name, type, and metadata
        """
        return {
            "node_name": self.node_name,
            "execution_pattern": self.execution_pattern,
            "cognitive_speed": self.metadata.cognitive_speed,
            "cognitive_depth": self.metadata.cognitive_depth,
            "processing_pattern": self.metadata.processing_pattern,
            "pipeline_role": self.metadata.pipeline_role,
            "bounded_context": self.metadata.bounded_context,
            "capabilities": self.metadata.capabilities,
            "fallback_patterns": self.get_fallback_patterns(),
        }

    def validate_context(self, context: NodeExecutionContext) -> List[str]:
        """
        Validate the execution context for common requirements.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context to validate

        Returns
        -------
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []

        if not context.correlation_id:
            errors.append("Missing correlation_id in context")

        if not context.workflow_id:
            errors.append("Missing workflow_id in context")

        if not context.task_classification:
            errors.append("Missing task_classification in context")

        if not context.cognitive_classification:
            errors.append("Missing cognitive_classification in context")

        return errors

    async def pre_execute(self, context: NodeExecutionContext) -> None:
        """
        Hook for pre-execution setup.

        Override this method to perform any setup required before execution.
        Default implementation adds the node to the execution path.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context
        """
        context.add_to_execution_path(self.node_name)

    async def post_execute(
        self, context: NodeExecutionContext, result: Dict[str, Any]
    ) -> None:
        """
        Hook for post-execution cleanup.

        Override this method to perform any cleanup after execution.
        Default implementation updates resource usage with end time.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context
        result : Dict[str, Any]
            The execution result
        """
        context.update_resource_usage({"end_time": datetime.now(timezone.utc)})

    def __repr__(self) -> str:
        """String representation of the node."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.node_name}', "
            f"pattern='{self.execution_pattern}')"
        )
