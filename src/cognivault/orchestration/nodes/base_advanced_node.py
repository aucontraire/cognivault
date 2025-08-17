"""
Base Advanced Node Infrastructure for CogniVault.

This module provides the foundation for advanced node types including
DECISION, AGGREGATOR, VALIDATOR, and TERMINATOR nodes. It defines
the base abstractions and execution context for all advanced nodes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ConfigDict, model_validator
from cognivault.agents.metadata import AgentMetadata, TaskClassification


class NodeExecutionContext(BaseModel):
    """
    Execution context for advanced nodes.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.

    This context carries all necessary information for node execution including
    correlation tracking, workflow identification, cognitive classification,
    and resource usage metrics.
    """

    # Required fields
    correlation_id: str = Field(
        ...,
        description="Unique identifier for request correlation",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "req-123e4567-e89b-12d3-a456-426614174000"},
    )
    workflow_id: str = Field(
        ...,
        description="Unique identifier for the workflow",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "wf-123e4567-e89b-12d3-a456-426614174000"},
    )
    cognitive_classification: Dict[str, str] = Field(
        ...,
        description="Multi-axis cognitive classification metadata",
        json_schema_extra={"example": {"speed": "fast", "depth": "shallow"}},
    )
    task_classification: TaskClassification = Field(
        ...,
        description="Task classification for execution",
    )

    # Optional fields with defaults
    execution_path: List[str] = Field(
        default_factory=list,
        description="Path of nodes executed so far",
        max_length=100,
        json_schema_extra={"example": ["refiner", "critic", "historian"]},
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score for the execution context",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    resource_usage: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Resource usage metrics and tracking",
        json_schema_extra={"example": {"cpu_usage": 45.2, "memory_mb": 128.5}},
    )

    # Metadata for advanced routing
    previous_nodes: List[str] = Field(
        default_factory=list,
        description="List of previously executed nodes",
        max_length=100,
        json_schema_extra={"example": ["refiner", "critic"]},
    )
    available_inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Available inputs from previous nodes",
        json_schema_extra={"example": {"refiner_output": "refined query"}},
    )
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata",
        json_schema_extra={"example": {"priority": "high", "timeout_ms": 30000}},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For TaskClassification and datetime objects
    )

    @model_validator(mode="after")
    def initialize_defaults_and_validate(self) -> "NodeExecutionContext":
        """Initialize default values and validate context."""
        # Handle None resource_usage and initialize defaults
        if self.resource_usage is None:
            self.resource_usage = {}

        # Initialize default resource tracking (avoid recursion by using dict operations)
        # At this point, resource_usage is guaranteed to be a dict, not None
        if self.resource_usage is not None and "start_time" not in self.resource_usage:
            self.resource_usage.update({"start_time": datetime.now(timezone.utc)})

        return self

    def add_to_execution_path(self, node_name: str) -> None:
        """Add a node to the execution path."""
        self.execution_path.append(node_name)
        self.previous_nodes.append(node_name)

    def update_resource_usage(self, metrics: Dict[str, Any]) -> None:
        """Update resource usage metrics."""
        if self.resource_usage is not None:
            self.resource_usage.update(metrics)

    def get_execution_time_ms(self) -> Optional[float]:
        """Calculate execution time in milliseconds if start_time is available."""
        if (
            self.resource_usage is not None
            and "start_time" in self.resource_usage
            and "end_time" in self.resource_usage
        ):
            start = self.resource_usage["start_time"]
            end = self.resource_usage["end_time"]

            # Type safety: ensure we have datetime objects before calculating
            if isinstance(start, datetime) and isinstance(end, datetime):
                return (end - start).total_seconds() * 1000.0

        return None

    def has_input_from(self, node_name: str) -> bool:
        """Check if context has input from a specific node."""
        return node_name in self.available_inputs

    def get_input_from(self, node_name: str) -> Optional[Any]:
        """Get input from a specific node if available."""
        return self.available_inputs.get(node_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization.

        Note: This method is kept for backward compatibility.
        For new code, use model_dump() instead.
        """
        data = self.model_dump()
        # Handle special serialization for TaskClassification
        if hasattr(self.task_classification, "to_dict"):
            data["task_classification"] = self.task_classification.to_dict()
        return data


class BaseAdvancedNode(ABC):
    """
    Base class for all advanced node types.

    This abstract class defines the interface and common functionality for
    DECISION, AGGREGATOR, VALIDATOR, and TERMINATOR nodes. Each node type
    must implement the execute and can_handle methods.
    """

    def __init__(self, metadata: AgentMetadata, node_name: str) -> None:
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
