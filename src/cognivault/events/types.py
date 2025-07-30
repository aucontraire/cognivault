"""
Enhanced Event Types with Multi-Axis Agent Classification.

This module defines the event types for the CogniVault event-driven architecture,
including rich metadata and multi-axis agent classification for intelligent
routing and observability.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from cognivault.agents.metadata import AgentMetadata, TaskClassification


class EventCategory(Enum):
    """
    Event category taxonomy for distinguishing event sources.

    This enum differentiates between orchestration-level events (DAG execution)
    and execution-level events (individual agent internals).
    """

    ORCHESTRATION = "orchestration"  # From node wrappers, DAG orchestration
    EXECUTION = "execution"  # From individual agents, retry logic, performance


class EventType(Enum):
    """Comprehensive event type taxonomy for observability."""

    # Workflow lifecycle events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"

    # Agent execution events
    AGENT_EXECUTION_STARTED = "agent.execution.started"
    AGENT_EXECUTION_COMPLETED = "agent.execution.completed"
    AGENT_EXECUTION_FAILED = "agent.execution.failed"

    # Orchestration events
    ROUTING_DECISION_MADE = "routing.decision.made"
    PATTERN_SELECTED = "pattern.selected"
    GRAPH_COMPILED = "graph.compiled"
    CHECKPOINT_CREATED = "checkpoint.created"

    # Performance and monitoring
    PERFORMANCE_METRIC_COLLECTED = "performance.metric.collected"
    HEALTH_CHECK_PERFORMED = "health.check.performed"

    # API and service boundary events
    API_REQUEST_RECEIVED = "api.request.received"
    API_RESPONSE_SENT = "api.response.sent"
    SERVICE_BOUNDARY_CROSSED = "service.boundary.crossed"

    # Advanced node type events
    DECISION_MADE = "node.decision.made"
    AGGREGATION_COMPLETED = "node.aggregation.completed"
    VALIDATION_COMPLETED = "node.validation.completed"
    TERMINATION_TRIGGERED = "node.termination.triggered"
    NODE_EXECUTION_STARTED = "node.execution.started"
    NODE_EXECUTION_COMPLETED = "node.execution.completed"


class WorkflowEvent(BaseModel):
    """
    Enhanced event model with multi-axis agent classification.

    Provides comprehensive event tracking with correlation context,
    agent metadata, and task classification for intelligent routing
    and service extraction preparation.
    """

    # Core event identification - required fields first
    event_type: EventType = Field(
        ...,
        description="Type of event being recorded",
        json_schema_extra={"example": "workflow.started"},
    )
    event_category: EventCategory = Field(
        ...,
        description="Category of event source (orchestration vs execution)",
        json_schema_extra={"example": "orchestration"},
    )
    workflow_id: str = Field(
        ...,
        description="Unique identifier for the workflow execution",
        min_length=1,
        max_length=200,
        json_schema_extra={"example": "workflow-12345-abcdef"},
    )

    # Optional fields with defaults
    event_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this specific event",
        json_schema_extra={"example": "a1b2c3d4e5f6789012345678901234ab"},
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this event occurred (UTC)",
        json_schema_extra={"example": "2024-01-01T12:00:00Z"},
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Correlation ID for distributed tracing",
        max_length=200,
        json_schema_extra={"example": "trace-abc123"},
    )
    parent_span_id: Optional[str] = Field(
        None,
        description="Parent span ID for nested event tracking",
        max_length=200,
        json_schema_extra={"example": "span-def456"},
    )

    # Multi-axis agent classification (architectural breakthrough)
    agent_metadata: Optional[AgentMetadata] = Field(
        None, description="Rich agent metadata with multi-axis classification"
    )
    task_classification: Optional[TaskClassification] = Field(
        None, description="Task classification for semantic routing"
    )
    capabilities_used: List[str] = Field(
        default_factory=list,
        description="List of capabilities utilized during execution",
        json_schema_extra={"example": ["critical_analysis", "context_retrieval"]},
    )

    # Event data and context
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data payload",
        json_schema_extra={"example": {"query": "test query", "status": "completed"}},
    )
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "schema_version": "2.0.0",
            "agent_taxonomy": "multi_axis",  # Evolved from "cognitive_only"
            "classification_model": "capability_based",
        },
        description="Event system metadata and versioning",
    )

    # Performance tracking
    execution_time_ms: Optional[float] = Field(
        None,
        description="Execution time in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 1250.5},
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Memory usage in megabytes",
        ge=0.0,
        json_schema_extra={"example": 128.5},
    )

    # Error information
    error_message: Optional[str] = Field(
        None,
        description="Error message if event represents a failure",
        max_length=5000,
        json_schema_extra={"example": "Agent 'critic' failed: timeout exceeded"},
    )
    error_type: Optional[str] = Field(
        None,
        description="Type of error that occurred",
        max_length=200,
        json_schema_extra={"example": "TimeoutError"},
    )

    # Service context (for future service extraction)
    service_name: str = Field(
        "cognivault-core",
        description="Name of the service that generated this event",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "cognivault-core"},
    )
    service_version: str = Field(
        "1.0.0",
        description="Version of the service that generated this event",
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$",
        json_schema_extra={"example": "1.0.0"},
    )

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v):
        """Validate event ID format."""
        if not isinstance(v, str) or len(v) != 32:
            raise ValueError("event_id must be a 32-character hex string")
        try:
            int(v, 16)  # Verify it's valid hex
        except ValueError:
            raise ValueError("event_id must contain only hexadecimal characters")
        return v

    @field_validator("capabilities_used")
    @classmethod
    def validate_capabilities(cls, v):
        """Validate capabilities list."""
        if not isinstance(v, list):
            raise ValueError("capabilities_used must be a list")
        for cap in v:
            if not isinstance(cap, str) or len(cap.strip()) == 0:
                raise ValueError("All capabilities must be non-empty strings")
        return v

    @model_validator(mode="after")
    def validate_error_consistency(self):
        """Validate error field consistency."""
        has_error_message = bool(self.error_message)
        has_error_type = bool(self.error_type)

        # If one error field is set, both should be set
        if has_error_message and not has_error_type:
            raise ValueError("error_type is required when error_message is provided")
        if has_error_type and not has_error_message:
            raise ValueError("error_message is required when error_type is provided")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage and transmission - backward compatibility."""
        return {
            "event_id": self.event_id,
            "event_type": (
                self.event_type.value
                if hasattr(self.event_type, "value")
                else self.event_type
            ),
            "event_category": (
                self.event_category.value
                if hasattr(self.event_category, "value")
                else self.event_category
            ),
            "timestamp": self.timestamp.isoformat(),
            "workflow_id": self.workflow_id,
            "correlation_id": self.correlation_id,
            "parent_span_id": self.parent_span_id,
            "agent_metadata": (
                self.agent_metadata.to_dict() if self.agent_metadata else None
            ),
            "task_classification": (
                self.task_classification.to_dict() if self.task_classification else None
            ),
            "capabilities_used": self.capabilities_used,
            "data": self.data,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "service_name": self.service_name,
            "service_version": self.service_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowEvent":
        """Deserialize event from storage - backward compatibility."""
        # Handle agent metadata reconstruction
        agent_metadata = None
        if data.get("agent_metadata"):
            agent_metadata = AgentMetadata.from_dict(data["agent_metadata"])

        # Handle task classification reconstruction
        task_classification = None
        if data.get("task_classification"):
            task_classification = TaskClassification.from_dict(
                data["task_classification"]
            )

        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            event_category=EventCategory(
                data.get("event_category", "orchestration")
            ),  # Default for backward compatibility
            timestamp=datetime.fromisoformat(data["timestamp"]),
            workflow_id=data["workflow_id"],
            correlation_id=data.get("correlation_id"),
            parent_span_id=data.get("parent_span_id"),
            agent_metadata=agent_metadata,
            task_classification=task_classification,
            capabilities_used=data.get("capabilities_used", []),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            execution_time_ms=data.get("execution_time_ms"),
            memory_usage_mb=data.get("memory_usage_mb"),
            error_message=data.get("error_message"),
            error_type=data.get("error_type"),
            service_name=data.get("service_name", "cognivault-core"),
            service_version=data.get("service_version", "1.0.0"),
        )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects, not string values
    )


class WorkflowStartedEvent(WorkflowEvent):
    """Workflow execution started event with enhanced metadata."""

    event_type: EventType = Field(
        default=EventType.WORKFLOW_STARTED,
        description="Type of event being recorded",
        json_schema_extra={"example": "workflow.started"},
    )
    event_category: EventCategory = Field(
        default=EventCategory.ORCHESTRATION,
        description="Category of event source (workflow events are orchestration-level)",
        json_schema_extra={"example": "orchestration"},
    )
    query: str = Field(
        "",
        description="The user query that initiated this workflow",
        max_length=50000,
        json_schema_extra={"example": "Analyze the impact of climate change"},
    )
    agents_requested: List[str] = Field(
        default_factory=list,
        description="List of agent names requested for execution",
        json_schema_extra={"example": ["refiner", "critic", "historian"]},
    )
    execution_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters for workflow execution",
        json_schema_extra={"example": {"timeout_seconds": 300, "parallel": True}},
    )
    orchestrator_type: str = Field(
        "langgraph-real",
        description="Type of orchestrator handling the workflow",
        pattern=r"^[a-zA-Z0-9._-]+$",
        json_schema_extra={"example": "langgraph-real"},
    )

    def model_post_init(self, __context) -> None:
        """Populate data after initialization."""
        self.data.update(
            {
                "query": (
                    self.query[:100] + "..." if len(self.query) > 100 else self.query
                ),
                "query_length": len(self.query),
                "agents_requested": self.agents_requested,
                "execution_config": self.execution_config,
                "orchestrator_type": self.orchestrator_type,
            }
        )


class WorkflowCompletedEvent(WorkflowEvent):
    """Workflow execution completed event with enhanced metadata."""

    event_type: EventType = Field(
        default=EventType.WORKFLOW_COMPLETED,
        description="Type of event being recorded",
        json_schema_extra={"example": "workflow.completed"},
    )
    event_category: EventCategory = Field(
        default=EventCategory.ORCHESTRATION,
        description="Category of event source (workflow events are orchestration-level)",
        json_schema_extra={"example": "orchestration"},
    )
    status: str = Field(
        "",
        description="Final status of the workflow execution",
        pattern=r"^(completed|failed|cancelled|partial_failure)$",
        json_schema_extra={"example": "completed"},
    )
    execution_time_seconds: float = Field(
        0.0,
        description="Total execution time in seconds",
        ge=0.0,
        json_schema_extra={"example": 42.5},
    )
    agent_outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Outputs from each executed agent",
        json_schema_extra={
            "example": {
                "refiner": "Refined query output",
                "critic": "Critical analysis result",
            }
        },
    )
    successful_agents: List[str] = Field(
        default_factory=list,
        description="List of agents that completed successfully",
        json_schema_extra={"example": ["refiner", "critic"]},
    )
    failed_agents: List[str] = Field(
        default_factory=list,
        description="List of agents that failed during execution",
        json_schema_extra={"example": ["historian"]},
    )

    @field_validator("agent_outputs")
    @classmethod
    def validate_agent_outputs(cls, v):
        """Validate agent outputs structure."""
        for agent_name, output in v.items():
            if not isinstance(agent_name, str) or len(agent_name.strip()) == 0:
                raise ValueError("Agent names must be non-empty strings")
            if not isinstance(output, str):
                raise ValueError(f"Output for agent '{agent_name}' must be a string")
        return v

    def model_post_init(self, __context) -> None:
        """Populate data after initialization."""
        self.execution_time_ms = self.execution_time_seconds * 1000
        self.data.update(
            {
                "status": self.status,
                "execution_time_seconds": self.execution_time_seconds,
                "agent_outputs_count": len(self.agent_outputs),
                "successful_agents": self.successful_agents,
                "failed_agents": self.failed_agents,
                "success_rate": len(self.successful_agents)
                / max(1, len(self.successful_agents) + len(self.failed_agents)),
            }
        )


class AgentExecutionStartedEvent(WorkflowEvent):
    """Agent execution started event with multi-axis classification."""

    event_type: EventType = Field(
        default=EventType.AGENT_EXECUTION_STARTED,
        description="Type of event being recorded",
        json_schema_extra={"example": "agent.execution.started"},
    )
    event_category: EventCategory = Field(
        default=EventCategory.EXECUTION,
        description="Category of event source (defaults to execution, override for orchestration)",
        json_schema_extra={"example": "execution"},
    )
    agent_name: str = Field(
        "",
        description="Name of the agent starting execution",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "critic"},
    )
    input_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input context provided to the agent",
        json_schema_extra={
            "example": {
                "query": "test query",
                "previous_outputs": {},
                "input_tokens": 150,
            }
        },
    )

    def model_post_init(self, __context) -> None:
        """Populate data after initialization."""
        self.data.update(
            {
                "agent_name": self.agent_name,
                "input_context_size": len(str(self.input_context)),
                "input_tokens": self.input_context.get("input_tokens", 0),
            }
        )


class AgentExecutionCompletedEvent(WorkflowEvent):
    """Agent execution completed event with performance metrics."""

    event_type: EventType = Field(
        default=EventType.AGENT_EXECUTION_COMPLETED,
        description="Type of event being recorded",
        json_schema_extra={"example": "agent.execution.completed"},
    )
    event_category: EventCategory = Field(
        default=EventCategory.EXECUTION,
        description="Category of event source (defaults to execution, override for orchestration)",
        json_schema_extra={"example": "execution"},
    )
    agent_name: str = Field(
        "",
        description="Name of the agent that completed execution",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "critic"},
    )
    success: bool = Field(
        True,
        description="Whether the agent execution was successful",
        json_schema_extra={"example": True},
    )
    output_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output context generated by the agent",
        json_schema_extra={
            "example": {
                "result": "Critical analysis complete",
                "confidence": 0.85,
                "output_tokens": 200,
            }
        },
    )

    def model_post_init(self, __context) -> None:
        """Populate data after initialization."""
        self.data.update(
            {
                "agent_name": self.agent_name,
                "success": self.success,
                "output_context_size": len(str(self.output_context)),
                "output_tokens": self.output_context.get("output_tokens", 0),
            }
        )


class RoutingDecisionEvent(WorkflowEvent):
    """Routing decision event for analytics and optimization."""

    event_type: EventType = Field(
        default=EventType.ROUTING_DECISION_MADE,
        description="Type of event being recorded",
        json_schema_extra={"example": "routing.decision.made"},
    )
    event_category: EventCategory = Field(
        default=EventCategory.ORCHESTRATION,
        description="Category of event source (routing decisions are orchestration-level)",
        json_schema_extra={"example": "orchestration"},
    )
    selected_agents: List[str] = Field(
        default_factory=list,
        description="List of agents selected for execution",
        json_schema_extra={"example": ["refiner", "critic", "historian"]},
    )
    routing_strategy: str = Field(
        "",
        description="Strategy used for agent selection",
        max_length=200,
        json_schema_extra={"example": "capability_based"},
    )
    confidence_score: float = Field(
        0.0,
        description="Confidence in the routing decision (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    reasoning: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reasoning behind the routing decision",
        json_schema_extra={
            "example": {
                "criteria": ["task_complexity", "domain_expertise"],
                "explanation": "Selected agents based on query analysis",
            }
        },
    )

    @field_validator("selected_agents")
    @classmethod
    def validate_selected_agents(cls, v):
        """Validate selected agents list."""
        for agent in v:
            if not isinstance(agent, str) or len(agent.strip()) == 0:
                raise ValueError("All agent names must be non-empty strings")
        return v

    def model_post_init(self, __context) -> None:
        """Populate data after initialization."""
        self.data.update(
            {
                "selected_agents": self.selected_agents,
                "routing_strategy": self.routing_strategy,
                "confidence_score": self.confidence_score,
                "reasoning": self.reasoning,
                "agent_count": len(self.selected_agents),
            }
        )


# Event filtering and statistics
class EventFilters(BaseModel):
    """Filters for querying events with comprehensive validation."""

    event_type: Optional[EventType] = Field(
        None,
        description="Filter by specific event type",
        json_schema_extra={"example": "workflow.started"},
    )
    workflow_id: Optional[str] = Field(
        None,
        description="Filter by workflow identifier",
        max_length=200,
        json_schema_extra={"example": "workflow-12345"},
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Filter by correlation identifier",
        max_length=200,
        json_schema_extra={"example": "trace-abc123"},
    )
    agent_name: Optional[str] = Field(
        None,
        description="Filter by agent name",
        max_length=100,
        json_schema_extra={"example": "critic"},
    )
    capability: Optional[str] = Field(
        None,
        description="Filter by capability used",
        max_length=100,
        json_schema_extra={"example": "critical_analysis"},
    )
    bounded_context: Optional[str] = Field(
        None,
        description="Filter by bounded context",
        pattern=r"^(reflection|transformation|retrieval)$",
        json_schema_extra={"example": "reflection"},
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Filter events after this timestamp",
        json_schema_extra={"example": "2024-01-01T00:00:00Z"},
    )
    end_time: Optional[datetime] = Field(
        None,
        description="Filter events before this timestamp",
        json_schema_extra={"example": "2024-01-01T23:59:59Z"},
    )
    has_errors: Optional[bool] = Field(
        None,
        description="Filter by presence of errors",
        json_schema_extra={"example": False},
    )

    @model_validator(mode="after")
    def validate_time_range(self):
        """Validate that start_time is before end_time."""
        if self.start_time and self.end_time and self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
        return self

    def matches(self, event: WorkflowEvent) -> bool:
        """Check if an event matches these filters."""
        if self.event_type and event.event_type != self.event_type:
            return False
        if self.workflow_id and event.workflow_id != self.workflow_id:
            return False
        if self.correlation_id and event.correlation_id != self.correlation_id:
            return False
        if self.agent_name and event.data.get("agent_name") != self.agent_name:
            return False
        if self.capability and self.capability not in event.capabilities_used:
            return False
        if (
            self.bounded_context
            and event.agent_metadata
            and event.agent_metadata.bounded_context != self.bounded_context
        ):
            return False
        if self.start_time and event.timestamp < self.start_time:
            return False
        if self.end_time and event.timestamp > self.end_time:
            return False
        if self.has_errors is not None:
            has_error = bool(event.error_message)
            if has_error != self.has_errors:
                return False
        return True

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EventStatistics(BaseModel):
    """Statistics about processed events with validation."""

    total_events: int = Field(
        0,
        description="Total number of events processed",
        ge=0,
        json_schema_extra={"example": 1250},
    )
    events_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of events by event type",
        json_schema_extra={
            "example": {
                "workflow.started": 100,
                "workflow.completed": 95,
                "agent.execution.completed": 380,
            }
        },
    )
    events_by_agent: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of events by agent name",
        json_schema_extra={
            "example": {"refiner": 95, "critic": 95, "historian": 90, "synthesis": 95}
        },
    )
    events_by_capability: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of events by capability used",
        json_schema_extra={
            "example": {
                "critical_analysis": 95,
                "context_retrieval": 90,
                "multi_perspective_synthesis": 95,
            }
        },
    )
    average_execution_time_ms: float = Field(
        0.0,
        description="Average execution time in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 1250.5},
    )
    error_rate: float = Field(
        0.0,
        description="Error rate as a percentage (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.05},
    )

    @field_validator("events_by_type", "events_by_agent", "events_by_capability")
    @classmethod
    def validate_event_counts(cls, v):
        """Validate that all counts are non-negative."""
        for key, count in v.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Count for '{key}' must be a non-negative integer")
        return v

    def update_with_event(self, event: WorkflowEvent):
        """Update statistics with a new event."""
        self.total_events += 1

        # Update by type
        event_type = event.event_type.value
        self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1

        # Update by agent
        agent_name = event.data.get("agent_name")
        if agent_name:
            self.events_by_agent[agent_name] = (
                self.events_by_agent.get(agent_name, 0) + 1
            )

        # Update by capability
        for capability in event.capabilities_used:
            self.events_by_capability[capability] = (
                self.events_by_capability.get(capability, 0) + 1
            )

        # Update execution time (simple average for now)
        if event.execution_time_ms:
            old_avg = self.average_execution_time_ms
            self.average_execution_time_ms = (
                old_avg * (self.total_events - 1) + event.execution_time_ms
            ) / self.total_events

        # Update error rate
        if event.error_message:
            error_count = sum(
                1 for events in self.events_by_type.values() if events > 0
            )
            self.error_rate = error_count / self.total_events

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
