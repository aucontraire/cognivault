"""
Enhanced Event Types with Multi-Axis Agent Classification.

This module defines the event types for the CogniVault event-driven architecture,
including rich metadata and multi-axis agent classification for intelligent
routing and observability.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List

from cognivault.agents.metadata import AgentMetadata, TaskClassification


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


@dataclass
class WorkflowEvent:
    """
    Enhanced event model with multi-axis agent classification.

    Provides comprehensive event tracking with correlation context,
    agent metadata, and task classification for intelligent routing
    and service extraction preparation.
    """

    # Core event identification - required fields first
    event_type: EventType
    workflow_id: str

    # Optional fields with defaults
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Multi-axis agent classification (architectural breakthrough)
    agent_metadata: Optional[AgentMetadata] = None
    task_classification: Optional[TaskClassification] = None
    capabilities_used: List[str] = field(default_factory=list)

    # Event data and context
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "schema_version": "2.0.0",
            "agent_taxonomy": "multi_axis",  # Evolved from "cognitive_only"
            "classification_model": "capability_based",
        }
    )

    # Performance tracking
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Service context (for future service extraction)
    service_name: str = "cognivault-core"
    service_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage and transmission."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
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
        """Deserialize event from storage."""
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


@dataclass
class WorkflowStartedEvent(WorkflowEvent):
    """Workflow execution started event with enhanced metadata."""

    query: str = ""
    agents_requested: List[str] = field(default_factory=list)
    execution_config: Dict[str, Any] = field(default_factory=dict)
    orchestrator_type: str = "langgraph-real"

    def __post_init__(self):
        self.event_type = EventType.WORKFLOW_STARTED
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


@dataclass
class WorkflowCompletedEvent(WorkflowEvent):
    """Workflow execution completed event with enhanced metadata."""

    status: str = ""  # "completed", "failed", "cancelled", "partial_failure"
    execution_time_seconds: float = 0.0
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    successful_agents: List[str] = field(default_factory=list)
    failed_agents: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.event_type = EventType.WORKFLOW_COMPLETED
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


@dataclass
class AgentExecutionStartedEvent(WorkflowEvent):
    """Agent execution started event with multi-axis classification."""

    agent_name: str = ""
    input_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.AGENT_EXECUTION_STARTED
        self.data.update(
            {
                "agent_name": self.agent_name,
                "input_context_size": len(str(self.input_context)),
                "input_tokens": self.input_context.get("input_tokens", 0),
            }
        )


@dataclass
class AgentExecutionCompletedEvent(WorkflowEvent):
    """Agent execution completed event with performance metrics."""

    agent_name: str = ""
    success: bool = True
    output_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.AGENT_EXECUTION_COMPLETED
        self.data.update(
            {
                "agent_name": self.agent_name,
                "success": self.success,
                "output_context_size": len(str(self.output_context)),
                "output_tokens": self.output_context.get("output_tokens", 0),
            }
        )


@dataclass
class RoutingDecisionEvent(WorkflowEvent):
    """Routing decision event for analytics and optimization."""

    selected_agents: List[str] = field(default_factory=list)
    routing_strategy: str = ""
    confidence_score: float = 0.0
    reasoning: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.ROUTING_DECISION_MADE
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
@dataclass
class EventFilters:
    """Filters for querying events."""

    event_type: Optional[EventType] = None
    workflow_id: Optional[str] = None
    correlation_id: Optional[str] = None
    agent_name: Optional[str] = None
    capability: Optional[str] = None
    bounded_context: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    has_errors: Optional[bool] = None

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


@dataclass
class EventStatistics:
    """Statistics about processed events."""

    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_agent: Dict[str, int] = field(default_factory=dict)
    events_by_capability: Dict[str, int] = field(default_factory=dict)
    average_execution_time_ms: float = 0.0
    error_rate: float = 0.0

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
