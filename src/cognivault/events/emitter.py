"""
Enhanced Event Emitter with Multi-Axis Agent Classification.

This module provides the event emission system for CogniVault's event-driven
architecture, supporting agent-level event tracking with multi-axis classification
and correlation context propagation.
"""

import os
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime, timezone

from cognivault.observability import get_logger
from cognivault.correlation import get_correlation_id
from cognivault.agents.metadata import AgentMetadata
from cognivault.routing.routing_decision import RoutingDecision

# NOTE: Removed import from test factories to fix circular import
# Production code should not depend on test factories

if TYPE_CHECKING:
    from .sinks import EventSink

from .types import (
    WorkflowEvent,
    EventType,
    EventCategory,
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    RoutingDecisionEvent,
)

logger = get_logger(__name__)


class EventEmitter:
    """Enhanced event emitter with multi-axis agent classification support."""

    def __init__(self) -> None:
        self.enabled = self._get_enabled_from_env()
        self.sinks: List["EventSink"] = []
        self.logger = get_logger(f"{__name__}.EventEmitter")

        # Initialize default sinks based on configuration
        self._initialize_default_sinks()

        if self.enabled:
            self.logger.info(f"EventEmitter initialized with {len(self.sinks)} sinks")
        else:
            self.logger.debug("EventEmitter initialized but disabled")

    def _get_enabled_from_env(self) -> bool:
        """Check if event emission is enabled via environment variable."""
        return os.getenv("COGNIVAULT_EVENTS_ENABLED", "false").lower() == "true"

    def _initialize_default_sinks(self) -> None:
        """Initialize default event sinks based on configuration."""
        if not self.enabled:
            return

        # Import sinks here to avoid circular imports
        try:
            from .sinks import ConsoleEventSink, InMemoryEventSink

            # Always add console sink when enabled
            self.add_sink(ConsoleEventSink())

            # Add in-memory sink for development/testing
            if os.getenv("COGNIVAULT_EVENTS_IN_MEMORY", "false").lower() == "true":
                max_events = int(os.getenv("COGNIVAULT_EVENTS_MAX_MEMORY", "1000"))
                self.add_sink(InMemoryEventSink(max_events=max_events))

        except ImportError as e:
            self.logger.warning(f"Could not initialize default sinks: {e}")

    def add_sink(self, sink: "EventSink") -> None:
        """Add an event sink."""
        self.sinks.append(sink)
        self.logger.debug(f"Added event sink: {sink.__class__.__name__}")

    def remove_sink(self, sink: "EventSink") -> bool:
        """Remove an event sink."""
        if sink in self.sinks:
            self.sinks.remove(sink)
            self.logger.debug(f"Removed event sink: {sink.__class__.__name__}")
            return True
        return False

    async def emit(self, event: WorkflowEvent) -> None:
        """Emit workflow event to all configured sinks."""
        if not self.enabled or not self.sinks:
            return

        self.logger.debug(
            f"Emitting event: {event.event_type.value} for workflow {event.workflow_id}"
        )

        # Emit to all sinks, but don't let one sink's failure affect others
        for sink in self.sinks:
            try:
                await sink.emit(event)
            except Exception as e:
                self.logger.error(
                    f"Error emitting to sink {sink.__class__.__name__}: {e}"
                )

    def enable(self) -> None:
        """Enable event emission."""
        was_enabled = self.enabled
        self.enabled = True
        if not was_enabled:
            self._initialize_default_sinks()
            self.logger.info("EventEmitter enabled")

    def disable(self) -> None:
        """Disable event emission."""
        if self.enabled:
            self.enabled = False
            self.logger.info("EventEmitter disabled")

    async def close(self) -> None:
        """Close all event sinks and cleanup resources."""
        self.logger.info(f"Closing EventEmitter with {len(self.sinks)} sinks")

        for sink in self.sinks:
            try:
                await sink.close()
            except Exception as e:
                self.logger.error(f"Error closing sink {sink.__class__.__name__}: {e}")

        self.sinks.clear()
        self.enabled = False


# Global event emitter instance
_global_event_emitter: Optional[EventEmitter] = None


def get_global_event_emitter() -> EventEmitter:
    """Get the global event emitter instance."""
    global _global_event_emitter
    if _global_event_emitter is None:
        _global_event_emitter = EventEmitter()
    return _global_event_emitter


def reset_global_event_emitter() -> None:
    """Reset the global event emitter instance for testing."""
    global _global_event_emitter
    if _global_event_emitter is not None:
        # Synchronously clean up sinks without async
        for sink in _global_event_emitter.sinks:
            try:
                # Try to close synchronously if sink supports it
                if hasattr(sink, "clear_events"):
                    sink.clear_events()
            except Exception:
                # Ignore cleanup errors during reset
                pass
        _global_event_emitter.sinks.clear()
        _global_event_emitter.enabled = False
    _global_event_emitter = None


# Convenience functions for event emission
async def emit_workflow_started(
    workflow_id: str,
    query: str,
    agents: Optional[List[str]] = None,
    execution_config: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit workflow started event."""
    emitter = get_global_event_emitter()

    event = WorkflowStartedEvent(
        event_type=EventType.WORKFLOW_STARTED,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        query=query,
        agents_requested=agents or [],
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        execution_config=execution_config or {},
        data={},
        metadata=metadata or {},
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
        orchestrator_type="langgraph",
    )

    await emitter.emit(event)


async def emit_workflow_completed(
    workflow_id: str,
    status: str,
    execution_time_seconds: float,
    agent_outputs: Optional[Dict[str, str]] = None,
    successful_agents: Optional[List[str]] = None,
    failed_agents: Optional[List[str]] = None,
    error_message: Optional[str] = None,
    error_type: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit workflow completed event."""
    emitter = get_global_event_emitter()

    # Auto-infer error_type if error_message is provided but error_type is not
    if error_message and not error_type:
        error_type = "WorkflowError"

    # Build event arguments, only including non-None values for optional fields
    event_kwargs: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "status": status,
        "execution_time_seconds": execution_time_seconds,
        "event_type": EventType.WORKFLOW_COMPLETED,
        "event_category": EventCategory.ORCHESTRATION,
        "timestamp": datetime.now(timezone.utc),
        "correlation_id": correlation_id or get_correlation_id(),
        "metadata": metadata or {},
    }

    # Only add optional parameters if provided (let default factories handle None values)
    if agent_outputs is not None:
        event_kwargs["agent_outputs"] = agent_outputs
    if successful_agents is not None:
        event_kwargs["successful_agents"] = successful_agents
    if failed_agents is not None:
        event_kwargs["failed_agents"] = failed_agents
    if error_message is not None:
        event_kwargs["error_message"] = error_message
    if error_type is not None:
        event_kwargs["error_type"] = error_type

    event = WorkflowCompletedEvent(**event_kwargs)

    await emitter.emit(event)


async def emit_agent_execution_started(
    workflow_id: str,
    agent_name: str,
    input_context: Dict[str, Any],
    agent_metadata: Optional[AgentMetadata] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    event_category: EventCategory = EventCategory.EXECUTION,
) -> None:
    """Emit agent execution started event."""
    emitter = get_global_event_emitter()

    event = AgentExecutionStartedEvent(
        event_type=EventType.AGENT_EXECUTION_STARTED,
        event_category=event_category,
        workflow_id=workflow_id,
        agent_name=agent_name,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=agent_metadata,
        task_classification=None,
        capabilities_used=[],
        input_context=input_context,
        data={},
        metadata=metadata or {},
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_agent_execution_completed(
    workflow_id: str,
    agent_name: str,
    success: bool,
    output_context: Dict[str, Any],
    agent_metadata: Optional[AgentMetadata] = None,
    execution_time_ms: Optional[float] = None,
    error_message: Optional[str] = None,
    error_type: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    event_category: EventCategory = EventCategory.EXECUTION,
) -> None:
    """Emit agent execution completed event."""
    emitter = get_global_event_emitter()

    # Auto-infer error_type if error_message is provided but error_type is not
    if error_message and not error_type:
        error_type = "AgentExecutionError"

    event = AgentExecutionCompletedEvent(
        event_type=EventType.AGENT_EXECUTION_COMPLETED,
        event_category=event_category,
        workflow_id=workflow_id,
        agent_name=agent_name,
        success=success,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=agent_metadata,
        task_classification=None,
        capabilities_used=[],
        output_context=output_context,
        data={},
        metadata=metadata or {},
        execution_time_ms=execution_time_ms,
        memory_usage_mb=None,
        error_message=error_message,
        error_type=error_type,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_routing_decision(
    workflow_id: str,
    selected_agents: List[str],
    routing_strategy: str,
    confidence_score: float = 0.0,
    reasoning: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit routing decision event."""
    emitter = get_global_event_emitter()

    event = RoutingDecisionEvent(
        event_type=EventType.ROUTING_DECISION_MADE,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        selected_agents=selected_agents,
        routing_strategy=routing_strategy,
        confidence_score=confidence_score,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        reasoning=reasoning or {},
        data={},
        metadata=metadata or {},
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_routing_decision_from_object(
    routing_decision: "RoutingDecision",
    workflow_id: str,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit routing decision event from RoutingDecision object.

    Parameters
    ----------
    routing_decision : RoutingDecision
        The routing decision object to emit
    workflow_id : str
        Workflow identifier
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    # Import here to avoid circular imports

    emitter = get_global_event_emitter()

    # Create enhanced event with full routing decision data
    event = RoutingDecisionEvent(
        event_type=EventType.ROUTING_DECISION_MADE,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        selected_agents=routing_decision.selected_agents,
        routing_strategy=routing_decision.routing_strategy,
        confidence_score=routing_decision.confidence_score,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        reasoning=routing_decision.reasoning.to_dict(),
        data={},
        metadata={
            **(metadata or {}),
            "decision_id": routing_decision.decision_id,
            "confidence_level": routing_decision.confidence_level.value,
            "available_agents": routing_decision.available_agents,
            "excluded_agents": routing_decision.get_excluded_agents(),
            "execution_order": routing_decision.execution_order,
            "parallel_groups": routing_decision.parallel_groups,
            "entry_point": routing_decision.entry_point,
            "exit_points": routing_decision.exit_points,
            "estimated_total_time_ms": routing_decision.estimated_total_time_ms,
            "estimated_success_probability": routing_decision.estimated_success_probability,
            "optimization_opportunities": routing_decision.optimization_opportunities,
            "is_high_confidence": routing_decision.is_high_confidence(),
            "is_risky": routing_decision.is_risky(),
            "has_fallbacks": routing_decision.has_fallbacks(),
        },
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_health_check_performed(
    component_name: str,
    status: str,
    response_time_ms: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit health check performed event.

    Parameters
    ----------
    component_name : str
        Name of the component being health checked (e.g., "api", "orchestrator", "llm_gateway")
    status : str
        Health check status ("healthy", "degraded", "unhealthy", "unknown")
    response_time_ms : Optional[float]
        Response time for the health check in milliseconds
    details : Optional[Dict[str, Any]]
        Additional health check details (e.g., resource usage, error messages)
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    # Generate a unique workflow_id for health check events
    import uuid

    workflow_id = f"health_check_{uuid.uuid4().hex[:8]}"

    event = WorkflowEvent(
        event_type=EventType.HEALTH_CHECK_PERFORMED,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "component_name": component_name,
            "status": status,
            "response_time_ms": response_time_ms,
            "details": details or {},
        },
        metadata={
            "event_category": "monitoring",
            "component_type": "system",
            **(metadata or {}),
        },
        execution_time_ms=response_time_ms,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_api_request_received(
    workflow_id: str,
    endpoint: str,
    request_size_bytes: Optional[int] = None,
    client_info: Optional[Dict[str, Any]] = None,
    request_data: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit API request received event.

    Parameters
    ----------
    workflow_id : str
        Workflow identifier for the API request
    endpoint : str
        The API endpoint that received the request (e.g., "execute_workflow", "health_check")
    request_size_bytes : Optional[int]
        Size of the request payload in bytes
    client_info : Optional[Dict[str, Any]]
        Information about the client making the request (e.g., user_agent, ip_address)
    request_data : Optional[Dict[str, Any]]
        Summary of request data (avoid including sensitive information)
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    event = WorkflowEvent(
        event_type=EventType.API_REQUEST_RECEIVED,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "endpoint": endpoint,
            "request_size_bytes": request_size_bytes,
            "client_info": client_info or {},
            "request_data": request_data or {},
        },
        metadata={
            "event_category": "api",
            "component_type": "api_gateway",
            **(metadata or {}),
        },
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_api_response_sent(
    workflow_id: str,
    status: str,
    response_size_bytes: Optional[int] = None,
    execution_time_ms: Optional[float] = None,
    response_data: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit API response sent event.

    Parameters
    ----------
    workflow_id : str
        Workflow identifier for the API response
    status : str
        Response status ("success", "error", "timeout", "partial_success")
    response_size_bytes : Optional[int]
        Size of the response payload in bytes
    execution_time_ms : Optional[float]
        Total execution time for the API request in milliseconds
    response_data : Optional[Dict[str, Any]]
        Summary of response data (avoid including sensitive information)
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    event = WorkflowEvent(
        event_type=EventType.API_RESPONSE_SENT,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "status": status,
            "response_size_bytes": response_size_bytes,
            "execution_time_ms": execution_time_ms,
            "response_data": response_data or {},
        },
        metadata={
            "event_category": "api",
            "component_type": "api_gateway",
            **(metadata or {}),
        },
        execution_time_ms=execution_time_ms,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_service_boundary_crossed(
    workflow_id: str,
    source_service: str,
    target_service: str,
    operation: str,
    boundary_type: str = "internal",
    payload_size_bytes: Optional[int] = None,
    operation_data: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit service boundary crossed event.

    Parameters
    ----------
    workflow_id : str
        Workflow identifier for the boundary crossing
    source_service : str
        Name of the service initiating the call (e.g., "orchestrator", "api_gateway")
    target_service : str
        Name of the service being called (e.g., "llm_gateway", "diagnostics", "external_api")
    operation : str
        The operation being performed (e.g., "complete", "health_check", "execute")
    boundary_type : str
        Type of boundary crossing ("internal", "external", "microservice")
    payload_size_bytes : Optional[int]
        Size of the data being transmitted across the boundary
    operation_data : Optional[Dict[str, Any]]
        Summary of operation data (avoid including sensitive information)
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    event = WorkflowEvent(
        event_type=EventType.SERVICE_BOUNDARY_CROSSED,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "source_service": source_service,
            "target_service": target_service,
            "operation": operation,
            "boundary_type": boundary_type,
            "payload_size_bytes": payload_size_bytes,
            "operation_data": operation_data or {},
        },
        metadata={
            "event_category": "service_interaction",
            "component_type": "boundary",
            **(metadata or {}),
        },
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_decision_made(
    workflow_id: str,
    decision_criteria: List[str],
    selected_path: str,
    confidence_score: float,
    alternative_paths: Optional[List[str]] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit decision made event for DECISION nodes.

    Parameters
    ----------
    workflow_id : str
        Workflow identifier for the decision
    decision_criteria : List[str]
        List of criteria used to make the decision
    selected_path : str
        The path that was selected based on the decision
    confidence_score : float
        Confidence score for the decision (0.0 to 1.0)
    alternative_paths : Optional[List[str]]
        Other paths that were considered but not selected
    reasoning : Optional[Dict[str, Any]]
        Detailed reasoning for the decision
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    event = WorkflowEvent(
        event_type=EventType.DECISION_MADE,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "decision_criteria": decision_criteria,
            "selected_path": selected_path,
            "confidence_score": confidence_score,
            "alternative_paths": alternative_paths or [],
            "reasoning": reasoning or {},
        },
        metadata={
            "event_category": "node_execution",
            "node_type": "decision",
            "execution_pattern": "decision",
            **(metadata or {}),
        },
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_aggregation_completed(
    workflow_id: str,
    aggregation_strategy: str,
    input_sources: List[str],
    output_quality_score: float,
    conflicts_resolved: int = 0,
    aggregation_time_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit aggregation completed event for AGGREGATOR nodes.

    Parameters
    ----------
    workflow_id : str
        Workflow identifier for the aggregation
    aggregation_strategy : str
        Strategy used for aggregation (e.g., "consensus", "weighted", "hierarchical")
    input_sources : List[str]
        List of input sources that were aggregated
    output_quality_score : float
        Quality score of the aggregated output (0.0 to 1.0)
    conflicts_resolved : int
        Number of conflicts resolved during aggregation
    aggregation_time_ms : Optional[float]
        Time taken for aggregation in milliseconds
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    event = WorkflowEvent(
        event_type=EventType.AGGREGATION_COMPLETED,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "aggregation_strategy": aggregation_strategy,
            "input_sources": input_sources,
            "output_quality_score": output_quality_score,
            "conflicts_resolved": conflicts_resolved,
            "aggregation_time_ms": aggregation_time_ms,
        },
        metadata={
            "event_category": "node_execution",
            "node_type": "aggregator",
            "execution_pattern": "aggregator",
            **(metadata or {}),
        },
        execution_time_ms=aggregation_time_ms,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_validation_completed(
    workflow_id: str,
    validation_result: str,  # "pass", "fail", "warning"
    quality_score: float,
    validation_criteria: List[str],
    recommendations: Optional[List[str]] = None,
    validation_time_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit validation completed event for VALIDATOR nodes.

    Parameters
    ----------
    workflow_id : str
        Workflow identifier for the validation
    validation_result : str
        Result of the validation ("pass", "fail", "warning")
    quality_score : float
        Quality score from the validation (0.0 to 1.0)
    validation_criteria : List[str]
        List of criteria that were validated
    recommendations : Optional[List[str]]
        List of recommendations based on validation results
    validation_time_ms : Optional[float]
        Time taken for validation in milliseconds
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    event = WorkflowEvent(
        event_type=EventType.VALIDATION_COMPLETED,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "validation_result": validation_result,
            "quality_score": quality_score,
            "validation_criteria": validation_criteria,
            "recommendations": recommendations or [],
            "validation_time_ms": validation_time_ms,
        },
        metadata={
            "event_category": "node_execution",
            "node_type": "validator",
            "execution_pattern": "validator",
            **(metadata or {}),
        },
        execution_time_ms=validation_time_ms,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)


async def emit_termination_triggered(
    workflow_id: str,
    termination_reason: str,
    confidence_score: Optional[float] = None,
    threshold: Optional[float] = None,
    resources_saved: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit termination triggered event for TERMINATOR nodes.

    Parameters
    ----------
    workflow_id : str
        Workflow identifier for the termination
    termination_reason : str
        Reason for early termination (e.g., "confidence_threshold_met", "quality_gate_passed", "resource_limit")
    confidence_score : Optional[float]
        Confidence score that triggered termination (0.0 to 1.0)
    threshold : Optional[float]
        Threshold value that was met/exceeded
    resources_saved : Optional[Dict[str, Any]]
        Resources saved by early termination (e.g., time_ms, compute_units, tokens)
    correlation_id : Optional[str]
        Correlation identifier
    metadata : Optional[Dict[str, Any]]
        Additional metadata
    """
    emitter = get_global_event_emitter()

    event = WorkflowEvent(
        event_type=EventType.TERMINATION_TRIGGERED,
        event_category=EventCategory.ORCHESTRATION,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        parent_span_id=None,
        agent_metadata=None,
        task_classification=None,
        capabilities_used=[],
        data={
            "termination_reason": termination_reason,
            "confidence_score": confidence_score,
            "threshold": threshold,
            "resources_saved": resources_saved or {},
        },
        metadata={
            "event_category": "node_execution",
            "node_type": "terminator",
            "execution_pattern": "terminator",
            **(metadata or {}),
        },
        execution_time_ms=None,
        memory_usage_mb=None,
        error_message=None,
        error_type=None,
        service_name="cognivault-core",
        service_version="1.0.0",
    )

    await emitter.emit(event)
