"""
Enhanced Event System for CogniVault.

This module provides an enhanced event-driven architecture with multi-axis
agent classification, correlation context propagation, and production-ready
event sinks for observability and future service extraction.
"""

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
from .emitter import (
    EventEmitter,
    get_global_event_emitter,
    emit_workflow_started,
    emit_workflow_completed,
    emit_agent_execution_started,
    emit_agent_execution_completed,
    emit_routing_decision,
    emit_routing_decision_from_object,
    emit_health_check_performed,
    emit_api_request_received,
    emit_api_response_sent,
    emit_service_boundary_crossed,
    emit_decision_made,
    emit_aggregation_completed,
    emit_validation_completed,
    emit_termination_triggered,
)
from .sinks import (
    EventSink,
    FileEventSink,
    ConsoleEventSink,
    InMemoryEventSink,
)

__all__ = [
    # Core event types
    "WorkflowEvent",
    "EventType",
    "EventCategory",
    "WorkflowStartedEvent",
    "WorkflowCompletedEvent",
    "AgentExecutionStartedEvent",
    "AgentExecutionCompletedEvent",
    "RoutingDecisionEvent",
    # Event emission
    "EventEmitter",
    "get_global_event_emitter",
    "emit_workflow_started",
    "emit_workflow_completed",
    "emit_agent_execution_started",
    "emit_agent_execution_completed",
    "emit_routing_decision",
    "emit_routing_decision_from_object",
    "emit_health_check_performed",
    "emit_api_request_received",
    "emit_api_response_sent",
    "emit_service_boundary_crossed",
    "emit_decision_made",
    "emit_aggregation_completed",
    "emit_validation_completed",
    "emit_termination_triggered",
    # Event sinks
    "EventSink",
    "FileEventSink",
    "ConsoleEventSink",
    "InMemoryEventSink",
]
