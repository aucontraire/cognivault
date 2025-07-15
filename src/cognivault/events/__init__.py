"""
Enhanced Event System for CogniVault.

This module provides an enhanced event-driven architecture with multi-axis
agent classification, correlation context propagation, and production-ready
event sinks for observability and future service extraction.
"""

from .types import (
    WorkflowEvent,
    EventType,
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
    # Event sinks
    "EventSink",
    "FileEventSink",
    "ConsoleEventSink",
    "InMemoryEventSink",
]
