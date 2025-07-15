"""
Enhanced Event Emitter with Multi-Axis Agent Classification.

This module provides the event emission system for CogniVault's event-driven
architecture, supporting agent-level event tracking with multi-axis classification
and correlation context propagation.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from datetime import datetime, timezone

from cognivault.observability import get_logger
from cognivault.correlation import get_correlation_id, get_workflow_id
from cognivault.agents.metadata import AgentMetadata

if TYPE_CHECKING:
    from .sinks import EventSink

from .types import (
    WorkflowEvent,
    EventType,
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
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        query=query,
        agents_requested=agents or [],
        execution_config=execution_config or {},
        metadata=metadata or {},
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
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit workflow completed event."""
    emitter = get_global_event_emitter()

    event = WorkflowCompletedEvent(
        event_type=EventType.WORKFLOW_COMPLETED,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        status=status,
        execution_time_seconds=execution_time_seconds,
        agent_outputs=agent_outputs or {},
        successful_agents=successful_agents or [],
        failed_agents=failed_agents or [],
        error_message=error_message,
        metadata=metadata or {},
    )

    await emitter.emit(event)


async def emit_agent_execution_started(
    workflow_id: str,
    agent_name: str,
    input_context: Dict[str, Any],
    agent_metadata: Optional[AgentMetadata] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit agent execution started event."""
    emitter = get_global_event_emitter()

    event = AgentExecutionStartedEvent(
        event_type=EventType.AGENT_EXECUTION_STARTED,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        agent_metadata=agent_metadata,
        agent_name=agent_name,
        input_context=input_context,
        metadata=metadata or {},
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
) -> None:
    """Emit agent execution completed event."""
    emitter = get_global_event_emitter()

    event = AgentExecutionCompletedEvent(
        event_type=EventType.AGENT_EXECUTION_COMPLETED,
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        agent_metadata=agent_metadata,
        agent_name=agent_name,
        success=success,
        output_context=output_context,
        execution_time_ms=execution_time_ms,
        error_message=error_message,
        error_type=error_type,
        metadata=metadata or {},
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
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or get_correlation_id(),
        selected_agents=selected_agents,
        routing_strategy=routing_strategy,
        confidence_score=confidence_score,
        reasoning=reasoning or {},
        metadata=metadata or {},
    )

    await emitter.emit(event)
