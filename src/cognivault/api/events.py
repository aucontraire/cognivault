"""
Event system bridge for API layer observability.

Prepares for Phase 3B event-driven architecture.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from cognivault.observability import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowEvent:
    """Base workflow event."""

    event_type: str
    workflow_id: str
    timestamp: datetime
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class WorkflowStarted(WorkflowEvent):
    """Workflow execution started event."""

    query: str = ""
    agents: Optional[List[str]] = None
    execution_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "workflow_started"


@dataclass
class WorkflowCompleted(WorkflowEvent):
    """Workflow execution completed event."""

    status: str = ""  # "completed", "failed", "cancelled"
    execution_time_seconds: float = 0.0
    agent_outputs: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "workflow_completed"


@dataclass
class WorkflowProgress(WorkflowEvent):
    """Workflow execution progress event."""

    current_agent: Optional[str] = None
    progress_percentage: float = 0.0
    estimated_completion_seconds: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "workflow_progress"


@dataclass
class APIHealthCheck(WorkflowEvent):
    """API health check event."""

    api_name: str = ""
    health_status: str = ""  # "healthy", "degraded", "unhealthy"
    health_details: Optional[str] = None
    health_checks: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = "api_health_check"


class EventSink(ABC):
    """Abstract base class for event sinks."""

    @abstractmethod
    async def emit(self, event: WorkflowEvent) -> None:
        """Emit an event to this sink."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the event sink and cleanup resources."""
        pass


class LogEventSink(EventSink):
    """Event sink that logs events using the standard logger."""

    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level.upper()
        self.logger = get_logger(f"{__name__}.LogEventSink")

    async def emit(self, event: WorkflowEvent) -> None:
        """Log the event."""
        event_data: Dict[str, Any] = {
            "event_type": event.event_type,
            "workflow_id": event.workflow_id,
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": event.correlation_id,
        }

        # Add event-specific data
        if isinstance(event, WorkflowStarted):
            event_data.update(
                {
                    "query_length": len(event.query),
                    "query_preview": event.query[:100],
                    "agents": event.agents,
                    "execution_config": event.execution_config,
                }
            )
        elif isinstance(event, WorkflowCompleted):
            event_data.update(
                {
                    "status": event.status,
                    "execution_time_seconds": event.execution_time_seconds,
                    "agent_count": (
                        len(event.agent_outputs) if event.agent_outputs else 0
                    ),
                    "error_message": event.error_message,
                }
            )
        elif isinstance(event, WorkflowProgress):
            event_data.update(
                {
                    "current_agent": event.current_agent,
                    "progress_percentage": event.progress_percentage,
                    "estimated_completion_seconds": event.estimated_completion_seconds,
                }
            )
        elif isinstance(event, APIHealthCheck):
            event_data.update(
                {
                    "api_name": event.api_name,
                    "health_status": event.health_status,
                    "health_details": event.health_details,
                }
            )

        # Add metadata if present
        if event.metadata:
            event_data["metadata"] = event.metadata

        message = f"[EVENT] {event.event_type}: {event.workflow_id}"
        if self.log_level == "DEBUG":
            message += f" | {event_data}"

        if self.log_level == "DEBUG":
            self.logger.debug(message)
        else:
            self.logger.info(message)

    async def close(self) -> None:
        """No cleanup needed for log sink."""
        pass


class InMemoryEventSink(EventSink):
    """Event sink that stores events in memory for testing and development."""

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: List[WorkflowEvent] = []
        self.logger = get_logger(f"{__name__}.InMemoryEventSink")

    async def emit(self, event: WorkflowEvent) -> None:
        """Store the event in memory."""
        self.events.append(event)

        # Maintain max size by removing oldest events
        if len(self.events) > self.max_events:
            removed_count = len(self.events) - self.max_events
            self.events = self.events[removed_count:]
            self.logger.debug(
                f"Removed {removed_count} old events to maintain max size {self.max_events}"
            )

    async def close(self) -> None:
        """Clear stored events."""
        self.events.clear()

    def get_events(
        self, event_type: Optional[str] = None, workflow_id: Optional[str] = None
    ) -> List[WorkflowEvent]:
        """Get stored events with optional filtering."""
        filtered_events = self.events

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        if workflow_id:
            filtered_events = [
                e for e in filtered_events if e.workflow_id == workflow_id
            ]

        return filtered_events

    def get_recent_events(self, count: int = 10) -> List[WorkflowEvent]:
        """Get the most recent events."""
        return self.events[-count:] if count <= len(self.events) else self.events

    def clear_events(self) -> int:
        """Clear all stored events and return count of cleared events."""
        count = len(self.events)
        self.events.clear()
        return count


class EventEmitter:
    """Event emitter for workflow observability."""

    def __init__(self) -> None:
        self.enabled = self._get_enabled_from_env()
        self.sinks: List[EventSink] = []
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

        # Always add log sink when enabled
        log_level = os.getenv("COGNIVAULT_EVENT_LOG_LEVEL", "INFO")
        self.add_sink(LogEventSink(log_level=log_level))

        # Add in-memory sink for development/testing
        if os.getenv("COGNIVAULT_EVENTS_IN_MEMORY", "false").lower() == "true":
            max_events = int(os.getenv("COGNIVAULT_EVENTS_MAX_MEMORY", "1000"))
            self.add_sink(InMemoryEventSink(max_events=max_events))

    def add_sink(self, sink: EventSink) -> None:
        """Add an event sink."""
        self.sinks.append(sink)
        self.logger.debug(f"Added event sink: {sink.__class__.__name__}")

    def remove_sink(self, sink: EventSink) -> bool:
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
            f"Emitting event: {event.event_type} for workflow {event.workflow_id}"
        )

        # Emit to all sinks, but don't let one sink's failure affect others
        for sink in self.sinks:
            try:
                await sink.emit(event)
            except Exception as e:
                self.logger.error(
                    f"Error emitting to sink {sink.__class__.__name__}: {e}"
                )

    async def emit_workflow_started(
        self,
        workflow_id: str,
        query: str,
        agents: Optional[List[str]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method to emit workflow started event."""
        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id=workflow_id,
            timestamp=datetime.now(timezone.utc),
            correlation_id=correlation_id,
            metadata=metadata,
            query=query,
            agents=agents,
            execution_config=execution_config,
        )
        await self.emit(event)

    async def emit_workflow_completed(
        self,
        workflow_id: str,
        status: str,
        execution_time_seconds: float,
        agent_outputs: Optional[Dict[str, str]] = None,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method to emit workflow completed event."""
        event = WorkflowCompleted(
            event_type="workflow_completed",
            workflow_id=workflow_id,
            timestamp=datetime.now(timezone.utc),
            correlation_id=correlation_id,
            metadata=metadata,
            status=status,
            execution_time_seconds=execution_time_seconds,
            agent_outputs=agent_outputs,
            error_message=error_message,
        )
        await self.emit(event)

    async def emit_api_health_check(
        self,
        workflow_id: str,
        api_name: str,
        health_status: str,
        health_details: Optional[str] = None,
        health_checks: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method to emit API health check event."""
        event = APIHealthCheck(
            event_type="api_health_check",
            workflow_id=workflow_id,
            timestamp=datetime.now(timezone.utc),
            correlation_id=correlation_id,
            metadata=metadata,
            api_name=api_name,
            health_status=health_status,
            health_details=health_details,
            health_checks=health_checks,
        )
        await self.emit(event)

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

    def get_sink_by_type(self, sink_type: type) -> Optional[EventSink]:
        """Get the first sink of the specified type."""
        for sink in self.sinks:
            if isinstance(sink, sink_type):
                return sink
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the event emitter."""
        stats = {
            "enabled": self.enabled,
            "sink_count": len(self.sinks),
            "sink_types": [sink.__class__.__name__ for sink in self.sinks],
        }

        # Get stats from in-memory sink if present
        memory_sink = self.get_sink_by_type(InMemoryEventSink)
        if memory_sink and isinstance(memory_sink, InMemoryEventSink):
            stats["memory_events_count"] = len(memory_sink.events)
            stats["memory_max_events"] = memory_sink.max_events

        return stats


# Global event emitter instance
event_emitter = EventEmitter()


# Convenience functions for global event emitter
async def emit_workflow_started(
    workflow_id: str,
    query: str,
    agents: Optional[List[str]] = None,
    execution_config: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Global convenience function to emit workflow started event."""
    await event_emitter.emit_workflow_started(
        workflow_id, query, agents, execution_config, correlation_id, metadata
    )


async def emit_workflow_completed(
    workflow_id: str,
    status: str,
    execution_time_seconds: float,
    agent_outputs: Optional[Dict[str, str]] = None,
    error_message: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Global convenience function to emit workflow completed event."""
    await event_emitter.emit_workflow_completed(
        workflow_id,
        status,
        execution_time_seconds,
        agent_outputs,
        error_message,
        correlation_id,
        metadata,
    )


def get_event_stats() -> Dict[str, Any]:
    """Get statistics about the global event emitter."""
    return event_emitter.get_stats()


def enable_events() -> None:
    """Enable global event emission."""
    event_emitter.enable()


def disable_events() -> None:
    """Disable global event emission."""
    event_emitter.disable()
