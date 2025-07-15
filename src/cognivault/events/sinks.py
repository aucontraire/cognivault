"""
Event Sinks for CogniVault Event System.

This module provides various event sinks for handling workflow events,
including console output, file storage, and in-memory collection for
testing and development.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from cognivault.observability import get_logger
from .types import WorkflowEvent, EventFilters, EventStatistics

logger = get_logger(__name__)


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


class ConsoleEventSink(EventSink):
    """Event sink that outputs events to console with formatting."""

    def __init__(self, include_metadata: bool = False, max_line_length: int = 120):
        self.include_metadata = include_metadata
        self.max_line_length = max_line_length
        self.logger = get_logger(f"{__name__}.ConsoleEventSink")

    async def emit(self, event: WorkflowEvent) -> None:
        """Output the event to console."""
        # Format timestamp
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]  # ms precision

        # Create basic event line
        event_line = f"[{timestamp}] {event.event_type.value}"

        # Add agent name if it's an agent event
        if hasattr(event, "agent_name") and event.agent_name:
            event_line += f" | {event.agent_name}"

        # Add workflow ID (shortened)
        workflow_short = (
            event.workflow_id[:8] if len(event.workflow_id) > 8 else event.workflow_id
        )
        event_line += f" | {workflow_short}"

        # Add execution status for completion events
        if hasattr(event, "success"):
            status = "✓" if event.success else "✗"
            event_line += f" {status}"

        # Add execution time if available
        if event.execution_time_ms:
            event_line += f" | {event.execution_time_ms:.1f}ms"

        # Add error information if present
        if event.error_message:
            error_preview = (
                event.error_message[:50] + "..."
                if len(event.error_message) > 50
                else event.error_message
            )
            event_line += f" | ERROR: {error_preview}"

        # Truncate if too long
        if len(event_line) > self.max_line_length:
            event_line = event_line[: self.max_line_length - 3] + "..."

        print(event_line)

        # Optionally print metadata
        if self.include_metadata and (event.metadata or event.data):
            metadata_info = {}
            if event.metadata:
                metadata_info.update(event.metadata)
            if event.data:
                metadata_info.update({"data": event.data})

            metadata_str = json.dumps(metadata_info, indent=2, default=str)
            print(f"  Metadata: {metadata_str}")

    async def close(self) -> None:
        """No cleanup needed for console sink."""
        pass


class FileEventSink(EventSink):
    """Event sink that writes events to a file in JSON Lines format."""

    def __init__(
        self,
        file_path: str,
        max_file_size_mb: int = 100,
        rotate_files: bool = True,
        filters: Optional[EventFilters] = None,
    ):
        self.file_path = Path(file_path)
        self.max_file_size_mb = max_file_size_mb
        self.rotate_files = rotate_files
        self.filters = filters
        self.logger = get_logger(f"{__name__}.FileEventSink")
        self.statistics = EventStatistics()

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"FileEventSink initialized: {self.file_path}")

    async def emit(self, event: WorkflowEvent) -> None:
        """Write the event to file."""
        # Apply filters if configured
        if self.filters and not self.filters.matches(event):
            return

        # Update statistics
        self.statistics.update_with_event(event)

        # Check file size and rotate if needed
        if self.rotate_files and self.file_path.exists():
            file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                await self._rotate_file()

        # Write event as JSON line
        try:
            event_json = json.dumps(event.to_dict(), default=str, separators=(",", ":"))

            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(event_json + "\n")

        except Exception as e:
            self.logger.error(f"Failed to write event to file {self.file_path}: {e}")

    async def _rotate_file(self) -> None:
        """Rotate the current file by renaming it with timestamp."""
        if not self.file_path.exists():
            return

        # Create rotated filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = (
            self.file_path.parent
            / f"{self.file_path.stem}_{timestamp}{self.file_path.suffix}"
        )

        try:
            self.file_path.rename(rotated_path)
            self.logger.info(f"Rotated log file: {self.file_path} -> {rotated_path}")
        except Exception as e:
            self.logger.error(f"Failed to rotate file {self.file_path}: {e}")

    def get_statistics(self) -> EventStatistics:
        """Get event statistics for this sink."""
        return self.statistics

    async def close(self) -> None:
        """Close the file sink."""
        self.logger.info(f"Closing FileEventSink: {self.file_path}")


class InMemoryEventSink(EventSink):
    """Event sink that stores events in memory for testing and development."""

    def __init__(self, max_events: int = 1000, filters: Optional[EventFilters] = None):
        self.max_events = max_events
        self.filters = filters
        self.events: List[WorkflowEvent] = []
        self.statistics = EventStatistics()
        self.logger = get_logger(f"{__name__}.InMemoryEventSink")

    async def emit(self, event: WorkflowEvent) -> None:
        """Store the event in memory."""
        # Apply filters if configured
        if self.filters and not self.filters.matches(event):
            return

        # Update statistics
        self.statistics.update_with_event(event)

        # Add event
        self.events.append(event)

        # Maintain max size by removing oldest events
        if len(self.events) > self.max_events:
            removed_count = len(self.events) - self.max_events
            self.events = self.events[removed_count:]
            self.logger.debug(
                f"Removed {removed_count} old events to maintain max size {self.max_events}"
            )

    def get_events(
        self,
        event_type: Optional[str] = None,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> List[WorkflowEvent]:
        """Get stored events with optional filtering."""
        filtered_events = self.events

        if event_type:
            filtered_events = [
                e for e in filtered_events if e.event_type.value == event_type
            ]

        if workflow_id:
            filtered_events = [
                e for e in filtered_events if e.workflow_id == workflow_id
            ]

        if agent_name:
            filtered_events = [
                e
                for e in filtered_events
                if getattr(e, "agent_name", None) == agent_name
            ]

        return filtered_events

    def get_recent_events(self, count: int = 10) -> List[WorkflowEvent]:
        """Get the most recent events."""
        return self.events[-count:] if count <= len(self.events) else self.events

    def get_statistics(self) -> EventStatistics:
        """Get event statistics for this sink."""
        return self.statistics

    def clear_events(self) -> int:
        """Clear all stored events and return count of cleared events."""
        count = len(self.events)
        self.events.clear()
        self.statistics = EventStatistics()  # Reset statistics
        return count

    async def close(self) -> None:
        """Clear stored events."""
        self.events.clear()
        self.statistics = EventStatistics()
        self.logger.info("InMemoryEventSink closed and cleared")


# Factory function for creating file sinks
def create_file_sink(
    file_path: str,
    max_file_size_mb: int = 100,
    rotate_files: bool = True,
    event_types: Optional[List[str]] = None,
    agent_names: Optional[List[str]] = None,
) -> FileEventSink:
    """Create a file event sink with optional filtering."""
    filters = None

    if event_types or agent_names:
        filters = EventFilters()

        if event_types:
            # Note: For now we'll create separate sinks per event type
            # In a full implementation, we'd support multiple event types in one filter
            pass

        if agent_names:
            # Agent name filtering would be handled in the filter logic
            pass

    return FileEventSink(
        file_path=file_path,
        max_file_size_mb=max_file_size_mb,
        rotate_files=rotate_files,
        filters=filters,
    )
