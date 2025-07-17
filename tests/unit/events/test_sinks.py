"""
Comprehensive tests for Event Sinks module.

This module tests all event sink implementations including:
- ConsoleEventSink with various configurations
- FileEventSink with file operations, rotation, and error handling
- InMemoryEventSink with filtering and memory management
- EventFilters and EventStatistics functionality
"""

import pytest
import json
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO

from cognivault.events.sinks import (
    EventSink,
    ConsoleEventSink,
    FileEventSink,
    InMemoryEventSink,
    create_file_sink,
)
from cognivault.events.types import (
    WorkflowEvent,
    EventType,
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    EventFilters,
    EventStatistics,
)


@pytest.fixture
def sample_workflow_started_event():
    """Create a sample workflow started event for testing."""
    return WorkflowStartedEvent(
        event_type=EventType.WORKFLOW_STARTED,
        workflow_id="test-workflow-123",
        timestamp=datetime.now(timezone.utc),
        correlation_id="test-correlation-456",
        query="Test query for workflow",
        agents_requested=["refiner", "critic"],
        execution_config={"timeout": 30},
        metadata={"test_key": "test_value"},
    )


@pytest.fixture
def sample_workflow_completed_event():
    """Create a sample workflow completed event for testing."""
    return WorkflowCompletedEvent(
        event_type=EventType.WORKFLOW_COMPLETED,
        workflow_id="test-workflow-123",
        timestamp=datetime.now(timezone.utc),
        correlation_id="test-correlation-456",
        status="completed",
        execution_time_seconds=5.2,
        agent_outputs={"refiner": "refined output", "critic": "critical analysis"},
        successful_agents=["refiner", "critic"],
        failed_agents=[],
        error_message=None,
        metadata={"result_quality": "high"},
    )


@pytest.fixture
def sample_agent_execution_completed_event():
    """Create a sample agent execution completed event for testing."""
    event = AgentExecutionCompletedEvent(
        event_type=EventType.AGENT_EXECUTION_COMPLETED,
        workflow_id="test-workflow-123",
        timestamp=datetime.now(timezone.utc),
        correlation_id="test-correlation-456",
        agent_metadata=None,  # Simplified for testing
        agent_name="refiner",
        success=True,
        output_context={"output": "refined content"},
        metadata={"confidence": 0.95},
    )
    # Manually set execution time for testing
    event.execution_time_ms = 150.5
    return event


@pytest.fixture
def sample_failed_agent_event():
    """Create a sample failed agent execution event for testing."""
    event = AgentExecutionCompletedEvent(
        event_type=EventType.AGENT_EXECUTION_COMPLETED,
        workflow_id="test-workflow-456",
        timestamp=datetime.now(timezone.utc),
        correlation_id="test-correlation-789",
        agent_metadata=None,
        agent_name="critic",
        success=False,
        output_context={},
        metadata={"retry_count": 3},
    )
    # Set error message and execution time manually
    event.error_message = "Analysis failed due to timeout"
    event.execution_time_ms = 75.0
    return event


class TestEventSinkAbstractBase:
    """Test the abstract EventSink base class."""

    def test_cannot_instantiate_abstract_base(self):
        """Test that EventSink cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EventSink()

    def test_abstract_methods_required(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteEventSink(EventSink):
            pass

        with pytest.raises(TypeError):
            IncompleteEventSink()


class TestConsoleEventSink:
    """Test ConsoleEventSink functionality."""

    @pytest.mark.asyncio
    async def test_console_sink_initialization(self):
        """Test ConsoleEventSink initialization with different options."""
        # Test default initialization
        sink = ConsoleEventSink()
        assert sink.include_metadata is False
        assert sink.max_line_length == 120

        # Test custom initialization
        sink = ConsoleEventSink(include_metadata=True, max_line_length=80)
        assert sink.include_metadata is True
        assert sink.max_line_length == 80

    @pytest.mark.asyncio
    async def test_console_sink_basic_event_output(
        self, sample_workflow_started_event, capsys
    ):
        """Test basic event output to console."""
        sink = ConsoleEventSink()

        await sink.emit(sample_workflow_started_event)

        captured = capsys.readouterr()
        output = captured.out

        # Check that key information is in the output
        assert "workflow.started" in output
        assert "test-wor" in output  # Shortened workflow ID
        assert sample_workflow_started_event.timestamp.strftime("%H:%M:%S") in output

    @pytest.mark.asyncio
    async def test_console_sink_agent_event_output(
        self, sample_agent_execution_completed_event, capsys
    ):
        """Test agent event output with agent name."""
        sink = ConsoleEventSink()

        await sink.emit(sample_agent_execution_completed_event)

        captured = capsys.readouterr()
        output = captured.out

        # Check agent-specific formatting
        assert "agent.execution.completed" in output
        assert "refiner" in output
        assert "✓" in output  # Success indicator
        assert "150.5ms" in output  # Execution time

    @pytest.mark.asyncio
    async def test_console_sink_failed_event_output(
        self, sample_failed_agent_event, capsys
    ):
        """Test failed event output with error information."""
        sink = ConsoleEventSink()

        await sink.emit(sample_failed_agent_event)

        captured = capsys.readouterr()
        output = captured.out

        # Check error formatting
        assert "✗" in output  # Failure indicator
        assert "ERROR:" in output
        assert "Analysis failed due to timeout" in output

    @pytest.mark.asyncio
    async def test_console_sink_with_metadata(
        self, sample_workflow_completed_event, capsys
    ):
        """Test console output with metadata included."""
        sink = ConsoleEventSink(include_metadata=True)

        await sink.emit(sample_workflow_completed_event)

        captured = capsys.readouterr()
        output = captured.out

        # Check metadata output
        assert "Metadata:" in output
        assert "result_quality" in output
        assert "high" in output

    @pytest.mark.asyncio
    async def test_console_sink_line_truncation(
        self, sample_workflow_started_event, capsys
    ):
        """Test line truncation for long output."""
        sink = ConsoleEventSink(max_line_length=30)  # Very short to ensure truncation

        await sink.emit(sample_workflow_started_event)

        captured = capsys.readouterr()
        output = captured.out.strip()

        # Check truncation - output should be exactly 30 chars or less
        assert len(output) <= 30
        if len(output) == 30:
            assert output.endswith("...")

    @pytest.mark.asyncio
    async def test_console_sink_close(self):
        """Test console sink close operation."""
        sink = ConsoleEventSink()
        await sink.close()  # Should not raise any exception


class TestFileEventSink:
    """Test FileEventSink functionality."""

    @pytest.mark.asyncio
    async def test_file_sink_initialization(self):
        """Test FileEventSink initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_events.jsonl"

            sink = FileEventSink(
                file_path=str(file_path),
                max_file_size_mb=50,
                rotate_files=True,
            )

            assert sink.file_path == file_path
            assert sink.max_file_size_mb == 50
            assert sink.rotate_files is True
            assert sink.file_path.parent.exists()

    @pytest.mark.asyncio
    async def test_file_sink_directory_creation(self):
        """Test that FileEventSink creates directories if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "events" / "test.jsonl"

            sink = FileEventSink(file_path=str(nested_path))

            assert nested_path.parent.exists()

    @pytest.mark.asyncio
    async def test_file_sink_event_writing(self, sample_workflow_started_event):
        """Test basic event writing to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            sink = FileEventSink(file_path=str(file_path))

            await sink.emit(sample_workflow_started_event)

            # Check file exists and contains event
            assert file_path.exists()

            with file_path.open("r") as f:
                content = f.read()

            # Parse JSON line
            event_data = json.loads(content.strip())
            assert event_data["event_type"] == "workflow.started"
            assert event_data["workflow_id"] == "test-workflow-123"

    @pytest.mark.asyncio
    async def test_file_sink_multiple_events(
        self, sample_workflow_started_event, sample_workflow_completed_event
    ):
        """Test writing multiple events to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            sink = FileEventSink(file_path=str(file_path))

            await sink.emit(sample_workflow_started_event)
            await sink.emit(sample_workflow_completed_event)

            # Check both events are written
            with file_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 2

            event1 = json.loads(lines[0])
            event2 = json.loads(lines[1])

            assert event1["event_type"] == "workflow.started"
            assert event2["event_type"] == "workflow.completed"

    @pytest.mark.asyncio
    async def test_file_sink_statistics_tracking(
        self, sample_workflow_started_event, sample_agent_execution_completed_event
    ):
        """Test that FileEventSink tracks statistics correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            sink = FileEventSink(file_path=str(file_path))

            await sink.emit(sample_workflow_started_event)
            await sink.emit(sample_agent_execution_completed_event)

            stats = sink.get_statistics()
            assert stats.total_events == 2
            assert "workflow.started" in stats.events_by_type
            assert "agent.execution.completed" in stats.events_by_type

    @pytest.mark.asyncio
    async def test_file_sink_with_filters(
        self, sample_workflow_started_event, sample_workflow_completed_event
    ):
        """Test FileEventSink with event filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"

            # Create filter that only allows workflow events
            filters = EventFilters(event_type=EventType.WORKFLOW_STARTED)
            sink = FileEventSink(file_path=str(file_path), filters=filters)

            await sink.emit(sample_workflow_started_event)  # Should be written
            await sink.emit(sample_workflow_completed_event)  # Should be filtered out

            # Check only one event was written
            with file_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 1
            event = json.loads(lines[0])
            assert event["event_type"] == "workflow.started"

    @pytest.mark.asyncio
    async def test_file_sink_rotation(self, sample_workflow_started_event):
        """Test file rotation when size limit is exceeded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"

            # Create a very small size limit to trigger rotation
            sink = FileEventSink(
                file_path=str(file_path), max_file_size_mb=0.000001, rotate_files=True
            )

            # Write first event
            await sink.emit(sample_workflow_started_event)

            # Get the original file content
            original_content = file_path.read_text()

            # Write second event - should trigger rotation
            await sink.emit(sample_workflow_started_event)

            # Check that a rotated file exists
            rotated_files = list(
                file_path.parent.glob(f"{file_path.stem}_*{file_path.suffix}")
            )
            assert len(rotated_files) == 1

            # Check that the rotated file contains the original content
            rotated_content = rotated_files[0].read_text()
            assert rotated_content == original_content

    @pytest.mark.asyncio
    async def test_file_sink_rotation_disabled(self, sample_workflow_started_event):
        """Test that rotation can be disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"

            sink = FileEventSink(
                file_path=str(file_path), max_file_size_mb=0.000001, rotate_files=False
            )

            # Write multiple events
            await sink.emit(sample_workflow_started_event)
            await sink.emit(sample_workflow_started_event)

            # Check that no rotated files exist
            rotated_files = list(
                file_path.parent.glob(f"{file_path.stem}_*{file_path.suffix}")
            )
            assert len(rotated_files) == 0

    @pytest.mark.asyncio
    async def test_file_sink_write_error_handling(self, sample_workflow_started_event):
        """Test error handling when file writing fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            sink = FileEventSink(file_path=str(file_path))

            # Mock file open to raise an exception
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                # Should not raise exception, but log error
                await sink.emit(sample_workflow_started_event)

    @pytest.mark.asyncio
    async def test_file_sink_rotation_error_handling(
        self, sample_workflow_started_event
    ):
        """Test error handling when file rotation fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            sink = FileEventSink(file_path=str(file_path), max_file_size_mb=0.000001)

            # Write first event
            await sink.emit(sample_workflow_started_event)

            # Mock file rename to raise an exception
            with patch.object(
                Path, "rename", side_effect=PermissionError("Permission denied")
            ):
                # Should not raise exception, but log error
                await sink.emit(sample_workflow_started_event)

    @pytest.mark.asyncio
    async def test_file_sink_close(self):
        """Test file sink close operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            sink = FileEventSink(file_path=str(file_path))

            await sink.close()  # Should not raise any exception


class TestInMemoryEventSink:
    """Test InMemoryEventSink functionality."""

    @pytest.mark.asyncio
    async def test_in_memory_sink_initialization(self):
        """Test InMemoryEventSink initialization."""
        # Test default initialization
        sink = InMemoryEventSink()
        assert sink.max_events == 1000
        assert len(sink.events) == 0

        # Test custom initialization
        sink = InMemoryEventSink(max_events=500)
        assert sink.max_events == 500

    @pytest.mark.asyncio
    async def test_in_memory_sink_event_storage(
        self, sample_workflow_started_event, sample_workflow_completed_event
    ):
        """Test basic event storage."""
        sink = InMemoryEventSink()

        await sink.emit(sample_workflow_started_event)
        await sink.emit(sample_workflow_completed_event)

        assert len(sink.events) == 2
        assert sink.events[0] == sample_workflow_started_event
        assert sink.events[1] == sample_workflow_completed_event

    @pytest.mark.asyncio
    async def test_in_memory_sink_max_events_limit(self, sample_workflow_started_event):
        """Test that max_events limit is enforced."""
        sink = InMemoryEventSink(max_events=2)

        # Add 3 events
        for i in range(3):
            event = WorkflowStartedEvent(
                event_type=EventType.WORKFLOW_STARTED,
                workflow_id=f"workflow-{i}",
                timestamp=datetime.now(timezone.utc),
                correlation_id=f"corr-{i}",
                query=f"Query {i}",
                agents_requested=["refiner"],
                execution_config={},
                metadata={},
            )
            await sink.emit(event)

        # Should only keep the last 2 events
        assert len(sink.events) == 2
        assert sink.events[0].workflow_id == "workflow-1"
        assert sink.events[1].workflow_id == "workflow-2"

    @pytest.mark.asyncio
    async def test_in_memory_sink_filtering(
        self, sample_workflow_started_event, sample_workflow_completed_event
    ):
        """Test InMemoryEventSink with filters."""
        filters = EventFilters(event_type=EventType.WORKFLOW_STARTED)
        sink = InMemoryEventSink(filters=filters)

        await sink.emit(sample_workflow_started_event)  # Should be stored
        await sink.emit(sample_workflow_completed_event)  # Should be filtered out

        assert len(sink.events) == 1
        assert sink.events[0] == sample_workflow_started_event

    @pytest.mark.asyncio
    async def test_in_memory_sink_get_events_filtering(
        self, sample_workflow_started_event, sample_agent_execution_completed_event
    ):
        """Test get_events with filtering parameters."""
        sink = InMemoryEventSink()

        await sink.emit(sample_workflow_started_event)
        await sink.emit(sample_agent_execution_completed_event)

        # Test event type filtering
        workflow_events = sink.get_events(event_type="workflow.started")
        assert len(workflow_events) == 1
        assert workflow_events[0] == sample_workflow_started_event

        # Test workflow ID filtering
        workflow_id_events = sink.get_events(workflow_id="test-workflow-123")
        assert len(workflow_id_events) == 2

        # Test agent name filtering
        agent_events = sink.get_events(agent_name="refiner")
        assert len(agent_events) == 1
        assert agent_events[0] == sample_agent_execution_completed_event

    @pytest.mark.asyncio
    async def test_in_memory_sink_get_recent_events(
        self, sample_workflow_started_event
    ):
        """Test get_recent_events functionality."""
        sink = InMemoryEventSink()

        # Add multiple events
        for i in range(5):
            event = WorkflowStartedEvent(
                event_type=EventType.WORKFLOW_STARTED,
                workflow_id=f"workflow-{i}",
                timestamp=datetime.now(timezone.utc),
                correlation_id=f"corr-{i}",
                query=f"Query {i}",
                agents_requested=["refiner"],
                execution_config={},
                metadata={},
            )
            await sink.emit(event)

        # Get recent events
        recent = sink.get_recent_events(count=3)
        assert len(recent) == 3
        assert recent[-1].workflow_id == "workflow-4"  # Most recent

        # Test when count exceeds available events
        all_recent = sink.get_recent_events(count=10)
        assert len(all_recent) == 5

    @pytest.mark.asyncio
    async def test_in_memory_sink_statistics(
        self, sample_workflow_started_event, sample_agent_execution_completed_event
    ):
        """Test statistics tracking."""
        sink = InMemoryEventSink()

        await sink.emit(sample_workflow_started_event)
        await sink.emit(sample_agent_execution_completed_event)

        stats = sink.get_statistics()
        assert stats.total_events == 2
        assert "workflow.started" in stats.events_by_type
        assert "agent.execution.completed" in stats.events_by_type

    @pytest.mark.asyncio
    async def test_in_memory_sink_clear_events(self, sample_workflow_started_event):
        """Test clearing events."""
        sink = InMemoryEventSink()

        await sink.emit(sample_workflow_started_event)
        assert len(sink.events) == 1

        cleared_count = sink.clear_events()
        assert cleared_count == 1
        assert len(sink.events) == 0

        # Statistics should also be reset
        stats = sink.get_statistics()
        assert stats.total_events == 0

    @pytest.mark.asyncio
    async def test_in_memory_sink_close(self, sample_workflow_started_event):
        """Test in-memory sink close operation."""
        sink = InMemoryEventSink()

        await sink.emit(sample_workflow_started_event)
        assert len(sink.events) == 1

        await sink.close()
        assert len(sink.events) == 0


class TestEventFilters:
    """Test EventFilters functionality."""

    @pytest.fixture
    def sample_events(self):
        """Create a variety of events for filter testing."""
        return [
            WorkflowStartedEvent(
                event_type=EventType.WORKFLOW_STARTED,
                workflow_id="workflow-1",
                timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                correlation_id="corr-1",
                query="Test query 1",
                agents_requested=["refiner"],
                execution_config={},
                metadata={},
            ),
            WorkflowCompletedEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                workflow_id="workflow-1",
                timestamp=datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
                correlation_id="corr-1",
                status="completed",
                execution_time_seconds=5.0,
                agent_outputs={"refiner": "output"},
                successful_agents=["refiner"],
                failed_agents=[],
                error_message=None,
                metadata={},
            ),
            AgentExecutionCompletedEvent(
                event_type=EventType.AGENT_EXECUTION_COMPLETED,
                workflow_id="workflow-2",
                timestamp=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                correlation_id="corr-2",
                agent_metadata=None,
                agent_name="critic",
                success=False,
                output_context={},
                metadata={},
            ),
        ]

    def test_event_filters_event_type(self, sample_events):
        """Test filtering by event type."""
        filters = EventFilters(event_type=EventType.WORKFLOW_STARTED)

        matching = [event for event in sample_events if filters.matches(event)]
        assert len(matching) == 1
        assert matching[0].event_type == EventType.WORKFLOW_STARTED

    def test_event_filters_workflow_id(self, sample_events):
        """Test filtering by workflow ID."""
        filters = EventFilters(workflow_id="workflow-1")

        matching = [event for event in sample_events if filters.matches(event)]
        assert len(matching) == 2
        assert all(event.workflow_id == "workflow-1" for event in matching)

    def test_event_filters_correlation_id(self, sample_events):
        """Test filtering by correlation ID."""
        filters = EventFilters(correlation_id="corr-2")

        matching = [event for event in sample_events if filters.matches(event)]
        assert len(matching) == 1
        assert matching[0].correlation_id == "corr-2"

    def test_event_filters_has_errors(self, sample_events):
        """Test filtering by error presence."""
        # Add an event with an error for testing
        error_event = AgentExecutionCompletedEvent(
            event_type=EventType.AGENT_EXECUTION_COMPLETED,
            workflow_id="workflow-error",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            correlation_id="corr-error",
            agent_metadata=None,
            agent_name="error_agent",
            success=False,
            output_context={},
            metadata={},
        )
        error_event.error_message = "Test error message"
        events_with_error = sample_events + [error_event]

        # Filter for events with errors
        filters = EventFilters(has_errors=True)
        matching = [event for event in events_with_error if filters.matches(event)]
        assert len(matching) == 1
        assert matching[0].error_message == "Test error message"

        # Filter for events without errors
        filters = EventFilters(has_errors=False)
        matching = [event for event in events_with_error if filters.matches(event)]
        assert len(matching) == 3  # The original 3 events without errors
        assert all(not event.error_message for event in matching)

    def test_event_filters_time_range(self, sample_events):
        """Test filtering by time range."""
        start_time = datetime(2024, 1, 1, 10, 3, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 10, 7, 0, tzinfo=timezone.utc)

        filters = EventFilters(start_time=start_time, end_time=end_time)

        matching = [event for event in sample_events if filters.matches(event)]
        assert len(matching) == 1
        assert matching[0].event_type == EventType.WORKFLOW_COMPLETED

    def test_event_filters_combined(self, sample_events):
        """Test combining multiple filters."""
        filters = EventFilters(workflow_id="workflow-1", has_errors=False)

        matching = [event for event in sample_events if filters.matches(event)]
        assert len(matching) == 2
        assert all(event.workflow_id == "workflow-1" for event in matching)
        assert all(not event.error_message for event in matching)


class TestEventStatistics:
    """Test EventStatistics functionality."""

    def test_event_statistics_initialization(self):
        """Test EventStatistics initialization."""
        stats = EventStatistics()
        assert stats.total_events == 0
        assert len(stats.events_by_type) == 0
        assert len(stats.events_by_agent) == 0
        assert stats.average_execution_time_ms == 0.0
        assert stats.error_rate == 0.0

    def test_event_statistics_update_with_workflow_event(
        self, sample_workflow_started_event
    ):
        """Test updating statistics with workflow events."""
        stats = EventStatistics()
        stats.update_with_event(sample_workflow_started_event)

        assert stats.total_events == 1
        assert stats.events_by_type["workflow.started"] == 1

    def test_event_statistics_update_with_agent_event(
        self, sample_agent_execution_completed_event
    ):
        """Test updating statistics with agent events."""
        stats = EventStatistics()
        stats.update_with_event(sample_agent_execution_completed_event)

        assert stats.total_events == 1
        assert stats.events_by_type["agent.execution.completed"] == 1
        assert stats.average_execution_time_ms == 150.5

    def test_event_statistics_multiple_events(
        self, sample_workflow_started_event, sample_agent_execution_completed_event
    ):
        """Test statistics with multiple events."""
        stats = EventStatistics()

        stats.update_with_event(sample_workflow_started_event)
        stats.update_with_event(sample_agent_execution_completed_event)

        assert stats.total_events == 2
        assert len(stats.events_by_type) == 2
        assert stats.events_by_type["workflow.started"] == 1
        assert stats.events_by_type["agent.execution.completed"] == 1

    def test_event_statistics_execution_time_average(self):
        """Test execution time averaging."""
        stats = EventStatistics()

        # Create events with different execution times
        event1 = AgentExecutionCompletedEvent(
            event_type=EventType.AGENT_EXECUTION_COMPLETED,
            workflow_id="test",
            timestamp=datetime.now(timezone.utc),
            correlation_id="test",
            agent_metadata=None,
            agent_name="test",
            success=True,
            output_context={},
            metadata={},
        )
        event1.execution_time_ms = 100.0

        event2 = AgentExecutionCompletedEvent(
            event_type=EventType.AGENT_EXECUTION_COMPLETED,
            workflow_id="test",
            timestamp=datetime.now(timezone.utc),
            correlation_id="test",
            agent_metadata=None,
            agent_name="test",
            success=True,
            output_context={},
            metadata={},
        )
        event2.execution_time_ms = 200.0

        stats.update_with_event(event1)
        stats.update_with_event(event2)

        # Average should be (100 + 200) / 2 = 150
        assert stats.average_execution_time_ms == 150.0


class TestCreateFileSink:
    """Test the create_file_sink factory function."""

    def test_create_file_sink_basic(self):
        """Test basic file sink creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.jsonl"

            sink = create_file_sink(str(file_path))

            assert isinstance(sink, FileEventSink)
            assert sink.file_path == file_path
            assert sink.max_file_size_mb == 100
            assert sink.rotate_files is True

    def test_create_file_sink_with_options(self):
        """Test file sink creation with custom options."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.jsonl"

            sink = create_file_sink(
                str(file_path),
                max_file_size_mb=50,
                rotate_files=False,
                event_types=["workflow.started"],
                agent_names=["refiner"],
            )

            assert isinstance(sink, FileEventSink)
            assert sink.max_file_size_mb == 50
            assert sink.rotate_files is False
            # Note: Filter creation is partially implemented
            assert sink.filters is not None


class TestSinkIntegration:
    """Test integration scenarios with multiple sinks."""

    @pytest.mark.asyncio
    async def test_multiple_sinks_parallel_processing(
        self, sample_workflow_started_event
    ):
        """Test that multiple sinks can process events in parallel."""
        # Create multiple sinks
        memory_sink = InMemoryEventSink()
        console_sink = ConsoleEventSink()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            file_sink = FileEventSink(file_path=str(file_path))

            # Emit event to all sinks simultaneously
            await asyncio.gather(
                memory_sink.emit(sample_workflow_started_event),
                console_sink.emit(sample_workflow_started_event),
                file_sink.emit(sample_workflow_started_event),
            )

            # Verify all sinks processed the event
            assert len(memory_sink.events) == 1
            assert file_path.exists()

            # Check file content
            with file_path.open("r") as f:
                content = f.read()
            event_data = json.loads(content.strip())
            assert event_data["workflow_id"] == "test-workflow-123"

    @pytest.mark.asyncio
    async def test_sink_resilience_to_failures(self, sample_workflow_started_event):
        """Test that sink failures don't affect other sinks."""
        memory_sink = InMemoryEventSink()

        # Create a file sink and mock the file operations to fail
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            file_sink = FileEventSink(file_path=str(file_path))

            # Mock the file writing operation to fail instead of file.exists()
            with patch("builtins.open", side_effect=OSError("Disk full")):
                # Both sinks process event, file sink should fail gracefully
                await memory_sink.emit(sample_workflow_started_event)
                await file_sink.emit(sample_workflow_started_event)  # Should not raise

                # Memory sink should still work
                assert len(memory_sink.events) == 1
