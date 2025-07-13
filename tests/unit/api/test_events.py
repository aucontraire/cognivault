"""
Unit tests for event system.

Comprehensive test coverage for event system components.
"""

import pytest
import os
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, Mock, AsyncMock

from cognivault.api.events import (
    WorkflowEvent,
    WorkflowStarted,
    WorkflowCompleted,
    WorkflowProgress,
    APIHealthCheck,
    EventSink,
    LogEventSink,
    InMemoryEventSink,
    EventEmitter,
    event_emitter,
    emit_workflow_started,
    emit_workflow_completed,
    get_event_stats,
    enable_events,
    disable_events,
)


class TestWorkflowEvents:
    """Test workflow event dataclasses."""

    def test_workflow_event_base(self):
        """Test base WorkflowEvent creation."""
        timestamp = datetime.now(timezone.utc)
        event = WorkflowEvent(
            event_type="test_event",
            workflow_id="workflow-123",
            timestamp=timestamp,
            correlation_id="corr-456",
            metadata={"key": "value"},
        )

        assert event.event_type == "test_event"
        assert event.workflow_id == "workflow-123"
        assert event.timestamp == timestamp
        assert event.correlation_id == "corr-456"
        assert event.metadata == {"key": "value"}

    def test_workflow_event_timezone_conversion(self):
        """Test that naive timestamps are converted to UTC."""
        naive_timestamp = datetime.now()  # No timezone
        event = WorkflowEvent(
            event_type="test", workflow_id="123", timestamp=naive_timestamp
        )

        assert event.timestamp.tzinfo == timezone.utc

    def test_workflow_started_event(self):
        """Test WorkflowStarted event creation."""
        timestamp = datetime.now(timezone.utc)
        event = WorkflowStarted(
            event_type="workflow_started",  # Will be overridden
            workflow_id="workflow-123",
            timestamp=timestamp,
            query="Test query",
            agents=["refiner", "critic"],
            execution_config={"mode": "test"},
        )

        assert event.event_type == "workflow_started"  # Auto-set in __post_init__
        assert event.query == "Test query"
        assert event.agents == ["refiner", "critic"]
        assert event.execution_config == {"mode": "test"}

    def test_workflow_completed_event(self):
        """Test WorkflowCompleted event creation."""
        timestamp = datetime.now(timezone.utc)
        event = WorkflowCompleted(
            event_type="workflow_completed",
            workflow_id="workflow-123",
            timestamp=timestamp,
            status="completed",
            execution_time_seconds=5.2,
            agent_outputs={"refiner": "output1", "critic": "output2"},
            error_message=None,
        )

        assert event.event_type == "workflow_completed"
        assert event.status == "completed"
        assert event.execution_time_seconds == 5.2
        assert event.agent_outputs == {"refiner": "output1", "critic": "output2"}
        assert event.error_message is None

    def test_workflow_progress_event(self):
        """Test WorkflowProgress event creation."""
        timestamp = datetime.now(timezone.utc)
        event = WorkflowProgress(
            event_type="workflow_progress",
            workflow_id="workflow-123",
            timestamp=timestamp,
            current_agent="critic",
            progress_percentage=75.0,
            estimated_completion_seconds=2.5,
        )

        assert event.event_type == "workflow_progress"
        assert event.current_agent == "critic"
        assert event.progress_percentage == 75.0
        assert event.estimated_completion_seconds == 2.5

    def test_api_health_check_event(self):
        """Test APIHealthCheck event creation."""
        timestamp = datetime.now(timezone.utc)
        event = APIHealthCheck(
            event_type="api_health_check",
            workflow_id="health-123",
            timestamp=timestamp,
            api_name="LangGraphOrchestrationAPI",
            health_status="healthy",
            health_details="All systems operational",
            health_checks={"db": True, "cache": True},
        )

        assert event.event_type == "api_health_check"
        assert event.api_name == "LangGraphOrchestrationAPI"
        assert event.health_status == "healthy"
        assert event.health_details == "All systems operational"
        assert event.health_checks == {"db": True, "cache": True}


class TestLogEventSink:
    """Test LogEventSink functionality."""

    def test_log_event_sink_creation(self):
        """Test LogEventSink creation."""
        sink = LogEventSink(log_level="DEBUG")
        assert sink.log_level == "DEBUG"

    @pytest.mark.asyncio
    async def test_log_event_sink_emit_workflow_started(self):
        """Test logging WorkflowStarted event."""
        sink = LogEventSink(log_level="DEBUG")

        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test query for logging",
            agents=["refiner", "critic"],
            execution_config={"test": True},
        )

        with patch.object(sink.logger, "debug") as mock_debug:
            await sink.emit(event)
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            assert "workflow_started" in call_args
            assert "test-123" in call_args

    @pytest.mark.asyncio
    async def test_log_event_sink_emit_workflow_completed(self):
        """Test logging WorkflowCompleted event."""
        sink = LogEventSink(log_level="INFO")

        event = WorkflowCompleted(
            event_type="workflow_completed",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            status="completed",
            execution_time_seconds=3.5,
            agent_outputs={"refiner": "output"},
            error_message=None,
        )

        with patch.object(sink.logger, "info") as mock_info:
            await sink.emit(event)
            mock_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_event_sink_emit_workflow_progress(self):
        """Test logging WorkflowProgress event."""
        sink = LogEventSink()

        event = WorkflowProgress(
            event_type="workflow_progress",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            current_agent="critic",
            progress_percentage=50.0,
            estimated_completion_seconds=5.0,
        )

        with patch.object(sink.logger, "info") as mock_info:
            await sink.emit(event)
            mock_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_event_sink_emit_api_health_check(self):
        """Test logging APIHealthCheck event."""
        sink = LogEventSink()

        event = APIHealthCheck(
            event_type="api_health_check",
            workflow_id="health-123",
            timestamp=datetime.now(timezone.utc),
            api_name="TestAPI",
            health_status="healthy",
            health_details="All good",
        )

        with patch.object(sink.logger, "info") as mock_info:
            await sink.emit(event)
            mock_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_event_sink_with_metadata(self):
        """Test logging event with metadata."""
        sink = LogEventSink(log_level="DEBUG")

        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test",
            metadata={"user": "test_user", "session": "session_123"},
        )

        with patch.object(sink.logger, "debug") as mock_debug:
            await sink.emit(event)
            mock_debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_event_sink_close(self):
        """Test LogEventSink close method."""
        sink = LogEventSink()
        await sink.close()  # Should not raise an exception


class TestInMemoryEventSink:
    """Test InMemoryEventSink functionality."""

    def test_in_memory_event_sink_creation(self):
        """Test InMemoryEventSink creation."""
        sink = InMemoryEventSink(max_events=500)
        assert sink.max_events == 500
        assert len(sink.events) == 0

    @pytest.mark.asyncio
    async def test_in_memory_event_sink_emit(self):
        """Test storing events in memory."""
        sink = InMemoryEventSink(max_events=10)

        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test query",
        )

        await sink.emit(event)

        assert len(sink.events) == 1
        assert sink.events[0] == event

    @pytest.mark.asyncio
    async def test_in_memory_event_sink_max_events_limit(self):
        """Test max events limit enforcement."""
        sink = InMemoryEventSink(max_events=3)

        # Add more events than the limit
        for i in range(5):
            event = WorkflowStarted(
                event_type="workflow_started",
                workflow_id=f"test-{i}",
                timestamp=datetime.now(timezone.utc),
                query=f"Query {i}",
            )
            await sink.emit(event)

        # Should only keep the last 3 events
        assert len(sink.events) == 3
        assert sink.events[0].workflow_id == "test-2"  # Oldest kept
        assert sink.events[-1].workflow_id == "test-4"  # Newest

    def test_get_events_no_filter(self):
        """Test getting all events without filtering."""
        sink = InMemoryEventSink()

        # Add different types of events
        events = [
            WorkflowStarted(
                event_type="workflow_started",
                workflow_id="test-1",
                timestamp=datetime.now(timezone.utc),
                query="Query 1",
            ),
            WorkflowCompleted(
                event_type="workflow_completed",
                workflow_id="test-1",
                timestamp=datetime.now(timezone.utc),
                status="completed",
                execution_time_seconds=1.0,
            ),
            WorkflowStarted(
                event_type="workflow_started",
                workflow_id="test-2",
                timestamp=datetime.now(timezone.utc),
                query="Query 2",
            ),
        ]

        sink.events = events

        retrieved_events = sink.get_events()
        assert len(retrieved_events) == 3
        assert retrieved_events == events

    def test_get_events_filter_by_type(self):
        """Test getting events filtered by type."""
        sink = InMemoryEventSink()

        events = [
            WorkflowStarted(
                event_type="workflow_started",
                workflow_id="test-1",
                timestamp=datetime.now(timezone.utc),
                query="Query 1",
            ),
            WorkflowCompleted(
                event_type="workflow_completed",
                workflow_id="test-1",
                timestamp=datetime.now(timezone.utc),
                status="completed",
                execution_time_seconds=1.0,
            ),
            WorkflowStarted(
                event_type="workflow_started",
                workflow_id="test-2",
                timestamp=datetime.now(timezone.utc),
                query="Query 2",
            ),
        ]

        sink.events = events

        started_events = sink.get_events(event_type="workflow_started")
        assert len(started_events) == 2
        assert all(e.event_type == "workflow_started" for e in started_events)

    def test_get_events_filter_by_workflow_id(self):
        """Test getting events filtered by workflow ID."""
        sink = InMemoryEventSink()

        events = [
            WorkflowStarted(
                event_type="workflow_started",
                workflow_id="test-1",
                timestamp=datetime.now(timezone.utc),
                query="Query 1",
            ),
            WorkflowCompleted(
                event_type="workflow_completed",
                workflow_id="test-1",
                timestamp=datetime.now(timezone.utc),
                status="completed",
                execution_time_seconds=1.0,
            ),
            WorkflowStarted(
                event_type="workflow_started",
                workflow_id="test-2",
                timestamp=datetime.now(timezone.utc),
                query="Query 2",
            ),
        ]

        sink.events = events

        workflow1_events = sink.get_events(workflow_id="test-1")
        assert len(workflow1_events) == 2
        assert all(e.workflow_id == "test-1" for e in workflow1_events)

    def test_get_recent_events(self):
        """Test getting recent events."""
        sink = InMemoryEventSink()

        # Add 5 events
        for i in range(5):
            event = WorkflowStarted(
                event_type="workflow_started",
                workflow_id=f"test-{i}",
                timestamp=datetime.now(timezone.utc),
                query=f"Query {i}",
            )
            sink.events.append(event)

        recent = sink.get_recent_events(count=3)
        assert len(recent) == 3
        assert recent[0].workflow_id == "test-2"  # 3rd from end
        assert recent[-1].workflow_id == "test-4"  # Last

    def test_get_recent_events_more_than_available(self):
        """Test getting recent events when requesting more than available."""
        sink = InMemoryEventSink()

        # Add 2 events
        for i in range(2):
            event = WorkflowStarted(
                event_type="workflow_started",
                workflow_id=f"test-{i}",
                timestamp=datetime.now(timezone.utc),
                query=f"Query {i}",
            )
            sink.events.append(event)

        recent = sink.get_recent_events(count=5)
        assert len(recent) == 2  # Only returns what's available

    def test_clear_events(self):
        """Test clearing events."""
        sink = InMemoryEventSink()

        # Add events
        for i in range(3):
            event = WorkflowStarted(
                event_type="workflow_started",
                workflow_id=f"test-{i}",
                timestamp=datetime.now(timezone.utc),
                query=f"Query {i}",
            )
            sink.events.append(event)

        count = sink.clear_events()
        assert count == 3
        assert len(sink.events) == 0

    @pytest.mark.asyncio
    async def test_in_memory_event_sink_close(self):
        """Test InMemoryEventSink close method."""
        sink = InMemoryEventSink()

        # Add some events
        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test",
        )
        await sink.emit(event)

        await sink.close()
        assert len(sink.events) == 0


class TestEventEmitter:
    """Test EventEmitter functionality."""

    def setup_method(self):
        """Clean up before each test."""
        # Reset environment variables
        for env_var in [
            "COGNIVAULT_EVENTS_ENABLED",
            "COGNIVAULT_EVENT_LOG_LEVEL",
            "COGNIVAULT_EVENTS_IN_MEMORY",
            "COGNIVAULT_EVENTS_MAX_MEMORY",
        ]:
            if env_var in os.environ:
                del os.environ[env_var]

    def test_event_emitter_creation_disabled(self):
        """Test EventEmitter creation when disabled."""
        emitter = EventEmitter()
        assert emitter.enabled is False
        assert len(emitter.sinks) == 0

    def test_event_emitter_creation_enabled(self):
        """Test EventEmitter creation when enabled."""
        with patch.dict(os.environ, {"COGNIVAULT_EVENTS_ENABLED": "true"}):
            emitter = EventEmitter()
            assert emitter.enabled is True
            assert len(emitter.sinks) >= 1  # At least log sink

    def test_event_emitter_with_in_memory_sink(self):
        """Test EventEmitter creation with in-memory sink."""
        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_EVENTS_ENABLED": "true",
                "COGNIVAULT_EVENTS_IN_MEMORY": "true",
                "COGNIVAULT_EVENTS_MAX_MEMORY": "500",
            },
        ):
            emitter = EventEmitter()
            assert emitter.enabled is True
            assert len(emitter.sinks) == 2  # Log + in-memory

    def test_add_sink(self):
        """Test adding event sink."""
        emitter = EventEmitter()
        sink = InMemoryEventSink()

        emitter.add_sink(sink)
        assert sink in emitter.sinks

    def test_remove_sink(self):
        """Test removing event sink."""
        emitter = EventEmitter()
        sink = InMemoryEventSink()

        emitter.add_sink(sink)
        assert sink in emitter.sinks

        result = emitter.remove_sink(sink)
        assert result is True
        assert sink not in emitter.sinks

    def test_remove_sink_not_found(self):
        """Test removing sink that's not present."""
        emitter = EventEmitter()
        sink = InMemoryEventSink()

        result = emitter.remove_sink(sink)
        assert result is False

    @pytest.mark.asyncio
    async def test_emit_event_disabled(self):
        """Test emitting event when disabled."""
        emitter = EventEmitter()
        assert emitter.enabled is False

        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test",
        )

        # Should not raise an exception
        await emitter.emit(event)

    @pytest.mark.asyncio
    async def test_emit_event_enabled_no_sinks(self):
        """Test emitting event when enabled but no sinks."""
        emitter = EventEmitter()
        emitter.enabled = True  # Force enable without sinks

        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test",
        )

        # Should not raise an exception
        await emitter.emit(event)

    @pytest.mark.asyncio
    async def test_emit_event_with_sinks(self):
        """Test emitting event with configured sinks."""
        emitter = EventEmitter()
        emitter.enabled = True

        sink1 = InMemoryEventSink()
        sink2 = InMemoryEventSink()
        emitter.add_sink(sink1)
        emitter.add_sink(sink2)

        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test",
        )

        await emitter.emit(event)

        assert len(sink1.events) == 1
        assert len(sink2.events) == 1
        assert sink1.events[0] == event
        assert sink2.events[0] == event

    @pytest.mark.asyncio
    async def test_emit_event_sink_failure(self):
        """Test emitting event when one sink fails."""
        emitter = EventEmitter()
        emitter.enabled = True

        good_sink = InMemoryEventSink()
        bad_sink = Mock(spec=EventSink)
        bad_sink.emit = AsyncMock(side_effect=Exception("Sink failure"))

        emitter.add_sink(good_sink)
        emitter.add_sink(bad_sink)

        event = WorkflowStarted(
            event_type="workflow_started",
            workflow_id="test-123",
            timestamp=datetime.now(timezone.utc),
            query="Test",
        )

        with patch.object(emitter.logger, "error") as mock_error:
            await emitter.emit(event)
            mock_error.assert_called_once()

        # Good sink should still receive the event
        assert len(good_sink.events) == 1

    @pytest.mark.asyncio
    async def test_emit_workflow_started_convenience(self):
        """Test emit_workflow_started convenience method."""
        emitter = EventEmitter()
        emitter.enabled = True

        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        await emitter.emit_workflow_started(
            workflow_id="test-123",
            query="Test query",
            agents=["refiner", "critic"],
            execution_config={"test": True},
            correlation_id="corr-456",
            metadata={"user": "test"},
        )

        assert len(sink.events) == 1
        event = sink.events[0]
        assert isinstance(event, WorkflowStarted)
        assert event.workflow_id == "test-123"
        assert event.query == "Test query"
        assert event.agents == ["refiner", "critic"]

    @pytest.mark.asyncio
    async def test_emit_workflow_completed_convenience(self):
        """Test emit_workflow_completed convenience method."""
        emitter = EventEmitter()
        emitter.enabled = True

        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        await emitter.emit_workflow_completed(
            workflow_id="test-123",
            status="completed",
            execution_time_seconds=5.2,
            agent_outputs={"refiner": "output"},
            error_message=None,
            correlation_id="corr-456",
            metadata={"session": "test"},
        )

        assert len(sink.events) == 1
        event = sink.events[0]
        assert isinstance(event, WorkflowCompleted)
        assert event.workflow_id == "test-123"
        assert event.status == "completed"
        assert event.execution_time_seconds == 5.2

    @pytest.mark.asyncio
    async def test_emit_api_health_check_convenience(self):
        """Test emit_api_health_check convenience method."""
        emitter = EventEmitter()
        emitter.enabled = True

        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        await emitter.emit_api_health_check(
            workflow_id="health-123",
            api_name="TestAPI",
            health_status="healthy",
            health_details="All good",
            health_checks={"db": True},
            correlation_id="corr-456",
            metadata={"check_time": "now"},
        )

        assert len(sink.events) == 1
        event = sink.events[0]
        assert isinstance(event, APIHealthCheck)
        assert event.api_name == "TestAPI"
        assert event.health_status == "healthy"

    def test_enable_emitter(self):
        """Test enabling the emitter."""
        emitter = EventEmitter()
        assert emitter.enabled is False

        with patch.object(emitter, "_initialize_default_sinks") as mock_init:
            emitter.enable()
            assert emitter.enabled is True
            mock_init.assert_called_once()

    def test_enable_emitter_already_enabled(self):
        """Test enabling already enabled emitter."""
        emitter = EventEmitter()
        emitter.enabled = True

        with patch.object(emitter, "_initialize_default_sinks") as mock_init:
            emitter.enable()
            assert emitter.enabled is True
            mock_init.assert_not_called()

    def test_disable_emitter(self):
        """Test disabling the emitter."""
        emitter = EventEmitter()
        emitter.enabled = True

        emitter.disable()
        assert emitter.enabled is False

    def test_disable_emitter_already_disabled(self):
        """Test disabling already disabled emitter."""
        emitter = EventEmitter()
        assert emitter.enabled is False

        emitter.disable()
        assert emitter.enabled is False

    @pytest.mark.asyncio
    async def test_close_emitter(self):
        """Test closing the emitter."""
        emitter = EventEmitter()
        emitter.enabled = True

        sink1 = Mock(spec=EventSink)
        sink1.close = AsyncMock()
        sink2 = Mock(spec=EventSink)
        sink2.close = AsyncMock()

        emitter.add_sink(sink1)
        emitter.add_sink(sink2)

        await emitter.close()

        sink1.close.assert_called_once()
        sink2.close.assert_called_once()
        assert len(emitter.sinks) == 0
        assert emitter.enabled is False

    @pytest.mark.asyncio
    async def test_close_emitter_sink_failure(self):
        """Test closing emitter when sink close fails."""
        emitter = EventEmitter()
        emitter.enabled = True

        good_sink = Mock(spec=EventSink)
        good_sink.close = AsyncMock()
        bad_sink = Mock(spec=EventSink)
        bad_sink.close = AsyncMock(side_effect=Exception("Close failure"))

        emitter.add_sink(good_sink)
        emitter.add_sink(bad_sink)

        with patch.object(emitter.logger, "error") as mock_error:
            await emitter.close()
            mock_error.assert_called_once()

        good_sink.close.assert_called_once()
        bad_sink.close.assert_called_once()

    def test_get_sink_by_type(self):
        """Test getting sink by type."""
        emitter = EventEmitter()

        memory_sink = InMemoryEventSink()
        log_sink = LogEventSink()

        emitter.add_sink(memory_sink)
        emitter.add_sink(log_sink)

        found_memory = emitter.get_sink_by_type(InMemoryEventSink)
        assert found_memory is memory_sink

        found_log = emitter.get_sink_by_type(LogEventSink)
        assert found_log is log_sink

        found_none = emitter.get_sink_by_type(Mock)
        assert found_none is None

    def test_get_stats_basic(self):
        """Test getting basic stats."""
        emitter = EventEmitter()
        emitter.enabled = True

        sink = LogEventSink()
        emitter.add_sink(sink)

        stats = emitter.get_stats()

        assert stats["enabled"] is True
        assert stats["sink_count"] == 1
        assert stats["sink_types"] == ["LogEventSink"]

    def test_get_stats_with_memory_sink(self):
        """Test getting stats with memory sink."""
        emitter = EventEmitter()
        emitter.enabled = True

        memory_sink = InMemoryEventSink(max_events=100)
        # Add some events
        for i in range(5):
            event = WorkflowStarted(
                event_type="workflow_started",
                workflow_id=f"test-{i}",
                timestamp=datetime.now(timezone.utc),
                query=f"Query {i}",
            )
            memory_sink.events.append(event)

        emitter.add_sink(memory_sink)

        stats = emitter.get_stats()

        assert stats["memory_events_count"] == 5
        assert stats["memory_max_events"] == 100


class TestGlobalEventFunctions:
    """Test global event functions."""

    def setup_method(self):
        """Reset global emitter state."""
        event_emitter.enabled = False
        event_emitter.sinks.clear()

    def teardown_method(self):
        """Clean up after tests."""
        event_emitter.enabled = False
        event_emitter.sinks.clear()

    @pytest.mark.asyncio
    async def test_global_emit_workflow_started(self):
        """Test global emit_workflow_started function."""
        event_emitter.enabled = True
        sink = InMemoryEventSink()
        event_emitter.add_sink(sink)

        await emit_workflow_started(
            workflow_id="global-test",
            query="Global test query",
            agents=["refiner"],
            correlation_id="global-corr",
        )

        assert len(sink.events) == 1
        event = sink.events[0]
        assert event.workflow_id == "global-test"
        assert event.query == "Global test query"

    @pytest.mark.asyncio
    async def test_global_emit_workflow_completed(self):
        """Test global emit_workflow_completed function."""
        event_emitter.enabled = True
        sink = InMemoryEventSink()
        event_emitter.add_sink(sink)

        await emit_workflow_completed(
            workflow_id="global-test",
            status="completed",
            execution_time_seconds=2.5,
            agent_outputs={"refiner": "output"},
            correlation_id="global-corr",
        )

        assert len(sink.events) == 1
        event = sink.events[0]
        assert event.workflow_id == "global-test"
        assert event.status == "completed"

    def test_get_event_stats_global(self):
        """Test global get_event_stats function."""
        event_emitter.enabled = True
        sink = LogEventSink()
        event_emitter.add_sink(sink)

        stats = get_event_stats()

        assert stats["enabled"] is True
        assert stats["sink_count"] == 1

    def test_enable_events_global(self):
        """Test global enable_events function."""
        assert event_emitter.enabled is False

        with patch.object(event_emitter, "enable") as mock_enable:
            enable_events()
            mock_enable.assert_called_once()

    def test_disable_events_global(self):
        """Test global disable_events function."""
        event_emitter.enabled = True

        with patch.object(event_emitter, "disable") as mock_disable:
            disable_events()
            mock_disable.assert_called_once()
