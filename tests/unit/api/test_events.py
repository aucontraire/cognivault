"""
Tests for event system integration with API layer.

This module tests that the main event system works correctly with the API layer
after the event system consolidation.
"""

import pytest
import os
from datetime import datetime, timezone
from unittest.mock import patch, Mock, AsyncMock

from cognivault.events import (
    get_global_event_emitter,
    emit_workflow_started,
    emit_workflow_completed,
    EventType,
    InMemoryEventSink,
)


class TestAPIEventIntegration:
    """Test event system integration with API layer."""

    def setup_method(self):
        """Setup for each test."""
        # Reset global event emitter
        emitter = get_global_event_emitter()
        emitter.sinks.clear()
        emitter.enabled = True

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset global event emitter
        emitter = get_global_event_emitter()
        emitter.sinks.clear()
        emitter.enabled = False

    @pytest.mark.asyncio
    async def test_api_workflow_started_event(self):
        """Test that API layer can emit workflow started events."""
        emitter = get_global_event_emitter()
        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        # Emit workflow started event like the API does
        await emit_workflow_started(
            workflow_id="test-123",
            query="Test query",
            agents=["refiner", "critic"],
            correlation_id="corr-456",
        )

        # Verify event was emitted
        assert len(sink.events) == 1
        event = sink.events[0]
        assert event.event_type == EventType.WORKFLOW_STARTED
        assert event.workflow_id == "test-123"
        assert event.correlation_id == "corr-456"
        assert event.data.get("query") == "Test query"
        assert event.data.get("agents_requested") == ["refiner", "critic"]

    @pytest.mark.asyncio
    async def test_api_workflow_completed_event(self):
        """Test that API layer can emit workflow completed events."""
        emitter = get_global_event_emitter()
        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        # Emit workflow completed event like the API does
        await emit_workflow_completed(
            workflow_id="test-123",
            status="completed",
            execution_time_seconds=5.2,
            agent_outputs={"refiner": "refined output", "critic": "critical analysis"},
            correlation_id="corr-456",
        )

        # Verify event was emitted
        assert len(sink.events) == 1
        event = sink.events[0]
        assert event.event_type == EventType.WORKFLOW_COMPLETED
        assert event.workflow_id == "test-123"
        assert event.correlation_id == "corr-456"
        assert event.data.get("status") == "completed"
        assert event.data.get("execution_time_seconds") == 5.2
        assert event.agent_outputs == {
            "refiner": "refined output",
            "critic": "critical analysis",
        }

    @pytest.mark.asyncio
    async def test_api_workflow_failed_event(self):
        """Test that API layer can emit workflow failed events."""
        emitter = get_global_event_emitter()
        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        # Emit workflow failed event like the API does
        await emit_workflow_completed(
            workflow_id="test-123",
            status="failed",
            execution_time_seconds=2.1,
            error_message="Test error",
            correlation_id="corr-456",
        )

        # Verify event was emitted
        assert len(sink.events) == 1
        event = sink.events[0]
        assert event.event_type == EventType.WORKFLOW_COMPLETED
        assert event.workflow_id == "test-123"
        assert event.correlation_id == "corr-456"
        assert event.data.get("status") == "failed"
        assert event.data.get("execution_time_seconds") == 2.1
        assert event.error_message == "Test error"

    @pytest.mark.asyncio
    async def test_event_emission_disabled(self):
        """Test that event emission can be disabled."""
        emitter = get_global_event_emitter()
        emitter.enabled = False
        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        # Emit event when disabled
        await emit_workflow_started(
            workflow_id="test-123",
            query="Test query",
        )

        # Verify no event was emitted
        assert len(sink.events) == 0

    @pytest.mark.asyncio
    async def test_multiple_events_in_sequence(self):
        """Test that multiple events can be emitted in sequence."""
        emitter = get_global_event_emitter()
        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        # Emit workflow started
        await emit_workflow_started(
            workflow_id="test-123",
            query="Test query",
            correlation_id="corr-456",
        )

        # Emit workflow completed
        await emit_workflow_completed(
            workflow_id="test-123",
            status="completed",
            execution_time_seconds=3.5,
            correlation_id="corr-456",
        )

        # Verify both events were emitted
        assert len(sink.events) == 2

        start_event = sink.events[0]
        assert start_event.event_type == EventType.WORKFLOW_STARTED
        assert start_event.workflow_id == "test-123"
        assert start_event.correlation_id == "corr-456"

        complete_event = sink.events[1]
        assert complete_event.event_type == EventType.WORKFLOW_COMPLETED
        assert complete_event.workflow_id == "test-123"
        assert complete_event.correlation_id == "corr-456"

    def test_event_emitter_singleton(self):
        """Test that get_global_event_emitter returns the same instance."""
        emitter1 = get_global_event_emitter()
        emitter2 = get_global_event_emitter()
        assert emitter1 is emitter2

    @pytest.mark.asyncio
    async def test_event_correlation_context(self):
        """Test that correlation context is preserved in events."""
        emitter = get_global_event_emitter()
        sink = InMemoryEventSink()
        emitter.add_sink(sink)

        # Emit event with correlation context
        await emit_workflow_started(
            workflow_id="test-123",
            query="Test query",
            correlation_id="parent-correlation-id",
        )

        # Verify correlation context is preserved
        assert len(sink.events) == 1
        event = sink.events[0]
        assert event.correlation_id == "parent-correlation-id"
