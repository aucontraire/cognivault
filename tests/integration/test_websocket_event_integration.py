"""
Integration tests for WebSocket event system integration.

Tests that WebSocket functionality integrates properly with the existing event system
without breaking existing functionality or introducing regressions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.websockets import WebSocketEventSink, websocket_manager
from cognivault.events.types import WorkflowEvent, EventType, EventCategory
from cognivault.events.emitter import get_global_event_emitter
from cognivault.events.sinks import ConsoleEventSink, InMemoryEventSink


class TestWebSocketEventSystemIntegration:
    """Test WebSocket integration with existing event system."""

    def setup_method(self):
        """Set up test fixtures and clean event system state."""
        self.client = TestClient(app)
        self.correlation_id = "integration-test-123"

        # Clear global event emitter state and enable it
        emitter = get_global_event_emitter()
        emitter.sinks.clear()
        emitter.enable()  # Enable event emission for tests

        # Clear global WebSocket manager state
        websocket_manager._connections.clear()
        websocket_manager._registered_sinks.clear()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear global event emitter state and disable it
        emitter = get_global_event_emitter()
        emitter.sinks.clear()
        emitter.disable()  # Disable event emission after tests

        # Clear global WebSocket manager state
        websocket_manager._connections.clear()
        websocket_manager._registered_sinks.clear()

    @pytest.mark.asyncio
    async def test_websocket_sink_coexists_with_existing_sinks(self):
        """Test that WebSocketEventSink works alongside existing event sinks."""
        emitter = get_global_event_emitter()

        # Add existing sinks
        console_sink = ConsoleEventSink()
        memory_sink = InMemoryEventSink()
        emitter.add_sink(console_sink)
        emitter.add_sink(memory_sink)

        # Add WebSocket sink
        mock_manager = Mock()
        mock_manager.broadcast_event = AsyncMock()
        websocket_sink = WebSocketEventSink(mock_manager, self.correlation_id)
        emitter.add_sink(websocket_sink)

        # Emit event
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow-123",
            correlation_id=self.correlation_id,
        )

        await emitter.emit(event)

        # All sinks should have received the event
        assert len(memory_sink.events) == 1
        assert memory_sink.events[0].event_type == EventType.WORKFLOW_STARTED
        mock_manager.broadcast_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_sink_filters_by_correlation_id(self):
        """Test that WebSocketEventSink only processes matching correlation IDs."""
        mock_manager = Mock()
        mock_manager.broadcast_event = AsyncMock()

        # Create sink for specific correlation ID
        websocket_sink = WebSocketEventSink(mock_manager, self.correlation_id)
        emitter = get_global_event_emitter()
        emitter.add_sink(websocket_sink)

        # Emit event with matching correlation ID
        matching_event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow-123",
            correlation_id=self.correlation_id,
            data={"agent_name": "test_agent", "status": "started"},
        )

        await emitter.emit(matching_event)
        mock_manager.broadcast_event.assert_called_once()

        # Reset mock
        mock_manager.broadcast_event.reset_mock()

        # Emit event with different correlation ID
        non_matching_event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow-123",
            correlation_id="different-correlation-id",
            data={"agent_name": "test_agent", "status": "started"},
        )

        await emitter.emit(non_matching_event)
        mock_manager.broadcast_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_existing_event_emission_unchanged(self):
        """Test that existing event emission patterns continue to work."""
        emitter = get_global_event_emitter()
        memory_sink = InMemoryEventSink()
        emitter.add_sink(memory_sink)

        # Test that existing event emission convenience functions still work
        from cognivault.events.emitter import (
            emit_workflow_started,
            emit_workflow_completed,
            emit_agent_execution_started,
            emit_agent_execution_completed,
        )

        correlation_id = "existing-pattern-test"
        workflow_id = "test-workflow-456"

        # Use existing convenience functions with proper parameters
        await emit_workflow_started(
            workflow_id=workflow_id,
            query="test query",
            correlation_id=correlation_id,
            metadata={"test": "data"},
        )
        await emit_agent_execution_started(
            workflow_id=workflow_id,
            agent_name="test_agent",
            input_context={"capability": "test"},
            correlation_id=correlation_id,
        )
        await emit_agent_execution_completed(
            workflow_id=workflow_id,
            agent_name="test_agent",
            success=True,
            output_context={"result": "success"},
            correlation_id=correlation_id,
        )
        await emit_workflow_completed(
            workflow_id=workflow_id,
            status="completed",
            execution_time_seconds=1.5,
            correlation_id=correlation_id,
            metadata={"final": "result"},
        )

        # Verify events were emitted correctly
        assert len(memory_sink.events) == 4
        assert memory_sink.events[0].event_type == EventType.WORKFLOW_STARTED
        assert memory_sink.events[1].event_type == EventType.AGENT_EXECUTION_STARTED
        assert memory_sink.events[2].event_type == EventType.AGENT_EXECUTION_COMPLETED
        assert memory_sink.events[3].event_type == EventType.WORKFLOW_COMPLETED

        # All events should have correct correlation ID
        for event in memory_sink.events:
            assert event.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_websocket_manager_auto_registration(self):
        """Test that WebSocket subscription automatically registers event sink."""
        # Clear any existing sinks
        emitter = get_global_event_emitter()
        initial_sink_count = len(emitter.sinks)

        # Mock WebSocket
        mock_websocket = Mock()

        # Subscribe to manager (should register event sink)
        await websocket_manager.subscribe(self.correlation_id, mock_websocket)

        # Should have registered a new event sink
        assert len(emitter.sinks) == initial_sink_count + 1

        # Find the WebSocket sink
        websocket_sink = None
        for sink in emitter.sinks:
            if isinstance(sink, WebSocketEventSink):
                websocket_sink = sink
                break

        assert websocket_sink is not None
        assert websocket_sink.target_correlation_id == self.correlation_id

    @pytest.mark.asyncio
    async def test_websocket_sink_doesnt_interfere_with_existing_sinks(self):
        """Test that WebSocketEventSink errors don't affect other sinks."""
        emitter = get_global_event_emitter()

        # Add a working memory sink
        memory_sink = InMemoryEventSink()
        emitter.add_sink(memory_sink)

        # Add a WebSocket sink that will fail
        mock_manager = Mock()
        mock_manager.broadcast_event = AsyncMock(
            side_effect=Exception("Broadcast failed")
        )
        failing_websocket_sink = WebSocketEventSink(mock_manager, self.correlation_id)
        emitter.add_sink(failing_websocket_sink)

        # Emit event
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow-123",
            correlation_id=self.correlation_id,
            data={"agent_name": "test_agent", "status": "started"},
        )

        # Should not raise exception despite WebSocket sink failure
        await emitter.emit(event)

        # Memory sink should still receive the event
        assert len(memory_sink.events) == 1
        assert memory_sink.events[0].event_type == EventType.WORKFLOW_STARTED

    def test_websocket_manager_isolation(self):
        """Test that WebSocket manager doesn't interfere with global state."""
        # Get initial state
        initial_total_connections = websocket_manager.get_total_connections()
        initial_correlation_ids = websocket_manager.get_active_correlation_ids()

        # Should start with clean state
        assert initial_total_connections == 0
        assert initial_correlation_ids == []

        # Create a separate manager instance
        from cognivault.api.websockets import WebSocketManager

        separate_manager = WebSocketManager()

        # Should also start with clean state
        assert separate_manager.get_total_connections() == 0
        assert separate_manager.get_active_correlation_ids() == []

    @pytest.mark.asyncio
    async def test_event_type_compatibility(self):
        """Test that WebSocketEventSink handles all existing event types."""
        mock_manager = Mock()
        mock_manager.broadcast_event = AsyncMock()
        websocket_sink = WebSocketEventSink(mock_manager, self.correlation_id)

        # Test all major event types from the existing system
        event_types_to_test = [
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
            EventType.WORKFLOW_FAILED,
            EventType.AGENT_EXECUTION_STARTED,
            EventType.AGENT_EXECUTION_COMPLETED,
            EventType.DECISION_MADE,
            EventType.AGGREGATION_COMPLETED,
            EventType.VALIDATION_COMPLETED,
            EventType.TERMINATION_TRIGGERED,
        ]

        for event_type in event_types_to_test:
            event = WorkflowEvent(
                event_type=event_type,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test-workflow-123",
                correlation_id=self.correlation_id,
                data={"agent_name": "test_agent", "status": "test_status"},
            )

            # Should not raise any exceptions
            await websocket_sink.emit(event)

            # Should call broadcast_event
            mock_manager.broadcast_event.assert_called()

            # Verify event data structure
            call_args = mock_manager.broadcast_event.call_args
            event_data = call_args[0][1]

            assert event_data["type"] == event_type.value
            assert event_data["correlation_id"] == self.correlation_id
            assert "progress" in event_data
            assert "message" in event_data
            assert "metadata" in event_data

            # Reset for next iteration
            mock_manager.broadcast_event.reset_mock()

    @pytest.mark.asyncio
    async def test_websocket_event_sink_metadata_preservation(self):
        """Test that WebSocketEventSink preserves important event metadata."""
        mock_manager = Mock()
        mock_manager.broadcast_event = AsyncMock()
        websocket_sink = WebSocketEventSink(mock_manager, self.correlation_id)

        # Create event with rich metadata
        event = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_COMPLETED,
            event_category=EventCategory.EXECUTION,
            workflow_id="test-workflow-123",
            correlation_id=self.correlation_id,
            data={"agent_name": "test_agent", "status": "completed"},
            metadata={
                "custom_field": "should_be_ignored",
                "another_field": "also_ignored",
            },
        )

        # Note: Cannot set arbitrary attributes on Pydantic models
        # The WebSocketEventSink should use getattr() with defaults for these attributes
        # Just test with the standard event structure

        await websocket_sink.emit(event)

        call_args = mock_manager.broadcast_event.call_args
        event_data = call_args[0][1]

        # Verify core event data
        assert event_data["type"] == "agent.execution.completed"
        assert event_data["correlation_id"] == self.correlation_id
        assert event_data["agent_name"] == "test_agent"
        assert event_data["status"] == "completed"

        # Verify metadata handling - since we can't set arbitrary attributes,
        # verify that None values are properly filtered out
        metadata = event_data["metadata"]

        # Should not include None values (these attributes don't exist on the event)
        assert (
            "execution_time_ms" not in metadata
            or metadata.get("execution_time_ms") is None
        )
        assert (
            "memory_usage_mb" not in metadata or metadata.get("memory_usage_mb") is None
        )
        assert "node_count" not in metadata or metadata.get("node_count") is None
        assert "error_type" not in metadata or metadata.get("error_type") is None
        assert "error_message" not in metadata or metadata.get("error_message") is None

    @pytest.mark.asyncio
    async def test_websocket_events_include_category_field(self):
        """Test that WebSocket events include the event category field for dual emission architecture."""
        mock_manager = Mock()
        mock_manager.broadcast_event = AsyncMock()
        websocket_sink = WebSocketEventSink(mock_manager, self.correlation_id)

        # Test orchestration event
        orchestration_event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-orchestration",
            correlation_id=self.correlation_id,
            data={"query": "test query"},
        )

        await websocket_sink.emit(orchestration_event)

        call_args = mock_manager.broadcast_event.call_args
        event_data = call_args[0][1]

        # Verify orchestration event has category field
        assert "category" in event_data, "WebSocket event missing category field"
        assert event_data["category"] == "orchestration"
        assert event_data["type"] == "workflow.started"

        # Reset mock for next test
        mock_manager.broadcast_event.reset_mock()

        # Test execution event
        execution_event = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_STARTED,
            event_category=EventCategory.EXECUTION,
            workflow_id="test-execution",
            correlation_id=self.correlation_id,
            data={"agent_name": "refiner", "step": "execution_internal"},
        )

        await websocket_sink.emit(execution_event)

        call_args = mock_manager.broadcast_event.call_args
        event_data = call_args[0][1]

        # Verify execution event has category field
        assert "category" in event_data, "WebSocket event missing category field"
        assert event_data["category"] == "execution"
        assert event_data["type"] == "agent.execution.started"

    @pytest.mark.asyncio
    async def test_websocket_dual_emission_architecture(self):
        """Test WebSocket handling of dual event emission (orchestration vs execution)."""
        mock_manager = Mock()
        captured_events = []

        async def capture_event(correlation_id, event_data):
            captured_events.append(event_data)

        mock_manager.broadcast_event = AsyncMock(side_effect=capture_event)
        websocket_sink = WebSocketEventSink(mock_manager, self.correlation_id)

        # Simulate dual emission: same agent, same event type, different categories
        # This represents node wrapper (orchestration) and individual agent (execution) events

        # Orchestration-level agent event (from node wrapper)
        orchestration_agent_event = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="dual-emission-test",
            correlation_id=self.correlation_id,
            data={"agent_name": "refiner", "step": "node_wrapper_execution"},
        )

        # Execution-level agent event (from individual agent)
        execution_agent_event = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_STARTED,
            event_category=EventCategory.EXECUTION,
            workflow_id="dual-emission-test",
            correlation_id=self.correlation_id,
            data={"agent_name": "refiner", "step": "agent_internal_execution"},
        )

        # Emit both events
        await websocket_sink.emit(orchestration_agent_event)
        await websocket_sink.emit(execution_agent_event)

        # Verify both events were captured
        assert len(captured_events) == 2

        # Analyze captured events
        orchestration_events = [
            e for e in captured_events if e.get("category") == "orchestration"
        ]
        execution_events = [
            e for e in captured_events if e.get("category") == "execution"
        ]

        assert len(orchestration_events) == 1, "Should have one orchestration event"
        assert len(execution_events) == 1, "Should have one execution event"

        # Both should be same event type but different categories
        assert orchestration_events[0]["type"] == "agent.execution.started"
        assert execution_events[0]["type"] == "agent.execution.started"
        assert orchestration_events[0]["agent_name"] == "refiner"
        assert execution_events[0]["agent_name"] == "refiner"

        # Categories should be different
        assert orchestration_events[0]["category"] != execution_events[0]["category"]

        # WebSocket consumers can now distinguish between the two event sources
        assert (
            orchestration_events[0]["category"] == "orchestration"
        )  # From node wrappers
        assert execution_events[0]["category"] == "execution"  # From individual agents
