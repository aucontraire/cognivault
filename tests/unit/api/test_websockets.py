"""
Unit tests for WebSocket connection management and event broadcasting.

Tests the WebSocketManager and WebSocketEventSink functionality in isolation
with comprehensive coverage of connection handling, event filtering, and error scenarios.
"""

import pytest
from typing import Any
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from fastapi import WebSocket

from cognivault.api.websockets import (
    WebSocketManager,
    WebSocketEventSink,
    websocket_manager,
)
from cognivault.events.types import WorkflowEvent, EventType, EventCategory
from cognivault.events.emitter import get_global_event_emitter


class TestWebSocketManager:
    """Test suite for WebSocketManager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.manager = WebSocketManager()
        self.mock_websocket1 = Mock(spec=WebSocket)
        self.mock_websocket2 = Mock(spec=WebSocket)
        self.correlation_id = "test-correlation-123"

    @pytest.mark.asyncio
    async def test_subscribe_single_client(self) -> None:
        """Test subscribing a single WebSocket client."""
        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)

        assert self.manager.get_connection_count(self.correlation_id) == 1
        assert self.manager.get_total_connections() == 1
        assert self.correlation_id in self.manager.get_active_correlation_ids()

    @pytest.mark.asyncio
    async def test_subscribe_multiple_clients_same_correlation(self) -> None:
        """Test subscribing multiple clients to the same correlation ID."""
        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.subscribe(self.correlation_id, self.mock_websocket2)

        assert self.manager.get_connection_count(self.correlation_id) == 2
        assert self.manager.get_total_connections() == 2

    @pytest.mark.asyncio
    async def test_subscribe_multiple_correlation_ids(self) -> None:
        """Test subscribing clients to different correlation IDs."""
        correlation_id2 = "test-correlation-456"

        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.subscribe(correlation_id2, self.mock_websocket2)

        assert self.manager.get_connection_count(self.correlation_id) == 1
        assert self.manager.get_connection_count(correlation_id2) == 1
        assert self.manager.get_total_connections() == 2
        assert len(self.manager.get_active_correlation_ids()) == 2

    @pytest.mark.asyncio
    @patch("cognivault.api.websockets.get_global_event_emitter")
    async def test_subscribe_registers_event_sink(self, mock_get_emitter: Mock) -> None:
        """Test that subscribing registers a WebSocketEventSink."""
        mock_emitter: Mock = Mock()
        mock_get_emitter.return_value = mock_emitter

        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)

        # Should register a WebSocketEventSink
        mock_emitter.add_sink.assert_called_once()
        sink_arg = mock_emitter.add_sink.call_args[0][0]
        assert isinstance(sink_arg, WebSocketEventSink)
        assert sink_arg.target_correlation_id == self.correlation_id

    @pytest.mark.asyncio
    @patch("cognivault.api.websockets.get_global_event_emitter")
    async def test_subscribe_avoids_duplicate_sink_registration(
        self, mock_get_emitter: Mock
    ) -> None:
        """Test that multiple subscriptions to same correlation_id don't create duplicate sinks."""
        mock_emitter: Mock = Mock()
        mock_get_emitter.return_value = mock_emitter

        # Subscribe two clients to same correlation_id
        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.subscribe(self.correlation_id, self.mock_websocket2)

        # Should only register sink once
        assert mock_emitter.add_sink.call_count == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_single_client(self) -> None:
        """Test unsubscribing a single WebSocket client."""
        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.unsubscribe(self.correlation_id, self.mock_websocket1)

        assert self.manager.get_connection_count(self.correlation_id) == 0
        assert self.manager.get_total_connections() == 0
        assert self.correlation_id not in self.manager.get_active_correlation_ids()

    @pytest.mark.asyncio
    async def test_unsubscribe_one_of_multiple_clients(self) -> None:
        """Test unsubscribing one client when multiple are connected."""
        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.subscribe(self.correlation_id, self.mock_websocket2)

        await self.manager.unsubscribe(self.correlation_id, self.mock_websocket1)

        assert self.manager.get_connection_count(self.correlation_id) == 1
        assert self.manager.get_total_connections() == 1
        assert self.correlation_id in self.manager.get_active_correlation_ids()

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_client(self) -> None:
        """Test unsubscribing a client that was never subscribed."""
        # Should not raise an exception
        await self.manager.unsubscribe("nonexistent-id", self.mock_websocket1)

        # Should handle removing non-subscribed websocket gracefully
        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.unsubscribe(self.correlation_id, self.mock_websocket2)

        assert self.manager.get_connection_count(self.correlation_id) == 1

    @pytest.mark.asyncio
    async def test_broadcast_event_successful(self) -> None:
        """Test successful event broadcasting to subscribed clients."""
        self.mock_websocket1.send_text = AsyncMock()
        self.mock_websocket2.send_text = AsyncMock()

        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.subscribe(self.correlation_id, self.mock_websocket2)

        event_data = {"type": "TEST_EVENT", "message": "test"}
        await self.manager.broadcast_event(self.correlation_id, event_data)

        expected_message = json.dumps(event_data)
        self.mock_websocket1.send_text.assert_called_once_with(expected_message)
        self.mock_websocket2.send_text.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_broadcast_event_no_connections(self) -> None:
        """Test broadcasting when no clients are connected."""
        event_data = {"type": "TEST_EVENT", "message": "test"}

        # Should not raise an exception
        await self.manager.broadcast_event("nonexistent-id", event_data)

    @pytest.mark.asyncio
    async def test_broadcast_event_handles_websocket_disconnect(self) -> None:
        """Test that broadcast handles WebSocket disconnections gracefully."""
        from fastapi import WebSocketDisconnect

        self.mock_websocket1.send_text = AsyncMock()
        self.mock_websocket2.send_text = AsyncMock(side_effect=WebSocketDisconnect())

        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.subscribe(self.correlation_id, self.mock_websocket2)

        event_data = {"type": "TEST_EVENT", "message": "test"}
        await self.manager.broadcast_event(self.correlation_id, event_data)

        # First websocket should receive the message
        self.mock_websocket1.send_text.assert_called_once()

        # Disconnected websocket should be removed from connections
        assert self.manager.get_connection_count(self.correlation_id) == 1

    @pytest.mark.asyncio
    async def test_broadcast_event_handles_general_exception(self) -> None:
        """Test that broadcast handles general exceptions during send."""
        self.mock_websocket1.send_text = AsyncMock(
            side_effect=Exception("Network error")
        )
        self.mock_websocket2.send_text = AsyncMock()

        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)
        await self.manager.subscribe(self.correlation_id, self.mock_websocket2)

        event_data = {"type": "TEST_EVENT", "message": "test"}
        await self.manager.broadcast_event(self.correlation_id, event_data)

        # Second websocket should still receive the message
        self.mock_websocket2.send_text.assert_called_once()

        # Failed websocket should be removed
        assert self.manager.get_connection_count(self.correlation_id) == 1

    @pytest.mark.asyncio
    async def test_broadcast_event_invalid_json(self) -> None:
        """Test broadcasting with data that can't be JSON serialized."""
        await self.manager.subscribe(self.correlation_id, self.mock_websocket1)

        # Create object that can't be JSON serialized
        event_data = {"function": lambda x: x}

        # Should not raise exception, but should log error
        await self.manager.broadcast_event(self.correlation_id, event_data)

    def test_get_connection_count_nonexistent_correlation(self) -> None:
        """Test getting connection count for non-existent correlation ID."""
        assert self.manager.get_connection_count("nonexistent-id") == 0

    def test_get_total_connections_empty(self) -> None:
        """Test getting total connections when none exist."""
        assert self.manager.get_total_connections() == 0

    def test_get_active_correlation_ids_empty(self) -> None:
        """Test getting active correlation IDs when none exist."""
        assert self.manager.get_active_correlation_ids() == []


class TestWebSocketEventSink:
    """Test suite for WebSocketEventSink functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.mock_manager = Mock(spec=WebSocketManager)
        self.mock_manager.broadcast_event = AsyncMock()
        self.correlation_id = "test-correlation-123"
        self.sink = WebSocketEventSink(self.mock_manager, self.correlation_id)

    def create_test_event(self, event_type: EventType, **kwargs: Any) -> WorkflowEvent:
        """Create a test WorkflowEvent with verified structure."""
        # Extract agent_name and status for data dict, don't pass as direct params
        agent_name = kwargs.pop("agent_name", "test_agent")
        status = kwargs.pop("status", "processing")

        defaults = {
            "event_type": event_type,
            "event_category": EventCategory.ORCHESTRATION,  # Default for test events
            "workflow_id": "test-workflow-123",
            "correlation_id": self.correlation_id,
            "data": {"agent_name": agent_name, "status": status},
        }
        defaults.update(kwargs)
        return WorkflowEvent(**defaults)

    @pytest.mark.asyncio
    async def test_emit_matching_correlation_id(self) -> None:
        """Test that events with matching correlation_id are broadcasted."""
        event = self.create_test_event(EventType.WORKFLOW_STARTED)

        await self.sink.emit(event)

        self.mock_manager.broadcast_event.assert_called_once()
        call_args = self.mock_manager.broadcast_event.call_args
        assert call_args[0][0] == self.correlation_id  # correlation_id argument

        # Verify event data structure
        event_data = call_args[0][1]
        assert event_data["type"] == "workflow.started"
        assert event_data["correlation_id"] == self.correlation_id
        assert "progress" in event_data
        assert "message" in event_data

    @pytest.mark.asyncio
    async def test_emit_non_matching_correlation_id(self) -> None:
        """Test that events with non-matching correlation_id are ignored."""
        event = self.create_test_event(
            EventType.WORKFLOW_STARTED, correlation_id="different-correlation-id"
        )

        await self.sink.emit(event)

        self.mock_manager.broadcast_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_handles_broadcast_exception(self) -> None:
        """Test that emit handles exceptions from broadcast_event gracefully."""
        self.mock_manager.broadcast_event.side_effect = Exception("Broadcast failed")

        event = self.create_test_event(EventType.WORKFLOW_STARTED)

        # Should not raise exception
        await self.sink.emit(event)

    def test_calculate_progress_workflow_events(self) -> None:
        """Test progress calculation for workflow-level events."""
        # Test workflow started
        event = self.create_test_event(EventType.WORKFLOW_STARTED)
        progress = self.sink._calculate_progress(event)
        assert progress == 0.0

        # Test workflow completed
        event = self.create_test_event(EventType.WORKFLOW_COMPLETED)
        progress = self.sink._calculate_progress(event)
        assert progress == 100.0

        # Test workflow failed
        event = self.create_test_event(EventType.WORKFLOW_FAILED)
        progress = self.sink._calculate_progress(event)
        assert progress == 0.0

    def test_calculate_progress_agent_events(self) -> None:
        """Test progress calculation for agent-specific events."""
        # Test agent execution started
        event = self.create_test_event(
            EventType.AGENT_EXECUTION_STARTED, agent_name="refiner"
        )
        progress = self.sink._calculate_progress(event)
        assert progress == 5.0

        # Test agent execution completed
        event = self.create_test_event(
            EventType.AGENT_EXECUTION_COMPLETED, agent_name="synthesis"
        )
        progress = self.sink._calculate_progress(event)
        assert progress == 95.0

        # Test unknown agent
        event = self.create_test_event(
            EventType.AGENT_EXECUTION_STARTED, agent_name="unknown_agent"
        )
        progress = self.sink._calculate_progress(event)
        assert progress == 10.0  # Default for unknown agents

    def test_calculate_progress_advanced_node_events(self) -> None:
        """Test progress calculation for advanced node type events."""
        # Test decision made
        event = self.create_test_event(EventType.DECISION_MADE)
        progress = self.sink._calculate_progress(event)
        assert progress == 15.0

        # Test aggregation completed
        event = self.create_test_event(EventType.AGGREGATION_COMPLETED)
        progress = self.sink._calculate_progress(event)
        assert progress == 85.0

        # Test validation completed
        event = self.create_test_event(EventType.VALIDATION_COMPLETED)
        progress = self.sink._calculate_progress(event)
        assert progress == 90.0

    def test_calculate_progress_unknown_event_type(self) -> None:
        """Test progress calculation for unknown event types."""
        # Create event with mock event type that's not in progress_map
        event = self.create_test_event(EventType.API_REQUEST_RECEIVED)
        progress = self.sink._calculate_progress(event)
        assert progress == 10.0  # Default for unknown events

    def test_get_user_friendly_message_workflow_events(self) -> None:
        """Test user-friendly message generation for workflow events."""
        event = self.create_test_event(EventType.WORKFLOW_STARTED)
        message = self.sink._get_user_friendly_message(event)
        assert message == "Workflow execution started"

        event = self.create_test_event(EventType.WORKFLOW_COMPLETED)
        message = self.sink._get_user_friendly_message(event)
        assert message == "Workflow completed successfully"

        event = self.create_test_event(EventType.WORKFLOW_FAILED)
        message = self.sink._get_user_friendly_message(event)
        assert message == "Workflow execution failed"

    def test_get_user_friendly_message_agent_events(self) -> None:
        """Test user-friendly message generation for agent events."""
        event = self.create_test_event(
            EventType.AGENT_EXECUTION_STARTED, agent_name="refiner"
        )
        message = self.sink._get_user_friendly_message(event)
        assert message == "Starting Refiner agent"

        event = self.create_test_event(
            EventType.AGENT_EXECUTION_COMPLETED, agent_name="critic"
        )
        message = self.sink._get_user_friendly_message(event)
        assert message == "Critic agent completed"

    def test_get_user_friendly_message_with_status(self) -> None:
        """Test user-friendly message generation includes status when relevant."""
        event = self.create_test_event(
            EventType.AGENT_EXECUTION_STARTED,
            agent_name="refiner",
            status="initializing",
        )
        message = self.sink._get_user_friendly_message(event)
        assert message == "Starting Refiner agent (initializing)"

    def test_get_user_friendly_message_unknown_event(self) -> None:
        """Test user-friendly message generation for unknown event types."""
        event = self.create_test_event(EventType.API_REQUEST_RECEIVED)
        message = self.sink._get_user_friendly_message(event)
        assert message == "Api Request Received"

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test that close method completes without error."""
        # Should not raise any exceptions
        await self.sink.close()

    @pytest.mark.asyncio
    async def test_event_data_metadata_handling(self) -> None:
        """Test that event metadata is properly included and filtered."""
        event = self.create_test_event(
            EventType.AGENT_EXECUTION_COMPLETED,
            metadata={
                "execution_time_ms": 1500.0,
                "memory_usage_mb": 256.0,
                "some_other_field": "ignored",
            },
        )

        # Add execution_time_ms as attribute (as it would be in real events)
        event.execution_time_ms = 1500.0
        event.memory_usage_mb = 256.0

        await self.sink.emit(event)

        call_args = self.mock_manager.broadcast_event.call_args
        event_data = call_args[0][1]

        # Should include execution_time_ms and memory_usage_mb
        assert event_data["metadata"]["execution_time_ms"] == 1500.0
        assert event_data["metadata"]["memory_usage_mb"] == 256.0

        # Should not include None values
        assert "node_count" not in event_data["metadata"]  # None values removed
        assert "error_type" not in event_data["metadata"]


class TestGlobalWebSocketManager:
    """Test suite for global websocket_manager instance."""

    def test_global_manager_instance(self) -> None:
        """Test that global websocket_manager is properly initialized."""
        assert websocket_manager is not None
        assert isinstance(websocket_manager, WebSocketManager)
        assert websocket_manager.get_total_connections() == 0
        assert websocket_manager.get_active_correlation_ids() == []

    def test_global_manager_singleton_behavior(self) -> None:
        """Test that importing websocket_manager gives same instance."""
        from cognivault.api.websockets import websocket_manager as imported_manager

        assert websocket_manager is imported_manager
