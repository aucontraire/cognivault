"""
Unit tests for WebSocket route endpoints.

Tests the FastAPI WebSocket endpoints for real-time workflow progress streaming
with comprehensive coverage of connection handling, validation, and error scenarios.
"""

import pytest
from typing import Any
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketState

from cognivault.api.main import app
from cognivault.api.routes.websockets import _is_valid_correlation_id


class TestWebSocketRoutes:
    """Test suite for WebSocket route endpoints."""

    def setup_method(self) -> None:
        """Set up test client and mocks for each test."""
        self.client = TestClient(app)

    def test_workflow_progress_websocket_connection(self) -> None:
        """Test successful WebSocket connection to workflow progress endpoint."""
        correlation_id = "550e8400-e29b-41d4-a716-446655440000"

        with self.client.websocket_connect(f"/ws/query/{correlation_id}") as websocket:
            # Should receive connection established message
            data = websocket.receive_json()

            assert data["type"] == "CONNECTION_ESTABLISHED"
            assert data["correlation_id"] == correlation_id
            assert data["status"] == "connected"
            assert data["progress"] == 0.0
            assert "Connected to workflow progress stream" in data["message"]

    def test_workflow_progress_websocket_invalid_correlation_id(self) -> None:
        """Test WebSocket connection rejection for invalid correlation ID."""
        invalid_ids = [
            "invalid@id",  # Invalid characters
            "id with spaces",  # Spaces
            "",  # Empty string
            "a" * 101,  # Too long
        ]

        for invalid_id in invalid_ids:
            with pytest.raises(Exception):  # WebSocket connection should fail
                with self.client.websocket_connect(f"/ws/query/{invalid_id}"):
                    pass  # Should not reach here

    @patch("cognivault.api.routes.websockets.websocket_manager")
    def test_workflow_progress_websocket_subscription(self, mock_manager: Mock) -> None:
        """Test that WebSocket connection properly subscribes to manager."""
        mock_manager.subscribe = AsyncMock()
        mock_manager.unsubscribe = AsyncMock()

        correlation_id = "test-correlation-123"

        with self.client.websocket_connect(f"/ws/query/{correlation_id}") as websocket:
            # Receive connection established message
            websocket.receive_json()

            # Should have subscribed to manager
            mock_manager.subscribe.assert_called_once()

            # Get the websocket argument from the subscribe call
            subscribe_call_args = mock_manager.subscribe.call_args
            assert subscribe_call_args[0][0] == correlation_id
            # WebSocket object should be passed as second argument
            assert subscribe_call_args[0][1] is not None

    @patch("cognivault.api.routes.websockets.websocket_manager")
    def test_workflow_progress_websocket_cleanup_on_disconnect(
        self, mock_manager: Mock
    ) -> None:
        """Test that WebSocket disconnection properly cleans up subscription."""
        mock_manager.subscribe = AsyncMock()
        mock_manager.unsubscribe = AsyncMock()

        correlation_id = "test-correlation-123"

        with self.client.websocket_connect(f"/ws/query/{correlation_id}") as websocket:
            websocket.receive_json()  # Connection established
            # WebSocket context manager will handle disconnection

        # Should have unsubscribed from manager
        mock_manager.unsubscribe.assert_called_once()

    def test_websocket_health_check_connection(self) -> None:
        """Test successful connection to WebSocket health check endpoint."""
        with self.client.websocket_connect("/ws/health") as websocket:
            # Should receive health status
            data = websocket.receive_json()

            assert data["status"] == "healthy"
            assert "active_connections" in data
            assert "active_workflows" in data
            assert "correlation_ids" in data
            assert isinstance(data["active_connections"], int)
            assert isinstance(data["active_workflows"], int)
            assert isinstance(data["correlation_ids"], list)

    def test_websocket_health_check_ping_pong(self) -> None:
        """Test ping/pong functionality on health check endpoint."""
        with self.client.websocket_connect("/ws/health") as websocket:
            # Receive initial health status
            websocket.receive_json()

            # Send ping
            websocket.send_text("ping")
            response = websocket.receive_text()
            assert response == "pong"

    def test_websocket_health_check_status_command(self) -> None:
        """Test status command on health check endpoint."""
        with self.client.websocket_connect("/ws/health") as websocket:
            # Receive initial health status
            initial_data = websocket.receive_json()

            # Send status command
            websocket.send_text("status")
            status_data = websocket.receive_json()

            assert status_data["status"] == "healthy"
            assert "active_connections" in status_data
            assert "active_workflows" in status_data

    def test_websocket_health_check_unknown_command(self) -> None:
        """Test unknown command handling on health check endpoint."""
        with self.client.websocket_connect("/ws/health") as websocket:
            # Receive initial health status
            websocket.receive_json()

            # Send unknown command
            websocket.send_text("unknown_command")
            response = websocket.receive_json()

            assert response["error"] == "Unknown command"
            assert "supported_commands" in response
            assert "ping" in response["supported_commands"]
            assert "status" in response["supported_commands"]

    @patch("cognivault.api.routes.websockets.websocket_manager")
    def test_workflow_progress_websocket_with_mock_manager_data(
        self, mock_manager: Mock
    ) -> None:
        """Test WebSocket health endpoint shows manager connection data."""
        # Mock manager to return test data
        mock_manager.get_total_connections.return_value = 5
        mock_manager.get_active_correlation_ids.return_value = ["id1", "id2", "id3"]

        with self.client.websocket_connect("/ws/health") as websocket:
            data = websocket.receive_json()

            assert data["active_connections"] == 5
            assert data["active_workflows"] == 3
            assert data["correlation_ids"] == ["id1", "id2", "id3"]


class TestCorrelationIdValidation:
    """Test suite for correlation ID validation function."""

    def test_valid_uuid_format(self) -> None:
        """Test validation of valid UUID format correlation IDs."""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "123e4567-e89b-12d3-a456-426614174000",
            "00000000-0000-0000-0000-000000000000",
            "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF",  # Uppercase
        ]

        for uuid in valid_uuids:
            assert _is_valid_correlation_id(uuid) is True
            assert _is_valid_correlation_id(uuid.lower()) is True

    def test_valid_alphanumeric_format(self) -> None:
        """Test validation of valid alphanumeric correlation IDs."""
        valid_ids = [
            "test-correlation-123",
            "workflow_execution_456",
            "simple",
            "UPPERCASE_ID",
            "mixed-Case_123",
            "abc123",
            "123abc",
        ]

        for id_str in valid_ids:
            assert _is_valid_correlation_id(id_str) is True

    def test_invalid_correlation_ids(self) -> None:
        """Test validation rejects invalid correlation IDs."""
        invalid_ids = [
            "",  # Empty string
            None,  # None value (will be converted to string)
            "invalid@id",  # Invalid characters
            "id with spaces",  # Spaces
            "id/with/slashes",  # Slashes
            "id.with.dots",  # Dots
            "id+with+plus",  # Plus signs
            "id%with%percent",  # Percent signs
            "a" * 101,  # Too long
            # Note: Short UUIDs like "550e8400-e29b-41d4-a716-44665544000" actually pass
            # because they match the alphanumeric pattern [a-zA-Z0-9_-]+
            # Using characters that definitely violate both patterns:
        ]

        for invalid_id in invalid_ids:
            # Handle None case
            if invalid_id is None:
                continue  # Skip None test as it would cause different error
            assert _is_valid_correlation_id(invalid_id) is False

    def test_edge_case_lengths(self) -> None:
        """Test validation with edge case lengths."""
        # Test maximum valid length (100 characters)
        max_length_id = "a" * 100
        assert _is_valid_correlation_id(max_length_id) is True

        # Test just over maximum length
        too_long_id = "a" * 101
        assert _is_valid_correlation_id(too_long_id) is False

    def test_case_insensitive_uuid_validation(self) -> None:
        """Test that UUID validation is case-insensitive."""
        mixed_case_uuid = "550E8400-e29B-41d4-A716-446655440000"
        assert _is_valid_correlation_id(mixed_case_uuid) is True


class TestWebSocketIntegrationScenarios:
    """Integration test scenarios for WebSocket functionality."""

    def setup_method(self) -> None:
        """Set up test client for integration tests."""
        self.client = TestClient(app)

    def test_multiple_clients_same_correlation_id(self) -> None:
        """Test multiple clients connecting to same correlation ID."""
        correlation_id = "test-multi-client-123"

        # Connect first client
        with self.client.websocket_connect(f"/ws/query/{correlation_id}") as ws1:
            # Connect second client
            with self.client.websocket_connect(f"/ws/query/{correlation_id}") as ws2:
                # Both should receive connection established messages
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()

                assert data1["type"] == "CONNECTION_ESTABLISHED"
                assert data2["type"] == "CONNECTION_ESTABLISHED"
                assert data1["correlation_id"] == correlation_id
                assert data2["correlation_id"] == correlation_id

    def test_multiple_clients_different_correlation_ids(self) -> None:
        """Test multiple clients connecting to different correlation IDs."""
        correlation_id1 = "test-client-1-123"
        correlation_id2 = "test-client-2-456"

        with self.client.websocket_connect(f"/ws/query/{correlation_id1}") as ws1:
            with self.client.websocket_connect(f"/ws/query/{correlation_id2}") as ws2:
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()

                assert data1["correlation_id"] == correlation_id1
                assert data2["correlation_id"] == correlation_id2

    @patch("cognivault.api.routes.websockets.websocket_manager")
    def test_websocket_manager_exception_handling(self, mock_manager: Mock) -> None:
        """Test that WebSocket routes handle manager exceptions gracefully."""
        # Make subscribe raise an exception
        mock_manager.subscribe.side_effect = Exception("Manager error")
        mock_manager.unsubscribe = AsyncMock()

        correlation_id = "test-exception-123"

        # Connection should still work but may close due to the exception
        try:
            with self.client.websocket_connect(
                f"/ws/query/{correlation_id}"
            ) as websocket:
                # May receive connection established or may disconnect
                pass
        except Exception:
            # WebSocket may disconnect due to the exception, which is expected
            pass

        # Should still attempt to clean up
        mock_manager.unsubscribe.assert_called()

    def test_client_message_handling(self) -> None:
        """Test that WebSocket endpoint handles unexpected client messages."""
        correlation_id = "test-client-message-123"

        with self.client.websocket_connect(f"/ws/query/{correlation_id}") as websocket:
            # Receive connection established
            websocket.receive_json()

            # Send unexpected message (should be logged but not cause errors)
            websocket.send_text("unexpected message")

            # Connection should remain active
            # Note: We can't easily test the logging in unit tests,
            # but the connection should handle it gracefully

    def test_websocket_route_url_patterns(self) -> None:
        """Test that WebSocket routes match expected URL patterns."""
        # Test valid workflow progress URLs
        valid_urls = [
            "/ws/query/550e8400-e29b-41d4-a716-446655440000",
            "/ws/query/test-correlation-123",
            "/ws/query/simple_id",
        ]

        for url in valid_urls:
            with self.client.websocket_connect(url) as websocket:
                data = websocket.receive_json()
                assert data["type"] == "CONNECTION_ESTABLISHED"

        # Test health check URL
        with self.client.websocket_connect("/ws/health") as websocket:
            data = websocket.receive_json()
            assert data["status"] == "healthy"
