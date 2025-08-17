"""
Tests for Event Emitter convenience functions.

This module tests the convenience functions for event emission including:
- emit_health_check_performed
- emit_api_request_received
- emit_api_response_sent
- emit_service_boundary_crossed
"""

import pytest
from typing import Any
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

from cognivault.events.emitter import (
    emit_health_check_performed,
    emit_api_request_received,
    emit_api_response_sent,
    emit_service_boundary_crossed,
    get_global_event_emitter,
)
from cognivault.events.types import EventType, WorkflowEvent


class TestEmitHealthCheckPerformed:
    """Test emit_health_check_performed convenience function."""

    @pytest.mark.asyncio
    async def test_emit_health_check_performed_basic(self) -> None:
        """Test basic health check event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_health_check_performed(component_name="api", status="healthy")

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.HEALTH_CHECK_PERFORMED
            assert emitted_event.data["component_name"] == "api"
            assert emitted_event.data["status"] == "healthy"
            assert emitted_event.data["response_time_ms"] is None
            assert emitted_event.data["details"] == {}
            assert emitted_event.metadata["event_category"] == "monitoring"
            assert emitted_event.metadata["component_type"] == "system"

    @pytest.mark.asyncio
    async def test_emit_health_check_performed_with_all_params(self) -> None:
        """Test health check event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            details = {"cpu_usage": 45.2, "memory_usage": 67.8}
            metadata = {"environment": "production"}

            await emit_health_check_performed(
                component_name="orchestrator",
                status="degraded",
                response_time_ms=150.5,
                details=details,
                correlation_id="test-correlation-123",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.HEALTH_CHECK_PERFORMED
            assert emitted_event.data["component_name"] == "orchestrator"
            assert emitted_event.data["status"] == "degraded"
            assert emitted_event.data["response_time_ms"] == 150.5
            assert emitted_event.data["details"] == details
            assert emitted_event.correlation_id == "test-correlation-123"
            assert emitted_event.metadata["environment"] == "production"
            assert emitted_event.metadata["event_category"] == "monitoring"
            assert emitted_event.metadata["component_type"] == "system"

    @pytest.mark.asyncio
    async def test_emit_health_check_performed_workflow_id_generation(self) -> None:
        """Test that health check events generate unique workflow IDs."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Emit two events
            await emit_health_check_performed("api", "healthy")
            await emit_health_check_performed("orchestrator", "healthy")

            # Verify two calls were made
            assert mock_emitter.emit.call_count == 2

            # Get the workflow IDs from both events
            first_event = mock_emitter.emit.call_args_list[0][0][0]
            second_event = mock_emitter.emit.call_args_list[1][0][0]

            # Verify workflow IDs are different and follow expected pattern
            assert first_event.workflow_id != second_event.workflow_id
            assert first_event.workflow_id.startswith("health_check_")
            assert second_event.workflow_id.startswith("health_check_")
            assert len(first_event.workflow_id.split("_")[2]) == 8  # 8 character hex
            assert len(second_event.workflow_id.split("_")[2]) == 8  # 8 character hex

    @pytest.mark.asyncio
    async def test_emit_health_check_performed_with_correlation_fallback(self) -> None:
        """Test health check event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-correlation-456"

            await emit_health_check_performed(
                component_name="llm_gateway", status="unhealthy"
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-correlation-456"

    @pytest.mark.asyncio
    async def test_emit_health_check_performed_timestamp_generation(self) -> None:
        """Test that health check events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_health_check_performed("api", "healthy")
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_emit_health_check_performed_empty_details_handling(self) -> None:
        """Test health check event emission with None details."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_health_check_performed(
                component_name="api", status="healthy", details=None
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify details default to empty dict
            assert emitted_event.data["details"] == {}

    @pytest.mark.asyncio
    async def test_emit_health_check_performed_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "environment": "staging",
                "version": "1.2.3",
                "event_category": "custom_monitoring",  # This should override default
            }

            await emit_health_check_performed(
                component_name="api", status="healthy", metadata=custom_metadata
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["environment"] == "staging"
            assert emitted_event.metadata["version"] == "1.2.3"
            assert (
                emitted_event.metadata["event_category"] == "custom_monitoring"
            )  # Overridden
            assert (
                emitted_event.metadata["component_type"] == "system"
            )  # Default preserved


class TestEmitApiRequestReceived:
    """Test emit_api_request_received convenience function."""

    @pytest.mark.asyncio
    async def test_emit_api_request_received_basic(self) -> None:
        """Test basic API request received event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_api_request_received(
                workflow_id="workflow-123", endpoint="execute_workflow"
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.API_REQUEST_RECEIVED
            assert emitted_event.workflow_id == "workflow-123"
            assert emitted_event.data["endpoint"] == "execute_workflow"
            assert emitted_event.data["request_size_bytes"] is None
            assert emitted_event.data["client_info"] == {}
            assert emitted_event.data["request_data"] == {}
            assert emitted_event.metadata["event_category"] == "api"
            assert emitted_event.metadata["component_type"] == "api_gateway"

    @pytest.mark.asyncio
    async def test_emit_api_request_received_with_all_params(self) -> None:
        """Test API request received event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            client_info = {
                "user_agent": "test-client/1.0",
                "ip_address": "192.168.1.100",
            }
            request_data = {
                "query_length": 250,
                "agents_requested": ["refiner", "critic"],
            }
            metadata = {"environment": "production", "api_version": "v1"}

            await emit_api_request_received(
                workflow_id="workflow-456",
                endpoint="health_check",
                request_size_bytes=1024,
                client_info=client_info,
                request_data=request_data,
                correlation_id="test-correlation-789",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.API_REQUEST_RECEIVED
            assert emitted_event.workflow_id == "workflow-456"
            assert emitted_event.data["endpoint"] == "health_check"
            assert emitted_event.data["request_size_bytes"] == 1024
            assert emitted_event.data["client_info"] == client_info
            assert emitted_event.data["request_data"] == request_data
            assert emitted_event.correlation_id == "test-correlation-789"
            assert emitted_event.metadata["environment"] == "production"
            assert emitted_event.metadata["api_version"] == "v1"
            assert emitted_event.metadata["event_category"] == "api"
            assert emitted_event.metadata["component_type"] == "api_gateway"

    @pytest.mark.asyncio
    async def test_emit_api_request_received_with_correlation_fallback(self) -> None:
        """Test API request received event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-correlation-999"

            await emit_api_request_received(
                workflow_id="workflow-789", endpoint="get_status"
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-correlation-999"

    @pytest.mark.asyncio
    async def test_emit_api_request_received_none_handling(self) -> None:
        """Test API request received event emission with None values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_api_request_received(
                workflow_id="workflow-abc",
                endpoint="execute_workflow",
                client_info=None,
                request_data=None,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify None values default to empty dicts
            assert emitted_event.data["client_info"] == {}
            assert emitted_event.data["request_data"] == {}

    @pytest.mark.asyncio
    async def test_emit_api_request_received_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "service": "orchestration-api",
                "version": "2.1.0",
                "event_category": "custom_api",  # This should override default
            }

            await emit_api_request_received(
                workflow_id="workflow-def",
                endpoint="execute_workflow",
                metadata=custom_metadata,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["service"] == "orchestration-api"
            assert emitted_event.metadata["version"] == "2.1.0"
            assert (
                emitted_event.metadata["event_category"] == "custom_api"
            )  # Overridden
            assert (
                emitted_event.metadata["component_type"] == "api_gateway"
            )  # Default preserved

    @pytest.mark.asyncio
    async def test_emit_api_request_received_timestamp_generation(self) -> None:
        """Test that API request received events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_api_request_received("workflow-ghi", "health_check")
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc


class TestEmitApiResponseSent:
    """Test emit_api_response_sent convenience function."""

    @pytest.mark.asyncio
    async def test_emit_api_response_sent_basic(self) -> None:
        """Test basic API response sent event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_api_response_sent(workflow_id="workflow-123", status="success")

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.API_RESPONSE_SENT
            assert emitted_event.workflow_id == "workflow-123"
            assert emitted_event.data["status"] == "success"
            assert emitted_event.data["response_size_bytes"] is None
            assert emitted_event.data["execution_time_ms"] is None
            assert emitted_event.data["response_data"] == {}
            assert emitted_event.execution_time_ms is None
            assert emitted_event.metadata["event_category"] == "api"
            assert emitted_event.metadata["component_type"] == "api_gateway"

    @pytest.mark.asyncio
    async def test_emit_api_response_sent_with_all_params(self) -> None:
        """Test API response sent event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            response_data = {
                "agent_outputs": 3,
                "successful_agents": ["refiner", "critic", "synthesis"],
            }
            metadata = {"environment": "production", "api_version": "v2"}

            await emit_api_response_sent(
                workflow_id="workflow-456",
                status="error",
                response_size_bytes=2048,
                execution_time_ms=1250.75,
                response_data=response_data,
                correlation_id="test-correlation-789",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.API_RESPONSE_SENT
            assert emitted_event.workflow_id == "workflow-456"
            assert emitted_event.data["status"] == "error"
            assert emitted_event.data["response_size_bytes"] == 2048
            assert emitted_event.data["execution_time_ms"] == 1250.75
            assert emitted_event.data["response_data"] == response_data
            assert (
                emitted_event.execution_time_ms == 1250.75
            )  # Should be set on event object too
            assert emitted_event.correlation_id == "test-correlation-789"
            assert emitted_event.metadata["environment"] == "production"
            assert emitted_event.metadata["api_version"] == "v2"
            assert emitted_event.metadata["event_category"] == "api"
            assert emitted_event.metadata["component_type"] == "api_gateway"

    @pytest.mark.asyncio
    async def test_emit_api_response_sent_with_correlation_fallback(self) -> None:
        """Test API response sent event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-correlation-888"

            await emit_api_response_sent(workflow_id="workflow-789", status="timeout")

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-correlation-888"

    @pytest.mark.asyncio
    async def test_emit_api_response_sent_none_handling(self) -> None:
        """Test API response sent event emission with None values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_api_response_sent(
                workflow_id="workflow-abc", status="partial_success", response_data=None
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify None values default appropriately
            assert emitted_event.data["response_data"] == {}

    @pytest.mark.asyncio
    async def test_emit_api_response_sent_execution_time_consistency(self) -> None:
        """Test that execution_time_ms is set consistently in data and event object."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            execution_time = 987.65

            await emit_api_response_sent(
                workflow_id="workflow-def",
                status="success",
                execution_time_ms=execution_time,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify execution time is set in both places
            assert emitted_event.data["execution_time_ms"] == execution_time
            assert emitted_event.execution_time_ms == execution_time

    @pytest.mark.asyncio
    async def test_emit_api_response_sent_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "service": "orchestration-api",
                "version": "3.0.1",
                "event_category": "custom_api_response",  # This should override default
            }

            await emit_api_response_sent(
                workflow_id="workflow-ghi", status="success", metadata=custom_metadata
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["service"] == "orchestration-api"
            assert emitted_event.metadata["version"] == "3.0.1"
            assert (
                emitted_event.metadata["event_category"] == "custom_api_response"
            )  # Overridden
            assert (
                emitted_event.metadata["component_type"] == "api_gateway"
            )  # Default preserved

    @pytest.mark.asyncio
    async def test_emit_api_response_sent_timestamp_generation(self) -> None:
        """Test that API response sent events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_api_response_sent("workflow-jkl", "success")
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc


class TestEmitServiceBoundaryCrossed:
    """Test emit_service_boundary_crossed convenience function."""

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_basic(self) -> None:
        """Test basic service boundary crossed event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_service_boundary_crossed(
                workflow_id="workflow-123",
                source_service="orchestrator",
                target_service="llm_gateway",
                operation="complete",
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.SERVICE_BOUNDARY_CROSSED
            assert emitted_event.workflow_id == "workflow-123"
            assert emitted_event.data["source_service"] == "orchestrator"
            assert emitted_event.data["target_service"] == "llm_gateway"
            assert emitted_event.data["operation"] == "complete"
            assert emitted_event.data["boundary_type"] == "internal"  # Default value
            assert emitted_event.data["payload_size_bytes"] is None
            assert emitted_event.data["operation_data"] == {}
            assert emitted_event.metadata["event_category"] == "service_interaction"
            assert emitted_event.metadata["component_type"] == "boundary"

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_with_all_params(self) -> None:
        """Test service boundary crossed event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            operation_data = {
                "model": "gpt-4",
                "tokens_requested": 1000,
                "temperature": 0.7,
            }
            metadata = {"environment": "production", "service_version": "2.0.1"}

            await emit_service_boundary_crossed(
                workflow_id="workflow-456",
                source_service="api_gateway",
                target_service="external_api",
                operation="health_check",
                boundary_type="external",
                payload_size_bytes=512,
                operation_data=operation_data,
                correlation_id="test-correlation-777",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.SERVICE_BOUNDARY_CROSSED
            assert emitted_event.workflow_id == "workflow-456"
            assert emitted_event.data["source_service"] == "api_gateway"
            assert emitted_event.data["target_service"] == "external_api"
            assert emitted_event.data["operation"] == "health_check"
            assert emitted_event.data["boundary_type"] == "external"
            assert emitted_event.data["payload_size_bytes"] == 512
            assert emitted_event.data["operation_data"] == operation_data
            assert emitted_event.correlation_id == "test-correlation-777"
            assert emitted_event.metadata["environment"] == "production"
            assert emitted_event.metadata["service_version"] == "2.0.1"
            assert emitted_event.metadata["event_category"] == "service_interaction"
            assert emitted_event.metadata["component_type"] == "boundary"

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_with_correlation_fallback(
        self,
    ) -> None:
        """Test service boundary crossed event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-correlation-666"

            await emit_service_boundary_crossed(
                workflow_id="workflow-789",
                source_service="diagnostics",
                target_service="orchestrator",
                operation="profile",
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-correlation-666"

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_boundary_types(self) -> None:
        """Test service boundary crossed event emission with different boundary types."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test microservice boundary
            await emit_service_boundary_crossed(
                workflow_id="workflow-abc",
                source_service="core_service",
                target_service="llm_service",
                operation="execute",
                boundary_type="microservice",
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify boundary type
            assert emitted_event.data["boundary_type"] == "microservice"

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_none_handling(self) -> None:
        """Test service boundary crossed event emission with None values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_service_boundary_crossed(
                workflow_id="workflow-def",
                source_service="source",
                target_service="target",
                operation="test",
                operation_data=None,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify None values default appropriately
            assert emitted_event.data["operation_data"] == {}

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "trace_id": "trace-12345",
                "deployment": "blue_green",
                "event_category": "custom_service_interaction",  # This should override default
            }

            await emit_service_boundary_crossed(
                workflow_id="workflow-ghi",
                source_service="api",
                target_service="database",
                operation="query",
                metadata=custom_metadata,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["trace_id"] == "trace-12345"
            assert emitted_event.metadata["deployment"] == "blue_green"
            assert (
                emitted_event.metadata["event_category"] == "custom_service_interaction"
            )  # Overridden
            assert (
                emitted_event.metadata["component_type"] == "boundary"
            )  # Default preserved

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_timestamp_generation(self) -> None:
        """Test that service boundary crossed events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_service_boundary_crossed(
                "workflow-jkl", "service1", "service2", "test"
            )
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_emit_service_boundary_crossed_comprehensive_data(self) -> None:
        """Test service boundary crossed event with comprehensive data for service architecture."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Simulate comprehensive service boundary tracking
            operation_data = {
                "request_type": "POST",
                "endpoint": "/api/v2/complete",
                "timeout_ms": 30000,
                "retry_count": 0,
                "circuit_breaker_state": "closed",
            }

            await emit_service_boundary_crossed(
                workflow_id="workflow-production-001",
                source_service="orchestrator_v2",
                target_service="llm_gateway_v3",
                operation="completion_request",
                boundary_type="internal",
                payload_size_bytes=2048,
                operation_data=operation_data,
                metadata={
                    "deployment_zone": "us-west-2",
                    "service_mesh": "istio",
                    "security_context": "authenticated",
                },
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify comprehensive data capture
            assert emitted_event.data["source_service"] == "orchestrator_v2"
            assert emitted_event.data["target_service"] == "llm_gateway_v3"
            assert emitted_event.data["operation"] == "completion_request"
            assert emitted_event.data["payload_size_bytes"] == 2048
            assert emitted_event.data["operation_data"]["request_type"] == "POST"
            assert (
                emitted_event.data["operation_data"]["circuit_breaker_state"]
                == "closed"
            )
            assert emitted_event.metadata["deployment_zone"] == "us-west-2"
            assert emitted_event.metadata["service_mesh"] == "istio"
