"""
WebSocket connection management and broadcasting for real-time workflow progress.

Provides WebSocketManager for handling client connections and broadcasting
workflow events to subscribed clients based on correlation IDs.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Any, Union
from fastapi import WebSocket, WebSocketDisconnect
import json

from cognivault.events.types import WorkflowEvent, EventType
from cognivault.events.sinks import EventSink
from cognivault.events.emitter import get_global_event_emitter
from cognivault.observability import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events to subscribed clients."""

    def __init__(self) -> None:
        # Map correlation_id -> list of connected WebSocket clients
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)
        # Track which sinks have been registered to avoid duplicates
        self._registered_sinks: Set[str] = set()
        # Lock for thread-safe connection management
        self._connection_lock = asyncio.Lock()

    async def subscribe(self, correlation_id: str, websocket: WebSocket) -> None:
        """
        Subscribe a WebSocket client to receive events for a specific correlation ID.

        Args:
            correlation_id: The workflow correlation ID to subscribe to
            websocket: The WebSocket connection to add
        """
        async with self._connection_lock:
            self._connections[correlation_id].append(websocket)
            logger.info(
                f"WebSocket client subscribed to correlation_id: {correlation_id}. "
                f"Total connections for this workflow: {len(self._connections[correlation_id])}"
            )

            # Register WebSocketEventSink if not already registered for this correlation_id
            if correlation_id not in self._registered_sinks:
                sink = WebSocketEventSink(self, correlation_id)
                get_global_event_emitter().add_sink(sink)
                self._registered_sinks.add(correlation_id)
                logger.debug(
                    f"Registered WebSocketEventSink for correlation_id: {correlation_id}"
                )

    async def unsubscribe(self, correlation_id: str, websocket: WebSocket) -> None:
        """
        Unsubscribe a WebSocket client from receiving events.

        Args:
            correlation_id: The workflow correlation ID to unsubscribe from
            websocket: The WebSocket connection to remove
        """
        async with self._connection_lock:
            if correlation_id in self._connections:
                try:
                    self._connections[correlation_id].remove(websocket)
                    logger.info(
                        f"WebSocket client unsubscribed from correlation_id: {correlation_id}. "
                        f"Remaining connections: {len(self._connections[correlation_id])}"
                    )

                    # Clean up empty connection lists
                    if not self._connections[correlation_id]:
                        del self._connections[correlation_id]
                        # Note: We keep the EventSink registered as other workflows
                        # with the same correlation_id might connect later
                        logger.debug(
                            f"Removed empty connection list for correlation_id: {correlation_id}"
                        )

                except ValueError:
                    logger.warning(
                        f"Attempted to remove WebSocket that wasn't in connections for {correlation_id}"
                    )

    async def broadcast_event(
        self, correlation_id: str, event_data: Dict[str, Any]
    ) -> None:
        """
        Broadcast an event to all WebSocket clients subscribed to a correlation ID.

        Args:
            correlation_id: The workflow correlation ID to broadcast to
            event_data: The event data to send (must be JSON serializable)
        """
        logger.debug(f"Broadcasting to {correlation_id}")

        if correlation_id not in self._connections:
            logger.debug(
                f"No WebSocket connections found for correlation_id: {correlation_id}"
            )
            return

        # Create a copy of connections to avoid modification during iteration
        try:
            async with self._connection_lock:
                connections = self._connections[correlation_id].copy()
        except Exception as e:
            logger.error(f"Error acquiring lock or copying connections: {e}")
            return

        if not connections:
            logger.debug("No connections available for broadcast")
            return

        # Prepare JSON message
        try:
            message = json.dumps(event_data)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize event data for WebSocket broadcast: {e}")
            return

        # Broadcast to all connections, handling disconnections
        disconnected_clients: List[WebSocket] = []
        successful_broadcasts = 0

        for websocket in connections:
            try:
                await websocket.send_text(message)
                successful_broadcasts += 1
            except WebSocketDisconnect:
                logger.debug(
                    f"WebSocket client disconnected during broadcast for {correlation_id}"
                )
                disconnected_clients.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket client: {e}")
                disconnected_clients.append(websocket)

        # Clean up disconnected clients
        if disconnected_clients:
            async with self._connection_lock:
                for websocket in disconnected_clients:
                    try:
                        self._connections[correlation_id].remove(websocket)
                    except ValueError:
                        pass  # Already removed

        logger.debug(
            f"Broadcasted event to {successful_broadcasts}/{len(connections)} "
            f"WebSocket clients for correlation_id: {correlation_id}"
        )

    def get_connection_count(self, correlation_id: str) -> int:
        """
        Get the number of active connections for a correlation ID.

        Args:
            correlation_id: The workflow correlation ID

        Returns:
            Number of active WebSocket connections
        """
        return len(self._connections.get(correlation_id, []))

    def get_total_connections(self) -> int:
        """
        Get the total number of active WebSocket connections across all correlation IDs.

        Returns:
            Total number of active WebSocket connections
        """
        return sum(len(connections) for connections in self._connections.values())

    def get_active_correlation_ids(self) -> List[str]:
        """
        Get list of correlation IDs that have active WebSocket connections.

        Returns:
            List of correlation IDs with active connections
        """
        return list(self._connections.keys())


class WebSocketEventSink(EventSink):
    """Event sink that broadcasts workflow events to WebSocket clients."""

    def __init__(self, manager: WebSocketManager, target_correlation_id: str) -> None:
        """
        Initialize WebSocket event sink.

        Args:
            manager: The WebSocketManager to broadcast through
            target_correlation_id: Only broadcast events with this correlation ID
        """
        self.manager = manager
        self.target_correlation_id = target_correlation_id

    async def emit(self, event: WorkflowEvent) -> None:
        """
        Emit a workflow event to WebSocket clients if correlation ID matches.

        Args:
            event: The workflow event to potentially broadcast
        """
        # Enhanced debugging for agent events
        agent_name = getattr(event, "agent_name", None)
        logger.debug(
            f"WebSocket sink received event: {event.event_type.value} "
            f"for correlation_id={event.correlation_id}, agent_name={agent_name}, "
            f"target_correlation_id={self.target_correlation_id}"
        )

        # Only broadcast events for our target correlation ID
        if event.correlation_id != self.target_correlation_id:
            return

        # Calculate progress based on event type and agent
        progress = self._calculate_progress(event)

        # Prepare event data for WebSocket broadcast
        event_data = {
            "type": event.event_type.value,
            "category": (
                event_category.value
                if (event_category := getattr(event, "event_category", None))
                is not None
                and hasattr(event_category, "value")
                else "unknown"
            ),
            "timestamp": (
                event.timestamp.isoformat()
                if hasattr(event.timestamp, "isoformat")
                else event.timestamp
            ),
            "correlation_id": event.correlation_id,
            "agent_name": getattr(event, "agent_name", None)
            or event.data.get("agent_name"),
            "status": self._derive_event_status(event),
            "progress": progress,
            "message": self._get_user_friendly_message(event),
            "metadata": {
                "execution_time_ms": getattr(event, "execution_time_ms", None)
                or event.data.get("execution_time_ms"),
                "memory_usage_mb": getattr(event, "memory_usage_mb", None)
                or event.data.get("memory_usage_mb"),
                "node_count": event.data.get("node_count"),
                "error_type": getattr(event, "error_type", None)
                or event.data.get("error_type"),
                "error_message": getattr(event, "error_message", None)
                or event.data.get("error_message"),
                # Add additional structured metadata from agent events
                "success": getattr(event, "success", None),
                "agent_metadata": (
                    getattr(event, "agent_metadata").to_dict()
                    if getattr(event, "agent_metadata", None) is not None
                    else None
                ),
                "input_context": getattr(event, "input_context", None),
                "output_context": getattr(event, "output_context", None),
                # Add workflow-specific metadata
                "workflow_id": getattr(event, "workflow_id", None),
                "query": getattr(event, "query", None) or event.data.get("query"),
                "agents_requested": getattr(event, "agents_requested", None)
                or event.data.get("agents_requested"),
                "orchestrator_type": event.data.get("orchestrator_type"),
            },
        }

        # Remove None values from metadata
        metadata_dict = event_data["metadata"]
        if isinstance(metadata_dict, dict):
            event_data["metadata"] = {
                k: v for k, v in metadata_dict.items() if v is not None
            }

        try:
            await self.manager.broadcast_event(self.target_correlation_id, event_data)
        except Exception as e:
            logger.error(
                f"Failed to broadcast event via WebSocket for correlation_id "
                f"{self.target_correlation_id}: {e}"
            )

    async def close(self) -> None:
        """Close the event sink and cleanup resources."""
        logger.debug(
            f"Closing WebSocketEventSink for correlation_id: {self.target_correlation_id}"
        )
        # No cleanup needed - WebSocketManager handles connection lifecycle

    def _calculate_progress(self, event: WorkflowEvent) -> float:
        """
        Calculate workflow progress percentage based on event type and agent.

        Args:
            event: The workflow event

        Returns:
            Progress percentage (0.0 to 100.0)
        """
        # Progress mapping based on typical 4-agent workflow
        # Use actual event type values (e.g., "workflow.started")
        progress_map: Dict[str, Union[float, Dict[str, float]]] = {
            "workflow.started": 0.0,
            "agent.execution.started": {
                "refiner": 5.0,
                "historian": 30.0,
                "critic": 55.0,
                "synthesis": 80.0,
            },
            "agent.execution.completed": {
                "refiner": 25.0,
                "historian": 50.0,
                "critic": 75.0,
                "synthesis": 95.0,
            },
            "node.decision.made": 15.0,
            "node.aggregation.completed": 85.0,
            "node.validation.completed": 90.0,
            "workflow.completed": 100.0,
            "workflow.failed": 0.0,  # Keep current progress on failure
        }

        event_type = event.event_type.value
        agent_name = getattr(event, "agent_name", None) or event.data.get(
            "agent_name", "unknown"
        )

        if event_type in progress_map:
            progress_value = progress_map[event_type]

            # Handle agent-specific progress
            if isinstance(progress_value, dict):
                return float(progress_value.get(agent_name.lower(), 10.0))

            return float(progress_value)

        # Default progress for unknown event types
        return 10.0

    def _get_user_friendly_message(self, event: WorkflowEvent) -> str:
        """
        Generate user-friendly message from workflow event.

        Args:
            event: The workflow event

        Returns:
            Human-readable message describing the event
        """
        event_type = event.event_type.value
        agent_name = getattr(event, "agent_name", None) or event.data.get(
            "agent_name", "system"
        )
        status = getattr(event, "status", None) or event.data.get(
            "status", "processing"
        )

        message_templates = {
            "workflow.started": "Workflow execution started",
            "workflow.completed": "Workflow completed successfully",
            "workflow.failed": "Workflow execution failed",
            "agent.execution.started": f"Starting {agent_name.title()} agent",
            "agent.execution.completed": f"{agent_name.title()} agent completed",
            "node.decision.made": f"Decision made by {agent_name}",
            "node.aggregation.completed": f"Results aggregated by {agent_name}",
            "node.validation.completed": f"Validation completed by {agent_name}",
            "node.termination.triggered": f"Early termination triggered by {agent_name}",
        }

        base_message = message_templates.get(
            event_type, f"{event_type.replace('.', ' ').replace('_', ' ').title()}"
        )

        # Add status information if available and relevant
        if (
            status
            and status != "processing"
            and event_type not in ["workflow.started", "workflow.completed"]
        ):
            base_message += f" ({status})"

        return base_message

    def _derive_event_status(self, event: WorkflowEvent) -> str:
        """
        Derive meaningful status from event type and context.

        Args:
            event: The workflow event

        Returns:
            Meaningful status string based on event context
        """
        # Check for explicit status first
        explicit_status = getattr(event, "status", None) or event.data.get("status")
        if explicit_status:
            return str(explicit_status)

        # Derive status based on event type
        event_type = event.event_type.value

        # Agent execution events
        if event_type == "agent.execution.started":
            return "running"
        elif event_type == "agent.execution.completed":
            success = getattr(event, "success", None) or event.data.get("success")
            if success is not None:
                return "completed" if success else "failed"
            # Fallback: check for error indicators
            if getattr(event, "error_message", None) or event.data.get("error_message"):
                return "failed"
            return "completed"
        elif event_type == "agent.execution.failed":
            return "failed"

        # Workflow events
        elif event_type == "workflow.started":
            return "starting"
        elif event_type == "workflow.completed":
            return "completed"
        elif event_type == "workflow.failed":
            return "failed"
        elif event_type == "workflow.cancelled":
            return "cancelled"

        # Routing and decision events
        elif event_type == "routing.decision.made":
            return "decided"
        elif event_type.startswith("node."):
            if "started" in event_type:
                return "running"
            elif "completed" in event_type:
                return "completed"
            elif "made" in event_type:
                return "decided"

        # Default status for unknown events
        return "processing"


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
