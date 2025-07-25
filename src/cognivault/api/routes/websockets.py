"""
WebSocket endpoints for real-time workflow progress streaming.

Provides WebSocket endpoints that stream live workflow execution progress
to clients based on correlation IDs with comprehensive error handling.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import re
from typing import Optional

from cognivault.api.websockets import websocket_manager
from cognivault.observability import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/query/{correlation_id}")
async def workflow_progress(websocket: WebSocket, correlation_id: str) -> None:
    """
    WebSocket endpoint for streaming real-time workflow progress updates.

    Clients connect to this endpoint with a correlation_id to receive live updates
    about workflow execution including agent progress, completion status, and errors.

    Args:
        websocket: The WebSocket connection
        correlation_id: Unique identifier for the workflow to monitor

    WebSocket Message Format:
        {
            "type": "WORKFLOW_STARTED" | "AGENT_EXECUTION_STARTED" | "AGENT_EXECUTION_COMPLETED" | "WORKFLOW_COMPLETED" | "WORKFLOW_FAILED",
            "timestamp": float,
            "correlation_id": str,
            "agent_name": str | None,
            "status": str,
            "progress": float,  # 0.0 to 100.0
            "message": str,  # Human-readable status message
            "metadata": {
                "execution_time_ms": float | None,
                "memory_usage_mb": float | None,
                "node_count": int | None,
                "error_type": str | None,
                "error_message": str | None
            }
        }

    Connection Lifecycle:
        1. Client connects to /ws/query/{correlation_id}
        2. Server validates correlation_id format
        3. WebSocket connection accepted
        4. Client subscribed to receive events for that correlation_id
        5. Real-time events streamed as workflow executes
        6. Connection cleaned up on disconnect

    Error Handling:
        - Invalid correlation_id format: Connection rejected with 1008 code
        - Network disconnection: Automatic cleanup of subscriptions
        - Broadcast errors: Logged but don't terminate connection

    Examples:
        - Connect: ws://localhost:8000/ws/query/550e8400-e29b-41d4-a716-446655440000
        - Receive: {"type": "WORKFLOW_STARTED", "progress": 0.0, "message": "Workflow execution started"}
        - Receive: {"type": "AGENT_EXECUTION_STARTED", "agent_name": "refiner", "progress": 5.0, "message": "Starting Refiner agent"}
        - Receive: {"type": "WORKFLOW_COMPLETED", "progress": 100.0, "message": "Workflow completed successfully"}
    """
    # Validate correlation_id format before accepting connection
    if not _is_valid_correlation_id(correlation_id):
        logger.warning(
            f"WebSocket connection rejected: invalid correlation_id format: {correlation_id}"
        )
        await websocket.close(code=1008, reason="Invalid correlation_id format")
        return

    try:
        # Accept the WebSocket connection
        await websocket.accept()
        logger.info(
            f"WebSocket connection accepted for correlation_id: {correlation_id}"
        )

        # Subscribe to events for this correlation_id
        await websocket_manager.subscribe(correlation_id, websocket)

        # Send initial connection confirmation
        await websocket.send_json(
            {
                "type": "CONNECTION_ESTABLISHED",
                "timestamp": 0.0,
                "correlation_id": correlation_id,
                "agent_name": None,
                "status": "connected",
                "progress": 0.0,
                "message": f"Connected to workflow progress stream for {correlation_id}",
                "metadata": {},
            }
        )

        # Keep connection alive and handle client messages
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                # We don't expect clients to send messages, but we need to handle them
                # to detect disconnections and potential future client commands
                message = await websocket.receive_text()

                # Log unexpected client messages
                logger.debug(
                    f"Received unexpected message from WebSocket client: {message}"
                )

                # Future: Could handle client commands like:
                # - {"command": "ping"} -> {"type": "pong"}
                # - {"command": "get_status"} -> current workflow status

        except WebSocketDisconnect:
            logger.info(
                f"WebSocket client disconnected from correlation_id: {correlation_id}"
            )
        except Exception as e:
            logger.error(f"Error in WebSocket message loop for {correlation_id}: {e}")

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket client disconnected during setup for correlation_id: {correlation_id}"
        )
    except Exception as e:
        logger.error(f"Error setting up WebSocket connection for {correlation_id}: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass  # Connection may already be closed
    finally:
        # Always cleanup subscription on disconnect
        try:
            await websocket_manager.unsubscribe(correlation_id, websocket)
        except Exception as e:
            logger.error(
                f"Error cleaning up WebSocket subscription for {correlation_id}: {e}"
            )


@router.websocket("/ws/health")
async def websocket_health_check(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for health checking and monitoring.

    Provides a simple WebSocket endpoint for testing connectivity and
    monitoring WebSocket service health without requiring workflow execution.

    Args:
        websocket: The WebSocket connection

    Usage:
        - Connect: ws://localhost:8000/ws/health
        - Receive: {"status": "healthy", "active_connections": 5, "active_workflows": 3}
        - Send: "ping" -> Receive: "pong"
    """
    try:
        await websocket.accept()
        logger.debug("WebSocket health check connection accepted")

        # Send health status
        health_data = {
            "status": "healthy",
            "active_connections": websocket_manager.get_total_connections(),
            "active_workflows": len(websocket_manager.get_active_correlation_ids()),
            "correlation_ids": websocket_manager.get_active_correlation_ids(),
        }

        await websocket.send_json(health_data)

        # Handle ping/pong for connection testing
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                message = await websocket.receive_text()

                if message.lower() == "ping":
                    await websocket.send_text("pong")
                elif message.lower() == "status":
                    # Refresh health data
                    health_data = {
                        "status": "healthy",
                        "active_connections": websocket_manager.get_total_connections(),
                        "active_workflows": len(
                            websocket_manager.get_active_correlation_ids()
                        ),
                        "correlation_ids": websocket_manager.get_active_correlation_ids(),
                    }
                    await websocket.send_json(health_data)
                else:
                    await websocket.send_json(
                        {
                            "error": "Unknown command",
                            "supported_commands": ["ping", "status"],
                        }
                    )

        except WebSocketDisconnect:
            logger.debug("WebSocket health check client disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket health check message loop: {e}")

    except WebSocketDisconnect:
        logger.debug("WebSocket health check client disconnected during setup")
    except Exception as e:
        logger.error(f"Error setting up WebSocket health check connection: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass  # Connection may already be closed


def _is_valid_correlation_id(correlation_id: str) -> bool:
    """
    Validate correlation_id format.

    Accepts UUID format or alphanumeric with hyphens/underscores.

    Args:
        correlation_id: The correlation ID to validate

    Returns:
        True if valid format, False otherwise
    """
    if not correlation_id or len(correlation_id) > 100:
        return False

    # Accept UUID format (most common)
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    if re.match(uuid_pattern, correlation_id.lower()):
        return True

    # Accept alphanumeric with hyphens and underscores
    alphanumeric_pattern = r"^[a-zA-Z0-9_-]+$"
    if re.match(alphanumeric_pattern, correlation_id):
        return True

    return False
