"""
Query execution endpoints for CogniVault API.

Provides endpoints for executing multi-agent workflows using the existing
orchestration infrastructure.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, List

from cognivault.api.models import (
    WorkflowRequest,
    WorkflowResponse,
    WorkflowHistoryResponse,
    WorkflowHistoryItem,
    StatusResponse,
)
from cognivault.api.factory import get_orchestration_api
from cognivault.observability import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/query", response_model=WorkflowResponse)
async def execute_query(request: WorkflowRequest) -> WorkflowResponse:
    """
    Execute a multi-agent workflow using existing orchestration.

    Args:
        request: Workflow execution request with query and optional configuration

    Returns:
        WorkflowResponse with execution results and metadata

    Raises:
        HTTPException: If workflow execution fails
    """
    try:
        logger.info(f"Executing query: {request.query[:100]}...")

        # Use existing factory pattern and business logic
        orchestration_api = get_orchestration_api()
        response = await orchestration_api.execute_workflow(request)

        logger.info(
            f"Query executed successfully, correlation_id: {response.correlation_id}"
        )
        return response

    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Workflow execution failed",
                "message": str(e),
                "type": type(e).__name__,
            },
        )


@router.get("/query/status/{correlation_id}", response_model=StatusResponse)
async def get_query_status(correlation_id: str) -> StatusResponse:
    """
    Get the status of a previously submitted query.

    Args:
        correlation_id: Unique identifier for the query execution

    Returns:
        StatusResponse with current workflow status and progress information

    Raises:
        HTTPException: If correlation_id is not found or API unavailable
    """
    try:
        logger.info(f"Getting status for correlation_id: {correlation_id}")

        # Get orchestration API instance
        orchestration_api = get_orchestration_api()

        # Get status using correlation_id to workflow_id mapping
        status_response = await orchestration_api.get_status_by_correlation_id(
            correlation_id
        )

        logger.info(
            f"Status retrieved for correlation_id {correlation_id}: "
            f"workflow_id={status_response.workflow_id}, status={status_response.status}"
        )

        return status_response

    except KeyError as e:
        logger.warning(f"Correlation ID not found: {correlation_id}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Correlation ID not found",
                "message": str(e),
                "correlation_id": correlation_id,
            },
        )
    except Exception as e:
        logger.error(f"Failed to get status for correlation_id {correlation_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve workflow status",
                "message": str(e),
                "type": type(e).__name__,
            },
        )


@router.get("/query/history", response_model=WorkflowHistoryResponse)
async def get_query_history(
    limit: int = Query(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
) -> WorkflowHistoryResponse:
    """
    Get recent query execution history.

    Retrieves workflow execution history from the orchestration API with pagination support.
    History includes workflow status, execution time, and query details.

    Args:
        limit: Maximum number of results to return (1-100, default: 10)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        WorkflowHistoryResponse with paginated workflow history

    Raises:
        HTTPException: If the orchestration API is unavailable or fails
    """
    try:
        logger.info(f"Fetching workflow history: limit={limit}, offset={offset}")

        # Get orchestration API instance
        orchestration_api = get_orchestration_api()

        # Get workflow history from orchestration API
        # Note: Current implementation returns in-memory active workflows
        # In production, this would be from persistent storage
        raw_history: List[Dict[str, Any]] = orchestration_api.get_workflow_history(
            limit=limit + offset  # Get more to handle offset
        )

        logger.debug(f"Raw history retrieved: {len(raw_history)} workflows")

        # Apply offset pagination
        paginated_history = raw_history[offset : offset + limit]

        # Convert raw history to typed models
        workflow_items: List[WorkflowHistoryItem] = []
        for workflow_data in paginated_history:
            try:
                # Convert raw workflow data to typed model
                workflow_item = WorkflowHistoryItem(
                    workflow_id=workflow_data["workflow_id"],
                    status=workflow_data["status"],
                    query=workflow_data[
                        "query"
                    ],  # Already truncated in orchestration API
                    start_time=workflow_data["start_time"],
                    execution_time_seconds=workflow_data["execution_time"],
                )
                workflow_items.append(workflow_item)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid workflow data: {e}")
                continue

        # Calculate pagination metadata
        total_workflows = len(raw_history)  # In production, this would be a count query
        has_more = (offset + len(workflow_items)) < total_workflows

        response = WorkflowHistoryResponse(
            workflows=workflow_items,
            total=total_workflows,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )

        logger.info(
            f"Workflow history retrieved: {len(workflow_items)} items, "
            f"total={total_workflows}, has_more={has_more}"
        )

        return response

    except Exception as e:
        logger.error(f"Failed to retrieve workflow history: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve workflow history",
                "message": str(e),
                "type": type(e).__name__,
            },
        )
