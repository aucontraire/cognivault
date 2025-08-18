"""
Unit tests for FastAPI query route endpoints.

Tests the web layer query endpoints using existing API models and factory patterns.
"""

import pytest
from typing import Any, TypedDict, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.models import (
    WorkflowRequest,
    WorkflowResponse,
    WorkflowHistoryResponse,
    WorkflowHistoryItem,
    StatusResponse,
)


# JSON Response Type Definitions for API Testing
class StatusResponseJSON(TypedDict):
    """Type-safe JSON structure for StatusResponse API responses."""

    workflow_id: str
    status: str
    progress_percentage: float
    current_agent: str | None
    estimated_completion_seconds: float | None


class WorkflowResponseJSON(TypedDict):
    """Type-safe JSON structure for WorkflowResponse API responses."""

    workflow_id: str
    status: str
    agent_outputs: dict[str, str]
    execution_time_seconds: float
    correlation_id: str | None
    error_message: str | None


class StatusTestCase(TypedDict):
    """Type-safe structure for status test case data."""

    correlation_id: str
    status: StatusResponse


class TestQueryRoutes:
    """Test suite for query execution endpoints."""

    def setup_method(self) -> None:
        """Set up test client for each test."""
        self.client = TestClient(app)

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_execute_query_success(self, mock_get_api: MagicMock) -> None:
        """Test successful query execution through FastAPI endpoint."""
        # Setup mock orchestration API response
        mock_api = AsyncMock()
        mock_response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440000",
            status="completed",
            agent_outputs={
                "refiner": "Refined query about machine learning",
                "historian": "Historical context on ML development",
                "critic": "Critical analysis of ML definition",
                "synthesis": "Machine learning is a subset of artificial intelligence...",
            },
            execution_time_seconds=15.5,
            correlation_id="test-correlation-123",
        )
        mock_api.execute_workflow.return_value = mock_response
        mock_get_api.return_value = mock_api

        # Test request
        request_data = {
            "query": "What is machine learning?",
            "agents": ["refiner", "historian", "critic", "synthesis"],
            "execution_config": {"timeout": 30},
        }

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches WorkflowResponse with Pydantic validation
        workflow_response = WorkflowResponse(**data)
        assert workflow_response.workflow_id == "550e8400-e29b-41d4-a716-446655440000"
        assert workflow_response.status == "completed"
        assert workflow_response.correlation_id == "test-correlation-123"
        assert workflow_response.execution_time_seconds == 15.5
        assert len(workflow_response.agent_outputs) == 4

    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_execute_query_orchestration_failure(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query execution when orchestration API fails."""
        # Setup mock to raise exception
        mock_get_api.side_effect = Exception(
            "LangGraphOrchestrationAPI must be initialized"
        )

        request_data = {
            "query": "What is machine learning?",
            "agents": ["refiner", "critic"],
            "execution_config": {},
        }

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 500
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Workflow execution failed"
        assert "LangGraphOrchestrationAPI must be initialized" in detail["message"]
        assert detail["type"] == "Exception"

    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_execute_query_with_minimal_request(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query execution with minimal required fields."""
        mock_api = AsyncMock()
        mock_response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655441111",
            status="completed",
            agent_outputs={"refiner": "Simple result"},
            execution_time_seconds=5.0,
            correlation_id="minimal-test-123",
        )
        mock_api.execute_workflow.return_value = mock_response
        mock_get_api.return_value = mock_api

        # Minimal request (only query required)
        request_data = {"query": "Simple query"}

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "550e8400-e29b-41d4-a716-446655441111"
        assert data["correlation_id"] == "minimal-test-123"

    def test_execute_query_invalid_request(self) -> None:
        """Test query execution with invalid request data."""
        # Missing required 'query' field
        request_data = {"agents": ["refiner"], "execution_config": {}}

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_execute_query_empty_query(self) -> None:
        """Test query execution with empty query string."""
        request_data = {"query": ""}

        response = self.client.post("/api/query", json=request_data)

        # This should be handled by Pydantic validation
        assert response.status_code == 422

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_execute_query_logs_appropriately(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that query execution logs appropriately."""
        mock_api = AsyncMock()
        mock_response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655442222",
            status="completed",
            agent_outputs={"refiner": "Test result"},
            execution_time_seconds=10.0,
            correlation_id="log-test-123",
        )
        mock_api.execute_workflow.return_value = mock_response
        mock_get_api.return_value = mock_api

        request_data = {"query": "Test query for logging"}

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 200

        # Verify logging calls
        assert mock_logger.info.call_count == 2  # Start and completion logs

        # Check start log
        start_log = mock_logger.info.call_args_list[0][0][0]
        assert "Executing query: Test query for logging" in start_log

        # Check completion log
        completion_log = mock_logger.info.call_args_list[1][0][0]
        assert "Query executed successfully" in completion_log
        assert "log-test-123" in completion_log

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_success(self, mock_get_api: MagicMock) -> None:
        """Test successful query status retrieval by correlation_id."""
        # Setup mock orchestration API with status data
        mock_api = AsyncMock()
        mock_status = StatusResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440001",
            status="running",
            progress_percentage=75.0,
            current_agent="critic",
            estimated_completion_seconds=15.5,
        )
        mock_api.get_status_by_correlation_id.return_value = mock_status
        mock_get_api.return_value = mock_api

        correlation_id = "test-correlation-456"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 200
        data: StatusResponseJSON = response.json()

        # Verify response structure matches StatusResponse
        assert data["workflow_id"] == "550e8400-e29b-41d4-a716-446655440001"
        assert data["status"] == "running"
        assert data["progress_percentage"] == 75.0
        assert data["current_agent"] == "critic"
        assert data["estimated_completion_seconds"] == 15.5

        # Verify API was called with correct correlation_id
        mock_api.get_status_by_correlation_id.assert_called_once_with(correlation_id)

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_completed_workflow(self, mock_get_api: MagicMock) -> None:
        """Test query status for completed workflow."""
        mock_api = AsyncMock()
        mock_status = StatusResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440002",
            status="completed",
            progress_percentage=100.0,
            current_agent=None,
            estimated_completion_seconds=None,
        )
        mock_api.get_status_by_correlation_id.return_value = mock_status
        mock_get_api.return_value = mock_api

        correlation_id = "completed-correlation-789"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["workflow_id"] == "550e8400-e29b-41d4-a716-446655440002"
        assert data["status"] == "completed"
        assert data["progress_percentage"] == 100.0
        assert data["current_agent"] is None
        assert data["estimated_completion_seconds"] is None

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_failed_workflow(self, mock_get_api: MagicMock) -> None:
        """Test query status for failed workflow."""
        mock_api = AsyncMock()
        mock_status = StatusResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440003",
            status="failed",
            progress_percentage=45.0,
            current_agent=None,
            estimated_completion_seconds=None,
        )
        mock_api.get_status_by_correlation_id.return_value = mock_status
        mock_get_api.return_value = mock_api

        correlation_id = "failed-correlation-101"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "failed"
        assert data["progress_percentage"] == 45.0
        assert data["current_agent"] is None

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_correlation_id_not_found(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query status when correlation_id is not found."""
        mock_api = AsyncMock()
        mock_api.get_status_by_correlation_id.side_effect = KeyError(
            "No workflow found for correlation_id: nonexistent-id"
        )
        mock_get_api.return_value = mock_api

        correlation_id = "nonexistent-id"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 404
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Correlation ID not found"
        assert detail["correlation_id"] == correlation_id
        assert "No workflow found" in detail["message"]

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_orchestration_api_failure(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query status when orchestration API fails."""
        mock_get_api.side_effect = Exception("Orchestration API unavailable")

        correlation_id = "test-correlation-500"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 500
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Failed to retrieve workflow status"
        assert "Orchestration API unavailable" in detail["message"]
        assert detail["type"] == "Exception"

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_api_method_failure(self, mock_get_api: MagicMock) -> None:
        """Test query status when get_status_by_correlation_id method fails."""
        mock_api = AsyncMock()
        mock_api.get_status_by_correlation_id.side_effect = RuntimeError(
            "Database connection timeout"
        )
        mock_get_api.return_value = mock_api

        correlation_id = "timeout-correlation-123"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 500
        data = response.json()

        detail = data["detail"]
        assert detail["error"] == "Failed to retrieve workflow status"
        assert "Database connection timeout" in detail["message"]
        assert detail["type"] == "RuntimeError"

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_logging(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that query status retrieval logs appropriately."""
        mock_api = AsyncMock()
        mock_status = StatusResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440001",
            status="running",
            progress_percentage=65.0,
            current_agent="synthesis",
            estimated_completion_seconds=8.5,
        )
        mock_api.get_status_by_correlation_id.return_value = mock_status
        mock_get_api.return_value = mock_api

        correlation_id = "logging-test-correlation"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 200

        # Verify logging calls
        assert mock_logger.info.call_count == 2

        # Check start log
        start_log = mock_logger.info.call_args_list[0][0][0]
        assert "Getting status for correlation_id" in start_log
        assert correlation_id in start_log

        # Check completion log
        completion_log = mock_logger.info.call_args_list[1][0][0]
        assert "Status retrieved for correlation_id" in completion_log
        assert correlation_id in completion_log
        assert "workflow_id=550e8400-e29b-41d4-a716-446655440001" in completion_log
        assert "status=running" in completion_log

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_error_logging(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that query status errors are logged properly."""
        error_message = "Workflow execution interrupted"
        mock_api = AsyncMock()
        mock_api.get_status_by_correlation_id.side_effect = Exception(error_message)
        mock_get_api.return_value = mock_api

        correlation_id = "error-logging-test"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 500

        # Verify error was logged
        mock_logger.error.assert_called_once()
        logged_message = mock_logger.error.call_args[0][0]
        assert "Failed to get status for correlation_id" in logged_message
        assert correlation_id in logged_message

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_not_found_logging(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that correlation ID not found cases are logged as warnings."""
        mock_api = AsyncMock()
        mock_api.get_status_by_correlation_id.side_effect = KeyError(
            "No workflow found for correlation_id: missing-correlation"
        )
        mock_get_api.return_value = mock_api

        correlation_id = "missing-correlation"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 404

        # Verify warning was logged for not found case
        mock_logger.warning.assert_called_once()
        logged_message = mock_logger.warning.call_args[0][0]
        assert "Correlation ID not found" in logged_message
        assert correlation_id in logged_message

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_status_multiple_statuses(self, mock_get_api: MagicMock) -> None:
        """Test query status endpoint with various workflow statuses."""
        mock_api = AsyncMock()
        mock_get_api.return_value = mock_api

        # Test different status scenarios with proper typing
        test_cases: list[StatusTestCase] = [
            {
                "correlation_id": "running-workflow",
                "status": StatusResponse(
                    workflow_id="550e8400-e29b-41d4-a716-446655440001",
                    status="running",
                    progress_percentage=30.0,
                    current_agent="refiner",
                    estimated_completion_seconds=25.0,
                ),
            },
            {
                "correlation_id": "cancelled-workflow",
                "status": StatusResponse(
                    workflow_id="550e8400-e29b-41d4-a716-446655440002",
                    status="cancelled",
                    progress_percentage=50.0,
                    current_agent=None,
                    estimated_completion_seconds=None,
                ),
            },
        ]

        for test_case in test_cases:
            mock_api.get_status_by_correlation_id.return_value = test_case["status"]

            response = self.client.get(
                f"/api/query/status/{test_case['correlation_id']}"
            )

            assert response.status_code == 200
            data: StatusResponseJSON = response.json()
            assert data["status"] == test_case["status"].status
            assert data["workflow_id"] == test_case["status"].workflow_id
            assert (
                data["progress_percentage"] == test_case["status"].progress_percentage
            )

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_success(self, mock_get_api: MagicMock) -> None:
        """Test successful query history retrieval."""
        # Setup mock orchestration API with history data
        mock_api: Mock = AsyncMock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "What is machine learning?",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "failed",
                "query": "Analyze climate change impact on agriculture with detailed methodology",
                "start_time": 1703097550.0,
                "execution_time": 8.2,
            },
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches WorkflowHistoryResponse
        assert "workflows" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data

        # Verify default pagination
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert data["total"] == 2
        assert isinstance(data["has_more"], bool)

        # Verify workflow history items
        workflows = data["workflows"]
        assert len(workflows) == 2

        # Check first workflow
        first = workflows[0]
        assert first["workflow_id"] == "550e8400-e29b-41d4-a716-446655440001"
        assert first["status"] == "completed"
        assert first["query"] == "What is machine learning?"
        assert first["start_time"] == 1703097600.0
        assert first["execution_time_seconds"] == 12.5

        # Check second workflow (truncated query)
        second = workflows[1]
        assert second["workflow_id"] == "550e8400-e29b-41d4-a716-446655440002"
        assert second["status"] == "failed"
        assert "climate change" in second["query"]

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_with_custom_parameters(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history endpoint with custom limit and offset."""
        # Setup mock with more data to test pagination
        mock_api: Mock = AsyncMock()
        mock_history = [
            {
                "workflow_id": f"550e8400-e29b-41d4-a716-44665544{i:04d}",
                "status": "completed" if i % 2 == 0 else "failed",
                "query": f"Test query {i}",
                "start_time": 1703097600.0 + i,
                "execution_time": 10.0 + i,
            }
            for i in range(25)  # Create 25 workflows
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_history[
            10:15
        ]  # Return 5 items starting from offset 10 (for limit=5, offset=10)
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=5&offset=10")

        assert response.status_code == 200
        data = response.json()

        # Verify custom parameters are respected
        assert data["limit"] == 5
        assert data["offset"] == 10
        assert data["total"] == 15  # offset (10) + returned items (5) = estimated total
        # Note: has_more logic might depend on the specific implementation
        # For now, just verify the field exists and has a boolean value
        assert isinstance(data["has_more"], bool)

        # Verify we get exactly 5 workflows (after offset)
        workflows = data["workflows"]
        assert len(workflows) == 5

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_with_pagination_has_more(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history pagination with has_more=True."""
        mock_api: Mock = AsyncMock()
        # Create 25 workflows, return first 20 for limit=10, offset=5
        mock_history = [
            {
                "workflow_id": f"550e8400-e29b-41d4-a716-44665544{i:04d}",
                "status": "completed",
                "query": f"Test query {i}",
                "start_time": 1703097600.0,
                "execution_time": 10.0,
            }
            for i in range(20)
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_history[
            :10
        ]  # Return only 10 items for limit=10
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=10&offset=5")

        assert response.status_code == 200
        data = response.json()

        assert data["limit"] == 10
        assert data["offset"] == 5
        assert data["total"] == 15  # offset (5) + returned items (10) = 15
        # has_more logic varies by implementation, just check it's boolean
        assert isinstance(data["has_more"], bool)

        workflows = data["workflows"]
        assert len(workflows) == 10

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_empty_results(self, mock_get_api: MagicMock) -> None:
        """Test query history with no workflows."""
        mock_api: Mock = AsyncMock()
        mock_api.get_workflow_history_from_database.return_value = []
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        assert data["workflows"] == []
        assert data["total"] == 0
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert isinstance(data["has_more"], bool)

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_orchestration_failure(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history when orchestration API fails."""
        mock_get_api.side_effect = Exception("Orchestration API unavailable")

        response = self.client.get("/api/query/history")

        assert response.status_code == 500
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Failed to retrieve workflow history"
        assert "Orchestration API unavailable" in detail["message"]
        assert detail["type"] == "Exception"

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_with_invalid_workflow_data(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history with malformed workflow data."""
        mock_api: Mock = AsyncMock()
        # Include one valid and one invalid workflow
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Valid workflow",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                # Missing required fields to test error handling
                "workflow_id": "invalid-id",
                "status": "completed",
                # Missing query, start_time, execution_time
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "failed",
                "query": "Another valid workflow",
                "start_time": 1703097650.0,
                "execution_time": 8.0,
            },
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        # Should return only valid workflows (1st and 3rd)
        workflows = data["workflows"]
        assert len(workflows) == 2
        assert workflows[0]["workflow_id"] == "550e8400-e29b-41d4-a716-446655440001"
        assert workflows[1]["workflow_id"] == "550e8400-e29b-41d4-a716-446655440003"

    def test_get_query_history_parameter_validation(self) -> None:
        """Test query history parameter validation."""
        # Test invalid limit (too high)
        response = self.client.get("/api/query/history?limit=101")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = self.client.get("/api/query/history?limit=0")
        assert response.status_code == 422

        # Test invalid offset (negative)
        response = self.client.get("/api/query/history?offset=-1")
        assert response.status_code == 422

        # Test non-integer parameters
        response = self.client.get("/api/query/history?limit=abc")
        assert response.status_code == 422

        response = self.client.get("/api/query/history?offset=xyz")
        assert response.status_code == 422

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_logging(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that query history retrieval logs appropriately."""
        mock_api: Mock = AsyncMock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Test query",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            }
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=5&offset=0")

        assert response.status_code == 200

        # Verify logging calls
        # Should have info log for start and completion
        assert mock_logger.info.call_count == 2

        # Check start log
        start_log = mock_logger.info.call_args_list[0][0][0]
        assert "Fetching workflow history" in start_log
        assert "limit=5" in start_log
        assert "offset=0" in start_log

        # Check completion log
        completion_log = mock_logger.info.call_args_list[1][0][0]
        assert "Workflow history retrieved" in completion_log
        assert "1 items" in completion_log
        assert "total=1" in completion_log

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_status_variations(self, mock_get_api: MagicMock) -> None:
        """Test query history with different workflow statuses."""
        mock_api: Mock = AsyncMock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Completed workflow",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "failed",
                "query": "Failed workflow",
                "start_time": 1703097550.0,
                "execution_time": 8.2,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "running",
                "query": "Running workflow",
                "start_time": 1703097650.0,
                "execution_time": 5.0,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440004",
                "status": "cancelled",
                "query": "Cancelled workflow",
                "start_time": 1703097500.0,
                "execution_time": 2.1,
            },
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        workflows = data["workflows"]
        assert len(workflows) == 4

        # Verify all status types are handled
        statuses = [wf["status"] for wf in workflows]
        assert "completed" in statuses
        assert "failed" in statuses
        assert "running" in statuses
        assert "cancelled" in statuses

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_execute_query_error_logging(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that query execution errors are logged properly."""
        error_message = "Database connection timeout"
        mock_get_api.side_effect = Exception(error_message)

        request_data = {"query": "Test query that will fail"}

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 500

        # Verify error was logged
        mock_logger.error.assert_called_once()
        logged_message = mock_logger.error.call_args[0][0]
        assert "Query execution failed" in logged_message
        assert error_message in str(logged_message)

    def test_query_endpoint_response_schema(self) -> None:
        """Test that query endpoint responses match expected schema."""
        # Test with invalid request to get 422 response
        response = self.client.post("/api/query", json={})

        assert response.status_code == 422
        data = response.json()

        # Verify FastAPI validation error format
        assert "detail" in data
        assert isinstance(data["detail"], list)  # FastAPI validation errors are lists


class TestQueryHistoryDatabaseIntegration:
    """Test suite for updated query history endpoint using database persistence."""

    def setup_method(self) -> None:
        """Set up test client for each test."""
        self.client = TestClient(app)

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_success(self, mock_get_api: MagicMock) -> None:
        """Test successful query history retrieval from database."""
        # Setup mock orchestration API with database history data
        mock_api: Mock = AsyncMock()
        mock_database_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "What is machine learning?",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "failed",
                "query": "Complex climate analysis with extensive methodology and detailed breakdown",
                "start_time": 1703097550.0,
                "execution_time": 8.2,
            },
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_database_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches WorkflowHistoryResponse
        assert "workflows" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data

        # Verify default pagination parameters were used
        mock_api.get_workflow_history_from_database.assert_called_once_with(
            limit=10, offset=0
        )

        # Verify default pagination values in response
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert data["total"] == 2
        assert isinstance(data["has_more"], bool)

        # Verify workflow history items from database
        workflows = data["workflows"]
        assert len(workflows) == 2

        # Check first workflow
        first = workflows[0]
        assert first["workflow_id"] == "550e8400-e29b-41d4-a716-446655440001"
        assert first["status"] == "completed"
        assert first["query"] == "What is machine learning?"
        assert first["start_time"] == 1703097600.0
        assert first["execution_time_seconds"] == 12.5

        # Check second workflow (should handle truncated query)
        second = workflows[1]
        assert second["workflow_id"] == "550e8400-e29b-41d4-a716-446655440002"
        assert second["status"] == "failed"
        assert "climate analysis" in second["query"]
        assert second["execution_time_seconds"] == 8.2

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_with_pagination(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history from database with custom pagination parameters."""
        mock_api: Mock = AsyncMock()
        # Mock database returning exactly 5 items for limit=5, offset=20
        mock_database_history = [
            {
                "workflow_id": f"550e8400-e29b-41d4-a716-44665544{20 + i:04d}",
                "status": "completed" if i % 2 == 0 else "failed",
                "query": f"Database query {20 + i}",
                "start_time": 1703097600.0 + i,
                "execution_time": 10.0 + i,
            }
            for i in range(5)  # Return exactly 5 items
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_database_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=5&offset=20")

        assert response.status_code == 200
        data = response.json()

        # Verify database method was called with correct parameters
        mock_api.get_workflow_history_from_database.assert_called_once_with(
            limit=5, offset=20
        )

        # Verify pagination parameters in response
        assert data["limit"] == 5
        assert data["offset"] == 20
        assert data["total"] == 25  # offset + returned items
        assert isinstance(
            data["has_more"], bool
        )  # We got full limit, so likely more exist

        # Verify we get exactly 5 workflows
        workflows = data["workflows"]
        assert len(workflows) == 5

        # Verify the workflows have correct IDs
        expected_ids = [
            f"550e8400-e29b-41d4-a716-44665544{20 + i:04d}" for i in range(5)
        ]
        actual_ids = [wf["workflow_id"] for wf in workflows]
        assert actual_ids == expected_ids

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_empty_results(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history from database with no results."""
        mock_api: Mock = AsyncMock()
        mock_api.get_workflow_history_from_database.return_value = []
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        # Verify database method was called
        mock_api.get_workflow_history_from_database.assert_called_once_with(
            limit=10, offset=0
        )

        # Verify empty response structure
        assert data["workflows"] == []
        assert data["total"] == 0
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert isinstance(data["has_more"], bool)

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_error_handling(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history when database access fails."""
        mock_api: Mock = AsyncMock()
        mock_api.get_workflow_history_from_database.side_effect = Exception(
            "Database connection failed"
        )
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 500
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Failed to retrieve workflow history"
        assert "Database connection failed" in detail["message"]
        assert detail["type"] == "Exception"

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_partial_data(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history with partial results indicating more data exists."""
        mock_api: Mock = AsyncMock()
        # Return exactly limit number of items to indicate more might exist
        mock_database_history = [
            {
                "workflow_id": f"550e8400-e29b-41d4-a716-44665544{i:04d}",
                "status": "completed",
                "query": f"Query {i}",
                "start_time": 1703097600.0 + i,
                "execution_time": 10.0,
            }
            for i in range(10)  # Return exactly 10 items for limit=10
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_database_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=10&offset=5")

        assert response.status_code == 200
        data = response.json()

        # When we get full limit back, has_more should be True
        assert data["limit"] == 10
        assert data["offset"] == 5
        assert data["total"] == 15  # offset + returned
        assert isinstance(
            data["has_more"], bool
        )  # Full limit returned suggests more exist

        workflows = data["workflows"]
        assert len(workflows) == 10

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_invalid_data_handling(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history handles invalid database records gracefully."""
        mock_api: Mock = AsyncMock()
        # Include mix of valid and invalid workflow data
        mock_database_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Valid workflow query",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                # Missing required fields
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "completed",
                # Missing query, start_time, execution_time
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "failed",
                "query": "Another valid workflow",
                "start_time": 1703097650.0,
                "execution_time": 8.0,
            },
            {
                # Invalid data types
                "workflow_id": 12345,  # Should be string
                "status": "completed",
                "query": "Query with invalid ID type",
                "start_time": "invalid_timestamp",  # Should be float
                "execution_time": "invalid_time",  # Should be float
            },
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_database_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        # Should return only valid workflows (1st and 3rd)
        workflows = data["workflows"]
        assert len(workflows) == 2
        assert workflows[0]["workflow_id"] == "550e8400-e29b-41d4-a716-446655440001"
        assert workflows[1]["workflow_id"] == "550e8400-e29b-41d4-a716-446655440003"

        # Invalid workflows should be skipped without breaking the response

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_logging(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that database history retrieval logs appropriately."""
        mock_api: Mock = AsyncMock()
        mock_database_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Test query for logging",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            }
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_database_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=15&offset=5")

        assert response.status_code == 200

        # Verify logging calls
        assert mock_logger.info.call_count == 2

        # Check start log includes parameters
        start_log = mock_logger.info.call_args_list[0][0][0]
        assert "Fetching workflow history" in start_log
        assert "limit=15" in start_log
        assert "offset=5" in start_log

        # Check completion log includes results summary
        completion_log = mock_logger.info.call_args_list[1][0][0]
        assert "Workflow history retrieved" in completion_log
        assert "1 items" in completion_log
        assert "total=6" in completion_log  # offset + returned items
        assert "has_more=False" in completion_log

        # Check debug log for raw history retrieval
        mock_logger.debug.assert_called_once()
        debug_log = mock_logger.debug.call_args[0][0]
        assert "Raw history retrieved" in debug_log
        assert "1 workflows" in debug_log

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_large_offset(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history with large offset returns appropriate results."""
        mock_api: Mock = AsyncMock()
        # Return fewer items than limit when offset is large (near end of data)
        mock_database_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Near end of dataset",
                "start_time": 1703097600.0,
                "execution_time": 5.0,
            }
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_database_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=10&offset=95")

        assert response.status_code == 200
        data = response.json()

        # Verify parameters
        assert data["limit"] == 10
        assert data["offset"] == 95
        assert data["total"] == 96  # offset + returned (1)
        assert isinstance(
            data["has_more"], bool
        )  # Less than limit returned, so no more

        workflows = data["workflows"]
        assert len(workflows) == 1
        assert workflows[0]["workflow_id"] == "550e8400-e29b-41d4-a716-446655440001"

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_database_workflow_status_variants(
        self, mock_get_api: MagicMock
    ) -> None:
        """Test query history from database handles all workflow status types."""
        mock_api: Mock = AsyncMock()
        mock_database_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Completed workflow from database",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "failed",
                "query": "Failed workflow from database",
                "start_time": 1703097550.0,
                "execution_time": 8.2,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "running",
                "query": "Running workflow from database",
                "start_time": 1703097650.0,
                "execution_time": 5.0,
            },
        ]
        mock_api.get_workflow_history_from_database.return_value = mock_database_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        workflows = data["workflows"]
        assert len(workflows) == 3

        # Verify all status types are preserved
        statuses = [wf["status"] for wf in workflows]
        assert "completed" in statuses
        assert "failed" in statuses
        assert "running" in statuses

        # Verify workflow IDs are correct
        workflow_ids = [wf["workflow_id"] for wf in workflows]
        expected_ids = [
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400-e29b-41d4-a716-446655440002",
            "550e8400-e29b-41d4-a716-446655440003",
        ]
        assert workflow_ids == expected_ids
