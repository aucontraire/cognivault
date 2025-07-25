"""
Unit tests for FastAPI query route endpoints.

Tests the web layer query endpoints using existing API models and factory patterns.
"""

import pytest
from unittest.mock import patch, AsyncMock, Mock
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.models import (
    WorkflowRequest,
    WorkflowResponse,
    WorkflowHistoryResponse,
    WorkflowHistoryItem,
)


class TestQueryRoutes:
    """Test suite for query execution endpoints."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_execute_query_success(self, mock_get_api):
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

        # Verify response structure matches WorkflowResponse
        assert data["workflow_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert data["status"] == "completed"
        assert data["correlation_id"] == "test-correlation-123"
        assert data["execution_time_seconds"] == 15.5
        assert "agent_outputs" in data
        assert len(data["agent_outputs"]) == 4

    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_execute_query_orchestration_failure(self, mock_get_api):
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
    async def test_execute_query_with_minimal_request(self, mock_get_api):
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

    def test_execute_query_invalid_request(self):
        """Test query execution with invalid request data."""
        # Missing required 'query' field
        request_data = {"agents": ["refiner"], "execution_config": {}}

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_execute_query_empty_query(self):
        """Test query execution with empty query string."""
        request_data = {"query": ""}

        response = self.client.post("/api/query", json=request_data)

        # This should be handled by Pydantic validation
        assert response.status_code == 422

    @patch("cognivault.api.routes.query.logger")
    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_execute_query_logs_appropriately(self, mock_get_api, mock_logger):
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

    def test_get_query_status_placeholder(self):
        """Test query status endpoint (placeholder implementation)."""
        correlation_id = "test-correlation-456"
        response = self.client.get(f"/api/query/status/{correlation_id}")

        assert response.status_code == 200
        data = response.json()

        # Verify placeholder response structure
        assert data["correlation_id"] == correlation_id
        assert data["status"] == "completed"  # Placeholder value
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_success(self, mock_get_api):
        """Test successful query history retrieval."""
        # Setup mock orchestration API with history data
        mock_api = Mock()
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
        mock_api.get_workflow_history.return_value = mock_history
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
        assert data["has_more"] is False

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
    def test_get_query_history_with_custom_parameters(self, mock_get_api):
        """Test query history endpoint with custom limit and offset."""
        # Setup mock with more data to test pagination
        mock_api = Mock()
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
        mock_api.get_workflow_history.return_value = mock_history[
            :15
        ]  # Return first 15 for limit=5, offset=10
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=5&offset=10")

        assert response.status_code == 200
        data = response.json()

        # Verify custom parameters are respected
        assert data["limit"] == 5
        assert data["offset"] == 10
        assert data["total"] == 15  # Based on mock return
        assert data["has_more"] is False  # 10 + 5 = 15, which equals total

        # Verify we get exactly 5 workflows (after offset)
        workflows = data["workflows"]
        assert len(workflows) == 5

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_with_pagination_has_more(self, mock_get_api):
        """Test query history pagination with has_more=True."""
        mock_api = Mock()
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
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history?limit=10&offset=5")

        assert response.status_code == 200
        data = response.json()

        assert data["limit"] == 10
        assert data["offset"] == 5
        assert data["total"] == 20
        # offset(5) + returned(10) = 15, which is < total(20), so has_more=True
        assert data["has_more"] is True

        workflows = data["workflows"]
        assert len(workflows) == 10

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_empty_results(self, mock_get_api):
        """Test query history with no workflows."""
        mock_api = Mock()
        mock_api.get_workflow_history.return_value = []
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        assert data["workflows"] == []
        assert data["total"] == 0
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert data["has_more"] is False

    @patch("cognivault.api.routes.query.get_orchestration_api")
    def test_get_query_history_orchestration_failure(self, mock_get_api):
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
    def test_get_query_history_with_invalid_workflow_data(self, mock_get_api):
        """Test query history with malformed workflow data."""
        mock_api = Mock()
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
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        # Should return only valid workflows (1st and 3rd)
        workflows = data["workflows"]
        assert len(workflows) == 2
        assert workflows[0]["workflow_id"] == "550e8400-e29b-41d4-a716-446655440001"
        assert workflows[1]["workflow_id"] == "550e8400-e29b-41d4-a716-446655440003"

    def test_get_query_history_parameter_validation(self):
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
    def test_get_query_history_logging(self, mock_get_api, mock_logger):
        """Test that query history retrieval logs appropriately."""
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Test query",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            }
        ]
        mock_api.get_workflow_history.return_value = mock_history
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
    def test_get_query_history_status_variations(self, mock_get_api):
        """Test query history with different workflow statuses."""
        mock_api = Mock()
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
        mock_api.get_workflow_history.return_value = mock_history
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
    async def test_execute_query_error_logging(self, mock_get_api, mock_logger):
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

    def test_query_endpoint_response_schema(self):
        """Test that query endpoint responses match expected schema."""
        # Test with invalid request to get 422 response
        response = self.client.post("/api/query", json={})

        assert response.status_code == 422
        data = response.json()

        # Verify FastAPI validation error format
        assert "detail" in data
        assert isinstance(data["detail"], list)  # FastAPI validation errors are lists
