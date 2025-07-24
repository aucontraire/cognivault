"""
Unit tests for FastAPI query route endpoints.

Tests the web layer query endpoints using existing API models and factory patterns.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.models import WorkflowRequest, WorkflowResponse


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

    def test_get_query_history_placeholder(self):
        """Test query history endpoint (placeholder implementation)."""
        response = self.client.get("/api/query/history")

        assert response.status_code == 200
        data = response.json()

        # Verify placeholder response structure
        assert data["queries"] == []
        assert data["total"] == 0
        assert data["limit"] == 10  # Default value
        assert data["offset"] == 0  # Default value
        assert "message" in data

    def test_get_query_history_with_parameters(self):
        """Test query history endpoint with custom limit and offset."""
        response = self.client.get("/api/query/history?limit=20&offset=5")

        assert response.status_code == 200
        data = response.json()

        # Verify parameters are respected
        assert data["limit"] == 20
        assert data["offset"] == 5

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
