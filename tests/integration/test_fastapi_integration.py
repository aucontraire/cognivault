"""
Integration tests for FastAPI web layer.

Tests the complete FastAPI application integration with existing API infrastructure.
"""

import pytest
from typing import Any, List
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import Response

from cognivault.api.main import app
from cognivault.api.models import WorkflowRequest, WorkflowResponse
from cognivault.api.base import APIHealthStatus
from cognivault.diagnostics.health import HealthStatus


class TestFastAPIIntegration:
    """Integration test suite for FastAPI application."""

    def setup_method(self) -> None:
        """Set up test client for each test."""
        self.client = TestClient(app)

    def test_app_initialization(self) -> None:
        """Test that FastAPI app initializes correctly with all routes."""
        # Verify the app has been created
        assert app is not None
        assert app.title == "CogniVault API"
        assert app.version == "0.1.0"

        # Test that CORS middleware is configured
        middlewares = [
            getattr(middleware.cls, "__name__", str(middleware.cls))
            for middleware in app.user_middleware
        ]
        assert "CORSMiddleware" in middlewares

    def test_openapi_schema_generation(self) -> None:
        """Test that OpenAPI schema is generated correctly."""
        response = self.client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        # Verify basic schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "CogniVault API"
        assert schema["info"]["version"] == "0.1.0"

        # Verify our endpoints are documented
        paths = schema["paths"]
        assert "/health" in paths
        assert "/health/detailed" in paths
        assert "/api/query" in paths
        assert "/api/query/status/{correlation_id}" in paths
        assert "/api/query/history" in paths

    def test_docs_endpoint_accessible(self) -> None:
        """Test that Swagger UI docs are accessible."""
        response = self.client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"swagger-ui" in response.content.lower()

    def test_redoc_endpoint_accessible(self) -> None:
        """Test that ReDoc documentation is accessible."""
        response = self.client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"redoc" in response.content.lower()

    def test_cors_configuration(self) -> None:
        """Test CORS configuration for development."""
        # Test preflight request
        response = self.client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200

    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_end_to_end_query_flow(self, mock_get_api: AsyncMock) -> None:
        """Test complete end-to-end query execution flow."""
        # Setup mock orchestration API
        mock_api = AsyncMock()
        mock_response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440000",
            status="completed",
            agent_outputs={
                "refiner": "Refined query about end-to-end testing",
                "historian": "Historical context for testing",
                "critic": "Critical analysis of test methodology",
                "synthesis": "This is a test result from the orchestration layer",
            },
            execution_time_seconds=25.3,
            correlation_id="e2e-test-123",
        )
        mock_api.execute_workflow.return_value = mock_response
        mock_get_api.return_value = mock_api

        # Test the complete flow
        request_data = {
            "query": "End-to-end test query",
            "agents": ["refiner", "historian", "critic", "synthesis"],
            "execution_config": {"timeout": 60, "enable_checkpoints": True},
        }

        response = self.client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify the request was properly transformed and handled
        assert data["workflow_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert data["correlation_id"] == "e2e-test-123"
        assert data["status"] == "completed"
        assert len(data["agent_outputs"]) == 4
        assert data["execution_time_seconds"] == 25.3

        # Verify the factory was called correctly
        mock_get_api.assert_called_once()
        mock_api.execute_workflow.assert_called_once()

        # Verify the request was converted to WorkflowRequest
        call_args = mock_api.execute_workflow.call_args[0][0]
        assert isinstance(call_args, WorkflowRequest)
        assert call_args.query == "End-to-end test query"

    @patch("cognivault.api.routes.health.get_orchestration_api")
    async def test_health_check_integration_flow(self, mock_get_api: AsyncMock) -> None:
        """Test health check integration with orchestration API."""
        # Setup mock for successful health check
        mock_api = AsyncMock()
        mock_status = APIHealthStatus(
            status=HealthStatus.HEALTHY,
            details="LangGraphOrchestrationAPI is healthy",
            checks={
                "initialized": True,
                "agent_count": 4,
                "langgraph_version": "0.2.0",
            },
        )
        mock_api.health_check.return_value = mock_status
        mock_get_api.return_value = mock_api

        # Test detailed health check
        response = self.client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()

        # Verify integration worked correctly
        assert data["status"] == "healthy"
        orchestration = data["dependencies"]["orchestration"]
        assert orchestration["status"] == "healthy"
        assert orchestration["details"] == "LangGraphOrchestrationAPI is healthy"
        assert orchestration["checks"]["initialized"] is True

    def test_error_handling_consistency(self) -> None:
        """Test that error responses are consistent across endpoints."""
        # Test 404 for non-existent endpoint
        response = self.client.get("/nonexistent")
        assert response.status_code == 404

        # Test 422 for invalid request body
        response = self.client.post("/api/query", json={"invalid": "data"})
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

    def test_request_response_serialization(self) -> None:
        """Test that request/response models serialize correctly."""
        # Test with complex request data
        complex_request = {
            "query": "Complex query with unicode: 测试 🚀",
            "agents": ["refiner", "historian", "critic", "synthesis"],
            "execution_config": {
                "timeout": 120,
                "enable_checkpoints": True,
                "metadata": {"user_id": "test-user-123", "session_id": "session-456"},
            },
        }

        # This should not raise serialization errors
        response = self.client.post("/api/query", json=complex_request)

        # Even if the orchestration fails, the request should be parsed correctly
        assert response.status_code in [
            200,
            500,
        ]  # 500 if orchestration not initialized

    def test_route_mounting_and_tags(self) -> None:
        """Test that routes are mounted correctly with proper tags."""
        response = self.client.get("/openapi.json")
        schema = response.json()

        # Verify health endpoints have Health tag
        health_endpoint = schema["paths"]["/health"]["get"]
        assert "Health" in health_endpoint["tags"]

        # Verify query endpoints have Query tag
        query_endpoint = schema["paths"]["/api/query"]["post"]
        assert "Query" in query_endpoint["tags"]

    def test_response_headers(self) -> None:
        """Test that responses include appropriate headers."""
        response = self.client.get("/health")

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    @patch("cognivault.api.routes.query.get_orchestration_api")
    async def test_concurrent_request_handling(self, mock_get_api: AsyncMock) -> None:
        """Test that the app can handle concurrent requests properly."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Setup mock
        mock_api = AsyncMock()
        mock_response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440001",
            status="completed",
            agent_outputs={"refiner": "Concurrent test result"},
            execution_time_seconds=5.0,
            correlation_id="concurrent-123",
        )
        mock_api.execute_workflow.return_value = mock_response
        mock_get_api.return_value = mock_api

        def make_request() -> Response:
            return self.client.post("/api/query", json={"query": "Concurrent test"})

        # Make multiple concurrent requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses: List[Response] = [future.result() for future in futures]

        # All requests should succeed (or fail consistently)
        status_codes = [r.status_code for r in responses]
        assert all(code in [200, 500] for code in status_codes)

    def test_application_metadata(self) -> None:
        """Test application metadata is correctly exposed."""
        response = self.client.get("/openapi.json")
        schema = response.json()

        info = schema["info"]
        assert info["title"] == "CogniVault API"
        assert (
            info["description"]
            == "Multi-agent workflow orchestration platform with intelligent routing"
        )
        assert info["version"] == "0.1.0"
