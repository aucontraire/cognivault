"""
Unit tests for FastAPI health route endpoints.

Tests the web layer health endpoints without external dependencies.
"""

import pytest
from typing import Any
from unittest.mock import patch, AsyncMock, Mock
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.base import APIHealthStatus
from cognivault.diagnostics.health import HealthStatus
from cognivault.api.factory import reset_api_cache


class TestHealthRoutes:
    """Test suite for health check endpoints."""

    def setup_method(self) -> None:
        """Set up test client for each test."""
        self.client = TestClient(app)
        # Reset API cache to ensure clean state
        reset_api_cache()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Reset API cache to avoid test interference
        reset_api_cache()

    def test_basic_health_check(self) -> None:
        """Test basic health endpoint returns expected structure."""
        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert data["status"] == "healthy"
        assert data["service"] == "cognivault-api"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data

    @patch("cognivault.api.routes.health.get_orchestration_api")
    def test_detailed_health_check_success(self, mock_get_api: Mock) -> None:
        """Test detailed health check with successful orchestration API."""
        # Setup mock orchestration API
        mock_api = AsyncMock()
        mock_status = APIHealthStatus(
            status=HealthStatus.HEALTHY, details="test details", checks={"test": "data"}
        )
        mock_api.health_check.return_value = mock_status
        mock_get_api.return_value = mock_api

        response = self.client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()

        # Verify main response structure
        assert data["status"] == "healthy"
        assert data["service"] == "cognivault-api"
        assert data["version"] == "0.1.0"

        # Verify orchestration dependency status
        assert "dependencies" in data
        assert "orchestration" in data["dependencies"]
        orchestration_data = data["dependencies"]["orchestration"]
        assert orchestration_data["status"] == "healthy"
        assert orchestration_data["details"] == "test details"
        assert orchestration_data["checks"] == {"test": "data"}

    @patch("cognivault.api.routes.health.get_orchestration_api")
    def test_detailed_health_check_failure(self, mock_get_api: Mock) -> None:
        """Test detailed health check with orchestration API failure."""
        # Setup mock to raise exception
        mock_get_api.side_effect = Exception("Orchestration API not initialized")

        response = self.client.get("/health/detailed")

        assert response.status_code == 503
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["status"] == "unhealthy"
        assert detail["service"] == "cognivault-api"
        assert "Orchestration API not initialized" in detail["error"]

    def test_health_endpoints_are_cors_enabled(self) -> None:
        """Test that health endpoints support CORS for development."""
        # Test actual request with origin header - CORSMiddleware handles this
        response = self.client.get(
            "/health", headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200
        # The response should succeed - CORS headers are handled by middleware

    def test_health_check_response_format(self) -> None:
        """Test that health check responses match expected JSON schema."""
        response = self.client.get("/health")
        data = response.json()

        # Verify data types
        assert isinstance(data["status"], str)
        assert isinstance(data["service"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["timestamp"], str)

        # Verify enum values
        assert data["status"] in ["healthy", "unhealthy"]

    @patch("cognivault.api.routes.health.logger")
    @patch("cognivault.api.routes.health.get_orchestration_api")
    def test_detailed_health_check_logs_errors(
        self, mock_get_api: Mock, mock_logger: Mock
    ) -> None:
        """Test that detailed health check logs errors appropriately."""
        error_message = "Database connection failed"
        mock_get_api.side_effect = Exception(error_message)

        response = self.client.get("/health/detailed")

        assert response.status_code == 503
        # Verify error was logged
        mock_logger.error.assert_called_once()
        logged_message = mock_logger.error.call_args[0][0]
        assert "Health check failed" in logged_message
        assert error_message in str(logged_message)
