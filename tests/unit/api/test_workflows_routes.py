"""
Unit tests for FastAPI workflows route endpoints.

Tests the workflow discovery and management endpoints using mock data and services.
"""

import pytest
from typing import Any
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.models import WorkflowMetadata, WorkflowsResponse


class TestWorkflowsRoutes:
    """Test suite for workflows discovery endpoints."""

    def setup_method(self) -> None:
        """Set up test client for each test."""
        self.client = TestClient(app)
        # Clear cache between tests to ensure test isolation
        from cognivault.api.routes.workflows import workflow_service

        workflow_service._workflow_cache = {}
        workflow_service._cache_timestamp = 0.0

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_success(self, mock_get_dirs: Mock) -> None:
        """Test successful workflow discovery with mock workflow files."""
        # Create temporary directory structure with mock workflow files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock workflow files
            workflows_data = [
                {
                    "workflow_id": "academic_research",
                    "name": "Academic Research Analysis",
                    "description": "Comprehensive academic research workflow with peer-review standards",
                    "version": "1.0.0",
                    "created_by": "CogniVault Team",
                    "metadata": {
                        "domain": "academic",
                        "complexity_level": "high",
                        "estimated_execution_time": "45-60 seconds",
                        "tags": ["academic", "research", "scholarly"],
                        "use_cases": ["dissertation_research", "literature_review"],
                    },
                    "nodes": [
                        {"name": "refiner", "type": "processor"},
                        {"name": "historian", "type": "processor"},
                        {"name": "critic", "type": "processor"},
                        {"name": "synthesis", "type": "aggregator"},
                    ],
                },
                {
                    "workflow_id": "legal_analysis",
                    "name": "Legal Document Analysis",
                    "description": "Legal compliance and analysis workflow",
                    "version": "1.2.0",
                    "created_by": "Legal Team",
                    "metadata": {
                        "domain": "legal",
                        "complexity_level": "expert",
                        "estimated_execution_time": "60-90 seconds",
                        "tags": ["legal", "compliance", "analysis"],
                        "use_cases": ["contract_review", "compliance_check"],
                    },
                    "nodes": [
                        {"name": "refiner", "type": "processor"},
                        {"name": "legal_analyzer", "type": "processor"},
                        {"name": "compliance_checker", "type": "validator"},
                    ],
                },
            ]

            # Write workflow files
            for i, workflow_data in enumerate(workflows_data):
                workflow_file = temp_path / f"workflow_{i}.yaml"
                with open(workflow_file, "w") as f:
                    yaml.dump(workflow_data, f)

            # Mock the directory discovery
            mock_get_dirs.return_value = [temp_path]

            # Make request
            response = self.client.get("/api/workflows")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure matches WorkflowsResponse
            assert "workflows" in data
            assert "categories" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert "has_more" in data

            # Verify default pagination
            assert data["limit"] == 10
            assert data["offset"] == 0
            assert data["search_query"] is None
            assert data["category_filter"] is None
            assert data["complexity_filter"] is None

            # Verify workflows were discovered
            workflows = data["workflows"]
            assert len(workflows) == 2
            assert data["total"] == 2
            assert data["has_more"] is False

            # Verify categories were extracted
            categories = data["categories"]
            assert "academic" in categories
            assert "legal" in categories

            # Verify workflow structure
            for workflow in workflows:
                assert "workflow_id" in workflow
                assert "name" in workflow
                assert "description" in workflow
                assert "version" in workflow
                assert "category" in workflow
                assert "tags" in workflow
                assert "created_by" in workflow
                assert "created_at" in workflow
                assert "estimated_execution_time" in workflow
                assert "complexity_level" in workflow
                assert "node_count" in workflow
                assert "use_cases" in workflow

                # Verify field types and constraints
                assert isinstance(workflow["workflow_id"], str)
                assert isinstance(workflow["name"], str)
                assert len(workflow["name"]) > 0
                assert isinstance(workflow["description"], str)
                assert isinstance(workflow["version"], str)
                assert isinstance(workflow["category"], str)
                assert isinstance(workflow["tags"], list)
                assert len(workflow["tags"]) > 0
                assert isinstance(workflow["created_by"], str)
                assert isinstance(workflow["created_at"], float)
                assert isinstance(workflow["estimated_execution_time"], str)
                assert workflow["complexity_level"] in [
                    "low",
                    "medium",
                    "high",
                    "expert",
                ]
                assert isinstance(workflow["node_count"], int)
                assert workflow["node_count"] > 0
                assert isinstance(workflow["use_cases"], list)

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_with_search(self, mock_get_dirs: Mock) -> None:
        """Test workflow discovery with search filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create workflows with different content for search testing
            workflows_data = [
                {
                    "workflow_id": "machine_learning_analysis",
                    "name": "Machine Learning Research",
                    "description": "Advanced machine learning and AI research workflow",
                    "metadata": {
                        "domain": "research",
                        "tags": ["machine_learning", "ai", "research"],
                        "use_cases": ["model_training", "data_analysis"],
                    },
                    "nodes": [{"name": "analyzer", "type": "processor"}],
                },
                {
                    "workflow_id": "climate_study",
                    "name": "Climate Change Analysis",
                    "description": "Environmental climate change impact study",
                    "metadata": {
                        "domain": "environmental",
                        "tags": ["climate", "environment", "study"],
                        "use_cases": ["climate_modeling", "impact_assessment"],
                    },
                    "nodes": [{"name": "modeler", "type": "processor"}],
                },
            ]

            for i, workflow_data in enumerate(workflows_data):
                workflow_file = temp_path / f"workflow_{i}.yaml"
                with open(workflow_file, "w") as f:
                    yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            # Test search for machine learning workflows
            response = self.client.get("/api/workflows?search=machine learning")

            assert response.status_code == 200
            data = response.json()

            assert data["search_query"] == "machine learning"

            # Should find workflows related to machine learning
            workflows = data["workflows"]
            assert len(workflows) >= 1

            # Check that returned workflows are relevant to search
            found_ml_workflow = False
            for workflow in workflows:
                searchable_text = (
                    workflow["name"]
                    + " "
                    + workflow["description"]
                    + " "
                    + " ".join(workflow["tags"])
                    + " "
                    + " ".join(workflow["use_cases"])
                ).lower()
                if "machine" in searchable_text and "learning" in searchable_text:
                    found_ml_workflow = True
                    break

            assert found_ml_workflow

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_with_category_filter(self, mock_get_dirs: Mock) -> None:
        """Test workflow discovery with category filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            workflows_data = [
                {
                    "workflow_id": "academic_workflow",
                    "name": "Academic Analysis",
                    "description": "Academic research workflow",
                    "metadata": {
                        "domain": "academic",
                        "tags": ["academic", "research"],
                    },
                    "nodes": [{"name": "analyzer", "type": "processor"}],
                },
                {
                    "workflow_id": "business_workflow",
                    "name": "Business Analysis",
                    "description": "Business intelligence workflow",
                    "metadata": {
                        "domain": "business",
                        "tags": ["business", "intelligence"],
                    },
                    "nodes": [{"name": "analyzer", "type": "processor"}],
                },
            ]

            for i, workflow_data in enumerate(workflows_data):
                workflow_file = temp_path / f"workflow_{i}.yaml"
                with open(workflow_file, "w") as f:
                    yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            # Test category filter
            response = self.client.get("/api/workflows?category=academic")

            assert response.status_code == 200
            data = response.json()

            assert data["category_filter"] == "academic"

            workflows = data["workflows"]
            assert len(workflows) == 1
            assert workflows[0]["category"] == "academic"

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_with_complexity_filter(self, mock_get_dirs: Mock) -> None:
        """Test workflow discovery with complexity filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            workflows_data = [
                {
                    "workflow_id": "simple_workflow",
                    "name": "Simple Analysis",
                    "description": "Basic analysis workflow",
                    "metadata": {"domain": "general", "complexity_level": "low"},
                    "nodes": [{"name": "analyzer", "type": "processor"}],
                },
                {
                    "workflow_id": "complex_workflow",
                    "name": "Complex Analysis",
                    "description": "Advanced analysis workflow",
                    "metadata": {"domain": "general", "complexity_level": "expert"},
                    "nodes": [{"name": "analyzer", "type": "processor"}],
                },
            ]

            for i, workflow_data in enumerate(workflows_data):
                workflow_file = temp_path / f"workflow_{i}.yaml"
                with open(workflow_file, "w") as f:
                    yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            # Test complexity filter
            response = self.client.get("/api/workflows?complexity=expert")

            assert response.status_code == 200
            data = response.json()

            assert data["complexity_filter"] == "expert"

            workflows = data["workflows"]
            assert len(workflows) == 1
            assert workflows[0]["complexity_level"] == "expert"

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_with_pagination(self, mock_get_dirs: Mock) -> None:
        """Test workflow discovery with custom pagination parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple workflows for pagination testing
            workflows_data = []
            for i in range(15):  # Create 15 workflows
                workflows_data.append(
                    {
                        "workflow_id": f"workflow_{i:02d}",
                        "name": f"Workflow {i:02d}",
                        "description": f"Test workflow number {i}",
                        "metadata": {
                            "domain": "test",
                            "tags": ["test", f"workflow_{i}"],
                        },
                        "nodes": [{"name": "processor", "type": "processor"}],
                    }
                )

            for i, workflow_data in enumerate(workflows_data):
                workflow_file = temp_path / f"workflow_{i}.yaml"
                with open(workflow_file, "w") as f:
                    yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            # Test custom pagination
            response = self.client.get("/api/workflows?limit=5&offset=3")

            assert response.status_code == 200
            data = response.json()

            assert data["limit"] == 5
            assert data["offset"] == 3
            assert data["total"] == 15

            # Should return at most 5 workflows
            workflows = data["workflows"]
            assert len(workflows) == 5

            # Check has_more is calculated correctly
            assert data["has_more"] is True  # 3 + 5 = 8 < 15

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_empty_directory(self, mock_get_dirs: Mock) -> None:
        """Test workflow discovery with no workflow files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_get_dirs.return_value = [temp_path]

            response = self.client.get("/api/workflows")

            assert response.status_code == 200
            data = response.json()

            assert data["workflows"] == []
            assert data["total"] == 0
            assert data["categories"] == []
            assert data["limit"] == 10
            assert data["offset"] == 0
            assert data["has_more"] is False

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_directory_error(self, mock_get_dirs: Mock) -> None:
        """Test workflow discovery when directory access fails."""
        # Simulate directory access error
        mock_get_dirs.side_effect = Exception("Directory access denied")

        response = self.client.get("/api/workflows")

        assert response.status_code == 500
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Failed to discover workflows"
        assert "Directory access denied" in detail["message"]
        assert detail["type"] == "Exception"

    def test_get_workflows_parameter_validation(self) -> None:
        """Test workflow endpoint parameter validation."""
        # Test invalid limit (too high)
        response = self.client.get("/api/workflows?limit=101")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = self.client.get("/api/workflows?limit=0")
        assert response.status_code == 422

        # Test invalid offset (negative)
        response = self.client.get("/api/workflows?offset=-1")
        assert response.status_code == 422

        # Test non-integer parameters
        response = self.client.get("/api/workflows?limit=abc")
        assert response.status_code == 422

        response = self.client.get("/api/workflows?offset=xyz")
        assert response.status_code == 422

        # Test search query too long
        long_search = "a" * 201  # Exceeds 200 character limit
        response = self.client.get(f"/api/workflows?search={long_search}")
        assert response.status_code == 422

        # Test invalid complexity filter
        response = self.client.get("/api/workflows?complexity=invalid")
        assert response.status_code == 422

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_search_no_results(self, mock_get_dirs: Mock) -> None:
        """Test workflow discovery when search returns no matching workflows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create workflow that won't match the search
            workflow_data = {
                "workflow_id": "climate_analysis",
                "name": "Climate Change Study",
                "description": "Environmental climate impact analysis",
                "metadata": {
                    "domain": "environmental",
                    "tags": ["climate", "environment"],
                },
                "nodes": [{"name": "analyzer", "type": "processor"}],
            }

            workflow_file = temp_path / "workflow.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            # Search for something completely unrelated
            response = self.client.get("/api/workflows?search=quantum_computing")

            assert response.status_code == 200
            data = response.json()

            assert data["workflows"] == []
            assert data["total"] == 0
            assert data["search_query"] == "quantum_computing"
            assert data["has_more"] is False

    @patch("cognivault.api.routes.workflows.logger")
    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_logging(
        self, mock_get_dirs: Mock, mock_logger: Mock
    ) -> None:
        """Test that workflow discovery logs appropriately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            workflow_data = {
                "workflow_id": "test_workflow",
                "name": "Test Workflow",
                "description": "Test workflow for logging",
                "metadata": {"domain": "test"},
                "nodes": [{"name": "test", "type": "processor"}],
            }

            workflow_file = temp_path / "workflow.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            response = self.client.get("/api/workflows?search=test&limit=5&offset=0")

            assert response.status_code == 200

            # Verify logging calls
            assert mock_logger.info.call_count >= 2

            # Check start log
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            start_logs = [
                log for log in info_calls if "Workflow discovery request" in log
            ]
            assert len(start_logs) > 0
            start_log = start_logs[0]
            assert "search='test'" in start_log
            assert "limit=5" in start_log
            assert "offset=0" in start_log

            # Check completion log
            completion_logs = [
                log for log in info_calls if "Workflow discovery completed" in log
            ]
            assert len(completion_logs) > 0

    @patch("cognivault.api.routes.workflows.logger")
    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflows_error_logging(
        self, mock_get_dirs: Mock, mock_logger: Mock
    ) -> None:
        """Test that workflow discovery errors are logged properly."""
        error_message = "File system access error"
        mock_get_dirs.side_effect = Exception(error_message)

        response = self.client.get("/api/workflows")

        assert response.status_code == 500

        # Verify error was logged
        mock_logger.error.assert_called_once()
        logged_message = mock_logger.error.call_args[0][0]
        assert "Workflows endpoint failed" in logged_message

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_workflow_metadata_extraction(self, mock_get_dirs: Mock) -> None:
        """Test the workflow metadata extraction functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create workflow with comprehensive metadata
            workflow_data = {
                "workflow_id": "comprehensive_test",
                "name": "Comprehensive Test Workflow",
                "description": "A workflow for testing comprehensive metadata extraction",
                "version": "2.1.0",
                "created_by": "Test Team",
                "metadata": {
                    "domain": "testing",
                    "complexity_level": "medium",
                    "estimated_execution_time": "15-30 seconds",
                    "tags": ["testing", "comprehensive", "metadata"],
                    "use_cases": [
                        "unit_testing",
                        "integration_testing",
                        "metadata_validation",
                    ],
                },
                "nodes": [
                    {"name": "setup", "type": "processor"},
                    {"name": "execute", "type": "processor"},
                    {"name": "validate", "type": "validator"},
                    {"name": "cleanup", "type": "processor"},
                ],
            }

            workflow_file = temp_path / "comprehensive.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            response = self.client.get("/api/workflows")

            assert response.status_code == 200
            data = response.json()

            workflows = data["workflows"]
            assert len(workflows) == 1

            workflow = workflows[0]

            # Verify all metadata was extracted correctly
            assert workflow["workflow_id"] == "comprehensive_test"
            assert workflow["name"] == "Comprehensive Test Workflow"
            assert workflow["version"] == "2.1.0"
            assert workflow["category"] == "testing"
            assert workflow["complexity_level"] == "medium"
            assert workflow["estimated_execution_time"] == "15-30 seconds"
            assert workflow["node_count"] == 4
            assert "testing" in workflow["tags"]
            assert "comprehensive" in workflow["tags"]
            assert "unit_testing" in workflow["use_cases"]

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_workflow_deduplication(self, mock_get_dirs: Mock) -> None:
        """Test that duplicate workflows are properly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create two workflows with the same ID but different versions
            workflow_data_v1 = {
                "workflow_id": "duplicate_test",
                "name": "Duplicate Test v1",
                "version": "1.0.0",
                "metadata": {"domain": "test"},
                "nodes": [{"name": "processor", "type": "processor"}],
            }

            workflow_data_v2 = {
                "workflow_id": "duplicate_test",
                "name": "Duplicate Test v2",
                "version": "2.0.0",
                "metadata": {"domain": "test"},
                "nodes": [{"name": "processor", "type": "processor"}],
            }

            # Write both files (v2 should be newer)
            v1_file = temp_path / "duplicate_v1.yaml"
            with open(v1_file, "w") as f:
                yaml.dump(workflow_data_v1, f)

            # Sleep briefly to ensure different timestamps
            import time

            time.sleep(0.1)

            v2_file = temp_path / "duplicate_v2.yaml"
            with open(v2_file, "w") as f:
                yaml.dump(workflow_data_v2, f)

            mock_get_dirs.return_value = [temp_path]

            response = self.client.get("/api/workflows")

            assert response.status_code == 200
            data = response.json()

            workflows = data["workflows"]
            # Should only have one workflow after deduplication
            assert len(workflows) == 1

            workflow = workflows[0]
            # Should keep the newer version (v2)
            assert workflow["name"] == "Duplicate Test v2"
            assert workflow["version"] == "2.0.0"

    def test_workflows_endpoint_response_schema(self) -> None:
        """Test that workflows endpoint responses match expected schema."""
        # Test with invalid request to get 422 response
        response = self.client.get("/api/workflows?limit=invalid")

        assert response.status_code == 422
        data = response.json()

        # Verify FastAPI validation error format
        assert "detail" in data
        assert isinstance(data["detail"], list)  # FastAPI validation errors are lists


class TestWorkflowByIdRoutes:
    """Test suite for individual workflow retrieval endpoints."""

    def setup_method(self) -> None:
        """Set up test client for each test."""
        self.client = TestClient(app)
        # Clear cache between tests to ensure test isolation
        from cognivault.api.routes.workflows import workflow_service

        workflow_service._workflow_cache = {}
        workflow_service._cache_timestamp = 0.0

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflow_by_id_success(self, mock_get_dirs: Mock) -> None:
        """Test successful workflow retrieval by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            workflow_data = {
                "workflow_id": "test_workflow_123",
                "name": "Test Workflow for ID Retrieval",
                "description": "A comprehensive test workflow for ID-based retrieval",
                "version": "1.5.0",
                "created_by": "API Test Team",
                "metadata": {
                    "domain": "testing",
                    "complexity_level": "high",
                    "estimated_execution_time": "30-45 seconds",
                    "tags": ["testing", "api", "retrieval"],
                    "use_cases": ["api_testing", "workflow_validation"],
                },
                "nodes": [
                    {"name": "setup", "type": "processor"},
                    {"name": "execute", "type": "processor"},
                    {"name": "validate", "type": "validator"},
                ],
            }

            workflow_file = temp_path / "test_workflow.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            # Get workflow by ID
            response = self.client.get("/api/workflows/test_workflow_123")

            assert response.status_code == 200
            data = response.json()

            # Verify all workflow metadata is returned
            assert data["workflow_id"] == "test_workflow_123"
            assert data["name"] == "Test Workflow for ID Retrieval"
            assert (
                data["description"]
                == "A comprehensive test workflow for ID-based retrieval"
            )
            assert data["version"] == "1.5.0"
            assert data["category"] == "testing"
            assert data["complexity_level"] == "high"
            assert data["estimated_execution_time"] == "30-45 seconds"
            assert data["node_count"] == 3
            assert "testing" in data["tags"]
            assert "api_testing" in data["use_cases"]

    def test_get_workflow_by_id_invalid_format(self) -> None:
        """Test workflow retrieval with invalid ID format."""
        # Test IDs that reach our validation logic (no slashes)
        invalid_ids_422 = [
            "invalid@id",  # Contains invalid character
            "spaces in id",  # Contains spaces
        ]

        for invalid_id in invalid_ids_422:
            response = self.client.get(f"/api/workflows/{invalid_id}")
            assert response.status_code == 422
            data = response.json()

            assert "detail" in data
            detail = data["detail"]
            assert detail["error"] == "Invalid workflow ID format"
            assert detail["workflow_id"] == invalid_id

        # Test IDs with slashes - FastAPI returns 404 (route not found)
        invalid_ids_404 = [
            "id with/slash",  # Contains slash - FastAPI routing issue
        ]

        for invalid_id in invalid_ids_404:
            response = self.client.get(f"/api/workflows/{invalid_id}")
            assert response.status_code == 404  # FastAPI routing 404

        # Test empty string separately - it hits different route
        response = self.client.get("/api/workflows/")  # Trailing slash for empty ID
        # This should either be 404 (route not found) or redirect to /api/workflows
        assert response.status_code in [404, 200, 307]  # 307 is redirect

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflow_by_id_not_found(self, mock_get_dirs: Mock) -> None:
        """Test workflow retrieval when workflow ID is not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_get_dirs.return_value = [temp_path]

            nonexistent_id = "nonexistent_workflow_id"
            response = self.client.get(f"/api/workflows/{nonexistent_id}")

            assert response.status_code == 404
            data = response.json()

            assert "detail" in data
            detail = data["detail"]
            assert detail["error"] == "Workflow not found"
            assert detail["workflow_id"] == nonexistent_id

    @patch("cognivault.api.routes.workflows.logger")
    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflow_by_id_logging(
        self, mock_get_dirs: Mock, mock_logger: Mock
    ) -> None:
        """Test that workflow retrieval logs appropriately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            workflow_data = {
                "workflow_id": "logging_test_workflow",
                "name": "Logging Test",
                "description": "Test workflow for logging validation",
                "metadata": {"domain": "test"},
                "nodes": [{"name": "processor", "type": "processor"}],
            }

            workflow_file = temp_path / "logging_test.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            response = self.client.get("/api/workflows/logging_test_workflow")

            assert response.status_code == 200

            # Verify logging calls
            assert mock_logger.info.call_count >= 2

            # Check start log
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            start_logs = [
                log for log in info_calls if "Retrieving workflow metadata" in log
            ]
            assert len(start_logs) > 0

            # Check completion log
            completion_logs = [
                log for log in info_calls if "Workflow metadata retrieved" in log
            ]
            assert len(completion_logs) > 0

    @patch("cognivault.api.routes.workflows.logger")
    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_get_workflow_by_id_error_handling(
        self, mock_get_dirs: Mock, mock_logger: Mock
    ) -> None:
        """Test workflow retrieval error handling and logging."""
        # Simulate service error
        mock_get_dirs.side_effect = Exception("Service unavailable")

        response = self.client.get("/api/workflows/test_id")

        assert response.status_code == 500
        data = response.json()

        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Failed to retrieve workflow"
        assert detail["type"] == "Exception"
        assert detail["workflow_id"] == "test_id"

        # Verify error was logged
        mock_logger.error.assert_called_once()
        logged_message = mock_logger.error.call_args[0][0]
        assert "Workflow retrieval failed" in logged_message

    @patch("cognivault.api.routes.workflows.workflow_service._get_workflow_directories")
    def test_workflow_id_validation_edge_cases(self, mock_get_dirs: Mock) -> None:
        """Test workflow ID validation with various edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create workflow with valid ID
            workflow_data = {
                "workflow_id": "valid_workflow-123",
                "name": "Valid Workflow",
                "description": "Test workflow with valid ID format",
                "metadata": {"domain": "test"},
                "nodes": [{"name": "processor", "type": "processor"}],
            }

            workflow_file = temp_path / "valid.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)

            mock_get_dirs.return_value = [temp_path]

            # Test valid IDs
            valid_ids = [
                "valid_workflow-123",
                "simple",
                "workflow_with_underscores",
                "workflow-with-hyphens",
                "WorkflowWithNumbers123",
            ]

            for valid_id in valid_ids:
                # Update the workflow data to match the ID we're testing
                workflow_data["workflow_id"] = valid_id
                with open(workflow_file, "w") as f:
                    yaml.dump(workflow_data, f)

                response = self.client.get(f"/api/workflows/{valid_id}")
                # Should either return 200 (found) or 404 (valid format but not found)
                assert response.status_code in [200, 404]
                if response.status_code == 404:
                    # Verify it's a proper 404, not a 422 format error
                    data = response.json()
                    assert data["detail"]["error"] == "Workflow not found"
