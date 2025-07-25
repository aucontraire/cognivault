"""
Unit tests for FastAPI topics route endpoints.

Tests the topic discovery and management endpoints using mock data and services.
"""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.models import TopicSummary, TopicsResponse


class TestTopicsRoutes:
    """Test suite for topics discovery endpoints."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_success(self, mock_get_api):
        """Test successful topics discovery with mock workflow history."""
        # Setup mock orchestration API with workflow history
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "What is machine learning and how does it work?",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "completed",
                "query": "Explain deep learning neural networks",
                "start_time": 1703097550.0,
                "execution_time": 15.2,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "completed",
                "query": "Climate change impact on agriculture",
                "start_time": 1703097500.0,
                "execution_time": 18.7,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/topics")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches TopicsResponse
        assert "topics" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data
        assert "search_query" in data

        # Verify default pagination
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert data["search_query"] is None

        # Verify topics were discovered
        topics = data["topics"]
        assert len(topics) > 0
        assert data["total"] == len(topics)

        # Verify topic structure
        for topic in topics:
            assert "topic_id" in topic
            assert "name" in topic
            assert "description" in topic
            assert "query_count" in topic
            assert "last_updated" in topic

            # Verify topic field types and formats
            assert isinstance(topic["topic_id"], str)
            assert len(topic["topic_id"]) == 36  # UUID format
            assert isinstance(topic["name"], str)
            assert len(topic["name"]) > 0
            assert isinstance(topic["description"], str)
            assert isinstance(topic["query_count"], int)
            assert topic["query_count"] > 0
            assert isinstance(topic["last_updated"], float)

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_with_search(self, mock_get_api):
        """Test topics discovery with search filtering."""
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Machine learning algorithms and applications",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "completed",
                "query": "Deep learning for computer vision",
                "start_time": 1703097550.0,
                "execution_time": 15.2,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "completed",
                "query": "Climate change and environmental policy",
                "start_time": 1703097500.0,
                "execution_time": 18.7,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # Test search for machine learning topics
        response = self.client.get("/api/topics?search=learning")

        assert response.status_code == 200
        data = response.json()

        assert data["search_query"] == "learning"

        # Should find topics related to learning
        topics = data["topics"]
        assert len(topics) > 0

        # Check that returned topics are relevant to search
        topic_texts = [t["name"] + " " + t["description"] for t in topics]
        learning_topics = [text for text in topic_texts if "learning" in text.lower()]
        assert len(learning_topics) > 0

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_with_pagination(self, mock_get_api):
        """Test topics discovery with custom pagination parameters."""
        mock_api = Mock()
        # Create more mock data to test pagination
        mock_history = []
        queries = [
            "Machine learning fundamentals",
            "Deep learning networks",
            "Neural network architectures",
            "Computer vision applications",
            "Natural language processing",
            "Reinforcement learning algorithms",
            "Data science methodologies",
            "Statistical analysis techniques",
            "Predictive modeling approaches",
            "Artificial intelligence ethics",
        ]

        for i, query in enumerate(queries):
            mock_history.append(
                {
                    "workflow_id": f"550e8400-e29b-41d4-a716-44665544{i:04d}",
                    "status": "completed",
                    "query": query,
                    "start_time": 1703097600.0 + i,
                    "execution_time": 10.0 + i,
                }
            )

        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # Test custom pagination
        response = self.client.get("/api/topics?limit=3&offset=2")

        assert response.status_code == 200
        data = response.json()

        assert data["limit"] == 3
        assert data["offset"] == 2

        # Should return at most 3 topics
        topics = data["topics"]
        assert len(topics) <= 3

        # Check has_more is calculated correctly
        expected_has_more = (data["offset"] + len(topics)) < data["total"]
        assert data["has_more"] == expected_has_more

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_empty_history(self, mock_get_api):
        """Test topics discovery with no workflow history."""
        mock_api = Mock()
        mock_api.get_workflow_history.return_value = []
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/topics")

        assert response.status_code == 200
        data = response.json()

        assert data["topics"] == []
        assert data["total"] == 0
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert data["has_more"] is False

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_orchestration_failure(self, mock_get_api):
        """Test topics discovery when orchestration API fails."""
        mock_get_api.side_effect = Exception("Orchestration API unavailable")

        response = self.client.get("/api/topics")

        assert response.status_code == 500
        data = response.json()

        # Verify error response structure
        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Failed to discover topics"
        assert "Orchestration API unavailable" in detail["message"]
        assert detail["type"] == "Exception"

    def test_get_topics_parameter_validation(self):
        """Test topics endpoint parameter validation."""
        # Test invalid limit (too high)
        response = self.client.get("/api/topics?limit=101")
        assert response.status_code == 422

        # Test invalid limit (too low)
        response = self.client.get("/api/topics?limit=0")
        assert response.status_code == 422

        # Test invalid offset (negative)
        response = self.client.get("/api/topics?offset=-1")
        assert response.status_code == 422

        # Test non-integer parameters
        response = self.client.get("/api/topics?limit=abc")
        assert response.status_code == 422

        response = self.client.get("/api/topics?offset=xyz")
        assert response.status_code == 422

        # Test search query too long
        long_search = "a" * 201  # Exceeds 200 character limit
        response = self.client.get(f"/api/topics?search={long_search}")
        assert response.status_code == 422

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_search_no_results(self, mock_get_api):
        """Test topics discovery when search returns no matching topics."""
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Climate change impact on agriculture",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # Search for something completely unrelated
        response = self.client.get("/api/topics?search=quantum_physics")

        assert response.status_code == 200
        data = response.json()

        assert data["topics"] == []
        assert data["total"] == 0
        assert data["search_query"] == "quantum_physics"
        assert data["has_more"] is False

    @patch("cognivault.api.routes.topics.logger")
    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_logging(self, mock_get_api, mock_logger):
        """Test that topics discovery logs appropriately."""
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Test query for logging",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/topics?search=test&limit=5&offset=0")

        assert response.status_code == 200

        # Verify logging calls
        assert mock_logger.info.call_count >= 2

        # Check start log
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        start_logs = [log for log in info_calls if "Topic discovery request" in log]
        assert len(start_logs) > 0
        start_log = start_logs[0]
        assert "search='test'" in start_log
        assert "limit=5" in start_log
        assert "offset=0" in start_log

        # Check completion log
        completion_logs = [
            log for log in info_calls if "Topic discovery completed" in log
        ]
        assert len(completion_logs) > 0

    @patch("cognivault.api.routes.topics.logger")
    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_error_logging(self, mock_get_api, mock_logger):
        """Test that topics discovery errors are logged properly."""
        error_message = "Database connection timeout"
        mock_get_api.side_effect = Exception(error_message)

        response = self.client.get("/api/topics")

        assert response.status_code == 500

        # Verify error was logged
        mock_logger.error.assert_called_once()
        logged_message = mock_logger.error.call_args[0][0]
        assert "Topics endpoint failed" in logged_message

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_topic_discovery_service_keyword_extraction(self, mock_get_api):
        """Test the keyword extraction functionality."""
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "What are the main benefits of machine learning algorithms for data analysis?",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/topics")

        assert response.status_code == 200
        data = response.json()

        topics = data["topics"]
        assert len(topics) > 0

        # Check that topic names contain meaningful keywords
        topic_names = [topic["name"] for topic in topics]
        # Should contain words like "Machine", "Learning", "Data", "Analysis"
        combined_names = " ".join(topic_names).lower()
        meaningful_words = ["machine", "learning", "data", "analysis", "algorithms"]
        found_words = [word for word in meaningful_words if word in combined_names]
        assert len(found_words) > 0

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_topic_discovery_query_grouping(self, mock_get_api):
        """Test that similar queries are grouped into the same topic."""
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Machine learning algorithms for classification",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "completed",
                "query": "Classification algorithms in machine learning",
                "start_time": 1703097550.0,
                "execution_time": 15.2,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "completed",
                "query": "Climate change effects on polar ice",
                "start_time": 1703097500.0,
                "execution_time": 18.7,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/topics")

        assert response.status_code == 200
        data = response.json()

        topics = data["topics"]
        assert len(topics) >= 1

        # Find the machine learning topic
        ml_topics = [
            t
            for t in topics
            if "machine" in t["name"].lower() or "learning" in t["name"].lower()
        ]
        if ml_topics:
            ml_topic = ml_topics[0]
            # Should have grouped the two similar ML queries
            assert ml_topic["query_count"] >= 2

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_topic_discovery_service_error_handling(self, mock_get_api):
        """Test that topic discovery service handles errors gracefully."""
        # Simulate orchestration API returning malformed data
        mock_api = Mock()
        mock_api.get_workflow_history.side_effect = RuntimeError("Database error")
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/topics")

        # Should return 500 error with proper error details
        assert response.status_code == 500
        data = response.json()

        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Failed to discover topics"
        assert detail["type"] == "RuntimeError"

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topics_edge_cases(self, mock_get_api):
        """Test topics discovery with edge case scenarios."""
        mock_api = Mock()

        # Test with queries that have minimal keywords
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "What?",  # Very short query
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "completed",
                "query": "The the the a an and or",  # Only stop words
                "start_time": 1703097550.0,
                "execution_time": 15.2,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "completed",
                "query": "",  # Empty query
                "start_time": 1703097500.0,
                "execution_time": 18.7,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        response = self.client.get("/api/topics")

        assert response.status_code == 200
        data = response.json()

        # Should handle edge cases gracefully and still return valid response
        assert "topics" in data
        assert isinstance(data["topics"], list)
        assert data["total"] >= 0

    def test_topics_endpoint_response_schema(self):
        """Test that topics endpoint responses match expected schema."""
        # Test with invalid request to get 422 response
        response = self.client.get("/api/topics?limit=invalid")

        assert response.status_code == 422
        data = response.json()

        # Verify FastAPI validation error format
        assert "detail" in data
        assert isinstance(data["detail"], list)  # FastAPI validation errors are lists
