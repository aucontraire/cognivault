"""
Unit tests for FastAPI topics route endpoints.

Tests the topic discovery and management endpoints using mock data and services.
"""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from cognivault.api.main import app
from cognivault.api.models import TopicSummary, TopicsResponse, TopicWikiResponse


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

        # Test search for machine-related topics
        response = self.client.get("/api/topics?search=machine")

        assert response.status_code == 200
        data = response.json()

        assert data["search_query"] == "machine"

        # Should find topics related to machine
        topics = data["topics"]
        assert len(topics) > 0

        # Check that returned topics are relevant to search
        topic_texts = [t["name"] + " " + t["description"] for t in topics]
        machine_topics = [text for text in topic_texts if "machine" in text.lower()]
        assert len(machine_topics) > 0

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


class TestTopicWikiRoutes:
    """Test suite for topic wiki knowledge retrieval endpoints."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topic_wiki_success(self, mock_get_api):
        """Test successful topic wiki knowledge retrieval."""
        # Setup mock orchestration API
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
                "query": "Explain machine learning algorithms for beginners",
                "start_time": 1703097550.0,
                "execution_time": 15.2,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # First, we need to get a topic ID by calling the topics endpoint
        topics_response = self.client.get("/api/topics")
        assert topics_response.status_code == 200
        topics_data = topics_response.json()

        # Should have at least one topic
        assert len(topics_data["topics"]) > 0
        topic_id = topics_data["topics"][0]["topic_id"]

        # Now test the wiki endpoint
        response = self.client.get(f"/api/topics/{topic_id}/wiki")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches TopicWikiResponse
        assert "topic_id" in data
        assert "topic_name" in data
        assert "content" in data
        assert "last_updated" in data
        assert "sources" in data
        assert "query_count" in data
        assert "confidence_score" in data

        # Verify field types and constraints
        assert data["topic_id"] == topic_id
        assert isinstance(data["topic_name"], str)
        assert len(data["topic_name"]) > 0
        assert isinstance(data["content"], str)
        assert len(data["content"]) > 0
        assert isinstance(data["last_updated"], float)
        assert isinstance(data["sources"], list)
        assert isinstance(data["query_count"], int)
        assert data["query_count"] >= 0
        assert isinstance(data["confidence_score"], float)
        assert 0.0 <= data["confidence_score"] <= 1.0

        # Verify sources are valid UUIDs if present
        for source_id in data["sources"]:
            assert isinstance(source_id, str)
            assert len(source_id) == 36  # UUID format

    def test_get_topic_wiki_invalid_uuid_format(self):
        """Test topic wiki with invalid UUID format."""
        invalid_topic_id = "invalid-uuid-format"
        response = self.client.get(f"/api/topics/{invalid_topic_id}/wiki")

        assert response.status_code == 422
        data = response.json()

        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Invalid topic ID format"
        assert detail["topic_id"] == invalid_topic_id

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topic_wiki_topic_not_found(self, mock_get_api):
        """Test topic wiki when topic ID is not found."""
        # Setup mock with empty history so no topics are discovered
        mock_api = Mock()
        mock_api.get_workflow_history.return_value = []
        mock_get_api.return_value = mock_api

        # Use a valid UUID format but non-existent topic
        nonexistent_topic_id = "550e8400-e29b-41d4-a716-446655440999"
        response = self.client.get(f"/api/topics/{nonexistent_topic_id}/wiki")

        assert response.status_code == 404
        data = response.json()

        assert "detail" in data
        detail = data["detail"]
        assert detail["error"] == "Topic not found"
        assert detail["topic_id"] == nonexistent_topic_id

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topic_wiki_with_minimal_data(self, mock_get_api):
        """Test topic wiki with minimal workflow data."""
        # Setup mock with one simple workflow
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "Simple test query",
                "start_time": 1703097600.0,
                "execution_time": 5.0,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # Get a topic first
        topics_response = self.client.get("/api/topics")
        assert topics_response.status_code == 200
        topics_data = topics_response.json()

        if len(topics_data["topics"]) > 0:
            topic_id = topics_data["topics"][0]["topic_id"]

            # Test wiki endpoint
            response = self.client.get(f"/api/topics/{topic_id}/wiki")

            assert response.status_code == 200
            data = response.json()

            # Should have basic content even with minimal data
            assert len(data["content"]) > 0
            assert data["confidence_score"] > 0.0
            assert data["query_count"] >= 0

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topic_wiki_with_multiple_related_workflows(self, mock_get_api):
        """Test topic wiki with multiple related workflows."""
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
                "status": "completed",
                "query": "How does machine learning work?",
                "start_time": 1703097550.0,
                "execution_time": 15.2,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440003",
                "status": "completed",
                "query": "Machine learning algorithms explained",
                "start_time": 1703097500.0,
                "execution_time": 18.7,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440004",
                "status": "completed",
                "query": "Unrelated topic about cooking",  # Should not match
                "start_time": 1703097450.0,
                "execution_time": 10.0,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # Get topics first
        topics_response = self.client.get("/api/topics")
        assert topics_response.status_code == 200
        topics_data = topics_response.json()

        # Find machine learning topic
        ml_topic = None
        for topic in topics_data["topics"]:
            if (
                "machine" in topic["name"].lower()
                or "learning" in topic["name"].lower()
            ):
                ml_topic = topic
                break

        if ml_topic:
            # Test wiki for machine learning topic
            response = self.client.get(f"/api/topics/{ml_topic['topic_id']}/wiki")

            assert response.status_code == 200
            data = response.json()

            # Should have higher confidence with multiple related workflows
            assert data["confidence_score"] > 0.2
            assert len(data["sources"]) >= 2  # Should have multiple sources
            assert (
                "machine" in data["content"].lower()
                or "learning" in data["content"].lower()
            )

    @patch("cognivault.api.routes.topics.topic_service.find_topic_by_id")
    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topic_wiki_orchestration_failure(self, mock_get_api, mock_find_topic):
        """Test topic wiki when orchestration API fails during knowledge synthesis."""
        # Create a mock topic to return
        from cognivault.api.models import TopicSummary

        mock_topic = TopicSummary(
            topic_id="550e8400-e29b-41d4-a716-446655440001",
            name="Machine Learning",
            description="Test topic for failure scenario",
            query_count=1,
            last_updated=1703097600.0,
            similarity_score=1.0,
        )
        mock_find_topic.return_value = mock_topic

        # Make orchestration API fail during synthesis
        mock_get_api.side_effect = Exception("Orchestration API unavailable")

        # Test wiki endpoint - should return fallback content due to exception
        response = self.client.get(f"/api/topics/{mock_topic.topic_id}/wiki")

        assert response.status_code == 200  # Fallback content should be provided
        data = response.json()

        # Should have fallback content
        assert len(data["content"]) > 0
        assert data["confidence_score"] == 0.5  # Default fallback confidence
        assert "This topic represents" in data["content"]  # Should be fallback content
        assert data["topic_id"] == mock_topic.topic_id
        assert data["topic_name"] == mock_topic.name

    @patch("cognivault.api.routes.topics.logger")
    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_get_topic_wiki_logging(self, mock_get_api, mock_logger):
        """Test that topic wiki retrieval logs appropriately."""
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

        # Get a topic first
        topics_response = self.client.get("/api/topics")
        if topics_response.status_code == 200:
            topics_data = topics_response.json()
            if len(topics_data["topics"]) > 0:
                topic_id = topics_data["topics"][0]["topic_id"]

                response = self.client.get(f"/api/topics/{topic_id}/wiki")

                assert response.status_code == 200

                # Verify logging calls
                assert mock_logger.info.call_count >= 2

                # Check for start log
                info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                start_logs = [
                    log for log in info_calls if "Retrieving topic wiki" in log
                ]
                assert len(start_logs) > 0

                # Check for completion log
                completion_logs = [
                    log for log in info_calls if "Topic wiki retrieved" in log
                ]
                assert len(completion_logs) > 0

    def test_get_topic_wiki_edge_cases(self):
        """Test topic wiki endpoint with edge case inputs."""
        # Test with various invalid formats
        invalid_ids = [
            "123",  # Too short
            "not-a-uuid-at-all",  # Invalid format
            "550e8400-e29b-41d4-a716-446655440000-extra",  # Too long
            "550e8400-e29b-41d4-a716-44665544000g",  # Invalid character
        ]

        for invalid_id in invalid_ids:
            response = self.client.get(f"/api/topics/{invalid_id}/wiki")
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
            assert data["detail"]["error"] == "Invalid topic ID format"

        # Test empty string separately - FastAPI treats this as a different route
        response = self.client.get("/api/topics//wiki")  # Double slash for empty
        assert response.status_code == 404  # FastAPI route not found

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_topic_wiki_content_synthesis_quality(self, mock_get_api):
        """Test the quality and structure of synthesized wiki content."""
        mock_api = Mock()
        mock_history = [
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "completed",
                "query": "What are the fundamentals of machine learning?",
                "start_time": 1703097600.0,
                "execution_time": 12.5,
            },
            {
                "workflow_id": "550e8400-e29b-41d4-a716-446655440002",
                "status": "completed",
                "query": "How do machine learning algorithms work?",
                "start_time": 1703097550.0,
                "execution_time": 15.2,
            },
        ]
        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # Get topics and find ML topic
        topics_response = self.client.get("/api/topics")
        assert topics_response.status_code == 200
        topics_data = topics_response.json()

        ml_topic = None
        for topic in topics_data["topics"]:
            if (
                "machine" in topic["name"].lower()
                or "learning" in topic["name"].lower()
            ):
                ml_topic = topic
                break

        if ml_topic:
            response = self.client.get(f"/api/topics/{ml_topic['topic_id']}/wiki")

            assert response.status_code == 200
            data = response.json()

            content = data["content"]

            # Content quality checks
            assert len(content) > 100  # Should be substantial
            assert len(content.split()) > 20  # Should have multiple words
            assert "\n" in content  # Should have structure (paragraphs)

            # Should contain topic-relevant keywords
            content_lower = content.lower()
            assert any(
                keyword in content_lower
                for keyword in ["machine", "learning", "analysis"]
            )

    @patch("cognivault.api.routes.topics.get_orchestration_api")
    def test_topic_wiki_source_tracking(self, mock_get_api):
        """Test that topic wiki properly tracks source workflows."""
        mock_api = Mock()

        # Create workflows with distinct IDs
        workflow_ids = [
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400-e29b-41d4-a716-446655440002",
            "550e8400-e29b-41d4-a716-446655440003",
        ]

        mock_history = []
        for i, wf_id in enumerate(workflow_ids):
            mock_history.append(
                {
                    "workflow_id": wf_id,
                    "status": "completed",
                    "query": f"Machine learning query {i + 1}",
                    "start_time": 1703097600.0 + i,
                    "execution_time": 10.0 + i,
                }
            )

        mock_api.get_workflow_history.return_value = mock_history
        mock_get_api.return_value = mock_api

        # Get topics and find ML topic
        topics_response = self.client.get("/api/topics")
        assert topics_response.status_code == 200
        topics_data = topics_response.json()

        ml_topic = None
        for topic in topics_data["topics"]:
            if (
                "machine" in topic["name"].lower()
                or "learning" in topic["name"].lower()
            ):
                ml_topic = topic
                break

        if ml_topic:
            response = self.client.get(f"/api/topics/{ml_topic['topic_id']}/wiki")

            assert response.status_code == 200
            data = response.json()

            # Should track source workflows
            sources = data["sources"]
            assert len(sources) > 0
            assert len(sources) <= len(workflow_ids)

            # All sources should be valid UUIDs
            for source in sources:
                assert len(source) == 36
                assert source in workflow_ids
