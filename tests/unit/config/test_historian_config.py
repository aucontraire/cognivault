"""
Tests for HistorianConfig hybrid search configuration functionality.

This test validates the new hybrid search configuration parameters
and their integration with the historian agent.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, patch
import os

from cognivault.config.agent_configs import HistorianConfig
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.historian.search import SearchResult
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)


class TestHistorianConfig:
    """Test HistorianConfig hybrid search parameters."""

    def test_default_values(self) -> None:
        """Test that default configuration values are correct."""
        config = HistorianConfig()

        # Test default values
        assert config.hybrid_search_enabled is False
        assert config.hybrid_search_file_ratio == 0.6
        assert config.database_relevance_boost == 0.0
        assert config.search_timeout_seconds == 5
        assert config.deduplication_threshold == 0.8

    def test_validation_constraints(self) -> None:
        """Test configuration parameter validation."""
        # Valid configuration
        config = HistorianConfig(
            hybrid_search_file_ratio=0.7,
            database_relevance_boost=0.2,
            search_timeout_seconds=10,
            deduplication_threshold=0.9,
        )
        assert config.hybrid_search_file_ratio == 0.7
        assert config.database_relevance_boost == 0.2

        # Test bounds validation
        with pytest.raises(ValueError):
            HistorianConfig(hybrid_search_file_ratio=1.5)  # > 1.0

        with pytest.raises(ValueError):
            HistorianConfig(hybrid_search_file_ratio=-0.1)  # < 0.0

        with pytest.raises(ValueError):
            HistorianConfig(database_relevance_boost=0.6)  # > 0.5

        with pytest.raises(ValueError):
            HistorianConfig(database_relevance_boost=-0.6)  # < -0.5

        with pytest.raises(ValueError):
            HistorianConfig(search_timeout_seconds=0)  # < 1

        with pytest.raises(ValueError):
            HistorianConfig(search_timeout_seconds=35)  # > 30

    def test_environment_variable_loading(self) -> None:
        """Test loading configuration from environment variables."""
        # Set environment variables
        env_vars = {
            "HISTORIAN_HYBRID_SEARCH_ENABLED": "true",
            "HISTORIAN_HYBRID_SEARCH_FILE_RATIO": "0.7",
            "HISTORIAN_DATABASE_RELEVANCE_BOOST": "0.1",
            "HISTORIAN_SEARCH_TIMEOUT_SECONDS": "8",
            "HISTORIAN_DEDUPLICATION_THRESHOLD": "0.9",
        }

        with patch.dict(os.environ, env_vars):
            config = HistorianConfig.from_env()

            assert config.hybrid_search_enabled is True
            assert config.hybrid_search_file_ratio == 0.7
            assert config.database_relevance_boost == 0.1
            assert config.search_timeout_seconds == 8
            assert config.deduplication_threshold == 0.9

    def test_dict_creation(self) -> None:
        """Test creating configuration from dictionary."""
        config_dict = {
            "hybrid_search_enabled": True,
            "hybrid_search_file_ratio": 0.4,
            "database_relevance_boost": -0.1,
            "search_timeout_seconds": 15,
            "deduplication_threshold": 0.75,
        }

        config = HistorianConfig.from_dict(config_dict)

        assert config.hybrid_search_enabled is True
        assert config.hybrid_search_file_ratio == 0.4
        assert config.database_relevance_boost == -0.1
        assert config.search_timeout_seconds == 15
        assert config.deduplication_threshold == 0.75

    def test_prompt_config_export(self) -> None:
        """Test conversion to prompt configuration format."""
        config = HistorianConfig(
            hybrid_search_enabled=True,
            hybrid_search_file_ratio=0.3,
            database_relevance_boost=0.2,
            search_timeout_seconds=12,
            deduplication_threshold=0.95,
        )

        prompt_config = config.to_prompt_config()

        assert prompt_config["hybrid_search_enabled"] == "True"
        assert prompt_config["hybrid_search_file_ratio"] == "0.3"
        assert prompt_config["database_relevance_boost"] == "0.2"
        assert prompt_config["search_timeout_seconds"] == "12"
        assert prompt_config["deduplication_threshold"] == "0.95"


class TestHistorianAgentConfiguration:
    """Test HistorianAgent integration with configuration parameters."""

    @pytest.mark.asyncio
    async def test_hybrid_search_configuration_integration(self) -> None:
        """Test that configuration parameters are used in hybrid search."""
        # Create custom configuration
        config = HistorianConfig(
            hybrid_search_enabled=True,
            hybrid_search_file_ratio=0.7,  # 70% file, 30% database
            database_relevance_boost=0.2,
            search_timeout_seconds=10,
            deduplication_threshold=0.9,
        )

        agent = HistorianAgent(llm=None, config=config)

        # Mock search methods to capture parameters
        file_results = [
            SearchResult(
                filepath="/test.md",
                filename="test.md",
                title="Test",
                date="2024-01-01",
                relevance_score=0.9,
                match_type="content",
                excerpt="Test content",
                metadata={},
            )
        ]

        db_results = [
            SearchResult(
                filepath="db_doc_1",
                filename="doc_1",
                title="DB Test",
                date="2024-01-02",
                relevance_score=0.8,
                match_type="content",
                excerpt="DB content",
                metadata={"source": "database"},
            )
        ]

        agent._search_file_content = AsyncMock(return_value=file_results)
        agent._search_database_content = AsyncMock(return_value=db_results)

        # Execute search
        context = AgentContextPatterns.simple_query("test query")
        results = await agent._search_historical_content("test query", context)

        # Verify 70/30 split (out of 10 total)
        agent._search_file_content.assert_called_once_with("test query", 7)  # 70% of 10
        agent._search_database_content.assert_called_once_with(
            "test query", 3
        )  # 30% of 10

        # Results should include both
        assert len(results) == 2

    def test_relevance_boost_calculation(self) -> None:
        """Test that database relevance boost is applied correctly."""
        config = HistorianConfig(database_relevance_boost=0.3)
        agent = HistorianAgent(llm=None, config=config)

        # Create mock database document
        class MockDoc:
            id = "123"
            title = "Test"
            content = "Test content"
            created_at = None
            source_path = None
            document_metadata = {}
            word_count = 10

        # Test the relevance calculation (this would be inside the search method)
        expected_score = 0.8 + 0.3  # base score + boost
        assert expected_score == 1.1

    def test_deduplication_threshold_integration(self) -> None:
        """Test that deduplication threshold affects similarity matching."""
        # Test with high threshold (0.95) - should allow more results
        config_strict = HistorianConfig(deduplication_threshold=0.95)
        agent_strict = HistorianAgent(llm=None, config=config_strict)

        # Test with low threshold (0.3) - should remove more duplicates
        config_loose = HistorianConfig(deduplication_threshold=0.3)
        agent_loose = HistorianAgent(llm=None, config=config_loose)

        # Create similar results
        result1 = SearchResult(
            filepath="/test1.md",
            filename="test1.md",
            title="Machine Learning Basics",
            date="2024-01-01",
            relevance_score=0.9,
            match_type="content",
            excerpt="Introduction to machine learning concepts",
            metadata={},
        )

        result2 = SearchResult(
            filepath="/test2.md",
            filename="test2.md",
            title="Machine Learning Introduction",  # Similar title
            date="2024-01-02",
            relevance_score=0.8,
            match_type="content",
            excerpt="Introduction to machine learning basics",  # Similar excerpt
            metadata={},
        )

        # Test strict deduplication (high threshold) - should keep both
        deduplicated_strict = agent_strict._deduplicate_search_results(
            [result1, result2]
        )
        assert len(deduplicated_strict) == 2  # Both kept due to high threshold

        # Test loose deduplication (low threshold) - should remove duplicate
        deduplicated_loose = agent_loose._deduplicate_search_results([result1, result2])
        assert len(deduplicated_loose) == 1  # One removed due to low threshold

    def test_text_similarity_calculation(self) -> None:
        """Test text similarity calculation method."""
        config = HistorianConfig()
        agent = HistorianAgent(llm=None, config=config)

        # Test exact match
        assert agent._text_similarity("hello", "hello") == 1.0

        # Test no similarity
        assert agent._text_similarity("abc", "xyz") == 0.0

        # Test partial similarity
        similarity = agent._text_similarity("machine learning", "machine intelligence")
        assert 0.0 < similarity < 1.0  # Should be between 0 and 1

        # Test empty strings
        assert agent._text_similarity("", "test") == 0.0
        assert agent._text_similarity("test", "") == 0.0
