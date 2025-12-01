"""
Integration test for historian agent hybrid search functionality.

This test verifies that the historian agent can successfully integrate
file-based and database search when hybrid search is enabled.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.historian.search import SearchResult
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)


class TestHistorianHybridSearch:
    """Test hybrid search integration functionality."""

    @pytest.mark.asyncio
    @patch("cognivault.agents.historian.agent.get_config")
    async def test_hybrid_search_enabled(self, mock_get_config: Any) -> None:
        """Test that hybrid search combines file and database results when enabled."""

        # Configure hybrid search to be enabled
        class MockTesting:
            historian_search_limit = 10
            enable_hybrid_search = True

        class MockConfig:
            testing = MockTesting()

        mock_get_config.return_value = MockConfig()

        # Create agent and explicitly enable hybrid search in agent config
        agent = HistorianAgent(llm=None)
        agent.config.hybrid_search_enabled = True  # Ensure hybrid search is enabled

        # Mock file search results
        file_results = [
            SearchResult(
                filepath="/file/test1.md",
                filename="test1.md",
                title="File Result 1",
                date="2024-01-01",
                relevance_score=0.9,
                match_type="content",
                excerpt="File content excerpt...",
                metadata={"source": "file"},
            )
        ]

        # Mock database search results
        db_results = [
            SearchResult(
                filepath="db_doc_123",
                filename="document_123",
                title="Database Result 1",
                date="2024-01-02",
                relevance_score=0.8,
                match_type="content",
                excerpt="Database content excerpt...",
                metadata={"source": "database", "database_id": "123"},
            )
        ]

        # Mock the individual search methods
        with (
            patch.object(
                agent, "_search_file_content", new_callable=AsyncMock
            ) as mock_file_search,
            patch.object(
                agent, "_search_database_content", new_callable=AsyncMock
            ) as mock_db_search,
        ):
            mock_file_search.return_value = file_results
            mock_db_search.return_value = db_results

            # Execute hybrid search
            context = AgentContextPatterns.simple_query("test query")
            results = await agent._search_historical_content("test query", context)

            # Verify both search methods were called with correct limits
            mock_file_search.assert_called_once_with("test query", 6)  # 60% of 10
            mock_db_search.assert_called_once_with("test query", 4)  # 40% of 10

            # Verify results include both file and database results
            assert len(results) == 2
            result_sources = [r.metadata.get("source") for r in results]
            assert "file" in result_sources
            assert "database" in result_sources

    @pytest.mark.asyncio
    @patch("cognivault.agents.historian.agent.get_config")
    async def test_hybrid_search_disabled_fallback(self, mock_get_config: Any) -> None:
        """Test that hybrid search falls back to file-only when disabled."""

        # Configure hybrid search to be disabled explicitly
        class MockTesting:
            historian_search_limit = 10
            enable_hybrid_search = False  # Explicitly disable hybrid search

        class MockConfig:
            testing = MockTesting()

        mock_get_config.return_value = MockConfig()

        # Create agent with hybrid search disabled in config
        agent = HistorianAgent(llm=None)
        agent.config.hybrid_search_enabled = (
            False  # Ensure it's disabled at agent level too
        )

        # Mock file search results
        file_results = [
            SearchResult(
                filepath="/file/test1.md",
                filename="test1.md",
                title="File Result 1",
                date="2024-01-01",
                relevance_score=0.9,
                match_type="content",
                excerpt="File content excerpt...",
                metadata={"source": "file"},
            )
        ]

        # Mock the search methods
        with (
            patch.object(
                agent, "_search_file_content", new_callable=AsyncMock
            ) as mock_file_search,
            patch.object(
                agent, "_search_database_content", new_callable=AsyncMock
            ) as mock_db_search,
        ):
            mock_file_search.return_value = file_results
            mock_db_search.return_value = []

            # Execute search
            context = AgentContextPatterns.simple_query("test query")
            results = await agent._search_historical_content("test query", context)

            # Verify only file search was called with full limit
            mock_file_search.assert_called_once_with("test query", 10)
            mock_db_search.assert_not_called()

            # Verify results are only file results
            assert len(results) == 1
            assert results[0].metadata.get("source") == "file"

    @pytest.mark.asyncio
    @patch("cognivault.agents.historian.agent.get_config")
    async def test_hybrid_search_deduplication(self, mock_get_config: Any) -> None:
        """Test that hybrid search properly deduplicates results."""

        # Configure hybrid search to be enabled
        class MockTesting:
            historian_search_limit = 10
            enable_hybrid_search = True

        class MockConfig:
            testing = MockTesting()

        mock_get_config.return_value = MockConfig()

        # Create agent and explicitly enable hybrid search in agent config
        agent = HistorianAgent(llm=None)
        agent.config.hybrid_search_enabled = True  # Ensure hybrid search is enabled

        # Mock identical results from both sources (should be deduplicated)
        duplicate_result = SearchResult(
            filepath="/shared/content.md",
            filename="content.md",
            title="Shared Content",
            date="2024-01-01",
            relevance_score=0.9,
            match_type="content",
            excerpt="This content appears in both sources...",
            metadata={"source": "both"},
        )

        file_results = [duplicate_result]
        db_results = [duplicate_result]  # Same content

        # Mock the search methods
        with (
            patch.object(
                agent, "_search_file_content", new_callable=AsyncMock
            ) as mock_file_search,
            patch.object(
                agent, "_search_database_content", new_callable=AsyncMock
            ) as mock_db_search,
        ):
            mock_file_search.return_value = file_results
            mock_db_search.return_value = db_results

            # Execute hybrid search
            context = AgentContextPatterns.simple_query("test query")
            results = await agent._search_historical_content("test query", context)

            # Verify deduplication occurred - should only have 1 result, not 2
            assert len(results) == 1
            assert results[0].title == "Shared Content"

    @pytest.mark.asyncio
    @patch("cognivault.agents.historian.agent.get_config")
    async def test_hybrid_search_database_failure_fallback(
        self, mock_get_config: Any
    ) -> None:
        """Test that hybrid search gracefully handles database failures."""

        # Configure hybrid search to be enabled
        class MockTesting:
            historian_search_limit = 10
            enable_hybrid_search = True

        class MockConfig:
            testing = MockTesting()

        mock_get_config.return_value = MockConfig()

        # Create agent and explicitly enable hybrid search in agent config
        agent = HistorianAgent(llm=None)
        agent.config.hybrid_search_enabled = True  # Ensure hybrid search is enabled

        # Mock file search success and database search failure
        file_results = [
            SearchResult(
                filepath="/file/test1.md",
                filename="test1.md",
                title="File Result 1",
                date="2024-01-01",
                relevance_score=0.9,
                match_type="content",
                excerpt="File content excerpt...",
                metadata={"source": "file"},
            )
        ]

        with (
            patch.object(
                agent, "_search_file_content", new_callable=AsyncMock
            ) as mock_file_search,
            patch.object(
                agent, "_search_database_content", new_callable=AsyncMock
            ) as mock_db_search,
        ):
            mock_file_search.return_value = file_results
            mock_db_search.return_value = []  # Database returns empty (failure)

            # Execute hybrid search
            context = AgentContextPatterns.simple_query("test query")
            results = await agent._search_historical_content("test query", context)

            # Verify both methods were called
            mock_file_search.assert_called_once()
            mock_db_search.assert_called_once()

            # Verify we still get file results even with database failure
            assert len(results) == 1
            assert results[0].metadata.get("source") == "file"
