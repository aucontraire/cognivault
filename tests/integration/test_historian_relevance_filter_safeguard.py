"""
Integration tests for the Historian agent's LLM relevance filter safeguard.

This module tests the end-to-end behavior of the safeguard that prevents
the LLM from filtering out ALL search results when relevant documents exist.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock, Mock, patch

from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.historian.search import SearchResult
from cognivault.config.agent_configs import HistorianConfig
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)


class OverAggressiveLLM(LLMInterface):
    """Mock LLM that always filters out all results (returns NONE)."""

    def __init__(self) -> None:
        self.relevance_call_count = 0
        self.synthesis_call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> Mock:
        """Generate mock response that filters everything for relevance."""
        if "RELEVANT INDICES" in prompt:
            self.relevance_call_count += 1
            response_text = "NONE"  # Filter everything
        elif "HISTORICAL SYNTHESIS" in prompt:
            self.synthesis_call_count += 1
            response_text = (
                "Historical context synthesis based on safeguard-retained results."
            )
        else:
            response_text = "Default response"

        mock_response = Mock()
        mock_response.text = response_text
        mock_response.tokens_used = 150
        mock_response.input_tokens = 100
        mock_response.output_tokens = 50
        return mock_response

    async def agenerate(self, prompt: str, **kwargs: Any) -> Mock:
        """Async version of generate."""
        return self.generate(prompt, **kwargs)


class SelectiveLLM(LLMInterface):
    """Mock LLM that returns specific indices (normal behavior)."""

    def __init__(self, relevant_indices: str = "0,1") -> None:
        self.relevant_indices = relevant_indices
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> Mock:
        """Generate mock response with selective filtering."""
        self.call_count += 1

        if "RELEVANT INDICES" in prompt:
            response_text = self.relevant_indices
        elif "HISTORICAL SYNTHESIS" in prompt:
            response_text = "Selective historical synthesis."
        else:
            response_text = "Default response"

        mock_response = Mock()
        mock_response.text = response_text
        mock_response.tokens_used = 150
        mock_response.input_tokens = 100
        mock_response.output_tokens = 50
        return mock_response

    async def agenerate(self, prompt: str, **kwargs: Any) -> Mock:
        """Async version of generate."""
        return self.generate(prompt, **kwargs)


@pytest.mark.integration
class TestHistorianRelevanceFilterSafeguardIntegration:
    """Integration tests for the LLM relevance filter safeguard."""

    @pytest.mark.asyncio
    async def test_end_to_end_safeguard_activation(self) -> None:
        """Test complete workflow with safeguard activation."""
        # Setup: Over-aggressive LLM + realistic search results
        aggressive_llm = OverAggressiveLLM()
        agent = HistorianAgent(llm=aggressive_llm)

        # Mock search engine with realistic results
        search_results = [
            SearchResult(
                filepath="/notes/machine_learning.md",
                filename="machine_learning.md",
                title="Introduction to Machine Learning",
                date="2024-01-15T10:00:00",
                relevance_score=0.92,
                match_type="content",
                matched_terms=["machine", "learning", "algorithms"],
                excerpt="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data...",
                metadata={"topics": ["AI", "machine learning", "algorithms"]},
            ),
            SearchResult(
                filepath="/notes/neural_networks.md",
                filename="neural_networks.md",
                title="Deep Neural Networks Overview",
                date="2024-01-16T14:30:00",
                relevance_score=0.88,
                match_type="content",
                matched_terms=["neural", "networks", "deep learning"],
                excerpt="Deep neural networks are computational models inspired by the human brain's structure...",
                metadata={"topics": ["AI", "neural networks", "deep learning"]},
            ),
            SearchResult(
                filepath="/notes/data_preprocessing.md",
                filename="data_preprocessing.md",
                title="Data Preprocessing Techniques",
                date="2024-01-17T09:15:00",
                relevance_score=0.75,
                match_type="tag",
                matched_terms=["data", "preprocessing"],
                excerpt="Effective data preprocessing is crucial for training robust machine learning models...",
                metadata={"topics": ["data science", "preprocessing", "ML"]},
            ),
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = search_results

        # Execute
        context = AgentContextPatterns.simple_query(
            "What are the key concepts in machine learning?"
        )
        result_context = await agent.run(context)

        # Assertions
        # 1. Safeguard should have activated
        # Note: LLM may be called twice - once in structured output path (which may fail and fall back),
        # and once in traditional path. Both paths correctly apply the safeguard.
        assert aggressive_llm.relevance_call_count >= 1  # LLM was consulted
        assert aggressive_llm.synthesis_call_count >= 1  # Synthesis still happened

        # 2. Should have retrieved notes despite LLM filtering all
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) == 3  # Default threshold

        # 3. Results should be sorted by relevance
        assert result_context.retrieved_notes[0] == "/notes/machine_learning.md"
        assert result_context.retrieved_notes[1] == "/notes/neural_networks.md"
        assert result_context.retrieved_notes[2] == "/notes/data_preprocessing.md"

        # 4. Should have produced output
        assert agent.name in result_context.agent_outputs
        assert len(result_context.agent_outputs[agent.name]) > 0

    @pytest.mark.asyncio
    async def test_safeguard_with_custom_configuration(self) -> None:
        """Test safeguard with custom minimum_results_threshold configuration."""
        # Custom config with threshold of 2
        config = HistorianConfig(minimum_results_threshold=2)
        aggressive_llm = OverAggressiveLLM()
        agent = HistorianAgent(llm=aggressive_llm, config=config)

        # Mock search with 5 results
        search_results = [
            SearchResult(
                filepath=f"/notes/doc{i}.md",
                filename=f"doc{i}.md",
                title=f"Document {i}",
                date=f"2024-01-{i:02d}T10:00:00",
                relevance_score=0.9 - (i * 0.05),
                match_type="content",
                matched_terms=["term"],
                excerpt=f"Content {i}...",
                metadata={"topics": ["test"]},
            )
            for i in range(1, 6)
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = search_results

        context = AgentContextPatterns.simple_query("test query")
        result_context = await agent.run(context)

        # Should keep only 2 results (custom threshold)
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) == 2

    @pytest.mark.asyncio
    async def test_safeguard_does_not_interfere_with_normal_llm(self) -> None:
        """Test that safeguard doesn't interfere when LLM works normally."""
        # Selective LLM that returns specific indices
        selective_llm = SelectiveLLM(relevant_indices="0,2,4")
        agent = HistorianAgent(llm=selective_llm)

        search_results = [
            SearchResult(
                filepath=f"/notes/doc{i}.md",
                filename=f"doc{i}.md",
                title=f"Document {i}",
                date=f"2024-01-{i:02d}T10:00:00",
                relevance_score=0.8,
                match_type="content",
                matched_terms=["term"],
                excerpt=f"Content {i}...",
                metadata={"topics": ["test"]},
            )
            for i in range(6)
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = search_results

        context = AgentContextPatterns.simple_query("test query")
        result_context = await agent.run(context)

        # Should have only the LLM-selected results (no safeguard activation)
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) == 3
        assert result_context.retrieved_notes[0] == "/notes/doc0.md"
        assert result_context.retrieved_notes[1] == "/notes/doc2.md"
        assert result_context.retrieved_notes[2] == "/notes/doc4.md"

    @pytest.mark.asyncio
    async def test_safeguard_with_real_search_and_database_fallback(self) -> None:
        """Test safeguard behavior with realistic search and database integration."""
        aggressive_llm = OverAggressiveLLM()

        # Use hybrid search configuration
        config = HistorianConfig(
            hybrid_search_enabled=True,
            hybrid_search_file_ratio=0.7,
            minimum_results_threshold=3,
        )
        agent = HistorianAgent(llm=aggressive_llm, config=config, search_type="hybrid")

        # Mock both file and database search results
        combined_results = [
            SearchResult(
                filepath="/notes/file_result.md",
                filename="file_result.md",
                title="File-based Result",
                date="2024-01-01T10:00:00",
                relevance_score=0.95,
                match_type="content",
                matched_terms=["file"],
                excerpt="From file search...",
                metadata={"topics": ["file"], "source": "file"},
            ),
            SearchResult(
                filepath="db_doc_123",
                filename="document_123",
                title="Database Result",
                date="2024-01-02T10:00:00",
                relevance_score=0.85,
                match_type="content",
                matched_terms=["database"],
                excerpt="From database search...",
                metadata={"topics": ["database"], "source": "database"},
            ),
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = combined_results

        context = AgentContextPatterns.simple_query("hybrid search test")
        result_context = await agent.run(context)

        # Safeguard should work with both file and database results
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) >= 2

    @pytest.mark.asyncio
    async def test_safeguard_logs_activation_details(self, caplog: Any) -> None:
        """Test that safeguard activation is logged with detailed information."""
        import logging

        caplog.set_level(logging.INFO)

        aggressive_llm = OverAggressiveLLM()
        config = HistorianConfig(minimum_results_threshold=2)
        agent = HistorianAgent(llm=aggressive_llm, config=config)

        search_results = [
            SearchResult(
                filepath="/test1.md",
                filename="test1.md",
                title="Test Document 1",
                date="2024-01-01T10:00:00",
                relevance_score=0.9,
                match_type="content",
                matched_terms=["test"],
                excerpt="Test 1...",
                metadata={"topics": ["test"]},
            ),
            SearchResult(
                filepath="/test2.md",
                filename="test2.md",
                title="Test Document 2",
                date="2024-01-02T10:00:00",
                relevance_score=0.7,
                match_type="content",
                matched_terms=["test"],
                excerpt="Test 2...",
                metadata={"topics": ["test"]},
            ),
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = search_results

        context = AgentContextPatterns.simple_query("test query")
        await agent.run(context)

        # Check for detailed logging
        log_messages = [record.message for record in caplog.records]

        # Should log safeguard activation
        assert any("SAFEGUARD ACTIVATED" in msg for msg in log_messages)

        # Should log kept results with scores
        assert any("Safeguard kept results:" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_safeguard_handles_empty_search_results_gracefully(self) -> None:
        """Test that safeguard doesn't activate when search finds nothing."""
        aggressive_llm = OverAggressiveLLM()
        agent = HistorianAgent(llm=aggressive_llm)

        # No search results
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = []

        context = AgentContextPatterns.simple_query("nonexistent topic")
        result_context = await agent.run(context)

        # Should not crash, and should have empty results
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) == 0

        # LLM should not be called for relevance if no results
        assert aggressive_llm.relevance_call_count == 0
