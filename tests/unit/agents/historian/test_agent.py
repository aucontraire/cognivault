"""
Tests for the enhanced Historian agent with LLM integration.

This module tests the HistorianAgent's search integration, LLM-powered relevance
analysis, historical synthesis, and error handling capabilities.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import Mock, AsyncMock, patch

from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.historian.search import SearchResult
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from cognivault.agents.base_agent import NodeType


class MockLLM(LLMInterface):
    """Mock LLM for testing purposes."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    def generate(self, prompt: str, **kwargs) -> Mock:
        """Generate mock response based on prompt content."""
        self.call_count += 1
        self.last_prompt = prompt

        # Return responses based on prompt content
        if "RELEVANT INDICES" in prompt:
            # Relevance analysis response
            response_text = self.responses.get("relevance", "0,1,2")
        elif "HISTORICAL SYNTHESIS" in prompt:
            # Synthesis response
            response_text = self.responses.get(
                "synthesis", "This is a test historical synthesis."
            )
        else:
            # Default response
            response_text = self.responses.get("default", "Mock response")

        mock_response = Mock()
        mock_response.text = response_text
        mock_response.tokens_used = 180
        mock_response.input_tokens = 120
        mock_response.output_tokens = 60
        return mock_response

    async def agenerate(self, prompt: str, **kwargs) -> Mock:
        """Async version of generate."""
        return self.generate(prompt, **kwargs)


class TestHistorianAgentInitialization:
    """Test HistorianAgent initialization and setup."""

    def test_default_initialization(self):
        """Test default agent initialization."""
        # Use llm=None to prevent real API calls during testing
        agent = HistorianAgent(llm=None)

        assert agent.name == "historian"
        assert agent.search_type == "hybrid"
        assert agent.search_engine is not None
        assert agent.llm is None

    def test_initialization_with_custom_llm(self):
        """Test initialization with custom LLM."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        assert agent.name == "historian"
        assert agent.llm is mock_llm
        assert agent.search_type == "hybrid"

    def test_initialization_with_search_type(self):
        """Test initialization with custom search type."""
        agent = HistorianAgent(search_type="tag")

        assert agent.search_type == "tag"
        assert agent.search_engine is not None

    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig")
    def test_default_llm_creation_success(self, mock_config_class, mock_llm_class):
        """Test successful default LLM creation."""
        # Mock config
        mock_config = Mock()
        mock_config.api_key = "test-key"
        mock_config.model = "gpt-4"
        mock_config.base_url = "https://api.openai.com/v1"
        mock_config_class.load.return_value = mock_config

        # Mock LLM
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        agent = HistorianAgent()

        assert agent.llm is mock_llm
        mock_config_class.load.assert_called_once()
        mock_llm_class.assert_called_once_with(
            api_key="test-key", model="gpt-4", base_url="https://api.openai.com/v1"
        )

    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig")
    def test_default_llm_creation_failure(self, mock_config_class, mock_llm_class):
        """Test default LLM creation failure handling."""
        mock_config_class.load.side_effect = Exception("Config error")

        agent = HistorianAgent()

        assert agent.llm is None
        mock_config_class.load.assert_called_once()
        mock_llm_class.assert_not_called()


class TestHistorianAgentExecution:
    """Test HistorianAgent execution workflows."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create mock search results
        self.mock_search_results = [
            SearchResult(
                filepath=str(self.notes_dir / "ai_note.md"),
                filename="ai_note.md",
                title="AI Fundamentals",
                date="2024-01-01T10:00:00",
                relevance_score=0.95,
                match_type="topic",
                matched_terms=["ai", "artificial_intelligence"],
                excerpt="This note covers artificial intelligence fundamentals...",
                metadata={
                    "topics": ["artificial_intelligence", "machine_learning"],
                    "domain": "technology",
                    "uuid": "ai-note-uuid",
                },
            ),
            SearchResult(
                filepath=str(self.notes_dir / "ml_note.md"),
                filename="ml_note.md",
                title="Machine Learning Basics",
                date="2024-01-02T11:00:00",
                relevance_score=0.87,
                match_type="content",
                matched_terms=["machine", "learning"],
                excerpt="Machine learning algorithms and applications...",
                metadata={
                    "topics": ["machine_learning", "algorithms"],
                    "domain": "technology",
                    "uuid": "ml-note-uuid",
                },
            ),
            SearchResult(
                filepath=str(self.notes_dir / "old_note.md"),
                filename="old_note.md",
                title="Historical AI Development",
                date="2023-06-15T09:00:00",
                relevance_score=0.45,
                match_type="title",
                matched_terms=["ai"],
                excerpt="The history of AI development from early concepts...",
                metadata={
                    "topics": ["history", "artificial_intelligence"],
                    "domain": "history",
                    "uuid": "old-note-uuid",
                },
            ),
        ]

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_successful_execution_with_llm(self):
        """Test successful agent execution with LLM."""
        # Set up mock LLM with appropriate responses
        mock_llm = MockLLM(
            {
                "relevance": "0,1",  # Select first two results
                "synthesis": "Based on the historical context, AI and machine learning have evolved significantly...",
            }
        )

        # Create agent with mock LLM and mock search
        agent = HistorianAgent(llm=mock_llm)

        # Mock the search engine
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = self.mock_search_results

        # Create test context
        context = AgentContext(query="What is artificial intelligence?")

        # Execute agent
        result_context = await agent.run(context)

        # Verify execution
        assert agent.name in result_context.agent_outputs
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) > 0
        assert (
            "AI and machine learning have evolved"
            in result_context.agent_outputs[agent.name]
        )

        # Verify LLM was called
        assert mock_llm.call_count == 2  # Once for relevance, once for synthesis

    @pytest.mark.asyncio
    async def test_successful_execution_without_llm(self):
        """Test successful agent execution without LLM."""
        # Create agent without LLM
        agent = HistorianAgent(llm=None)

        # Verify no LLM is set
        assert agent.llm is None

        # Mock the search engine
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = self.mock_search_results

        # Create test context
        context = AgentContext(query="What is machine learning?")

        # Execute agent
        result_context = await agent.run(context)

        # Verify execution
        assert agent.name in result_context.agent_outputs
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) > 0

        # Should get basic summary format (not LLM-generated)
        output = result_context.agent_outputs[agent.name]
        assert "Found" in output or "Machine Learning Basics" in output

    @pytest.mark.asyncio
    async def test_execution_with_no_search_results(self):
        """Test execution when search returns no results."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        # Mock search to return empty results
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = []

        context = AgentContext(query="nonexistent topic")
        result_context = await agent.run(context)

        # Should handle gracefully
        assert agent.name in result_context.agent_outputs
        assert (
            "No relevant historical context found"
            in result_context.agent_outputs[agent.name]
        )

    @pytest.mark.asyncio
    async def test_execution_with_search_failure(self):
        """Test execution when search fails."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        # Mock search to raise exception
        agent.search_engine = AsyncMock()
        agent.search_engine.search.side_effect = Exception("Search failed")

        context = AgentContext(query="test query")

        # Should handle gracefully
        result_context = await agent.run(context)
        assert agent.name in result_context.agent_outputs

    @pytest.mark.asyncio
    async def test_execution_with_llm_failure(self):
        """Test execution when LLM calls fail."""
        # Create LLM that raises exceptions
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM failed")

        agent = HistorianAgent(llm=mock_llm)

        # Mock successful search
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = self.mock_search_results

        context = AgentContext(query="test query")
        result_context = await agent.run(context)

        # Should fall back gracefully
        assert agent.name in result_context.agent_outputs
        assert result_context.retrieved_notes is not None

    @pytest.mark.asyncio
    @patch("cognivault.agents.historian.agent.get_config")
    async def test_execution_with_mock_fallback(self, mock_get_config):
        """Test execution with mock data fallback."""

        # Create proper mock objects with necessary attributes
        class MockTesting:
            mock_history_entries = ["mock_note_1.md", "mock_note_2.md"]

        class MockExecution:
            enable_simulation_delay = False

        class MockConfig:
            execution = MockExecution()
            testing = MockTesting()

        mock_get_config.return_value = MockConfig()

        agent = HistorianAgent(llm=None)

        # Mock search to raise exception
        agent.search_engine = AsyncMock()
        agent.search_engine.search.side_effect = Exception("Search failed")

        context = AgentContext(query="test query")
        result_context = await agent.run(context)

        # Should use mock fallback when mock_history_entries exists
        assert agent.name in result_context.agent_outputs
        # Check if mock fallback was used
        if "mock_note_1.md" in str(result_context.retrieved_notes):
            assert result_context.retrieved_notes == [
                "mock_note_1.md",
                "mock_note_2.md",
            ]
            assert "fallback data" in result_context.agent_outputs[agent.name]
        else:
            # Or regular failure handling if config not working
            assert (
                "No historical context available"
                in result_context.agent_outputs[agent.name]
                or "No relevant historical context found"
                in result_context.agent_outputs[agent.name]
            )

    @pytest.mark.asyncio
    @patch("cognivault.agents.historian.agent.get_config")
    async def test_execution_with_simulation_delay(self, mock_get_config):
        """Test execution with simulation delay enabled."""

        # Create proper mock objects
        class MockTesting:
            historian_search_limit = 5
            mock_history_entries = []

        class MockExecution:
            enable_simulation_delay = True
            simulation_delay_seconds = 0.1  # Short delay for testing

        class MockConfig:
            execution = MockExecution()
            testing = MockTesting()

        mock_get_config.return_value = MockConfig()

        agent = HistorianAgent(llm=None)
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = []

        context = AgentContext(query="test query")

        # Time the execution
        import time

        start_time = time.time()
        await agent.run(context)
        execution_time = time.time() - start_time

        # Should include delay
        assert execution_time >= 0.1


class TestHistorianAgentSearchIntegration:
    """Test integration with search functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_search_historical_content_success(self):
        """Test successful historical content search."""
        agent = HistorianAgent(llm=None)

        # Mock search results
        expected_results = [
            SearchResult(
                filepath="/test/path.md",
                filename="path.md",
                title="Test Note",
                date="2024-01-01",
                relevance_score=0.8,
                match_type="topic",
                matched_terms=["test"],
                excerpt="Test excerpt",
                metadata={},
            )
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = expected_results

        context = AgentContext(query="test query")
        results = await agent._search_historical_content("test query", context)

        assert results == expected_results
        agent.search_engine.search.assert_called_once_with("test query", limit=10)

    @pytest.mark.asyncio
    async def test_search_historical_content_failure(self):
        """Test search failure handling."""
        agent = HistorianAgent(llm=None)

        agent.search_engine = AsyncMock()
        agent.search_engine.search.side_effect = Exception("Search error")

        context = AgentContext(query="test query")
        results = await agent._search_historical_content("test query", context)

        assert results == []

    @pytest.mark.asyncio
    @patch("cognivault.agents.historian.agent.get_config")
    async def test_search_with_custom_limit(self, mock_get_config):
        """Test search with custom search limit from config."""

        # Create a more sophisticated mock that properly handles getattr
        class MockTesting:
            historian_search_limit = 15

        class MockConfig:
            testing = MockTesting()

        mock_get_config.return_value = MockConfig()

        agent = HistorianAgent(llm=None)
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = []

        context = AgentContext(query="test query")
        await agent._search_historical_content("test query", context)

        agent.search_engine.search.assert_called_once_with("test query", limit=15)


class TestHistorianAgentRelevanceAnalysis:
    """Test LLM-powered relevance analysis functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_search_results = [
            SearchResult(
                filepath="/path1.md",
                filename="note1.md",
                title="Relevant Note",
                date="2024-01-01",
                relevance_score=0.9,
                match_type="topic",
                matched_terms=["ai"],
                excerpt="This note is highly relevant...",
                metadata={"topics": ["ai"]},
            ),
            SearchResult(
                filepath="/path2.md",
                filename="note2.md",
                title="Somewhat Relevant",
                date="2024-01-02",
                relevance_score=0.6,
                match_type="content",
                matched_terms=["technology"],
                excerpt="This note has some relevance...",
                metadata={"topics": ["technology"]},
            ),
            SearchResult(
                filepath="/path3.md",
                filename="note3.md",
                title="Less Relevant",
                date="2024-01-03",
                relevance_score=0.3,
                match_type="title",
                matched_terms=["general"],
                excerpt="This note is less relevant...",
                metadata={"topics": ["general"]},
            ),
        ]

    @pytest.mark.asyncio
    async def test_analyze_relevance_with_llm(self):
        """Test relevance analysis with LLM."""
        mock_llm = MockLLM({"relevance": "0,2"})  # Select indices 0 and 2
        agent = HistorianAgent(llm=mock_llm)

        context = AgentContext(query="test query")
        filtered_results = await agent._analyze_relevance(
            "test query", self.mock_search_results, context
        )

        assert len(filtered_results) == 2
        assert filtered_results[0].title == "Relevant Note"
        assert filtered_results[1].title == "Less Relevant"
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_analyze_relevance_without_llm(self):
        """Test relevance analysis fallback without LLM."""
        agent = HistorianAgent(llm=None)

        context = AgentContext(query="test query")

        # Verify agent has no LLM
        assert agent.llm is None

        filtered_results = await agent._analyze_relevance(
            "test query", self.mock_search_results, context
        )

        # Should return top 5 results (or all if fewer than 5)
        assert len(filtered_results) <= 5
        # For 3 mock results, should return all 3
        assert len(filtered_results) == 3

    @pytest.mark.asyncio
    async def test_analyze_relevance_empty_results(self):
        """Test relevance analysis with empty search results."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        context = AgentContext(query="test query")
        filtered_results = await agent._analyze_relevance("test query", [], context)

        assert filtered_results == []
        assert mock_llm.call_count == 0  # LLM should not be called for empty results

    @pytest.mark.asyncio
    async def test_analyze_relevance_llm_failure(self):
        """Test relevance analysis when LLM fails."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM error")

        agent = HistorianAgent(llm=mock_llm)

        context = AgentContext(query="test query")
        filtered_results = await agent._analyze_relevance(
            "test query", self.mock_search_results, context
        )

        # Should fall back to top 5 search results
        assert len(filtered_results) <= 5
        assert len(filtered_results) > 0

    def test_build_relevance_prompt(self):
        """Test relevance prompt building."""
        agent = HistorianAgent(llm=None)

        prompt = agent._build_relevance_prompt("test query", self.mock_search_results)

        assert "test query" in prompt
        assert "RELEVANT INDICES" in prompt
        assert "Relevant Note" in prompt
        assert "maximum 5 most relevant" in prompt
        assert "[0]" in prompt and "[1]" in prompt and "[2]" in prompt

    def test_parse_relevance_response_valid(self):
        """Test parsing valid relevance response."""
        agent = HistorianAgent(llm=None)

        # Test comma-separated indices
        indices = agent._parse_relevance_response("0,2,4")
        assert indices == [0, 2, 4]

        # Test with extra text
        indices = agent._parse_relevance_response("The relevant indices are: 1, 3, 5")
        assert indices == [1, 3, 5]

        # Test NONE response
        indices = agent._parse_relevance_response("NONE")
        assert indices == []

    def test_parse_relevance_response_invalid(self):
        """Test parsing invalid relevance response."""
        agent = HistorianAgent(llm=None)

        # Test malformed response
        indices = agent._parse_relevance_response("invalid response")
        assert isinstance(indices, list)  # Should return default fallback

        # Test empty response
        indices = agent._parse_relevance_response("")
        assert isinstance(indices, list)

    def test_parse_relevance_response_limit(self):
        """Test that relevance response parsing limits to 5 results."""
        agent = HistorianAgent(llm=None)

        indices = agent._parse_relevance_response("0,1,2,3,4,5,6,7,8,9")
        assert len(indices) <= 5


class TestHistorianAgentSynthesis:
    """Test historical context synthesis functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_filtered_results = [
            SearchResult(
                filepath="/ai_history.md",
                filename="ai_history.md",
                title="AI Development Timeline",
                date="2024-01-01T10:00:00",
                relevance_score=0.95,
                match_type="topic",
                matched_terms=["ai", "development"],
                excerpt="The development of AI has progressed through several key phases...",
                metadata={
                    "topics": ["artificial_intelligence", "history"],
                    "domain": "technology",
                },
            ),
            SearchResult(
                filepath="/ml_evolution.md",
                filename="ml_evolution.md",
                title="Machine Learning Evolution",
                date="2024-01-02T11:00:00",
                relevance_score=0.87,
                match_type="content",
                matched_terms=["machine_learning", "evolution"],
                excerpt="Machine learning has evolved from simple statistical methods...",
                metadata={
                    "topics": ["machine_learning", "algorithms"],
                    "domain": "technology",
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_synthesize_historical_context_with_llm(self):
        """Test historical context synthesis with LLM."""
        mock_llm = MockLLM(
            {
                "synthesis": "The historical context reveals that AI development has progressed through distinct phases, from early statistical methods to modern deep learning approaches. This evolution demonstrates the iterative nature of technological advancement."
            }
        )

        agent = HistorianAgent(llm=mock_llm)

        context = AgentContext(query="How has AI evolved?")
        synthesis = await agent._synthesize_historical_context(
            "How has AI evolved?", self.mock_filtered_results, context
        )

        assert "historical context reveals" in synthesis
        assert "AI development has progressed" in synthesis
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_synthesize_historical_context_without_llm(self):
        """Test historical context synthesis without LLM."""
        agent = HistorianAgent(llm=None)

        # Verify agent has no LLM
        assert agent.llm is None

        context = AgentContext(query="How has AI evolved?")
        synthesis = await agent._synthesize_historical_context(
            "How has AI evolved?", self.mock_filtered_results, context
        )

        # Should create basic summary since no LLM is available
        assert "Found" in synthesis
        # Check for titles from the mock data
        assert "AI Development Timeline" in synthesis
        assert "Machine Learning Evolution" in synthesis

    @pytest.mark.asyncio
    async def test_synthesize_historical_context_empty_results(self):
        """Test synthesis with empty filtered results."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        context = AgentContext(query="test query")
        synthesis = await agent._synthesize_historical_context(
            "test query", [], context
        )

        assert "No relevant historical context found" in synthesis
        assert mock_llm.call_count == 0

    @pytest.mark.asyncio
    async def test_synthesize_historical_context_llm_failure(self):
        """Test synthesis when LLM fails."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM synthesis error")

        agent = HistorianAgent(llm=mock_llm)

        context = AgentContext(query="test query")
        synthesis = await agent._synthesize_historical_context(
            "test query", self.mock_filtered_results, context
        )

        # Should fall back to basic summary
        assert any(
            word in synthesis for word in ["Found", "relevant", "historical", "notes"]
        )

    def test_build_synthesis_prompt(self):
        """Test synthesis prompt building."""
        agent = HistorianAgent(llm=None)

        prompt = agent._build_synthesis_prompt(
            "How has AI evolved?", self.mock_filtered_results
        )

        assert "How has AI evolved?" in prompt
        assert "HISTORICAL SYNTHESIS" in prompt
        assert "AI Development Timeline" in prompt
        assert "Machine Learning Evolution" in prompt
        assert "coherent narrative" in prompt

    def test_create_basic_summary(self):
        """Test basic summary creation."""
        agent = HistorianAgent(llm=None)

        summary = agent._create_basic_summary("test query", self.mock_filtered_results)

        assert any(
            word in summary for word in ["Found", "relevant", "historical", "notes"]
        )
        assert "AI Development Timeline" in summary
        assert "Machine Learning Evolution" in summary

    def test_create_basic_summary_empty(self):
        """Test basic summary with empty results."""
        agent = HistorianAgent(llm=None)

        summary = agent._create_basic_summary("test query", [])

        assert "No relevant historical context found" in summary


class TestHistorianAgentFallbackMethods:
    """Test fallback and error handling methods."""

    @pytest.mark.asyncio
    async def test_create_fallback_output(self):
        """Test fallback output creation."""
        agent = HistorianAgent(llm=None)

        mock_history = ["note1.md", "note2.md", "note3.md"]
        output = await agent._create_fallback_output("test query", mock_history)

        assert "Historical context for: test query" in output
        assert "fallback data" in output
        assert "note1.md" in output
        assert "note2.md" in output

    @pytest.mark.asyncio
    async def test_create_no_context_output(self):
        """Test no context output creation."""
        agent = HistorianAgent(llm=None)

        output = await agent._create_no_context_output("test query")

        assert "No historical context available" in output
        assert "test query" in output
        assert "new topic" in output


class TestHistorianAgentNodeMetadata:
    """Test LangGraph node metadata definition."""

    def test_define_node_metadata(self):
        """Test node metadata definition."""
        agent = HistorianAgent(llm=None)

        metadata = agent.define_node_metadata()

        assert metadata["node_type"] == NodeType.PROCESSOR
        assert metadata["dependencies"] == []
        assert "inputs" in metadata
        assert "outputs" in metadata
        assert len(metadata["inputs"]) == 1
        assert len(metadata["outputs"]) == 1
        assert metadata["inputs"][0].name == "context"
        assert metadata["outputs"][0].name == "context"
        assert "historian" in metadata["tags"]
        assert "parallel" in metadata["tags"]


class TestHistorianAgentContextTracking:
    """Test context tracking and execution metadata."""

    @pytest.mark.asyncio
    async def test_context_execution_tracking(self):
        """Test that agent properly tracks execution in context."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        # Mock successful search
        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = []

        context = AgentContext(query="test query")

        # Track initial state
        initial_executions = len(context.agent_trace)

        result_context = await agent.run(context)

        # Verify execution tracking
        assert len(result_context.agent_trace) > initial_executions
        assert agent.name in result_context.agent_outputs

    @pytest.mark.asyncio
    async def test_context_retrieved_notes_tracking(self):
        """Test that retrieved notes are properly tracked in context."""
        mock_llm = MockLLM({"relevance": "0", "synthesis": "Test synthesis"})
        agent = HistorianAgent(llm=mock_llm)

        # Mock search with results
        mock_results = [
            SearchResult(
                filepath="/test/note.md",
                filename="note.md",
                title="Test Note",
                date="2024-01-01",
                relevance_score=0.8,
                match_type="topic",
                matched_terms=["test"],
                excerpt="Test content",
                metadata={},
            )
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = mock_results

        context = AgentContext(query="test query")
        result_context = await agent.run(context)

        # Verify retrieved notes tracking
        assert result_context.retrieved_notes is not None
        assert len(result_context.retrieved_notes) > 0
        assert "/test/note.md" in result_context.retrieved_notes


class TestHistorianAgentErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_search_failure(self):
        """Test graceful degradation when search completely fails."""
        agent = HistorianAgent(llm=None)

        # Mock search to raise exception
        agent.search_engine = AsyncMock()
        agent.search_engine.search.side_effect = Exception("Complete search failure")

        context = AgentContext(query="test query")

        # Should not raise exception
        result_context = await agent.run(context)

        # Should have some output
        assert agent.name in result_context.agent_outputs
        output = result_context.agent_outputs[agent.name]
        # Check for either failure message format
        assert (
            "No historical context available" in output
            or "No relevant historical context found" in output
        )

    @pytest.mark.asyncio
    async def test_graceful_degradation_llm_failure(self):
        """Test graceful degradation when LLM completely fails."""
        # Create LLM that always fails
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM completely down")

        agent = HistorianAgent(llm=mock_llm)

        # Mock successful search
        mock_results = [
            SearchResult(
                filepath="/test.md",
                filename="test.md",
                title="Test",
                date="2024-01-01",
                relevance_score=0.8,
                match_type="topic",
                matched_terms=["test"],
                excerpt="Test excerpt",
                metadata={},
            )
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = mock_results

        context = AgentContext(query="test query")
        result_context = await agent.run(context)

        # Should still produce output using fallback methods
        assert agent.name in result_context.agent_outputs
        assert result_context.retrieved_notes is not None

    @pytest.mark.asyncio
    async def test_edge_case_malformed_search_results(self):
        """Test handling of malformed search results."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        # Create minimal search results that pass basic validation
        # but represent edge cases the agent should handle gracefully
        malformed_results = [
            SearchResult(
                filepath="/empty/result.md",  # Valid but minimal path
                filename="empty.md",  # Valid but minimal filename
                title="Empty",  # Valid but minimal title
                date="",  # Empty date (still valid)
                relevance_score=0.0,
                match_type="content",  # Valid match type
                matched_terms=[],
                excerpt="",  # Empty excerpt (valid)
                metadata={},
            )
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = malformed_results

        context = AgentContext(query="test query")

        # Should handle gracefully
        result_context = await agent.run(context)
        assert agent.name in result_context.agent_outputs

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self):
        """Test that agent can handle concurrent executions safely."""
        mock_llm = MockLLM()
        agent = HistorianAgent(llm=mock_llm)

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = []

        # Create multiple contexts
        contexts = [AgentContext(query=f"query {i}") for i in range(3)]

        # Execute concurrently
        results = await asyncio.gather(*[agent.run(context) for context in contexts])

        # All should complete successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert agent.name in result.agent_outputs
            assert f"query {i}" in result.query


class TestHistorianAgentIntegration:
    """Test integration scenarios with the broader agent system."""

    @pytest.mark.asyncio
    async def test_integration_with_agent_context(self):
        """Test full integration with AgentContext system."""
        mock_llm = MockLLM(
            {
                "relevance": "0",
                "synthesis": "Comprehensive historical analysis shows clear patterns in the evolution of this topic.",
            }
        )

        agent = HistorianAgent(llm=mock_llm)

        # Mock comprehensive search results
        comprehensive_results = [
            SearchResult(
                filepath="/comprehensive.md",
                filename="comprehensive.md",
                title="Comprehensive Analysis",
                date="2024-01-01T10:00:00",
                relevance_score=0.95,
                match_type="topic",
                matched_terms=["comprehensive", "analysis"],
                excerpt="This comprehensive analysis covers multiple aspects...",
                metadata={
                    "topics": ["analysis", "research", "comprehensive"],
                    "domain": "academic",
                    "difficulty": "advanced",
                    "uuid": "comp-analysis-uuid",
                },
            )
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = comprehensive_results

        # Create realistic context
        context = AgentContext(
            query="What are the comprehensive approaches to analyzing complex systems?"
        )

        # Execute
        result_context = await agent.run(context)

        # Verify comprehensive integration
        assert agent.name in result_context.agent_outputs
        assert result_context.retrieved_notes == ["/comprehensive.md"]
        assert "historical analysis" in result_context.agent_outputs[agent.name]
        assert len(result_context.agent_trace) > 0

    @pytest.mark.asyncio
    async def test_performance_with_large_result_sets(self):
        """Test performance and behavior with large search result sets."""
        mock_llm = MockLLM({"relevance": "0,1,2,3,4"})  # Select first 5
        agent = HistorianAgent(llm=mock_llm)

        # Create large result set
        large_results = [
            SearchResult(
                filepath=f"/note_{i}.md",
                filename=f"note_{i}.md",
                title=f"Note {i}",
                date=f"2024-01-{i:02d}T10:00:00",
                relevance_score=max(0.1, 0.9 - (i * 0.01)),
                match_type="topic",
                matched_terms=[f"term_{i}"],
                excerpt=f"This is excerpt {i}...",
                metadata={"topics": [f"topic_{i}"]},
            )
            for i in range(50)  # 50 results
        ]

        agent.search_engine = AsyncMock()
        agent.search_engine.search.return_value = large_results

        context = AgentContext(query="broad topic search")

        # Should handle large result sets efficiently
        result_context = await agent.run(context)

        assert agent.name in result_context.agent_outputs
        # Should limit to reasonable number of retrieved notes
        assert len(result_context.retrieved_notes) <= 5
