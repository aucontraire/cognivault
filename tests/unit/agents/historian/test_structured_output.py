"""
Comprehensive test suite for HistorianAgent structured output implementation.

Tests the integration of LangChain service for structured output,
fallback mechanisms, content pollution prevention, and hybrid search compatibility.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict, Optional, List

from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.historian.search import SearchResult
from cognivault.agents.models import (
    HistorianOutput,
    HistoricalReference,
    ProcessingMode,
    ConfidenceLevel,
)
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface, LLMResponse
from cognivault.services.langchain_service import StructuredOutputResult
from cognivault.config.agent_configs import HistorianConfig


class MockLLM(LLMInterface):
    """Mock LLM for testing."""

    def __init__(self, response_text: str = "Test historical synthesis", **kwargs: Any):
        super().__init__()
        self.response_text = response_text
        self.model = kwargs.get("model", "gpt-4")
        self.api_key = kwargs.get("api_key", "test-key")
        self.generate_called = False
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a mock response."""
        self.generate_called = True
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = kwargs.get("system_prompt", "")

        # Different responses for relevance analysis vs synthesis
        if "RELEVANT INDICES" in prompt:
            # Relevance analysis response
            return LLMResponse(
                text="0,1,2",  # Return first 3 indices as relevant
                tokens_used=50,
                input_tokens=30,
                output_tokens=20,
            )
        else:
            # Synthesis response
            return LLMResponse(
                text=self.response_text,
                tokens_used=100,
                input_tokens=60,
                output_tokens=40,
            )

    async def agenerate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Async generate."""
        return self.generate(prompt, **kwargs)


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a mock LLM."""
    return MockLLM(
        response_text="Historical synthesis: The query relates to past research on machine learning algorithms."
    )


@pytest.fixture
def agent_context() -> AgentContext:
    """Create a test agent context."""
    return AgentContext(query="What is the history of machine learning?")


@pytest.fixture
def historian_config() -> HistorianConfig:
    """Create a test historian configuration."""
    return HistorianConfig(
        search_depth="deep",
        relevance_threshold=0.7,
        hybrid_search_enabled=True,
        hybrid_search_file_ratio=0.5,
    )


@pytest.fixture
def mock_search_results() -> List[SearchResult]:
    """Create mock search results."""
    return [
        SearchResult(
            title="Early ML Research",
            excerpt="Machine learning originated in the 1950s with perceptrons...",
            filepath="/notes/ml_history.md",
            filename="ml_history.md",
            date="2024-01-15",
            match_type="content",
            relevance_score=0.9,
            metadata={"topics": ["machine learning", "history", "perceptrons"]},
        ),
        SearchResult(
            title="Neural Network Evolution",
            excerpt="The development of neural networks went through several winters...",
            filepath="/notes/nn_evolution.md",
            filename="nn_evolution.md",
            date="2024-02-20",
            match_type="content",
            relevance_score=0.85,
            metadata={"topics": ["neural networks", "AI winters", "deep learning"]},
        ),
        SearchResult(
            title="Modern Deep Learning",
            excerpt="The breakthrough in 2012 with AlexNet marked a new era...",
            filepath="/notes/deep_learning.md",
            filename="deep_learning.md",
            date="2024-03-10",
            match_type="content",
            relevance_score=0.8,
            metadata={"topics": ["deep learning", "AlexNet", "ImageNet"]},
        ),
    ]


@pytest.fixture
def structured_historian_output() -> HistorianOutput:
    """Create a structured historian output for testing."""
    return HistorianOutput(
        agent_name="historian",
        processing_mode=ProcessingMode.ACTIVE,
        confidence=ConfidenceLevel.HIGH,
        processing_time_ms=250.5,
        relevant_sources=[
            HistoricalReference(
                source_id=None,
                title="Early ML Research",
                relevance_score=0.9,
                content_snippet="Machine learning originated in the 1950s...",
            ),
            HistoricalReference(
                source_id=None,
                title="Neural Network Evolution",
                relevance_score=0.85,
                content_snippet="The development of neural networks...",
            ),
        ],
        historical_synthesis="Machine learning has evolved from early perceptrons to modern deep learning systems.",
        themes_identified=["technological evolution", "research breakthroughs"],
        time_periods_covered=["1950s", "2012-present"],
        contextual_connections=["Historical patterns inform current research"],
        sources_searched=10,
        relevant_sources_found=2,
    )


class TestHistorianAgentStructuredOutput:
    """Test suite for HistorianAgent structured output functionality."""

    @pytest.mark.asyncio
    async def test_successful_structured_output(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        structured_historian_output: HistorianOutput,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test successful structured output processing."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Mock search and structured service
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                with patch.object(
                    agent.structured_service,
                    "get_structured_output",
                    new_callable=AsyncMock,
                ) as mock_get_structured:
                    mock_get_structured.return_value = structured_historian_output

                    result = await agent.run(agent_context)

                    # Verify structured output was called
                    mock_get_structured.assert_called_once()
                    call_args = mock_get_structured.call_args
                    assert call_args.kwargs["output_class"] == HistorianOutput
                    assert call_args.kwargs["max_retries"] == 3

                    # Verify context was updated
                    assert "historian" in result.agent_outputs
                    assert (
                        "Machine learning has evolved"
                        in result.agent_outputs["historian"]
                    )

                    # Verify retrieved notes were set
                    assert result.retrieved_notes is not None
                    assert len(result.retrieved_notes) > 0

    @pytest.mark.asyncio
    async def test_structured_output_with_result_wrapper(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        structured_historian_output: HistorianOutput,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test structured output when result is wrapped in StructuredOutputResult."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Create wrapped result
            wrapped_result = StructuredOutputResult(
                parsed=structured_historian_output,
                raw="raw JSON response",
                method_used="with_structured_output",
                fallback_used=False,
            )

            # Mock search and structured service
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                with patch.object(
                    agent.structured_service,
                    "get_structured_output",
                    new_callable=AsyncMock,
                ) as mock_get_structured:
                    mock_get_structured.return_value = wrapped_result

                    result = await agent.run(agent_context)

                    # Verify context was updated correctly
                    assert "historian" in result.agent_outputs
                    assert (
                        "Machine learning has evolved"
                        in result.agent_outputs["historian"]
                    )

    @pytest.mark.asyncio
    async def test_fallback_to_traditional_on_structured_failure(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test fallback to traditional LLM when structured output fails."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Mock search
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                # Mock structured service to fail
                with patch.object(
                    agent.structured_service,
                    "get_structured_output",
                    new_callable=AsyncMock,
                ) as mock_get_structured:
                    mock_get_structured.side_effect = Exception(
                        "Structured output failed"
                    )

                    result = await agent.run(agent_context)

                    # Verify fallback to traditional LLM
                    assert mock_llm.generate_called
                    assert mock_llm.call_count >= 2  # For relevance and synthesis

                    # Verify context was updated
                    assert "historian" in result.agent_outputs
                    assert "Historical synthesis" in result.agent_outputs["historian"]

    @pytest.mark.asyncio
    async def test_no_structured_service_fallback(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test behavior when structured service is not available."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Disable structured service
            agent.structured_service = None

            # Mock search
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                result = await agent.run(agent_context)

                # Verify traditional LLM was used
                assert mock_llm.generate_called
                assert "historian" in result.agent_outputs

    @pytest.mark.asyncio
    async def test_execution_state_storage(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        structured_historian_output: HistorianOutput,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test that structured output is stored in execution state."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Mock search and structured service
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                with patch.object(
                    agent.structured_service,
                    "get_structured_output",
                    new_callable=AsyncMock,
                ) as mock_get_structured:
                    mock_get_structured.return_value = structured_historian_output

                    result = await agent.run(agent_context)

                    # Verify structured output was stored in execution_state
                    assert "structured_outputs" in result.execution_state
                    assert "historian" in result.execution_state["structured_outputs"]

                    stored_output = result.execution_state["structured_outputs"][
                        "historian"
                    ]
                    assert (
                        stored_output["processing_mode"] == ProcessingMode.ACTIVE.value
                    )
                    assert stored_output["confidence"] == ConfidenceLevel.HIGH.value
                    assert stored_output["sources_searched"] == 10

    @pytest.mark.asyncio
    async def test_hybrid_search_with_structured_output(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        structured_historian_output: HistorianOutput,
    ) -> None:
        """Test that hybrid search works with structured output."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Mock both file and database search
            file_results = [
                SearchResult(
                    title="File Result",
                    excerpt="From file search...",
                    filepath="/notes/file.md",
                    filename="file.md",
                    date="2024-01-01",
                    match_type="content",
                    relevance_score=0.8,
                    metadata={"topics": ["file"], "source": "file"},
                )
            ]

            db_results = [
                SearchResult(
                    title="Database Result",
                    excerpt="From database search...",
                    filepath="db_doc_1",
                    filename="document_1",
                    date="2024-01-02",
                    match_type="content",
                    relevance_score=0.85,
                    metadata={"topics": ["database"], "source": "database"},
                )
            ]

            with patch.object(
                agent, "_search_file_content", new_callable=AsyncMock
            ) as mock_file_search:
                mock_file_search.return_value = file_results

                with patch.object(
                    agent, "_search_database_content", new_callable=AsyncMock
                ) as mock_db_search:
                    mock_db_search.return_value = db_results

                    with patch.object(
                        agent.structured_service,
                        "get_structured_output",
                        new_callable=AsyncMock,
                    ) as mock_get_structured:
                        mock_get_structured.return_value = structured_historian_output

                        result = await agent.run(agent_context)

                        # Verify both search methods were called
                        mock_file_search.assert_called_once()
                        mock_db_search.assert_called_once()

                        # Verify output was generated
                        assert "historian" in result.agent_outputs

    @pytest.mark.asyncio
    async def test_configuration_update(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
    ) -> None:
        """Test configuration update and prompt recomposition."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            initial_config = HistorianConfig(search_depth="shallow")
            agent = HistorianAgent(llm=mock_llm, config=initial_config)

            # Update configuration
            new_config = HistorianConfig(
                search_depth="deep",
                relevance_threshold=0.8,
                hybrid_search_enabled=True,
            )
            agent.update_config(new_config)

            assert agent.config == new_config
            assert agent.config.search_depth == "deep"
            assert agent.config.relevance_threshold == 0.8

    @pytest.mark.asyncio
    async def test_service_initialization_failure(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test graceful handling when service initialization fails."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            with patch(
                "cognivault.agents.historian.agent.LangChainService",
                side_effect=Exception("Service init failed"),
            ):
                agent = HistorianAgent(llm=mock_llm)

                # Should fall back to None
                assert agent.structured_service is None

                # Mock search
                with patch.object(
                    agent, "_search_historical_content", new_callable=AsyncMock
                ) as mock_search:
                    mock_search.return_value = mock_search_results

                    # Should still work with traditional LLM
                    result = await agent.run(agent_context)
                    assert "historian" in result.agent_outputs

    @pytest.mark.asyncio
    async def test_no_llm_available(
        self,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
    ) -> None:
        """Test behavior when no LLM is available."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=None, config=historian_config)

            # Structured service should be None
            assert agent.structured_service is None

            # Mock search to return results
            mock_results = [
                SearchResult(
                    title="Test Result",
                    excerpt="Test excerpt",
                    filepath="/test.md",
                    filename="test.md",
                    date="2024-01-01",
                    match_type="content",
                    relevance_score=0.8,
                    metadata={"topics": ["test"]},
                )
            ]

            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_results

                result = await agent.run(agent_context)

                # Should create basic summary without LLM
                assert "historian" in result.agent_outputs
                assert (
                    "Found 1 relevant historical notes"
                    in result.agent_outputs["historian"]
                )

    def test_type_annotations(self) -> None:
        """Verify type annotations are correct."""
        # This test ensures MyPy compliance
        from typing import get_type_hints

        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            hints = get_type_hints(HistorianAgent.__init__)
            assert hints["config"] == Optional[HistorianConfig]
            assert hints["return"] == type(None)

            hints = get_type_hints(HistorianAgent.run)
            assert hints["context"] == AgentContext
            assert hints["return"] == AgentContext

            hints = get_type_hints(HistorianAgent._run_structured)
            assert hints["query"] == str
            assert hints["context"] == AgentContext
            assert hints["return"] == str


class TestHistorianAgentContentPollutionPrevention:
    """Test content pollution prevention in structured outputs."""

    @pytest.mark.asyncio
    async def test_content_pollution_validation(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test that content pollution is caught by validation."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Mock search
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                with patch.object(
                    agent.structured_service,
                    "get_structured_output",
                    new_callable=AsyncMock,
                ) as mock_get_structured:
                    # Simulate validation error from Pydantic
                    mock_get_structured.side_effect = ValueError(
                        "Content pollution detected: historical_synthesis contains process description"
                    )

                    # Should fall back to traditional
                    result = await agent.run(agent_context)

                    # Verify fallback occurred
                    assert mock_llm.generate_called
                    assert "historian" in result.agent_outputs

    @pytest.mark.asyncio
    async def test_clean_structured_output(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test that clean structured output passes validation."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Create clean output without pollution
            clean_output = HistorianOutput(
                agent_name="historian",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                processing_time_ms=200,
                relevant_sources=[
                    HistoricalReference(
                        source_id=None,
                        title="Clean Source",
                        relevance_score=0.9,
                        content_snippet="Historical facts without process description",
                    )
                ],
                historical_synthesis="Pure historical synthesis focusing on content only",
                themes_identified=["evolution", "innovation"],
                time_periods_covered=["1950s", "2020s"],
                contextual_connections=["Past innovations inform present"],
                sources_searched=15,
                relevant_sources_found=1,
            )

            # Mock search
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                with patch.object(
                    agent.structured_service,
                    "get_structured_output",
                    new_callable=AsyncMock,
                ) as mock_get_structured:
                    mock_get_structured.return_value = clean_output

                    result = await agent.run(agent_context)

                    # Verify clean output was accepted
                    assert "historian" in result.agent_outputs
                    assert (
                        "Pure historical synthesis" in result.agent_outputs["historian"]
                    )
                    # Traditional LLM should still be called for relevance analysis
                    assert mock_llm.generate_called  # For relevance filtering


class TestHistorianAgentTokenUsage:
    """Test token usage tracking for both structured and traditional paths."""

    @pytest.mark.asyncio
    async def test_token_usage_with_structured_output(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        structured_historian_output: HistorianOutput,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test token usage is properly recorded with structured output."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Mock search and structured service
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                with patch.object(
                    agent.structured_service,
                    "get_structured_output",
                    new_callable=AsyncMock,
                ) as mock_get_structured:
                    mock_get_structured.return_value = structured_historian_output

                    result = await agent.run(agent_context)

                    # Verify token usage was recorded
                    assert "historian" in result.agent_token_usage
                    usage = result.agent_token_usage["historian"]
                    # Should have tokens from relevance analysis
                    assert usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_token_usage_with_traditional_path(
        self,
        mock_llm: MockLLM,
        agent_context: AgentContext,
        historian_config: HistorianConfig,
        mock_search_results: List[SearchResult],
    ) -> None:
        """Test token usage is properly recorded with traditional path."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=historian_config)

            # Disable structured service to force traditional path
            agent.structured_service = None

            # Mock search
            with patch.object(
                agent, "_search_historical_content", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results

                result = await agent.run(agent_context)

                # Verify token usage was recorded
                assert "historian" in result.agent_token_usage
                usage = result.agent_token_usage["historian"]
                # Should have accumulated tokens from relevance + synthesis
                assert (
                    usage["total_tokens"] == 150
                )  # 50 from relevance + 100 from synthesis
                assert usage["input_tokens"] == 90  # 30 + 60
                assert usage["output_tokens"] == 60  # 20 + 40
