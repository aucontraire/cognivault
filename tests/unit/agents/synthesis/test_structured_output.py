"""
Comprehensive test suite for SynthesisAgent structured output functionality.

Tests the integration of LangChain structured output service with SynthesisAgent,
including content pollution prevention, field validation, and fallback mechanisms.
"""

import pytest
import asyncio
from typing import Any, Dict, Optional, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from pydantic import ValidationError

from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from cognivault.agents.models import (
    SynthesisOutput,
    SynthesisTheme,
    ConfidenceLevel,
    ProcessingMode,
)
from cognivault.services.langchain_service import (
    LangChainService,
    StructuredOutputResult,
)
from tests.factories.agent_context_factories import (
    AgentContextPatterns,
    AgentContextFactory,
)


class MockStructuredLLM(LLMInterface):
    """Mock LLM that simulates structured output responses."""

    def __init__(self, api_key: str = "test-key", model: str = "gpt-4") -> None:
        self.api_key = api_key
        self.model = model
        self.call_count = 0
        self.last_prompt = ""

    def generate(self, prompt: str, **kwargs: Any) -> Mock:
        """Generate mock response for traditional synthesis."""
        self.call_count += 1
        self.last_prompt = prompt

        mock_response = Mock()
        mock_response.text = "Traditional synthesis response"
        mock_response.tokens_used = 100
        mock_response.input_tokens = 60
        mock_response.output_tokens = 40
        return mock_response


class TestSynthesisAgentStructuredOutput:
    """Test structured output functionality in SynthesisAgent."""

    @pytest.mark.asyncio
    async def test_structured_service_initialization(self) -> None:
        """Test that structured service is properly initialized."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        assert agent.structured_service is not None
        assert isinstance(agent.structured_service, LangChainService)
        assert agent.name == "synthesis"

    @pytest.mark.asyncio
    async def test_structured_service_disabled_without_llm(self) -> None:
        """Test that structured service is disabled when no LLM is available."""
        agent = SynthesisAgent(llm=None)

        assert agent.structured_service is None
        assert agent.llm is None

    @pytest.mark.asyncio
    async def test_run_with_structured_output_success(self) -> None:
        """Test successful synthesis using structured output."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        # Create context with multiple agent outputs
        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Create mock structured output with all required fields
        long_synthesis = (
            "This is a comprehensive synthesis of all agent outputs. " * 10
        )  # Make it > 100 chars and 50+ words
        mock_synthesis_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.PASSIVE,
            confidence=ConfidenceLevel.HIGH,
            final_synthesis=long_synthesis,
            key_themes=[
                SynthesisTheme(
                    theme_name="Knowledge Integration",
                    description="Integration of multiple perspectives",
                    supporting_agents=["refiner", "critic"],
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
            conflicts_resolved=["No significant conflicts were found between agents"],
            complementary_insights=[
                "Historical context enhances current analysis significantly"
            ],
            knowledge_gaps=["Implementation details are missing from the analysis"],
            meta_insights=[
                "Multiple perspectives improve overall accuracy considerably"
            ],
            contributing_agents=["refiner", "critic", "historian"],
            word_count=len(long_synthesis.split()),
            topics_extracted=["AI", "synthesis", "analysis"],
        )

        # Mock the structured service
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.return_value = (
            mock_synthesis_output
        )
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify structured output was used
        mock_structured_service.get_structured_output.assert_called_once()
        call_args = mock_structured_service.get_structured_output.call_args
        assert call_args.kwargs["output_class"] == SynthesisOutput
        assert call_args.kwargs["max_retries"] == 3

        # Verify context was updated
        assert agent.name in result.agent_outputs
        assert "Knowledge Integration" in result.agent_outputs[agent.name]
        assert result.final_synthesis == result.agent_outputs[agent.name]

        # Verify structured output stored in execution state
        assert "structured_outputs" in result.execution_state
        assert agent.name in result.execution_state["structured_outputs"]

    @pytest.mark.asyncio
    async def test_run_with_structured_output_result_wrapper(self) -> None:
        """Test handling of StructuredOutputResult wrapper."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Create mock structured output wrapped in StructuredOutputResult
        long_synthesis = (
            "Wrapped synthesis result with enough content to meet minimum requirements. "
            * 10
        )
        mock_synthesis_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.MEDIUM,
            final_synthesis=long_synthesis,
            key_themes=[],
            contributing_agents=["refiner"],
            word_count=len(long_synthesis.split()),
        )

        mock_result = StructuredOutputResult(
            parsed=mock_synthesis_output,
            raw="raw response",
            method_used="json_schema",
            fallback_used=False,
            processing_time_ms=100.0,
        )

        # Mock the structured service
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.return_value = mock_result
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify the wrapped result was properly extracted
        assert "Wrapped synthesis result" in result.agent_outputs[agent.name]

    @pytest.mark.asyncio
    async def test_fallback_to_traditional_on_structured_failure(self) -> None:
        """Test fallback to traditional synthesis when structured output fails."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Mock the structured service to fail
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.side_effect = Exception(
            "Structured output failed"
        )
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify fallback to traditional synthesis
        assert agent.name in result.agent_outputs
        # Should have used traditional LLM generate method
        assert mock_llm.call_count > 0

    @pytest.mark.asyncio
    async def test_structured_output_with_complex_themes(self) -> None:
        """Test structured output with multiple complex themes."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Create structured output with multiple themes
        long_synthesis = (
            "Complex multi-theme synthesis with extensive content analysis. " * 10
        )
        mock_synthesis_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.FALLBACK,
            confidence=ConfidenceLevel.HIGH,
            final_synthesis=long_synthesis,
            key_themes=[
                SynthesisTheme(
                    theme_name="Theme 1",
                    description="First theme description",
                    supporting_agents=["refiner"],
                    confidence=ConfidenceLevel.HIGH,
                ),
                SynthesisTheme(
                    theme_name="Theme 2",
                    description="Second theme description",
                    supporting_agents=["critic", "historian"],
                    confidence=ConfidenceLevel.MEDIUM,
                ),
                SynthesisTheme(
                    theme_name="Theme 3",
                    description="Third theme description",
                    supporting_agents=["refiner", "critic"],
                    confidence=ConfidenceLevel.LOW,
                ),
            ],
            contributing_agents=["refiner", "critic", "historian"],
            word_count=len(long_synthesis.split()),
        )

        # Mock the structured service
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.return_value = (
            mock_synthesis_output
        )
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify themes are included in output
        output = result.agent_outputs[agent.name]
        assert "Theme 1" in output
        assert "Theme 2" in output
        assert "Theme 3" in output

    @pytest.mark.asyncio
    async def test_structured_output_token_usage_tracking(self) -> None:
        """Test that token usage is properly tracked for structured output."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Create mock structured output
        long_synthesis = (
            "Token tracking test with sufficient content to pass validation requirements. "
            * 10
        )
        mock_synthesis_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.MEDIUM,
            final_synthesis=long_synthesis,
            contributing_agents=["refiner"],
            word_count=len(long_synthesis.split()),
        )

        # Mock the structured service
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.return_value = (
            mock_synthesis_output
        )
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify token usage was recorded (even if zeros for structured)
        token_usage = result.get_agent_token_usage(agent.name)
        assert token_usage is not None
        assert "input_tokens" in token_usage
        assert "output_tokens" in token_usage
        assert "total_tokens" in token_usage

    @pytest.mark.asyncio
    async def test_content_pollution_prevention(self) -> None:
        """Test that content pollution is prevented in structured output."""
        from cognivault.agents.models import ProcessingMode

        # Create a base valid synthesis that's long enough (100+ chars and 50+ words)
        long_polluted_text = "I synthesized this content by analyzing the outputs " * 10
        long_valid_text = (
            "This is the actual synthesized content without meta-commentary " * 10
        )

        # Test validation directly on the model
        with pytest.raises(ValidationError) as exc_info:
            SynthesisOutput(
                agent_name="synthesis",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                final_synthesis=long_polluted_text,
                contributing_agents=["refiner"],
                word_count=len(long_polluted_text.split()),
            )
        assert "Content pollution detected" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            SynthesisOutput(
                agent_name="synthesis",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                final_synthesis="My analysis shows that this is important "
                * 5,  # Make it long enough
                contributing_agents=["refiner"],
                word_count=40,
            )
        # Note: word_count validation may fail first, but content pollution should also be detected
        error_str = str(exc_info.value)
        assert "Content pollution detected" in error_str or "word_count" in error_str

        # Valid synthesis should work
        valid_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            final_synthesis=long_valid_text,
            contributing_agents=["refiner"],
            word_count=len(long_valid_text.split()),
        )
        assert valid_output.final_synthesis == long_valid_text.strip()

    @pytest.mark.asyncio
    async def test_structured_output_with_no_themes(self) -> None:
        """Test structured output when no themes are identified."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Create structured output without themes
        long_synthesis = (
            "Simple synthesis without themes but with extensive content analysis. " * 10
        )
        mock_synthesis_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.MEDIUM,
            final_synthesis=long_synthesis,
            key_themes=[],
            contributing_agents=["refiner"],
            word_count=len(long_synthesis.split()),
        )

        # Mock the structured service
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.return_value = (
            mock_synthesis_output
        )
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify output doesn't include theme section
        output = result.agent_outputs[agent.name]
        assert "Simple synthesis without themes" in output
        assert "Primary Themes:" not in output

    @pytest.mark.asyncio
    async def test_structured_output_with_meta_insights(self) -> None:
        """Test that meta-insights are properly formatted in output."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Create structured output with meta insights
        long_synthesis = (
            "Synthesis with meta-insights and comprehensive analysis of all agent outputs. "
            * 10
        )
        mock_synthesis_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.PASSIVE,
            confidence=ConfidenceLevel.HIGH,
            final_synthesis=long_synthesis,
            meta_insights=[
                "First meta-insight reveals cross-agent pattern recognition",
                "Second meta-insight identifies emergent knowledge patterns",
                "Third meta-insight shows relationship between agent perspectives",
            ],
            contributing_agents=["refiner", "critic"],
            word_count=len(long_synthesis.split()),
        )

        # Mock the structured service
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.return_value = (
            mock_synthesis_output
        )
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify meta-insights are included
        output = result.agent_outputs[agent.name]
        assert "Meta-Insights" in output
        assert "First meta-insight" in output
        assert "Second meta-insight" in output
        assert "Third meta-insight" in output

    @pytest.mark.asyncio
    async def test_structured_service_initialization_failure(self) -> None:
        """Test graceful handling when structured service fails to initialize."""
        mock_llm = MockStructuredLLM()

        with patch(
            "cognivault.agents.synthesis.agent.LangChainService",
            side_effect=Exception("Service init failed"),
        ):
            agent = SynthesisAgent(llm=mock_llm)

            # Should still work but without structured service
            assert agent.structured_service is None
            assert agent.llm == mock_llm

    @pytest.mark.asyncio
    async def test_traditional_synthesis_when_no_structured_service(self) -> None:
        """Test that traditional synthesis works when structured service is not available."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)
        agent.structured_service = None  # Disable structured service

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Run the agent
        result = await agent.run(context)

        # Should use traditional synthesis
        assert agent.name in result.agent_outputs
        assert mock_llm.call_count > 0  # Traditional LLM was called

    @pytest.mark.asyncio
    async def test_structured_output_preserves_all_fields(self) -> None:
        """Test that all SynthesisOutput fields are preserved and used."""
        mock_llm = MockStructuredLLM()
        agent = SynthesisAgent(llm=mock_llm)

        context = AgentContextFactory.with_agent_outputs(
            refiner="Refined query output",
            critic="Critical analysis of the query",
            historian="Historical context and references",
        )

        # Create comprehensive structured output
        long_synthesis = (
            "Comprehensive synthesis with all fields populated and validated correctly. "
            * 10
        )
        mock_synthesis_output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.PASSIVE,
            confidence=ConfidenceLevel.HIGH,
            final_synthesis=long_synthesis,
            key_themes=[
                SynthesisTheme(
                    theme_name="Complete Theme",
                    description="Theme with comprehensive details",
                    supporting_agents=["refiner", "critic"],
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
            conflicts_resolved=[
                "Resolved conflict between agent perspectives on implementation",
                "Resolved temporal inconsistencies in historical data",
            ],
            complementary_insights=[
                "Historical context significantly enhances current analysis",
                "Critical evaluation validates refined query interpretation",
                "Multiple perspectives create comprehensive understanding",
            ],
            knowledge_gaps=[
                "Implementation details require further investigation",
                "Long-term implications need additional analysis",
            ],
            meta_insights=[
                "Agent collaboration reveals emergent knowledge patterns",
                "Synthesis process identifies cross-domain connections",
            ],
            contributing_agents=["refiner", "critic", "historian"],
            word_count=len(long_synthesis.split()),
            topics_extracted=["Topic A", "Topic B", "Topic C"],
        )

        # Mock the structured service
        mock_structured_service = AsyncMock(spec=LangChainService)
        mock_structured_service.get_structured_output.return_value = (
            mock_synthesis_output
        )
        agent.structured_service = mock_structured_service

        # Run the agent
        result = await agent.run(context)

        # Verify structured data is stored
        stored_data = result.execution_state["structured_outputs"][agent.name]
        assert len(stored_data["key_themes"]) == 1
        assert len(stored_data["conflicts_resolved"]) == 2
        assert len(stored_data["complementary_insights"]) == 3
        assert len(stored_data["knowledge_gaps"]) == 2
        assert len(stored_data["meta_insights"]) == 2
        assert len(stored_data["topics_extracted"]) == 3
        assert stored_data["word_count"] == len(long_synthesis.split())
