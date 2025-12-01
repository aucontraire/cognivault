"""
Integration tests for LangChainService that validate actual structured output behavior.

These tests use real models (when API keys are available) or realistic mocks
to validate the complete integration patterns from the LangChain structured output article.

Environment Variables for Real API Tests:
- Real API tests are DISABLED BY DEFAULT for test stability and CI/CD reliability
- To enable real API tests, set BOTH:
  * ENABLE_REAL_API_TESTS=true (explicit opt-in required)
  * SKIP_REAL_API_TESTS=false (override default skip behavior)
- OPENAI_API_KEY: Must be set with valid API key for real tests to work
- CI=true: Automatically skip real API tests in CI environment

Most tests use realistic mocks to ensure consistent behavior without API dependencies.
"""

import pytest
import os
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from unittest.mock import AsyncMock, MagicMock, patch

from cognivault.services.langchain_service import (
    LangChainService,
    StructuredOutputResult,
)
from cognivault.exceptions import LLMError, LLMValidationError


# Real-world test models that match our use cases
class QueryRefinementOutput(BaseModel):
    """Model for query refinement - our main content pollution prevention use case."""

    refined_query: str = Field(
        ...,
        description="The refined query without meta-commentary",
        min_length=5,
        max_length=500,
    )
    confidence_score: float = Field(
        ..., description="Confidence in the refinement quality", ge=0.0, le=1.0
    )
    refinement_type: str = Field(
        ...,
        description="Type of refinement applied",
        pattern=r"^(clarification|expansion|simplification|correction)$",
    )
    original_preserved: bool = Field(
        default=True, description="Whether original intent was preserved"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Additional refinement suggestions"
    )


class SynthesisAnalysisOutput(BaseModel):
    """Model for synthesis analysis with rich validation."""

    summary: str = Field(..., min_length=50, max_length=1000)
    key_insights: List[str] = Field(..., min_length=1, max_length=10)
    confidence_rating: int = Field(..., ge=1, le=5)
    complexity_score: float = Field(..., ge=0.0, le=1.0)
    theme_categories: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class CriticFeedbackOutput(BaseModel):
    """Model for critic feedback with strict validation."""

    critique: str = Field(..., min_length=20, description="Detailed critique")
    strengths: List[str] = Field(..., min_length=1, description="Identified strengths")
    weaknesses: List[str] = Field(
        ..., min_length=1, description="Identified weaknesses"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    severity: str = Field(
        ..., description="Severity level", pattern=r"^(low|medium|high|critical)$"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")


@pytest.fixture
def integration_service() -> LangChainService:
    """Create service configured for integration testing."""
    # Use environment variable for API key - no fallback for integration tests
    api_key = os.getenv("OPENAI_API_KEY")
    return LangChainService(model="gpt-4o", temperature=0.1, api_key=api_key)


@pytest.fixture
def has_openai_key() -> bool:
    """Check if OpenAI API key is available for real integration tests."""
    return bool(os.getenv("OPENAI_API_KEY"))


class TestRealWorldIntegrationPatterns:
    """Test integration with real-world use case patterns."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY", "").startswith("test-key")
        or os.getenv("CI", "false").lower() == "true"
        or os.getenv("SKIP_REAL_API_TESTS", "true").lower()
        == "true"  # Default to skipping
        or not os.getenv("ENABLE_REAL_API_TESTS", "false").lower()
        == "true",  # Require explicit opt-in
        reason="Real API tests disabled by default for stability - set ENABLE_REAL_API_TESTS=true and SKIP_REAL_API_TESTS=false to enable",
    )
    async def test_query_refinement_integration_real(
        self, integration_service: LangChainService
    ) -> None:
        """Test query refinement with real API (when available)."""

        # Real query that needs refinement
        original_query = "tell me about AI"

        result = await integration_service.get_structured_output(
            f"Refine this query to be more specific and detailed: '{original_query}'",
            QueryRefinementOutput,
            system_prompt=(
                "You are a query refinement expert. Refine the user's query to be "
                "more specific and searchable while preserving the original intent. "
                "Do NOT include meta-commentary like 'I refined this' in the output."
            ),
        )

        # Validate structure
        assert isinstance(result, QueryRefinementOutput)
        assert len(result.refined_query) >= 5
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.refinement_type in [
            "clarification",
            "expansion",
            "simplification",
            "correction",
        ]

        # Validate content pollution prevention (our main goal)
        assert "I refined" not in result.refined_query
        assert "I changed" not in result.refined_query
        assert "After analyzing" not in result.refined_query
        assert "Upon review" not in result.refined_query

        # Validate the refinement is actually better
        assert len(result.refined_query) > len(original_query)
        assert result.refined_query.lower() != original_query.lower()

    @pytest.mark.asyncio
    async def test_query_refinement_integration_mock(
        self, integration_service: LangChainService
    ) -> None:
        """Test query refinement with realistic mock (always runs)."""

        # Create realistic mock response
        expected_result = QueryRefinementOutput(
            refined_query="What are the main applications and current capabilities of artificial intelligence technology?",
            confidence_score=0.85,
            refinement_type="expansion",
            original_preserved=True,
            suggestions=[
                "Consider specifying a particular domain",
                "Add time frame for 'current' capabilities",
            ],
        )

        # Mock LLM that returns realistic structured output
        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Refine this query: 'tell me about AI'", QueryRefinementOutput
            )

            # Validate structured output
            assert isinstance(result, QueryRefinementOutput)
            assert result.refinement_type == "expansion"
            assert result.confidence_score == 0.85
            assert result.original_preserved is True

            # Validate content pollution prevention
            assert "I refined" not in result.refined_query
            assert (
                "tell me about" not in result.refined_query.lower()
            )  # Should be improved

            # Verify method was called with correct parameters
            mock_llm.with_structured_output.assert_called_once_with(
                QueryRefinementOutput,
                method="json_schema",  # GPT-4o should use json_schema
                include_raw=False,
            )
        finally:
            integration_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_synthesis_analysis_integration(
        self, integration_service: LangChainService
    ) -> None:
        """Test synthesis analysis with complex validation."""

        # Mock realistic synthesis output
        expected_result = SynthesisAnalysisOutput(
            summary="This analysis reveals three key patterns in machine learning adoption: enterprise integration challenges, data quality requirements, and skill gap concerns. Organizations are increasingly focused on practical implementation rather than theoretical capabilities.",
            key_insights=[
                "Enterprise adoption is limited by integration complexity",
                "Data quality is the primary blocker for ML success",
                "Skills gap is wider than initially estimated",
            ],
            confidence_rating=4,
            complexity_score=0.75,
            theme_categories=["enterprise", "data-quality", "workforce"],
            metadata={"analysis_type": "trend_analysis", "data_sources": 3},
        )

        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Analyze the key themes in machine learning adoption surveys",
                SynthesisAnalysisOutput,
            )

            # Validate complex field validation
            assert isinstance(result, SynthesisAnalysisOutput)
            assert len(result.summary) >= 50
            assert len(result.key_insights) >= 1
            assert 1 <= result.confidence_rating <= 5
            assert 0.0 <= result.complexity_score <= 1.0
        finally:
            integration_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_critic_feedback_integration(
        self, integration_service: LangChainService
    ) -> None:
        """Test critic feedback with strict field validation."""

        expected_result = CriticFeedbackOutput(
            critique="The analysis provides good coverage of technical aspects but lacks discussion of ethical implications and potential societal impacts",
            strengths=["Technical accuracy", "Clear structure", "Good examples"],
            weaknesses=[
                "Missing ethical considerations",
                "Limited scope",
                "No future outlook",
            ],
            suggestions=[
                "Add ethical framework discussion",
                "Include societal impact analysis",
            ],
            severity="medium",
            confidence=0.82,
        )

        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Provide detailed critique of this AI analysis paper",
                CriticFeedbackOutput,
            )

            # Validate strict validation patterns
            assert isinstance(result, CriticFeedbackOutput)
            assert len(result.critique) >= 20
            assert len(result.strengths) >= 1
            assert len(result.weaknesses) >= 1
            assert result.severity in ["low", "medium", "high", "critical"]
            assert 0.0 <= result.confidence <= 1.0
        finally:
            integration_service.llm = original_llm


class TestProviderSpecificMethods:
    """Test provider-specific method selection from the article."""

    @pytest.mark.asyncio
    async def test_gpt_4o_uses_json_schema(self) -> None:
        """Test that GPT-4o uses json_schema method as specified in article."""
        service = LangChainService(model="gpt-4o")
        expected_result = QueryRefinementOutput(
            refined_query="Test refined query",
            confidence_score=0.8,
            refinement_type="clarification",
        )

        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = service.llm
        service.llm = mock_llm

        try:
            await service.get_structured_output("test", QueryRefinementOutput)

            # Verify method selection per article
            mock_llm.with_structured_output.assert_called_once_with(
                QueryRefinementOutput,
                method="json_schema",  # Article specifies this for GPT-4o
                include_raw=False,
            )
        finally:
            service.llm = original_llm

    @pytest.mark.asyncio
    async def test_gpt_35_uses_function_calling(self) -> None:
        """Test that GPT-3.5 uses function_calling method as specified in article."""
        service = LangChainService(model="gpt-3.5-turbo")
        expected_result = QueryRefinementOutput(
            refined_query="Test refined query",
            confidence_score=0.8,
            refinement_type="clarification",
        )

        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = service.llm
        service.llm = mock_llm

        try:
            await service.get_structured_output("test", QueryRefinementOutput)

            # Verify method selection per article
            mock_llm.with_structured_output.assert_called_once_with(
                QueryRefinementOutput,
                method="function_calling",  # Article specifies this for GPT-3.5
                include_raw=False,
            )
        finally:
            service.llm = original_llm

    @pytest.mark.asyncio
    async def test_claude_uses_function_calling(self) -> None:
        """Test that Claude uses function_calling method as specified in article."""
        service = LangChainService(model="claude-3-sonnet")
        expected_result = QueryRefinementOutput(
            refined_query="Test refined query",
            confidence_score=0.8,
            refinement_type="clarification",
        )

        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = service.llm
        service.llm = mock_llm

        try:
            await service.get_structured_output("test", QueryRefinementOutput)

            # Verify method selection per article
            mock_llm.with_structured_output.assert_called_once_with(
                QueryRefinementOutput,
                method="function_calling",  # Article specifies this for Claude
                include_raw=False,
            )
        finally:
            service.llm = original_llm


class TestFallbackBehavior:
    """Test fallback to PydanticOutputParser as described in the article."""

    @pytest.mark.asyncio
    async def test_fallback_parser_integration(
        self, integration_service: LangChainService
    ) -> None:
        """Test complete fallback to PydanticOutputParser when native fails."""

        # Mock response that PydanticOutputParser will need to parse
        mock_response = MagicMock()
        mock_response.content = """
{
    "refined_query": "What are the specific applications, benefits, and limitations of artificial intelligence in healthcare?",
    "confidence_score": 0.9,
    "refinement_type": "expansion",
    "original_preserved": true,
    "suggestions": ["Consider specific healthcare domains", "Add current vs future capabilities"]
}
"""

        # Mock LLM where native structured output fails
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(
            side_effect=AttributeError("Model doesn't support structured output")
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Refine: 'AI in healthcare'", QueryRefinementOutput
            )

            # Verify fallback worked and parsed correctly
            assert isinstance(result, QueryRefinementOutput)
            assert result.refined_query.startswith("What are the specific applications")
            assert result.confidence_score == 0.9
            assert result.refinement_type == "expansion"
            assert result.original_preserved is True
            assert len(result.suggestions) == 2

            # Verify metrics tracked fallback usage
            assert integration_service.metrics["fallback_used"] == 1
            assert integration_service.metrics["total_calls"] == 1
        finally:
            integration_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_fallback_with_format_instructions(
        self, integration_service: LangChainService
    ) -> None:
        """Test that fallback includes PydanticOutputParser format instructions."""

        # Mock response in the format PydanticOutputParser expects
        mock_response = MagicMock()
        mock_response.content = """{"critique": "The analysis demonstrates strong technical understanding but could benefit from broader contextual considerations.", "strengths": ["Technical accuracy", "Clear methodology"], "weaknesses": ["Limited scope", "Missing context"], "suggestions": ["Add broader context", "Include limitations"], "severity": "medium", "confidence": 0.75}"""

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(
            side_effect=Exception("Native method failed")
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Critique this AI analysis", CriticFeedbackOutput
            )

            # Verify result was parsed correctly by fallback
            assert isinstance(result, CriticFeedbackOutput)
            assert result.severity == "medium"
            assert result.confidence == 0.75
            assert len(result.strengths) == 2
            assert len(result.weaknesses) == 2

            # Verify ainvoke was called with enhanced prompt
            call_args = mock_llm.ainvoke.call_args[0][0]
            human_message = call_args[-1][1]  # Last message should be human message

            # Should contain format instructions (article pattern)
            assert "json" in human_message.lower() or "format" in human_message.lower()
        finally:
            integration_service.llm = original_llm


class TestContentPollutionPrevention:
    """Test that structured output prevents content pollution (our main goal)."""

    @pytest.mark.asyncio
    async def test_structured_prevents_meta_commentary(
        self, integration_service: LangChainService
    ) -> None:
        """Test that structured output prevents meta-commentary pollution."""

        # This would be a polluted response in traditional prompting
        clean_result = QueryRefinementOutput(
            refined_query="What are the primary machine learning algorithms used in natural language processing?",
            confidence_score=0.88,
            refinement_type="expansion",
            original_preserved=True,
        )

        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=clean_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Refine this query: 'ML for NLP'", QueryRefinementOutput
            )

            # Verify no meta-commentary pollution
            assert isinstance(result, QueryRefinementOutput)  # Type narrowing
            content = result.refined_query.lower()

            # Common pollution patterns that should NOT appear
            pollution_patterns = [
                "i refined",
                "i changed",
                "i improved",
                "after analyzing",
                "upon review",
                "i made it",
                "i converted",
                "here's the refined",
                "the refined version",
                "i've refined",
            ]

            for pattern in pollution_patterns:
                assert pattern not in content, f"Found pollution pattern: '{pattern}'"

            # Should be clean, direct content
            assert "machine learning" in content
            assert "natural language processing" in content
        finally:
            integration_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_structured_separates_content_and_metadata(
        self, integration_service: LangChainService
    ) -> None:
        """Test that structured output properly separates content from metadata."""

        synthesis_result = SynthesisAnalysisOutput(
            summary="Machine learning adoption in enterprise environments shows three distinct phases: evaluation, pilot implementation, and scaled deployment. Success factors include data readiness, organizational change management, and technical infrastructure alignment.",
            key_insights=[
                "Data quality determines 70% of ML project success",
                "Change management is often underestimated",
                "Technical debt compounds during scaling",
            ],
            confidence_rating=4,
            complexity_score=0.8,
            theme_categories=["enterprise", "adoption", "scaling"],
            metadata={"source_count": 15, "analysis_depth": "comprehensive"},
        )

        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=synthesis_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Synthesize insights from enterprise ML adoption studies",
                SynthesisAnalysisOutput,
            )

            # Verify clean separation of concerns
            assert isinstance(result, SynthesisAnalysisOutput)

            # Content should be clean (no metadata mixed in)
            assert "confidence_rating" not in result.summary
            assert "complexity_score" not in result.summary
            assert str(result.confidence_rating) not in result.summary

            # Metadata should be properly structured
            assert isinstance(result.confidence_rating, int)
            assert isinstance(result.complexity_score, float)
            assert isinstance(result.key_insights, list)
            assert isinstance(result.theme_categories, list)

            # Content should focus on actual insights
            assert "machine learning adoption" in result.summary.lower()
            assert "enterprise" in result.summary.lower()
        finally:
            integration_service.llm = original_llm


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    @pytest.mark.asyncio
    async def test_validation_error_with_detailed_context(
        self, integration_service: LangChainService
    ) -> None:
        """Test that validation errors provide detailed context."""

        # Mock response that will fail validation
        invalid_response = MagicMock()
        invalid_response.content = '{"invalid_field": "wrong structure"}'

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(
            side_effect=Exception("Native failed")
        )
        mock_llm.ainvoke = AsyncMock(return_value=invalid_response)

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            with pytest.raises(LLMValidationError) as exc_info:
                await integration_service.get_structured_output(
                    "Generate query refinement", QueryRefinementOutput
                )

            # Verify error context is detailed
            error = exc_info.value
            assert "Failed to get structured output" in error.message
            assert "QueryRefinementOutput" in error.message
            assert error.model_name == "gpt-4o"
            assert "fallback_attempted" in error.context
            assert error.context["fallback_attempted"] is True
        finally:
            integration_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_retry_logic_integration(
        self, integration_service: LangChainService
    ) -> None:
        """Test retry logic with transient failures."""
        call_count = 0
        expected_result = QueryRefinementOutput(
            refined_query="Successfully refined after retries",
            confidence_score=0.9,
            refinement_type="clarification",
        )

        def create_failing_then_succeeding_mock() -> MagicMock:
            nonlocal call_count
            mock_llm = MagicMock()

            def with_structured_side_effect(*args: Any, **kwargs: Any) -> AsyncMock:
                nonlocal call_count
                call_count += 1

                mock_structured = AsyncMock()
                if call_count < 2:  # Fail first attempt
                    mock_structured.ainvoke = AsyncMock(
                        side_effect=Exception("Transient error")
                    )
                else:  # Succeed on second attempt
                    mock_structured.ainvoke = AsyncMock(return_value=expected_result)

                return mock_structured

            mock_llm.with_structured_output = MagicMock(
                side_effect=with_structured_side_effect
            )
            return mock_llm

        mock_llm = create_failing_then_succeeding_mock()

        original_llm = integration_service.llm
        integration_service.llm = mock_llm

        try:
            result = await integration_service.get_structured_output(
                "Test retry logic", QueryRefinementOutput, max_retries=3
            )

            # Verify success after retry
            assert isinstance(result, QueryRefinementOutput)
            assert result.refined_query == "Successfully refined after retries"
            assert call_count == 2  # Should have retried once
        finally:
            integration_service.llm = original_llm
