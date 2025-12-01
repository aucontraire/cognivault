"""
Tests for LangChainService with structured output support.

Tests the implementation of patterns from the LangChain structured output article.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError

from cognivault.services.langchain_service import (
    LangChainService,
    StructuredOutputResult,
)
from cognivault.exceptions import LLMError, LLMValidationError


# Test models with strict typing (renamed to avoid pytest collection warnings)
class OutputModel(BaseModel):
    """Test output model for structured output testing."""

    content: str = Field(..., description="Main content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata"
    )


class ComplexModel(BaseModel):
    """More complex test model."""

    summary: str = Field(..., min_length=10, max_length=200)
    score: int = Field(..., ge=1, le=10)
    categories: List[str] = Field(..., min_length=1)  # Fix deprecated min_items
    is_valid: bool = Field(default=True)


@pytest.fixture
def langchain_service() -> LangChainService:
    """Create LangChainService instance for testing."""
    return LangChainService(model="gpt-4o", temperature=0.1)


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Mock OpenAI response object."""
    response = MagicMock()
    response.content = (
        '{"content": "test content", "confidence": 0.95, "tags": ["test"]}'
    )
    return response


class TestLangChainServiceInit:
    """Test LangChainService initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        service = LangChainService(use_pool=False, use_discovery=False)

        # The service uses gpt-4o as the default fallback model
        assert service.model_name == "gpt-4o"
        assert service.llm is not None
        assert service.metrics["total_calls"] == 0

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        service = LangChainService(
            model="gpt-3.5-turbo", temperature=0.5, api_key="test-key"
        )

        assert service.model_name == "gpt-3.5-turbo"
        assert service.llm is not None

    def test_provider_method_selection(
        self, langchain_service: LangChainService
    ) -> None:
        """Test provider-specific method selection."""
        # Test GPT models
        assert (
            langchain_service._get_structured_output_method("gpt-4o") == "json_schema"
        )
        assert (
            langchain_service._get_structured_output_method("gpt-4")
            == "function_calling"
        )

        # Test Claude models
        assert (
            langchain_service._get_structured_output_method("claude-3-sonnet")
            == "function_calling"
        )

        # Test unknown models
        assert (
            langchain_service._get_structured_output_method("unknown-model")
            == "json_mode"
        )


class TestStructuredOutput:
    """Test structured output functionality."""

    @pytest.mark.asyncio
    async def test_successful_native_structured_output(
        self, langchain_service: LangChainService
    ) -> None:
        """Test successful native structured output."""
        expected_result = OutputModel(
            content="test content", confidence=0.95, tags=["test"]
        )

        # Create a mock LLM with the with_structured_output method
        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            result = await langchain_service.get_structured_output(
                "test prompt", OutputModel
            )

            # Verify result
            assert isinstance(result, OutputModel)
            assert result.content == "test content"
            assert result.confidence == 0.95
            assert result.tags == ["test"]

            # Verify method was called with correct parameters
            mock_llm.with_structured_output.assert_called_once_with(
                OutputModel, method="json_schema", include_raw=False
            )

            # Verify metrics
            assert langchain_service.metrics["total_calls"] == 1
            assert langchain_service.metrics["successful_structured"] == 1
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_structured_output_with_raw(
        self, langchain_service: LangChainService
    ) -> None:
        """Test structured output with raw response included."""
        expected_result = {
            "parsed": OutputModel(content="test", confidence=0.8, tags=[]),
            "raw": "raw response content",
        }

        # Create a mock LLM with the with_structured_output method
        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            result = await langchain_service.get_structured_output(
                "test prompt", OutputModel, include_raw=True
            )

            # Verify result structure
            assert isinstance(result, StructuredOutputResult)
            assert isinstance(result.parsed, OutputModel)
            assert result.raw == "raw response content"
            assert result.method_used == "json_schema"
            assert result.fallback_used is False
            assert result.processing_time_ms is not None

            # Verify method was called with correct parameters
            mock_llm.with_structured_output.assert_called_once_with(
                OutputModel, method="json_schema", include_raw=True
            )
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_fallback_to_parser(
        self, langchain_service: LangChainService, mock_openai_response: MagicMock
    ) -> None:
        """Test fallback to PydanticOutputParser when native fails."""

        # Create mock LLM that fails structured output but works with ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(
            side_effect=AttributeError("Method not supported")
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_openai_response)

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            result = await langchain_service.get_structured_output(
                "test prompt", OutputModel
            )

            # Verify result from parser
            assert isinstance(result, OutputModel)
            assert result.content == "test content"
            assert result.confidence == 0.95
            assert result.tags == ["test"]

            # Verify metrics show fallback was used
            assert langchain_service.metrics["fallback_used"] == 1
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_validation_error_handling(
        self, langchain_service: LangChainService
    ) -> None:
        """Test handling of validation errors."""
        # Mock invalid response that can't be parsed
        invalid_response = MagicMock()
        invalid_response.content = '{"invalid": "structure"}'

        # Create mock LLM that fails structured output and returns invalid data
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(
            side_effect=Exception("Native failed")
        )
        mock_llm.ainvoke = AsyncMock(return_value=invalid_response)

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            with pytest.raises(LLMValidationError) as exc_info:
                await langchain_service.get_structured_output(
                    "test prompt", OutputModel
                )

            # Verify error details
            error = exc_info.value
            assert "Failed to get structured output" in error.message
            assert error.model_name == "gpt-4o"
            assert "fallback_attempted" in error.context
            assert langchain_service.metrics["validation_failures"] == 1
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_retry_logic(self, langchain_service: LangChainService) -> None:
        """Test retry logic for transient failures."""
        call_count = 0
        expected_result = OutputModel(content="success", confidence=1.0, tags=[])

        def mock_with_structured_side_effect(*_args: Any, **_kwargs: Any) -> AsyncMock:
            nonlocal call_count
            call_count += 1

            inner_mock_llm = AsyncMock()
            if call_count < 3:  # Fail first 2 attempts
                inner_mock_llm.ainvoke = AsyncMock(
                    side_effect=Exception("Transient error")
                )
            else:  # Succeed on 3rd attempt
                inner_mock_llm.ainvoke = AsyncMock(return_value=expected_result)

            return inner_mock_llm

        # Create mock LLM with retry behavior
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(
            side_effect=mock_with_structured_side_effect
        )

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            result = await langchain_service.get_structured_output(
                "test prompt", OutputModel, max_retries=3
            )

            # Verify success after retries
            assert isinstance(result, OutputModel)
            assert result.content == "success"
            assert call_count == 3  # Verify it tried 3 times
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_system_prompt_handling(
        self, langchain_service: LangChainService
    ) -> None:
        """Test that system prompts are properly included."""
        expected_result = OutputModel(content="test", confidence=0.7, tags=[])

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)

        mock_with_structured_output = MagicMock(return_value=mock_structured_llm)

        # Create mock LLM with structured output
        mock_llm = MagicMock()
        mock_llm.with_structured_output = mock_with_structured_output

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            await langchain_service.get_structured_output(
                "user prompt", OutputModel, system_prompt="system instructions"
            )

            # Verify ainvoke was called with both system and human messages
            call_args = mock_structured_llm.ainvoke.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0] == ("system", "system instructions")
            assert call_args[1] == ("human", "user prompt")
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm


class TestComplexModels:
    """Test with more complex Pydantic models."""

    @pytest.mark.asyncio
    async def test_complex_model_validation(
        self, langchain_service: LangChainService
    ) -> None:
        """Test structured output with complex validation rules."""
        expected_result = ComplexModel(
            summary="This is a valid summary with sufficient length",
            score=8,
            categories=["test", "validation"],
            is_valid=True,
        )

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)

        mock_with_structured_output = MagicMock(return_value=mock_structured_llm)

        # Create mock LLM with structured output
        mock_llm = MagicMock()
        mock_llm.with_structured_output = mock_with_structured_output

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            result = await langchain_service.get_structured_output(
                "test prompt", ComplexModel
            )

            # Verify complex model validation
            assert isinstance(result, ComplexModel)
            assert len(result.summary) >= 10
            assert 1 <= result.score <= 10
            assert len(result.categories) >= 1
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

    def test_model_field_validation(self) -> None:
        """Test that Pydantic field validation works correctly."""
        # Test valid model
        valid_model = ComplexModel(
            summary="Valid summary text here", score=5, categories=["category1"]
        )
        assert valid_model.is_valid is True

        # Test invalid models raise ValidationError
        with pytest.raises(ValidationError):
            ComplexModel(
                summary="Short",  # Too short (min_length=10)
                score=5,
                categories=["category1"],
            )

        with pytest.raises(ValidationError):
            ComplexModel(
                summary="Valid summary text here",
                score=11,  # Too high (le=10)
                categories=["category1"],
            )

        with pytest.raises(ValidationError):
            ComplexModel(
                summary="Valid summary text here",
                score=5,
                categories=[],  # Empty (min_items=1)
            )


class TestServiceMetrics:
    """Test service metrics and cache functionality."""

    def test_get_cache_stats_empty(self, langchain_service: LangChainService) -> None:
        """Test cache stats when no calls have been made."""
        stats = langchain_service.get_cache_stats()

        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["fallback_rate"] == 0.0
        assert stats["validation_failure_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, langchain_service: LangChainService) -> None:
        """Test that metrics are properly tracked."""
        expected_result = OutputModel(content="test", confidence=0.8, tags=[])

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_result)

        mock_with_structured_output = MagicMock(return_value=mock_structured_llm)

        # Create mock LLM with structured output
        mock_llm = MagicMock()
        mock_llm.with_structured_output = mock_with_structured_output

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            # Make a successful call
            await langchain_service.get_structured_output("test prompt", OutputModel)
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

        stats = langchain_service.get_cache_stats()

        assert stats["total_calls"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["fallback_rate"] == 0.0
        assert stats["validation_failure_rate"] == 0.0

    def test_clear_cache(self, langchain_service: LangChainService) -> None:
        """Test cache clearing functionality."""
        # Set some metrics
        langchain_service.metrics["total_calls"] = 5
        langchain_service.metrics["successful_structured"] = 3

        # Clear cache
        langchain_service.clear_cache()

        # Verify metrics are reset
        assert langchain_service.metrics["total_calls"] == 0
        assert langchain_service.metrics["successful_structured"] == 0


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_llm_error_propagation(
        self, langchain_service: LangChainService
    ) -> None:
        """Test that LLM errors are properly propagated."""
        mock_with_structured_failing = MagicMock(
            side_effect=AttributeError("Model doesn't support structured output")
        )
        mock_ainvoke_failing = AsyncMock(side_effect=Exception("Network error"))

        # Create mock LLM with failing methods
        mock_llm = MagicMock()
        mock_llm.with_structured_output = mock_with_structured_failing
        mock_llm.ainvoke = mock_ainvoke_failing

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            with pytest.raises(LLMValidationError):
                await langchain_service.get_structured_output(
                    "test prompt", OutputModel
                )
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm

    @pytest.mark.asyncio
    async def test_invalid_model_class(
        self, langchain_service: LangChainService
    ) -> None:
        """Test handling of invalid model classes."""
        # This test ensures we handle non-BaseModel classes gracefully
        mock_with_structured_failing = MagicMock(
            side_effect=Exception("Invalid model class")
        )

        # Create mock LLM with failing structured output
        mock_llm = MagicMock()
        mock_llm.with_structured_output = mock_with_structured_failing

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            with pytest.raises(LLMValidationError):
                # This should fail because str is not a BaseModel
                await langchain_service.get_structured_output(
                    "test prompt",
                    str,  # type: ignore
                )
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm


# Integration-style tests
class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_content_pollution_prevention(
        self, langchain_service: LangChainService
    ) -> None:
        """Test that structured output prevents content pollution."""
        # Simulate a response with meta-commentary (the pollution we want to prevent)
        clean_result = OutputModel(
            content="Clean content without meta-commentary",  # No pollution
            confidence=0.9,
            tags=["clean", "structured"],
        )

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=clean_result)

        mock_with_structured_output = MagicMock(return_value=mock_structured_llm)

        # Create mock LLM with structured output
        mock_llm = MagicMock()
        mock_llm.with_structured_output = mock_with_structured_output

        # Replace the service's LLM with our mock
        original_llm = langchain_service.llm
        langchain_service.llm = mock_llm

        try:
            result = await langchain_service.get_structured_output(
                "Refine this query: What is AI?", OutputModel
            )

            # Verify clean separation
            assert isinstance(result, OutputModel)
            assert "I refined" not in result.content
            assert "I changed" not in result.content
            assert result.confidence is not None
            assert isinstance(result.tags, list)
        finally:
            # Restore original LLM
            langchain_service.llm = original_llm
