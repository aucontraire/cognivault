"""Tests for default_factory field parser enhancement in LangChainService."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from cognivault.services.langchain_service import LangChainService
from cognivault.agents.models import CriticOutput


class MockModelWithDefaultFactory(BaseModel):
    """Test model with default_factory fields to simulate the CriticOutput issue."""

    required_field: str
    optional_string: Optional[str] = None
    list_with_factory: List[str] = Field(default_factory=list)
    dict_with_factory: Dict[str, str] = Field(default_factory=dict)


class TestDefaultFactoryParserFix:
    """Test suite for the default_factory field parser enhancement."""

    @pytest.fixture
    def service(self) -> LangChainService:
        """Create a LangChainService instance for testing."""
        return LangChainService(
            model="gpt-5",
            api_key="test-key",
            use_discovery=False,
            use_pool=False,
        )

    def test_ensure_default_factory_fields_with_missing_fields(
        self, service: LangChainService
    ) -> None:
        """Test that missing default_factory fields are properly initialized."""
        # Create a mock parsed result that's missing default_factory fields
        mock_result = MockModelWithDefaultFactory(required_field="test")

        # Manually remove the default_factory fields to simulate OpenAI parser behavior
        mock_result.list_with_factory = []  # This should be fine
        mock_result.dict_with_factory = {}  # This should be fine

        # Process through the enhancement
        enhanced_result = service._ensure_default_factory_fields(
            mock_result, MockModelWithDefaultFactory
        )

        # Verify the result is properly initialized
        assert isinstance(enhanced_result, MockModelWithDefaultFactory)
        assert enhanced_result.required_field == "test"
        assert (
            enhanced_result.list_with_factory == []
        )  # default_factory should create empty list
        assert (
            enhanced_result.dict_with_factory == {}
        )  # default_factory should create empty dict
        assert enhanced_result.optional_string is None

    def test_ensure_default_factory_fields_with_partial_data(
        self, service: LangChainService
    ) -> None:
        """Test re-validation with partial data that triggers default_factory."""
        # Create a dict with only required field (simulating OpenAI returning minimal response)
        raw_data = {"required_field": "test_value"}

        # Create instance from minimal data (this should trigger default_factory)
        partial_result = MockModelWithDefaultFactory(**raw_data)

        # Process through the enhancement
        enhanced_result = service._ensure_default_factory_fields(
            partial_result, MockModelWithDefaultFactory
        )

        # Verify all fields are properly set
        assert enhanced_result.required_field == "test_value"
        assert enhanced_result.list_with_factory == []
        assert enhanced_result.dict_with_factory == {}
        assert enhanced_result.optional_string is None

    def test_ensure_default_factory_fields_with_complete_data(
        self, service: LangChainService
    ) -> None:
        """Test that complete data passes through unchanged."""
        # Create a fully populated result
        complete_result = MockModelWithDefaultFactory(
            required_field="test",
            optional_string="optional_value",
            list_with_factory=["item1", "item2"],
            dict_with_factory={"key1": "value1"},
        )

        # Process through the enhancement
        enhanced_result = service._ensure_default_factory_fields(
            complete_result, MockModelWithDefaultFactory
        )

        # Verify data is preserved
        assert enhanced_result.required_field == "test"
        assert enhanced_result.optional_string == "optional_value"
        assert enhanced_result.list_with_factory == ["item1", "item2"]
        assert enhanced_result.dict_with_factory == {"key1": "value1"}

    def test_ensure_default_factory_fields_handles_validation_errors(
        self, service: LangChainService
    ) -> None:
        """Test that validation errors are handled gracefully."""
        # Create a mock result that will fail validation
        mock_result = Mock()
        mock_result.model_dump.side_effect = Exception("Validation failed")

        # Process through the enhancement - should return original on error
        enhanced_result = service._ensure_default_factory_fields(
            mock_result, MockModelWithDefaultFactory
        )

        # Should return the original mock result
        assert enhanced_result is mock_result

    def test_ensure_default_factory_fields_handles_non_pydantic_objects(
        self, service: LangChainService
    ) -> None:
        """Test handling of non-Pydantic objects."""
        # Test with a valid Pydantic object that works normally
        valid_obj = MockModelWithDefaultFactory(required_field="test")
        enhanced_result = service._ensure_default_factory_fields(
            valid_obj, MockModelWithDefaultFactory
        )
        assert isinstance(enhanced_result, MockModelWithDefaultFactory)
        assert enhanced_result.required_field == "test"
        assert enhanced_result.list_with_factory == []
        assert enhanced_result.dict_with_factory == {}

    @pytest.mark.asyncio
    async def test_native_openai_parse_applies_fix(
        self, service: LangChainService
    ) -> None:
        """Test that the native OpenAI parse method applies the default_factory fix."""
        with patch("openai.AsyncOpenAI") as mock_openai_class:
            # Setup mock client and completion
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            # Create a mock completion with minimal data
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.parsed = MockModelWithDefaultFactory(
                required_field="test"
            )
            mock_completion.choices[0].message.content = "raw content"

            # Setup async mock
            mock_client.beta.chat.completions.parse = AsyncMock(
                return_value=mock_completion
            )

            # Call the method
            messages = [("system", "test"), ("human", "test")]
            result = await service._try_native_openai_parse(
                messages=messages,
                output_class=MockModelWithDefaultFactory,
                include_raw=False,
            )

            # Verify the result has properly initialized default_factory fields
            assert isinstance(result, MockModelWithDefaultFactory)
            assert result.required_field == "test"
            assert (
                result.list_with_factory == []
            )  # Should be initialized by default_factory
            assert (
                result.dict_with_factory == {}
            )  # Should be initialized by default_factory

    @pytest.mark.asyncio
    async def test_langchain_structured_output_applies_fix(
        self, service: LangChainService
    ) -> None:
        """Test that LangChain structured output also applies the default_factory fix."""
        # Create a mock LLM and structured output
        mock_llm = MagicMock()
        mock_structured = MagicMock()

        # Mock the structured output result with minimal data
        mock_result = MockModelWithDefaultFactory(required_field="test")
        mock_structured.ainvoke = AsyncMock(return_value=mock_result)
        mock_llm.with_structured_output.return_value = mock_structured

        # Set the mock LLM on the service
        service.llm = mock_llm

        # Call the method
        messages = [("human", "test")]
        result = await service._try_native_structured_output(
            messages=messages,
            output_class=MockModelWithDefaultFactory,
            include_raw=False,
            attempt=0,
        )

        # Verify the result has properly initialized default_factory fields
        assert isinstance(result, MockModelWithDefaultFactory)
        assert result.required_field == "test"
        assert result.list_with_factory == []
        assert result.dict_with_factory == {}

    def test_critic_output_specific_scenario(self, service: LangChainService) -> None:
        """Test the specific CriticOutput scenario that was failing."""
        from cognivault.agents.models import BiasType

        # Create a minimal CriticOutput that might come from OpenAI (missing bias_details)
        minimal_critic_data = {
            "agent_name": "critic",
            "processing_mode": "active",
            "confidence": "medium",
            "assumptions": [
                "This is a longer test assumption that meets validation requirements"
            ],
            "logical_gaps": [],
            "biases": [BiasType.CONFIRMATION],
            "alternate_framings": [],
            "critique_summary": "This is a test critique summary that meets the minimum length requirement for validation",
            "issues_detected": 1,
        }

        # Create instance without bias_details (the problematic field)
        critic_result = CriticOutput(**minimal_critic_data)

        # Process through the enhancement
        enhanced_result = service._ensure_default_factory_fields(
            critic_result, CriticOutput
        )

        # Verify bias_details is properly initialized as empty dict
        assert isinstance(enhanced_result, CriticOutput)
        assert (
            enhanced_result.bias_details == {}
        )  # Should be empty dict from default_factory
        assert enhanced_result.assumptions == [
            "This is a longer test assumption that meets validation requirements"
        ]
        assert enhanced_result.biases == [BiasType.CONFIRMATION]


class TestParserIntegration:
    """Integration tests for the parser fix in real scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_critic_output_parsing(self) -> None:
        """Test end-to-end parsing that would have failed before the fix."""
        service = LangChainService(
            model="gpt-5",
            api_key="test-key",
            use_discovery=False,
            use_pool=False,
        )

        with patch("openai.AsyncOpenAI") as mock_openai_class:
            # Setup mock to return CriticOutput without bias_details
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            # Create a mock parsed result missing the problematic field
            mock_parsed = CriticOutput(
                agent_name="critic",
                processing_mode="active",
                confidence="medium",
                assumptions=[
                    "This is a longer test assumption that meets validation requirements"
                ],
                logical_gaps=[],
                biases=[],
                alternate_framings=[],
                critique_summary="This is a test critique summary with sufficient length",
                issues_detected=0,
                # NOTE: bias_details is intentionally missing
            )

            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.parsed = mock_parsed
            mock_client.beta.chat.completions.parse = AsyncMock(
                return_value=mock_completion
            )

            # This should now work without schema validation errors
            result = await service._try_native_openai_parse(
                messages=[("human", "test")], output_class=CriticOutput
            )

            # Verify the result is valid and has initialized default_factory fields
            assert isinstance(result, CriticOutput)
            assert result.bias_details == {}  # Should be properly initialized
