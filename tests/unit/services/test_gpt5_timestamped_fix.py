"""Test GPT-5 timestamped model fix for structured output."""

import pytest
from typing import Optional, Type
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pydantic import BaseModel

from cognivault.services.langchain_service import LangChainService


class MockOutput(BaseModel):
    """Mock output model for testing."""

    answer: str
    confidence: float


class TestGPT5TimestampedFix:
    """Test suite for GPT-5 timestamped model structured output fix."""

    @pytest.mark.asyncio
    async def test_timestamped_gpt5_uses_function_calling(self) -> None:
        """Test that timestamped GPT-5 models use function_calling method."""
        service = LangChainService(
            model="gpt-5-2025-08-07",
            api_key="test-key",
            use_discovery=False,
            use_pool=False,
        )

        # Test method detection
        method = service._get_structured_output_method("gpt-5-2025-08-07")
        assert method == "function_calling", (
            "Timestamped GPT-5 should use function_calling"
        )

    def test_base_gpt5_uses_json_schema(self) -> None:
        """Test that base GPT-5 models still use json_schema."""
        service = LangChainService(
            model="gpt-5", api_key="test-key", use_discovery=False, use_pool=False
        )

        # Test method detection
        method = service._get_structured_output_method("gpt-5")
        assert method == "json_schema", "Base GPT-5 should use json_schema"

    def test_gpt5_nano_uses_json_schema(self) -> None:
        """Test that GPT-5 nano still uses json_schema."""
        service = LangChainService(
            model="gpt-5-nano", api_key="test-key", use_discovery=False, use_pool=False
        )

        method = service._get_structured_output_method("gpt-5-nano")
        assert method == "json_schema", "GPT-5 nano should use json_schema"

    def test_chat_variant_uses_json_mode(self) -> None:
        """Test that chat variants use json_mode fallback."""
        service = LangChainService(
            model="gpt-5-chat-latest",
            api_key="test-key",
            use_discovery=False,
            use_pool=False,
        )

        method = service._get_structured_output_method("gpt-5-chat-latest")
        assert method == "json_mode", "Chat variants should use json_mode"

    @pytest.mark.asyncio
    async def test_method_fallback_on_failure(self) -> None:
        """Test that method fallback works when primary method fails."""
        service = LangChainService(
            model="gpt-5-2025-08-07",
            api_key="test-key",
            use_discovery=False,
            use_pool=False,
        )

        # Mock the LLM to simulate method failure
        mock_llm = MagicMock()
        mock_structured = MagicMock()

        # First call with function_calling fails, second with json_mode succeeds
        call_count = 0

        def with_structured_side_effect(
            schema: Type[BaseModel],
            method: Optional[str] = None,
            include_raw: bool = False,
        ) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1 and method == "function_calling":
                raise ValueError("Invalid schema for function_calling")
            return mock_structured

        mock_llm.with_structured_output.side_effect = with_structured_side_effect
        mock_structured.ainvoke = AsyncMock(
            return_value=MockOutput(answer="test", confidence=0.9)
        )

        service.llm = mock_llm

        # Should succeed with fallback
        result = await service._try_native_structured_output(
            [("user", "test prompt")], MockOutput, include_raw=False, attempt=0
        )

        assert isinstance(result, MockOutput)
        assert mock_llm.with_structured_output.call_count >= 2, (
            "Should try multiple methods"
        )

    def test_temperature_excluded_for_all_gpt5_variants(self) -> None:
        """Test that temperature is excluded for all GPT-5 variants."""
        models_to_test = [
            "gpt-5",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5-2025-08-07",
            "gpt-5-chat-latest",
        ]

        for model in models_to_test:
            with patch(
                "cognivault.services.langchain_service.ChatOpenAI"
            ) as mock_openai:
                service = LangChainService(
                    model=model,
                    temperature=0.5,  # Should be excluded
                    api_key="test-key",
                    use_discovery=False,
                    use_pool=False,
                )

                # Check ChatOpenAI was called without temperature
                call_kwargs = mock_openai.call_args[1]
                assert "temperature" not in call_kwargs, (
                    f"Temperature should be excluded for {model}"
                )
                assert call_kwargs["model"] == model
                assert call_kwargs["api_key"] == "test-key"
