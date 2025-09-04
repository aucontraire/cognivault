"""Unit tests for GPT-5 temperature constraint handling."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from cognivault.services.langchain_service import LangChainService
from cognivault.services.llm_pool import LLMServicePool
from langchain_openai import ChatOpenAI


class TestGPT5TemperatureHandling:
    """Test that GPT-5 models correctly exclude temperature parameter."""

    def test_gpt5_excludes_temperature(self) -> None:
        """Test that GPT-5 models don't get temperature parameter."""
        with patch("cognivault.services.langchain_service.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            # Create service with GPT-5 and temperature=0.1
            service = LangChainService(
                model="gpt-5",
                temperature=0.1,
                api_key="test-key",
                use_discovery=False,
                use_pool=False,
            )

            # Verify ChatOpenAI was called WITHOUT temperature
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "temperature" not in call_kwargs
            assert call_kwargs["model"] == "gpt-5"

    def test_gpt5_mini_excludes_temperature(self) -> None:
        """Test that GPT-5-mini models don't get temperature parameter."""
        with patch("cognivault.services.langchain_service.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            # Create service with GPT-5-mini
            service = LangChainService(
                model="gpt-5-mini",
                temperature=0.0,  # Even 0.0 should be excluded
                api_key="test-key",
                use_discovery=False,
                use_pool=False,
            )

            # Verify ChatOpenAI was called WITHOUT temperature
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "temperature" not in call_kwargs
            assert call_kwargs["model"] == "gpt-5-mini"

    def test_gpt5_nano_excludes_temperature(self) -> None:
        """Test that GPT-5-nano models don't get temperature parameter."""
        with patch("cognivault.services.langchain_service.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            # Create service with GPT-5-nano
            service = LangChainService(
                model="gpt-5-nano",
                temperature=0.2,
                api_key="test-key",
                use_discovery=False,
                use_pool=False,
            )

            # Verify ChatOpenAI was called WITHOUT temperature
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "temperature" not in call_kwargs
            assert call_kwargs["model"] == "gpt-5-nano"

    def test_gpt4_includes_temperature(self) -> None:
        """Test that GPT-4 models still get temperature parameter."""
        with patch("cognivault.services.langchain_service.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            # Create service with GPT-4o
            service = LangChainService(
                model="gpt-4o",
                temperature=0.1,
                api_key="test-key",
                use_discovery=False,
                use_pool=False,
            )

            # Verify ChatOpenAI was called WITH temperature
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.1
            assert call_kwargs["model"] == "gpt-4o"

    def test_unknown_model_includes_temperature(self) -> None:
        """Test that unknown models still get temperature parameter."""
        with patch("cognivault.services.langchain_service.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            # Create service with unknown model
            service = LangChainService(
                model="some-future-model",
                temperature=0.5,
                api_key="test-key",
                use_discovery=False,
                use_pool=False,
            )

            # Verify ChatOpenAI was called WITH temperature
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_native_parse_excludes_temperature_for_gpt5(self) -> None:
        """Test that native OpenAI parse excludes temperature for GPT-5."""
        with patch("openai.AsyncOpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            # Mock the parse response
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.parsed = {"test": "result"}
            mock_completion.choices[0].message.content = "raw content"

            # Make the parse method async
            from unittest.mock import AsyncMock

            mock_client.beta.chat.completions.parse = AsyncMock(
                return_value=mock_completion
            )

            # Create service with GPT-5
            service = LangChainService(
                model="gpt-5",
                temperature=0.0,  # This should be excluded
                api_key="test-key",
                use_discovery=False,
                use_pool=False,
            )

            # Call native parse
            from pydantic import BaseModel

            class TestModel(BaseModel):
                test: str

            messages = [("system", "test"), ("human", "test")]
            result = await service._try_native_openai_parse(
                messages=messages, output_class=TestModel, include_raw=False
            )

            # Verify parse was called WITHOUT temperature
            mock_client.beta.chat.completions.parse.assert_called_once()
            call_kwargs = mock_client.beta.chat.completions.parse.call_args[1]
            assert "temperature" not in call_kwargs
            assert call_kwargs["model"] == "gpt-5"


class TestLLMPoolGPT5Handling:
    """Test that LLMServicePool correctly handles GPT-5 temperature constraints."""

    def test_pool_excludes_temperature_for_gpt5(self) -> None:
        """Test that pool doesn't pass temperature to GPT-5 models."""
        with patch("cognivault.services.llm_pool.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            pool = LLMServicePool.get_instance()
            pool._api_key = "test-key"

            # Get client for GPT-5
            client = pool.get_or_create_client("gpt-5", temperature=0.1)

            # Verify ChatOpenAI was called WITHOUT temperature
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "temperature" not in call_kwargs
            assert call_kwargs["model"] == "gpt-5"

            # Verify cache key uses "default" for GPT-5
            assert "gpt-5@default" in pool._llm_clients

    def test_pool_includes_temperature_for_gpt4(self) -> None:
        """Test that pool still passes temperature to GPT-4 models."""
        with patch("cognivault.services.llm_pool.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            pool = LLMServicePool.get_instance()
            pool._api_key = "test-key"
            pool._llm_clients.clear()  # Clear cache

            # Get client for GPT-4
            client = pool.get_or_create_client("gpt-4o", temperature=0.2)

            # Verify ChatOpenAI was called WITH temperature
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.2
            assert call_kwargs["model"] == "gpt-4o"

            # Verify cache key uses temperature value for GPT-4
            assert "gpt-4o@0.2" in pool._llm_clients
