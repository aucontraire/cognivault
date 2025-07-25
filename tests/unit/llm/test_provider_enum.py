"""
Unit tests for LLM provider and model enumerations.

Tests type safety, extensibility, and integration capabilities
of the LLM model and provider enums.
"""

import pytest
from enum import Enum

from cognivault.llm.provider_enum import LLMProvider, LLMModel


class TestLLMProvider:
    """Test suite for LLM provider enumeration."""

    def test_provider_enum_values(self):
        """Test that LLM provider enum has expected values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.STUB == "stub"

    def test_provider_enum_inheritance(self):
        """Test that LLMProvider inherits from str and Enum."""
        assert issubclass(LLMProvider, str)
        assert issubclass(LLMProvider, Enum)

        # Test instance behavior
        provider = LLMProvider.OPENAI
        assert isinstance(provider, str)
        assert isinstance(provider, LLMProvider)
        assert provider == "openai"
        assert provider.value == "openai"

    def test_provider_enum_iteration(self):
        """Test iteration over provider enum values."""
        providers = list(LLMProvider)
        assert len(providers) == 2
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.STUB in providers


class TestLLMModel:
    """Test suite for LLM model enumeration."""

    def test_openai_model_values(self):
        """Test OpenAI model enum values."""
        assert LLMModel.GPT_4 == "gpt-4"
        assert LLMModel.GPT_4_TURBO == "gpt-4-turbo"
        assert LLMModel.GPT_4O == "gpt-4o"
        assert LLMModel.GPT_4O_MINI == "gpt-4o-mini"
        assert LLMModel.GPT_3_5_TURBO == "gpt-3.5-turbo"

    def test_future_model_values(self):
        """Test future/extensibility model enum values."""
        assert LLMModel.CLAUDE_OPUS == "claude-3-opus"
        assert LLMModel.CLAUDE_SONNET == "claude-3-sonnet"
        assert LLMModel.CLAUDE_HAIKU == "claude-3-haiku"
        assert LLMModel.MISTRAL_7B == "mistral-7b"
        assert LLMModel.LLAMA_3 == "llama-3"

    def test_special_model_values(self):
        """Test special model enum values for testing and custom use."""
        assert LLMModel.STUB == "stub"
        assert LLMModel.LOCAL_CUSTOM == "local-custom"

    def test_model_enum_inheritance(self):
        """Test that LLMModel inherits from str and Enum."""
        assert issubclass(LLMModel, str)
        assert issubclass(LLMModel, Enum)

        # Test instance behavior
        model = LLMModel.GPT_4
        assert isinstance(model, str)
        assert isinstance(model, LLMModel)
        assert model == "gpt-4"
        assert model.value == "gpt-4"

    def test_model_enum_completeness(self):
        """Test that all expected models are present."""
        models = list(LLMModel)

        # Should have at least 12 models (5 OpenAI + 5 future + 2 special)
        assert len(models) >= 12

        # Check key models are present
        expected_models = [
            LLMModel.GPT_4,
            LLMModel.GPT_4_TURBO,
            LLMModel.CLAUDE_OPUS,
            LLMModel.MISTRAL_7B,
            LLMModel.STUB,
        ]
        for model in expected_models:
            assert model in models

    def test_model_enum_type_safety(self):
        """Test type safety benefits of using enum vs raw strings."""

        # Enum provides type safety
        def process_model(model: LLMModel) -> str:
            return f"Processing with {model}"

        # Valid usage
        result = process_model(LLMModel.GPT_4)
        assert "GPT_4" in result  # Enum name representation in f-strings

        # String comparison still works
        model = LLMModel.GPT_4_TURBO
        assert model == "gpt-4-turbo"

        # Can be used in dictionaries/sets
        model_costs = {
            LLMModel.GPT_4: 0.03,
            LLMModel.GPT_4_TURBO: 0.01,
            LLMModel.GPT_3_5_TURBO: 0.0015,
        }
        assert model_costs[LLMModel.GPT_4] == 0.03

    def test_model_categorization(self):
        """Test ability to categorize models by provider."""
        openai_models = [
            LLMModel.GPT_4,
            LLMModel.GPT_4_TURBO,
            LLMModel.GPT_4O,
            LLMModel.GPT_4O_MINI,
            LLMModel.GPT_3_5_TURBO,
        ]

        claude_models = [
            LLMModel.CLAUDE_OPUS,
            LLMModel.CLAUDE_SONNET,
            LLMModel.CLAUDE_HAIKU,
        ]

        # Test that we can identify OpenAI models
        for model in openai_models:
            assert "gpt" in model.value.lower() or "gpt-" in model.value

        # Test that we can identify Claude models
        for model in claude_models:
            assert "claude" in model.value.lower()

    def test_model_enum_extensibility(self):
        """Test that the enum supports future extension patterns."""
        # All models should be strings
        for model in LLMModel:
            assert isinstance(model.value, str)
            assert len(model.value) > 0

        # Models should follow naming conventions
        for model in LLMModel:
            # No spaces in model names
            assert " " not in model.value
            # Lowercase with hyphens or underscores
            assert (
                model.value.replace("-", "").replace("_", "").replace(".", "").islower()
            )

    def test_enum_serialization_compatibility(self):
        """Test that enums serialize properly for API/JSON usage."""
        model = LLMModel.GPT_4

        # Test JSON serialization compatibility
        import json

        serialized = json.dumps({"model": model})
        assert '"model": "gpt-4"' in serialized

        # Test deserialization
        data = json.loads(serialized)
        assert data["model"] == "gpt-4"

        # Test reconstruction from string
        reconstructed = LLMModel(data["model"])
        assert reconstructed == LLMModel.GPT_4


class TestEnumIntegration:
    """Integration tests for provider and model enums."""

    def test_provider_model_mapping(self):
        """Test logical mapping between providers and models."""
        # OpenAI provider should map to OpenAI models
        openai_models = [
            LLMModel.GPT_4,
            LLMModel.GPT_4_TURBO,
            LLMModel.GPT_4O,
            LLMModel.GPT_3_5_TURBO,
        ]

        provider_model_map = {
            LLMProvider.OPENAI: openai_models,
            LLMProvider.STUB: [LLMModel.STUB],
        }

        # Test mapping logic
        for provider, models in provider_model_map.items():
            assert isinstance(provider, LLMProvider)
            for model in models:
                assert isinstance(model, LLMModel)

    def test_cost_calculation_integration(self):
        """Test integration with cost calculation functionality."""
        # This would integrate with the WorkflowExecutionMetadata cost calculation
        cost_map = {
            LLMModel.GPT_4: 0.03,
            LLMModel.GPT_4_TURBO: 0.01,
            LLMModel.GPT_4O: 0.005,
            LLMModel.GPT_4O_MINI: 0.00015,
            LLMModel.GPT_3_5_TURBO: 0.0015,
        }

        # Test that enum can be used as dictionary keys
        test_model = LLMModel.GPT_4
        assert cost_map[test_model] == 0.03

        # Test cost calculation with enum
        tokens = 1000
        cost = (tokens / 1000) * cost_map[LLMModel.GPT_4_TURBO]
        assert cost == 0.01

    def test_future_provider_extensibility(self):
        """Test that the design supports adding new providers."""
        # Current enum structure should support easy extension
        providers = list(LLMProvider)
        models = list(LLMModel)

        # Should be easy to add new providers like:
        # ANTHROPIC = "anthropic"
        # MISTRAL = "mistral"
        # HUGGINGFACE = "huggingface"

        # Should be easy to add new models like:
        # CLAUDE_3_5_SONNET = "claude-3.5-sonnet"
        # MIXTRAL_8X7B = "mixtral-8x7b"

        # Verify current structure supports this pattern
        assert all(isinstance(p.value, str) for p in providers)
        assert all(isinstance(m.value, str) for m in models)
        assert len(providers) >= 2  # Room for growth
        assert len(models) >= 10  # Room for growth
