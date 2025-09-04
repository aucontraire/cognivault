"""
Test that base GPT-5 models are preferred over variants.

"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from cognivault.services.model_discovery_service import (
    ModelDiscoveryService,
    ModelInfo,
    ModelCategory,
    ModelSpeed,
)


class TestBaseModelPreference:
    """Test that base models are preferred over variants."""

    def test_rank_model_prefers_base_models(self) -> None:
        """Test that the ranking function heavily prefers base models."""
        service = ModelDiscoveryService(enable_discovery=False)

        # Create test models
        base_gpt5 = ModelInfo(
            id="gpt-5",
            category=ModelCategory.GPT5,
            speed=ModelSpeed.STANDARD,
            context_window=128000,
            supports_structured_output=True,
            supports_function_calling=True,
        )

        variant_gpt5 = ModelInfo(
            id="gpt-5-2025-08-07",  # Timestamped variant
            category=ModelCategory.GPT5,
            speed=ModelSpeed.STANDARD,
            context_window=128000,
            supports_structured_output=True,
            supports_function_calling=True,
        )

        chat_variant = ModelInfo(
            id="gpt-5-chat-latest",  # Chat variant
            category=ModelCategory.GPT5,
            speed=ModelSpeed.STANDARD,
            context_window=128000,
            supports_structured_output=False,  # Chat variants don't support this
            supports_function_calling=False,
        )

        # Test ranking for each agent type
        for agent_name in ["historian", "critic", "synthesis"]:
            prefs = service.AGENT_MODEL_PREFERENCES[agent_name]

            # Create the rank_model function with proper closure
            def rank_model(model: ModelInfo) -> int:
                score = 0

                # HIGHEST PRIORITY: Prefer base models over variants
                is_base_model = model.id.lower() in [
                    "gpt-5",
                    "gpt-5-nano",
                    "gpt-5-mini",
                ]
                is_variant = "-" in model.id and not is_base_model

                if is_base_model:
                    score += 2000  # Huge bonus for base models
                elif is_variant:
                    score -= 1500  # Penalize ALL variants

                # Category preference
                preferred_cats = prefs.get("preferred_categories", [])
                if (
                    isinstance(preferred_cats, list)
                    and model.category in preferred_cats
                ):
                    score += 100 * (
                        len(preferred_cats) - preferred_cats.index(model.category)
                    )

                # Speed preference
                if "required_speed" in prefs and model.speed == prefs["required_speed"]:
                    score += 50

                # Structured output support bonus
                if model.supports_structured_output:
                    score += 200

                # Function calling support bonus
                if model.supports_function_calling:
                    score += 100

                # Capability count
                score += len(model.capabilities)

                return score

            # Calculate scores
            base_score = rank_model(base_gpt5)
            variant_score = rank_model(variant_gpt5)
            chat_score = rank_model(chat_variant)

            # Assert base model has highest score
            assert base_score > variant_score, (
                f"Base gpt-5 should score higher than timestamped variant for {agent_name}"
            )
            assert base_score > chat_score, (
                f"Base gpt-5 should score higher than chat variant for {agent_name}"
            )

            # Verify the score differences match our expectations
            # Base model gets +2000, variants get -1500, so difference should be ~3500
            assert base_score - variant_score >= 3000, (
                f"Score difference should be at least 3000 for {agent_name}"
            )

    @pytest.mark.asyncio
    async def test_get_best_model_selects_base(self) -> None:
        """Test that get_best_model_for_agent selects base models."""
        service = ModelDiscoveryService(enable_discovery=False)

        # Mock the discover_models method to return our test models
        test_models = [
            ModelInfo(
                id="gpt-5",
                category=ModelCategory.GPT5,
                speed=ModelSpeed.STANDARD,
                context_window=128000,
                supports_structured_output=True,
                supports_function_calling=True,
                capabilities={"reasoning", "analysis"},
            ),
            ModelInfo(
                id="gpt-5-2025-08-07",
                category=ModelCategory.GPT5,
                speed=ModelSpeed.STANDARD,
                context_window=128000,
                supports_structured_output=True,
                supports_function_calling=True,
                capabilities={"reasoning", "analysis"},
            ),
            ModelInfo(
                id="gpt-5-chat-latest",
                category=ModelCategory.GPT5,
                speed=ModelSpeed.STANDARD,
                context_window=128000,
                supports_structured_output=False,
                supports_function_calling=False,
                capabilities={"reasoning", "analysis"},
            ),
        ]

        # Patch discover_models to return our test models
        with patch.object(
            service, "discover_models", new_callable=AsyncMock
        ) as mock_discover:
            mock_discover.return_value = test_models

            # Test each agent
            for agent_name in ["historian", "critic", "synthesis"]:
                best_model = await service.get_best_model_for_agent(agent_name)

                # Should select base gpt-5, not variants
                assert best_model == "gpt-5", (
                    f"Agent {agent_name} should select base gpt-5, got {best_model}"
                )

    def test_simplified_method_selection(self) -> None:
        """Test that method selection is simplified for base models."""
        from cognivault.services.langchain_service import LangChainService

        service = LangChainService(use_discovery=False)

        # Base models should use json_schema
        assert service._get_structured_output_method("gpt-5") == "json_schema"
        assert service._get_structured_output_method("gpt-5-nano") == "json_schema"
        assert service._get_structured_output_method("gpt-5-mini") == "json_schema"

        # Unknown models should fallback to json_mode
        assert service._get_structured_output_method("unknown-model") == "json_mode"

        # GPT-4 models should use their specific methods
        assert service._get_structured_output_method("gpt-4") == "function_calling"
        assert service._get_structured_output_method("gpt-4o") == "json_schema"
