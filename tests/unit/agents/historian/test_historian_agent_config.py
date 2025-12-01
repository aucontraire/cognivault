"""
Unit tests for HistorianAgent configuration integration.

Tests the configuration system integration including:
- Backward compatibility (no config parameter)
- Configuration parameter acceptance with existing search_type param
- PromptComposer integration
- Dynamic prompt updates
- Fallback behavior
- Preservation of existing HistorianAgent features
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import asyncio

from cognivault.agents.historian.agent import HistorianAgent
from cognivault.config.agent_configs import HistorianConfig
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from tests.factories.agent_context_factories import AgentContextFactory
from tests.factories import HistorianConfigFactory


class TestHistorianAgentConfig:
    """Test HistorianAgent configuration system integration."""

    def test_historian_agent_backward_compatibility_no_params(self) -> None:
        """Test that HistorianAgent works without any parameters (backward compatibility)."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            # Should work without any parameters (existing code compatibility)
            agent = HistorianAgent()

            # Should have default config
            assert agent.config is not None
            assert isinstance(agent.config, HistorianConfig)
            assert agent.config.search_depth == "standard"  # Default value
            assert agent.config.relevance_threshold == 0.6  # Default value

            # Should have PromptComposer
            assert agent._prompt_composer is not None

            # Should preserve existing attributes
            assert agent.search_type == "hybrid"  # Default value
            assert hasattr(agent, "search_engine")

            # Should have composed prompt (or None if composition failed)
            assert hasattr(agent, "_composed_prompt")

    def test_historian_agent_backward_compatibility_with_existing_params(self) -> None:
        """Test that HistorianAgent works with existing parameters (backward compatibility)."""
        mock_llm = Mock(spec=LLMInterface)

        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            # Should work with existing parameter pattern
            agent = HistorianAgent(llm=mock_llm, search_type="vector")

            # Should use provided parameters
            assert agent.llm is mock_llm
            assert agent.search_type == "vector"

            # Should have default config (new feature)
            assert agent.config is not None
            assert isinstance(agent.config, HistorianConfig)

    def test_historian_agent_with_config_parameter(self) -> None:
        """Test that HistorianAgent accepts config parameter while preserving existing params."""
        mock_llm = Mock(spec=LLMInterface)

        # Create custom config
        custom_config = HistorianConfigFactory.deep_search(relevance_threshold=0.85)

        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(
                llm=mock_llm, search_type="hybrid", config=custom_config
            )

            # Should use the provided config
            assert agent.config is custom_config
            assert agent.config.search_depth == "deep"
            assert agent.config.relevance_threshold == 0.85
            assert agent.config.context_expansion is True
            assert agent.config.memory_scope == "full"

            # Should preserve existing functionality
            assert agent.llm is mock_llm
            assert agent.search_type == "hybrid"

    @patch("cognivault.agents.historian.agent.PromptComposer")
    def test_prompt_composer_integration(self, mock_prompt_composer_class: Any) -> None:
        """Test that HistorianAgent integrates with PromptComposer correctly."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer: Mock = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock prompt composition
        mock_composed_prompt: Mock = Mock()
        mock_composed_prompt.system_prompt = "Custom historian system prompt"
        mock_composer.compose_historian_prompt.return_value = mock_composed_prompt
        mock_composer.validate_composition.return_value = True

        config = HistorianConfigFactory.deep_search()

        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm, config=config)

            # Should call compose_historian_prompt during initialization
            mock_composer.compose_historian_prompt.assert_called_once_with(config)

            # Should use composed prompt
            system_prompt = agent._get_system_prompt()
            assert system_prompt == "Custom historian system prompt"

    @patch("cognivault.agents.historian.agent.PromptComposer")
    def test_prompt_composition_fallback(self, mock_prompt_composer_class: Any) -> None:
        """Test fallback to default prompt when composition fails."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer: Mock = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock composition failure
        mock_composer.compose_historian_prompt.side_effect = Exception(
            "Composition failed"
        )

        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm)

            # Should fallback to default prompt
            system_prompt = agent._get_system_prompt()
            # Should be either from prompts.py or embedded fallback
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 0

    @patch("cognivault.agents.historian.agent.PromptComposer")
    def test_update_config_method(self, mock_prompt_composer_class: Any) -> None:
        """Test the update_config method updates configuration and recomposes prompts."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer: Mock = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent(llm=mock_llm)
            original_config = agent.config

            # Update config
            new_config = HistorianConfigFactory.exhaustive_search(
                relevance_threshold=0.9
            )

            agent.update_config(new_config)

            # Should update config
            assert agent.config is new_config
            assert agent.config.search_depth == "exhaustive"

            # Should call compose_historian_prompt again (once during init, once during update)
            assert mock_composer.compose_historian_prompt.call_count == 2

    def test_agent_has_required_attributes(self) -> None:
        """Test that HistorianAgent has all required attributes for config integration."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            agent = HistorianAgent()

            # Required attributes for config integration
            assert hasattr(agent, "config")
            assert hasattr(agent, "_prompt_composer")
            assert hasattr(agent, "_composed_prompt")
            assert hasattr(agent, "update_config")
            assert hasattr(agent, "_update_composed_prompt")
            assert hasattr(agent, "_get_system_prompt")

            # Preserve existing HistorianAgent attributes
            assert hasattr(agent, "search_engine")
            assert hasattr(agent, "search_type")
            assert hasattr(agent, "llm")

            # Required methods should be callable
            assert callable(agent.update_config)
            assert callable(agent._update_composed_prompt)
            assert callable(agent._get_system_prompt)

    def test_config_property_type_safety(self) -> None:
        """Test that config property maintains type safety."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            # With custom config
            custom_config = HistorianConfigFactory.deep_search()
            agent_with_config = HistorianAgent(config=custom_config)
            assert isinstance(agent_with_config.config, HistorianConfig)

            # With default config
            agent_default = HistorianAgent()
            assert isinstance(agent_default.config, HistorianConfig)

    def test_parameter_order_and_compatibility(self) -> None:
        """Test that parameter order and types are preserved for backward compatibility."""
        mock_llm = Mock(spec=LLMInterface)

        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            # Test various parameter combinations

            # Original pattern
            agent1 = HistorianAgent(llm=mock_llm)
            assert agent1.llm is mock_llm
            assert agent1.search_type == "hybrid"

            # With search_type
            agent2 = HistorianAgent(llm=mock_llm, search_type="vector")
            assert agent2.llm is mock_llm
            assert agent2.search_type == "vector"

            # With config (new)
            config = HistorianConfigFactory.deep_search()
            agent3 = HistorianAgent(llm=mock_llm, search_type="semantic", config=config)
            assert agent3.llm is mock_llm
            assert agent3.search_type == "semantic"
            assert agent3.config is config

    @pytest.mark.asyncio
    async def test_run_method_preserves_functionality(self) -> None:
        """Test that run method preserves existing HistorianAgent functionality."""
        mock_llm = Mock(spec=LLMInterface)

        with patch(
            "cognivault.agents.historian.agent.SearchFactory.create_search"
        ) as mock_search_factory:
            with patch("cognivault.config.app_config.get_config") as mock_get_config:
                # Mock config
                mock_config: Mock = Mock()
                mock_config.execution.enable_simulation_delay = False
                mock_config.testing.mock_history_entries = False
                mock_get_config.return_value = mock_config

                # Mock search engine
                mock_search_engine: Mock = Mock()
                mock_search_engine.search = AsyncMock(return_value=[])
                mock_search_factory.create_search.return_value = mock_search_engine

                agent = HistorianAgent(llm=mock_llm)

                # Create context
                context = AgentContextFactory.basic(
                    query="Test historical query",
                    user_config={"user_id": "test_user", "session_id": "test_session"},
                    metadata={"workflow_metadata": {}},
                )

                # Run agent
                result_context = await agent.run(context)

                # Should complete without errors and preserve existing behavior
                assert "historian" in result_context.agent_outputs
                assert isinstance(
                    result_context.agent_outputs["historian"], (str, dict)
                )

    def test_default_llm_creation_with_config(self) -> None:
        """Test that default LLM creation works with config integration."""
        with patch("cognivault.agents.historian.agent.SearchFactory.create_search"):
            with patch(
                "cognivault.agents.historian.agent.HistorianAgent._create_default_llm"
            ) as mock_create_llm:
                mock_llm = Mock(spec=LLMInterface)
                mock_create_llm.return_value = mock_llm

                # Test default LLM creation (llm="default")
                agent = HistorianAgent()  # Uses default llm="default"

                # Should call default LLM creation
                mock_create_llm.assert_called_once()
                assert agent.llm is mock_llm

                # Should have config integration
                assert isinstance(agent.config, HistorianConfig)

    def test_search_type_and_config_interaction(self) -> None:
        """Test that search_type parameter and config work together correctly."""
        custom_config = HistorianConfigFactory.deep_search(relevance_threshold=0.8)

        with patch(
            "cognivault.agents.historian.agent.SearchFactory.create_search"
        ) as mock_search_factory:
            mock_search_engine: Mock = Mock()
            mock_search_factory.return_value = mock_search_engine

            agent = HistorianAgent(search_type="vector", config=custom_config)

            # Should use both parameters correctly
            assert agent.search_type == "vector"
            assert agent.config is custom_config
            assert agent.config.search_depth == "deep"

            # Should create search engine with correct type (called inside __init__)
            mock_search_factory.assert_called_once_with("vector")
