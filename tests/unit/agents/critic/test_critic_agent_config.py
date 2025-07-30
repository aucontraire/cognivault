"""
Unit tests for CriticAgent configuration integration.

Tests the configuration system integration including:
- Backward compatibility (no config parameter)
- Configuration parameter acceptance
- PromptComposer integration
- Dynamic prompt updates
- Fallback behavior
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from cognivault.agents.critic.agent import CriticAgent
from cognivault.config.agent_configs import CriticConfig
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface


class TestCriticAgentConfig:
    """Test CriticAgent configuration system integration."""

    def test_critic_agent_backward_compatibility(self):
        """Test that CriticAgent works without config parameter (backward compatibility)."""
        mock_llm = Mock(spec=LLMInterface)

        # Should work without config parameter (existing code compatibility)
        agent = CriticAgent(llm=mock_llm)

        # Should have default config
        assert agent.config is not None
        assert isinstance(agent.config, CriticConfig)
        assert agent.config.analysis_depth == "medium"  # Default value
        assert agent.config.confidence_reporting is True  # Default value

        # Should have PromptComposer
        assert agent._prompt_composer is not None

        # Should have composed prompt (or None if composition failed)
        assert hasattr(agent, "_composed_prompt")

    def test_critic_agent_with_config_parameter(self):
        """Test that CriticAgent accepts config parameter."""
        mock_llm = Mock(spec=LLMInterface)

        # Create custom config
        custom_config = CriticConfig(
            analysis_depth="deep",
            confidence_reporting=False,
            bias_detection=True,
            scoring_criteria=["accuracy", "thoroughness"],
        )

        agent = CriticAgent(llm=mock_llm, config=custom_config)

        # Should use the provided config
        assert agent.config is custom_config
        assert agent.config.analysis_depth == "deep"
        assert agent.config.confidence_reporting is False
        assert agent.config.bias_detection is True
        assert "accuracy" in agent.config.scoring_criteria

    @patch("cognivault.agents.critic.agent.PromptComposer")
    def test_prompt_composer_integration(self, mock_prompt_composer_class):
        """Test that CriticAgent integrates with PromptComposer correctly."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock prompt composition
        mock_composed_prompt = Mock()
        mock_composed_prompt.system_prompt = "Custom critic system prompt"
        mock_composer.compose_critic_prompt.return_value = mock_composed_prompt
        mock_composer.validate_composition.return_value = True

        config = CriticConfig(analysis_depth="deep")
        agent = CriticAgent(llm=mock_llm, config=config)

        # Should call compose_critic_prompt during initialization
        mock_composer.compose_critic_prompt.assert_called_once_with(config)

        # Should use composed prompt
        system_prompt = agent._get_system_prompt()
        assert system_prompt == "Custom critic system prompt"

    @patch("cognivault.agents.critic.agent.PromptComposer")
    def test_prompt_composition_fallback(self, mock_prompt_composer_class):
        """Test fallback to default prompt when composition fails."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock composition failure
        mock_composer.compose_critic_prompt.side_effect = Exception(
            "Composition failed"
        )

        agent = CriticAgent(llm=mock_llm)

        # Should fallback to default prompt
        from cognivault.agents.critic.prompts import CRITIC_SYSTEM_PROMPT

        system_prompt = agent._get_system_prompt()
        assert system_prompt == CRITIC_SYSTEM_PROMPT

    @patch("cognivault.agents.critic.agent.PromptComposer")
    def test_update_config_method(self, mock_prompt_composer_class):
        """Test the update_config method updates configuration and recomposes prompts."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        agent = CriticAgent(llm=mock_llm)
        original_config = agent.config

        # Update config
        new_config = CriticConfig(
            analysis_depth="comprehensive", confidence_reporting=False
        )

        agent.update_config(new_config)

        # Should update config
        assert agent.config is new_config
        assert agent.config.analysis_depth == "comprehensive"

        # Should call compose_critic_prompt again (once during init, once during update)
        assert mock_composer.compose_critic_prompt.call_count == 2

    @pytest.mark.asyncio
    @patch("cognivault.agents.critic.agent.PromptComposer")
    async def test_run_method_uses_configurable_prompt(
        self, mock_prompt_composer_class
    ):
        """Test that run method uses configurable system prompt."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock composed prompt
        mock_composed_prompt = Mock()
        mock_composed_prompt.system_prompt = "Custom deep analysis prompt"
        mock_composer.compose_critic_prompt.return_value = mock_composed_prompt
        mock_composer.validate_composition.return_value = True

        # Mock LLM response
        mock_response = Mock()
        mock_response.text = "Detailed critique of the refined query"
        mock_llm.generate = Mock(return_value=mock_response)

        config = CriticConfig(analysis_depth="deep")
        agent = CriticAgent(llm=mock_llm, config=config)

        # Create context with refiner output
        context = AgentContext(
            user_id="test_user",
            session_id="test_session",
            query="Original query",
            workflow_metadata={},
        )
        context.add_agent_output("refiner", "Refined query output")

        # Run agent
        result_context = await agent.run(context)

        # Should use custom system prompt
        mock_llm.generate.assert_called_once_with(
            prompt="Refined query output", system_prompt="Custom deep analysis prompt"
        )

        # Should add critique to context
        assert "critic" in result_context.agent_outputs
        assert (
            result_context.agent_outputs["critic"]
            == "Detailed critique of the refined query"
        )

    @pytest.mark.asyncio
    async def test_run_method_backward_compatibility(self):
        """Test that run method works with original behavior (no config)."""
        mock_llm = Mock(spec=LLMInterface)

        # Mock LLM response
        mock_response = Mock()
        mock_response.text = "Standard critique"
        mock_llm.generate = Mock(return_value=mock_response)

        # Create agent without config (backward compatibility)
        agent = CriticAgent(llm=mock_llm)

        # Create context
        context = AgentContext(
            user_id="test_user",
            session_id="test_session",
            query="Test query",
            workflow_metadata={},
        )
        context.add_agent_output("refiner", "Refined test query")

        # Run agent
        result_context = await agent.run(context)

        # Should work and add output
        assert "critic" in result_context.agent_outputs
        assert result_context.agent_outputs["critic"] == "Standard critique"

        # Should call LLM generate (system prompt will be default or composed)
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_method_handles_missing_refiner_output(self):
        """Test that run method handles missing refiner output gracefully."""
        mock_llm = Mock(spec=LLMInterface)
        agent = CriticAgent(llm=mock_llm)

        # Create context without refiner output
        context = AgentContext(
            user_id="test_user",
            session_id="test_session",
            query="Test query",
            workflow_metadata={},
        )

        # Run agent
        result_context = await agent.run(context)

        # Should handle missing refiner output
        assert "critic" in result_context.agent_outputs
        assert "No refined output available" in result_context.agent_outputs["critic"]

        # Should not call LLM when no refiner output
        mock_llm.generate.assert_not_called()

    def test_agent_has_required_attributes(self):
        """Test that CriticAgent has all required attributes for config integration."""
        mock_llm = Mock(spec=LLMInterface)
        agent = CriticAgent(llm=mock_llm)

        # Required attributes for config integration
        assert hasattr(agent, "config")
        assert hasattr(agent, "_prompt_composer")
        assert hasattr(agent, "_composed_prompt")
        assert hasattr(agent, "update_config")
        assert hasattr(agent, "_update_composed_prompt")
        assert hasattr(agent, "_get_system_prompt")

        # Required methods should be callable
        assert callable(agent.update_config)
        assert callable(agent._update_composed_prompt)
        assert callable(agent._get_system_prompt)

    def test_config_property_type_safety(self):
        """Test that config property maintains type safety."""
        mock_llm = Mock(spec=LLMInterface)

        # With custom config
        custom_config = CriticConfig(analysis_depth="deep")
        agent_with_config = CriticAgent(llm=mock_llm, config=custom_config)
        assert isinstance(agent_with_config.config, CriticConfig)

        # With default config
        agent_default = CriticAgent(llm=mock_llm)
        assert isinstance(agent_default.config, CriticConfig)
