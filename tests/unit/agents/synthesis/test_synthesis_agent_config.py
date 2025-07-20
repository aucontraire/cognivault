"""
Unit tests for SynthesisAgent configuration integration.

Tests the configuration system integration including:
- Backward compatibility (no config parameter)
- Configuration parameter acceptance
- PromptComposer integration
- Dynamic prompt updates
- Fallback behavior
- Preservation of existing SynthesisAgent features
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.config.agent_configs import SynthesisConfig
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface


class TestSynthesisAgentConfig:
    """Test SynthesisAgent configuration system integration."""

    def test_synthesis_agent_backward_compatibility_no_params(self):
        """Test that SynthesisAgent works without any parameters (backward compatibility)."""
        # Should work without any parameters (existing code compatibility)
        agent = SynthesisAgent()

        # Should have default config
        assert agent.config is not None
        assert isinstance(agent.config, SynthesisConfig)
        assert agent.config.synthesis_strategy == "balanced"  # Default value
        assert agent.config.meta_analysis is True  # Default value

        # Should have PromptComposer
        assert agent._prompt_composer is not None

        # Should have composed prompt (or None if composition failed)
        assert hasattr(agent, "_composed_prompt")

    def test_synthesis_agent_backward_compatibility_with_llm(self):
        """Test that SynthesisAgent works with existing LLM parameter (backward compatibility)."""
        mock_llm = Mock(spec=LLMInterface)

        # Should work with existing parameter pattern
        agent = SynthesisAgent(llm=mock_llm)

        # Should use provided LLM
        assert agent.llm is mock_llm

        # Should have default config (new feature)
        assert agent.config is not None
        assert isinstance(agent.config, SynthesisConfig)

    def test_synthesis_agent_with_config_parameter(self):
        """Test that SynthesisAgent accepts config parameter while preserving existing params."""
        mock_llm = Mock(spec=LLMInterface)

        # Create custom config
        custom_config = SynthesisConfig(
            synthesis_strategy="focused",
            meta_analysis=False,
            thematic_focus="analytical",
            integration_mode="sequential",
        )

        agent = SynthesisAgent(llm=mock_llm, config=custom_config)

        # Should use the provided config
        assert agent.config is custom_config
        assert agent.config.synthesis_strategy == "focused"
        assert agent.config.meta_analysis is False
        assert agent.config.thematic_focus == "analytical"
        assert agent.config.integration_mode == "sequential"

        # Should preserve existing functionality
        assert agent.llm is mock_llm

    @patch("cognivault.agents.synthesis.agent.PromptComposer")
    def test_prompt_composer_integration(self, mock_prompt_composer_class):
        """Test that SynthesisAgent integrates with PromptComposer correctly."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock prompt composition
        mock_composed_prompt = Mock()
        mock_composed_prompt.system_prompt = "Custom synthesis system prompt"
        mock_composer.compose_synthesis_prompt.return_value = mock_composed_prompt
        mock_composer.validate_composition.return_value = True

        config = SynthesisConfig(synthesis_strategy="focused")
        agent = SynthesisAgent(llm=mock_llm, config=config)

        # Should call compose_synthesis_prompt during initialization
        mock_composer.compose_synthesis_prompt.assert_called_once_with(config)

        # Should use composed prompt
        system_prompt = agent._get_system_prompt()
        assert system_prompt == "Custom synthesis system prompt"

    @patch("cognivault.agents.synthesis.agent.PromptComposer")
    def test_prompt_composition_fallback(self, mock_prompt_composer_class):
        """Test fallback to default prompt when composition fails."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock composition failure
        mock_composer.compose_synthesis_prompt.side_effect = Exception(
            "Composition failed"
        )

        agent = SynthesisAgent(llm=mock_llm)

        # Should fallback to default prompt
        system_prompt = agent._get_system_prompt()
        # Should be either from prompts.py or embedded fallback
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0

    @patch("cognivault.agents.synthesis.agent.PromptComposer")
    def test_update_config_method(self, mock_prompt_composer_class):
        """Test the update_config method updates configuration and recomposes prompts."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        agent = SynthesisAgent(llm=mock_llm)
        original_config = agent.config

        # Update config
        new_config = SynthesisConfig(
            synthesis_strategy="comprehensive", meta_analysis=False
        )

        agent.update_config(new_config)

        # Should update config
        assert agent.config is new_config
        assert agent.config.synthesis_strategy == "comprehensive"

        # Should call compose_synthesis_prompt again (once during init, once during update)
        assert mock_composer.compose_synthesis_prompt.call_count == 2

    @pytest.mark.asyncio
    async def test_run_method_preserves_functionality(self):
        """Test that run method preserves existing SynthesisAgent functionality."""
        mock_llm = Mock(spec=LLMInterface)

        # Mock LLM responses for analysis and synthesis
        mock_analysis_response = Mock()
        mock_analysis_response.text = (
            "THEMES: test theme\nTOPICS: test topic\nCONFLICTS: none"
        )

        mock_synthesis_response = Mock()
        mock_synthesis_response.text = "Comprehensive synthesis of agent outputs"

        mock_llm.generate.side_effect = [
            mock_analysis_response,
            mock_synthesis_response,
        ]

        agent = SynthesisAgent(llm=mock_llm)

        # Create context with multiple agent outputs
        context = AgentContext(
            user_id="test_user",
            session_id="test_session",
            query="Test synthesis query",
            workflow_metadata={},
        )
        context.add_agent_output("refiner", "Refined query output")
        context.add_agent_output("critic", "Critical analysis output")
        context.add_agent_output("historian", "Historical context output")

        # Run agent
        result_context = await agent.run(context)

        # Should complete without errors and preserve existing behavior
        assert "Synthesis" in result_context.agent_outputs
        assert isinstance(result_context.agent_outputs["Synthesis"], str)
        assert len(result_context.agent_outputs["Synthesis"]) > 0

    @pytest.mark.asyncio
    async def test_run_method_with_no_llm_fallback(self):
        """Test that run method works with fallback synthesis when no LLM."""
        agent = SynthesisAgent(llm=None)

        # Create context with agent outputs
        context = AgentContext(
            user_id="test_user",
            session_id="test_session",
            query="Test fallback query",
            workflow_metadata={},
        )
        context.add_agent_output("refiner", "Refined output")
        context.add_agent_output("critic", "Critical output")

        # Run agent
        result_context = await agent.run(context)

        # Should use fallback synthesis
        assert "Synthesis" in result_context.agent_outputs
        synthesis_output = result_context.agent_outputs["Synthesis"]
        assert "Test fallback query" in synthesis_output
        assert "Refined output" in synthesis_output
        assert "Critical output" in synthesis_output

    def test_agent_has_required_attributes(self):
        """Test that SynthesisAgent has all required attributes for config integration."""
        agent = SynthesisAgent()

        # Required attributes for config integration
        assert hasattr(agent, "config")
        assert hasattr(agent, "_prompt_composer")
        assert hasattr(agent, "_composed_prompt")
        assert hasattr(agent, "update_config")
        assert hasattr(agent, "_update_composed_prompt")
        assert hasattr(agent, "_get_system_prompt")

        # Preserve existing SynthesisAgent attributes
        assert hasattr(agent, "llm")

        # Required methods should be callable
        assert callable(agent.update_config)
        assert callable(agent._update_composed_prompt)
        assert callable(agent._get_system_prompt)

    def test_config_property_type_safety(self):
        """Test that config property maintains type safety."""
        # With custom config
        custom_config = SynthesisConfig(synthesis_strategy="focused")
        agent_with_config = SynthesisAgent(config=custom_config)
        assert isinstance(agent_with_config.config, SynthesisConfig)

        # With default config
        agent_default = SynthesisAgent()
        assert isinstance(agent_default.config, SynthesisConfig)

    def test_parameter_order_and_compatibility(self):
        """Test that parameter order and types are preserved for backward compatibility."""
        mock_llm = Mock(spec=LLMInterface)

        # Test various parameter combinations

        # Original pattern (no params)
        agent1 = SynthesisAgent()
        assert (
            agent1.llm is not None or agent1.llm is None
        )  # Could be either based on OpenAI setup

        # With LLM
        agent2 = SynthesisAgent(llm=mock_llm)
        assert agent2.llm is mock_llm

        # With config (new)
        config = SynthesisConfig(synthesis_strategy="comprehensive")
        agent3 = SynthesisAgent(llm=mock_llm, config=config)
        assert agent3.llm is mock_llm
        assert agent3.config is config

    @patch("cognivault.agents.synthesis.agent.PromptComposer")
    def test_composed_prompt_usage_in_build_methods(self, mock_prompt_composer_class):
        """Test that build methods can use composed prompts from PromptComposer."""
        mock_llm = Mock(spec=LLMInterface)
        mock_composer = Mock()
        mock_prompt_composer_class.return_value = mock_composer

        # Mock composed prompt with analysis and synthesis templates
        mock_composed_prompt = Mock()
        mock_composed_prompt.analysis_prompt = (
            "Custom analysis: {query} - {outputs_text}"
        )
        mock_composed_prompt.synthesis_prompt = (
            "Custom synthesis: {query} - {themes_text}"
        )
        mock_composer.compose_synthesis_prompt.return_value = mock_composed_prompt
        mock_composer.validate_composition.return_value = True

        agent = SynthesisAgent(llm=mock_llm)

        # Test analysis prompt building
        outputs = {"refiner": "test output"}
        analysis_prompt = agent._build_analysis_prompt("test query", outputs)
        assert "Custom analysis: test query" in analysis_prompt

        # Test synthesis prompt building
        analysis = {"themes": ["test"], "conflicts": [], "key_topics": []}
        synthesis_prompt = agent._build_synthesis_prompt(
            "test query", outputs, analysis
        )
        assert "Custom synthesis: test query" in synthesis_prompt

    def test_default_llm_creation_with_config(self):
        """Test that default LLM creation works with config integration."""
        with patch(
            "cognivault.agents.synthesis.agent.SynthesisAgent._create_default_llm"
        ) as mock_create_llm:
            mock_llm = Mock(spec=LLMInterface)
            mock_create_llm.return_value = mock_llm

            # Test default LLM creation (llm="default")
            agent = SynthesisAgent()  # Uses default llm="default"

            # Should call default LLM creation
            mock_create_llm.assert_called_once()
            assert agent.llm is mock_llm

            # Should have config integration
            assert isinstance(agent.config, SynthesisConfig)

    @pytest.mark.asyncio
    async def test_emergency_fallback_preserves_functionality(self):
        """Test that emergency fallback works when all synthesis methods fail."""
        # Create agent with mock LLM that will fail
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate.side_effect = Exception("LLM failure")

        agent = SynthesisAgent(llm=mock_llm)

        # Create context
        context = AgentContext(
            user_id="test_user",
            session_id="test_session",
            query="Test emergency fallback",
            workflow_metadata={},
        )
        context.add_agent_output("refiner", "Test output")

        # Run agent - should handle failure gracefully
        result_context = await agent.run(context)

        # Should have emergency fallback output
        assert "Synthesis" in result_context.agent_outputs
        synthesis_output = result_context.agent_outputs["Synthesis"]
        assert "Test emergency fallback" in synthesis_output
        assert "Test output" in synthesis_output
