"""
Unit tests for workflow composer agent configuration integration.

Tests the new create_agent_config function and the updated workflow composer
that integrates with the agent configuration system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict

from cognivault.workflows.composer import (
    create_agent_config,
    get_agent_class,
    NodeFactory,
)
from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
    PromptConfig,
    BehavioralConfig,
)


class TestCreateAgentConfig:
    """Test the create_agent_config function for workflow integration."""

    def test_create_default_config_without_dict(self) -> None:
        """Test creating default configuration when no config dict provided."""
        # Test each agent type gets its appropriate default config
        refiner_config = create_agent_config("refiner")
        assert isinstance(refiner_config, RefinerConfig)
        assert refiner_config.refinement_level == "standard"
        assert refiner_config.behavioral_mode == "adaptive"

        critic_config = create_agent_config("critic")
        assert isinstance(critic_config, CriticConfig)
        assert critic_config.analysis_depth == "medium"
        assert critic_config.confidence_reporting is True

        historian_config = create_agent_config("historian")
        assert isinstance(historian_config, HistorianConfig)
        assert historian_config.search_depth == "standard"
        assert historian_config.relevance_threshold == 0.6

        synthesis_config = create_agent_config("synthesis")
        assert isinstance(synthesis_config, SynthesisConfig)
        assert synthesis_config.synthesis_strategy == "balanced"
        assert synthesis_config.meta_analysis is True

    def test_create_config_with_empty_dict(self) -> None:
        """Test creating configuration with empty dictionary."""
        config = create_agent_config("refiner", {})
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "standard"  # Default value

    def test_create_refiner_config_with_agent_specific_fields(self) -> None:
        """Test creating RefinerConfig with agent-specific configuration."""
        config_dict = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            "output_format": "structured",
        }

        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "comprehensive"
        assert config.behavioral_mode == "active"
        assert config.output_format == "structured"

    def test_create_critic_config_with_agent_specific_fields(self) -> None:
        """Test creating CriticConfig with agent-specific configuration."""
        config_dict = {
            "analysis_depth": "deep",
            "confidence_reporting": False,
            "bias_detection": True,
            "scoring_criteria": ["accuracy", "objectivity"],
        }

        config = create_agent_config("critic", config_dict)
        assert isinstance(config, CriticConfig)
        assert config.analysis_depth == "deep"
        assert config.confidence_reporting is False
        assert config.bias_detection is True
        assert config.scoring_criteria == ["accuracy", "objectivity"]

    def test_create_historian_config_with_agent_specific_fields(self) -> None:
        """Test creating HistorianConfig with agent-specific configuration."""
        config_dict = {
            "search_depth": "exhaustive",
            "relevance_threshold": 0.8,
            "context_expansion": False,
            "memory_scope": "full",
        }

        config = create_agent_config("historian", config_dict)
        assert isinstance(config, HistorianConfig)
        assert config.search_depth == "exhaustive"
        assert config.relevance_threshold == 0.8
        assert config.context_expansion is False
        assert config.memory_scope == "full"

    def test_create_synthesis_config_with_agent_specific_fields(self) -> None:
        """Test creating SynthesisConfig with agent-specific configuration."""
        config_dict = {
            "synthesis_strategy": "creative",
            "thematic_focus": "innovation",
            "meta_analysis": False,
            "integration_mode": "hierarchical",
        }

        config = create_agent_config("synthesis", config_dict)
        assert isinstance(config, SynthesisConfig)
        assert config.synthesis_strategy == "creative"
        assert config.thematic_focus == "innovation"
        assert config.meta_analysis is False
        assert config.integration_mode == "hierarchical"

    def test_create_config_with_legacy_prompt_format(self) -> None:
        """Test creating configuration with legacy prompt format."""
        config_dict = {
            "prompts": {
                "system_prompt": "Custom system prompt for testing",
                "templates": {"analysis": "Custom analysis template"},
            }
        }

        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert (
            config.prompt_config.custom_system_prompt
            == "Custom system prompt for testing"
        )
        assert config.prompt_config.custom_templates == {
            "analysis": "Custom analysis template"
        }

    def test_create_config_with_behavioral_constraints(self) -> None:
        """Test creating configuration with behavioral constraints."""
        config_dict = {
            "custom_constraints": ["preserve_intent", "enhance_clarity"],
            "fallback_mode": "strict",
        }

        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert config.behavioral_config.custom_constraints == [
            "preserve_intent",
            "enhance_clarity",
        ]
        assert config.behavioral_config.fallback_mode == "strict"

    def test_create_config_with_mixed_configuration(self) -> None:
        """Test creating configuration with both agent-specific and nested configs."""
        config_dict = {
            "refinement_level": "detailed",
            "prompts": {"system_prompt": "Mixed config prompt"},
            "custom_constraints": ["mixed_constraint"],
            "behavioral_mode": "passive",
        }

        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "detailed"
        assert config.behavioral_mode == "passive"
        assert config.prompt_config.custom_system_prompt == "Mixed config prompt"
        assert config.behavioral_config.custom_constraints == ["mixed_constraint"]

    def test_create_config_ignores_unknown_fields(self) -> None:
        """Test that unknown fields are ignored without errors."""
        config_dict = {
            "refinement_level": "standard",
            "unknown_field": "should_be_ignored",
            "another_unknown": 42,
        }

        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "standard"
        # Unknown fields should not cause errors or be included

    def test_create_config_fallback_for_unknown_agent_type(self) -> None:
        """Test that unknown agent types fall back to RefinerConfig."""
        config = create_agent_config("unknown_agent")
        assert isinstance(config, RefinerConfig)

    def test_create_config_with_template_variables(self) -> None:
        """Test creating configuration with template variables."""
        config_dict = {
            "prompts": {
                "templates": {"custom": "Template with {variable}"},
                "template_variables": {"variable": "value"},
            }
        }

        config = create_agent_config("refiner", config_dict)
        assert config.prompt_config.custom_templates == {
            "custom": "Template with {variable}"
        }
        # Note: template_variables are handled by PromptConfig, not extracted separately


class TestNodeFactoryConfigIntegration:
    """Test NodeFactory integration with agent configuration system."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.node_factory = NodeFactory()

    @pytest.fixture
    def mock_node_config(self) -> Any:
        """Create a mock NodeConfiguration for testing."""
        mock_config: Mock = Mock()
        mock_config.node_id = "test_refiner"
        mock_config.node_type = "refiner"
        mock_config.category = "BASE"
        mock_config.config = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
        }
        return mock_config

    @pytest.fixture
    def mock_node_config_with_prompts(self) -> Any:
        """Create a mock NodeConfiguration with legacy prompt config."""
        mock_config: Mock = Mock()
        mock_config.node_id = "test_critic"
        mock_config.node_type = "critic"
        mock_config.category = "BASE"
        mock_config.config = {
            "analysis_depth": "deep",
            "prompts": {"system_prompt": "Custom critic prompt"},
        }
        return mock_config

    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig.load")
    def test_create_base_node_uses_agent_config(
        self, mock_config_load: Any, mock_llm_class: Any, mock_node_config: Any
    ) -> None:
        """Test that _create_base_node creates agents with proper configuration."""
        # Setup mocks
        mock_config_load.return_value = Mock(
            api_key="test-key", model="gpt-4", base_url=None
        )
        mock_llm: Mock = Mock()
        mock_llm_class.return_value = mock_llm

        # Mock the agent class and instance
        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            # Create a mock agent class that accepts config parameter
            mock_agent_class: Mock = Mock()
            # Set up the mock to have a signature that accepts config
            mock_signature: Mock = Mock()
            mock_signature.parameters.keys.return_value = ["self", "llm", "config"]

            with patch("inspect.signature", return_value=mock_signature):
                mock_agent_instance: Mock = Mock()
                mock_agent_instance.name = "Refiner"
                mock_agent_instance.run = AsyncMock(
                    return_value=Mock(agent_outputs={"Refiner": "Test output"})
                )
                mock_agent_class.return_value = mock_agent_instance
                mock_get_agent.return_value = mock_agent_class

                # Create the node function
                node_func = self.node_factory._create_base_node(mock_node_config)

                # Execute the node function with test state
                import asyncio

                test_state = {"query": "test query"}
                result = asyncio.run(node_func(test_state))

                # Verify agent was created with configuration
                mock_agent_class.assert_called_once()
                call_args = mock_agent_class.call_args
                assert len(call_args[0]) == 2  # llm and config arguments
                llm_arg, config_arg = call_args[0]

                # Verify configuration is properly typed
                assert isinstance(config_arg, RefinerConfig)
                assert config_arg.refinement_level == "comprehensive"
                assert config_arg.behavioral_mode == "active"

                # Verify result
                assert "test_refiner" in result
                assert result["test_refiner"]["output"] == "Test output"

    @patch("cognivault.workflows.prompt_loader.apply_prompt_configuration")
    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig.load")
    def test_create_base_node_applies_legacy_prompts(
        self,
        mock_config_load: Any,
        mock_llm_class: Any,
        mock_apply_prompts: Any,
        mock_node_config_with_prompts: Any,
    ) -> None:
        """Test that legacy prompt configuration is still applied for backward compatibility."""
        # Setup mocks
        mock_config_load.return_value = Mock(
            api_key="test-key", model="gpt-4", base_url=None
        )
        mock_llm: Mock = Mock()
        mock_llm_class.return_value = mock_llm
        mock_apply_prompts.return_value = {"system_prompt": "Applied legacy prompt"}

        # Mock the agent class and instance
        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class: Mock = Mock()
            mock_agent_instance: Mock = Mock()
            mock_agent_instance.name = "Critic"
            mock_agent_instance.system_prompt = None
            mock_agent_instance.run = AsyncMock(
                return_value=Mock(agent_outputs={"Critic": "Test output"})
            )
            mock_agent_class.return_value = mock_agent_instance
            mock_get_agent.return_value = mock_agent_class

            # Create the node function
            node_func = self.node_factory._create_base_node(
                mock_node_config_with_prompts
            )

            # Execute the node function
            import asyncio

            test_state = {"query": "test query"}
            result = asyncio.run(node_func(test_state))

            # Verify legacy prompt configuration was applied
            mock_apply_prompts.assert_called_once_with(
                "critic", mock_node_config_with_prompts.config
            )
            assert mock_agent_instance.system_prompt == "Applied legacy prompt"

    @patch("cognivault.llm.openai.OpenAIChatLLM")
    @patch("cognivault.config.openai_config.OpenAIConfig.load")
    def test_create_base_node_with_no_config(
        self, mock_config_load: Any, mock_llm_class: Any
    ) -> None:
        """Test that node creation works with no configuration (backward compatibility)."""
        # Setup mocks
        mock_config_load.return_value = Mock(
            api_key="test-key", model="gpt-4", base_url=None
        )
        mock_llm: Mock = Mock()
        mock_llm_class.return_value = mock_llm

        # Create node config without configuration
        mock_node_config: Mock = Mock()
        mock_node_config.node_id = "test_agent"
        mock_node_config.node_type = "refiner"
        mock_node_config.category = "BASE"
        mock_node_config.config = None

        # Mock the agent class and instance
        with patch("cognivault.workflows.composer.get_agent_class") as mock_get_agent:
            mock_agent_class: Mock = Mock()
            # Set up the mock to have a signature that accepts config
            mock_signature: Mock = Mock()
            mock_signature.parameters.keys.return_value = ["self", "llm", "config"]

            with patch("inspect.signature", return_value=mock_signature):
                mock_agent_instance: Mock = Mock()
                mock_agent_instance.name = "Refiner"
                mock_agent_instance.run = AsyncMock(
                    return_value=Mock(agent_outputs={"Refiner": "Default output"})
                )
                mock_agent_class.return_value = mock_agent_instance
                mock_get_agent.return_value = mock_agent_class

                # Create the node function
                node_func = self.node_factory._create_base_node(mock_node_config)

                # Execute the node function
                import asyncio

                test_state = {"query": "test query"}
                result = asyncio.run(node_func(test_state))

                # Verify agent was created with default configuration
                mock_agent_class.assert_called_once()
                call_args = mock_agent_class.call_args
                llm_arg, config_arg = call_args[0]

                # Should get default RefinerConfig
                assert isinstance(config_arg, RefinerConfig)
                assert config_arg.refinement_level == "standard"  # Default

    def test_create_base_node_handles_agent_creation_error(self) -> None:
        """Test that node creation handles agent creation errors gracefully."""
        mock_node_config: Mock = Mock()
        mock_node_config.node_id = "failing_agent"
        mock_node_config.node_type = "nonexistent"
        mock_node_config.category = "BASE"
        mock_node_config.config = {}

        # This should not raise an exception during node function creation
        node_func = self.node_factory._create_base_node(mock_node_config)

        # The error should be handled during execution
        import asyncio

        test_state = {"query": "test query"}
        result = asyncio.run(node_func(test_state))

        # Should return fallback output
        assert "failing_agent" in result
        assert "error" in result["failing_agent"]["output"].lower()


class TestGetAgentClass:
    """Test the get_agent_class function."""

    def test_get_known_agent_classes(self) -> None:
        """Test getting known agent classes."""
        from cognivault.agents.refiner.agent import RefinerAgent
        from cognivault.agents.critic.agent import CriticAgent
        from cognivault.agents.historian.agent import HistorianAgent
        from cognivault.agents.synthesis.agent import SynthesisAgent

        assert get_agent_class("refiner") == RefinerAgent
        assert get_agent_class("critic") == CriticAgent
        assert get_agent_class("historian") == HistorianAgent
        assert get_agent_class("synthesis") == SynthesisAgent

    def test_get_unknown_agent_class_returns_default(self) -> None:
        """Test that unknown agent types return RefinerAgent as default."""
        from cognivault.agents.refiner.agent import RefinerAgent

        assert get_agent_class("unknown") == RefinerAgent
        assert get_agent_class("nonexistent") == RefinerAgent
        assert get_agent_class("") == RefinerAgent


class TestWorkflowComposerBackwardCompatibility:
    """Test that workflow composer maintains backward compatibility."""

    def test_legacy_workflow_still_works(self) -> None:
        """Test that legacy workflows without agent configs still work."""
        # This would be tested more comprehensively in integration tests
        # but we can verify the factory patterns work with legacy configs

        config_dict = {
            "prompts": {"system_prompt": "Legacy system prompt"},
            "refinement_level": "detailed",  # Old style mixing
        }

        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "detailed"
        assert config.prompt_config.custom_system_prompt == "Legacy system prompt"

    def test_mixed_old_and_new_config_format(self) -> None:
        """Test that mixing old and new configuration formats works."""
        config_dict = {
            # New agent-specific format
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            # Legacy prompt format
            "prompts": {"system_prompt": "Mixed prompt"},
            # Behavioral config format
            "custom_constraints": ["mixed_constraint"],
        }

        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "comprehensive"
        assert config.behavioral_mode == "active"
        assert config.prompt_config.custom_system_prompt == "Mixed prompt"
        assert config.behavioral_config.custom_constraints == ["mixed_constraint"]
