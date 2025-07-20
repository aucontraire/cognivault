"""
Tests for RefinerAgent configuration integration.

This module tests the enhanced RefinerAgent that supports configurable prompt
composition while maintaining backward compatibility.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.config.agent_configs import (
    RefinerConfig,
    PromptConfig,
    BehavioralConfig,
)
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface


class MockLLMResponse:
    """Mock LLM response for testing."""

    def __init__(self, text: str):
        self.text = text


class TestRefinerAgentBackwardCompatibility:
    """Test that RefinerAgent maintains backward compatibility."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLMInterface)
        self.mock_response = MockLLMResponse("Refined test query")
        self.mock_llm.generate.return_value = self.mock_response

    def test_backward_compatible_initialization(self):
        """Test that RefinerAgent can be initialized without config (backward compatibility)."""
        agent = RefinerAgent(self.mock_llm)

        assert agent.llm == self.mock_llm
        assert isinstance(agent.config, RefinerConfig)
        assert agent.config.refinement_level == "standard"  # Default value
        assert agent.config.behavioral_mode == "adaptive"  # Default value
        assert agent._prompt_composer is not None
        assert agent._composed_prompt is not None

    def test_backward_compatible_execution(self):
        """Test that RefinerAgent executes correctly without configuration."""
        agent = RefinerAgent(self.mock_llm)
        context = AgentContext(query="test query")

        # Mock the config to avoid actual file access
        with patch("cognivault.agents.refiner.agent.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.execution.enable_simulation_delay = False
            mock_get_config.return_value = mock_config

            # Run the agent
            result = run_async(agent.run(context))

            # Verify LLM was called
            assert self.mock_llm.generate.called
            # Verify output was added to context
            assert "Refiner" in result.agent_outputs
            assert "Refined test query" in result.agent_outputs["Refiner"]


class TestRefinerAgentConfiguration:
    """Test RefinerAgent with configuration support."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLMInterface)
        self.mock_response = MockLLMResponse("Enhanced refined query")
        self.mock_llm.generate.return_value = self.mock_response

    def test_initialization_with_config(self):
        """Test RefinerAgent initialization with custom configuration."""
        config = RefinerConfig(
            refinement_level="comprehensive",
            behavioral_mode="active",
            output_format="structured",
        )

        agent = RefinerAgent(self.mock_llm, config)

        assert agent.llm == self.mock_llm
        assert agent.config == config
        assert agent.config.refinement_level == "comprehensive"
        assert agent.config.behavioral_mode == "active"
        assert agent._composed_prompt is not None

    def test_config_influences_prompt_composition(self):
        """Test that configuration actually influences prompt composition."""
        # Create agent with comprehensive configuration
        config = RefinerConfig(
            refinement_level="comprehensive", behavioral_mode="active"
        )
        agent = RefinerAgent(self.mock_llm, config)

        # Get the composed prompt
        system_prompt = agent._get_system_prompt()

        # Verify that configuration influences are present
        assert (
            "exhaustive" in system_prompt.lower()
            or "comprehensive" in system_prompt.lower()
        )
        assert "proactive" in system_prompt.lower() or "active" in system_prompt.lower()

        # Verify it's different from default
        assert len(system_prompt) > len(
            agent._prompt_composer.get_default_prompt("refiner")
        )

    def test_custom_constraints_integration(self):
        """Test that custom constraints are integrated into the prompt."""
        config = RefinerConfig()
        config.behavioral_config.custom_constraints = [
            "preserve_technical_terminology",
            "maintain_academic_tone",
        ]

        agent = RefinerAgent(self.mock_llm, config)
        system_prompt = agent._get_system_prompt()

        # Verify custom constraints are in the prompt
        assert "preserve_technical_terminology" in system_prompt
        assert "maintain_academic_tone" in system_prompt

    def test_custom_system_prompt_override(self):
        """Test that custom system prompt completely overrides default."""
        config = RefinerConfig()
        config.prompt_config.custom_system_prompt = (
            "Custom refiner system prompt for testing"
        )

        agent = RefinerAgent(self.mock_llm, config)
        system_prompt = agent._get_system_prompt()

        assert system_prompt.startswith("Custom refiner system prompt for testing")

    def test_update_config_method(self):
        """Test dynamic configuration updates."""
        # Start with default config
        agent = RefinerAgent(self.mock_llm)
        original_config = agent.config

        # Update to comprehensive config
        new_config = RefinerConfig(
            refinement_level="comprehensive", behavioral_mode="active"
        )
        agent.update_config(new_config)

        assert agent.config == new_config
        assert agent.config != original_config
        assert agent.config.refinement_level == "comprehensive"

        # Verify prompt was recomposed
        system_prompt = agent._get_system_prompt()
        assert (
            "exhaustive" in system_prompt.lower()
            or "comprehensive" in system_prompt.lower()
        )

    def test_prompt_validation_fallback(self):
        """Test that invalid prompt composition falls back gracefully."""
        # Create agent with config
        config = RefinerConfig()
        agent = RefinerAgent(self.mock_llm, config)

        # Mock validation to fail
        with patch.object(
            agent._prompt_composer, "validate_composition", return_value=False
        ):
            system_prompt = agent._get_system_prompt()

            # Should fall back to default prompt
            from cognivault.agents.refiner.prompts import REFINER_SYSTEM_PROMPT

            assert system_prompt == REFINER_SYSTEM_PROMPT

    def test_execution_with_configuration(self):
        """Test agent execution uses configured prompts."""
        config = RefinerConfig(refinement_level="detailed", behavioral_mode="active")
        agent = RefinerAgent(self.mock_llm, config)
        context = AgentContext(query="test query")

        with patch("cognivault.agents.refiner.agent.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.execution.enable_simulation_delay = False
            mock_get_config.return_value = mock_config

            # Run the agent
            result = run_async(agent.run(context))

            # Verify LLM was called with configured prompt
            assert self.mock_llm.generate.called
            call_args = self.mock_llm.generate.call_args

            # The system_prompt should be the configured one, not the default
            system_prompt_used = call_args[1]["system_prompt"]  # keyword argument
            assert len(system_prompt_used) > 0

            # Verify output
            assert "Refiner" in result.agent_outputs
            assert "Enhanced refined query" in result.agent_outputs["Refiner"]


class TestRefinerAgentConfigurationEdgeCases:
    """Test edge cases and error handling for RefinerAgent configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLMInterface)
        self.mock_response = MockLLMResponse("Fallback response")
        self.mock_llm.generate.return_value = self.mock_response

    def test_composer_failure_fallback(self):
        """Test that composer failure falls back to default prompt."""
        config = RefinerConfig()

        # Mock the agent's prompt composer after initialization
        agent = RefinerAgent(self.mock_llm, config)

        # Mock the compose method to fail during update
        agent._prompt_composer.compose_refiner_prompt = Mock(
            side_effect=Exception("Composition failed")
        )

        # Trigger prompt update which should fail and set composed_prompt to None
        agent._update_composed_prompt()

        # Should have fallen back
        assert agent._composed_prompt is None

        # Should use default prompt
        from cognivault.agents.refiner.prompts import REFINER_SYSTEM_PROMPT

        system_prompt = agent._get_system_prompt()
        assert system_prompt == REFINER_SYSTEM_PROMPT

    def test_environment_variable_config_loading(self):
        """Test that configuration can be loaded from environment variables."""
        # This tests the integration with RefinerConfig.from_env()
        config = RefinerConfig.from_env()
        agent = RefinerAgent(self.mock_llm, config)

        assert isinstance(agent.config, RefinerConfig)
        assert agent._composed_prompt is not None

    def test_node_metadata_unchanged(self):
        """Test that configuration doesn't affect node metadata."""
        config = RefinerConfig(refinement_level="comprehensive")
        agent = RefinerAgent(self.mock_llm, config)

        metadata = agent.define_node_metadata()

        # Metadata should remain consistent regardless of configuration
        assert metadata["node_type"].value == "processor"
        assert metadata["dependencies"] == []
        assert "refiner" in metadata["tags"]
        assert len(metadata["inputs"]) == 1
        assert len(metadata["outputs"]) == 1


class TestRefinerAgentIntegration:
    """Integration tests for RefinerAgent with configuration system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLMInterface)
        self.mock_response = MockLLMResponse("Integration test response")
        self.mock_llm.generate.return_value = self.mock_response

    def test_full_workflow_with_config(self):
        """Test complete workflow from configuration to execution."""
        # Create configuration with multiple customizations
        config = RefinerConfig(
            refinement_level="comprehensive",
            behavioral_mode="active",
            output_format="structured",
        )
        config.behavioral_config.custom_constraints = [
            "preserve_intent",
            "enhance_clarity",
        ]
        config.prompt_config.template_variables = {"domain": "technical"}

        # Create agent
        agent = RefinerAgent(self.mock_llm, config)

        # Verify configuration was applied
        assert agent.config.refinement_level == "comprehensive"

        # Verify prompt composition worked
        system_prompt = agent._get_system_prompt()
        assert "preserve_intent" in system_prompt
        assert "enhance_clarity" in system_prompt

        # Execute agent
        context = AgentContext(query="test technical query")

        with patch("cognivault.agents.refiner.agent.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.execution.enable_simulation_delay = False
            mock_get_config.return_value = mock_config

            result = run_async(agent.run(context))

            # Verify execution completed successfully
            assert "Refiner" in result.agent_outputs
            assert self.mock_llm.generate.called

    def test_config_serialization_roundtrip(self):
        """Test that agent configuration can be serialized and restored."""
        # Create configured agent
        original_config = RefinerConfig(
            refinement_level="detailed", behavioral_mode="passive"
        )
        agent = RefinerAgent(self.mock_llm, original_config)

        # Serialize configuration
        config_dict = agent.config.model_dump()

        # Create new configuration from serialized data
        restored_config = RefinerConfig(**config_dict)

        # Create new agent with restored config
        new_agent = RefinerAgent(self.mock_llm, restored_config)

        # Verify they behave identically
        assert agent.config.refinement_level == new_agent.config.refinement_level
        assert agent.config.behavioral_mode == new_agent.config.behavioral_mode

        original_prompt = agent._get_system_prompt()
        restored_prompt = new_agent._get_system_prompt()
        assert original_prompt == restored_prompt


# Helper for running async tests in pytest
def run_async(coro):
    """Helper to run async functions in sync tests."""
    import asyncio

    return asyncio.run(coro)
