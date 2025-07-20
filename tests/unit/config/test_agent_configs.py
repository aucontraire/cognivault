"""
Tests for agent configuration classes.

This module tests the Pydantic-based configuration system for dynamic agent
behavior modification, including validation, environment loading, and integration
with the existing prompt system.
"""

import os
import pytest
from typing import Dict, Any
from unittest.mock import patch

from cognivault.config.agent_configs import (
    PromptConfig,
    BehavioralConfig,
    OutputConfig,
    ExecutionConfig,
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
    get_agent_config_class,
    create_agent_config,
)


class TestBaseConfigurations:
    """Test base configuration classes."""

    def test_prompt_config_defaults(self):
        """Test PromptConfig default values."""
        config = PromptConfig()
        assert config.custom_system_prompt is None
        assert config.custom_templates == {}
        assert config.template_variables == {}

    def test_prompt_config_custom_values(self):
        """Test PromptConfig with custom values."""
        config = PromptConfig(
            custom_system_prompt="Custom prompt",
            custom_templates={"greeting": "Hello {name}"},
            template_variables={"name": "Claude"},
        )
        assert config.custom_system_prompt == "Custom prompt"
        assert config.custom_templates["greeting"] == "Hello {name}"
        assert config.template_variables["name"] == "Claude"

    def test_behavioral_config_defaults(self):
        """Test BehavioralConfig default values."""
        config = BehavioralConfig()
        assert config.custom_constraints == []
        assert config.fallback_mode == "adaptive"

    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        config = OutputConfig()
        assert config.format_preference == "structured"
        assert config.include_metadata is True
        assert config.confidence_threshold == 0.7

    def test_output_config_validation(self):
        """Test OutputConfig validation constraints."""
        # Valid confidence threshold
        config = OutputConfig(confidence_threshold=0.5)
        assert config.confidence_threshold == 0.5

        # Invalid confidence threshold - too low
        with pytest.raises(ValueError):
            OutputConfig(confidence_threshold=-0.1)

        # Invalid confidence threshold - too high
        with pytest.raises(ValueError):
            OutputConfig(confidence_threshold=1.1)

    def test_execution_config_defaults(self):
        """Test ExecutionConfig default values."""
        config = ExecutionConfig()
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.enable_caching is True

    def test_execution_config_validation(self):
        """Test ExecutionConfig validation constraints."""
        # Valid timeout
        config = ExecutionConfig(timeout_seconds=60)
        assert config.timeout_seconds == 60

        # Invalid timeout - too low
        with pytest.raises(ValueError):
            ExecutionConfig(timeout_seconds=0)

        # Invalid timeout - too high
        with pytest.raises(ValueError):
            ExecutionConfig(timeout_seconds=400)

        # Valid retries
        config = ExecutionConfig(max_retries=5)
        assert config.max_retries == 5

        # Invalid retries - too high
        with pytest.raises(ValueError):
            ExecutionConfig(max_retries=15)


class TestRefinerConfig:
    """Test RefinerAgent configuration."""

    def test_refiner_config_defaults(self):
        """Test RefinerConfig default values."""
        config = RefinerConfig()
        assert config.refinement_level == "standard"
        assert config.behavioral_mode == "adaptive"
        assert config.output_format == "structured"
        assert isinstance(config.prompt_config, PromptConfig)
        assert isinstance(config.behavioral_config, BehavioralConfig)
        assert isinstance(config.output_config, OutputConfig)
        assert isinstance(config.execution_config, ExecutionConfig)

    def test_refiner_config_custom_values(self):
        """Test RefinerConfig with custom values."""
        config = RefinerConfig(
            refinement_level="comprehensive",
            behavioral_mode="active",
            output_format="prefixed",
        )
        assert config.refinement_level == "comprehensive"
        assert config.behavioral_mode == "active"
        assert config.output_format == "prefixed"

    def test_refiner_config_validation(self):
        """Test RefinerConfig validation."""
        # Valid values
        RefinerConfig(refinement_level="minimal")
        RefinerConfig(behavioral_mode="passive")
        RefinerConfig(output_format="raw")

        # Invalid refinement level
        with pytest.raises(ValueError):
            RefinerConfig(refinement_level="invalid")

    def test_refiner_config_from_dict(self):
        """Test RefinerConfig.from_dict() method."""
        config_dict = {
            "refinement_level": "detailed",
            "behavioral_mode": "active",
            "output_format": "structured",
        }
        config = RefinerConfig.from_dict(config_dict)
        assert config.refinement_level == "detailed"
        assert config.behavioral_mode == "active"
        assert config.output_format == "structured"

    def test_refiner_config_from_env(self):
        """Test RefinerConfig.from_env() method."""
        with patch.dict(
            os.environ,
            {
                "REFINER_REFINEMENT_LEVEL": "comprehensive",
                "REFINER_BEHAVIORAL_MODE": "passive",
                "REFINER_OUTPUT_FORMAT": "raw",
            },
        ):
            config = RefinerConfig.from_env()
            assert config.refinement_level == "comprehensive"
            assert config.behavioral_mode == "passive"
            assert config.output_format == "raw"

    def test_refiner_config_from_env_custom_prefix(self):
        """Test RefinerConfig.from_env() with custom prefix."""
        with patch.dict(
            os.environ,
            {"CUSTOM_REFINEMENT_LEVEL": "detailed", "CUSTOM_BEHAVIORAL_MODE": "active"},
        ):
            config = RefinerConfig.from_env(prefix="CUSTOM")
            assert config.refinement_level == "detailed"
            assert config.behavioral_mode == "active"

    def test_refiner_config_to_prompt_config(self):
        """Test RefinerConfig.to_prompt_config() method."""
        config = RefinerConfig(
            refinement_level="comprehensive",
            behavioral_mode="active",
            output_format="structured",
        )
        config.behavioral_config.custom_constraints = ["preserve_tone"]
        config.prompt_config.template_variables = {"style": "formal"}

        prompt_config = config.to_prompt_config()
        assert prompt_config["refinement_level"] == "comprehensive"
        assert prompt_config["behavioral_mode"] == "active"
        assert prompt_config["output_format"] == "structured"
        assert prompt_config["custom_constraints"] == ["preserve_tone"]
        assert prompt_config["template_variables"] == {"style": "formal"}


class TestCriticConfig:
    """Test CriticAgent configuration."""

    def test_critic_config_defaults(self):
        """Test CriticConfig default values."""
        config = CriticConfig()
        assert config.analysis_depth == "medium"
        assert config.confidence_reporting is True
        assert config.bias_detection is True
        assert config.scoring_criteria == ["accuracy", "completeness", "objectivity"]

    def test_critic_config_custom_values(self):
        """Test CriticConfig with custom values."""
        config = CriticConfig(
            analysis_depth="deep",
            confidence_reporting=False,
            bias_detection=False,
            scoring_criteria=["accuracy", "clarity"],
        )
        assert config.analysis_depth == "deep"
        assert config.confidence_reporting is False
        assert config.bias_detection is False
        assert config.scoring_criteria == ["accuracy", "clarity"]

    def test_critic_config_from_env(self):
        """Test CriticConfig.from_env() method."""
        with patch.dict(
            os.environ,
            {
                "CRITIC_ANALYSIS_DEPTH": "comprehensive",
                "CRITIC_CONFIDENCE_REPORTING": "false",
                "CRITIC_BIAS_DETECTION": "true",
            },
        ):
            config = CriticConfig.from_env()
            assert config.analysis_depth == "comprehensive"
            assert config.confidence_reporting is False
            assert config.bias_detection is True

    def test_critic_config_to_prompt_config(self):
        """Test CriticConfig.to_prompt_config() method."""
        config = CriticConfig(
            analysis_depth="deep",
            confidence_reporting=False,
            scoring_criteria=["accuracy", "depth"],
        )

        prompt_config = config.to_prompt_config()
        assert prompt_config["analysis_depth"] == "deep"
        assert prompt_config["confidence_reporting"] == "False"
        assert prompt_config["scoring_criteria"] == ["accuracy", "depth"]


class TestHistorianConfig:
    """Test HistorianAgent configuration."""

    def test_historian_config_defaults(self):
        """Test HistorianConfig default values."""
        config = HistorianConfig()
        assert config.search_depth == "standard"
        assert config.relevance_threshold == 0.6
        assert config.context_expansion is True
        assert config.memory_scope == "recent"

    def test_historian_config_validation(self):
        """Test HistorianConfig validation."""
        # Valid relevance threshold
        config = HistorianConfig(relevance_threshold=0.8)
        assert config.relevance_threshold == 0.8

        # Invalid relevance threshold
        with pytest.raises(ValueError):
            HistorianConfig(relevance_threshold=1.5)

    def test_historian_config_from_env(self):
        """Test HistorianConfig.from_env() method."""
        with patch.dict(
            os.environ,
            {
                "HISTORIAN_SEARCH_DEPTH": "exhaustive",
                "HISTORIAN_RELEVANCE_THRESHOLD": "0.8",
                "HISTORIAN_CONTEXT_EXPANSION": "false",
                "HISTORIAN_MEMORY_SCOPE": "full",
            },
        ):
            config = HistorianConfig.from_env()
            assert config.search_depth == "exhaustive"
            assert config.relevance_threshold == 0.8
            assert config.context_expansion is False
            assert config.memory_scope == "full"


class TestSynthesisConfig:
    """Test SynthesisAgent configuration."""

    def test_synthesis_config_defaults(self):
        """Test SynthesisConfig default values."""
        config = SynthesisConfig()
        assert config.synthesis_strategy == "balanced"
        assert config.thematic_focus is None
        assert config.meta_analysis is True
        assert config.integration_mode == "adaptive"

    def test_synthesis_config_custom_values(self):
        """Test SynthesisConfig with custom values."""
        config = SynthesisConfig(
            synthesis_strategy="creative",
            thematic_focus="innovation",
            meta_analysis=False,
            integration_mode="hierarchical",
        )
        assert config.synthesis_strategy == "creative"
        assert config.thematic_focus == "innovation"
        assert config.meta_analysis is False
        assert config.integration_mode == "hierarchical"

    def test_synthesis_config_to_prompt_config(self):
        """Test SynthesisConfig.to_prompt_config() method."""
        config = SynthesisConfig(
            synthesis_strategy="focused",
            thematic_focus="sustainability",
            meta_analysis=True,
        )

        prompt_config = config.to_prompt_config()
        assert prompt_config["synthesis_strategy"] == "focused"
        assert prompt_config["thematic_focus"] == "sustainability"
        assert prompt_config["meta_analysis"] == "True"


class TestFactoryFunctions:
    """Test factory functions for agent configurations."""

    def test_get_agent_config_class(self):
        """Test get_agent_config_class() function."""
        assert get_agent_config_class("refiner") == RefinerConfig
        assert get_agent_config_class("critic") == CriticConfig
        assert get_agent_config_class("historian") == HistorianConfig
        assert get_agent_config_class("synthesis") == SynthesisConfig

        with pytest.raises(ValueError):
            get_agent_config_class("unknown")

    def test_create_agent_config(self):
        """Test create_agent_config() factory function."""
        config_dict = {"refinement_level": "comprehensive"}
        config = create_agent_config("refiner", config_dict)
        assert isinstance(config, RefinerConfig)
        assert config.refinement_level == "comprehensive"

        config_dict = {"analysis_depth": "deep"}
        config = create_agent_config("critic", config_dict)
        assert isinstance(config, CriticConfig)
        assert config.analysis_depth == "deep"


class TestConfigurationIntegration:
    """Test integration between different configuration classes."""

    def test_nested_config_modification(self):
        """Test modifying nested configuration objects."""
        config = RefinerConfig()

        # Modify nested configurations
        config.prompt_config.custom_system_prompt = "Custom refiner prompt"
        config.behavioral_config.custom_constraints = ["preserve_context"]
        config.output_config.confidence_threshold = 0.8
        config.execution_config.timeout_seconds = 60

        assert config.prompt_config.custom_system_prompt == "Custom refiner prompt"
        assert config.behavioral_config.custom_constraints == ["preserve_context"]
        assert config.output_config.confidence_threshold == 0.8
        assert config.execution_config.timeout_seconds == 60

    def test_config_serialization(self):
        """Test configuration serialization to dict."""
        config = RefinerConfig(
            refinement_level="comprehensive", behavioral_mode="active"
        )
        config.behavioral_config.custom_constraints = ["test_constraint"]

        # Test that config can be converted to dict and back
        config_dict = config.model_dump()
        assert config_dict["refinement_level"] == "comprehensive"
        assert config_dict["behavioral_mode"] == "active"
        assert config_dict["behavioral_config"]["custom_constraints"] == [
            "test_constraint"
        ]

        # Test recreation from dict
        new_config = RefinerConfig(**config_dict)
        assert new_config.refinement_level == "comprehensive"
        assert new_config.behavioral_mode == "active"
        assert new_config.behavioral_config.custom_constraints == ["test_constraint"]

    def test_invalid_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValueError):
            RefinerConfig(invalid_field="should_fail")

    def test_prompt_config_integration(self):
        """Test prompt configuration integration across all agents."""
        agents_configs = [
            RefinerConfig(refinement_level="comprehensive"),
            CriticConfig(analysis_depth="deep"),
            HistorianConfig(search_depth="exhaustive"),
            SynthesisConfig(synthesis_strategy="creative"),
        ]

        for config in agents_configs:
            # All configs should have prompt_config
            assert hasattr(config, "prompt_config")
            assert isinstance(config.prompt_config, PromptConfig)

            # All configs should have to_prompt_config method
            assert hasattr(config, "to_prompt_config")
            prompt_config_dict = config.to_prompt_config()
            assert isinstance(prompt_config_dict, dict)
