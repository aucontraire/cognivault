"""
Tests for the PromptComposer system.

This module tests the dynamic prompt composition system that enables configurable
agent behavior through runtime prompt modification.
"""

import pytest
from typing import Any
from unittest.mock import patch, MagicMock

from cognivault.workflows.prompt_composer import PromptComposer, ComposedPrompt
from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
    PromptConfig,
    BehavioralConfig,
    OutputConfig,
)

from tests.factories import (
    RefinerConfigFactory,
    CriticConfigFactory,
    SynthesisConfigFactory,
    HistorianConfigFactory,
)


class TestComposedPrompt:
    """Test ComposedPrompt container class."""

    def test_composed_prompt_creation(self) -> None:
        """Test ComposedPrompt creation and basic functionality."""
        prompt = ComposedPrompt(
            system_prompt="Test system prompt",
            templates={"greeting": "Hello {name}"},
            variables={"name": "Claude"},
            metadata={"agent_type": "test"},
        )

        assert prompt.system_prompt == "Test system prompt"
        assert prompt.templates["greeting"] == "Hello {name}"
        assert prompt.variables["name"] == "Claude"
        assert prompt.metadata["agent_type"] == "test"

    def test_get_template(self) -> None:
        """Test template retrieval."""
        prompt = ComposedPrompt(
            system_prompt="Test",
            templates={"greeting": "Hello {name}", "farewell": "Goodbye {name}"},
            variables={"name": "Claude"},
            metadata={},
        )

        assert prompt.get_template("greeting") == "Hello {name}"
        assert prompt.get_template("farewell") == "Goodbye {name}"
        assert prompt.get_template("nonexistent") is None

    def test_substitute_variables(self) -> None:
        """Test variable substitution in templates."""
        prompt = ComposedPrompt(
            system_prompt="Test",
            templates={},
            variables={"name": "Claude", "role": "assistant"},
            metadata={},
        )

        # Successful substitution
        result = prompt.substitute_variables("Hello {name}, you are an {role}")
        assert result == "Hello Claude, you are an assistant"

        # Missing variable - should return template as-is
        result = prompt.substitute_variables("Hello {missing_var}")
        assert result == "Hello {missing_var}"


class TestPromptComposer:
    """Test PromptComposer class functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_composer_initialization(self) -> None:
        """Test PromptComposer initialization."""
        assert isinstance(self.composer.default_prompts, dict)
        assert "refiner" in self.composer.default_prompts
        assert "critic" in self.composer.default_prompts
        assert "historian" in self.composer.default_prompts
        assert "synthesis" in self.composer.default_prompts

        assert isinstance(self.composer.behavioral_templates, dict)
        assert "refinement_level" in self.composer.behavioral_templates
        assert "analysis_depth" in self.composer.behavioral_templates

    def test_get_default_prompt(self) -> None:
        """Test default prompt retrieval."""
        refiner_prompt = self.composer.get_default_prompt("refiner")
        assert isinstance(refiner_prompt, str)
        assert len(refiner_prompt) > 0

        unknown_prompt = self.composer.get_default_prompt("unknown")
        assert unknown_prompt == "You are a helpful assistant."


class TestRefinerPromptComposition:
    """Test refiner agent prompt composition."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_compose_refiner_prompt_defaults(self) -> None:
        """Test refiner prompt composition with default configuration."""
        # 🚩 LIBERATED: Using factory for refiner configuration
        config = RefinerConfigFactory.generate_minimal_data()
        composed = self.composer.compose_refiner_prompt(config)

        assert isinstance(composed, ComposedPrompt)
        assert len(composed.system_prompt) > 0
        assert composed.metadata["agent_type"] == "refiner"
        assert composed.variables["refinement_level"] == "standard"
        assert composed.variables["behavioral_mode"] == "adaptive"
        assert composed.variables["output_format"] == "structured"

    def test_compose_refiner_prompt_custom_values(self) -> None:
        """Test refiner prompt composition with custom configuration."""
        # 🚩 LIBERATED: Using factory with custom constraints
        config = RefinerConfigFactory.with_custom_constraints(
            ["preserve_tone", "maintain_clarity"],
            refinement_level="comprehensive",
            behavioral_mode="active",
            output_format="prefixed",
        )

        composed = self.composer.compose_refiner_prompt(config)

        assert "exhaustive refinement" in composed.system_prompt.lower()
        assert "proactive" in composed.system_prompt.lower()
        assert "preserve_tone" in composed.system_prompt
        assert "maintain_clarity" in composed.system_prompt
        assert composed.variables["refinement_level"] == "comprehensive"

    def test_compose_refiner_prompt_custom_system_prompt(self) -> None:
        """Test refiner prompt composition with custom system prompt."""
        config = RefinerConfig()
        config.prompt_config.custom_system_prompt = "Custom refiner system prompt"

        composed = self.composer.compose_refiner_prompt(config)

        assert composed.system_prompt.startswith("Custom refiner system prompt")

    def test_compose_refiner_prompt_custom_templates(self) -> None:
        """Test refiner prompt composition with custom templates."""
        config = RefinerConfig()
        config.prompt_config.custom_templates = {
            "refinement": "Refine this: {query}",
            "analysis": "Analyze: {content}",
        }
        config.prompt_config.template_variables = {"style": "formal"}

        composed = self.composer.compose_refiner_prompt(config)

        assert "refinement" in composed.templates
        assert "analysis" in composed.templates
        assert composed.templates["refinement"] == "Refine this: {query}"
        assert composed.variables["style"] == "formal"


class TestCriticPromptComposition:
    """Test critic agent prompt composition."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_compose_critic_prompt_defaults(self) -> None:
        """Test critic prompt composition with default configuration."""
        config = CriticConfig()
        composed = self.composer.compose_critic_prompt(config)

        assert isinstance(composed, ComposedPrompt)
        assert composed.metadata["agent_type"] == "critic"
        assert composed.variables["analysis_depth"] == "medium"
        assert composed.variables["confidence_reporting"] == "True"
        assert composed.variables["bias_detection"] == "True"

    def test_compose_critic_prompt_analysis_depth(self) -> None:
        """Test critic prompt composition with different analysis depths."""
        config = CriticConfig(analysis_depth="deep")
        composed = self.composer.compose_critic_prompt(config)

        assert "deep" in composed.system_prompt.lower()
        assert composed.variables["analysis_depth"] == "deep"

    def test_compose_critic_prompt_confidence_and_bias(self) -> None:
        """Test critic prompt composition with confidence and bias settings."""
        config = CriticConfig(confidence_reporting=False, bias_detection=True)
        composed = self.composer.compose_critic_prompt(config)

        # Should not include confidence reporting
        assert "confidence scores" not in composed.system_prompt.lower()
        # Should include bias detection
        assert "bias" in composed.system_prompt.lower()
        assert composed.variables["confidence_reporting"] == "False"
        assert composed.variables["bias_detection"] == "True"

    def test_compose_critic_prompt_scoring_criteria(self) -> None:
        """Test critic prompt composition with custom scoring criteria."""
        config = CriticConfig(scoring_criteria=["accuracy", "depth", "novelty"])
        composed = self.composer.compose_critic_prompt(config)

        assert "accuracy" in composed.system_prompt
        assert "depth" in composed.system_prompt
        assert "novelty" in composed.system_prompt
        assert composed.variables["scoring_criteria"] == [
            "accuracy",
            "depth",
            "novelty",
        ]


class TestHistorianPromptComposition:
    """Test historian agent prompt composition."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_compose_historian_prompt_defaults(self) -> None:
        """Test historian prompt composition with default configuration."""
        config = HistorianConfig()
        composed = self.composer.compose_historian_prompt(config)

        assert isinstance(composed, ComposedPrompt)
        assert composed.metadata["agent_type"] == "historian"
        assert composed.variables["search_depth"] == "standard"
        assert composed.variables["relevance_threshold"] == "0.6"
        assert composed.variables["context_expansion"] == "True"
        assert composed.variables["memory_scope"] == "recent"

    def test_compose_historian_prompt_search_settings(self) -> None:
        """Test historian prompt composition with custom search settings."""
        config = HistorianConfig(
            search_depth="exhaustive",
            relevance_threshold=0.8,
            context_expansion=False,
            memory_scope="full",
        )
        composed = self.composer.compose_historian_prompt(config)

        assert "exhaustive" in composed.system_prompt.lower()
        assert "0.8" in composed.system_prompt
        assert composed.variables["search_depth"] == "exhaustive"
        assert composed.variables["relevance_threshold"] == "0.8"
        assert composed.variables["context_expansion"] == "False"
        assert composed.variables["memory_scope"] == "full"

    def test_compose_historian_prompt_metadata(self) -> None:
        """Test historian prompt composition metadata."""
        config = HistorianConfig(search_depth="deep", relevance_threshold=0.7)
        composed = self.composer.compose_historian_prompt(config)

        metadata = composed.metadata
        assert metadata["search_parameters"]["depth"] == "deep"
        assert metadata["search_parameters"]["threshold"] == 0.7
        assert metadata["search_parameters"]["expansion"] is True
        assert metadata["search_parameters"]["scope"] == "recent"


class TestSynthesisPromptComposition:
    """Test synthesis agent prompt composition."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_compose_synthesis_prompt_defaults(self) -> None:
        """Test synthesis prompt composition with default configuration."""
        config = SynthesisConfig()
        composed = self.composer.compose_synthesis_prompt(config)

        assert isinstance(composed, ComposedPrompt)
        assert composed.metadata["agent_type"] == "synthesis"
        assert composed.variables["synthesis_strategy"] == "balanced"
        assert composed.variables["thematic_focus"] == ""
        assert composed.variables["meta_analysis"] == "True"
        assert composed.variables["integration_mode"] == "adaptive"

    def test_compose_synthesis_prompt_strategy_and_focus(self) -> None:
        """Test synthesis prompt composition with strategy and thematic focus."""
        config = SynthesisConfig(
            synthesis_strategy="creative",
            thematic_focus="innovation",
            meta_analysis=False,
        )
        composed = self.composer.compose_synthesis_prompt(config)

        assert "innovative" in composed.system_prompt.lower()
        assert "innovation" in composed.system_prompt
        assert composed.variables["synthesis_strategy"] == "creative"
        assert composed.variables["thematic_focus"] == "innovation"
        assert composed.variables["meta_analysis"] == "False"

    def test_compose_synthesis_prompt_integration_mode(self) -> None:
        """Test synthesis prompt composition with different integration modes."""
        config = SynthesisConfig(integration_mode="hierarchical")
        composed = self.composer.compose_synthesis_prompt(config)

        assert "hierarchical" in composed.system_prompt.lower()
        assert composed.variables["integration_mode"] == "hierarchical"

    def test_compose_synthesis_prompt_metadata(self) -> None:
        """Test synthesis prompt composition metadata."""
        config = SynthesisConfig(
            synthesis_strategy="focused",
            thematic_focus="sustainability",
            meta_analysis=True,
            integration_mode="parallel",
        )
        composed = self.composer.compose_synthesis_prompt(config)

        features = composed.metadata["synthesis_features"]
        assert features["strategy"] == "focused"
        assert features["thematic_focus"] == "sustainability"
        assert features["meta_analysis"] is True
        assert features["integration_mode"] == "parallel"


class TestUniversalComposition:
    """Test universal prompt composition interface."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_compose_prompt_refiner(self) -> None:
        """Test universal compose_prompt method with refiner config."""
        config = RefinerConfig(refinement_level="detailed")
        composed = self.composer.compose_prompt("refiner", config)

        assert isinstance(composed, ComposedPrompt)
        assert composed.metadata["agent_type"] == "refiner"
        assert composed.variables["refinement_level"] == "detailed"

    def test_compose_prompt_critic(self) -> None:
        """Test universal compose_prompt method with critic config."""
        config = CriticConfig(analysis_depth="comprehensive")
        composed = self.composer.compose_prompt("critic", config)

        assert composed.metadata["agent_type"] == "critic"
        assert composed.variables["analysis_depth"] == "comprehensive"

    def test_compose_prompt_historian(self) -> None:
        """Test universal compose_prompt method with historian config."""
        config = HistorianConfig(search_depth="deep")
        composed = self.composer.compose_prompt("historian", config)

        assert composed.metadata["agent_type"] == "historian"
        assert composed.variables["search_depth"] == "deep"

    def test_compose_prompt_synthesis(self) -> None:
        """Test universal compose_prompt method with synthesis config."""
        config = SynthesisConfig(synthesis_strategy="comprehensive")
        composed = self.composer.compose_prompt("synthesis", config)

        assert composed.metadata["agent_type"] == "synthesis"
        assert composed.variables["synthesis_strategy"] == "comprehensive"

    def test_compose_prompt_invalid_agent_type(self) -> None:
        """Test universal compose_prompt method with invalid agent type."""
        config = RefinerConfig()

        with pytest.raises(ValueError, match="Unsupported agent type"):
            self.composer.compose_prompt("invalid_agent", config)


class TestPromptValidation:
    """Test prompt validation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_validate_composition_valid_prompt(self) -> None:
        """Test validation of valid composed prompt."""
        prompt = ComposedPrompt(
            system_prompt="Valid system prompt",
            templates={"greeting": "Hello {name}"},
            variables={"name": "Claude"},
            metadata={"agent_type": "test"},
        )

        assert self.composer.validate_composition(prompt) is True

    def test_validate_composition_empty_system_prompt(self) -> None:
        """Test validation fails for empty system prompt."""
        # Pydantic validation now prevents creating ComposedPrompt with empty system_prompt
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ComposedPrompt(system_prompt="", templates={}, variables={}, metadata={})

        assert "system_prompt cannot be empty" in str(exc_info.value)

    def test_validate_composition_invalid_template(self) -> None:
        """Test validation fails for template with missing variables."""
        prompt = ComposedPrompt(
            system_prompt="Valid prompt",
            templates={"greeting": "Hello {missing_var}"},
            variables={"name": "Claude"},  # missing 'missing_var'
            metadata={},
        )

        assert self.composer.validate_composition(prompt) is False

    def test_validate_composition_valid_template_with_variables(self) -> None:
        """Test validation passes for template with all required variables."""
        prompt = ComposedPrompt(
            system_prompt="Valid prompt",
            templates={"greeting": "Hello {name}, you are {role}"},
            variables={"name": "Claude", "role": "assistant"},
            metadata={},
        )

        assert self.composer.validate_composition(prompt) is True


class TestBehavioralModifications:
    """Test behavioral modification patterns in prompt composition."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.composer = PromptComposer()

    def test_output_format_instructions(self) -> None:
        """Test output format instruction generation."""
        # Test direct format specification
        instructions = self.composer._get_output_format_instructions(
            "structured", "markdown"
        )
        assert "structure" in instructions.lower()

        # Test adaptive format falling back to general format
        instructions = self.composer._get_output_format_instructions(
            "adaptive", "markdown"
        )
        assert "markdown" in instructions.lower()

    def test_behavioral_template_integration(self) -> None:
        """Test that behavioral templates are properly integrated."""
        config = RefinerConfig(
            refinement_level="comprehensive", behavioral_mode="active"
        )
        composed = self.composer.compose_refiner_prompt(config)

        # Check that behavioral templates are included
        assert (
            "exhaustive" in composed.system_prompt.lower()
            or "comprehensive" in composed.system_prompt.lower()
        )
        assert (
            "proactive" in composed.system_prompt.lower()
            or "active" in composed.system_prompt.lower()
        )

        # Check that the base prompt is preserved
        assert len(composed.system_prompt) > len(
            self.composer.default_prompts["refiner"]
        )

    def test_custom_constraints_integration(self) -> None:
        """Test that custom constraints are properly integrated."""
        config = RefinerConfig()
        config.behavioral_config.custom_constraints = [
            "preserve_technical_terminology",
            "maintain_academic_tone",
            "include_source_citations",
        ]

        composed = self.composer.compose_refiner_prompt(config)

        for constraint in config.behavioral_config.custom_constraints:
            assert constraint in composed.system_prompt
