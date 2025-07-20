"""
Comprehensive unit tests for the legacy prompt_loader.py system.

This module tests the critical backward compatibility layer that integrates
our new Pydantic-based configuration system with the existing prompt loading
infrastructure. Missing coverage here poses immediate risk to architectural migration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from cognivault.workflows.prompt_loader import (
    load_agent_prompts,
    get_system_prompt,
    get_template_prompt,
    apply_rich_configuration_to_prompts,
    apply_prompt_configuration,
    _apply_refiner_config,
    _apply_historian_config,
    _apply_critic_config,
    _apply_synthesis_config,
)


class TestLoadAgentPrompts:
    """Test loading prompts from agent prompts.py modules."""

    @patch("cognivault.workflows.prompt_loader.importlib.import_module")
    def test_load_agent_prompts_success(self, mock_import):
        """Test successful loading of agent prompts."""
        # Create a mock prompts module
        mock_module = Mock()
        mock_module.REFINER_SYSTEM_PROMPT = "You are a query refiner."
        mock_module.REFINER_ANALYSIS_TEMPLATE = "Analyze: {query}"
        mock_module._private_var = "should be ignored"
        mock_module.NON_STRING_VAR = 123  # Should be ignored

        # Mock dir() to return attribute names
        with patch(
            "builtins.dir",
            return_value=[
                "REFINER_SYSTEM_PROMPT",
                "REFINER_ANALYSIS_TEMPLATE",
                "_private_var",
                "NON_STRING_VAR",
            ],
        ):
            mock_import.return_value = mock_module

            result = load_agent_prompts("refiner")

            # Verify correct module import (may be called multiple times due to nested calls)
            mock_import.assert_called_with("cognivault.agents.refiner.prompts")

            # Verify only public string attributes are included
            expected = {
                "REFINER_SYSTEM_PROMPT": "You are a query refiner.",
                "REFINER_ANALYSIS_TEMPLATE": "Analyze: {query}",
            }
            assert result == expected

    @patch("cognivault.workflows.prompt_loader.importlib.import_module")
    def test_load_agent_prompts_import_error(self, mock_import):
        """Test handling of missing prompts.py file."""
        mock_import.side_effect = ImportError(
            "No module named 'cognivault.agents.nonexistent.prompts'"
        )

        result = load_agent_prompts("nonexistent")

        assert result == {}
        mock_import.assert_called_once_with("cognivault.agents.nonexistent.prompts")

    @patch("cognivault.workflows.prompt_loader.importlib.import_module")
    def test_load_agent_prompts_general_exception(self, mock_import):
        """Test handling of general exceptions during prompt loading."""
        mock_import.side_effect = AttributeError("Some unexpected error")

        result = load_agent_prompts("broken")

        assert result == {}
        mock_import.assert_called_once_with("cognivault.agents.broken.prompts")

    @patch("cognivault.workflows.prompt_loader.importlib.import_module")
    def test_load_agent_prompts_empty_module(self, mock_import):
        """Test loading from module with no relevant attributes."""
        mock_module = Mock()
        with patch("builtins.dir", return_value=["__name__", "__doc__"]):
            mock_import.return_value = mock_module

            result = load_agent_prompts("empty")

            assert result == {}

    @patch("cognivault.workflows.prompt_loader.importlib.import_module")
    def test_load_agent_prompts_mixed_attributes(self, mock_import):
        """Test loading from module with mixed attribute types."""
        mock_module = Mock()
        mock_module.STRING_PROMPT = "Valid prompt"
        mock_module.DICT_CONFIG = {"key": "value"}
        mock_module.LIST_ITEMS = ["item1", "item2"]
        mock_module.INT_VALUE = 42
        mock_module.BOOL_FLAG = True
        mock_module.NONE_VALUE = None

        with patch(
            "builtins.dir",
            return_value=[
                "STRING_PROMPT",
                "DICT_CONFIG",
                "LIST_ITEMS",
                "INT_VALUE",
                "BOOL_FLAG",
                "NONE_VALUE",
            ],
        ):
            mock_import.return_value = mock_module

            result = load_agent_prompts("mixed")

            # Only string values should be included
            assert result == {"STRING_PROMPT": "Valid prompt"}


class TestGetSystemPrompt:
    """Test system prompt retrieval with custom overrides."""

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_system_prompt_with_custom_override(self, mock_load):
        """Test that custom prompt overrides default."""
        mock_load.return_value = {"REFINER_SYSTEM_PROMPT": "Default prompt"}

        result = get_system_prompt("refiner", "Custom prompt")

        assert result == "Custom prompt"
        # Should not load prompts when custom is provided
        mock_load.assert_not_called()

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_system_prompt_standard_key(self, mock_load):
        """Test retrieval using standard system prompt key."""
        mock_load.return_value = {"REFINER_SYSTEM_PROMPT": "Standard refiner prompt"}

        result = get_system_prompt("refiner")

        assert result == "Standard refiner prompt"
        mock_load.assert_called_once_with("refiner")

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_system_prompt_fallback_key(self, mock_load):
        """Test fallback to generic SYSTEM_PROMPT key."""
        mock_load.return_value = {"SYSTEM_PROMPT": "Generic system prompt"}

        result = get_system_prompt("custom_agent")

        assert result == "Generic system prompt"
        mock_load.assert_called_once_with("custom_agent")

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_system_prompt_no_prompt_found(self, mock_load):
        """Test handling when no system prompt is found."""
        mock_load.return_value = {"OTHER_PROMPT": "Not a system prompt"}

        result = get_system_prompt("agent_without_system_prompt")

        assert result is None
        mock_load.assert_called_once_with("agent_without_system_prompt")

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_system_prompt_empty_prompts(self, mock_load):
        """Test handling when no prompts are loaded."""
        mock_load.return_value = {}

        result = get_system_prompt("empty_agent")

        assert result is None
        mock_load.assert_called_once_with("empty_agent")


class TestGetTemplatePrompt:
    """Test template prompt retrieval and formatting."""

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_template_prompt_success(self, mock_load):
        """Test successful template prompt retrieval and formatting."""
        mock_load.return_value = {
            "REFINER_ANALYSIS_TEMPLATE": "Analyze this query: {query} with focus on {focus}"
        }

        result = get_template_prompt(
            "refiner", "analysis", query="test query", focus="clarity"
        )

        assert result == "Analyze this query: test query with focus on clarity"
        mock_load.assert_called_once_with("refiner")

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_template_prompt_missing_variable(self, mock_load):
        """Test handling of missing template variables."""
        mock_load.return_value = {
            "REFINER_ANALYSIS_TEMPLATE": "Analyze this query: {query} with focus on {missing_var}"
        }

        result = get_template_prompt("refiner", "analysis", query="test query")

        assert result is None  # Should return None when template formatting fails
        mock_load.assert_called_once_with("refiner")

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_template_prompt_not_found(self, mock_load):
        """Test handling when template is not found."""
        mock_load.return_value = {"OTHER_TEMPLATE": "Not the template we want"}

        result = get_template_prompt("refiner", "nonexistent")

        assert result is None
        mock_load.assert_called_once_with("refiner")

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_get_template_prompt_no_variables(self, mock_load):
        """Test template prompt without variables."""
        mock_load.return_value = {
            "CRITIC_FEEDBACK_TEMPLATE": "Provide feedback on the analysis."
        }

        result = get_template_prompt("critic", "feedback")

        assert result == "Provide feedback on the analysis."
        mock_load.assert_called_once_with("critic")


class TestApplyRichConfigurationToPrompts:
    """Test rich configuration application to prompts."""

    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_rich_config_refiner(self, mock_get_prompt):
        """Test applying rich configuration to refiner agent."""
        mock_get_prompt.return_value = "Base refiner prompt."

        config = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            "output_format": "structured",
            "custom_constraints": ["preserve_intent", "enhance_clarity"],
        }

        result = apply_rich_configuration_to_prompts("refiner", config)

        assert "Base refiner prompt." in result
        assert "ACTIVE MODE with maximum thoroughness" in result
        assert "Always actively refine" in result
        assert "structured bullet points" in result
        assert "preserve_intent" in result
        assert "enhance_clarity" in result
        mock_get_prompt.assert_called_once_with("refiner")

    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_rich_config_historian(self, mock_get_prompt):
        """Test applying rich configuration to historian agent."""
        mock_get_prompt.return_value = "Base historian prompt."

        config = {
            "search_depth": "exhaustive",
            "analysis_mode": "analytical",
            "focus_areas": ["recent_developments", "key_patterns"],
        }

        result = apply_rich_configuration_to_prompts("historian", config)

        assert "Base historian prompt." in result
        assert "Exhaustive - examine all available sources" in result
        assert "analytical approach with deep pattern recognition" in result
        assert "recent_developments" in result
        assert "key_patterns" in result
        mock_get_prompt.assert_called_once_with("historian")

    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_rich_config_critic(self, mock_get_prompt):
        """Test applying rich configuration to critic agent."""
        mock_get_prompt.return_value = "Base critic prompt."

        config = {
            "analysis_depth": "deep",
            "bias_detection": False,
            "categories": ["accuracy", "completeness"],
        }

        result = apply_rich_configuration_to_prompts("critic", config)

        assert "Base critic prompt." in result
        assert "Deep - perform comprehensive analysis" in result
        assert "BIAS DETECTION: Disabled" in result
        assert "accuracy" in result
        assert "completeness" in result
        mock_get_prompt.assert_called_once_with("critic")

    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_rich_config_synthesis(self, mock_get_prompt):
        """Test applying rich configuration to synthesis agent."""
        mock_get_prompt.return_value = "Base synthesis prompt."

        config = {
            "synthesis_mode": "basic",
            "output_style": "executive",
            "integration_strategy": "weighted",
            "structure": ["executive_summary", "key_findings", "recommendations"],
        }

        result = apply_rich_configuration_to_prompts("synthesis", config)

        assert "Base synthesis prompt." in result
        assert "Basic - provide concise integration" in result
        assert "Executive - clear, action-oriented" in result
        assert "weighted approach - prioritize higher-confidence" in result
        assert "executive_summary" in result
        assert "key_findings" in result
        assert "recommendations" in result
        mock_get_prompt.assert_called_once_with("synthesis")

    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_rich_config_unknown_agent(self, mock_get_prompt):
        """Test handling unknown agent type."""
        mock_get_prompt.return_value = "Base prompt."

        result = apply_rich_configuration_to_prompts("unknown_agent", {})

        assert result == "Base prompt."
        mock_get_prompt.assert_called_once_with("unknown_agent")

    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_rich_config_no_base_prompt(self, mock_get_prompt):
        """Test handling when no base prompt is available."""
        mock_get_prompt.return_value = None

        result = apply_rich_configuration_to_prompts("missing_agent", {})

        assert result == ""
        mock_get_prompt.assert_called_once_with("missing_agent")


class TestRefinerConfigApplication:
    """Test refiner-specific configuration application."""

    def test_apply_refiner_config_detailed_level(self):
        """Test detailed refinement level configuration."""
        base_prompt = "Base prompt."
        config = {"refinement_level": "detailed"}

        result = _apply_refiner_config(base_prompt, config)

        assert "Base prompt." in result
        assert "ACTIVE MODE. Provide comprehensive refinements" in result

    def test_apply_refiner_config_minimal_level(self):
        """Test minimal refinement level configuration."""
        base_prompt = "Base prompt."
        config = {"refinement_level": "minimal"}

        result = _apply_refiner_config(base_prompt, config)

        assert "PASSIVE MODE. Only refine if critically necessary" in result

    def test_apply_refiner_config_comprehensive_level(self):
        """Test comprehensive refinement level configuration."""
        base_prompt = "Base prompt."
        config = {"refinement_level": "comprehensive"}

        result = _apply_refiner_config(base_prompt, config)

        assert "ACTIVE MODE with maximum thoroughness" in result

    def test_apply_refiner_config_behavioral_modes(self):
        """Test behavioral mode configurations."""
        base_prompt = "Base prompt."

        # Test active mode
        result = _apply_refiner_config(base_prompt, {"behavioral_mode": "active"})
        assert "Always actively refine and improve" in result

        # Test passive mode
        result = _apply_refiner_config(base_prompt, {"behavioral_mode": "passive"})
        assert "Only refine when significant improvements" in result

    def test_apply_refiner_config_output_formats(self):
        """Test output format configurations."""
        base_prompt = "Base prompt."

        # Test structured format
        result = _apply_refiner_config(base_prompt, {"output_format": "structured"})
        assert "structured bullet points" in result

        # Test raw format
        result = _apply_refiner_config(base_prompt, {"output_format": "raw"})
        assert "without prefixes or formatting" in result

    def test_apply_refiner_config_custom_constraints(self):
        """Test custom constraints application."""
        base_prompt = "Base prompt."
        config = {
            "custom_constraints": ["maintain_technical_accuracy", "preserve_context"]
        }

        result = _apply_refiner_config(base_prompt, config)

        assert "ADDITIONAL CONSTRAINTS:" in result
        assert "maintain_technical_accuracy" in result
        assert "preserve_context" in result

    def test_apply_refiner_config_empty_config(self):
        """Test applying empty configuration."""
        base_prompt = "Base prompt."
        config = {}

        result = _apply_refiner_config(base_prompt, config)

        assert result == "Base prompt."


class TestHistorianConfigApplication:
    """Test historian-specific configuration application."""

    def test_apply_historian_config_search_depths(self):
        """Test search depth configurations."""
        base_prompt = "Base prompt."

        # Test exhaustive search
        result = _apply_historian_config(base_prompt, {"search_depth": "exhaustive"})
        assert "Exhaustive - examine all available sources" in result

        # Test basic search
        result = _apply_historian_config(base_prompt, {"search_depth": "basic"})
        assert "Basic - focus on most relevant and recent" in result

    def test_apply_historian_config_analysis_modes(self):
        """Test analysis mode configurations."""
        base_prompt = "Base prompt."

        # Test analytical mode
        result = _apply_historian_config(base_prompt, {"analysis_mode": "analytical"})
        assert "analytical approach with deep pattern recognition" in result

        # Test factual mode
        result = _apply_historian_config(base_prompt, {"analysis_mode": "factual"})
        assert "factual content and documented evidence" in result

    def test_apply_historian_config_focus_areas(self):
        """Test focus areas configuration."""
        base_prompt = "Base prompt."
        config = {
            "focus_areas": [
                "technology_trends",
                "market_analysis",
                "regulatory_changes",
            ]
        }

        result = _apply_historian_config(base_prompt, config)

        assert "FOCUS AREAS:" in result
        assert "technology_trends" in result
        assert "market_analysis" in result
        assert "regulatory_changes" in result

    def test_apply_historian_config_empty_focus_areas(self):
        """Test empty focus areas."""
        base_prompt = "Base prompt."
        config = {"focus_areas": []}

        result = _apply_historian_config(base_prompt, config)

        assert "FOCUS AREAS:" not in result


class TestCriticConfigApplication:
    """Test critic-specific configuration application."""

    def test_apply_critic_config_analysis_depths(self):
        """Test analysis depth configurations."""
        base_prompt = "Base prompt."

        # Test deep analysis
        result = _apply_critic_config(base_prompt, {"analysis_depth": "deep"})
        assert "Deep - perform comprehensive analysis" in result

        # Test shallow analysis
        result = _apply_critic_config(base_prompt, {"analysis_depth": "shallow"})
        assert "Shallow - focus on obvious issues" in result

    def test_apply_critic_config_bias_detection(self):
        """Test bias detection configuration."""
        base_prompt = "Base prompt."

        # Test disabled bias detection
        result = _apply_critic_config(base_prompt, {"bias_detection": False})
        assert "BIAS DETECTION: Disabled" in result

        # Test enabled bias detection (default)
        result = _apply_critic_config(base_prompt, {"bias_detection": True})
        assert "BIAS DETECTION: Disabled" not in result

    def test_apply_critic_config_categories(self):
        """Test categories focus configuration."""
        base_prompt = "Base prompt."
        config = {
            "categories": [
                "logical_consistency",
                "evidence_quality",
                "argument_strength",
            ]
        }

        result = _apply_critic_config(base_prompt, config)

        assert "FOCUS CATEGORIES:" in result
        assert "logical_consistency" in result
        assert "evidence_quality" in result
        assert "argument_strength" in result


class TestSynthesisConfigApplication:
    """Test synthesis-specific configuration application."""

    def test_apply_synthesis_config_synthesis_modes(self):
        """Test synthesis mode configurations."""
        base_prompt = "Base prompt."

        # Test basic synthesis
        result = _apply_synthesis_config(base_prompt, {"synthesis_mode": "basic"})
        assert "Basic - provide concise integration" in result

        # Test comprehensive synthesis
        result = _apply_synthesis_config(
            base_prompt, {"synthesis_mode": "comprehensive"}
        )
        assert "Comprehensive - detailed integration with full analysis" in result

    def test_apply_synthesis_config_output_styles(self):
        """Test output style configurations."""
        base_prompt = "Base prompt."

        # Test all output styles
        styles = {
            "executive": "Executive - clear, action-oriented, business-focused",
            "academic": "Academic - scholarly, detailed, evidence-based",
            "technical": "Technical - precise, factual, implementation-focused",
            "legal": "Legal - precise, risk-aware, compliance-focused",
        }

        for style, expected_text in styles.items():
            result = _apply_synthesis_config(base_prompt, {"output_style": style})
            assert expected_text in result

    def test_apply_synthesis_config_integration_strategies(self):
        """Test integration strategy configurations."""
        base_prompt = "Base prompt."

        # Test weighted integration
        result = _apply_synthesis_config(
            base_prompt, {"integration_strategy": "weighted"}
        )
        assert "weighted approach - prioritize higher-confidence" in result

        # Test sequential integration
        result = _apply_synthesis_config(
            base_prompt, {"integration_strategy": "sequential"}
        )
        assert "sequential approach - integrate in order" in result

    def test_apply_synthesis_config_structure(self):
        """Test structure requirements configuration."""
        base_prompt = "Base prompt."
        config = {
            "structure": ["executive_summary", "detailed_analysis", "conclusions"]
        }

        result = _apply_synthesis_config(base_prompt, config)

        assert "REQUIRED STRUCTURE:" in result
        assert "executive_summary" in result
        assert "detailed_analysis" in result
        assert "conclusions" in result


class TestApplyPromptConfiguration:
    """Test the main prompt configuration application function."""

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_prompt_config_custom_prompts(self, mock_get_prompt, mock_load):
        """Test applying custom prompt configuration."""
        mock_get_prompt.return_value = "Custom system prompt"
        mock_load.return_value = {
            "REFINER_FEEDBACK_TEMPLATE": "Default feedback template"
        }

        config = {
            "prompts": {
                "system_prompt": "Custom system prompt",
                "templates": {
                    "analysis": "Custom analysis template",
                    "feedback": "",  # Empty should use default
                },
            }
        }

        result = apply_prompt_configuration("refiner", config)

        expected = {
            "system_prompt": "Custom system prompt",
            "analysis_template": "Custom analysis template",
            "feedback_template": "Default feedback template",  # Uses default for empty
        }
        assert result == expected
        mock_get_prompt.assert_called_once_with("refiner", "Custom system prompt")

    @patch("cognivault.workflows.prompt_loader.apply_rich_configuration_to_prompts")
    def test_apply_prompt_config_rich_configuration(self, mock_rich_config):
        """Test applying rich configuration when no custom prompts."""
        mock_rich_config.return_value = "Rich configured prompt"

        config = {"refinement_level": "comprehensive", "behavioral_mode": "active"}

        result = apply_prompt_configuration("refiner", config)

        expected = {"system_prompt": "Rich configured prompt"}
        assert result == expected
        mock_rich_config.assert_called_once_with("refiner", config)

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_prompt_config_no_system_prompt(self, mock_get_prompt, mock_load):
        """Test handling when no system prompt is available."""
        mock_get_prompt.return_value = None
        mock_load.return_value = {}

        config = {"prompts": {"templates": {"analysis": "Custom template"}}}

        result = apply_prompt_configuration("refiner", config)

        expected = {"analysis_template": "Custom template"}
        assert result == expected

    @patch("cognivault.workflows.prompt_loader.apply_rich_configuration_to_prompts")
    def test_apply_prompt_config_empty_rich_result(self, mock_rich_config):
        """Test handling when rich configuration returns empty result."""
        mock_rich_config.return_value = ""

        config = {"refinement_level": "standard"}

        result = apply_prompt_configuration("refiner", config)

        assert result == {}
        mock_rich_config.assert_called_once_with("refiner", config)

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_apply_prompt_config_empty_config(self, mock_load):
        """Test handling empty configuration."""
        mock_load.return_value = {}

        # Empty config should trigger rich configuration path
        with patch(
            "cognivault.workflows.prompt_loader.apply_rich_configuration_to_prompts"
        ) as mock_rich:
            mock_rich.return_value = ""
            result = apply_prompt_configuration("refiner", {})

            assert result == {}
            mock_rich.assert_called_once_with("refiner", {})

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_prompt_config_missing_default_template(
        self, mock_get_prompt, mock_load
    ):
        """Test handling when default template is not available for empty custom template."""
        mock_get_prompt.return_value = (
            "Custom prompt"  # get_system_prompt returns custom prompt when provided
        )
        mock_load.return_value = {}  # No default templates

        config = {
            "prompts": {
                "system_prompt": "Custom prompt",
                "templates": {"missing": ""},  # Empty template with no default
            }
        }

        result = apply_prompt_configuration("refiner", config)

        # Should only include system prompt, not the missing template
        expected = {"system_prompt": "Custom prompt"}
        assert result == expected


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @patch("cognivault.workflows.prompt_loader.get_system_prompt")
    def test_apply_rich_config_all_agent_types(self, mock_get_prompt):
        """Test rich configuration for all supported agent types."""
        mock_get_prompt.return_value = "Base prompt"

        agent_types = ["refiner", "historian", "critic", "synthesis"]

        for agent_type in agent_types:
            result = apply_rich_configuration_to_prompts(agent_type, {})
            assert "Base prompt" in result

        # Verify all agent types were processed
        assert mock_get_prompt.call_count == len(agent_types)

    def test_refiner_config_all_options_combination(self):
        """Test refiner configuration with all options combined."""
        base_prompt = "Base prompt."
        config = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            "output_format": "structured",
            "custom_constraints": ["constraint1", "constraint2"],
        }

        result = _apply_refiner_config(base_prompt, config)

        # Verify all options are applied
        assert "Base prompt." in result
        assert "ACTIVE MODE with maximum thoroughness" in result
        assert "Always actively refine" in result
        assert "structured bullet points" in result
        assert "constraint1" in result
        assert "constraint2" in result

    def test_historian_config_all_options_combination(self):
        """Test historian configuration with all options combined."""
        base_prompt = "Base prompt."
        config = {
            "search_depth": "exhaustive",
            "analysis_mode": "analytical",
            "focus_areas": ["area1", "area2"],
        }

        result = _apply_historian_config(base_prompt, config)

        # Verify all options are applied
        assert "Base prompt." in result
        assert "Exhaustive - examine all available sources" in result
        assert "analytical approach with deep pattern recognition" in result
        assert "area1" in result
        assert "area2" in result

    def test_synthesis_config_all_options_combination(self):
        """Test synthesis configuration with all options combined."""
        base_prompt = "Base prompt."
        config = {
            "synthesis_mode": "comprehensive",
            "output_style": "academic",
            "integration_strategy": "weighted",
            "structure": ["section1", "section2"],
        }

        result = _apply_synthesis_config(base_prompt, config)

        # Verify all options are applied
        assert "Base prompt." in result
        assert "Comprehensive - detailed integration" in result
        assert "Academic - scholarly, detailed" in result
        assert "weighted approach - prioritize higher-confidence" in result
        assert "section1" in result
        assert "section2" in result

    @patch("cognivault.workflows.prompt_loader.load_agent_prompts")
    def test_template_prompt_case_sensitivity(self, mock_load):
        """Test template prompt key case sensitivity."""
        mock_load.return_value = {
            "REFINER_ANALYSIS_TEMPLATE": "Correct template",
            "refiner_analysis_template": "Wrong case template",
        }

        # Should find the uppercase version
        result = get_template_prompt("refiner", "analysis")

        assert result == "Correct template"

    def test_config_with_none_values(self):
        """Test configuration handling with None values."""
        base_prompt = "Base prompt."

        # Test refiner with None values
        config = {
            "refinement_level": None,
            "behavioral_mode": None,
            "custom_constraints": None,
        }

        result = _apply_refiner_config(base_prompt, config)

        # Should handle None values gracefully
        assert result == "Base prompt."

    def test_config_with_empty_lists(self):
        """Test configuration handling with empty lists."""
        base_prompt = "Base prompt."

        # Test all agents with empty constraint/area lists
        configs = [
            ("refiner", {"custom_constraints": []}),
            ("historian", {"focus_areas": []}),
            ("critic", {"categories": []}),
            ("synthesis", {"structure": []}),
        ]

        for agent_type, config in configs:
            if agent_type == "refiner":
                result = _apply_refiner_config(base_prompt, config)
            elif agent_type == "historian":
                result = _apply_historian_config(base_prompt, config)
            elif agent_type == "critic":
                result = _apply_critic_config(base_prompt, config)
            elif agent_type == "synthesis":
                result = _apply_synthesis_config(base_prompt, config)

            # Should not include empty list sections
            assert "ADDITIONAL CONSTRAINTS:" not in result
            assert "FOCUS AREAS:" not in result
            assert "FOCUS CATEGORIES:" not in result
            assert "REQUIRED STRUCTURE:" not in result
