"""
Test suite for config_mapper.py Pydantic integration.

This test suite validates the Pydantic migration of ConfigMapper including:
- Modernized model_validate_config() method using Pydantic's native validation
- Backward compatibility with existing methods
- Flat-to-nested configuration mapping
- Value transformation and validation
- Error handling and logging
"""

import pytest
from typing import Any, Dict
from unittest.mock import patch

from cognivault.config.config_mapper import ConfigMapper
from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
)


class TestConfigMapperPydanticIntegration:
    """Test Pydantic integration features of ConfigMapper."""

    def test_model_validate_config_nested_format(self) -> None:
        """Test model_validate_config with nested format (direct Pydantic validation)."""
        # Arrange - nested format that matches Pydantic schema
        nested_config = {
            "refinement_level": "detailed",
            "behavioral_mode": "adaptive",
            "output_format": "structured",
            "prompt_config": {
                "custom_system_prompt": "Custom refiner prompt",
                "template_variables": {"key": "value"},
            },
            "behavioral_config": {
                "custom_constraints": ["constraint1", "constraint2"],
                "fallback_mode": "graceful",
            },
            "output_config": {
                "format_preference": "markdown",
                "include_metadata": True,
                "confidence_threshold": 0.8,
            },
            "execution_config": {
                "timeout_seconds": 45,
                "max_retries": 2,
                "enable_caching": False,
            },
        }

        # Act
        result = ConfigMapper.model_validate_config(nested_config, "refiner")

        # Assert
        assert result is not None
        assert isinstance(result, RefinerConfig)
        assert result.refinement_level == "detailed"
        assert result.behavioral_mode == "adaptive"
        assert result.output_format == "structured"
        assert result.prompt_config.custom_system_prompt == "Custom refiner prompt"
        assert result.behavioral_config.custom_constraints == [
            "constraint1",
            "constraint2",
        ]
        assert result.output_config.confidence_threshold == 0.8
        assert result.execution_config.timeout_seconds == 45

    def test_model_validate_config_flat_format(self) -> None:
        """Test model_validate_config with flat format (fallback mapping)."""
        # Arrange - flat format that requires mapping
        flat_config = {
            "refinement_level": "comprehensive",
            "custom_constraints": ["flat_constraint1", "flat_constraint2"],
            "custom_system_prompt": "Flat format prompt",
            "template_variables": {"flat_key": "flat_value"},
            "format_preference": "raw",
            "include_metadata": False,
            "confidence_threshold": 0.9,
            "timeout_seconds": 60,
            "max_retries": 5,
            "enable_caching": True,
        }

        # Act
        result = ConfigMapper.model_validate_config(flat_config, "refiner")

        # Assert
        assert result is not None
        assert isinstance(result, RefinerConfig)
        assert result.refinement_level == "comprehensive"
        assert result.prompt_config.custom_system_prompt == "Flat format prompt"
        assert result.prompt_config.template_variables == {"flat_key": "flat_value"}
        assert result.behavioral_config.custom_constraints == [
            "flat_constraint1",
            "flat_constraint2",
        ]
        assert result.output_config.format_preference == "raw"
        assert result.output_config.include_metadata is False
        assert result.execution_config.timeout_seconds == 60

    def test_model_validate_config_all_agent_types(self) -> None:
        """Test model_validate_config works for all supported agent types."""
        # Test configurations for each agent type
        test_configs = {
            "refiner": {"refinement_level": "standard", "behavioral_mode": "active"},
            "critic": {
                "analysis_depth": "deep",
                "confidence_reporting": True,
                "bias_detection": False,
            },
            "historian": {
                "search_depth": "exhaustive",
                "relevance_threshold": 0.7,
                "context_expansion": True,
                "memory_scope": "full",
            },
            "synthesis": {
                "synthesis_strategy": "balanced",
                "thematic_focus": "integration",
                "meta_analysis": True,
                "integration_mode": "hierarchical",
            },
        }

        expected_classes = {
            "refiner": RefinerConfig,
            "critic": CriticConfig,
            "historian": HistorianConfig,
            "synthesis": SynthesisConfig,
        }

        for agent_type, config in test_configs.items():
            # Act
            assert isinstance(config, dict)
            typed_config: Dict[str, Any] = config
            result = ConfigMapper.model_validate_config(typed_config, agent_type)

            # Assert
            assert result is not None, f"Failed to create config for {agent_type}"
            assert isinstance(result, expected_classes[agent_type])

    def test_model_validate_config_invalid_agent_type(self) -> None:
        """Test model_validate_config with invalid agent type."""
        # Arrange
        config = {"some_field": "some_value"}

        # Act
        result = ConfigMapper.model_validate_config(config, "unknown_agent")

        # Assert
        assert result is None

    def test_model_validate_config_empty_config(self) -> None:
        """Test model_validate_config with empty configuration."""
        # Act
        result = ConfigMapper.model_validate_config({}, "refiner")

        # Assert
        assert result is None

    def test_model_validate_config_none_config(self) -> None:
        """Test model_validate_config with None configuration."""
        # Act
        result = ConfigMapper.model_validate_config(None, "refiner")

        # Assert
        assert result is None

    @patch("cognivault.config.config_mapper.logger")
    def test_model_validate_config_logging(self, mock_logger: Any) -> None:
        """Test that model_validate_config produces appropriate log messages."""
        # Test case 1: Empty config
        ConfigMapper.model_validate_config({}, "refiner")
        mock_logger.debug.assert_called_with("No config data provided for refiner")

        # Test case 2: Unknown agent type
        ConfigMapper.model_validate_config({"field": "value"}, "unknown")
        mock_logger.warning.assert_called_with("Unknown agent type: unknown")

        # Test case 3: Validation failure
        invalid_config = {"refinement_level": "invalid_value"}
        ConfigMapper.model_validate_config(invalid_config, "refiner")

        # Should have logged debug message about direct validation failure
        debug_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if "Direct validation failed for refiner" in str(call)
        ]
        assert len(debug_calls) > 0


class TestConfigMapperValueTransformation:
    """Test value transformation features of ConfigMapper."""

    def test_transform_value_historian_search_depth(self) -> None:
        """Test value transformation for historian search_depth field."""
        # Test mapping from chart values to schema values
        test_cases = [
            ("comprehensive", "deep"),
            ("standard", "standard"),
            ("shallow", "shallow"),
            ("exhaustive", "exhaustive"),
        ]

        for input_value, expected_output in test_cases:
            # Act
            result = ConfigMapper._transform_value(
                "search_depth", input_value, "historian"
            )

            # Assert
            assert result == expected_output

    def test_transform_value_historian_context_expansion(self) -> None:
        """Test value transformation for historian context_expansion field."""
        # Test mapping from string to boolean
        test_cases = [
            ("broad", True),
            ("narrow", False),
            ("true", True),
            ("false", False),
        ]

        for input_value, expected_output in test_cases:
            # Act
            result = ConfigMapper._transform_value(
                "context_expansion", input_value, "historian"
            )

            # Assert
            assert result == expected_output

    def test_transform_value_synthesis_strategy(self) -> None:
        """Test value transformation for synthesis strategy field."""
        # Test mapping from long names to schema values
        test_cases = [
            ("comprehensive_integration", "comprehensive"),
            ("balanced", "balanced"),
            ("focused", "focused"),
            ("creative", "creative"),
        ]

        for input_value, expected_output in test_cases:
            # Act
            result = ConfigMapper._transform_value(
                "synthesis_strategy", input_value, "synthesis"
            )

            # Assert
            assert result == expected_output

    def test_transform_value_boolean_strings(self) -> None:
        """Test transformation of boolean strings."""
        # Test boolean string conversion
        test_cases = [
            ("true", True),
            ("false", False),
            ("True", True),
            ("False", False),
            ("TRUE", True),
            ("FALSE", False),
        ]

        for input_value, expected_output in test_cases:
            # Act
            result = ConfigMapper._transform_value(
                "any_field", input_value, "any_agent"
            )

            # Assert
            assert result == expected_output

    def test_transform_value_no_transformation(self) -> None:
        """Test that values without transformation mappings are returned unchanged."""
        # Test various value types
        test_values = [
            "unchanged_string",
            42,
            3.14,
            ["list", "of", "values"],
            {"dict": "value"},
            None,
        ]

        for value in test_values:
            # Act
            result = ConfigMapper._transform_value("unmapped_field", value, "any_agent")

            # Assert
            assert result == value


class TestConfigMapperFlatToNestedMapping:
    """Test flat-to-nested configuration mapping."""

    def test_map_flat_to_nested_refiner_comprehensive(self) -> None:
        """Test comprehensive flat-to-nested mapping for refiner."""
        # Arrange
        flat_config = {
            "refinement_level": "detailed",
            "behavioral_mode": "adaptive",
            "output_format": "structured",
            "custom_constraints": ["constraint1", "constraint2"],
            "fallback_mode": "graceful",
            "custom_system_prompt": "Custom prompt",
            "template_variables": {"var1": "value1"},
            "custom_templates": {"template1": "content1"},
            "format_preference": "markdown",
            "include_metadata": True,
            "confidence_threshold": 0.85,
            "timeout_seconds": 30,
            "max_retries": 3,
            "enable_caching": True,
            "prompts": {
                "system_prompt": "Prompt from prompts section",
                "templates": {"template2": "content2"},
            },
        }

        # Act
        result = ConfigMapper.map_flat_to_nested(flat_config, "refiner")

        # Assert
        assert result["refinement_level"] == "detailed"
        assert result["behavioral_mode"] == "adaptive"
        assert result["output_format"] == "structured"

        assert result["behavioral_config"]["custom_constraints"] == [
            "constraint1",
            "constraint2",
        ]
        assert result["behavioral_config"]["fallback_mode"] == "graceful"

        # Custom system prompt from prompts section should override flat field
        assert (
            result["prompt_config"]["custom_system_prompt"]
            == "Prompt from prompts section"
        )
        assert result["prompt_config"]["template_variables"] == {"var1": "value1"}
        assert result["prompt_config"]["custom_templates"] == {"template2": "content2"}

        assert result["output_config"]["format_preference"] == "markdown"
        assert result["output_config"]["include_metadata"] is True
        assert result["output_config"]["confidence_threshold"] == 0.85

        assert result["execution_config"]["timeout_seconds"] == 30
        assert result["execution_config"]["max_retries"] == 3
        assert result["execution_config"]["enable_caching"] is True

    def test_map_flat_to_nested_unknown_fields(self) -> None:
        """Test that unknown fields are preserved in nested structure."""
        # Arrange
        flat_config = {
            "refinement_level": "standard",
            "unknown_field": "unknown_value",
            "another_unknown": {"nested": "data"},
        }

        # Act
        result = ConfigMapper.map_flat_to_nested(flat_config, "refiner")

        # Assert
        assert result["refinement_level"] == "standard"
        assert result["unknown_field"] == "unknown_value"
        assert result["another_unknown"] == {"nested": "data"}

    def test_map_flat_to_nested_special_fields_ignored(self) -> None:
        """Test that special fields are properly ignored."""
        # Arrange
        flat_config = {
            "refinement_level": "standard",
            "agent_type": "refiner",  # Should be ignored
            "prompts": {  # Should be handled specially
                "system_prompt": "Special prompt"
            },
        }

        # Act
        result = ConfigMapper.map_flat_to_nested(flat_config, "refiner")

        # Assert
        assert "agent_type" not in result
        assert result["prompt_config"]["custom_system_prompt"] == "Special prompt"

    def test_map_flat_to_nested_all_agent_types(self) -> None:
        """Test flat-to-nested mapping for all agent types."""
        test_configs = {
            "refiner": {"refinement_level": "standard"},
            "critic": {"analysis_depth": "medium"},
            "historian": {"search_depth": "standard"},
            "synthesis": {"synthesis_strategy": "balanced"},
        }

        for agent_type, flat_config in test_configs.items():
            # Act
            result = ConfigMapper.map_flat_to_nested(flat_config, agent_type)

            # Assert
            assert isinstance(result, dict)
            assert "prompt_config" in result
            assert "behavioral_config" in result
            assert "output_config" in result
            assert "execution_config" in result


class TestConfigMapperBackwardCompatibility:
    """Test backward compatibility of ConfigMapper."""

    def test_validate_and_create_config_delegates_to_model_validate(self) -> None:
        """Test that validate_and_create_config delegates to model_validate_config."""
        # Arrange
        config = {"refinement_level": "standard"}

        with patch.object(ConfigMapper, "model_validate_config") as mock_validate:
            mock_validate.return_value = RefinerConfig()

            # Act
            result = ConfigMapper.validate_and_create_config(config, "refiner")

            # Assert
            mock_validate.assert_called_once_with(config, "refiner")
            assert result is not None

    def test_create_agent_config_still_works(self) -> None:
        """Test that create_agent_config method still works as before."""
        # Arrange
        flat_config = {
            "refinement_level": "detailed",
            "custom_constraints": ["constraint1"],
            "timeout_seconds": 45,
        }

        # Act
        result = ConfigMapper.create_agent_config(flat_config, "refiner")

        # Assert
        assert result is not None
        assert isinstance(result, RefinerConfig)
        assert result.refinement_level == "detailed"
        assert result.behavioral_config.custom_constraints == ["constraint1"]
        assert result.execution_config.timeout_seconds == 45

    def test_create_agent_config_error_handling(self) -> None:
        """Test error handling in create_agent_config."""
        # Test with unsupported agent type
        with pytest.raises(ValueError, match="Unknown agent type: invalid"):
            ConfigMapper.create_agent_config({}, "invalid")

        # Test with invalid configuration that causes Pydantic validation error
        invalid_config = {
            "refinement_level": "invalid_level",  # Not in allowed values
            "confidence_threshold": 1.5,  # Outside valid range
        }

        with pytest.raises(ValueError, match="Failed to create refiner configuration"):
            ConfigMapper.create_agent_config(invalid_config, "refiner")


class TestConfigMapperHelperMethods:
    """Test helper methods of ConfigMapper."""

    def test_get_config_class(self) -> None:
        """Test _get_config_class helper method."""
        # Test valid agent types
        assert ConfigMapper._get_config_class("refiner") == RefinerConfig
        assert ConfigMapper._get_config_class("critic") == CriticConfig
        assert ConfigMapper._get_config_class("historian") == HistorianConfig
        assert ConfigMapper._get_config_class("synthesis") == SynthesisConfig

        # Test invalid agent type
        assert ConfigMapper._get_config_class("invalid") is None

    def test_get_field_mapping(self) -> None:
        """Test _get_field_mapping helper method."""
        # Test valid agent types
        refiner_mapping = ConfigMapper._get_field_mapping("refiner")
        assert isinstance(refiner_mapping, dict)
        assert "refinement_level" in refiner_mapping
        assert "custom_constraints" in refiner_mapping

        critic_mapping = ConfigMapper._get_field_mapping("critic")
        assert isinstance(critic_mapping, dict)
        assert "analysis_depth" in critic_mapping

        # Test invalid agent type
        invalid_mapping = ConfigMapper._get_field_mapping("invalid")
        assert invalid_mapping == {}

    def test_set_nested_value(self) -> None:
        """Test _set_nested_value helper method."""
        # Test direct field setting
        config: Dict[str, Any] = {}
        ConfigMapper._set_nested_value(config, "direct_field", "direct_value")
        assert config["direct_field"] == "direct_value"

        # Test single-level nested field
        config = {}
        ConfigMapper._set_nested_value(config, "section.field", "nested_value")
        assert config["section"]["field"] == "nested_value"

        # Test multi-level nested field
        config = {}
        ConfigMapper._set_nested_value(config, "section.subsection.field", "deep_value")
        assert config["section"]["subsection"]["field"] == "deep_value"

        # Test overwriting existing structure
        config = {"section": {"existing": "value"}}
        ConfigMapper._set_nested_value(config, "section.new_field", "new_value")
        assert config["section"]["existing"] == "value"
        assert config["section"]["new_field"] == "new_value"


class TestConfigMapperEdgeCases:
    """Test edge cases and error conditions."""

    def test_model_validate_config_validation_errors(self) -> None:
        """Test handling of Pydantic validation errors with graceful fallback."""
        # Invalid configuration that will fail both direct and mapped validation
        invalid_config = {
            "refinement_level": "totally_invalid_level",
            "confidence_threshold": 5.0,  # Outside valid range (0.0-1.0)
            "timeout_seconds": -10,  # Negative value not allowed
        }

        with patch("cognivault.config.config_mapper.logger") as mock_logger:
            # Act
            result = ConfigMapper.model_validate_config(invalid_config, "refiner")

            # Assert - Should gracefully fall back to default config instead of None
            assert result is not None
            assert isinstance(result, RefinerConfig)
            # Should fall back to default values due to invalid input
            assert result.refinement_level == "standard"  # Default fallback
            assert result.output_config.confidence_threshold == 0.7  # Default fallback
            assert result.execution_config.timeout_seconds == 30  # Default fallback

            # Should have logged debug messages about validation attempts
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "Direct validation failed for refiner" in str(call)
            ]
            assert len(debug_calls) > 0

    def test_complex_nested_config_structures(self) -> None:
        """Test handling of complex nested configuration structures."""
        # Complex but valid configuration with deep nesting
        complex_config = {
            "refinement_level": "comprehensive",
            "prompt_config": {
                "custom_system_prompt": "Complex prompt with multiple lines\nand special formatting",
                "template_variables": {
                    "var1": "simple_value",
                    "var2": "another_value",
                    "complex_var": "value with spaces and symbols: @#$%",
                },
                "custom_templates": {
                    "template1": "Template content with {{variables}}",
                    "template2": "Another template with different content",
                },
            },
            "behavioral_config": {
                "custom_constraints": [
                    "constraint1",
                    "constraint2",
                    "complex constraint with spaces and symbols",
                    "constraint with unicode: émojis 🚀",
                ],
                "fallback_mode": "adaptive",
            },
            "output_config": {
                "format_preference": "structured",
                "include_metadata": True,
                "confidence_threshold": 0.95,
            },
            "execution_config": {
                "timeout_seconds": 120,
                "max_retries": 5,
                "enable_caching": False,
            },
        }

        # Act
        result = ConfigMapper.model_validate_config(complex_config, "refiner")

        # Assert
        assert result is not None
        assert isinstance(result, RefinerConfig)
        assert result.refinement_level == "comprehensive"
        assert (
            result.prompt_config.custom_system_prompt
            == "Complex prompt with multiple lines\nand special formatting"
        )
        assert len(result.prompt_config.template_variables) == 3
        assert len(result.behavioral_config.custom_constraints) == 4
        assert "🚀" in result.behavioral_config.custom_constraints[3]

    def test_large_configuration_data(self) -> None:
        """Test handling of large configuration data."""
        # Create a large configuration
        large_config = {
            "refinement_level": "detailed",
            "behavioral_config": {
                "custom_constraints": [f"constraint_{i}" for i in range(100)]
            },
            "prompt_config": {
                "template_variables": {f"var_{i}": f"value_{i}" for i in range(50)}
            },
        }

        # Act
        result = ConfigMapper.model_validate_config(large_config, "refiner")

        # Assert
        assert result is not None
        assert len(result.behavioral_config.custom_constraints) == 100
        assert len(result.prompt_config.template_variables) == 50

    def test_unicode_and_special_characters(self) -> None:
        """Test handling of Unicode and special characters in configuration."""
        # Configuration with Unicode and special characters
        unicode_config = {
            "refinement_level": "standard",
            "prompt_config": {
                "custom_system_prompt": "Prompt with émojis 🚀 and ünicode",
                "template_variables": {
                    "unicode_key": "Value with spëcial chars: áéíóú",
                    "emoji_key": "🎉 Celebration time! 🎊",
                },
            },
            "behavioral_config": {
                "custom_constraints": [
                    "Constraint with åccents",
                    "Constraint with 中文",
                    "Constraint with עברית",
                ]
            },
        }

        # Act
        result = ConfigMapper.model_validate_config(unicode_config, "refiner")

        # Assert
        assert result is not None
        prompt_config = result.prompt_config
        assert prompt_config.custom_system_prompt is not None
        assert "🚀" in prompt_config.custom_system_prompt
        assert "spëcial" in prompt_config.template_variables["unicode_key"]
        assert "中文" in result.behavioral_config.custom_constraints[1]
