"""
Configuration mapper for supporting both flat chart format and nested Pydantic format.

This module handles the mapping between:
1. Flat format (chart workflows): custom_constraints: ["term1", "term2"]
2. Nested format (Pydantic): behavioral_config.custom_constraints: ["term1", "term2"]

Modernized to use Pydantic's native validation and transformation capabilities
while maintaining backward compatibility with existing chart workflows.
"""

from typing import Dict, Any, Type, TypeVar, Optional, overload, Literal
from pydantic import BaseModel, ValidationError
import logging

from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
    AgentConfigType,
    get_agent_config_class,
)

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ConfigMapper:
    """
    Maps between flat configuration format (used in chart workflows)
    and nested Pydantic format (used in our configuration classes).
    """

    # Define the mapping from flat fields to nested Pydantic structure
    REFINER_FIELD_MAPPING = {
        # Direct fields (no mapping needed)
        "refinement_level": "refinement_level",
        "behavioral_mode": "behavioral_mode",
        "output_format": "output_format",
        # Fields that map to nested configs
        "custom_constraints": "behavioral_config.custom_constraints",
        "fallback_mode": "behavioral_config.fallback_mode",
        "custom_system_prompt": "prompt_config.custom_system_prompt",
        "template_variables": "prompt_config.template_variables",
        "custom_templates": "prompt_config.custom_templates",
        "format_preference": "output_config.format_preference",
        "include_metadata": "output_config.include_metadata",
        "confidence_threshold": "output_config.confidence_threshold",
        "timeout_seconds": "execution_config.timeout_seconds",
        "max_retries": "execution_config.max_retries",
        "enable_caching": "execution_config.enable_caching",
        # Special fields that need handling
        "strict_mode": "behavioral_config.strict_mode",
        "simulation_delay": "execution_config.simulation_delay",
        "show_refinement_prefix": "output_config.show_refinement_prefix",
        "preserve_unchanged_indicator": "output_config.preserve_unchanged_indicator",
    }

    CRITIC_FIELD_MAPPING = {
        # Direct fields
        "analysis_depth": "analysis_depth",
        "confidence_reporting": "confidence_reporting",
        "bias_detection": "bias_detection",
        "scoring_criteria": "scoring_criteria",
        # Nested fields
        "custom_constraints": "behavioral_config.custom_constraints",
        "fallback_mode": "behavioral_config.fallback_mode",
        "custom_system_prompt": "prompt_config.custom_system_prompt",
        "template_variables": "prompt_config.template_variables",
        "custom_templates": "prompt_config.custom_templates",
        "format_preference": "output_config.format_preference",
        "include_metadata": "output_config.include_metadata",
        "confidence_threshold": "output_config.confidence_threshold",
        "timeout_seconds": "execution_config.timeout_seconds",
        "max_retries": "execution_config.max_retries",
        "enable_caching": "execution_config.enable_caching",
        # Chart-specific fields
        "categories": "behavioral_config.categories",
        "focus_on": "behavioral_config.focus_on",
    }

    HISTORIAN_FIELD_MAPPING = {
        # Direct fields
        "search_depth": "search_depth",
        "relevance_threshold": "relevance_threshold",
        "context_expansion": "context_expansion",
        "memory_scope": "memory_scope",
        # Nested fields
        "custom_constraints": "behavioral_config.custom_constraints",
        "fallback_mode": "behavioral_config.fallback_mode",
        "custom_system_prompt": "prompt_config.custom_system_prompt",
        "template_variables": "prompt_config.template_variables",
        "custom_templates": "prompt_config.custom_templates",
        "format_preference": "output_config.format_preference",
        "include_metadata": "output_config.include_metadata",
        "confidence_threshold": "output_config.confidence_threshold",
        "timeout_seconds": "execution_config.timeout_seconds",
        "max_retries": "execution_config.max_retries",
        "enable_caching": "execution_config.enable_caching",
        # Chart-specific fields and legacy fields from enhanced_prompts_example
        "max_results": "execution_config.max_results",
        "search_type": "behavioral_config.search_type",  # Legacy field
        "analysis_mode": "behavioral_config.analysis_mode",
        "focus_areas": "behavioral_config.focus_areas",
        "time_horizon": "behavioral_config.time_horizon",
    }

    # Value mappings for fields that need transformation
    VALUE_MAPPINGS: Dict[str, Dict[str, Dict[str, Any]]] = {
        "historian": {
            "search_depth": {
                "comprehensive": "deep",  # Map chart value to schema value
                "standard": "standard",
                "shallow": "shallow",
                "exhaustive": "exhaustive",
            },
            "context_expansion": {
                "broad": True,  # Map string to boolean
                "narrow": False,
                "true": True,
                "false": False,
            },
        },
        "synthesis": {
            "synthesis_strategy": {
                "comprehensive_integration": "comprehensive",  # Map long names to schema values
                "balanced": "balanced",
                "focused": "focused",
                "creative": "creative",
            }
        },
    }

    SYNTHESIS_FIELD_MAPPING = {
        # Direct fields
        "synthesis_strategy": "synthesis_strategy",
        "thematic_focus": "thematic_focus",
        "meta_analysis": "meta_analysis",
        "integration_mode": "integration_mode",
        # Nested fields
        "custom_constraints": "behavioral_config.custom_constraints",
        "fallback_mode": "behavioral_config.fallback_mode",
        "custom_system_prompt": "prompt_config.custom_system_prompt",
        "template_variables": "prompt_config.template_variables",
        "custom_templates": "prompt_config.custom_templates",
        "format_preference": "output_config.format_preference",
        "include_metadata": "output_config.include_metadata",
        "confidence_threshold": "output_config.confidence_threshold",
        "timeout_seconds": "execution_config.timeout_seconds",
        "max_retries": "execution_config.max_retries",
        "enable_caching": "execution_config.enable_caching",
        # Chart-specific fields
        "synthesis_mode": "behavioral_config.synthesis_mode",
        "integration_strategy": "behavioral_config.integration_strategy",
        "output_style": "output_config.output_style",
        "include_confidence": "output_config.include_confidence",
        "structure": "output_config.structure",
        "tone": "output_config.tone",
        "min_agent_outputs": "execution_config.min_agent_outputs",
    }

    @classmethod
    def map_flat_to_nested(
        cls, flat_config: Dict[str, Any], agent_type: str
    ) -> Dict[str, Any]:
        """
        Convert flat configuration format to nested Pydantic format.

        Args:
            flat_config: Configuration in flat format (e.g., from chart workflows)
            agent_type: Type of agent ('refiner', 'critic', 'historian', 'synthesis')

        Returns:
            Configuration in nested format suitable for Pydantic validation
        """
        # Get the appropriate field mapping
        field_mapping = cls._get_field_mapping(agent_type)

        # Initialize nested structure
        nested_config: Dict[str, Any] = {
            "prompt_config": {},
            "behavioral_config": {},
            "output_config": {},
            "execution_config": {},
        }

        # Process each field in flat config
        for flat_field, value in flat_config.items():
            # Handle special cases first
            if flat_field in ["prompts", "agent_type"]:
                # Skip these - they're handled separately
                continue

            # Apply value transformation if needed
            transformed_value = cls._transform_value(flat_field, value, agent_type)

            # Check if field has a mapping
            if flat_field in field_mapping:
                nested_path = field_mapping[flat_field]
                cls._set_nested_value(nested_config, nested_path, transformed_value)
            else:
                # Unknown field - add as direct field for forward compatibility
                nested_config[flat_field] = transformed_value

        # Handle special 'prompts' configuration
        if "prompts" in flat_config:
            prompts_config = flat_config["prompts"]
            if "system_prompt" in prompts_config:
                nested_config["prompt_config"]["custom_system_prompt"] = prompts_config[
                    "system_prompt"
                ]
            if "templates" in prompts_config:
                nested_config["prompt_config"]["custom_templates"] = prompts_config[
                    "templates"
                ]

        # Don't clean up empty nested configs - Pydantic expects them
        # Just ensure they're always present for consistent structure
        return nested_config

    @classmethod
    def _get_field_mapping(cls, agent_type: str) -> Dict[str, str]:
        """Get the field mapping for the specified agent type."""
        mappings = {
            "refiner": cls.REFINER_FIELD_MAPPING,
            "critic": cls.CRITIC_FIELD_MAPPING,
            "historian": cls.HISTORIAN_FIELD_MAPPING,
            "synthesis": cls.SYNTHESIS_FIELD_MAPPING,
        }
        return mappings.get(agent_type, {})

    @classmethod
    def _get_config_class(cls, agent_type: str) -> Optional[Type[AgentConfigType]]:
        """Get the appropriate config class for an agent type."""
        config_classes: Dict[str, Type[AgentConfigType]] = {
            "refiner": RefinerConfig,
            "critic": CriticConfig,
            "historian": HistorianConfig,
            "synthesis": SynthesisConfig,
        }
        return config_classes.get(agent_type)

    @classmethod
    def _transform_value(cls, field_name: str, value: Any, agent_type: str) -> Any:
        """Transform field values from chart format to schema format."""
        # Get value mappings for this agent type
        agent_mappings = cls.VALUE_MAPPINGS.get(agent_type, {})
        field_mappings = agent_mappings.get(field_name, {}) if agent_mappings else {}

        # If there's a specific mapping for this value, use it
        if isinstance(value, str) and value in field_mappings:
            return field_mappings[value]

        # If it's a boolean string, convert it
        if isinstance(value, str) and value.lower() in ["true", "false"]:
            return value.lower() == "true"

        # Return original value if no transformation needed
        return value

    @classmethod
    def _set_nested_value(cls, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a value in nested configuration using dot notation path."""
        if "." not in path:
            # Direct field
            config[path] = value
        else:
            # Nested field
            parts = path.split(".", 1)
            section = parts[0]
            remaining_path = parts[1]

            if section not in config:
                config[section] = {}

            if "." in remaining_path:
                cls._set_nested_value(config[section], remaining_path, value)
            else:
                config[section][remaining_path] = value

    @classmethod
    def create_agent_config(
        cls, flat_config: Dict[str, Any], agent_type: str
    ) -> AgentConfigType:
        """
        Create a Pydantic agent configuration from flat configuration format.

        Args:
            flat_config: Configuration in flat format
            agent_type: Type of agent ('refiner', 'critic', 'historian', 'synthesis')

        Returns:
            Appropriate Pydantic configuration object

        Raises:
            ValueError: If agent_type is not supported
        """
        # Map flat config to nested format
        nested_config = cls.map_flat_to_nested(flat_config, agent_type)

        # Get appropriate config class using existing function
        config_class = get_agent_config_class(agent_type)

        try:
            return config_class(**nested_config)
        except Exception as e:
            # Provide helpful error message with both formats
            raise ValueError(
                f"Failed to create {agent_type} configuration.\n"
                f"Flat config: {flat_config}\n"
                f"Mapped config: {nested_config}\n"
                f"Error: {e}"
            ) from e

    # Overloads for better type safety
    @overload
    @classmethod
    def model_validate_config(
        cls, config_data: Optional[Dict[str, Any]], agent_type: Literal["refiner"]
    ) -> Optional[RefinerConfig]: ...

    @overload
    @classmethod
    def model_validate_config(
        cls, config_data: Optional[Dict[str, Any]], agent_type: Literal["critic"]
    ) -> Optional[CriticConfig]: ...

    @overload
    @classmethod
    def model_validate_config(
        cls, config_data: Optional[Dict[str, Any]], agent_type: Literal["historian"]
    ) -> Optional[HistorianConfig]: ...

    @overload
    @classmethod
    def model_validate_config(
        cls, config_data: Optional[Dict[str, Any]], agent_type: Literal["synthesis"]
    ) -> Optional[SynthesisConfig]: ...

    @overload
    @classmethod
    def model_validate_config(
        cls, config_data: Optional[Dict[str, Any]], agent_type: str
    ) -> Optional[AgentConfigType]: ...

    @classmethod
    def model_validate_config(
        cls, config_data: Optional[Dict[str, Any]], agent_type: str
    ) -> Optional[AgentConfigType]:
        """
        Validate configuration data using Pydantic's model_validate().

        This is the modernized approach that leverages Pydantic's native validation
        and error handling capabilities.

        Args:
            config_data: Configuration data in either format
            agent_type: Type of agent

        Returns:
            Pydantic configuration object or None if validation fails
        """
        if not config_data:
            logger.debug(f"No config data provided for {agent_type}")
            return None

        config_class = cls._get_config_class(agent_type)
        if not config_class:
            logger.warning(f"Unknown agent type: {agent_type}")
            return None

        # Try direct Pydantic validation first (nested format)
        try:
            return config_class.model_validate(config_data)
        except ValidationError as e:
            logger.debug(f"Direct validation failed for {agent_type}: {e}")
            # Fall back to flat format mapping
            try:
                nested_config = cls.map_flat_to_nested(config_data, agent_type)
                return config_class.model_validate(nested_config)
            except ValidationError as nested_e:
                logger.debug(f"Mapped validation failed for {agent_type}: {nested_e}")
                # Final fallback: try to create config with only known fields
                try:
                    return cls._create_config_with_known_fields(config_data, agent_type)
                except Exception as fallback_e:
                    logger.warning(
                        f"All validation attempts failed for {agent_type}: "
                        f"Direct: {e}, Mapped: {nested_e}, Fallback: {fallback_e}"
                    )
                    return None

    # Overloads for validate_and_create_config for better type safety
    @overload
    @classmethod
    def validate_and_create_config(
        cls, config_data: Dict[str, Any], agent_type: Literal["refiner"]
    ) -> Optional[RefinerConfig]: ...

    @overload
    @classmethod
    def validate_and_create_config(
        cls, config_data: Dict[str, Any], agent_type: Literal["critic"]
    ) -> Optional[CriticConfig]: ...

    @overload
    @classmethod
    def validate_and_create_config(
        cls, config_data: Dict[str, Any], agent_type: Literal["historian"]
    ) -> Optional[HistorianConfig]: ...

    @overload
    @classmethod
    def validate_and_create_config(
        cls, config_data: Dict[str, Any], agent_type: Literal["synthesis"]
    ) -> Optional[SynthesisConfig]: ...

    @overload
    @classmethod
    def validate_and_create_config(
        cls, config_data: Dict[str, Any], agent_type: str
    ) -> Optional[AgentConfigType]: ...

    @classmethod
    def validate_and_create_config(
        cls, config_data: Dict[str, Any], agent_type: str
    ) -> Optional[AgentConfigType]:
        """
        Validate configuration data and create appropriate config object.

        Maintained for backward compatibility. New code should use model_validate_config().

        Args:
            config_data: Configuration data in either format
            agent_type: Type of agent

        Returns:
            Pydantic configuration object or None if creation fails
        """
        # Delegate to the modernized method
        return cls.model_validate_config(config_data, agent_type)

    @classmethod
    def _create_config_with_known_fields(
        cls, config_data: Dict[str, Any], agent_type: str
    ) -> Optional[AgentConfigType]:
        """Create config using only fields that are known to the Pydantic schema."""

        config_class = get_agent_config_class(agent_type)

        # Get field mapping for this agent type
        field_mapping = cls._get_field_mapping(agent_type)

        # Initialize nested structure with defaults
        nested_config: Dict[str, Any] = {
            "prompt_config": {},
            "behavioral_config": {},
            "output_config": {},
            "execution_config": {},
        }

        # Only process fields that we know how to map
        for flat_field, value in config_data.items():
            if flat_field in ["prompts", "agent_type"]:
                continue

            if flat_field in field_mapping:
                transformed_value = cls._transform_value(flat_field, value, agent_type)
                nested_path = field_mapping[flat_field]
                cls._set_nested_value(nested_config, nested_path, transformed_value)

        # Handle prompts section
        if "prompts" in config_data:
            prompts_config = config_data["prompts"]
            if "system_prompt" in prompts_config:
                nested_config["prompt_config"]["custom_system_prompt"] = prompts_config[
                    "system_prompt"
                ]
            if "templates" in prompts_config:
                nested_config["prompt_config"]["custom_templates"] = prompts_config[
                    "templates"
                ]

        # Try to create config with only the mapped fields
        try:
            return config_class(**nested_config)
        except Exception:
            # If that fails, return a default config
            return config_class()
