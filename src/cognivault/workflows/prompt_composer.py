"""
Prompt Composer System for Dynamic Agent Behavior Configuration

This module provides dynamic prompt composition capabilities, allowing agents to
modify their behavior based on YAML configuration without code changes. The composer
integrates with the existing prompt system while enabling runtime customization.

Architecture:
- Template-based prompt composition with variable substitution
- Integration with existing prompt_loader.py system
- Configuration-driven behavioral modifications
- Fallback to default prompts for backward compatibility
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable, TypedDict, get_type_hints
from dataclasses import dataclass

from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
    AgentConfigType,
)

# Import existing prompts for fallback behavior
try:
    from cognivault.agents.refiner.prompts import REFINER_SYSTEM_PROMPT
    from cognivault.agents.critic.prompts import CRITIC_SYSTEM_PROMPT
    from cognivault.agents.historian.prompts import HISTORIAN_SYSTEM_PROMPT
    from cognivault.agents.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT
except ImportError:
    # Fallback if prompts not available
    REFINER_SYSTEM_PROMPT = "You are a query refinement assistant."
    CRITIC_SYSTEM_PROMPT = "You are a critical analysis assistant."
    HISTORIAN_SYSTEM_PROMPT = "You are a context retrieval assistant."
    SYNTHESIS_SYSTEM_PROMPT = "You are a synthesis assistant."

logger = logging.getLogger(__name__)


# Strict typing for template variables - keeping Python on a leash!
class TemplateVariables(TypedDict, total=False):
    """
    Strictly typed template variables for prompt composition.

    Uses TypedDict with total=False to allow partial dictionaries while maintaining
    strict typing for each key that is present. This ensures type safety while
    allowing different agents to use different subsets of variables.
    """

    # Common variables across all agents
    custom_user_variables: str  # From user's custom template_variables
    style: str  # Common template variable
    domain: str  # Common template variable

    # Refiner agent variables
    refinement_level: str
    behavioral_mode: str
    output_format: str

    # Critic agent variables
    analysis_depth: str
    confidence_reporting: str
    bias_detection: str
    scoring_criteria: List[str]

    # Historian agent variables
    search_depth: str
    relevance_threshold: str
    context_expansion: str
    memory_scope: str

    # Synthesis agent variables
    synthesis_strategy: str
    thematic_focus: str
    meta_analysis: str
    integration_mode: str


@dataclass
class ComposedPrompt:
    """Container for a composed prompt with metadata."""

    system_prompt: str
    templates: Dict[str, str]
    variables: TemplateVariables
    metadata: Dict[str, Any]

    def get_template(self, template_name: str) -> Optional[str]:
        """Get a specific template by name."""
        return self.templates.get(template_name)

    def substitute_variables(self, template: str) -> str:
        """Substitute variables in a template string."""
        try:
            return template.format(**self.variables)
        except KeyError as e:
            logger.warning(f"Template variable {e} not found, using template as-is")
            return template


class PromptComposer:
    """
    Dynamic prompt composition system for configurable agent behavior.

    The PromptComposer enables runtime modification of agent prompts based on
    configuration while maintaining backward compatibility with existing systems.
    """

    def __init__(self):
        """Initialize the prompt composer."""
        self.default_prompts = {
            "refiner": REFINER_SYSTEM_PROMPT,
            "critic": CRITIC_SYSTEM_PROMPT,
            "historian": HISTORIAN_SYSTEM_PROMPT,
            "synthesis": SYNTHESIS_SYSTEM_PROMPT,
        }

        # Template patterns for behavioral modifications
        self.behavioral_templates = {
            "refinement_level": {
                "minimal": "Focus on essential clarifications only.",
                "standard": "Provide balanced refinement with key improvements.",
                "detailed": "Thoroughly analyze and enhance the query with comprehensive improvements.",
                "comprehensive": "Perform exhaustive refinement with detailed analysis and multiple perspectives.",
            },
            "analysis_depth": {
                "shallow": "Provide surface-level analysis focusing on obvious patterns.",
                "medium": "Conduct moderate analysis with attention to key insights and patterns.",
                "deep": "Perform thorough analysis with detailed examination of underlying factors.",
                "comprehensive": "Execute exhaustive analysis with multi-layered examination and nuanced insights.",
            },
            "behavioral_mode": {
                "active": "Take an proactive, engagement-focused approach with direct recommendations.",
                "passive": "Provide neutral, observation-focused analysis without strong directives.",
                "adaptive": "Adjust approach based on context complexity and user needs.",
            },
            "synthesis_strategy": {
                "comprehensive": "Integrate all available perspectives into a complete unified analysis.",
                "focused": "Synthesize around specific themes and core insights.",
                "balanced": "Provide proportional integration of different viewpoints.",
                "creative": "Explore innovative connections and novel synthesis approaches.",
            },
        }

    def _create_template_variables(
        self, custom_variables: Dict[str, str]
    ) -> TemplateVariables:
        """
        Create a properly typed TemplateVariables dict from custom user variables.

        Only includes variables that match our TypedDict schema to maintain type safety.
        """
        # Get the valid keys from TemplateVariables TypedDict
        valid_keys = get_type_hints(TemplateVariables).keys()

        # Filter custom variables to only include valid TypedDict keys
        filtered_variables = {
            key: value for key, value in custom_variables.items() if key in valid_keys
        }

        # Cast to TemplateVariables - this is safe because we've filtered the keys
        return filtered_variables  # type: ignore

    def compose_refiner_prompt(self, config: RefinerConfig) -> ComposedPrompt:
        """
        Compose refiner agent prompt based on configuration.

        Args:
            config: RefinerConfig with behavioral and prompt settings

        Returns:
            ComposedPrompt with customized system prompt and templates
        """
        # Start with base prompt or custom override
        if config.prompt_config.custom_system_prompt:
            base_prompt = config.prompt_config.custom_system_prompt
        else:
            base_prompt = self.default_prompts["refiner"]

        # Apply behavioral modifications
        modifications = []

        # Add refinement level guidance
        if refinement_guide := self.behavioral_templates["refinement_level"].get(
            config.refinement_level
        ):
            modifications.append(f"Refinement Level: {refinement_guide}")

        # Add behavioral mode guidance
        if behavior_guide := self.behavioral_templates["behavioral_mode"].get(
            config.behavioral_mode
        ):
            modifications.append(f"Behavioral Mode: {behavior_guide}")

        # Add output format instructions
        output_instructions = self._get_output_format_instructions(
            config.output_format, config.output_config.format_preference
        )
        if output_instructions:
            modifications.append(f"Output Format: {output_instructions}")

        # Add custom constraints
        if config.behavioral_config.custom_constraints:
            constraints_text = "Additional Constraints: " + "; ".join(
                config.behavioral_config.custom_constraints
            )
            modifications.append(constraints_text)

        # Compose final prompt
        if modifications:
            modification_text = "\n\n" + "\n".join(f"- {mod}" for mod in modifications)
            composed_prompt = base_prompt + modification_text
        else:
            composed_prompt = base_prompt

        # Prepare templates and variables
        templates = dict(config.prompt_config.custom_templates)

        # Combine template variables from prompt config with configuration-derived variables
        all_template_vars = dict(config.prompt_config.template_variables)
        variables = self._create_template_variables(all_template_vars)

        # Add configuration-derived variables
        variables.update(
            {
                "refinement_level": config.refinement_level,
                "behavioral_mode": config.behavioral_mode,
                "output_format": config.output_format,
            }
        )

        metadata = {
            "agent_type": "refiner",
            "config_version": "1.0",
            "composition_timestamp": "runtime",
            "fallback_mode": config.behavioral_config.fallback_mode,
        }

        return ComposedPrompt(
            system_prompt=composed_prompt,
            templates=templates,
            variables=variables,
            metadata=metadata,
        )

    def compose_critic_prompt(self, config: CriticConfig) -> ComposedPrompt:
        """
        Compose critic agent prompt based on configuration.

        Args:
            config: CriticConfig with analysis and evaluation settings

        Returns:
            ComposedPrompt with customized critical analysis prompt
        """
        # Start with base prompt or custom override
        if config.prompt_config.custom_system_prompt:
            base_prompt = config.prompt_config.custom_system_prompt
        else:
            base_prompt = self.default_prompts["critic"]

        # Apply behavioral modifications
        modifications = []

        # Add analysis depth guidance
        if analysis_guide := self.behavioral_templates["analysis_depth"].get(
            config.analysis_depth
        ):
            modifications.append(f"Analysis Depth: {analysis_guide}")

        # Add confidence reporting instructions
        if config.confidence_reporting:
            modifications.append(
                "Confidence Reporting: Include confidence scores (0.0-1.0) for key assessments."
            )

        # Add bias detection instructions
        if config.bias_detection:
            modifications.append(
                "Bias Detection: Actively identify and report potential biases or assumptions."
            )

        # Add scoring criteria
        if config.scoring_criteria:
            criteria_text = (
                "Evaluation Criteria: Assess content based on: "
                + ", ".join(config.scoring_criteria)
            )
            modifications.append(criteria_text)

        # Add custom constraints
        if config.behavioral_config.custom_constraints:
            constraints_text = "Additional Guidelines: " + "; ".join(
                config.behavioral_config.custom_constraints
            )
            modifications.append(constraints_text)

        # Compose final prompt
        if modifications:
            modification_text = "\n\n" + "\n".join(f"- {mod}" for mod in modifications)
            composed_prompt = base_prompt + modification_text
        else:
            composed_prompt = base_prompt

        # Prepare templates and variables
        templates = dict(config.prompt_config.custom_templates)

        # Combine template variables from prompt config with configuration-derived variables
        all_template_vars = dict(config.prompt_config.template_variables)
        variables = self._create_template_variables(all_template_vars)

        # Add configuration-derived variables
        variables.update(
            {
                "analysis_depth": config.analysis_depth,
                "confidence_reporting": str(config.confidence_reporting),
                "bias_detection": str(config.bias_detection),
                "scoring_criteria": config.scoring_criteria,
            }
        )

        metadata = {
            "agent_type": "critic",
            "config_version": "1.0",
            "composition_timestamp": "runtime",
            "analysis_features": {
                "confidence_reporting": config.confidence_reporting,
                "bias_detection": config.bias_detection,
            },
        }

        return ComposedPrompt(
            system_prompt=composed_prompt,
            templates=templates,
            variables=variables,
            metadata=metadata,
        )

    def compose_historian_prompt(self, config: HistorianConfig) -> ComposedPrompt:
        """
        Compose historian agent prompt based on configuration.

        Args:
            config: HistorianConfig with search and context settings

        Returns:
            ComposedPrompt with customized context retrieval prompt
        """
        # Start with base prompt or custom override
        if config.prompt_config.custom_system_prompt:
            base_prompt = config.prompt_config.custom_system_prompt
        else:
            base_prompt = self.default_prompts["historian"]

        # Apply behavioral modifications
        modifications = []

        # Add search depth guidance
        search_instructions = {
            "shallow": "Focus on immediate, directly relevant context.",
            "standard": "Provide balanced context retrieval with key historical information.",
            "deep": "Conduct thorough context search including related background information.",
            "exhaustive": "Perform comprehensive context retrieval with extensive historical analysis.",
        }
        if search_guide := search_instructions.get(config.search_depth):
            modifications.append(f"Search Depth: {search_guide}")

        # Add relevance threshold guidance
        modifications.append(
            f"Relevance Threshold: Focus on context with relevance score ≥ {config.relevance_threshold}"
        )

        # Add context expansion instructions
        if config.context_expansion:
            modifications.append(
                "Context Expansion: Include related information that provides broader understanding."
            )

        # Add memory scope guidance
        scope_instructions = {
            "session": "Focus on current conversation context only.",
            "recent": "Include recent historical context and patterns.",
            "full": "Consider complete available historical context.",
            "selective": "Intelligently select most relevant historical context based on query.",
        }
        if scope_guide := scope_instructions.get(config.memory_scope):
            modifications.append(f"Memory Scope: {scope_guide}")

        # Add custom constraints
        if config.behavioral_config.custom_constraints:
            constraints_text = "Search Guidelines: " + "; ".join(
                config.behavioral_config.custom_constraints
            )
            modifications.append(constraints_text)

        # Compose final prompt
        if modifications:
            modification_text = "\n\n" + "\n".join(f"- {mod}" for mod in modifications)
            composed_prompt = base_prompt + modification_text
        else:
            composed_prompt = base_prompt

        # Prepare templates and variables
        templates = dict(config.prompt_config.custom_templates)

        # Combine template variables from prompt config with configuration-derived variables
        all_template_vars = dict(config.prompt_config.template_variables)
        variables = self._create_template_variables(all_template_vars)

        # Add configuration-derived variables
        variables.update(
            {
                "search_depth": config.search_depth,
                "relevance_threshold": str(config.relevance_threshold),
                "context_expansion": str(config.context_expansion),
                "memory_scope": config.memory_scope,
            }
        )

        metadata = {
            "agent_type": "historian",
            "config_version": "1.0",
            "composition_timestamp": "runtime",
            "search_parameters": {
                "depth": config.search_depth,
                "threshold": config.relevance_threshold,
                "expansion": config.context_expansion,
                "scope": config.memory_scope,
            },
        }

        return ComposedPrompt(
            system_prompt=composed_prompt,
            templates=templates,
            variables=variables,
            metadata=metadata,
        )

    def compose_synthesis_prompt(self, config: SynthesisConfig) -> ComposedPrompt:
        """
        Compose synthesis agent prompt based on configuration.

        Args:
            config: SynthesisConfig with integration and analysis settings

        Returns:
            ComposedPrompt with customized synthesis prompt
        """
        # Start with base prompt or custom override
        if config.prompt_config.custom_system_prompt:
            base_prompt = config.prompt_config.custom_system_prompt
        else:
            base_prompt = self.default_prompts["synthesis"]

        # Apply behavioral modifications
        modifications = []

        # Add synthesis strategy guidance
        if strategy_guide := self.behavioral_templates["synthesis_strategy"].get(
            config.synthesis_strategy
        ):
            modifications.append(f"Synthesis Strategy: {strategy_guide}")

        # Add thematic focus if specified
        if config.thematic_focus:
            modifications.append(
                f"Thematic Focus: Pay special attention to aspects related to '{config.thematic_focus}'."
            )

        # Add meta-analysis instructions
        if config.meta_analysis:
            modifications.append(
                "Meta-Analysis: Include analysis of the synthesis process and integration patterns."
            )

        # Add integration mode guidance
        integration_instructions = {
            "sequential": "Process and integrate agent outputs in logical sequence.",
            "parallel": "Consider all agent outputs simultaneously for synthesis.",
            "hierarchical": "Organize synthesis with clear hierarchical structure and priorities.",
            "adaptive": "Use the most appropriate integration approach based on content complexity.",
        }
        if integration_guide := integration_instructions.get(config.integration_mode):
            modifications.append(f"Integration Mode: {integration_guide}")

        # Add custom constraints
        if config.behavioral_config.custom_constraints:
            constraints_text = "Synthesis Guidelines: " + "; ".join(
                config.behavioral_config.custom_constraints
            )
            modifications.append(constraints_text)

        # Compose final prompt
        if modifications:
            modification_text = "\n\n" + "\n".join(f"- {mod}" for mod in modifications)
            composed_prompt = base_prompt + modification_text
        else:
            composed_prompt = base_prompt

        # Prepare templates and variables
        templates = dict(config.prompt_config.custom_templates)

        # Combine template variables from prompt config with configuration-derived variables
        all_template_vars = dict(config.prompt_config.template_variables)
        variables = self._create_template_variables(all_template_vars)

        # Add configuration-derived variables
        variables.update(
            {
                "synthesis_strategy": config.synthesis_strategy,
                "thematic_focus": config.thematic_focus or "",
                "meta_analysis": str(config.meta_analysis),
                "integration_mode": config.integration_mode,
            }
        )

        metadata = {
            "agent_type": "synthesis",
            "config_version": "1.0",
            "composition_timestamp": "runtime",
            "synthesis_features": {
                "strategy": config.synthesis_strategy,
                "thematic_focus": config.thematic_focus,
                "meta_analysis": config.meta_analysis,
                "integration_mode": config.integration_mode,
            },
        }

        return ComposedPrompt(
            system_prompt=composed_prompt,
            templates=templates,
            variables=variables,
            metadata=metadata,
        )

    def compose_prompt(
        self, agent_type: str, config: AgentConfigType
    ) -> ComposedPrompt:
        """
        Universal prompt composition method for any agent type.

        Args:
            agent_type: Type of agent ("refiner", "critic", "historian", "synthesis")
            config: Agent-specific configuration object

        Returns:
            ComposedPrompt with customized prompt for the agent

        Raises:
            ValueError: If agent_type is not supported
        """
        composer_methods: Dict[str, Callable[[Any], ComposedPrompt]] = {
            "refiner": self.compose_refiner_prompt,
            "critic": self.compose_critic_prompt,
            "historian": self.compose_historian_prompt,
            "synthesis": self.compose_synthesis_prompt,
        }

        if agent_type not in composer_methods:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        composer_method = composer_methods[agent_type]
        return composer_method(config)

    def _get_output_format_instructions(
        self, agent_format: str, general_format: str
    ) -> str:
        """Get output format instructions based on configuration."""
        format_instructions = {
            "raw": "Provide direct, unformatted output without additional structure.",
            "prefixed": "Prefix output with clear agent identification and section headers.",
            "structured": "Use clear structure with headers, bullet points, and logical organization.",
            "markdown": "Format output using markdown syntax for enhanced readability.",
        }

        # Agent-specific format takes precedence
        primary_format = agent_format if agent_format != "adaptive" else general_format
        return format_instructions.get(primary_format, "")

    def get_default_prompt(self, agent_type: str) -> str:
        """Get the default prompt for an agent type (fallback behavior)."""
        return self.default_prompts.get(agent_type, "You are a helpful assistant.")

    def validate_composition(self, composed_prompt: ComposedPrompt) -> bool:
        """
        Validate that a composed prompt is well-formed.

        Args:
            composed_prompt: ComposedPrompt to validate

        Returns:
            True if prompt is valid, False otherwise
        """
        try:
            # Check required fields
            if not composed_prompt.system_prompt:
                return False

            # Check that templates can be populated with variables
            for template_name, template in composed_prompt.templates.items():
                try:
                    template.format(**composed_prompt.variables)
                except (KeyError, ValueError):
                    logger.warning(
                        f"Template {template_name} has invalid variable references"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Prompt validation failed: {e}")
            return False
