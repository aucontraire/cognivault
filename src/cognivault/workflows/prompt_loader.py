"""
Prompt loading utilities for workflow execution.

This module provides functions to load prompts from agent prompts.py files
and apply custom prompt configurations from workflow definitions.
"""

import importlib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_agent_prompts(agent_type: str) -> Dict[str, str]:
    """
    Load prompts from an agent's prompts.py file.

    Parameters
    ----------
    agent_type : str
        The type of agent (e.g., 'refiner', 'critic', 'historian', 'synthesis')

    Returns
    -------
    Dict[str, str]
        Dictionary containing available prompts for the agent type
    """
    prompts = {}

    try:
        # Dynamic import of the agent's prompts module
        module_name = f"cognivault.agents.{agent_type}.prompts"
        prompts_module = importlib.import_module(module_name)

        # Extract all prompt variables from the module
        for attr_name in dir(prompts_module):
            if not attr_name.startswith("_"):  # Skip private attributes
                attr_value = getattr(prompts_module, attr_name)
                if isinstance(attr_value, str):
                    prompts[attr_name] = attr_value

        logger.debug(f"Loaded {len(prompts)} prompts for {agent_type} agent")

    except ImportError:
        logger.warning(f"No prompts.py file found for {agent_type} agent")
    except Exception as e:
        logger.error(f"Error loading prompts for {agent_type} agent: {e}")

    return prompts


def get_system_prompt(
    agent_type: str, custom_prompt: Optional[str] = None
) -> Optional[str]:
    """
    Get the system prompt for an agent, with optional custom override.

    Parameters
    ----------
    agent_type : str
        The type of agent (e.g., 'refiner', 'critic', 'historian', 'synthesis')
    custom_prompt : Optional[str]
        Custom prompt to override the default system prompt

    Returns
    -------
    Optional[str]
        The system prompt to use, or None if no prompt is available
    """
    if custom_prompt:
        logger.debug(f"Using custom system prompt for {agent_type} agent")
        return custom_prompt

    # Load prompts from the agent's prompts.py file
    prompts = load_agent_prompts(agent_type)

    # Look for the standard system prompt
    system_prompt_key = f"{agent_type.upper()}_SYSTEM_PROMPT"
    if system_prompt_key in prompts:
        logger.debug(f"Using default system prompt for {agent_type} agent")
        return prompts[system_prompt_key]

    # Fallback to generic system prompt if available
    if "SYSTEM_PROMPT" in prompts:
        logger.debug(f"Using fallback system prompt for {agent_type} agent")
        return prompts["SYSTEM_PROMPT"]

    logger.warning(f"No system prompt found for {agent_type} agent")
    return None


def get_template_prompt(
    agent_type: str, template_name: str, **kwargs: Any
) -> Optional[str]:
    """
    Get and format a template prompt for an agent.

    Parameters
    ----------
    agent_type : str
        The type of agent (e.g., 'refiner', 'critic', 'historian', 'synthesis')
    template_name : str
        The name of the template prompt to retrieve
    **kwargs
        Variables to substitute in the template

    Returns
    -------
    Optional[str]
        The formatted template prompt, or None if not found
    """
    prompts = load_agent_prompts(agent_type)

    # Look for the template prompt
    template_key = f"{agent_type.upper()}_{template_name.upper()}_TEMPLATE"
    if template_key in prompts:
        try:
            template = prompts[template_key]
            formatted_prompt = template.format(**kwargs)
            logger.debug(
                f"Formatted template prompt {template_name} for {agent_type} agent"
            )
            return formatted_prompt
        except KeyError as e:
            logger.error(
                f"Missing template variable {e} for {template_name} in {agent_type} agent"
            )
            return None

    logger.warning(f"Template prompt {template_name} not found for {agent_type} agent")
    return None


def apply_rich_configuration_to_prompts(agent_type: str, config: Dict[str, Any]) -> str:
    """
    Apply rich configuration options to generate customized system prompts.

    Maps configuration options like refinement_level, search_depth, etc. to
    prompt modifications that control agent behavior.

    Parameters
    ----------
    agent_type : str
        The type of agent (refiner, historian, critic, synthesis)
    config : Dict[str, Any]
        Rich configuration dictionary with agent-specific options

    Returns
    -------
    str
        Customized system prompt based on configuration
    """
    # Get base system prompt
    base_prompt = get_system_prompt(agent_type)
    if not base_prompt:
        return ""

    # Apply agent-specific configuration modifications
    if agent_type == "refiner":
        return _apply_refiner_config(base_prompt, config)
    elif agent_type == "historian":
        return _apply_historian_config(base_prompt, config)
    elif agent_type == "critic":
        return _apply_critic_config(base_prompt, config)
    elif agent_type == "synthesis":
        return _apply_synthesis_config(base_prompt, config)
    else:
        return base_prompt


def _apply_refiner_config(base_prompt: str, config: Dict[str, Any]) -> str:
    """Apply refiner-specific configuration to prompt."""
    prompt = base_prompt

    # Apply refinement level
    refinement_level = config.get("refinement_level", "standard")
    if refinement_level == "detailed":
        prompt += "\n\nMODE: Use ACTIVE MODE. Provide comprehensive refinements with detailed explanations."
    elif refinement_level == "minimal":
        prompt += "\n\nMODE: Prefer PASSIVE MODE. Only refine if critically necessary."
    elif refinement_level == "comprehensive":
        prompt += (
            "\n\nMODE: Use ACTIVE MODE with maximum thoroughness. Examine every aspect."
        )

    # Apply behavioral mode
    behavioral_mode = config.get("behavioral_mode", "adaptive")
    if behavioral_mode == "active":
        prompt += "\n\nBEHAVIOR: Always actively refine and improve queries."
    elif behavioral_mode == "passive":
        prompt += (
            "\n\nBEHAVIOR: Only refine when significant improvements are possible."
        )

    # Apply output format
    output_format = config.get("output_format", "prefixed")
    if output_format == "structured":
        prompt += "\n\nOUTPUT: Use structured bullet points for complex refinements."
    elif output_format == "raw":
        prompt += "\n\nOUTPUT: Return refined query without prefixes or formatting."

    # Apply custom constraints
    custom_constraints = config.get("custom_constraints", [])
    if custom_constraints:
        prompt += f"\n\nADDITIONAL CONSTRAINTS:\n" + "\n".join(
            f"- {c}" for c in custom_constraints
        )

    return prompt


def _apply_historian_config(base_prompt: str, config: Dict[str, Any]) -> str:
    """Apply historian-specific configuration to prompt."""
    prompt = base_prompt

    # Apply search depth
    search_depth = config.get("search_depth", "comprehensive")
    if search_depth == "exhaustive":
        prompt += (
            "\n\nSEARCH MODE: Exhaustive - examine all available sources thoroughly."
        )
    elif search_depth == "basic":
        prompt += "\n\nSEARCH MODE: Basic - focus on most relevant and recent sources."

    # Apply analysis mode
    analysis_mode = config.get("analysis_mode", "contextual")
    if analysis_mode == "analytical":
        prompt += "\n\nANALYSIS: Use analytical approach with deep pattern recognition."
    elif analysis_mode == "factual":
        prompt += "\n\nANALYSIS: Focus on factual content and documented evidence."

    # Apply focus areas
    focus_areas = config.get("focus_areas", [])
    if focus_areas:
        prompt += f"\n\nFOCUS AREAS:\n" + "\n".join(f"- {area}" for area in focus_areas)

    return prompt


def _apply_critic_config(base_prompt: str, config: Dict[str, Any]) -> str:
    """Apply critic-specific configuration to prompt."""
    prompt = base_prompt

    # Apply analysis depth
    analysis_depth = config.get("analysis_depth", "medium")
    if analysis_depth == "deep":
        prompt += "\n\nANALYSIS DEPTH: Deep - perform comprehensive analysis with detailed examination."
    elif analysis_depth == "shallow":
        prompt += (
            "\n\nANALYSIS DEPTH: Shallow - focus on obvious issues and quick insights."
        )

    # Apply bias detection
    bias_detection = config.get("bias_detection", True)
    if not bias_detection:
        prompt += "\n\nBIAS DETECTION: Disabled - focus on other analytical aspects."

    # Apply categories focus
    categories = config.get("categories", [])
    if categories:
        prompt += f"\n\nFOCUS CATEGORIES:\n" + "\n".join(
            f"- {cat}" for cat in categories
        )

    return prompt


def _apply_synthesis_config(base_prompt: str, config: Dict[str, Any]) -> str:
    """Apply synthesis-specific configuration to prompt."""
    prompt = base_prompt

    # Apply synthesis mode
    synthesis_mode = config.get("synthesis_mode", "comprehensive")
    if synthesis_mode == "basic":
        prompt += (
            "\n\nSYNTHESIS MODE: Basic - provide concise integration of key points."
        )
    elif synthesis_mode == "comprehensive":
        prompt += "\n\nSYNTHESIS MODE: Comprehensive - detailed integration with full analysis."

    # Apply output style
    output_style = config.get("output_style", "academic")
    if output_style == "executive":
        prompt += (
            "\n\nOUTPUT STYLE: Executive - clear, action-oriented, business-focused."
        )
    elif output_style == "academic":
        prompt += "\n\nOUTPUT STYLE: Academic - scholarly, detailed, evidence-based."
    elif output_style == "technical":
        prompt += (
            "\n\nOUTPUT STYLE: Technical - precise, factual, implementation-focused."
        )
    elif output_style == "legal":
        prompt += "\n\nOUTPUT STYLE: Legal - precise, risk-aware, compliance-focused."

    # Apply integration strategy
    integration_strategy = config.get("integration_strategy", "thematic")
    if integration_strategy == "weighted":
        prompt += "\n\nINTEGRATION: Use weighted approach - prioritize higher-confidence insights."
    elif integration_strategy == "sequential":
        prompt += (
            "\n\nINTEGRATION: Use sequential approach - integrate in order of analysis."
        )

    # Apply structure requirements
    structure = config.get("structure", [])
    if structure:
        prompt += f"\n\nREQUIRED STRUCTURE:\n" + "\n".join(
            f"- {section}" for section in structure
        )

    return prompt


def apply_prompt_configuration(
    agent_type: str, config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Apply prompt configuration from workflow node configuration.

    Supports both simple custom prompts and rich configuration options.

    Parameters
    ----------
    agent_type : str
        The type of agent
    config : Dict[str, Any]
        Configuration dictionary that may contain prompt overrides

    Returns
    -------
    Dict[str, str]
        Dictionary containing the configured prompts
    """
    configured_prompts = {}

    # Load default prompts
    default_prompts = load_agent_prompts(agent_type)

    # Check for custom prompts first (Phase 1a simple format)
    custom_prompts = config.get("prompts", {})
    if custom_prompts:
        # Apply custom system prompt if specified
        custom_system_prompt = custom_prompts.get("system_prompt")
        system_prompt = get_system_prompt(agent_type, custom_system_prompt)
        if system_prompt:
            configured_prompts["system_prompt"] = system_prompt

        # Apply custom template prompts if specified
        custom_templates = custom_prompts.get("templates", {})
        for template_name, template_content in custom_templates.items():
            if template_content:
                configured_prompts[f"{template_name}_template"] = template_content
            else:
                # Use default template if available
                default_template = default_prompts.get(
                    f"{agent_type.upper()}_{template_name.upper()}_TEMPLATE"
                )
                if default_template:
                    configured_prompts[f"{template_name}_template"] = default_template
    else:
        # Apply rich configuration options (existing workflow format)
        system_prompt = apply_rich_configuration_to_prompts(agent_type, config)
        if system_prompt:
            configured_prompts["system_prompt"] = system_prompt

    logger.debug(f"Configured {len(configured_prompts)} prompts for {agent_type} agent")
    return configured_prompts
