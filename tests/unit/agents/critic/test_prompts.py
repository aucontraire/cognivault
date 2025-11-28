"""Tests for the CriticAgent prompts module."""

from cognivault.agents.critic.prompts import CRITIC_SYSTEM_PROMPT


def test_critic_system_prompt_exists() -> None:
    """Test that the CRITIC_SYSTEM_PROMPT constant is defined."""
    assert CRITIC_SYSTEM_PROMPT is not None
    assert isinstance(CRITIC_SYSTEM_PROMPT, str)
    assert len(CRITIC_SYSTEM_PROMPT) > 0


def test_critic_system_prompt_contains_key_components() -> None:
    """Test that the system prompt contains all required components from the charter."""
    # Check for main purpose and role
    assert "CriticAgent" in CRITIC_SYSTEM_PROMPT
    assert "second stage" in CRITIC_SYSTEM_PROMPT
    assert "cognitive reflection pipeline" in CRITIC_SYSTEM_PROMPT

    # Check for primary responsibilities
    assert "implicit assumptions" in CRITIC_SYSTEM_PROMPT
    assert "logical gaps" in CRITIC_SYSTEM_PROMPT
    assert "hidden biases" in CRITIC_SYSTEM_PROMPT

    # Check for bias types
    assert "Temporal bias" in CRITIC_SYSTEM_PROMPT
    assert "Cultural bias" in CRITIC_SYSTEM_PROMPT
    assert "Methodological bias" in CRITIC_SYSTEM_PROMPT
    assert "Scale bias" in CRITIC_SYSTEM_PROMPT

    # Check for behavioral modes
    assert "ACTIVE MODE" in CRITIC_SYSTEM_PROMPT
    assert "PASSIVE MODE" in CRITIC_SYSTEM_PROMPT

    # Check for output format instructions
    assert "Simple Queries" in CRITIC_SYSTEM_PROMPT
    assert "Complex Queries" in CRITIC_SYSTEM_PROMPT
    assert "Minimal Issues" in CRITIC_SYSTEM_PROMPT

    # Check for confidence scoring
    assert "Confidence" in CRITIC_SYSTEM_PROMPT
    assert "High" in CRITIC_SYSTEM_PROMPT
    assert "Medium" in CRITIC_SYSTEM_PROMPT
    assert "Low" in CRITIC_SYSTEM_PROMPT


def test_critic_system_prompt_contains_constraints() -> None:
    """Test that the system prompt contains proper constraints."""
    # Check for what NOT to do
    assert "DO NOT answer the question yourself" in CRITIC_SYSTEM_PROMPT
    assert "DO NOT rewrite or rephrase" in CRITIC_SYSTEM_PROMPT
    assert "DO NOT cite historical examples" in CRITIC_SYSTEM_PROMPT
    assert "cite historical examples or perform synthesis" in CRITIC_SYSTEM_PROMPT


def test_critic_system_prompt_contains_examples() -> None:
    """Test that the system prompt contains examples for different scenarios."""
    # Check for input/output examples
    assert (
        "What are the societal impacts of artificial intelligence"
        in CRITIC_SYSTEM_PROMPT
    )
    assert (
        "How has the structure and function of democratic institutions"
        in CRITIC_SYSTEM_PROMPT
    )
    assert "What are the documented economic effects" in CRITIC_SYSTEM_PROMPT

    # Check for different output formats in examples
    assert "Assumes AI will have significant social impact" in CRITIC_SYSTEM_PROMPT
    assert "- Assumptions:" in CRITIC_SYSTEM_PROMPT
    assert "Query is well-scoped and neutral" in CRITIC_SYSTEM_PROMPT


def test_critic_system_prompt_structure() -> None:
    """Test that the system prompt has proper structure and formatting."""
    # Check for main sections
    assert "## PRIMARY RESPONSIBILITIES" in CRITIC_SYSTEM_PROMPT
    assert "## BEHAVIORAL MODES" in CRITIC_SYSTEM_PROMPT
    assert "## OUTPUT FORMAT" in CRITIC_SYSTEM_PROMPT
    assert "## CONFIDENCE SCORING" in CRITIC_SYSTEM_PROMPT
    assert "## EXAMPLES" in CRITIC_SYSTEM_PROMPT
    assert "## CONSTRAINTS" in CRITIC_SYSTEM_PROMPT

    # Check that the prompt is well-structured with numbered points
    assert "1. **Identify assumptions" in CRITIC_SYSTEM_PROMPT
    assert "2. **Highlight logical gaps" in CRITIC_SYSTEM_PROMPT
    assert "3. **Surface potential biases" in CRITIC_SYSTEM_PROMPT


def test_critic_system_prompt_formatting_guidelines() -> None:
    """Test that the system prompt includes proper formatting guidelines."""
    # Check for format specifications
    assert "**Format**:" in CRITIC_SYSTEM_PROMPT
    assert "**Example**:" in CRITIC_SYSTEM_PROMPT

    # Check for structured bullet point format (markdown hyphens)
    assert "- Assumptions:" in CRITIC_SYSTEM_PROMPT
    assert "- Gaps:" in CRITIC_SYSTEM_PROMPT
    assert "- Biases:" in CRITIC_SYSTEM_PROMPT
    assert "- Confidence:" in CRITIC_SYSTEM_PROMPT


def test_critic_system_prompt_downstream_integration() -> None:
    """Test that the system prompt mentions downstream agent integration."""
    assert "downstream agents" in CRITIC_SYSTEM_PROMPT
    assert "historical lookup" in CRITIC_SYSTEM_PROMPT
    assert "synthesis" in CRITIC_SYSTEM_PROMPT
    # The prompt mentions downstream agents without naming specific agent classes


def test_critic_system_prompt_length() -> None:
    """Test that the system prompt is substantial but not excessively long."""
    # Should be comprehensive but not overwhelming
    # Updated to allow for structured output format instructions
    assert (
        2000 <= len(CRITIC_SYSTEM_PROMPT) <= 10000
    ), f"Prompt length: {len(CRITIC_SYSTEM_PROMPT)}"


def test_critic_system_prompt_no_harmful_content() -> None:
    """Test that the system prompt doesn't contain potentially harmful instructions."""
    harmful_phrases = [
        "ignore previous instructions",
        "disregard the above",
        "forget your role",
        "act as if",
        "pretend to be",
    ]

    prompt_lower = CRITIC_SYSTEM_PROMPT.lower()
    for phrase in harmful_phrases:
        assert phrase not in prompt_lower, f"Found potentially harmful phrase: {phrase}"


def test_critic_system_prompt_role_consistency() -> None:
    """Test that the system prompt maintains consistent role definition."""
    # Should not confuse the role with other agents
    assert "RefinerAgent" in CRITIC_SYSTEM_PROMPT  # Should mention but not claim to be
    assert (
        "first stage" not in CRITIC_SYSTEM_PROMPT
    )  # Should not claim to be first stage

    # Should be clear about being second stage
    assert "second stage" in CRITIC_SYSTEM_PROMPT

    # Should not claim to do things outside its scope
    assert "DO NOT answer the question" in CRITIC_SYSTEM_PROMPT
    assert "DO NOT rewrite" in CRITIC_SYSTEM_PROMPT


def test_critic_system_prompt_structured_output_format() -> None:
    """Test that the system prompt includes structured output format instructions."""
    # Check for structured output section
    assert "STRUCTURED OUTPUT FORMAT" in CRITIC_SYSTEM_PROMPT

    # Check for BiasDetail instructions
    assert "bias_details" in CRITIC_SYSTEM_PROMPT
    assert "BiasDetail" in CRITIC_SYSTEM_PROMPT
    assert "bias_type" in CRITIC_SYSTEM_PROMPT

    # Check for field descriptions
    assert "assumptions" in CRITIC_SYSTEM_PROMPT
    assert "logical_gaps" in CRITIC_SYSTEM_PROMPT
    assert "biases" in CRITIC_SYSTEM_PROMPT

    # Check for structured output examples in JSON format
    assert "```json" in CRITIC_SYSTEM_PROMPT
