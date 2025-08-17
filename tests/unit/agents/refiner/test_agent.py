import pytest
from typing import Any
from unittest.mock import MagicMock
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext
from cognivault.config.app_config import ApplicationConfig, set_config, reset_config
from tests.factories.agent_context_factories import (
    AgentContextPatterns,
    AgentContextFactory,
)
from tests.factories.mock_llm_factories import MockLLMFactory, MockLLMResponseFactory


@pytest.mark.asyncio
async def test_refiner_agent_adds_output() -> None:
    context = AgentContextPatterns.simple_query(
        "What is the future of AI in education?"
    )

    # Mock the LLMInterface to return a valid response structure
    mock_llm = MockLLMFactory.with_response(
        "How will artificial intelligence transform educational practices and student learning outcomes over the next decade?",
        tokens_used=42,
        model_name="gpt-4",
    )

    agent = RefinerAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify agent added output to context
    assert "refiner" in updated_context.agent_outputs
    assert "Refined query:" in updated_context.agent_outputs["refiner"]
    assert "transform educational practices" in updated_context.agent_outputs["refiner"]

    # Verify system prompt was used in LLM call
    mock_llm.generate.assert_called_once()
    call_args = mock_llm.generate.call_args
    assert call_args[1]["system_prompt"] is not None
    assert "RefinerAgent" in call_args[1]["system_prompt"]


@pytest.mark.asyncio
async def test_refiner_agent_unchanged_query() -> None:
    context = AgentContextPatterns.simple_query(
        "How do economic policies affect income inequality?"
    )

    # Mock the LLMInterface to return an unchanged response
    mock_llm = MockLLMFactory.with_response(
        "[Unchanged] How do economic policies affect income inequality?",
        tokens_used=20,
        model_name="gpt-4",
    )

    agent = RefinerAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify unchanged queries are handled properly
    assert "refiner" in updated_context.agent_outputs
    assert "[Unchanged]" in updated_context.agent_outputs["refiner"]
    assert "income inequality" in updated_context.agent_outputs["refiner"]


@pytest.mark.asyncio
async def test_refiner_agent_raises_without_text_field() -> None:
    context = AgentContextPatterns.simple_query("Will this break?")

    mock_llm: MagicMock = MagicMock()
    mock_response: MagicMock = MagicMock()
    del mock_response.text  # Ensure 'text' attribute is missing
    mock_llm.generate.return_value = mock_response

    agent = RefinerAgent(llm=mock_llm)

    with pytest.raises(ValueError, match="LLMResponse missing 'text' field"):
        await agent.run(context)


@pytest.mark.asyncio
async def test_refiner_handles_nonsense_query() -> None:
    """Test that RefinerAgent handles malformed or unclear queries with appropriate fallbacks."""
    from cognivault.llm.llm_interface import LLMResponse

    mock_llm: MagicMock = MagicMock()
    agent = RefinerAgent(llm=mock_llm)

    # Test cases: input -> expected fallback response (removed empty strings - validation should reject them)
    test_cases = [
        (
            "???",
            "What specific question or topic are you interested in learning about?",
        ),
        (
            "How do?",
            "What specific question or topic are you interested in learning about?",
        ),
        (
            "huh",
            "What topic would you like to explore or discuss?",
        ),
    ]

    for input_query, expected_fallback in test_cases:
        # Mock LLM to return the expected fallback response
        mock_llm.generate.return_value = MockLLMResponseFactory.generate_valid_data(
            text=expected_fallback, tokens_used=15, model_name="gpt-4"
        )

        context = AgentContextPatterns.simple_query(input_query)
        updated_context = await agent.run(context)

        # Verify fallback behavior
        assert "refiner" in updated_context.agent_outputs
        output = updated_context.agent_outputs["refiner"]
        assert f"Refined query: {expected_fallback}" == output

        # Verify system prompt was used
        call_args = mock_llm.generate.call_args
        assert call_args[1]["system_prompt"] is not None
        assert "FALLBACK MODE" in call_args[1]["system_prompt"]


@pytest.mark.asyncio
async def test_refiner_rejects_empty_query() -> None:
    """Test that AgentContext validation properly rejects empty queries."""
    from pydantic import ValidationError

    # Empty queries should be rejected at validation level
    with pytest.raises(ValidationError, match="Query cannot be empty"):
        AgentContextPatterns.simple_query("")

    # Whitespace-only queries should also be rejected
    with pytest.raises(ValidationError, match="Query cannot be empty"):
        AgentContextPatterns.simple_query("   ")


@pytest.mark.asyncio
async def test_refiner_agent_with_simulation_delay() -> None:
    """Test that refiner agent respects simulation delay configuration."""
    # Set up configuration with simulation delay enabled
    config = ApplicationConfig()
    config.execution.enable_simulation_delay = True
    config.execution.simulation_delay_seconds = 0.01  # Very short for testing
    set_config(config)

    try:
        mock_llm = MockLLMFactory.with_response(
            "Test refined query with delay", tokens_used=10, model_name="gpt-4"
        )

        query = "Test query with simulation delay"
        context = AgentContextPatterns.simple_query(query)

        agent = RefinerAgent(llm=mock_llm)

        # Measure execution time to verify delay was applied
        import time

        start_time = time.time()
        result_context = await agent.run(context)
        end_time = time.time()

        # Should have taken at least the simulation delay time
        assert (end_time - start_time) >= 0.01

        # Verify normal functionality still works
        assert "refiner" in result_context.agent_outputs
        output = result_context.agent_outputs["refiner"]
        assert "refined" in output.lower()

    finally:
        reset_config()
