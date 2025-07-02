import pytest
from unittest.mock import MagicMock
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext
from cognivault.config.app_config import ApplicationConfig, set_config, reset_config


@pytest.mark.asyncio
async def test_refiner_agent_adds_output():
    context = AgentContext(query="What is the future of AI in education?")

    # Mock the LLMInterface to return a valid response structure
    from cognivault.llm.llm_interface import LLMResponse

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        text="How will artificial intelligence transform educational practices and student learning outcomes over the next decade?",
        tokens_used=42,
        model_name="gpt-4",
        finish_reason="stop",
    )

    agent = RefinerAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify agent added output to context
    assert "Refiner" in updated_context.agent_outputs
    assert "Refined query:" in updated_context.agent_outputs["Refiner"]
    assert "transform educational practices" in updated_context.agent_outputs["Refiner"]

    # Verify system prompt was used in LLM call
    mock_llm.generate.assert_called_once()
    call_args = mock_llm.generate.call_args
    assert call_args[1]["system_prompt"] is not None
    assert "RefinerAgent" in call_args[1]["system_prompt"]


@pytest.mark.asyncio
async def test_refiner_agent_unchanged_query():
    context = AgentContext(query="How do economic policies affect income inequality?")

    # Mock the LLMInterface to return an unchanged response
    from cognivault.llm.llm_interface import LLMResponse

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        text="[Unchanged] How do economic policies affect income inequality?",
        tokens_used=20,
        model_name="gpt-4",
        finish_reason="stop",
    )

    agent = RefinerAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify unchanged queries are handled properly
    assert "Refiner" in updated_context.agent_outputs
    assert "[Unchanged]" in updated_context.agent_outputs["Refiner"]
    assert "income inequality" in updated_context.agent_outputs["Refiner"]


@pytest.mark.asyncio
async def test_refiner_agent_raises_without_text_field():
    context = AgentContext(query="Will this break?")

    mock_llm = MagicMock()
    mock_response = MagicMock()
    del mock_response.text  # Ensure 'text' attribute is missing
    mock_llm.generate.return_value = mock_response

    agent = RefinerAgent(llm=mock_llm)

    with pytest.raises(ValueError, match="LLMResponse missing 'text' field"):
        await agent.run(context)


@pytest.mark.asyncio
async def test_refiner_handles_blank_or_nonsense_query():
    """Test that RefinerAgent handles blank, malformed, or nonsense queries with appropriate fallbacks."""
    from cognivault.llm.llm_interface import LLMResponse

    mock_llm = MagicMock()
    agent = RefinerAgent(llm=mock_llm)

    # Test cases: input -> expected fallback response
    test_cases = [
        ("", "What topic would you like to explore or discuss?"),
        ("   ", "What topic would you like to explore or discuss?"),
        (
            "???",
            "What specific question or topic are you interested in learning about?",
        ),
        (
            "How do?",
            "What specific question or topic are you interested in learning about?",
        ),
    ]

    for input_query, expected_fallback in test_cases:
        # Mock LLM to return the expected fallback response
        mock_llm.generate.return_value = LLMResponse(
            text=expected_fallback,
            tokens_used=15,
            model_name="gpt-4",
            finish_reason="stop",
        )

        context = AgentContext(query=input_query)
        updated_context = await agent.run(context)

        # Verify fallback behavior
        assert "Refiner" in updated_context.agent_outputs
        output = updated_context.agent_outputs["Refiner"]
        assert f"Refined query: {expected_fallback}" == output

        # Verify system prompt was used
        call_args = mock_llm.generate.call_args
        assert call_args[1]["system_prompt"] is not None
        assert "FALLBACK MODE" in call_args[1]["system_prompt"]


@pytest.mark.asyncio
async def test_refiner_agent_with_simulation_delay():
    """Test that refiner agent respects simulation delay configuration."""
    # Set up configuration with simulation delay enabled
    config = ApplicationConfig()
    config.execution.enable_simulation_delay = True
    config.execution.simulation_delay_seconds = 0.01  # Very short for testing
    set_config(config)

    try:
        from cognivault.llm.llm_interface import LLMResponse

        mock_llm = MagicMock()
        mock_llm.generate.return_value = LLMResponse(
            text="Test refined query with delay",
            tokens_used=10,
            model_name="gpt-4",
            finish_reason="stop",
        )

        query = "Test query with simulation delay"
        context = AgentContext(query=query)

        agent = RefinerAgent(llm=mock_llm)

        # Measure execution time to verify delay was applied
        import time

        start_time = time.time()
        result_context = await agent.run(context)
        end_time = time.time()

        # Should have taken at least the simulation delay time
        assert (end_time - start_time) >= 0.01

        # Verify normal functionality still works
        assert "Refiner" in result_context.agent_outputs
        output = result_context.agent_outputs["Refiner"]
        assert "refined" in output.lower()

    finally:
        reset_config()
