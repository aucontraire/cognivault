import pytest
from unittest.mock import MagicMock
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext


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
