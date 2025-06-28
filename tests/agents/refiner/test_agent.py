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
        text="AI will personalize learning and augment teachers.",
        tokens_used=42,
        model_name="gpt-4",
        finish_reason="stop",
    )

    agent = RefinerAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    assert "Refiner" in updated_context.agent_outputs
    assert "[Refined Note]" in updated_context.agent_outputs["Refiner"]
    assert "personalize learning" in updated_context.agent_outputs["Refiner"]


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
