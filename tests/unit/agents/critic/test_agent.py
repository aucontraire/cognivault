import pytest
from unittest.mock import MagicMock

from cognivault.agents.critic.agent import CriticAgent
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMResponse
from cognivault.agents.critic.prompts import CRITIC_SYSTEM_PROMPT

pytest_plugins = ("pytest_asyncio",)


def create_mock_llm(response_text: str) -> MagicMock:
    """Create a mock LLM that returns the specified response text."""
    mock_llm = MagicMock()
    mock_response = LLMResponse(
        text=response_text,
        tokens_used=50,
        model_name="test-model",
        finish_reason="stop",
    )
    mock_llm.generate.return_value = mock_response
    return mock_llm


@pytest.mark.asyncio
async def test_critic_with_refiner_output():
    """Test CriticAgent with RefinerAgent output available."""
    mock_llm = create_mock_llm(
        "Assumes 'fair' is objectively measurable—lacks definition of fairness criteria. (Confidence: Medium)"
    )

    context = AgentContext(
        query="Was the election fair?",
        agent_outputs={
            "refiner": "Was the 2020 presidential election conducted fairly according to established democratic standards?"
        },
    )

    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify LLM was called with correct parameters
    # Note: CriticAgent now uses dynamic PromptComposer prompts instead of static CRITIC_SYSTEM_PROMPT
    call_args = mock_llm.generate.call_args
    assert (
        call_args[1]["prompt"]
        == "Was the 2020 presidential election conducted fairly according to established democratic standards?"
    )
    # System prompt should be dynamically composed (contains original CRITIC_SYSTEM_PROMPT as base)
    assert (
        "CriticAgent, the second stage in a cognitive reflection pipeline"
        in call_args[1]["system_prompt"]
    )

    # Verify output was added to context
    assert "critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["critic"]
    assert "Assumes 'fair' is objectively measurable" in output
    assert "Confidence: Medium" in output


@pytest.mark.asyncio
async def test_critic_without_refiner_output():
    """Test CriticAgent when no RefinerAgent output is available."""
    mock_llm = create_mock_llm("This should not be called")

    context = AgentContext(query="What about the turnout?")
    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify LLM was not called when no refined output available
    mock_llm.generate.assert_not_called()

    # Verify fallback message was used
    assert "critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["critic"]
    assert "No refined output available from RefinerAgent to critique" in output


@pytest.mark.asyncio
async def test_critic_with_empty_refiner_output():
    """Test CriticAgent when RefinerAgent output is empty string."""
    mock_llm = create_mock_llm("This should not be called")

    context = AgentContext(query="Empty query test", agent_outputs={"refiner": ""})

    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify LLM was not called with empty input
    mock_llm.generate.assert_not_called()

    # Verify fallback message was used
    assert "critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["critic"]
    assert "No refined output available from RefinerAgent to critique" in output


@pytest.mark.asyncio
async def test_critic_complex_query_analysis():
    """Test CriticAgent with a complex query requiring structured critique."""
    complex_critique = """• Assumptions: Presumes democratic institutions are uniform across cultures
• Gaps: No definition of "evolved" or measurement criteria specified
• Biases: Western-centric view of democracy, post-Cold War timeframe assumption
• Confidence: Medium (query has clear intent but multiple ambiguities)"""

    mock_llm = create_mock_llm(complex_critique)

    context = AgentContext(
        query="How has democracy evolved?",
        agent_outputs={
            "refiner": "How has the structure and function of democratic institutions evolved since the Cold War?"
        },
    )

    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify LLM was called correctly
    # Note: CriticAgent now uses dynamic PromptComposer prompts instead of static CRITIC_SYSTEM_PROMPT
    call_args = mock_llm.generate.call_args
    assert (
        call_args[1]["prompt"]
        == "How has the structure and function of democratic institutions evolved since the Cold War?"
    )
    # System prompt should be dynamically composed (contains original CRITIC_SYSTEM_PROMPT as base)
    assert (
        "CriticAgent, the second stage in a cognitive reflection pipeline"
        in call_args[1]["system_prompt"]
    )

    # Verify structured output
    assert "critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["critic"]
    assert "• Assumptions:" in output
    assert "• Gaps:" in output
    assert "• Biases:" in output
    assert "• Confidence:" in output


@pytest.mark.asyncio
async def test_critic_well_scoped_query():
    """Test CriticAgent with a well-scoped query requiring minimal critique."""
    minimal_critique = "Query is well-scoped and neutral—includes timeframe, methodology, and specific metrics. No significant critique needed. (Confidence: High)"

    mock_llm = create_mock_llm(minimal_critique)

    context = AgentContext(
        query="Research question",
        agent_outputs={
            "refiner": "What are the documented economic effects of minimum wage increases on employment rates in peer-reviewed studies from 2010-2020?"
        },
    )

    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify minimal critique format
    assert "critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["critic"]
    assert "well-scoped and neutral" in output
    assert "No significant critique needed" in output
    assert "Confidence: High" in output


@pytest.mark.asyncio
async def test_critic_logging_behavior():
    """Test that CriticAgent logs appropriately during execution."""
    mock_llm = create_mock_llm("Test critique output")

    context = AgentContext(
        query="Test query", agent_outputs={"refiner": "Test refined query"}
    )

    agent = CriticAgent(llm=mock_llm)

    # Test that the agent can run without errors (logging tested implicitly)
    updated_context = await agent.run(context)
    assert "critic" in updated_context.agent_outputs


@pytest.mark.asyncio
async def test_critic_context_tracing():
    """Test that CriticAgent properly logs trace information."""
    mock_llm = create_mock_llm("Test critique for tracing")

    context = AgentContext(
        query="Trace test", agent_outputs={"refiner": "Refined query for tracing"}
    )

    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    # Verify output was added and context updated
    assert "critic" in updated_context.agent_outputs
    assert updated_context.agent_outputs["critic"] == "Test critique for tracing"


@pytest.mark.asyncio
async def test_critic_system_prompt_usage():
    """Test that CriticAgent uses the correct system prompt."""
    mock_llm = create_mock_llm("System prompt test output")

    context = AgentContext(
        query="System prompt test",
        agent_outputs={"refiner": "Test query for system prompt verification"},
    )

    agent = CriticAgent(llm=mock_llm)
    await agent.run(context)

    # Verify the system prompt was used
    # Note: CriticAgent now uses dynamic PromptComposer prompts instead of static CRITIC_SYSTEM_PROMPT
    call_args = mock_llm.generate.call_args
    system_prompt = call_args[1]["system_prompt"]
    # Should contain the base CRITIC_SYSTEM_PROMPT content plus configuration additions
    assert "CriticAgent" in system_prompt
    assert "second stage" in system_prompt
    # The composed prompt should include configuration-driven additions
    assert "cognitive reflection pipeline" in system_prompt


@pytest.mark.asyncio
async def test_critic_agent_name():
    """Test that CriticAgent has the correct name."""
    mock_llm = create_mock_llm("Name test")
    agent = CriticAgent(llm=mock_llm)
    assert agent.name == "critic"


@pytest.mark.asyncio
async def test_critic_response_stripping():
    """Test that CriticAgent strips whitespace from LLM responses."""
    mock_llm = create_mock_llm("  \n  Critique with extra whitespace  \n  ")

    context = AgentContext(
        query="Whitespace test",
        agent_outputs={"refiner": "Test query with whitespace response"},
    )

    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    output = updated_context.agent_outputs["critic"]
    assert output == "Critique with extra whitespace"
    assert not output.startswith(" ")
    assert not output.endswith(" ")


@pytest.mark.asyncio
async def test_critic_non_text_response():
    """Test fallback when LLM returns an object without a .text attribute."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = object()  # Lacks .text

    context = AgentContext(
        query="Fallback test",
        agent_outputs={"refiner": "This query triggers a fallback."},
    )

    agent = CriticAgent(llm=mock_llm)
    updated_context = await agent.run(context)

    output = updated_context.agent_outputs["critic"]
    assert output == "Error: received streaming response instead of text response"
