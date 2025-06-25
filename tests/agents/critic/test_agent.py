import pytest

pytest_plugins = ("pytest_asyncio",)
import asyncio
from cognivault.agents.critic.agent import CriticAgent
from cognivault.context import AgentContext


@pytest.mark.asyncio
async def test_critic_with_refiner_output():
    context = AgentContext(
        query="Was the election fair?",
        agent_outputs={"Refiner": "Some structured reflection on the election."},
    )
    agent = CriticAgent()
    updated_context = await agent.run(context)

    assert "Critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["Critic"]
    assert "[Critique]" in output
    assert "may lack depth" in output


@pytest.mark.asyncio
async def test_critic_without_refiner_output():
    context = AgentContext(query="What about the turnout?")
    agent = CriticAgent()
    updated_context = await agent.run(context)

    assert "Critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["Critic"]
    assert "No refined output found" in output
