import pytest
from cognivault.agents.critic import CriticAgent
from cognivault.context import AgentContext


def test_critic_with_refiner_output():
    context = AgentContext(
        query="Was the election fair?",
        agent_outputs={"Refiner": "Some structured reflection on the election."},
    )
    agent = CriticAgent()
    updated_context = agent.run(context)

    assert "Critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["Critic"]
    assert "[Critique]" in output
    assert "may lack depth" in output


def test_critic_without_refiner_output():
    context = AgentContext(query="What about the turnout?")
    agent = CriticAgent()
    updated_context = agent.run(context)

    assert "Critic" in updated_context.agent_outputs
    output = updated_context.agent_outputs["Critic"]
    assert "No refined output found" in output
