from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext


def test_refiner_agent_adds_output():
    context = AgentContext(query="What is the future of AI in education?")
    agent = RefinerAgent()
    updated_context = agent.run(context)

    assert "Refiner" in updated_context.agent_outputs
    assert "[Refined Note]" in updated_context.agent_outputs["Refiner"]
