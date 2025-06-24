from cognivault.agents.synthesis import SynthesisAgent
from cognivault.context import AgentContext


def test_synthesis_agent_combines_outputs():
    context = AgentContext(query="Test")
    context.add_agent_output("Refiner", "Refined output.")
    context.add_agent_output("Historian", "Historical context.")
    context.add_agent_output("Critic", "Critique here.")

    agent = SynthesisAgent()
    updated_context = agent.run(context)

    synthesized = updated_context.agent_outputs.get("Synthesis")
    assert synthesized is not None
    assert "Refiner" in synthesized
    assert "Historian" in synthesized
    assert "Critic" in synthesized
