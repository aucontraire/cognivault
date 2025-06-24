import pytest
from cognivault.agents.historian import HistorianAgent
from cognivault.context import AgentContext


def test_historian_agent_adds_mock_history():
    query = "Has Mexico seen electoral reforms recently?"
    context = AgentContext(query=query)

    agent = HistorianAgent()
    result_context = agent.run(context)

    assert "Historian" in result_context.agent_outputs
    output = result_context.agent_outputs["Historian"]

    assert "2024-10-15" in output
    assert "2024-11-05" in output
    assert len(result_context.retrieved_notes) == 2
