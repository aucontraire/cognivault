import pytest
from unittest.mock import patch, AsyncMock

from cognivault.agents.synthesis.main import run_synthesis


@pytest.mark.asyncio
@patch("cognivault.agents.synthesis.agent.SynthesisAgent")
async def test_run_synthesis_returns_expected_output(mock_agent_class):
    from cognivault.context import AgentContext

    mock_agent = AsyncMock()
    mock_agent.name = "Synthesis"

    async def mock_run(context: AgentContext):
        context.add_agent_output("Synthesis", "Mocked synthesis output")

    mock_agent.run.side_effect = mock_run
    mock_agent_class.return_value = mock_agent

    query = "What causes revolutions?"
    result = await run_synthesis(query)

    assert result == "Mocked synthesis output"
    mock_agent.run.assert_called_once()
