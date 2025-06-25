import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from cognivault.agents.synthesis.main import run_synthesis


@pytest.mark.asyncio
@patch("cognivault.agents.synthesis.agent.SynthesisAgent")
async def test_run_synthesis_returns_expected_output(mock_agent_class):
    mock_agent = AsyncMock()
    mock_agent.name = "Synthesis"

    mock_context = AsyncMock()
    mock_context.agent_outputs = {"Synthesis": "Mocked synthesis output"}
    mock_context.get_output = AsyncMock(return_value="Mocked synthesis output")
    mock_agent.run.return_value = mock_context
    mock_agent_class.return_value = mock_agent

    query = "What causes revolutions?"
    result = await run_synthesis(query)
    if asyncio.iscoroutine(result):
        result = await result

    assert result == "Mocked synthesis output"
    mock_agent.run.assert_called_once()
