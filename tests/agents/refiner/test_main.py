import pytest
from unittest.mock import patch, AsyncMock

from cognivault.agents.refiner.main import run_refiner


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.RefinerAgent")
async def test_run_refiner_returns_expected_output(mock_agent_class):
    mock_agent = AsyncMock()
    mock_agent.name = "Refiner"

    async def mock_run(context):
        context.add_agent_output("Refiner", "Mocked refiner output")

    mock_agent.run.side_effect = mock_run
    mock_agent_class.return_value = mock_agent

    query = "What causes revolutions?"
    result = await run_refiner(query)

    assert result == "Mocked refiner output"
    mock_agent.run.assert_awaited_once()
