import pytest
from unittest.mock import patch, MagicMock

from cognivault.agents.refiner.main import run_refiner


@patch("cognivault.agents.refiner.main.RefinerAgent")
def test_run_refiner_returns_expected_output(mock_agent_class):
    mock_agent = MagicMock()
    mock_agent.run.return_value.agent_outputs = {"Refiner": "Mocked refiner output"}
    mock_agent_class.return_value = mock_agent

    query = "What causes revolutions?"
    result = run_refiner(query)

    assert result == "Mocked refiner output"
    mock_agent.run.assert_called_once()
