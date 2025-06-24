import pytest
from unittest.mock import patch, MagicMock

from cognivault.agents.synthesis.main import run_synthesis


@patch("cognivault.agents.synthesis.main.SynthesisAgent")
def test_run_synthesis_returns_expected_output(mock_agent_class):
    mock_agent = MagicMock()
    mock_agent.run.return_value.agent_outputs = {"Synthesis": "Mocked synthesis output"}
    mock_agent_class.return_value = mock_agent

    query = "What causes revolutions?"
    result = run_synthesis(query)

    assert result == "Mocked synthesis output"
    mock_agent.run.assert_called_once()
