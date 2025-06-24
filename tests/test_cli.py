import pytest
from typer.testing import CliRunner
from cognivault.cli import app

runner = CliRunner()


def test_cli_runs_with_refiner():
    result = runner.invoke(app, ["Why is democracy shifting?", "--agents=refiner"])
    assert result.exit_code == 0
    assert "🧠 Refiner:" in result.output


def test_cli_runs_with_multiple_agents():
    result = runner.invoke(
        app, ["What causes political polarization?", "--agents=refiner,critic"]
    )
    assert result.exit_code == 0
    assert "🧠 Refiner:" in result.output
    assert "🤔 Critic:" in result.output


def test_cli_runs_with_all_agents():
    result = runner.invoke(app, ["Explain democratic backsliding."])
    assert result.exit_code == 0
    assert "🧠 Refiner:" in result.output
    assert "🤔 Critic:" in result.output
    assert "🕵️ Historian:" in result.output
    assert "🔗 Synthesis:" in result.output
