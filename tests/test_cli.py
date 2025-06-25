import pytest
from cognivault.cli import run as cli_main
from typer.testing import CliRunner
from cognivault.cli import app

runner = CliRunner()


@pytest.mark.asyncio
async def test_cli_runs_with_refiner(capsys):
    await cli_main("Why is democracy shifting?", agents="refiner", log_level="INFO")
    captured = capsys.readouterr()
    assert "🧠 Refiner:" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_multiple_agents(capsys):
    await cli_main(
        "What causes political polarization?", agents="refiner,critic", log_level="INFO"
    )
    captured = capsys.readouterr()
    assert "🧠 Refiner:" in captured.out
    assert "🤔 Critic:" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_all_agents(capsys):
    await cli_main("Explain democratic backsliding.", agents=None, log_level="INFO")
    captured = capsys.readouterr()
    assert "🧠 Refiner:" in captured.out
    assert "🤔 Critic:" in captured.out
    assert "🕵️ Historian:" in captured.out
    assert "🔗 Synthesis:" in captured.out


@pytest.mark.asyncio
async def test_cli_invalid_log_level(capsys):
    # You may want to wrap this in a try/except block to catch logging misconfigurations
    with pytest.raises(Exception):
        await cli_main("Test invalid log level", agents="refiner", log_level="INVALID")


@pytest.mark.asyncio
async def test_cli_malformed_agents(capsys):
    await cli_main("Test malformed agents", agents=",refiner,,", log_level="INFO")
    captured = capsys.readouterr()
    assert "🧠 Refiner:" in captured.out


def test_cli_main_entrypoint_runs():
    result = runner.invoke(
        app,
        [
            "What is cognitive dissonance?",
            "--agents=refiner",
            "--log-level=INFO",
        ],
    )
    assert result.exit_code == 0
    assert "🧠 Refiner:" in result.output
