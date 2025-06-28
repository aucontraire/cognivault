from unittest.mock import patch, MagicMock
from cognivault.context import AgentContext
import pytest
from cognivault.cli import run as cli_main
from typer.testing import CliRunner
from cognivault.cli import app

runner = CliRunner()


@pytest.mark.asyncio
async def test_cli_runs_with_refiner(capsys):
    fake_context = AgentContext(query="Why is democracy shifting?")
    fake_context.agent_outputs = {"Refiner": "[Refined Note] Democracy is evolving..."}
    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main("Why is democracy shifting?", agents="refiner", log_level="INFO")
        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_export_md(tmp_path, capsys):
    query = "What is the role of memory in cognition?"
    export_path = tmp_path / "dummy_export.md"

    # Patch the MarkdownExporter to write to a known location
    from unittest.mock import patch

    with patch("cognivault.store.wiki_adapter.MarkdownExporter.export") as mock_export:
        mock_export.return_value = str(export_path)
        await cli_main(query, agents="refiner", log_level="INFO", export_md=True)
        mock_export.assert_called_once()
        captured = capsys.readouterr()
        assert "📄 Markdown exported to:" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_multiple_agents(capsys):
    fake_context = AgentContext(query="What causes political polarization?")
    fake_context.agent_outputs = {
        "Refiner": "[Refined Note] Something something politics.",
        "Critic": "[Critique] Something something bias.",
    }
    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main(
            "What causes political polarization?",
            agents="refiner,critic",
            log_level="INFO",
        )
        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out
        assert "🤔 Critic:" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_all_agents(capsys):
    fake_context = AgentContext(query="Explain democratic backsliding.")
    fake_context.agent_outputs = {
        "Refiner": "[Refined Note] Democracy is sliding.",
        "Critic": "[Critique] You're missing economic drivers.",
        "Historian": "[Historical context]",
        "Synthesis": "[Synthesis result]",
    }
    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
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
    fake_context = AgentContext(query="Test malformed agents")
    fake_context.agent_outputs = {"Refiner": "[Refined Note] Cleaned up input."}
    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main("Test malformed agents", agents=",refiner,,", log_level="INFO")
        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out


def test_cli_main_entrypoint_runs():
    fake_context = AgentContext(query="What is cognitive dissonance?")
    fake_context.agent_outputs = {"Refiner": "[Refined Note] It's when thoughts clash."}
    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
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
