from unittest.mock import patch, mock_open, Mock
from cognivault.context import AgentContext
import pytest
import json
import tempfile
import os
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
                "main",
                "What is cognitive dissonance?",
                "--agents=refiner",
                "--log-level=INFO",
            ],
        )
        assert result.exit_code == 0
        assert "🧠 Refiner:" in result.output


# Tests for new CLI flags


@pytest.mark.asyncio
async def test_cli_health_check_flag(capsys):
    """Test the --health-check flag functionality."""
    fake_orchestrator = Mock()
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.orchestrator.AgentOrchestrator", return_value=fake_orchestrator
    ):
        await cli_main("test query", health_check=True, log_level="INFO")
        captured = capsys.readouterr()
        assert "🩺" in captured.out  # Health check emoji
        assert "Agent Health Checks" in captured.out
        assert "✅ All agents are healthy" in captured.out


@pytest.mark.asyncio
async def test_cli_health_check_with_failures(capsys):
    """Test health check with agent failures."""
    fake_registry = Mock()

    def mock_check_health(agent_name):
        if agent_name == "refiner":
            return True
        elif agent_name == "critic":
            return False
        elif agent_name == "historian":
            raise Exception("Test error")
        return True

    fake_registry.check_health.side_effect = mock_check_health

    with patch(
        "cognivault.orchestrator.get_agent_registry", return_value=fake_registry
    ):
        await cli_main(
            "test query",
            agents="refiner,critic,historian",
            health_check=True,
            log_level="INFO",
        )
        captured = capsys.readouterr()
        assert "❌ Some agents failed health checks" in captured.out


@pytest.mark.asyncio
async def test_cli_dry_run_flag(capsys):
    """Test the --dry-run flag functionality."""
    fake_orchestrator = Mock()
    fake_agents = [Mock(name="Refiner"), Mock(name="Critic")]
    fake_orchestrator.agents = fake_agents
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.orchestrator.AgentOrchestrator", return_value=fake_orchestrator
    ):
        await cli_main("test query", dry_run=True, log_level="INFO")
        captured = capsys.readouterr()
        assert "🧪" in captured.out  # Dry run emoji
        assert "Dry Run - Pipeline Validation" in captured.out
        assert "Pipeline Configuration" in captured.out
        assert "✅ Pipeline validation complete" in captured.out


@pytest.mark.asyncio
async def test_cli_trace_flag(capsys):
    """Test the --trace flag functionality."""
    fake_context = AgentContext(query="test query with trace")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "test_context_123"
    fake_context.current_size = 1024
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = [
        {"from_agent": "START", "to_agent": "Refiner", "edge_type": "normal"}
    ]

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main("test query", agents="refiner", trace=True, log_level="INFO")
        captured = capsys.readouterr()
        assert "🔍" in captured.out  # Trace emoji
        assert "Starting pipeline execution with detailed tracing" in captured.out
        assert "Execution Trace Summary" in captured.out
        assert "Pipeline ID:" in captured.out
        assert "test_context_123" in captured.out


@pytest.mark.asyncio
async def test_cli_export_trace_flag(tmp_path):
    """Test the --export-trace flag functionality."""
    fake_context = AgentContext(query="test query for export")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "test_export_123"
    fake_context.current_size = 2048
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []
    fake_context.conditional_routing = {}
    fake_context.path_metadata = {}
    fake_context.agent_trace = {}
    fake_context.execution_state = {}

    export_file = tmp_path / "test_trace.json"

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main(
            "test query",
            agents="refiner",
            export_trace=str(export_file),
            log_level="INFO",
        )

        # Verify the trace file was created
        assert export_file.exists()

        # Verify the content
        with open(export_file, "r") as f:
            trace_data = json.load(f)

        assert trace_data["pipeline_id"] == "test_export_123"
        assert trace_data["query"] == "test query for export"
        assert "Refiner" in trace_data["successful_agents"]
        assert trace_data["context_size_bytes"] == 2048


@pytest.mark.asyncio
async def test_cli_trace_with_conditional_routing(capsys):
    """Test trace display with conditional routing information."""
    fake_context = AgentContext(query="test query with routing")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "test_routing_123"
    fake_context.current_size = 512
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = [
        {"from_agent": "Refiner", "to_agent": "Critic", "edge_type": "conditional"}
    ]
    fake_context.conditional_routing = {
        "refiner_decision": {"condition": "success", "target": "critic"}
    }

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main("test query", agents="refiner", trace=True, log_level="INFO")
        captured = capsys.readouterr()
        assert "🔀" in captured.out  # Conditional routing emoji
        assert "Conditional Routing Decisions" in captured.out
        assert "refiner_decision" in captured.out


@pytest.mark.asyncio
async def test_cli_trace_with_failed_agents(capsys):
    """Test trace display with failed agents."""
    fake_context = AgentContext(query="test query with failures")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = {"Critic"}
    fake_context.context_id = "test_failure_123"
    fake_context.current_size = 256
    fake_context.agent_execution_status = {"Refiner": "completed", "Critic": "failed"}

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main(
            "test query", agents="refiner,critic", trace=True, log_level="INFO"
        )
        captured = capsys.readouterr()
        assert "Failed Agents: 1" in captured.out
        assert "✗ Failed" in captured.out


def test_cli_health_check_typer_interface():
    """Test health check flag through typer CLI interface."""
    fake_orchestrator = Mock()
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.orchestrator.AgentOrchestrator", return_value=fake_orchestrator
    ):
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--health-check",
                "--log-level=INFO",
            ],
        )
        assert result.exit_code == 0
        assert "Agent Health Checks" in result.output


def test_cli_dry_run_typer_interface():
    """Test dry run flag through typer CLI interface."""
    fake_orchestrator = Mock()
    fake_agents = [Mock(name="Refiner")]
    fake_orchestrator.agents = fake_agents
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.orchestrator.AgentOrchestrator", return_value=fake_orchestrator
    ):
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--dry-run",
                "--log-level=INFO",
            ],
        )
        assert result.exit_code == 0
        assert "Pipeline Validation" in result.output


def test_cli_trace_typer_interface():
    """Test trace flag through typer CLI interface."""
    fake_context = AgentContext(query="test typer trace")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "test_typer_123"
    fake_context.current_size = 128
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--agents=refiner",
                "--trace",
                "--log-level=INFO",
            ],
        )
        assert result.exit_code == 0
        assert "detailed tracing" in result.output
        assert "Execution Trace Summary" in result.output


def test_cli_export_trace_typer_interface():
    """Test export-trace flag through typer CLI interface."""
    fake_context = AgentContext(query="test typer export")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "test_typer_export_123"
    fake_context.current_size = 64
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []
    fake_context.conditional_routing = {}
    fake_context.path_metadata = {}
    fake_context.agent_trace = {}
    fake_context.execution_state = {}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name

    try:
        with patch(
            "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
        ):
            result = runner.invoke(
                app,
                [
                    "main",
                    "test query",
                    "--agents=refiner",
                    f"--export-trace={tmp_path}",
                    "--log-level=INFO",
                ],
            )
            assert result.exit_code == 0
            assert "Execution trace exported to" in result.output

            # Verify file was created and contains expected data
            assert os.path.exists(tmp_path)
            with open(tmp_path, "r") as f:
                trace_data = json.load(f)
            assert trace_data["pipeline_id"] == "test_typer_export_123"
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_cli_combined_flags(capsys, tmp_path):
    """Test combining multiple new flags together."""
    fake_context = AgentContext(query="test combined flags")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "test_combined_123"
    fake_context.current_size = 512
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []
    fake_context.conditional_routing = {}
    fake_context.path_metadata = {}
    fake_context.agent_trace = {}
    fake_context.execution_state = {}

    export_file = tmp_path / "combined_trace.json"

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main(
            "test query",
            agents="refiner",
            trace=True,
            export_trace=str(export_file),
            log_level="INFO",
        )
        captured = capsys.readouterr()

        # Should show trace output
        assert "detailed tracing" in captured.out
        assert "Execution Trace Summary" in captured.out

        # Should export trace file
        assert "Execution trace exported to" in captured.out
        assert export_file.exists()


@pytest.mark.asyncio
async def test_cli_health_check_specific_agents(capsys):
    """Test health check with specific agents list."""
    fake_registry = Mock()

    def mock_check_health(agent_name):
        if agent_name == "refiner":
            return True
        elif agent_name == "critic":
            return False
        return True

    fake_registry.check_health.side_effect = mock_check_health

    with patch(
        "cognivault.orchestrator.get_agent_registry", return_value=fake_registry
    ):
        await cli_main(
            "test query", agents="refiner,critic", health_check=True, log_level="INFO"
        )
        captured = capsys.readouterr()
        assert "Refiner" in captured.out
        assert "Critic" in captured.out
        assert "✓ Healthy" in captured.out
        assert "✗ Unhealthy" in captured.out
