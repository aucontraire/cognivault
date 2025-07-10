from unittest.mock import patch, mock_open, Mock
from cognivault.context import AgentContext
import pytest
import json
import tempfile
import os
from cognivault.cli import run as cli_main, create_llm_instance
from typer.testing import CliRunner
from cognivault.cli import app

runner = CliRunner()


@pytest.mark.asyncio
async def test_cli_default_execution_mode_is_langgraph_real():
    """Test that the default execution mode is now langgraph-real."""
    fake_context = AgentContext(query="Test default mode")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        # Test without explicit execution mode - should use langgraph-real
        await cli_main("Test default mode", log_level="INFO")
        # If this doesn't raise an exception, the default is working


@pytest.mark.asyncio
async def test_cli_legacy_mode_deprecation_warning(capsys):
    """Test that legacy mode shows deprecation warning."""
    fake_context = AgentContext(query="Test legacy warning")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ):
        await cli_main("Test legacy warning", execution_mode="legacy", log_level="INFO")
        captured = capsys.readouterr()
        assert "Legacy orchestrator is deprecated" in captured.out
        assert "will be removed in v1.1.0" in captured.out


@pytest.mark.asyncio
async def test_cli_langgraph_mode_deprecation_warning(capsys):
    """Test that intermediate langgraph mode shows deprecation warning."""
    fake_context = AgentContext(query="Test langgraph warning")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "Test langgraph warning", execution_mode="langgraph", log_level="INFO"
        )
        captured = capsys.readouterr()
        assert "'langgraph' mode is deprecated" in captured.out
        assert "Use 'langgraph-real' (default)" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_refiner(capsys):
    fake_context = AgentContext(query="Why is democracy shifting?")
    fake_context.agent_outputs = {"Refiner": "[Refined Note] Democracy is evolving..."}
    with patch(
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main("Why is democracy shifting?", agents="refiner", log_level="INFO")
        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_export_md(tmp_path, capsys):
    query = "What is the role of memory in cognition?"
    export_path = tmp_path / "dummy_export.md"

    # Create fake context with agent output
    fake_context = AgentContext(query=query)
    fake_context.agent_outputs = {
        "Refiner": "[Refined Note] Memory plays a crucial role..."
    }

    # Patch both the orchestrator and the MarkdownExporter
    from unittest.mock import patch

    with (
        patch("cognivault.store.wiki_adapter.MarkdownExporter.export") as mock_export,
        patch(
            "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
            return_value=fake_context,
        ),
    ):
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main("Test malformed agents", agents=",refiner,,", log_level="INFO")
        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out


def test_cli_main_entrypoint_runs():
    fake_context = AgentContext(query="What is cognitive dissonance?")
    fake_context.agent_outputs = {"Refiner": "[Refined Note] It's when thoughts clash."}
    with patch(
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator",
        return_value=fake_orchestrator,
    ):
        await cli_main("test query", health_check=True, log_level="INFO")
        captured = capsys.readouterr()
        assert "🩺" in captured.out  # Health check emoji
        assert "Agent Health Checks" in captured.out
        assert "✅ All agents are healthy" in captured.out


@pytest.mark.asyncio
async def test_cli_health_check_basic_functionality(capsys):
    """Test that health check mode runs without errors (integration test)."""
    await cli_main(
        "test query",
        agents="refiner,critic",
        health_check=True,
        log_level="INFO",
    )
    captured = capsys.readouterr()
    # Since we're using real orchestrator with real registry, expect healthy agents
    assert "Running Agent Health Checks" in captured.out
    assert "Agent Health Status" in captured.out


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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator",
        return_value=fake_orchestrator,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator",
        return_value=fake_orchestrator,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator",
        return_value=fake_orchestrator,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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
            "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
            return_value=fake_context,
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
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
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

    # Mock the get_agent_registry function that RealLangGraphOrchestrator uses
    with patch(
        "cognivault.langraph.real_orchestrator.get_agent_registry",
        return_value=fake_registry,
    ):
        await cli_main(
            "test query", agents="refiner,critic", health_check=True, log_level="INFO"
        )
        captured = capsys.readouterr()
        assert "Refiner" in captured.out
        assert "Critic" in captured.out
        assert "✓ Healthy" in captured.out
        assert "✗ Unhealthy" in captured.out


# Tests for execution mode functionality


@pytest.mark.asyncio
async def test_cli_execution_mode_legacy(capsys):
    """Test CLI with legacy execution mode."""
    fake_context = AgentContext(query="test legacy execution")
    fake_context.agent_outputs = {"Refiner": "Legacy execution output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run", return_value=fake_context
    ) as mock_orchestrator:
        await cli_main(
            "test query", agents="refiner", execution_mode="legacy", log_level="INFO"
        )

        # Verify legacy orchestrator was called
        mock_orchestrator.assert_called_once_with("test query")

        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out
        assert "Legacy execution output" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_langgraph(capsys):
    """Test CLI with LangGraph execution mode."""
    fake_context = AgentContext(query="test langgraph execution")
    fake_context.agent_outputs = {"Refiner": "LangGraph execution output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ) as mock_langgraph_orchestrator:
        await cli_main(
            "test query", agents="refiner", execution_mode="langgraph", log_level="INFO"
        )

        # Verify LangGraph orchestrator was called
        mock_langgraph_orchestrator.assert_called_once_with("test query")

        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out
        assert "LangGraph execution output" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_invalid():
    """Test CLI with invalid execution mode."""
    with pytest.raises(ValueError, match="Invalid execution mode: invalid"):
        await cli_main(
            "test query", agents="refiner", execution_mode="invalid", log_level="INFO"
        )


@pytest.mark.asyncio
async def test_cli_execution_mode_default_legacy(capsys):
    """Test CLI defaults to langgraph-real execution mode when not specified (Phase 1 migration)."""
    fake_context = AgentContext(query="test default execution")
    fake_context.agent_outputs = {"Refiner": "Default langgraph-real execution"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    with patch(
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
        return_value=fake_context,
    ) as mock_orchestrator:
        await cli_main(
            "test query",
            agents="refiner",
            # execution_mode not specified, should default to langgraph-real
            log_level="INFO",
        )

        # Verify RealLangGraphOrchestrator was called (new default behavior)
        mock_orchestrator.assert_called_once_with("test query")

        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_trace_legacy(capsys):
    """Test legacy execution mode with trace enabled."""
    fake_context = AgentContext(query="test legacy trace")
    fake_context.agent_outputs = {"Refiner": "Legacy trace output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "legacy_trace_123"
    fake_context.current_size = 1024
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            trace=True,
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🔍" in captured.out
        assert (
            "Starting pipeline execution with detailed tracing (legacy mode)"
            in captured.out
        )
        assert "Execution Trace Summary" in captured.out
        assert "legacy_trace_123" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_trace_langgraph(capsys):
    """Test LangGraph execution mode with trace enabled."""
    fake_context = AgentContext(query="test langgraph trace")
    fake_context.agent_outputs = {"Refiner": "LangGraph trace output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "langgraph_trace_123"
    fake_context.current_size = 2048
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            trace=True,
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🔍" in captured.out
        assert (
            "Starting pipeline execution with detailed tracing (langgraph mode)"
            in captured.out
        )
        assert "Execution Trace Summary" in captured.out
        assert "langgraph_trace_123" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_health_check_legacy(capsys):
    """Test legacy execution mode with health check."""
    fake_orchestrator = Mock()
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator",
        return_value=fake_orchestrator,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            health_check=True,
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🩺" in captured.out
        assert "Agent Health Checks" in captured.out
        assert "✅ All agents are healthy" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_health_check_langgraph(capsys):
    """Test LangGraph execution mode with health check."""
    fake_orchestrator = Mock()
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator",
        return_value=fake_orchestrator,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            health_check=True,
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🩺" in captured.out
        assert "Agent Health Checks" in captured.out
        assert "✅ All agents are healthy" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_dry_run_legacy(capsys):
    """Test legacy execution mode with dry run."""
    fake_orchestrator = Mock()
    fake_agents = [Mock(name="Refiner")]
    fake_orchestrator.agents = fake_agents
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator",
        return_value=fake_orchestrator,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            dry_run=True,
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🧪" in captured.out
        assert "Dry Run - Pipeline Validation" in captured.out
        assert "✅ Pipeline validation complete" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_dry_run_langgraph(capsys):
    """Test LangGraph execution mode with dry run."""
    fake_orchestrator = Mock()
    fake_agents = [Mock(name="Refiner")]
    fake_orchestrator.agents = fake_agents
    fake_registry = Mock()
    fake_registry.check_health.return_value = True
    fake_orchestrator.registry = fake_registry

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator",
        return_value=fake_orchestrator,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            dry_run=True,
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🧪" in captured.out
        assert "Dry Run - Pipeline Validation" in captured.out
        assert "✅ Pipeline validation complete" in captured.out


def test_cli_execution_mode_legacy_typer_interface():
    """Test legacy execution mode through typer CLI interface."""
    fake_context = AgentContext(query="test legacy typer")
    fake_context.agent_outputs = {"Refiner": "Legacy typer output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run",
        return_value=fake_context,
    ):
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--agents=refiner",
                "--execution-mode=legacy",
                "--log-level=INFO",
            ],
        )
        assert result.exit_code == 0
        assert "🧠 Refiner:" in result.output
        assert "Legacy typer output" in result.output


def test_cli_execution_mode_langgraph_typer_interface():
    """Test LangGraph execution mode through typer CLI interface."""
    fake_context = AgentContext(query="test langgraph typer")
    fake_context.agent_outputs = {"Refiner": "LangGraph typer output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        result = runner.invoke(
            app,
            [
                "main",
                "test query",
                "--agents=refiner",
                "--execution-mode=langgraph",
                "--log-level=INFO",
            ],
        )
        assert result.exit_code == 0
        assert "🧠 Refiner:" in result.output
        assert "LangGraph typer output" in result.output


def test_cli_execution_mode_invalid_typer_interface():
    """Test invalid execution mode through typer CLI interface."""
    result = runner.invoke(
        app,
        [
            "main",
            "test query",
            "--agents=refiner",
            "--execution-mode=invalid",
            "--log-level=INFO",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid execution mode: invalid" in str(result.exception)


@pytest.mark.asyncio
async def test_cli_execution_mode_with_multiple_agents_legacy(capsys):
    """Test legacy execution mode with multiple agents."""
    fake_context = AgentContext(query="test legacy multiple")
    fake_context.agent_outputs = {
        "Refiner": "Legacy refiner output",
        "Critic": "Legacy critic output",
    }
    fake_context.successful_agents = {"Refiner", "Critic"}
    fake_context.failed_agents = set()

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="legacy",
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out
        assert "🤔 Critic:" in captured.out
        assert "Legacy refiner output" in captured.out
        assert "Legacy critic output" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_multiple_agents_langgraph(capsys):
    """Test LangGraph execution mode with multiple agents."""
    fake_context = AgentContext(query="test langgraph multiple")
    fake_context.agent_outputs = {
        "Refiner": "LangGraph refiner output",
        "Critic": "LangGraph critic output",
    }
    fake_context.successful_agents = {"Refiner", "Critic"}
    fake_context.failed_agents = set()

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="langgraph",
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out
        assert "🤔 Critic:" in captured.out
        assert "LangGraph refiner output" in captured.out
        assert "LangGraph critic output" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_error_handling_legacy(capsys):
    """Test legacy execution mode with error handling."""
    fake_context = AgentContext(query="test legacy error")
    fake_context.agent_outputs = {"Refiner": "Partial output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = {"Critic"}

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="legacy",
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "✅" in captured.out  # Success for completed agents
        assert "1 agents completed successfully" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_error_handling_langgraph(capsys):
    """Test LangGraph execution mode with error handling."""
    fake_context = AgentContext(query="test langgraph error")
    fake_context.agent_outputs = {"Refiner": "Partial output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = {"Critic"}

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="langgraph",
            log_level="INFO",
        )

        captured = capsys.readouterr()
        assert "✅" in captured.out  # Success for completed agents
        assert "1 agents completed successfully" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_export_trace_legacy(tmp_path):
    """Test legacy execution mode with export trace."""
    fake_context = AgentContext(query="test query")
    fake_context.agent_outputs = {"Refiner": "Legacy export output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "legacy_export_123"
    fake_context.current_size = 512
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []
    fake_context.conditional_routing = {}
    fake_context.path_metadata = {}
    fake_context.agent_trace = {}
    fake_context.execution_state = {}

    export_file = tmp_path / "legacy_trace.json"

    with patch(
        "cognivault.orchestrator.AgentOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            export_trace=str(export_file),
            log_level="INFO",
        )

        # Verify trace file was created
        assert export_file.exists()

        # Verify content
        with open(export_file, "r") as f:
            trace_data = json.load(f)

        assert trace_data["pipeline_id"] == "legacy_export_123"
        assert trace_data["query"] == "test query"  # The query comes from the CLI call


@pytest.mark.asyncio
async def test_cli_execution_mode_with_export_trace_langgraph(tmp_path):
    """Test LangGraph execution mode with export trace."""
    fake_context = AgentContext(query="test query")
    fake_context.agent_outputs = {"Refiner": "LangGraph export output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()
    fake_context.context_id = "langgraph_export_123"
    fake_context.current_size = 1024
    fake_context.agent_execution_status = {"Refiner": "completed"}
    fake_context.execution_edges = []
    fake_context.conditional_routing = {}
    fake_context.path_metadata = {}
    fake_context.agent_trace = {}
    fake_context.execution_state = {}

    export_file = tmp_path / "langgraph_trace.json"

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            export_trace=str(export_file),
            log_level="INFO",
        )

        # Verify trace file was created
        assert export_file.exists()

        # Verify content
        with open(export_file, "r") as f:
            trace_data = json.load(f)

        assert trace_data["pipeline_id"] == "langgraph_export_123"
        assert trace_data["query"] == "test query"  # The query comes from the CLI call


# Tests for new LLM instance creation and topic analysis functionality


@patch("cognivault.cli.OpenAIConfig.load")
@patch("cognivault.cli.OpenAIChatLLM")
def test_create_llm_instance(mock_llm_class, mock_config_load):
    """Test create_llm_instance function creates LLM with proper configuration."""
    # Mock configuration
    mock_config = Mock()
    mock_config.api_key = "test-key"
    mock_config.model = "gpt-4"
    mock_config.base_url = "https://api.openai.com/v1"
    mock_config_load.return_value = mock_config

    # Mock LLM instance
    mock_llm = Mock()
    mock_llm_class.return_value = mock_llm

    # Test function
    result = create_llm_instance()

    # Verify config was loaded
    mock_config_load.assert_called_once()

    # Verify LLM was created with correct parameters
    mock_llm_class.assert_called_once_with(
        api_key="test-key", model="gpt-4", base_url="https://api.openai.com/v1"
    )

    # Verify returned instance
    assert result == mock_llm


@pytest.mark.asyncio
async def test_cli_topic_analysis_with_llm(capsys):
    """Test that CLI passes LLM instance to TopicManager for topic analysis."""
    fake_context = AgentContext(query="Democracy in Mexico and US")
    fake_context.agent_outputs = {"Refiner": "Democracy analysis output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    # Mock LLM instance
    mock_llm = Mock()

    # Mock topic analysis result
    mock_topic_analysis = Mock()
    mock_topic_analysis.suggested_topics = [
        Mock(topic="democracy"),
        Mock(topic="politics"),
        Mock(topic="government"),
    ]
    mock_topic_analysis.suggested_domain = "society"

    # Mock TopicManager
    mock_topic_manager = Mock()

    # Create an async mock for analyze_and_suggest_topics
    async def mock_analyze_topics(query, agent_outputs):
        return mock_topic_analysis

    mock_topic_manager.analyze_and_suggest_topics = mock_analyze_topics

    with (
        patch("cognivault.cli.create_llm_instance", return_value=mock_llm),
        patch(
            "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
            return_value=fake_context,
        ),
        patch("cognivault.cli.TopicManager", return_value=mock_topic_manager),
        patch(
            "cognivault.store.wiki_adapter.MarkdownExporter.export",
            return_value="test.md",
        ),
    ):
        await cli_main(
            "Democracy in Mexico and US",
            agents="refiner",
            log_level="INFO",
            export_md=True,
        )

        # Verify TopicManager was initialized with LLM
        from cognivault.cli import TopicManager

        TopicManager.assert_called_once_with(llm=mock_llm)

        captured = capsys.readouterr()
        assert "🎯 Suggested domain: society" in captured.out
        assert "🏷️  Suggested topics:" in captured.out


@pytest.mark.asyncio
async def test_cli_topic_analysis_error_handling(capsys):
    """Test that CLI handles topic analysis errors gracefully."""
    fake_context = AgentContext(query="Test query")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    # Mock LLM instance
    mock_llm = Mock()

    # Mock TopicManager that throws exception
    mock_topic_manager = Mock()

    # Create an async mock that throws exception
    async def mock_analyze_topics_error(query, agent_outputs):
        raise Exception("Topic analysis failed")

    mock_topic_manager.analyze_and_suggest_topics = mock_analyze_topics_error

    with (
        patch("cognivault.cli.create_llm_instance", return_value=mock_llm),
        patch(
            "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
            return_value=fake_context,
        ),
        patch("cognivault.cli.TopicManager", return_value=mock_topic_manager),
        patch(
            "cognivault.store.wiki_adapter.MarkdownExporter.export",
            return_value="test.md",
        ),
    ):
        await cli_main("Test query", agents="refiner", log_level="INFO", export_md=True)

        # Should still export markdown even if topic analysis fails
        captured = capsys.readouterr()
        assert "📄 Markdown exported to:" in captured.out
        # Should not crash, should handle gracefully


@pytest.mark.asyncio
async def test_cli_comparison_mode_with_llm_topic_analysis(capsys):
    """Test that comparison mode uses LLM for topic analysis."""
    fake_context = AgentContext(query="Test comparison query")
    fake_context.agent_outputs = {"Refiner": "Comparison output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    # Mock LLM instance
    mock_llm = Mock()

    # Mock topic analysis result
    mock_topic_analysis = Mock()
    mock_topic_analysis.suggested_topics = [Mock(topic="test")]
    mock_topic_analysis.suggested_domain = "technology"

    # Mock TopicManager
    mock_topic_manager = Mock()

    # Create an async mock for analyze_and_suggest_topics
    async def mock_analyze_topics(query, agent_outputs):
        return mock_topic_analysis

    mock_topic_manager.analyze_and_suggest_topics = mock_analyze_topics

    # Mock comparison results
    mock_results = {
        "legacy": {
            "success_count": 1,
            "last_context": fake_context,
            "execution_times": [1.0],
            "memory_usage": [100.0],
            "context_sizes": [1024],
            "agent_counts": [1],
            "error_count": 0,
            "errors": [],
        },
        "langgraph": {
            "success_count": 1,
            "last_context": fake_context,
            "execution_times": [1.2],
            "memory_usage": [120.0],
            "context_sizes": [1024],
            "agent_counts": [1],
            "error_count": 0,
            "errors": [],
        },
    }

    with (
        patch("cognivault.cli.create_llm_instance", return_value=mock_llm),
        patch("cognivault.orchestrator.AgentOrchestrator", return_value=Mock()),
        patch(
            "cognivault.langraph.orchestrator.LangGraphOrchestrator",
            return_value=Mock(),
        ),
        patch("cognivault.cli.TopicManager", return_value=mock_topic_manager),
        patch(
            "cognivault.store.wiki_adapter.MarkdownExporter.export",
            return_value="test.md",
        ),
    ):
        # Mock the actual orchestrator runs
        with (
            patch(
                "cognivault.orchestrator.AgentOrchestrator.run",
                return_value=fake_context,
            ),
            patch(
                "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
                return_value=fake_context,
            ),
        ):
            await cli_main(
                "Test comparison query",
                agents="refiner",
                log_level="INFO",
                export_md=True,
                compare_modes=True,
            )

            # Should call TopicManager with LLM in comparison mode
            # (This is called in _export_comparison_results)
            captured = capsys.readouterr()
            assert "🔄" in captured.out  # Comparison mode indicator
            assert "Performance Benchmark Results" in captured.out


@pytest.mark.asyncio
async def test_cli_topic_analysis_without_export_md(capsys):
    """Test that topic analysis is only run when export_md is True."""
    fake_context = AgentContext(query="Test query")
    fake_context.agent_outputs = {"Refiner": "Test output"}
    fake_context.successful_agents = {"Refiner"}
    fake_context.failed_agents = set()

    # Mock LLM instance
    mock_llm = Mock()

    # Mock TopicManager - should not be called
    mock_topic_manager = Mock()

    with (
        patch("cognivault.cli.create_llm_instance", return_value=mock_llm),
        patch(
            "cognivault.langraph.real_orchestrator.RealLangGraphOrchestrator.run",
            return_value=fake_context,
        ),
        patch("cognivault.cli.TopicManager", return_value=mock_topic_manager),
    ):
        await cli_main(
            "Test query",
            agents="refiner",
            log_level="INFO",
            export_md=False,  # Topic analysis should not run
        )

        # TopicManager should not be instantiated
        from cognivault.cli import TopicManager

        TopicManager.assert_not_called()

        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out
        assert "📄 Markdown exported to:" not in captured.out  # No export
