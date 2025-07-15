from unittest.mock import patch, Mock
from cognivault.context import AgentContext
import pytest
import json
import tempfile
import os
import typer
from cognivault.cli import (
    run as cli_main,
    create_llm_instance,
    _validate_langgraph_runtime,
)
from typer.testing import CliRunner
from cognivault.cli import app

runner = CliRunner()


@pytest.mark.asyncio
async def test_cli_default_execution_mode_is_langgraph_real():
    """Test that the default execution mode is now langgraph-real."""
    fake_context = AgentContext(query="Test default mode")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        # Test without explicit execution mode - should use langgraph-real
        await cli_main("Test default mode", log_level="INFO")
        # If this doesn't raise an exception, the default is working


@pytest.mark.asyncio
async def test_cli_legacy_mode_removed_error(capsys):
    """Test that legacy mode shows error message after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main("Test legacy error", execution_mode="legacy", log_level="INFO")

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out


@pytest.mark.asyncio
async def test_cli_langgraph_mode_removed_error(capsys):
    """Test that intermediate langgraph mode shows error message after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "Test langgraph error", execution_mode="langgraph", log_level="INFO"
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out


@pytest.mark.asyncio
async def test_cli_runs_with_refiner(capsys):
    fake_context = AgentContext(query="Why is democracy shifting?")
    fake_context.agent_outputs = {"Refiner": "[Refined Note] Democracy is evolving..."}
    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        await cli_main("Test malformed agents", agents=",refiner,,", log_level="INFO")
        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out


def test_cli_main_entrypoint_runs():
    fake_context = AgentContext(query="What is cognitive dissonance?")
    fake_context.agent_outputs = {"Refiner": "[Refined Note] It's when thoughts clash."}
    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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

    # Mock the get_agent_registry function that LangGraphOrchestrator uses
    with patch(
        "cognivault.langraph.orchestrator.get_agent_registry",
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
async def test_cli_execution_mode_legacy_fails(capsys):
    """Test CLI rejects legacy execution mode after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query", agents="refiner", execution_mode="legacy", log_level="INFO"
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_langgraph_fails(capsys):
    """Test CLI rejects intermediate langgraph execution mode after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query", agents="refiner", execution_mode="langgraph", log_level="INFO"
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_invalid():
    """Test CLI with invalid execution mode."""
    with pytest.raises(typer.Exit):
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
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ) as mock_orchestrator:
        await cli_main(
            "test query",
            agents="refiner",
            # execution_mode not specified, should default to langgraph-real
            log_level="INFO",
        )

        # Verify LangGraphOrchestrator was called (new default behavior)
        mock_orchestrator.assert_called_once_with("test query")

        captured = capsys.readouterr()
        assert "🧠 Refiner:" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_trace_legacy_fails(capsys):
    """Test legacy execution mode with trace fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            trace=True,
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_trace_langgraph_fails(capsys):
    """Test intermediate LangGraph execution mode with trace fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            trace=True,
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_health_check_legacy_fails(capsys):
    """Test legacy execution mode with health check fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            health_check=True,
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_health_check_langgraph_fails(capsys):
    """Test intermediate LangGraph execution mode with health check fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            health_check=True,
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_dry_run_legacy_fails(capsys):
    """Test legacy execution mode with dry run fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            dry_run=True,
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_dry_run_langgraph_fails(capsys):
    """Test intermediate LangGraph execution mode with dry run fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            dry_run=True,
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out


def test_cli_execution_mode_legacy_typer_interface():
    """Test legacy execution mode through typer CLI interface fails after Phase 3."""
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
    assert result.exit_code == 1
    assert "Unsupported execution mode: legacy" in result.output


def test_cli_execution_mode_langgraph_typer_interface():
    """Test intermediate LangGraph execution mode through typer CLI interface fails after Phase 3."""
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
    assert result.exit_code == 1
    assert "Unsupported execution mode: langgraph" in result.output


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
    assert result.exit_code == 1
    assert "Unsupported execution mode: invalid" in result.output


@pytest.mark.asyncio
async def test_cli_execution_mode_with_multiple_agents_legacy_fails(capsys):
    """Test legacy execution mode with multiple agents fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="legacy",
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_multiple_agents_langgraph_fails(capsys):
    """Test intermediate LangGraph execution mode with multiple agents fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="langgraph",
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_error_handling_legacy_fails(capsys):
    """Test legacy execution mode with error handling fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="legacy",
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_error_handling_langgraph_fails(capsys):
    """Test intermediate LangGraph execution mode with error handling fails after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner,critic",
            execution_mode="langgraph",
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out


@pytest.mark.asyncio
async def test_cli_execution_mode_with_export_trace_legacy_fails(tmp_path, capsys):
    """Test legacy execution mode with export trace fails after Phase 3."""
    export_file = tmp_path / "legacy_trace.json"

    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="legacy",
            export_trace=str(export_file),
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: legacy" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out

    # Verify trace file was NOT created since execution failed
    assert not export_file.exists()


@pytest.mark.asyncio
async def test_cli_execution_mode_with_export_trace_langgraph_fails(tmp_path, capsys):
    """Test intermediate LangGraph execution mode with export trace fails after Phase 3."""
    export_file = tmp_path / "langgraph_trace.json"

    with pytest.raises(typer.Exit):
        await cli_main(
            "test query",
            agents="refiner",
            execution_mode="langgraph",
            export_trace=str(export_file),
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Unsupported execution mode: langgraph" in captured.out
    assert "Only 'langgraph-real' mode is supported after Phase 3" in captured.out

    # Verify trace file was NOT created since execution failed
    assert not export_file.exists()


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
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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
async def test_cli_comparison_mode_disabled_after_phase3(capsys):
    """Test that comparison mode is disabled after Phase 3 legacy cleanup."""
    # Since legacy and intermediate modes are removed, comparison mode is no longer supported
    with pytest.raises(
        Exception
    ):  # Should fail due to missing comparison functionality
        await cli_main(
            "Test comparison query",
            agents="refiner",
            log_level="INFO",
            export_md=True,
            compare_modes=True,
        )

    # Note: After Phase 3, comparison mode needs to be reimplemented
    # to compare different configurations of langgraph-real mode only


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
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
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


# Rollback and Checkpoint Management Tests


@pytest.mark.asyncio
async def test_cli_rollback_functionality():
    """Test CLI rollback functionality with memory checkpoints."""
    fake_context = AgentContext(query="Test rollback")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    # Mock the rollback functionality
    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ) as mock_orchestrator:
        with patch("cognivault.cli._validate_langgraph_runtime") as mock_validate:
            mock_validate.return_value = None

            # Test rollback with memory checkpoints
            await cli_main(
                "Test rollback",
                execution_mode="langgraph-real",
                enable_checkpoints=True,
                log_level="INFO",
            )

            # Verify orchestrator was called with checkpoints enabled
            mock_orchestrator.assert_called_once()


@pytest.mark.asyncio
async def test_cli_checkpoint_management_with_thread_id():
    """Test CLI checkpoint management with custom thread ID."""
    fake_context = AgentContext(query="Test checkpoint thread")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ) as mock_run:
        with patch("cognivault.cli._validate_langgraph_runtime"):
            with patch("cognivault.cli.create_llm_instance"):
                # Test with custom thread ID
                await cli_main(
                    "Test checkpoint thread",
                    execution_mode="langgraph-real",
                    enable_checkpoints=True,
                    thread_id="test-thread-123",
                    log_level="INFO",
                )

                # Verify orchestrator run was called
                mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_cli_checkpoint_error_handling():
    """Test CLI error handling when checkpoint operations fail."""
    # Mock checkpoint failure
    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        side_effect=Exception("Checkpoint error"),
    ):
        with patch("cognivault.cli._validate_langgraph_runtime"):
            with patch("cognivault.cli.create_llm_instance"):
                # Test error handling
                with pytest.raises(Exception, match="Checkpoint error"):
                    await cli_main(
                        "Test checkpoint error",
                        execution_mode="langgraph-real",
                        enable_checkpoints=True,
                        log_level="INFO",
                    )


# LangGraph Runtime Validation Tests


def test_validate_langgraph_runtime_success():
    """Test successful LangGraph runtime validation."""
    with patch("langgraph.graph.StateGraph") as mock_state_graph:
        with patch("langgraph.checkpoint.memory.MemorySaver"):
            # Should not raise any exception
            _validate_langgraph_runtime()
            mock_state_graph.assert_called_once()


def test_validate_langgraph_runtime_import_error():
    """Test LangGraph runtime validation with import error."""
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 'langgraph'")
    ):
        with pytest.raises(ImportError, match="No module named 'langgraph'"):
            _validate_langgraph_runtime()


def test_validate_langgraph_runtime_functionality_error():
    """Test LangGraph runtime validation with functionality error."""
    with patch(
        "langgraph.graph.StateGraph", side_effect=RuntimeError("StateGraph failed")
    ):
        with pytest.raises(RuntimeError, match="StateGraph failed"):
            _validate_langgraph_runtime()


@pytest.mark.asyncio
async def test_cli_langgraph_validation_failure_handling(capsys):
    """Test CLI handling of LangGraph validation failures."""
    import typer

    with patch(
        "cognivault.cli._validate_langgraph_runtime",
        side_effect=ImportError("LangGraph not installed"),
    ):
        with pytest.raises((SystemExit, ImportError, typer.Exit)):
            await cli_main(
                "Test validation failure",
                execution_mode="langgraph-real",
                log_level="INFO",
            )


# Phase 3: Legacy analytics removed - tests no longer needed


# Phase 3: Analytics integration test removed


# Configuration and Error Handling Tests


def test_create_llm_instance_success():
    """Test successful LLM instance creation."""
    with patch("cognivault.cli.OpenAIConfig.load") as mock_config:
        mock_config.return_value = Mock(
            api_key="test-key", model="gpt-4", base_url="https://api.openai.com/v1"
        )

        with patch("cognivault.cli.OpenAIChatLLM") as mock_llm:
            llm_instance = create_llm_instance()

            mock_config.assert_called_once()
            mock_llm.assert_called_once_with(
                api_key="test-key", model="gpt-4", base_url="https://api.openai.com/v1"
            )


def test_create_llm_instance_config_error():
    """Test LLM instance creation with configuration error."""
    with patch(
        "cognivault.cli.OpenAIConfig.load",
        side_effect=ValueError("Invalid config"),
    ):
        with pytest.raises(ValueError, match="Invalid config"):
            create_llm_instance()


@pytest.mark.asyncio
async def test_cli_orchestrator_error_handling():
    """Test CLI error handling for orchestrator failures."""
    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        side_effect=RuntimeError("Orchestrator failed"),
    ):
        with patch("cognivault.cli._validate_langgraph_runtime"):
            with patch("cognivault.cli.create_llm_instance"):
                with pytest.raises(RuntimeError, match="Orchestrator failed"):
                    await cli_main(
                        "Test orchestrator error",
                        execution_mode="langgraph-real",
                        log_level="INFO",
                    )


@pytest.mark.asyncio
async def test_cli_agent_parsing_error_handling():
    """Test CLI error handling for invalid agent specifications."""
    fake_context = AgentContext(query="Test invalid agents")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        with patch("cognivault.cli._validate_langgraph_runtime"):
            # Test with empty agents string - should handle gracefully
            await cli_main(
                "Test invalid agents",
                execution_mode="langgraph-real",
                agents="",  # Empty agents string
                log_level="INFO",
            )


# Advanced Comparison Mode Tests


@pytest.mark.asyncio
async def test_cli_comparison_mode_with_benchmarking_disabled(capsys):
    """Test CLI comparison mode is disabled after Phase 3 legacy cleanup."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "Test comparison",
            execution_mode="langgraph-real",
            compare_modes=True,
            benchmark_runs=5,
            log_level="INFO",
        )

    captured = capsys.readouterr()
    assert "Comparison mode disabled after Phase 3 legacy cleanup" in captured.out
    assert "Comparison mode will be reimplemented" in captured.out


@pytest.mark.asyncio
async def test_cli_comparison_mode_memory_tracking_disabled(capsys):
    """Test CLI comparison mode memory tracking is disabled after Phase 3."""
    with pytest.raises(typer.Exit):
        await cli_main(
            "Test memory tracking",
            execution_mode="langgraph-real",
            compare_modes=True,
            log_level="DEBUG",
        )

    captured = capsys.readouterr()
    assert "Comparison mode disabled after Phase 3 legacy cleanup" in captured.out


# DAG Visualization Integration Tests


@pytest.mark.asyncio
async def test_cli_dag_visualization_integration():
    """Test CLI DAG visualization integration."""
    fake_context = AgentContext(query="Test DAG visualization")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch("cognivault.cli.cli_visualize_dag") as mock_visualize:
        with patch(
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
            return_value=fake_context,
        ):
            with patch("cognivault.cli._validate_langgraph_runtime"):
                mock_visualize.return_value = None

                # Test DAG visualization
                await cli_main(
                    "Test DAG visualization",
                    execution_mode="langgraph-real",
                    visualize_dag="stdout",
                    log_level="INFO",
                )

                # Verify visualization was called
                mock_visualize.assert_called_once()


@pytest.mark.asyncio
async def test_cli_dag_visualization_with_file_output():
    """Test CLI DAG visualization with file output."""
    fake_context = AgentContext(query="Test DAG file output")
    fake_context.agent_outputs = {"Refiner": "Test output", "Critic": "Test output"}

    with patch("cognivault.cli.cli_visualize_dag") as mock_visualize:
        with patch(
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
            return_value=fake_context,
        ):
            with patch("cognivault.cli._validate_langgraph_runtime"):
                mock_visualize.return_value = None

                # Test DAG visualization to file
                await cli_main(
                    "Test DAG file output",
                    execution_mode="langgraph-real",
                    visualize_dag="dag_output.md",
                    agents="refiner,critic",
                    log_level="INFO",
                )

                # Verify visualization was called with file output
                mock_visualize.assert_called_once()


@pytest.mark.asyncio
async def test_cli_dag_visualization_error_handling(capsys):
    """Test CLI DAG visualization error handling."""
    fake_context = AgentContext(query="Test DAG error")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.cli.cli_visualize_dag",
        side_effect=Exception("Visualization failed"),
    ) as mock_visualize:
        with patch(
            "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
            return_value=fake_context,
        ):
            with patch("cognivault.cli._validate_langgraph_runtime"):
                # Should handle visualization errors gracefully without raising
                await cli_main(
                    "Test DAG error",
                    execution_mode="langgraph-real",
                    visualize_dag="stdout",
                    log_level="INFO",
                )

                # Verify error was logged and execution continued
                captured = capsys.readouterr()
                assert "DAG visualization failed: Visualization failed" in captured.out


# Integration Tests for Complete CLI Workflows


@pytest.mark.asyncio
async def test_cli_complete_workflow_with_all_features():
    """Test complete CLI workflow with all features enabled."""
    fake_context = AgentContext(query="Complete workflow test")
    fake_context.agent_outputs = {
        "Refiner": "Refined query",
        "Critic": "Critical analysis",
        "Synthesis": "Final synthesis",
    }

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ):
        with patch("cognivault.cli._validate_langgraph_runtime"):
            with patch("cognivault.cli.cli_visualize_dag") as mock_visualize:
                mock_visualize.return_value = None

                # Test complete workflow
                await cli_main(
                    "Complete workflow test",
                    execution_mode="langgraph-real",
                    agents="refiner,critic,synthesis",
                    enable_checkpoints=True,
                    thread_id="workflow-123",
                    visualize_dag="stdout",
                    export_md=True,
                    log_level="DEBUG",
                )

                # Verify visualization was called (analytics removed in Phase 3)
                mock_visualize.assert_called_once()


@pytest.mark.asyncio
async def test_cli_error_recovery_workflow():
    """Test CLI error recovery and graceful degradation."""
    # Test recovery from LLM failure
    with patch(
        "cognivault.cli.create_llm_instance",
        side_effect=Exception("LLM creation failed"),
    ):
        with pytest.raises(Exception, match="LLM creation failed"):
            await cli_main(
                "Test error recovery", execution_mode="langgraph-real", log_level="INFO"
            )


@pytest.mark.asyncio
async def test_cli_performance_monitoring_integration():
    """Test CLI performance monitoring integration."""
    fake_context = AgentContext(query="Performance test")
    fake_context.agent_outputs = {"Refiner": "Test output"}

    with patch(
        "cognivault.langraph.orchestrator.LangGraphOrchestrator.run",
        return_value=fake_context,
    ) as mock_run:
        with patch("cognivault.cli._validate_langgraph_runtime"):
            with patch("cognivault.cli.create_llm_instance"):
                # Test performance monitoring
                await cli_main(
                    "Performance test",
                    execution_mode="langgraph-real",
                    trace=True,
                    log_level="DEBUG",
                )

                # Verify orchestrator run was called
                mock_run.assert_called_once()
