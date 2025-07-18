"""
CogniVault CLI Package

Comprehensive command-line interface for CogniVault operations including
legacy orchestration, declarative workflows, and system diagnostics.
"""

import typer
from .workflow_commands import workflow_app

# Import and re-export all functions needed for backward compatibility
from .main_commands import (
    create_llm_instance,
    _validate_langgraph_runtime,
    _run_with_api,
    _run_health_check,
    _run_dry_run,
    _display_standard_output,
    _display_detailed_trace,
    _export_trace_data,
    _run_rollback_mode,
    run,
)

# Import configuration and LLM classes for test compatibility
try:
    from cognivault.config.openai_config import OpenAIConfig
    from cognivault.llm.openai import OpenAIChatLLM
    from cognivault.diagnostics.visualize_dag import cli_visualize_dag
    from cognivault.store.topic_manager import TopicManager
    from cognivault.store.wiki_adapter import MarkdownExporter

    # Import API functions for compatibility
    from cognivault.api.factory import initialize_api, shutdown_api, get_api_mode
except ImportError:
    # Graceful degradation if modules not available - use type: ignore for optional imports
    OpenAIConfig = None  # type: ignore
    OpenAIChatLLM = None  # type: ignore
    cli_visualize_dag = None  # type: ignore
    TopicManager = None  # type: ignore
    MarkdownExporter = None  # type: ignore
    initialize_api = None  # type: ignore
    shutdown_api = None  # type: ignore
    get_api_mode = None  # type: ignore

# Create main CLI application
app = typer.Typer(
    name="cognivault",
    help="CogniVault - Intelligent multi-agent orchestration platform",
)

# Add command groups
app.add_typer(workflow_app, name="workflow", help="Declarative workflow operations")


# Add main commands directly to root (for backward compatibility)
# This preserves existing commands like: cognivault "query" --agents refiner,critic
@app.command()
def main(
    query: str,
    agents: str = typer.Option(
        None,
        help="Comma-separated list of agents to run (e.g., 'refiner,critic,historian,synthesis')",
    ),
    log_level: str = typer.Option(
        "INFO", help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
    export_md: bool = typer.Option(
        False, "--export-md", help="Export the agent outputs to a markdown file"
    ),
    trace: bool = typer.Option(
        False, "--trace", help="Show detailed execution trace with timing and metadata"
    ),
    health_check: bool = typer.Option(
        False,
        "--health-check",
        help="Run agent health checks without executing pipeline",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate pipeline configuration without execution"
    ),
    export_trace: str = typer.Option(
        None, "--export-trace", help="Export detailed execution trace to JSON file"
    ),
    execution_mode: str = typer.Option(
        "langgraph-real",
        "--execution-mode",
        help="Execution mode: 'langgraph-real' for production LangGraph integration (default)",
    ),
    compare_modes: bool = typer.Option(
        False,
        "--compare-modes",
        help="Run both langgraph and langgraph-real modes side-by-side for performance comparison",
    ),
    benchmark_runs: int = typer.Option(
        1,
        "--benchmark-runs",
        help="Number of runs for benchmarking (used with --compare-modes)",
    ),
    visualize_dag: str = typer.Option(
        None,
        "--visualize-dag",
        help="Visualize the DAG structure: 'stdout' for console output, or filepath",
    ),
    enable_checkpoints: bool = typer.Option(
        False,
        "--enable-checkpoints",
        help="Enable LangGraph checkpointing for conversation persistence",
    ),
    thread_id: str = typer.Option(
        None,
        "--thread-id",
        help="Thread ID for conversation scoping",
    ),
    rollback_last_checkpoint: bool = typer.Option(
        False,
        "--rollback-last-checkpoint",
        help="Rollback to the latest checkpoint for the thread",
    ),
    use_api: bool = typer.Option(
        False,
        "--use-api",
        help="Use API layer instead of direct orchestrator",
    ),
    api_mode: str = typer.Option(
        None,
        "--api-mode",
        help="API mode: 'real' (production) or 'mock' (testing)",
    ),
):
    """
    Run CogniVault agents based on the provided query and options.

    This is the main orchestration command that processes queries through
    the traditional agent pipeline (Refiner, Historian, Critic, Synthesis).

    For declarative DAG workflows, use: cognivault workflow run
    """
    # Import and delegate to main_commands
    from .main_commands import run
    import asyncio

    asyncio.run(
        run(
            query,
            agents,
            log_level,
            export_md,
            trace,
            health_check,
            dry_run,
            export_trace,
            execution_mode,
            compare_modes,
            benchmark_runs,
            visualize_dag,
            enable_checkpoints,
            thread_id,
            rollback_last_checkpoint,
            use_api,
            api_mode,
        )
    )


# Import diagnostics from main_commands and add to CLI
def setup_diagnostics():
    """Set up diagnostics subcommands."""
    try:
        from cognivault.diagnostics.cli import app as diagnostics_app

        app.add_typer(
            diagnostics_app,
            name="diagnostics",
            help="System diagnostics and monitoring",
        )
    except ImportError:
        pass  # Diagnostics not available


# Setup diagnostics on import
setup_diagnostics()


def cli_main():
    """Entry point for the cognivault CLI command."""
    app()


if __name__ == "__main__":
    cli_main()
