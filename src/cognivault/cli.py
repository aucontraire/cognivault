"""Command-line interface for running Cognivault agents with specified queries."""

import logging
import typer
import asyncio
import json
import time
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text

from cognivault.config.logging_config import setup_logging
from cognivault.orchestrator import AgentOrchestrator
from cognivault.store.wiki_adapter import MarkdownExporter
from cognivault.store.topic_manager import TopicManager
from cognivault.diagnostics.cli import app as diagnostics_app

app = typer.Typer()

# Add diagnostics subcommands
app.add_typer(
    diagnostics_app, name="diagnostics", help="System diagnostics and monitoring"
)


async def run(
    query: str,
    agents: Optional[str] = None,
    log_level: str = "INFO",
    export_md: bool = False,
    trace: bool = False,
    health_check: bool = False,
    dry_run: bool = False,
    export_trace: Optional[str] = None,
):
    cli_name = "CLI"
    # Configure logging based on CLI-provided level
    try:
        level_value = logging.getLevelName(log_level.upper())
        if not isinstance(level_value, int):
            raise ValueError(f"Invalid log level: {log_level}")
    except (ValueError, TypeError):
        raise ValueError(f"Invalid log level: {log_level}")

    setup_logging(level_value)
    logger = logging.getLogger(__name__)
    console = Console()
    start_time = time.time()

    agents_to_run = [agent.strip() for agent in agents.split(",")] if agents else None
    logger.info(f"[{cli_name}] Received query: %s", query)
    logger.info(
        f"[{cli_name}] Agents to run: %s",
        agents_to_run if agents_to_run else "All agents",
    )

    orchestrator = AgentOrchestrator(agents_to_run=agents_to_run)

    # Health check mode - validate agents without execution
    if health_check:
        await _run_health_check(orchestrator, console, agents_to_run)
        return

    # Dry run mode - validate pipeline without execution
    if dry_run:
        await _run_dry_run(orchestrator, console, query, agents_to_run)
        return

    # Execute the pipeline
    if trace:
        console.print(
            "🔍 [bold]Starting pipeline execution with detailed tracing...[/bold]"
        )

    context = await orchestrator.run(query)
    execution_time = time.time() - start_time

    # Display execution results with optional trace information
    if trace:
        _display_detailed_trace(console, context, execution_time)
    else:
        _display_standard_output(console, context, execution_time)

    # Export trace if requested
    if export_trace:
        _export_trace_data(context, export_trace, execution_time)
        console.print(f"📊 [bold]Execution trace exported to: {export_trace}[/bold]")

    if export_md:
        # Initialize topic manager for auto-tagging
        topic_manager = TopicManager()

        # Analyze and suggest topics
        try:
            logger.info(f"[{cli_name}] Analyzing topics for auto-tagging...")
            topic_analysis = await topic_manager.analyze_and_suggest_topics(
                query=query, agent_outputs=context.agent_outputs
            )

            # Extract suggested topics and domain
            suggested_topics = [s.topic for s in topic_analysis.suggested_topics]
            suggested_domain = topic_analysis.suggested_domain

            logger.info(f"[{cli_name}] Suggested topics: {suggested_topics}")
            if suggested_domain:
                logger.info(f"[{cli_name}] Suggested domain: {suggested_domain}")

        except Exception as e:
            logger.warning(f"[{cli_name}] Topic analysis failed: {e}")
            suggested_topics = []
            suggested_domain = None

        # Export with enhanced metadata
        exporter = MarkdownExporter()
        md_path = exporter.export(
            agent_outputs=context.agent_outputs,
            question=query,
            topics=suggested_topics,
            domain=suggested_domain,
        )
        print(f"📄 Markdown exported to: {md_path}")

        # Display topic suggestions to user
        if suggested_topics:
            print(f"🏷️  Suggested topics: {', '.join(suggested_topics[:5])}")
        if suggested_domain:
            print(f"🎯 Suggested domain: {suggested_domain}")


@app.command()
def main(
    query: str,
    agents: str = typer.Option(
        None, help="Comma-separated list of agents to run (e.g., 'refiner,critic')"
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
):
    """
    Run Cognivault agents based on the provided query and options.

    Parameters
    ----------
    query : str
        The query string to be processed by the agents.
    agents : str, optional
        Comma-separated list of agents to run (e.g., 'refiner,critic'). If None, all agents are run.
    log_level : str, optional
        Logging level to use (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is 'INFO'.
    export_md : bool, optional
        Whether to export the agent outputs to a markdown file. Default is False.
    """

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
        )
    )


async def _run_health_check(orchestrator, console, agents_to_run):
    """Run health checks for all agents without executing the pipeline."""
    console.print("🩺 [bold]Running Agent Health Checks[/bold]")

    health_table = Table(title="Agent Health Status")
    health_table.add_column("Agent", style="bold")
    health_table.add_column("Status", justify="center")
    health_table.add_column("Details")

    registry = orchestrator.registry
    agents_list = (
        agents_to_run
        if agents_to_run
        else ["refiner", "critic", "historian", "synthesis"]
    )

    all_healthy = True
    for agent_name in agents_list:
        try:
            agent_key = agent_name.lower()
            is_healthy = registry.check_health(agent_key)

            if is_healthy:
                status = "[green]✓ Healthy[/green]"
                details = "Agent is ready for execution"
            else:
                status = "[red]✗ Unhealthy[/red]"
                details = "Health check failed"
                all_healthy = False

            health_table.add_row(agent_name.title(), status, details)

        except Exception as e:
            status = "[red]✗ Error[/red]"
            details = f"Health check error: {str(e)}"
            health_table.add_row(agent_name.title(), status, details)
            all_healthy = False

    console.print(health_table)

    if all_healthy:
        console.print(
            "\n[green]✅ All agents are healthy and ready for execution[/green]"
        )
    else:
        console.print("\n[red]❌ Some agents failed health checks[/red]")


async def _run_dry_run(orchestrator, console, query, agents_to_run):
    """Validate pipeline configuration without executing agents."""
    console.print("🧪 [bold]Dry Run - Pipeline Validation[/bold]")

    # Display configuration
    config_panel = Panel(
        f"Query: {query[:100]}{'...' if len(query) > 100 else ''}\n"
        f"Agents: {', '.join(agents_to_run) if agents_to_run else 'All default agents'}\n"
        f"Total Agents: {len(orchestrator.agents)}",
        title="Pipeline Configuration",
        border_style="blue",
    )
    console.print(config_panel)

    # Validate agent dependencies
    console.print("\n📋 [bold]Agent Dependency Validation[/bold]")

    dependency_tree = Tree("Pipeline Execution Order")
    for i, agent in enumerate(orchestrator.agents):
        dependency_tree.add(f"{i + 1}. {agent.name}")

    console.print(dependency_tree)

    # Validate agent health
    await _run_health_check(
        orchestrator, console, [agent.name for agent in orchestrator.agents]
    )

    console.print(
        "\n[green]✅ Pipeline validation complete - ready for execution[/green]"
    )


def _display_standard_output(console, context, execution_time):
    """Display standard agent outputs with performance metrics."""
    emoji_map = {
        "Refiner": "🧠",
        "Critic": "🤔",
        "Historian": "🕵️",
        "Synthesis": "🔗",
    }

    # Display performance summary
    console.print(f"\n⏱️  [bold]Pipeline completed in {execution_time:.2f}s[/bold]")
    console.print(
        f"✅ [green]{len(context.successful_agents)} agents completed successfully[/green]"
    )
    if context.failed_agents:
        console.print(f"❌ [red]{len(context.failed_agents)} agents failed[/red]")

    # Display agent outputs
    for agent_name, output in context.agent_outputs.items():
        emoji = emoji_map.get(agent_name, "🧠")
        console.print(f"\n{emoji} [bold]{agent_name}:[/bold]")
        console.print(output.strip())


def _display_detailed_trace(console, context, execution_time):
    """Display detailed execution trace with timing and metadata."""
    # Main execution summary
    summary_panel = Panel(
        f"[bold]Pipeline ID:[/bold] {context.context_id}\n"
        f"[bold]Total Execution Time:[/bold] {execution_time:.3f}s\n"
        f"[bold]Successful Agents:[/bold] {len(context.successful_agents)}\n"
        f"[bold]Failed Agents:[/bold] {len(context.failed_agents)}\n"
        f"[bold]Context Size:[/bold] {context.current_size:,} bytes",
        title="🔍 Execution Trace Summary",
        border_style="green",
    )
    console.print(summary_panel)

    # Agent execution status
    if context.agent_execution_status:
        console.print("\n📊 [bold]Agent Execution Status[/bold]")
        status_table = Table()
        status_table.add_column("Agent", style="bold")
        status_table.add_column("Status", justify="center")
        status_table.add_column("Execution Time")

        for agent_name, status in context.agent_execution_status.items():
            if status == "completed":
                status_display = "[green]✓ Completed[/green]"
            elif status == "failed":
                status_display = "[red]✗ Failed[/red]"
            elif status == "running":
                status_display = "[yellow]⏳ Running[/yellow]"
            else:
                status_display = f"[gray]{status}[/gray]"

            # Get timing from trace if available
            timing = "N/A"
            if agent_name in context.agent_trace:
                trace_events = context.agent_trace[agent_name]
                if trace_events:
                    timing = f"{len(trace_events)} events"

            status_table.add_row(agent_name, status_display, timing)

        console.print(status_table)

    # Execution edges (dependency flow)
    if context.execution_edges:
        console.print("\n🔗 [bold]Execution Flow[/bold]")
        for edge in context.execution_edges:
            from_agent = edge.get("from_agent", "START")
            to_agent = edge.get("to_agent", "END")
            edge_type = edge.get("edge_type", "normal")
            console.print(f"  {from_agent} → {to_agent} ({edge_type})")

    # Conditional routing decisions
    if context.conditional_routing:
        console.print("\n🔀 [bold]Conditional Routing Decisions[/bold]")
        for decision_point, details in context.conditional_routing.items():
            console.print(f"  [bold]{decision_point}:[/bold] {details}")

    # Agent outputs
    console.print("\n📝 [bold]Agent Outputs[/bold]")
    emoji_map = {
        "Refiner": "🧠",
        "Critic": "🤔",
        "Historian": "🕵️",
        "Synthesis": "🔗",
    }

    for agent_name, output in context.agent_outputs.items():
        emoji = emoji_map.get(agent_name, "🧠")
        output_panel = Panel(
            output.strip(), title=f"{emoji} {agent_name}", border_style="blue"
        )
        console.print(output_panel)


def _export_trace_data(context, export_path, execution_time):
    """Export detailed trace data to JSON file."""
    trace_data = {
        "pipeline_id": context.context_id,
        "execution_time_seconds": execution_time,
        "query": context.query,
        "successful_agents": list(context.successful_agents),
        "failed_agents": list(context.failed_agents),
        "agent_execution_status": context.agent_execution_status,
        "execution_edges": context.execution_edges,
        "conditional_routing": context.conditional_routing,
        "path_metadata": context.path_metadata,
        "agent_trace": context.agent_trace,
        "context_size_bytes": context.current_size,
        "agent_outputs": context.agent_outputs,
        "execution_state": context.execution_state,
        "timestamp": time.time(),
    }

    with open(export_path, "w") as f:
        json.dump(trace_data, f, indent=2, default=str)


if __name__ == "__main__":  # pragma: no cover
    app()
