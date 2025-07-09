"""Command-line interface for running Cognivault agents with specified queries."""

import logging
import typer
import asyncio
import json
import time
import statistics
from typing import Optional, Union, Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from cognivault.config.logging_config import setup_logging
from cognivault.config.openai_config import OpenAIConfig
from cognivault.orchestrator import AgentOrchestrator
from cognivault.langraph.orchestrator import LangGraphOrchestrator
from cognivault.store.wiki_adapter import MarkdownExporter
from cognivault.store.topic_manager import TopicManager
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMInterface
from cognivault.diagnostics.cli import app as diagnostics_app

app = typer.Typer()

# Add diagnostics subcommands
app.add_typer(
    diagnostics_app, name="diagnostics", help="System diagnostics and monitoring"
)


def create_llm_instance() -> LLMInterface:
    """Create and configure an LLM instance for use by agents and topic manager."""
    llm_config = OpenAIConfig.load()
    return OpenAIChatLLM(
        api_key=llm_config.api_key,
        model=llm_config.model,
        base_url=llm_config.base_url,
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
    execution_mode: str = "legacy",
    compare_modes: bool = False,
    benchmark_runs: int = 1,
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

    # Validate execution mode
    if execution_mode not in ["legacy", "langgraph"]:
        raise ValueError(
            f"Invalid execution mode: {execution_mode}. Must be 'legacy' or 'langgraph'"
        )

    # Handle comparison mode
    if compare_modes:
        # Create shared LLM instance for comparison mode
        llm = create_llm_instance()
        await _run_comparison_mode(
            query,
            agents_to_run,
            console,
            trace,
            export_md,
            export_trace,
            benchmark_runs,
            llm,
        )
        return

    logger.info(f"[{cli_name}] Execution mode: {execution_mode}")

    # Create shared LLM instance for agents and topic manager
    llm = create_llm_instance()

    # Create orchestrator based on execution mode
    if execution_mode == "legacy":
        orchestrator: Union[AgentOrchestrator, LangGraphOrchestrator] = (
            AgentOrchestrator(agents_to_run=agents_to_run)
        )
    elif execution_mode == "langgraph":
        orchestrator = LangGraphOrchestrator(agents_to_run=agents_to_run)
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

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
            f"🔍 [bold]Starting pipeline execution with detailed tracing ({execution_mode} mode)...[/bold]"
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
        # Initialize topic manager for auto-tagging with shared LLM
        topic_manager = TopicManager(llm=llm)

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
    execution_mode: str = typer.Option(
        "legacy",
        "--execution-mode",
        help="Execution mode: 'legacy' for current orchestrator, 'langgraph' for DAG execution",
    ),
    compare_modes: bool = typer.Option(
        False,
        "--compare-modes",
        help="Run both legacy and langgraph modes side-by-side for performance comparison",
    ),
    benchmark_runs: int = typer.Option(
        1,
        "--benchmark-runs",
        help="Number of runs for benchmarking (used with --compare-modes for statistical accuracy)",
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
            execution_mode,
            compare_modes,
            benchmark_runs,
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


async def _run_comparison_mode(
    query: str,
    agents_to_run: Optional[list[str]],
    console: Console,
    trace: bool,
    export_md: bool,
    export_trace: Optional[str],
    benchmark_runs: int = 1,
    llm: Optional[LLMInterface] = None,
):
    """Run both legacy and langgraph modes side-by-side for comparison."""
    if benchmark_runs > 1:
        console.print(
            f"🔄 [bold magenta]Running Performance Benchmark ({benchmark_runs} runs per mode)[/bold magenta]\n"
        )
    else:
        console.print(
            "🔄 [bold magenta]Running Side-by-Side Execution Mode Comparison[/bold magenta]\n"
        )

    results: Dict[str, Any] = {}

    # Run both execution modes with benchmarking
    for mode in ["legacy", "langgraph"]:
        console.print(f"⚡ [bold]Running {mode.title()} Mode...[/bold]")

        mode_results: Dict[str, Any] = {
            "execution_times": [],
            "memory_usage": [],
            "context_sizes": [],
            "agent_counts": [],
            "success_count": 0,
            "error_count": 0,
            "last_context": None,
            "errors": [],
        }

        # Run multiple iterations for benchmarking
        for run_num in range(benchmark_runs):
            if benchmark_runs > 1:
                console.print(f"  📊 Run {run_num + 1}/{benchmark_runs}")

            # Create orchestrator for this mode
            if mode == "legacy":
                mode_orchestrator: Union[AgentOrchestrator, LangGraphOrchestrator] = (
                    AgentOrchestrator(agents_to_run=agents_to_run)
                )
            else:
                mode_orchestrator = LangGraphOrchestrator(agents_to_run=agents_to_run)

            # Measure memory before execution (if available)
            memory_before = 0.0
            process = None
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except (ImportError, OSError, AttributeError):
                    memory_before = 0.0
                    process = None

            # Execute with timing
            start_time = time.time()
            try:
                context = await mode_orchestrator.run(query)
                execution_time = time.time() - start_time

                # Measure memory after execution (if available)
                memory_used = 0.0
                if PSUTIL_AVAILABLE and process is not None and memory_before > 0:
                    try:
                        memory_after = process.memory_info().rss / 1024 / 1024  # MB
                        memory_used = memory_after - memory_before
                    except (OSError, AttributeError):
                        memory_used = 0.0

                # Record metrics
                mode_results["execution_times"].append(execution_time)
                mode_results["memory_usage"].append(memory_used)
                mode_results["context_sizes"].append(context.current_size)
                mode_results["agent_counts"].append(len(context.agent_outputs))
                mode_results["success_count"] += 1
                mode_results["last_context"] = context

                if benchmark_runs > 1:
                    console.print(f"    ✅ Completed in {execution_time:.3f}s")

            except Exception as e:
                execution_time = time.time() - start_time
                mode_results["execution_times"].append(execution_time)
                mode_results["error_count"] += 1
                mode_results["errors"].append(str(e))

                if benchmark_runs > 1:
                    console.print(f"    ❌ Failed after {execution_time:.3f}s: {e}")

        results[mode] = mode_results

        # Display summary for this mode
        if mode_results["success_count"] > 0:
            avg_time = statistics.mean(mode_results["execution_times"])
            console.print(
                f"📈 {mode.title()} summary: {mode_results['success_count']}/{benchmark_runs} successful, "
                f"avg time: {avg_time:.3f}s\n"
            )
        else:
            console.print(f"❌ {mode.title()} mode: All runs failed\n")

    # Display comparison results
    _display_comparison_results(results, console, query)

    # Handle export options if both modes succeeded
    if export_md and all(result["success_count"] > 0 for result in results.values()):
        await _export_comparison_results(results, export_md, query, llm)

    if export_trace:
        _export_comparison_trace(results, export_trace)


def _display_comparison_results(results: dict, console: Console, query: str):
    """Display side-by-side comparison of execution results with enhanced benchmarking data."""
    console.print("📊 [bold blue]Performance Benchmark Results[/bold blue]\n")

    # Performance comparison table
    perf_table = Table(title="Performance Comparison")
    perf_table.add_column("Metric", style="bold")
    perf_table.add_column("Legacy Mode", justify="right")
    perf_table.add_column("LangGraph Mode", justify="right")
    perf_table.add_column("Difference", justify="right")

    legacy_results = results["legacy"]
    langgraph_results = results["langgraph"]

    # Calculate statistics for execution times
    if legacy_results["execution_times"] and langgraph_results["execution_times"]:
        legacy_avg = statistics.mean(legacy_results["execution_times"])
        langgraph_avg = statistics.mean(langgraph_results["execution_times"])
        time_diff = legacy_avg - langgraph_avg
        time_diff_pct = (time_diff / legacy_avg) * 100 if legacy_avg > 0 else 0

        # Execution time (with std dev if multiple runs)
        if len(legacy_results["execution_times"]) > 1:
            legacy_std = statistics.stdev(legacy_results["execution_times"])
            legacy_time_str = f"{legacy_avg:.3f}s ±{legacy_std:.3f}"
        else:
            legacy_time_str = f"{legacy_avg:.3f}s"

        if len(langgraph_results["execution_times"]) > 1:
            langgraph_std = statistics.stdev(langgraph_results["execution_times"])
            langgraph_time_str = f"{langgraph_avg:.3f}s ±{langgraph_std:.3f}"
        else:
            langgraph_time_str = f"{langgraph_avg:.3f}s"

        perf_table.add_row(
            "Avg Execution Time",
            legacy_time_str,
            langgraph_time_str,
            f"{time_diff:+.3f}s ({time_diff_pct:+.1f}%)",
        )

        # Min/Max times if multiple runs
        if len(legacy_results["execution_times"]) > 1:
            legacy_min = min(legacy_results["execution_times"])
            legacy_max = max(legacy_results["execution_times"])
            langgraph_min = min(langgraph_results["execution_times"])
            langgraph_max = max(langgraph_results["execution_times"])

            perf_table.add_row(
                "Min Time",
                f"{legacy_min:.3f}s",
                f"{langgraph_min:.3f}s",
                f"{legacy_min - langgraph_min:+.3f}s",
            )
            perf_table.add_row(
                "Max Time",
                f"{legacy_max:.3f}s",
                f"{langgraph_max:.3f}s",
                f"{legacy_max - langgraph_max:+.3f}s",
            )

    # Success rate
    legacy_success_rate = (
        legacy_results["success_count"] / len(legacy_results["execution_times"])
        if legacy_results["execution_times"]
        else 0
    )
    langgraph_success_rate = (
        langgraph_results["success_count"] / len(langgraph_results["execution_times"])
        if langgraph_results["execution_times"]
        else 0
    )

    perf_table.add_row(
        "Success Rate",
        f"{legacy_success_rate:.1%}",
        f"{langgraph_success_rate:.1%}",
        f"{legacy_success_rate - langgraph_success_rate:+.1%}",
    )

    # Memory usage
    if legacy_results["memory_usage"] and langgraph_results["memory_usage"]:
        legacy_mem_avg = statistics.mean(legacy_results["memory_usage"])
        langgraph_mem_avg = statistics.mean(langgraph_results["memory_usage"])
        mem_diff = legacy_mem_avg - langgraph_mem_avg

        perf_table.add_row(
            "Avg Memory Usage",
            f"{legacy_mem_avg:.1f} MB",
            f"{langgraph_mem_avg:.1f} MB",
            f"{mem_diff:+.1f} MB",
        )

    # Context size
    if legacy_results["context_sizes"] and langgraph_results["context_sizes"]:
        legacy_size_avg = statistics.mean(legacy_results["context_sizes"])
        langgraph_size_avg = statistics.mean(langgraph_results["context_sizes"])
        size_diff = legacy_size_avg - langgraph_size_avg

        perf_table.add_row(
            "Avg Context Size",
            f"{legacy_size_avg:,.0f} bytes",
            f"{langgraph_size_avg:,.0f} bytes",
            f"{size_diff:+,.0f} bytes",
        )

    console.print(perf_table)
    console.print()

    # Display detailed results if both modes have successful runs
    if legacy_results["success_count"] > 0 and langgraph_results["success_count"] > 0:
        _display_output_comparison(results, console)
    else:
        # Show error details
        console.print("❌ [bold red]Execution Errors:[/bold red]")
        for mode, mode_results in results.items():
            if mode_results["errors"]:
                console.print(f"  {mode.title()} mode errors:")
                for error in mode_results["errors"]:
                    console.print(f"    • {error}")
        console.print()


def _display_output_comparison(results: dict, console: Console):
    """Display comparison of agent outputs between modes."""
    legacy_context = results["legacy"]["last_context"]
    langgraph_context = results["langgraph"]["last_context"]

    if not legacy_context or not langgraph_context:
        console.print(
            "⚠️  [yellow]Cannot compare outputs - missing context data[/yellow]"
        )
        return

    legacy_outputs = legacy_context.agent_outputs
    langgraph_outputs = langgraph_context.agent_outputs

    # Find all agents that ran in either mode
    all_agents = set(legacy_outputs.keys()) | set(langgraph_outputs.keys())

    for agent in all_agents:
        legacy_output = legacy_outputs.get(agent, "[Not executed]")
        langgraph_output = langgraph_outputs.get(agent, "[Not executed]")

        # Check if outputs are identical
        outputs_match = legacy_output == langgraph_output
        match_indicator = "✅ Identical" if outputs_match else "🔄 Different"

        console.print(f"🤖 [bold]{agent} Agent - {match_indicator}[/bold]")

        if not outputs_match:
            # Show truncated outputs for comparison
            legacy_preview = (
                (legacy_output[:100] + "...")
                if len(legacy_output) > 100
                else legacy_output
            )
            langgraph_preview = (
                (langgraph_output[:100] + "...")
                if len(langgraph_output) > 100
                else langgraph_output
            )

            comparison_table = Table(show_header=True)
            comparison_table.add_column("Legacy Mode", style="cyan", width=40)
            comparison_table.add_column("LangGraph Mode", style="green", width=40)
            comparison_table.add_row(legacy_preview, langgraph_preview)

            console.print(comparison_table)

        console.print()


async def _export_comparison_results(
    results: dict, export_md: bool, query: str, llm: Optional[LLMInterface] = None
):
    """Export comparison results to markdown."""
    # For now, export the legacy mode results as primary
    # TODO: Enhance to create a comprehensive comparison export
    if results["legacy"]["success_count"] > 0 and results["legacy"]["last_context"]:
        context = results["legacy"]["last_context"]

        # Initialize topic manager for auto-tagging with shared LLM
        topic_manager = TopicManager(llm=llm)

        # Analyze and suggest topics
        suggested_topics = []
        suggested_domain = None
        try:
            topic_analysis = await topic_manager.analyze_and_suggest_topics(
                query=query, agent_outputs=context.agent_outputs
            )
            suggested_topics = [s.topic for s in topic_analysis.suggested_topics]
            suggested_domain = topic_analysis.suggested_domain
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Topic analysis failed in comparison mode: {e}"
            )

        # Export with enhanced metadata
        exporter = MarkdownExporter()
        md_path = exporter.export(
            agent_outputs=context.agent_outputs,
            question=query,
            topics=suggested_topics,
            domain=suggested_domain,
        )
        print(f"📄 Comparison results exported to: {md_path}")


def _export_comparison_trace(results: dict, export_trace: str):
    """Export comparison trace data to JSON."""
    comparison_data: Dict[str, Any] = {
        "comparison_timestamp": time.time(),
        "query": "",
        "benchmark_summary": {},
        "modes": {},
    }

    # Get query from successful context
    for mode, result in results.items():
        if result["success_count"] > 0 and result["last_context"]:
            comparison_data["query"] = result["last_context"].query
            break

    # Add benchmark summary
    for mode, result in results.items():
        if result["execution_times"]:
            summary = {
                "total_runs": len(result["execution_times"]),
                "successful_runs": result["success_count"],
                "error_count": result["error_count"],
                "avg_execution_time": statistics.mean(result["execution_times"]),
                "execution_times": result["execution_times"],
            }

            if len(result["execution_times"]) > 1:
                summary["std_dev_time"] = statistics.stdev(result["execution_times"])
                summary["min_time"] = min(result["execution_times"])
                summary["max_time"] = max(result["execution_times"])

            if result["memory_usage"]:
                summary["avg_memory_usage_mb"] = statistics.mean(result["memory_usage"])
                summary["memory_usage"] = result["memory_usage"]

            if result["context_sizes"]:
                summary["avg_context_size"] = statistics.mean(result["context_sizes"])
                summary["context_sizes"] = result["context_sizes"]

            comparison_data["benchmark_summary"][mode] = summary

        # Detailed mode data
        if result["success_count"] > 0 and result["last_context"]:
            context = result["last_context"]
            comparison_data["modes"][mode] = {
                "success": True,
                "agent_outputs": context.agent_outputs,
                "agent_trace": context.agent_trace,
                "context_size_bytes": context.current_size,
                "successful_agents": list(context.successful_agents),
                "failed_agents": list(context.failed_agents),
                "errors": result["errors"],
            }
        else:
            comparison_data["modes"][mode] = {
                "success": False,
                "errors": result["errors"],
            }

    with open(export_trace, "w") as f:
        json.dump(comparison_data, f, indent=2, default=str)

    print(f"🔍 Comparison trace exported to: {export_trace}")


if __name__ == "__main__":  # pragma: no cover
    app()
