"""Command-line interface for running Cognivault agents with specified queries."""

import logging
import typer
import asyncio
import json
import time
import os
from typing import Optional, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from cognivault.config.logging_config import setup_logging
from cognivault.config.openai_config import OpenAIConfig
from cognivault.store.wiki_adapter import MarkdownExporter
from cognivault.store.topic_manager import TopicManager
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMInterface
from cognivault.diagnostics.cli import app as diagnostics_app
from cognivault.diagnostics.visualize_dag import cli_visualize_dag
from cognivault.context import AgentContext

# Import API functions at module level for testing
try:
    from cognivault.api.factory import get_api_mode, initialize_api, shutdown_api
except ImportError:
    # Provide dummy functions if API is not available
    def get_api_mode():  # type: ignore
        return "none"

    async def initialize_api(force_mode=None):  # type: ignore
        return None

    async def shutdown_api():  # type: ignore
        pass


# This module now contains the core orchestration logic
# The main CLI app is defined in __init__.py


def create_llm_instance() -> LLMInterface:
    """Create and configure an LLM instance for use by agents and topic manager."""
    llm_config = OpenAIConfig.load()
    return OpenAIChatLLM(
        api_key=llm_config.api_key,
        model=llm_config.model,
        base_url=llm_config.base_url,
    )


def _validate_langgraph_runtime() -> None:
    """Validate LangGraph runtime compatibility and show helpful warnings."""
    try:
        import langgraph

        version = getattr(langgraph, "__version__", None)

        # Check version compatibility
        if version and not version.startswith("0.6"):
            raise RuntimeError(
                f"LangGraph version {version} may not be compatible. "
                f"Expected version 0.6.x. Consider: pip install langgraph=={version or '0.6.4'}"
            )

        # Test essential imports
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        # Test basic StateGraph creation (lightweight validation)
        from typing import TypedDict

        class TestState(TypedDict):
            test: str

        def test_node(state: TestState) -> TestState:
            return {"test": "working"}

        # Quick validation - create but don't execute
        graph = StateGraph(TestState)
        graph.add_node("test", test_node)
        graph.add_edge("test", END)
        graph.set_entry_point("test")

        # Test compilation
        app = graph.compile()

        # If we get here, LangGraph is functional

    except ImportError as e:
        raise ImportError(f"LangGraph import failed: {e}")
    except Exception as e:
        raise RuntimeError(f"LangGraph runtime validation failed: {e}")


async def run(
    query: str,
    agents: Optional[str] = None,
    log_level: str = "INFO",
    export_md: bool = False,
    trace: bool = False,
    health_check: bool = False,
    dry_run: bool = False,
    export_trace: Optional[str] = None,
    execution_mode: str = "langgraph-real",
    compare_modes: bool = False,
    benchmark_runs: int = 1,
    visualize_dag: Optional[str] = None,
    enable_checkpoints: bool = False,
    thread_id: Optional[str] = None,
    rollback_last_checkpoint: bool = False,
    use_api: bool = False,
    api_mode: Optional[str] = None,
) -> None:
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

    # Determine if API should be used
    use_api_layer = (
        use_api or os.getenv("COGNIVAULT_USE_API", "false").lower() == "true"
    )

    # Set API mode if specified
    if api_mode:
        from cognivault.api.factory import set_api_mode

        original_mode = get_api_mode()
        set_api_mode(api_mode)
        logger.info(f"API mode set to: {api_mode} (was: {original_mode})")

    # Log execution mode
    if use_api_layer:
        current_api_mode = get_api_mode()
        logger.info(f"Using API layer with mode: {current_api_mode}")
        console.print(
            f"🔗 [bold cyan]Using API layer ({current_api_mode} mode)[/bold cyan]"
        )
    else:
        logger.info("Using direct orchestrator execution")
        console.print("🎯 [bold green]Using direct orchestrator execution[/bold green]")

    agents_to_run = [agent.strip() for agent in agents.split(",")] if agents else None
    logger.info(f"[{cli_name}] Received query: %s", query)
    logger.info(
        f"[{cli_name}] Agents to run: %s",
        agents_to_run if agents_to_run else "All agents",
    )

    # Handle DAG visualization (can be used with any execution mode)
    if visualize_dag:
        # Use specified agents or default for visualization
        dag_agents = agents_to_run if agents_to_run else None

        # For Phase 2.1, include historian support
        if dag_agents:
            # Filter to supported agents for Phase 2.1
            supported_agents = {"refiner", "critic", "historian", "synthesis"}
            dag_agents = [
                agent for agent in dag_agents if agent.lower() in supported_agents
            ]

        try:
            cli_visualize_dag(
                agents=dag_agents,
                output=visualize_dag,
                version="Phase 2.1",
                show_state_flow=True,
                show_details=True,
            )
            console.print(
                f"[green]✅ DAG visualization output to: {visualize_dag}[/green]"
            )
        except Exception as e:
            console.print(f"[red]❌ DAG visualization failed: {e}[/red]")
            logger.error(f"DAG visualization error: {e}")

        # If only visualization was requested (no query execution), exit
        if not query or query.strip() == "":
            return

    # Phase 3: Only langgraph-real mode supported
    if execution_mode != "langgraph-real":
        console.print(f"[red]❌ Unsupported execution mode: {execution_mode}[/red]")
        console.print(
            "[yellow]💡 Only 'langgraph-real' mode is supported after Phase 3 legacy cleanup[/yellow]"
        )
        raise typer.Exit(1)

    # Handle comparison mode (disabled after Phase 3 legacy cleanup)
    if compare_modes:
        console.print(
            "[red]❌ Comparison mode disabled after Phase 3 legacy cleanup[/red]"
        )
        console.print(
            "[yellow]💡 Comparison mode will be reimplemented to compare different configurations of langgraph-real mode[/yellow]"
        )
        raise typer.Exit(1)

    logger.info(f"[{cli_name}] Execution mode: {execution_mode}")

    # Create shared LLM instance for agents and topic manager
    llm = create_llm_instance()

    # Create LangGraphOrchestrator (only supported orchestrator after Phase 3)
    try:
        _validate_langgraph_runtime()

        # Validate checkpoint flags
        if rollback_last_checkpoint and not enable_checkpoints:
            console.print(
                "[red]❌ --rollback-last-checkpoint requires --enable-checkpoints[/red]"
            )
            raise typer.Exit(1)

        # Import LangGraphOrchestrator only when needed to avoid module-level LangGraph import
        from cognivault.orchestration.orchestrator import LangGraphOrchestrator

        orchestrator = LangGraphOrchestrator(
            agents_to_run=agents_to_run,
            enable_checkpoints=enable_checkpoints,
            thread_id=thread_id,
        )

    except ImportError as e:
        console.print(f"[red]❌ LangGraph is not installed or incompatible: {e}[/red]")
        console.print("[yellow]💡 Try: pip install langgraph==0.6.4[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ LangGraph runtime error: {e}[/red]")
        console.print(
            "[yellow]💡 Check LangGraph installation with: cognivault diagnostics health[/yellow]"
        )
        raise typer.Exit(1)

    # Handle rollback mode
    if rollback_last_checkpoint:
        if thread_id is None:
            console.print("[red]Error: thread_id is required for rollback mode[/red]")
            raise typer.Exit(1)
        await _run_rollback_mode(orchestrator, console, thread_id)
        return

    # Health check mode - validate agents without execution
    if health_check:
        await _run_health_check(orchestrator, console, agents_to_run)
        return

    # Dry run mode - validate pipeline without execution
    if dry_run:
        await _run_dry_run(orchestrator, console, query, agents_to_run)
        return

    # Execute the pipeline
    if use_api_layer:
        # API execution path
        context = await _run_with_api(
            query, agents_to_run, console, trace, execution_mode, api_mode
        )
        execution_time = time.time() - start_time
    else:
        # Direct orchestrator execution path (existing behavior)
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


# Main command is now handled in __init__.py for proper CLI structure


async def _run_with_api(
    query: str,
    agents_to_run: Optional[List[str]],
    console: Console,
    trace: bool,
    execution_mode: str,
    api_mode: Optional[str],
) -> AgentContext:
    """
    Execute workflow using the API layer.

    Args:
        query: The query to process
        agents_to_run: List of agents to run
        console: Rich console for output
        trace: Whether to show detailed tracing
        execution_mode: Execution mode (for compatibility)
        api_mode: API mode override

    Returns:
        AgentContext with results
    """
    logger = logging.getLogger(__name__)

    try:
        # Import API models when needed
        from cognivault.api.models import WorkflowRequest

        # Initialize API
        if trace:
            console.print(
                f"🔗 [bold]Initializing API layer ({get_api_mode()} mode)...[/bold]"
            )

        api = await initialize_api(force_mode=api_mode)

        if trace:
            console.print("🚀 [bold]Starting workflow execution via API...[/bold]")

        # Create workflow request
        request = WorkflowRequest(
            query=query,
            agents=agents_to_run,
            execution_config={"execution_mode": execution_mode, "trace": trace},
        )

        # Execute workflow
        response = await api.execute_workflow(request)

        if trace:
            console.print(
                f"✅ [bold]Workflow completed with status: {response.status}[/bold]"
            )
            console.print(
                f"⏱️  [bold]Execution time: {response.execution_time_seconds:.2f}s[/bold]"
            )

        # Convert API response back to AgentContext for compatibility
        context = AgentContext(query=query, agent_outputs=response.agent_outputs)

        # Determine the actual API mode used
        actual_api_mode = api_mode or get_api_mode()

        # Add workflow metadata to context for tracing
        if hasattr(context, "metadata"):
            context.metadata.update(
                {
                    "workflow_id": response.workflow_id,
                    "api_execution_time": response.execution_time_seconds,
                    "api_status": response.status,
                    "correlation_id": response.correlation_id,
                    "execution_mode": "api",
                    "api_mode": actual_api_mode,
                }
            )
        else:
            # If metadata doesn't exist, create it
            context.metadata = {
                "workflow_id": response.workflow_id,
                "api_execution_time": response.execution_time_seconds,
                "api_status": response.status,
                "correlation_id": response.correlation_id,
                "execution_mode": "api",
                "api_mode": actual_api_mode,
            }

        if response.status == "failed":
            if response.error_message:
                console.print(
                    f"[red]❌ Workflow failed: {response.error_message}[/red]"
                )
                logger.error(f"API workflow failed: {response.error_message}")
            else:
                console.print("[red]❌ Workflow failed with unknown error[/red]")
                logger.error("API workflow failed with unknown error")

        return context

    except Exception as e:
        console.print(f"[red]❌ API execution failed: {e}[/red]")
        logger.error(f"API execution failed: {e}")

        # Create error context for consistency
        error_context = AgentContext(
            query=query, agent_outputs={"error": f"API execution failed: {e}"}
        )
        actual_api_mode = api_mode or get_api_mode()
        error_context.metadata = {
            "execution_mode": "api",
            "api_mode": actual_api_mode,
            "error": str(e),
        }
        return error_context

    finally:
        # Cleanup API resources
        try:
            if trace:
                console.print("🧹 [bold]Cleaning up API resources...[/bold]")
            await shutdown_api()
        except Exception as e:
            logger.warning(f"API cleanup warning: {e}")


async def _run_health_check(
    orchestrator: Any, console: Console, agents_to_run: Optional[List[str]]
) -> None:
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

            health_table.add_row(str(agent_name).title(), status, details)

        except Exception as e:
            status = "[red]✗ Error[/red]"
            details = f"Health check error: {str(e)}"
            health_table.add_row(str(agent_name).title(), status, details)
            all_healthy = False

    console.print(health_table)

    if all_healthy:
        console.print(
            "\n[green]✅ All agents are healthy and ready for execution[/green]"
        )
    else:
        console.print("\n[red]❌ Some agents failed health checks[/red]")


async def _run_dry_run(
    orchestrator: Any, console: Console, query: str, agents_to_run: Optional[List[str]]
) -> None:
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


def _display_standard_output(
    console: Console, context: Any, execution_time: float
) -> None:
    """Display standard agent outputs with performance metrics."""
    emoji_map = {
        "refiner": "🧠",
        "critic": "🤔",
        "historian": "🕵️",
        "synthesis": "🔗",
    }

    # Map for display names (capitalized for user-friendly output)
    display_name_map = {
        "refiner": "Refiner",
        "critic": "Critic",
        "historian": "Historian",
        "synthesis": "Synthesis",
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
        display_name = display_name_map.get(agent_name, agent_name.capitalize())
        console.print(f"\n{emoji} [bold]{display_name}:[/bold]")
        console.print(output.strip())


def _display_detailed_trace(
    console: Console, context: Any, execution_time: float
) -> None:
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
        "refiner": "🧠",
        "critic": "🤔",
        "historian": "🕵️",
        "synthesis": "🔗",
    }

    # Map for display names (capitalized for user-friendly output)
    display_name_map = {
        "refiner": "Refiner",
        "critic": "Critic",
        "historian": "Historian",
        "synthesis": "Synthesis",
    }

    for agent_name, output in context.agent_outputs.items():
        emoji = emoji_map.get(agent_name, "🧠")
        display_name = display_name_map.get(agent_name, agent_name.capitalize())
        output_panel = Panel(
            output.strip(), title=f"{emoji} {display_name}", border_style="blue"
        )
        console.print(output_panel)


def _export_trace_data(context: Any, export_path: str, execution_time: float) -> None:
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


# NOTE: Comparison mode functions removed in Phase 3A.1
# These functions will be reimplemented to compare different configurations of langgraph-real mode only


# NOTE: Additional comparison mode display functions removed in Phase 3A.1
# These functions will be reimplemented to compare different configurations of langgraph-real mode only


async def _run_rollback_mode(
    orchestrator: Any, console: Console, thread_id: str
) -> None:
    """Handle rollback to latest checkpoint."""
    console.print("🔄 [bold]Rolling back to latest checkpoint[/bold]")

    # Check if this is a LangGraphOrchestrator with rollback capability
    if not hasattr(orchestrator, "rollback_to_checkpoint"):
        console.print("[red]❌ Rollback not supported for this orchestrator[/red]")
        return

    try:
        # Attempt rollback
        restored_context = await orchestrator.rollback_to_checkpoint(
            thread_id=thread_id
        )

        if restored_context:
            console.print(
                "[green]✅ Successfully rolled back to latest checkpoint[/green]"
            )

            # Display checkpoint information
            if thread_id:
                console.print(f"📋 Thread ID: {thread_id}")

            # Show checkpoint history if available
            if hasattr(orchestrator, "get_checkpoint_history"):
                history = orchestrator.get_checkpoint_history(thread_id)
                if history:
                    console.print(
                        f"📊 Checkpoint history: {len(history)} checkpoints found"
                    )

                    # Show latest checkpoint details
                    latest = history[0]
                    console.print(
                        f"🕒 Latest checkpoint: {latest['agent_step']} "
                        f"({latest['timestamp']}, {latest['state_size_bytes']} bytes)"
                    )

            # Display restored agent outputs
            if restored_context.agent_outputs:
                console.print("\n📝 [bold]Restored Agent Outputs[/bold]")
                emoji_map = {
                    "Refiner": "🧠",
                    "refiner": "🧠",
                    "Critic": "🤔",
                    "critic": "🤔",
                    "Historian": "🕵️",
                    "historian": "🕵️",
                    "Synthesis": "🔗",
                    "synthesis": "🔗",
                }

                for agent_name, output in restored_context.agent_outputs.items():
                    emoji = emoji_map.get(agent_name, "🧠")
                    console.print(f"\n{emoji} [bold]{agent_name.title()}:[/bold]")
                    console.print(
                        output.strip()[:200] + "..."
                        if len(output) > 200
                        else output.strip()
                    )
            else:
                console.print("📝 No agent outputs found in restored checkpoint")

        else:
            console.print("[yellow]⚠️ No checkpoint found to rollback to[/yellow]")
            if thread_id:
                console.print(f"Thread ID: {thread_id}")
            else:
                console.print(
                    "No thread ID specified - use --thread-id to specify a conversation"
                )

    except Exception as e:
        console.print(f"[red]❌ Rollback failed: {e}[/red]")


# Module contains shared functions for CLI operations
# Entry point is now in __init__.py
