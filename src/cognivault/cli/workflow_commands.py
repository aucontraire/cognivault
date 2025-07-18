"""
CLI commands for declarative workflow operations.

This module provides comprehensive CLI interface for workflow definition,
validation, execution, and management in the CogniVault ecosystem.
"""

import json
import os
import time
import yaml
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from cognivault.context import AgentContext
from cognivault.workflows import WorkflowDefinition
from cognivault.workflows.executor import DeclarativeOrchestrator


# Initialize CLI components
workflow_app = typer.Typer(help="Declarative workflow operations")
console = Console()


@workflow_app.command("run")
def run_workflow(
    workflow_file: str = typer.Argument(..., help="Path to workflow definition file"),
    query: str = typer.Option(..., "--query", "-q", help="Query to process"),
    output_format: str = typer.Option(
        "json", "--format", "-f", help="Output format (json, yaml, table)"
    ),
    save_result: Optional[str] = typer.Option(
        None, "--save", "-s", help="Save result to file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Execute a declarative workflow with the given query.

    Examples:
        cognivault workflow run examples/complex_analysis.yaml -q "Analyze this data"
        cognivault workflow run workflows/decision_flow.yaml -q "What should I do?" --format table
    """
    try:
        import asyncio

        asyncio.run(
            _run_workflow_async(
                workflow_file, query, output_format, save_result, verbose
            )
        )
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # In test environment, run directly
            import asyncio

            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                _run_workflow_async(
                    workflow_file, query, output_format, save_result, verbose
                )
            )
        else:
            raise


async def _run_workflow_async(
    workflow_file: str,
    query: str,
    output_format: str,
    save_result: Optional[str],
    verbose: bool,
):
    """Async implementation of workflow execution."""
    try:
        # Load workflow definition
        if verbose:
            console.print(f"[blue]Loading workflow from: {workflow_file}[/blue]")

        workflow = _load_workflow_file(workflow_file)

        if verbose:
            console.print(
                f"[green]Loaded workflow: {workflow.name} v{workflow.version}[/green]"
            )
            console.print(f"[dim]Created by: {workflow.created_by}[/dim]")
            console.print(
                f"[dim]Nodes: {len(workflow.nodes)}, Schema: v{workflow.workflow_schema_version}[/dim]"
            )

        # Create initial context
        initial_context = AgentContext(query=query)

        # Execute workflow
        orchestrator = DeclarativeOrchestrator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Executing workflow...", total=None)
            result = await orchestrator.execute_workflow(workflow, initial_context)
            progress.update(task, description="Workflow completed!")

        # Display results
        _display_workflow_result(result, output_format, verbose)

        # Save result if requested
        if save_result:
            _save_workflow_result(result, save_result, output_format)
            console.print(f"[green]Result saved to: {save_result}[/green]")

    except Exception as e:
        console.print(f"[red]Error executing workflow: {str(e)}[/red]")
        raise typer.Exit(1)


@workflow_app.command("validate")
def validate_workflow(
    workflow_file: str = typer.Argument(..., help="Path to workflow definition file"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose validation output"
    ),
):
    """
    Validate a workflow definition without executing it.

    Examples:
        cognivault workflow validate examples/complex_analysis.yaml
        cognivault workflow validate workflows/decision_flow.yaml --verbose
    """
    try:
        import asyncio

        asyncio.run(_validate_workflow_async(workflow_file, verbose))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # In test environment, run directly
            import asyncio

            loop = asyncio.get_event_loop()
            loop.run_until_complete(_validate_workflow_async(workflow_file, verbose))
        else:
            raise


async def _validate_workflow_async(workflow_file: str, verbose: bool):
    """Async implementation of workflow validation."""
    try:
        # Load workflow definition
        workflow = _load_workflow_file(workflow_file)

        # Validate workflow
        orchestrator = DeclarativeOrchestrator()
        validation_result = await orchestrator.validate_workflow(workflow)

        # Display validation results
        if validation_result["valid"]:
            console.print("[green]✓ Workflow validation passed[/green]")

            if verbose:
                table = Table(title="Workflow Information")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")

                table.add_row("Name", workflow.name)
                table.add_row("Version", workflow.version)
                table.add_row("Created By", workflow.created_by)
                table.add_row("Nodes", str(validation_result["node_count"]))
                table.add_row("Edges", str(validation_result["edge_count"]))
                table.add_row("Schema Version", workflow.workflow_schema_version)

                console.print(table)
        else:
            console.print("[red]✗ Workflow validation failed[/red]")

            for error in validation_result["errors"]:
                console.print(f"  [red]• {error}[/red]")

            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error validating workflow: {str(e)}[/red]")
        raise typer.Exit(1)


@workflow_app.command("list")
def list_workflows(
    directory: str = typer.Option(
        "examples", "--dir", "-d", help="Directory to search for workflows"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table, json)"
    ),
):
    """
    List available workflow examples and definitions.

    Examples:
        cognivault workflow list
        cognivault workflow list --dir my_workflows --format json
    """
    try:
        workflows = _find_workflow_files(directory)

        if not workflows:
            console.print(f"[yellow]No workflow files found in: {directory}[/yellow]")
            return

        if format == "table":
            _display_workflows_table(workflows)
        elif format == "json":
            _display_workflows_json(workflows)
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error listing workflows: {str(e)}[/red]")
        raise typer.Exit(1)


@workflow_app.command("show")
def show_workflow(
    workflow_file: str = typer.Argument(..., help="Path to workflow definition file"),
    format: str = typer.Option(
        "yaml", "--format", "-f", help="Display format (yaml, json)"
    ),
):
    """
    Display workflow definition in readable format.

    Examples:
        cognivault workflow show examples/complex_analysis.yaml
        cognivault workflow show workflows/decision_flow.yaml --format json
    """
    try:
        workflow = _load_workflow_file(workflow_file)

        if format.lower() == "yaml":
            content = workflow.export("yaml")
            syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
        elif format.lower() == "json":
            content = workflow.export("json")
            syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            raise typer.Exit(1)

        panel = Panel(
            syntax,
            title=f"[cyan]{workflow.name} v{workflow.version}[/cyan]",
            subtitle=f"[dim]Created by: {workflow.created_by}[/dim]",
        )
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Error displaying workflow: {str(e)}[/red]")
        raise typer.Exit(1)


@workflow_app.command("export")
def export_workflow(
    workflow_file: str = typer.Argument(..., help="Path to workflow definition file"),
    output_file: str = typer.Argument(..., help="Output file path"),
    format: str = typer.Option(
        "json", "--format", "-f", help="Export format (json, yaml)"
    ),
    include_snapshot: bool = typer.Option(
        False, "--snapshot", help="Include composition snapshot"
    ),
):
    """
    Export workflow definition to file with optional composition metadata.

    Examples:
        cognivault workflow export examples/complex_analysis.yaml output.json
        cognivault workflow export workflows/decision_flow.yaml output.yaml --format yaml --snapshot
    """
    try:
        import asyncio

        asyncio.run(
            _export_workflow_async(workflow_file, output_file, format, include_snapshot)
        )
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # In test environment, run directly
            import asyncio

            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                _export_workflow_async(
                    workflow_file, output_file, format, include_snapshot
                )
            )
        else:
            raise


async def _export_workflow_async(
    workflow_file: str,
    output_file: str,
    format: str,
    include_snapshot: bool,
):
    """Async implementation of workflow export."""
    try:
        # Load workflow definition
        workflow = _load_workflow_file(workflow_file)

        if include_snapshot:
            # Export with composition snapshot
            orchestrator = DeclarativeOrchestrator()
            snapshot = await orchestrator.export_workflow_snapshot(workflow)
            content = (
                json.dumps(snapshot, indent=2)
                if format == "json"
                else workflow.export(format)
            )
        else:
            # Export workflow definition only
            content = workflow.export(format)

        # Write to output file
        with open(output_file, "w") as f:
            f.write(content)

        console.print(f"[green]Workflow exported to: {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error exporting workflow: {str(e)}[/red]")
        raise typer.Exit(1)


# Helper functions


def _load_workflow_file(file_path: str) -> WorkflowDefinition:
    """Load workflow definition from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Workflow file not found: {file_path}")

    if file_path.endswith((".yaml", ".yml")):
        return WorkflowDefinition.from_yaml_file(file_path)
    elif file_path.endswith(".json"):
        return WorkflowDefinition.from_json_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def _find_workflow_files(directory: str) -> List[dict]:
    """Find workflow files in directory."""
    if not os.path.exists(directory):
        return []

    workflows = []
    for file_path in Path(directory).rglob("*.yaml"):
        try:
            workflow = WorkflowDefinition.from_yaml_file(str(file_path))
            workflows.append(
                {
                    "file": str(file_path),
                    "name": workflow.name,
                    "version": workflow.version,
                    "created_by": workflow.created_by,
                    "nodes": len(workflow.nodes),
                    "description": workflow.description or "No description",
                }
            )
        except Exception:
            # Skip invalid workflow files
            continue

    for file_path in Path(directory).rglob("*.json"):
        try:
            workflow = WorkflowDefinition.from_json_file(str(file_path))
            workflows.append(
                {
                    "file": str(file_path),
                    "name": workflow.name,
                    "version": workflow.version,
                    "created_by": workflow.created_by,
                    "nodes": len(workflow.nodes),
                    "description": workflow.description or "No description",
                }
            )
        except Exception:
            # Skip invalid workflow files
            continue

    return workflows


def _display_workflows_table(workflows: List[dict]):
    """Display workflows in table format."""
    table = Table(title="Available Workflows")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Created By", style="white")
    table.add_column("Nodes", justify="right", style="yellow")
    table.add_column("File", style="dim")

    for workflow in workflows:
        table.add_row(
            workflow["name"],
            workflow["version"],
            workflow["created_by"],
            str(workflow["nodes"]),
            workflow["file"],
        )

    console.print(table)


def _display_workflows_json(workflows: List[dict]):
    """Display workflows in JSON format."""
    console.print(json.dumps(workflows, indent=2))


def _display_workflow_result(result, output_format: str, verbose: bool):
    """Display workflow execution result."""
    if output_format == "json":
        console.print(json.dumps(result.to_dict(), indent=2))
    elif output_format == "table":
        _display_result_table(result, verbose)
    else:
        console.print(f"[red]Unsupported output format: {output_format}[/red]")


def _display_result_table(result, verbose: bool):
    """Display result in table format."""
    # Execution summary
    table = Table(title="Workflow Execution Result")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Success", "✓ Yes" if result.success else "✗ No")
    table.add_row("Execution Time", f"{result.execution_time_seconds:.2f}s")
    table.add_row("Nodes Executed", str(len(result.node_execution_order)))

    if result.error_message:
        table.add_row("Error", result.error_message)

    console.print(table)

    if verbose and result.node_execution_order:
        # Node execution order
        order_table = Table(title="Node Execution Order")
        order_table.add_column("Step", justify="right", style="cyan")
        order_table.add_column("Node ID", style="white")

        for i, node_id in enumerate(result.node_execution_order, 1):
            order_table.add_row(str(i), node_id)

        console.print(order_table)


def _save_workflow_result(result, file_path: str, format: str):
    """Save workflow result to file."""
    content = json.dumps(result.to_dict(), indent=2)

    with open(file_path, "w") as f:
        f.write(content)


# Test helper functions for the CLI commands


async def run_workflow_test_helper(
    query: str,
    workflow_file: str,
    agents: Optional[List[str]],
    trace: bool,
    dry_run: bool,
    export_trace: Optional[str],
    log_level: str,
) -> None:
    """Execute a workflow from CLI parameters (helper function)."""
    console = Console()

    try:
        # Load workflow definition
        workflow_def = load_workflow_definition(workflow_file)

        if dry_run:
            console.print("🧪 [bold]Running in dry-run mode - validation only[/bold]")
            orchestrator = DeclarativeOrchestrator()
            await orchestrator.validate_workflow(workflow_def)
            console.print("[green]✅ Workflow validation successful[/green]")
            return

        # Create orchestrator and execute
        orchestrator = DeclarativeOrchestrator()

        # Build execution config
        execution_config = {"trace": trace, "log_level": log_level}
        if agents:
            execution_config["agents"] = agents

        # Execute workflow
        context = await orchestrator.run(query, execution_config)

        # Display results
        execution_time = 2.5  # Placeholder
        display_execution_results(context, execution_time, trace)

        # Export trace if requested
        if export_trace:
            export_trace_data(context, export_trace, execution_time)
            console.print(f"📊 [bold]Trace exported to: {export_trace}[/bold]")

    except FileNotFoundError:
        console.print(f"[red]❌ Workflow file not found: {workflow_file}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]❌ Workflow execution failed: {e}[/red]")
        raise SystemExit(1)


async def validate_workflow_test_helper(workflow_file: str, strict: bool) -> None:
    """Validate a workflow definition (helper function)."""
    console = Console()

    try:
        workflow_def = load_workflow_definition(workflow_file)
        orchestrator = DeclarativeOrchestrator()
        await orchestrator.validate_workflow(workflow_def)
        console.print("[green]✅ Workflow validation successful[/green]")
    except Exception as e:
        console.print(f"[red]❌ Workflow validation failed: {e}[/red]")
        if strict:
            raise SystemExit(1)


def list_workflows_test_helper() -> None:
    """List available workflow examples (helper function)."""
    console = Console()

    # Check for examples directory
    examples_dir = Path("src/cognivault/workflows/examples")
    if not examples_dir.exists():
        console.print("[yellow]No workflow examples found[/yellow]")
        return

    # Find workflow files
    workflow_files = list(examples_dir.glob("*.yaml")) + list(
        examples_dir.glob("*.json")
    )

    if not workflow_files:
        console.print("[yellow]No workflow files found in examples[/yellow]")
        return

    # Display workflow list
    table = Table(title="Available Workflows")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    for file_path in workflow_files:
        try:
            workflow_def = load_workflow_definition(str(file_path))
            table.add_row(
                workflow_def.name, workflow_def.description or "No description"
            )
        except Exception:
            table.add_row(file_path.stem, "Failed to load")

    console.print(table)


def show_workflow_test_helper(workflow_file: str, detailed: bool) -> None:
    """Show workflow definition details (helper function)."""
    console = Console()

    try:
        workflow_def = load_workflow_definition(workflow_file)
        display_workflow_summary(workflow_def, detailed)
    except Exception as e:
        console.print(f"[red]❌ Failed to load workflow: {e}[/red]")
        raise SystemExit(1)


def export_workflow_test_helper(
    source_file: str, output_file: str, metadata: Dict[str, Any], include_runtime: bool
) -> None:
    """Export workflow definition with optional runtime information (helper function)."""
    console = Console()

    try:
        workflow_def = load_workflow_definition(source_file)

        # Add runtime metadata if requested
        if include_runtime:
            import time

            runtime_metadata = {
                "exported_at": time.time(),
                "exported_by": os.getenv("USER", "unknown"),
                "runtime_version": "1.0.0",
            }
            workflow_def.metadata.update(runtime_metadata)

        # Add custom metadata
        workflow_def.metadata.update(metadata)

        # Export using composer
        from cognivault.workflows.composer import DagComposer

        composer = DagComposer()
        composer.export_workflow_snapshot(workflow_def, output_file)

        console.print(f"[green]✅ Workflow exported to: {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Export failed: {e}[/red]")
        raise SystemExit(1)


def load_workflow_definition(file_path: str) -> WorkflowDefinition:
    """Load workflow definition from file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {file_path}")

    # Determine file format
    if path.suffix.lower() in [".yaml", ".yml"]:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return WorkflowDefinition.from_dict(data)


def display_workflow_summary(workflow_def: WorkflowDefinition, detailed: bool) -> None:
    """Display workflow summary information."""
    console = Console()

    # Basic information
    info_panel = Panel(
        f"[bold]Name:[/bold] {workflow_def.name}\n"
        f"[bold]Version:[/bold] {workflow_def.version}\n"
        f"[bold]Created by:[/bold] {workflow_def.created_by}\n"
        f"[bold]Description:[/bold] {workflow_def.description or 'No description'}\n"
        f"[bold]Tags:[/bold] {', '.join(workflow_def.tags) if workflow_def.tags else 'None'}",
        title="Workflow Information",
        border_style="blue",
    )
    console.print(info_panel)

    if detailed:
        # Node information
        node_table = Table(title="Nodes")
        node_table.add_column("ID", style="cyan")
        node_table.add_column("Type", style="white")
        node_table.add_column("Category", style="yellow")

        for node in workflow_def.nodes:
            node_table.add_row(node.node_id, node.node_type, node.category)

        console.print(node_table)

        # Flow information
        if workflow_def.flow.edges:
            edge_table = Table(title="Edges")
            edge_table.add_column("From", style="cyan")
            edge_table.add_column("To", style="white")
            edge_table.add_column("Type", style="yellow")

            for edge in workflow_def.flow.edges:
                edge_table.add_row(edge.from_node, edge.to_node, edge.edge_type)

            console.print(edge_table)


def display_execution_results(
    context: AgentContext, execution_time: float, trace: bool
) -> None:
    """Display workflow execution results."""
    console = Console()

    # Performance summary
    console.print(f"\n⏱️  [bold]Workflow completed in {execution_time:.2f}s[/bold]")
    console.print(
        f"✅ [green]{len(context.successful_agents)} agents completed successfully[/green]"
    )

    if context.failed_agents:
        console.print(f"❌ [red]{len(context.failed_agents)} agents failed[/red]")

    # Agent outputs
    console.print("\n📝 [bold]Agent Outputs[/bold]")
    for agent_name, output in context.agent_outputs.items():
        output_panel = Panel(
            output.strip()[:500] + ("..." if len(output) > 500 else ""),
            title=f"🧠 {agent_name}",
            border_style="blue",
        )
        console.print(output_panel)

    if trace:
        # Additional trace information
        if hasattr(context, "execution_state"):
            console.print("\n🔍 [bold]Execution Trace[/bold]")
            for key, value in context.execution_state.items():
                console.print(f"  [bold]{key}:[/bold] {value}")


def export_trace_data(
    context: AgentContext, output_path: str, execution_time: float
) -> None:
    """Export execution trace data to file."""
    trace_data = {
        "workflow_id": getattr(context, "context_id", "unknown"),
        "execution_time_seconds": execution_time,
        "query": context.query,
        "agent_outputs": context.agent_outputs,
        "successful_agents": list(context.successful_agents),
        "failed_agents": list(context.failed_agents),
        "execution_state": getattr(context, "execution_state", {}),
        "timestamp": time.time(),
    }

    with open(output_path, "w") as f:
        json.dump(trace_data, f, indent=2, default=str)
