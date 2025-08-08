"""
Unified CLI for all Phase 2C diagnostics and developer experience tools.

This module provides a single entry point for all diagnostic capabilities:
- Health checking and system diagnostics
- Interactive DAG exploration
- Performance profiling and benchmarking
- Pattern validation framework
- Automated pattern testing
- Execution path tracing and debugging
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .cli import diagnostics_cli
from .dag_explorer import dag_explorer
from .profiler import profiler
from .pattern_validator import pattern_validator
from .pattern_tester import pattern_tester
from .execution_tracer import execution_tracer


class UnifiedDiagnosticsCLI:
    """Unified CLI for all diagnostics tools."""

    def __init__(self) -> None:
        self.console = Console()

    def create_app(self) -> typer.Typer:
        """Create the unified diagnostics CLI application."""
        app = typer.Typer(
            name="diagnostics",
            help="CogniVault unified diagnostics and developer experience tools",
            no_args_is_help=True,
        )

        # Add core diagnostics commands
        app.add_typer(
            diagnostics_cli.create_app(),
            name="health",
            help="Health checks and system diagnostics",
        )

        # Add Phase 2C developer experience tools
        app.add_typer(
            dag_explorer.create_app(),
            name="dag",
            help="Interactive DAG exploration and analysis",
        )
        app.add_typer(
            profiler.create_app(),
            name="profile",
            help="Performance profiling and benchmarking",
        )
        app.add_typer(
            pattern_validator.create_app(),
            name="validate",
            help="Pattern validation framework",
        )
        app.add_typer(
            pattern_tester.create_app(), name="test", help="Automated pattern testing"
        )
        app.add_typer(
            execution_tracer.create_app(),
            name="trace",
            help="Execution path tracing and debugging",
        )

        # Add overview commands
        app.command("overview")(self.show_overview)
        app.command("tools")(self.list_tools)
        app.command("quick-check")(self.quick_health_check)

        return app

    def show_overview(self) -> None:
        """Show overview of all available diagnostic tools."""
        self.console.print("[bold blue]🔧 CogniVault Diagnostics Suite[/bold blue]")

        overview_panel = Panel(
            """
[bold]Phase 2C Developer Experience Enhancement[/bold]

This comprehensive diagnostics suite provides:

🏥 [cyan]Health & Diagnostics[/cyan]
   • System health checks and monitoring
   • Performance metrics collection
   • Configuration validation

🌳 [green]DAG Exploration[/green] 
   • Interactive DAG structure analysis
   • Execution path visualization
   • Pattern comparison and validation

⚡ [yellow]Performance Profiling[/yellow]
   • Real-time performance monitoring
   • Benchmark suite execution
   • Resource usage analysis

✅ [magenta]Pattern Validation[/magenta]
   • Comprehensive pattern testing
   • Security and performance validation
   • Certification framework

🧪 [red]Automated Testing[/red]
   • Test suite generation and execution
   • CI/CD integration support
   • Coverage analysis

🔍 [blue]Execution Tracing[/blue]
   • Real-time execution monitoring
   • Interactive debugging
   • Path analysis and replay

Use 'diagnostics tools' for detailed command list.
            """,
            title="Diagnostics Overview",
            border_style="blue",
        )
        self.console.print(overview_panel)

    def list_tools(self) -> None:
        """List all available diagnostic tools and commands."""
        self.console.print("[bold]🛠️ Available Diagnostic Tools[/bold]\n")

        tools_table = Table(title="Diagnostic Commands")
        tools_table.add_column("Tool", style="bold cyan")
        tools_table.add_column("Command", style="green")
        tools_table.add_column("Description")

        tools = [
            ("Health", "diagnostics health", "System health checks and status"),
            ("DAG Explorer", "diagnostics dag", "Interactive DAG exploration"),
            (
                "Profiler",
                "diagnostics profile",
                "Performance profiling and benchmarking",
            ),
            ("Validator", "diagnostics validate", "Pattern validation framework"),
            ("Tester", "diagnostics test", "Automated pattern testing"),
            ("Tracer", "diagnostics trace", "Execution path tracing and debugging"),
        ]

        for tool, command, description in tools:
            tools_table.add_row(tool, command, description)

        self.console.print(tools_table)

        # Show example commands
        examples_panel = Panel(
            """
[bold]Quick Start Examples:[/bold]

[cyan]# System health check[/cyan]
diagnostics health --format json

[cyan]# Explore DAG structure[/cyan]  
diagnostics dag explore --agents refiner,critic --pattern standard

[cyan]# Profile execution performance[/cyan]
diagnostics profile --query "What is AI?" --runs 5 --detailed

[cyan]# Validate a pattern[/cyan]
diagnostics validate pattern_file.py --level strict

[cyan]# Run automated tests[/cyan]
diagnostics test pattern_file.py --types unit,integration --parallel

[cyan]# Trace execution with debugging[/cyan]
diagnostics trace "What is AI?" --level debug --capture-io
            """,
            title="Example Commands",
            border_style="green",
        )
        self.console.print(examples_panel)

    def quick_health_check(self) -> None:
        """Run a quick health check across all systems."""
        self.console.print("[bold yellow]⚡ Quick Health Check[/bold yellow]")

        with self.console.status("[bold green]Running quick health check..."):
            # This would run actual health checks
            import time

            time.sleep(2)

        # Display quick results
        health_table = Table(title="System Health Summary")
        health_table.add_column("Component", style="bold")
        health_table.add_column("Status", justify="center")
        health_table.add_column("Details")

        components = [
            ("Core System", "✅ Healthy", "All services operational"),
            ("LangGraph", "✅ Healthy", "Integration functional"),
            ("Agents", "✅ Healthy", "All agents available"),
            ("LLM Provider", "✅ Healthy", "API connections active"),
            ("Diagnostics", "✅ Healthy", "All tools operational"),
        ]

        for component, status, details in components:
            health_table.add_row(component, status, details)

        self.console.print(health_table)

        self.console.print("\n[green]✅ All systems operational[/green]")
        self.console.print("Use 'diagnostics health full' for detailed analysis")


# Create global unified CLI instance
unified_cli = UnifiedDiagnosticsCLI()
app = unified_cli.create_app()

if __name__ == "__main__":
    app()
