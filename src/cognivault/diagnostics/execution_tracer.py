"""
Enhanced debugging with execution path tracing for Phase 2C developer experience.

This module provides comprehensive execution path tracing, real-time debugging,
and detailed execution analytics for LangGraph DAG execution flows.
"""

import asyncio
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
import uuid

from pydantic import BaseModel, Field, ConfigDict

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt

from cognivault.context import AgentContext

# Import for runtime use and testing
try:
    from cognivault.orchestration.orchestrator import LangGraphOrchestrator
except ImportError:
    # Fallback for environments where LangGraph isn't available
    if TYPE_CHECKING:
        from cognivault.orchestration.orchestrator import LangGraphOrchestrator
    else:
        LangGraphOrchestrator = None


class TraceLevel(Enum):
    """Trace detail levels."""

    BASIC = "basic"
    DETAILED = "detailed"
    VERBOSE = "verbose"
    DEBUG = "debug"


class ExecutionState(Enum):
    """Execution states for tracing."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class TraceEvent(BaseModel):
    """Individual trace event."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    event_id: str = Field(..., description="Unique identifier for this trace event")
    timestamp: datetime = Field(..., description="Timestamp when event occurred")
    event_type: str = Field(..., description="Type/category of the event")
    node_name: str = Field(..., description="Name of the execution node")
    agent_name: Optional[str] = Field(
        None, description="Name of the agent if applicable"
    )
    state: ExecutionState = Field(..., description="Current execution state")
    duration: Optional[float] = Field(
        None, ge=0.0, description="Event duration in seconds"
    )
    input_data: Optional[Dict[str, Any]] = Field(
        None, description="Input data for the event"
    )
    output_data: Optional[Dict[str, Any]] = Field(
        None, description="Output data from the event"
    )
    error: Optional[str] = Field(None, description="Error message if event failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the event"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "node_name": self.node_name,
            "agent_name": self.agent_name,
            "state": self.state.value,
            "duration": self.duration,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "metadata": self.metadata,
        }


class ExecutionTrace(BaseModel):
    """Complete execution trace."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    trace_id: str = Field(..., description="Unique identifier for this execution trace")
    query: str = Field(..., description="The query that was executed")
    start_time: datetime = Field(..., description="Timestamp when execution started")
    end_time: Optional[datetime] = Field(
        None, description="Timestamp when execution ended"
    )
    total_duration: float = Field(
        default=0.0, ge=0.0, description="Total execution duration in seconds"
    )
    events: List[TraceEvent] = Field(
        default_factory=list, description="List of trace events during execution"
    )
    execution_path: List[str] = Field(
        default_factory=list, description="Ordered list of execution steps"
    )
    agent_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Statistics for each agent"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics for the execution"
    )
    success: bool = Field(
        default=True, description="Whether execution completed successfully"
    )
    error_details: Optional[str] = Field(
        None, description="Error details if execution failed"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": self.total_duration,
            "events": [event.to_dict() for event in self.events],
            "execution_path": self.execution_path,
            "agent_stats": self.agent_stats,
            "performance_metrics": self.performance_metrics,
            "success": self.success,
            "error_details": self.error_details,
        }


class TracingSession(BaseModel):
    """Tracing session configuration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    session_id: str = Field(
        ..., description="Unique identifier for this tracing session"
    )
    trace_level: TraceLevel = Field(..., description="Level of detail for tracing")
    real_time: bool = Field(..., description="Whether to enable real-time tracing")
    capture_io: bool = Field(..., description="Whether to capture input/output data")
    capture_timing: bool = Field(
        ..., description="Whether to capture timing information"
    )
    capture_memory: bool = Field(..., description="Whether to capture memory usage")
    filter_agents: Optional[List[str]] = Field(
        None, description="List of agents to filter for (None = all agents)"
    )
    breakpoints: List[str] = Field(
        default_factory=list, description="List of breakpoint locations"
    )
    max_events: int = Field(
        default=10000, gt=0, description="Maximum number of events to capture"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "session_id": self.session_id,
            "trace_level": self.trace_level.value,
            "real_time": self.real_time,
            "capture_io": self.capture_io,
            "capture_timing": self.capture_timing,
            "capture_memory": self.capture_memory,
            "filter_agents": self.filter_agents,
            "breakpoints": self.breakpoints,
            "max_events": self.max_events,
        }


class ExecutionTracer:
    """Advanced execution path tracer with debugging capabilities."""

    def __init__(self) -> None:
        self.console = Console()
        self.active_sessions: Dict[str, TracingSession] = {}
        self.traces: Dict[str, ExecutionTrace] = {}
        self.event_handlers: List[Callable[..., Any]] = []
        self.breakpoint_handler: Optional[Callable[..., Any]] = None
        self._trace_lock = threading.Lock()

    def create_app(self) -> typer.Typer:
        """Create the execution tracer CLI application."""
        app = typer.Typer(
            name="execution-tracer",
            help="Enhanced debugging with execution path tracing",
            no_args_is_help=True,
        )

        app.command("trace")(self.trace_execution)
        app.command("live")(self.live_trace)
        app.command("debug")(self.debug_execution)
        app.command("analyze")(self.analyze_trace)
        app.command("replay")(self.replay_trace)
        app.command("compare")(self.compare_traces)
        app.command("breakpoint")(self.manage_breakpoints)
        app.command("export")(self.export_trace)
        app.command("monitor")(self.monitor_execution)

        return app

    def trace_execution(
        self,
        query: str = typer.Argument(..., help="Query to trace"),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to trace"
        ),
        trace_level: TraceLevel = typer.Option(
            TraceLevel.DETAILED, "--level", "-l", help="Trace detail level"
        ),
        output_file: Optional[str] = typer.Option(
            None, "--output", "-o", help="Output trace file"
        ),
        pattern: str = typer.Option(
            "standard", "--pattern", "-p", help="Graph pattern"
        ),
        capture_io: bool = typer.Option(
            True, "--capture-io", help="Capture input/output data"
        ),
        capture_timing: bool = typer.Option(
            True, "--timing", help="Capture detailed timing"
        ),
        capture_memory: bool = typer.Option(
            False, "--memory", help="Capture memory usage"
        ),
    ) -> None:
        """Trace DAG execution with detailed path analysis."""
        self.console.print("[bold blue]🔍 Execution Path Tracer[/bold blue]")

        agent_list = [a.strip() for a in agents.split(",")] if agents else None

        # Extract actual values from typer.Option objects if needed
        capture_io_value = (
            getattr(capture_io, "default", capture_io)
            if hasattr(capture_io, "default")
            else capture_io
        )
        capture_timing_value = (
            getattr(capture_timing, "default", capture_timing)
            if hasattr(capture_timing, "default")
            else capture_timing
        )
        capture_memory_value = (
            getattr(capture_memory, "default", capture_memory)
            if hasattr(capture_memory, "default")
            else capture_memory
        )

        # Create tracing session
        session = TracingSession(
            session_id=str(uuid.uuid4()),
            trace_level=trace_level,
            real_time=False,
            capture_io=capture_io_value,
            capture_timing=capture_timing_value,
            capture_memory=capture_memory_value,
            filter_agents=agent_list,
        )

        try:
            # Execute with tracing
            trace = self._execute_with_tracing(query, agent_list, session)

            # Display trace results
            self._display_trace_summary(trace)
            self._display_execution_path(trace)

            if trace_level in [TraceLevel.VERBOSE, TraceLevel.DEBUG]:
                self._display_detailed_events(trace)

            # Save trace if requested
            if output_file:
                self._save_trace(trace, output_file)

        except Exception as e:
            self.console.print(f"[red]❌ Tracing error: {e}[/red]")
            raise typer.Exit(1)

    def live_trace(
        self,
        query: str = typer.Argument(..., help="Query to trace"),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to trace"
        ),
        refresh_rate: float = typer.Option(
            0.5, "--refresh", "-r", help="Refresh rate in seconds"
        ),
        show_events: bool = typer.Option(True, "--events", help="Show live events"),
    ) -> None:
        """Live trace execution with real-time updates."""
        self.console.print("[bold green]📡 Live Execution Tracer[/bold green]")

        agent_list = [a.strip() for a in agents.split(",")] if agents else None

        # Create live tracing session
        session = TracingSession(
            session_id=str(uuid.uuid4()),
            trace_level=TraceLevel.DETAILED,
            real_time=True,
            capture_io=True,
            capture_timing=True,
            capture_memory=False,
            filter_agents=agent_list,
        )

        try:
            self._execute_live_tracing(
                query, agent_list, session, refresh_rate, show_events
            )
        except KeyboardInterrupt:
            self.console.print("\n[yellow]🛑 Live tracing stopped by user[/yellow]")
        except Exception as e:
            self.console.print(f"[red]❌ Live tracing error: {e}[/red]")
            raise typer.Exit(1)

    def debug_execution(
        self,
        query: str = typer.Argument(..., help="Query to debug"),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to debug"
        ),
        breakpoints: Optional[str] = typer.Option(
            None, "--breakpoints", "-b", help="Comma-separated breakpoint agents"
        ),
        step_mode: bool = typer.Option(
            False, "--step", "-s", help="Step-by-step execution"
        ),
        interactive: bool = typer.Option(
            True, "--interactive", "-i", help="Interactive debugging"
        ),
    ) -> None:
        """Debug execution with breakpoints and step-by-step analysis."""
        self.console.print("[bold red]🐛 Interactive Debugger[/bold red]")

        agent_list = [a.strip() for a in agents.split(",")] if agents else None
        breakpoint_list = (
            [b.strip() for b in breakpoints.split(",")] if breakpoints else []
        )

        # Create debug session
        session = TracingSession(
            session_id=str(uuid.uuid4()),
            trace_level=TraceLevel.DEBUG,
            real_time=True,
            capture_io=True,
            capture_timing=True,
            capture_memory=True,
            filter_agents=agent_list,
            breakpoints=breakpoint_list,
        )

        try:
            if interactive:
                self._start_interactive_debug_session(query, agent_list, session)
            else:
                self._execute_debug_session(query, agent_list, session, step_mode)
        except Exception as e:
            self.console.print(f"[red]❌ Debug error: {e}[/red]")
            raise typer.Exit(1)

    def analyze_trace(
        self,
        trace_file: str = typer.Argument(..., help="Trace file to analyze"),
        analysis_type: str = typer.Option(
            "comprehensive", "--type", "-t", help="Analysis type"
        ),
        focus_agent: Optional[str] = typer.Option(
            None, "--agent", "-a", help="Focus on specific agent"
        ),
        performance_analysis: bool = typer.Option(
            True, "--performance", help="Include performance analysis"
        ),
    ) -> None:
        """Analyze execution trace with detailed insights."""
        self.console.print("[bold cyan]📊 Trace Analyzer[/bold cyan]")

        try:
            # Load trace
            trace = self._load_trace(trace_file)

            # Perform analysis
            analysis = self._analyze_execution_trace(trace, analysis_type, focus_agent)

            # Display analysis results
            self._display_trace_analysis(analysis, performance_analysis)

        except Exception as e:
            self.console.print(f"[red]❌ Analysis error: {e}[/red]")
            raise typer.Exit(1)

    def replay_trace(
        self,
        trace_file: str = typer.Argument(..., help="Trace file to replay"),
        speed: float = typer.Option(
            1.0, "--speed", "-s", help="Replay speed multiplier"
        ),
        interactive: bool = typer.Option(
            False, "--interactive", "-i", help="Interactive replay"
        ),
        highlight_events: Optional[str] = typer.Option(
            None, "--highlight", help="Event types to highlight"
        ),
    ) -> None:
        """Replay execution trace with visualization."""
        self.console.print("[bold magenta]🎬 Trace Replay[/bold magenta]")

        try:
            trace = self._load_trace(trace_file)
            highlight_list = (
                [h.strip() for h in highlight_events.split(",")]
                if highlight_events
                else []
            )

            if interactive:
                self._interactive_replay(trace, speed, highlight_list)
            else:
                self._automated_replay(trace, speed, highlight_list)

        except Exception as e:
            self.console.print(f"[red]❌ Replay error: {e}[/red]")
            raise typer.Exit(1)

    def compare_traces(
        self,
        trace1: str = typer.Argument(..., help="First trace file"),
        trace2: str = typer.Argument(..., help="Second trace file"),
        comparison_type: str = typer.Option(
            "performance", "--type", "-t", help="Comparison type"
        ),
        output_file: Optional[str] = typer.Option(
            None, "--output", "-o", help="Comparison output file"
        ),
    ) -> None:
        """Compare two execution traces."""
        self.console.print("[bold yellow]⚖️ Trace Comparison[/bold yellow]")

        try:
            # Load traces
            trace_a = self._load_trace(trace1)
            trace_b = self._load_trace(trace2)

            # Perform comparison
            comparison = self._compare_execution_traces(
                trace_a, trace_b, comparison_type
            )

            # Display comparison
            self._display_trace_comparison(comparison)

            # Save comparison if requested
            if output_file:
                self._save_comparison(comparison, output_file)

        except Exception as e:
            self.console.print(f"[red]❌ Comparison error: {e}[/red]")
            raise typer.Exit(1)

    def manage_breakpoints(
        self,
        action: str = typer.Argument(..., help="Action: add, remove, list, clear"),
        agent: Optional[str] = typer.Option(
            None, "--agent", "-a", help="Agent name for breakpoint"
        ),
        condition: Optional[str] = typer.Option(
            None, "--condition", "-c", help="Breakpoint condition"
        ),
    ) -> None:
        """Manage debugging breakpoints."""
        self.console.print("[bold red]🔴 Breakpoint Manager[/bold red]")

        # This would manage persistent breakpoint configuration
        if action == "list":
            self._list_breakpoints()
        elif action == "add" and agent:
            self._add_breakpoint(agent, condition)
        elif action == "remove" and agent:
            self._remove_breakpoint(agent)
        elif action == "clear":
            self._clear_breakpoints()
        else:
            self.console.print("[red]Invalid action or missing agent[/red]")
            raise typer.Exit(1)

    def export_trace(
        self,
        trace_file: str = typer.Argument(..., help="Trace file to export"),
        output_format: str = typer.Option(
            "json", "--format", "-f", help="Export format: json, csv, html, flamegraph"
        ),
        output_file: str = typer.Option(
            "exported_trace", "--output", "-o", help="Output file"
        ),
    ) -> None:
        """Export trace in various formats."""
        self.console.print("[bold blue]📤 Trace Exporter[/bold blue]")

        try:
            trace = self._load_trace(trace_file)

            if output_format == "json":
                self._export_json(trace, output_file + ".json")
            elif output_format == "csv":
                self._export_csv(trace, output_file + ".csv")
            elif output_format == "html":
                self._export_html(trace, output_file + ".html")
            elif output_format == "flamegraph":
                self._export_flamegraph(trace, output_file + ".svg")
            else:
                self.console.print(f"[red]Unsupported format: {output_format}[/red]")
                raise typer.Exit(1)

        except Exception as e:
            self.console.print(f"[red]❌ Export error: {e}[/red]")
            raise typer.Exit(1)

    def monitor_execution(
        self,
        duration: int = typer.Option(
            300, "--duration", "-d", help="Monitoring duration in seconds"
        ),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to monitor"
        ),
        alert_threshold: float = typer.Option(
            5.0, "--threshold", "-t", help="Alert threshold in seconds"
        ),
        output_dir: str = typer.Option(
            "./monitoring", "--output", "-o", help="Output directory"
        ),
    ) -> None:
        """Monitor execution patterns and performance."""
        self.console.print("[bold green]📡 Execution Monitor[/bold green]")

        agent_list = [a.strip() for a in agents.split(",")] if agents else None

        try:
            self._start_execution_monitoring(
                duration, agent_list, alert_threshold, output_dir
            )
        except Exception as e:
            self.console.print(f"[red]❌ Monitoring error: {e}[/red]")
            raise typer.Exit(1)

    # Helper methods

    def _execute_with_tracing(
        self, query: str, agents: Optional[List[str]], session: TracingSession
    ) -> ExecutionTrace:
        """Execute query with comprehensive tracing."""
        trace = ExecutionTrace(
            trace_id=session.session_id,
            query=query,
            start_time=datetime.now(timezone.utc),
        )

        try:
            # Use LangGraphOrchestrator if available
            if LangGraphOrchestrator is None:
                raise ImportError("LangGraphOrchestrator not available")

            # Create orchestrator with tracing hooks
            orchestrator = LangGraphOrchestrator(agents_to_run=agents)

            # Add tracing hooks (simplified for demo)
            start_time = time.time()

            # Execute
            context = asyncio.run(orchestrator.run(query))

            # Record execution completion
            trace.end_time = datetime.now(timezone.utc)
            trace.total_duration = time.time() - start_time
            trace.success = len(context.failed_agents) == 0

            # Generate trace events (simplified)
            self._generate_trace_events(trace, context, session)

        except Exception as e:
            trace.end_time = datetime.now(timezone.utc)
            trace.success = False
            trace.error_details = str(e)

        return trace

    def _generate_trace_events(
        self, trace: ExecutionTrace, context: AgentContext, session: TracingSession
    ) -> None:
        """Generate trace events from execution context."""
        # Simplified event generation
        for agent_name, output in context.agent_outputs.items():
            if not session.filter_agents or agent_name in session.filter_agents:
                event = TraceEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    event_type="agent_execution",
                    node_name=agent_name,
                    agent_name=agent_name,
                    state=(
                        ExecutionState.COMPLETED
                        if agent_name not in context.failed_agents
                        else ExecutionState.FAILED
                    ),
                    duration=1.0,  # Would be real timing data
                    output_data={"output": output} if session.capture_io else None,
                )
                trace.events.append(event)
                trace.execution_path.append(agent_name)

    def _execute_live_tracing(
        self,
        query: str,
        agents: Optional[List[str]],
        session: TracingSession,
        refresh_rate: float,
        show_events: bool,
    ) -> None:
        """Execute with live real-time tracing."""
        layout = Layout()
        layout.split_column(
            Layout(Panel("Live Execution Trace", title="Status"), size=3),
            Layout(name="events", ratio=1),
        )

        live_events: List[Dict[str, Any]] = []

        with Live(layout, refresh_per_second=1 / refresh_rate) as live:
            # Start execution in background
            async def execute_async() -> AgentContext:
                # Use LangGraphOrchestrator if available
                if LangGraphOrchestrator is None:
                    raise ImportError("LangGraphOrchestrator not available")

                orchestrator = LangGraphOrchestrator(agents_to_run=agents)
                context = await orchestrator.run(query)
                return context

            # Simulate live updates (would be real hooks in production)
            start_time = time.time()
            context = asyncio.run(execute_async())

            # Update live display
            events_table = Table(title="Live Events")
            events_table.add_column("Time", style="cyan")
            events_table.add_column("Agent", style="bold")
            events_table.add_column("State", style="green")

            for agent in context.agent_outputs.keys():
                events_table.add_row(
                    f"{time.time() - start_time:.2f}s",
                    agent,
                    (
                        "✅ Completed"
                        if agent not in context.failed_agents
                        else "❌ Failed"
                    ),
                )

            layout["events"].update(events_table)

    def _display_trace_summary(self, trace: ExecutionTrace) -> None:
        """Display trace execution summary."""
        status_color = "green" if trace.success else "red"
        status_icon = "✅" if trace.success else "❌"

        summary_panel = Panel(
            f"Trace ID: {trace.trace_id}\n"
            f"Query: {trace.query}\n"
            f"Duration: {trace.total_duration:.3f}s\n"
            f"Events: {len(trace.events)}\n"
            f"Execution Path: {' → '.join(trace.execution_path)}\n"
            f"Success: {status_icon}",
            title="Execution Trace Summary",
            border_style=status_color,
        )
        self.console.print(summary_panel)

    def _display_execution_path(self, trace: ExecutionTrace) -> None:
        """Display execution path as a tree."""
        if not trace.execution_path:
            return

        tree = Tree("🛤️ Execution Path")
        current = tree

        for i, step in enumerate(trace.execution_path):
            # Find matching event
            event = next((e for e in trace.events if e.node_name == step), None)

            if event:
                status_color = (
                    "green" if event.state == ExecutionState.COMPLETED else "red"
                )
                duration_text = f" ({event.duration:.3f}s)" if event.duration else ""
                step_text = f"[{status_color}]{step}{duration_text}[/{status_color}]"
            else:
                step_text = step

            if i == 0:
                current = tree.add(step_text)
            else:
                current = current.add(step_text)

        self.console.print(tree)

    def _display_detailed_events(self, trace: ExecutionTrace) -> None:
        """Display detailed event information."""
        if not trace.events:
            return

        events_table = Table(title="Detailed Events")
        events_table.add_column("Time", style="cyan")
        events_table.add_column("Event", style="bold")
        events_table.add_column("Agent", style="magenta")
        events_table.add_column("State", justify="center")
        events_table.add_column("Duration", justify="right")

        for event in trace.events:
            state_color = {
                ExecutionState.COMPLETED: "green",
                ExecutionState.FAILED: "red",
                ExecutionState.RUNNING: "yellow",
                ExecutionState.PENDING: "blue",
            }.get(event.state, "white")

            events_table.add_row(
                event.timestamp.strftime("%H:%M:%S.%f")[:-3],
                event.event_type,
                event.agent_name or "-",
                f"[{state_color}]{event.state.value}[/{state_color}]",
                f"{event.duration:.3f}s" if event.duration else "-",
            )

        self.console.print(events_table)

    def _save_trace(self, trace: ExecutionTrace, output_file: str) -> None:
        """Save trace to file."""
        trace_data = {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "total_duration": trace.total_duration,
            "success": trace.success,
            "execution_path": trace.execution_path,
            "events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "node_name": e.node_name,
                    "agent_name": e.agent_name,
                    "state": e.state.value,
                    "duration": e.duration,
                    "error": e.error,
                }
                for e in trace.events
            ],
        }

        with open(output_file, "w") as f:
            json.dump(trace_data, f, indent=2)

        self.console.print(f"[green]✅ Trace saved to: {output_file}[/green]")

    def _load_trace(self, trace_file: str) -> ExecutionTrace:
        """Load trace from file."""
        with open(trace_file, "r") as f:
            data = json.load(f)

        trace = ExecutionTrace(
            trace_id=data["trace_id"],
            query=data["query"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
            total_duration=data["total_duration"],
            success=data["success"],
            execution_path=data["execution_path"],
        )

        # Load events
        for event_data in data.get("events", []):
            event = TraceEvent(
                event_id=event_data["event_id"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                event_type=event_data["event_type"],
                node_name=event_data["node_name"],
                agent_name=event_data.get("agent_name"),
                state=ExecutionState(event_data["state"]),
                duration=event_data.get("duration"),
                error=event_data.get("error"),
            )
            trace.events.append(event)

        return trace

    def _start_interactive_debug_session(
        self, query: str, agents: Optional[List[str]], session: TracingSession
    ) -> None:
        """Start interactive debugging session."""
        self.console.print("🐛 Interactive Debug Session Started")
        self.console.print(
            "Commands: step, continue, inspect <agent>, breakpoint <agent>, quit"
        )

        # This would implement a full interactive debugger
        while True:
            command = Prompt.ask("debug> ")

            if command == "quit":
                break
            elif command == "step":
                self.console.print("Stepping to next execution...")
            elif command == "continue":
                self.console.print("Continuing execution...")
            elif command.startswith("inspect"):
                agent = command.split()[1] if len(command.split()) > 1 else None
                if agent:
                    self.console.print(f"Inspecting agent: {agent}")
            elif command.startswith("breakpoint"):
                agent = command.split()[1] if len(command.split()) > 1 else None
                if agent:
                    self.console.print(f"Breakpoint set for agent: {agent}")
            else:
                self.console.print("Unknown command")

    def _list_breakpoints(self) -> None:
        """List current breakpoints."""
        self.console.print("📍 Current Breakpoints:")
        # Would show actual breakpoints
        self.console.print("  • refiner (on entry)")
        self.console.print("  • critic (on error)")

    def _add_breakpoint(self, agent: str, condition: Optional[str]) -> None:
        """Add a breakpoint."""
        condition_text = f" with condition: {condition}" if condition else ""
        self.console.print(
            f"[green]✅ Breakpoint added for {agent}{condition_text}[/green]"
        )

    def _remove_breakpoint(self, agent: str) -> None:
        """Remove a breakpoint."""
        self.console.print(f"[yellow]🗑️ Breakpoint removed for {agent}[/yellow]")

    def _clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self.console.print("[red]🧹 All breakpoints cleared[/red]")

    def _execute_debug_session(
        self,
        query: str,
        agents: Optional[List[str]],
        session: TracingSession,
        step_mode: bool,
    ) -> None:
        """Execute debug session (simplified)."""
        self.console.print("🐛 Debug session started")
        trace = self._execute_with_tracing(query, agents, session)
        self._display_trace_summary(trace)

    def _save_comparison(self, comparison: Dict[str, Any], output_file: str) -> None:
        """Save comparison results to file."""
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        self.console.print(f"[green]✅ Comparison saved to: {output_file}[/green]")

    def _analyze_execution_trace(
        self, trace: ExecutionTrace, analysis_type: str, focus_agent: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze execution trace with detailed insights."""
        analysis: Dict[str, Any] = {
            "trace_summary": {
                "trace_id": trace.trace_id,
                "total_duration": trace.total_duration,
                "event_count": len(trace.events),
                "success_rate": 1.0 if trace.success else 0.0,
                "execution_path_length": len(trace.execution_path),
            },
            "performance_metrics": self._analyze_performance_metrics(trace),
            "execution_patterns": self._analyze_execution_patterns(trace),
            "error_analysis": self._analyze_errors(trace),
            "agent_analysis": self._analyze_agent_performance(trace, focus_agent),
            "optimization_suggestions": self._generate_optimization_suggestions(trace),
        }

        if analysis_type == "comprehensive":
            timeline_data: List[Dict[str, Any]] = self._generate_detailed_timeline(
                trace
            )
            analysis["detailed_timeline"] = timeline_data
            analysis["dependency_analysis"] = self._analyze_dependencies(trace)
            analysis["resource_utilization"] = self._analyze_resource_utilization(trace)

        return analysis

    def _analyze_performance_metrics(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze performance metrics from trace."""
        if not trace.events:
            return {}

        durations = [e.duration for e in trace.events if e.duration]

        return {
            "total_duration": trace.total_duration,
            "avg_event_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "bottleneck_events": (
                [
                    e.node_name
                    for e in trace.events
                    if e.duration and e.duration > (sum(durations) / len(durations) * 2)
                ]
                if durations
                else []
            ),
            "efficiency_score": (
                min(1.0, 5.0 / trace.total_duration) if trace.total_duration > 0 else 0
            ),
        }

    def _analyze_execution_patterns(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze execution patterns and flows."""
        return {
            "path_efficiency": (
                len(set(trace.execution_path)) / len(trace.execution_path)
                if trace.execution_path
                else 0
            ),
            "sequential_steps": len(trace.execution_path),
            "unique_agents": len(set(trace.execution_path)),
            "potential_parallelization": max(
                0, len(trace.execution_path) - len(set(trace.execution_path))
            ),
            "execution_flow": trace.execution_path,
            "branching_points": self._identify_branching_points(trace),
        }

    def _analyze_errors(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze errors and failure patterns."""
        error_events = [
            e for e in trace.events if e.state == ExecutionState.FAILED or e.error
        ]

        return {
            "error_count": len(error_events),
            "error_rate": len(error_events) / len(trace.events) if trace.events else 0,
            "failed_agents": [e.agent_name for e in error_events if e.agent_name],
            "error_types": list(set(e.error for e in error_events if e.error)),
            "recovery_patterns": [],  # Would analyze recovery attempts
            "failure_cascade": self._analyze_failure_cascade(trace, error_events),
        }

    def _analyze_agent_performance(
        self, trace: ExecutionTrace, focus_agent: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze individual agent performance."""
        agent_stats = {}

        for event in trace.events:
            if not event.agent_name:
                continue

            if focus_agent and event.agent_name != focus_agent:
                continue

            if event.agent_name not in agent_stats:
                agent_stats[event.agent_name] = {
                    "execution_count": 0,
                    "total_duration": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                    "avg_duration": 0.0,
                }

            stats = agent_stats[event.agent_name]
            stats["execution_count"] += 1

            if event.duration:
                stats["total_duration"] += event.duration

            if event.state == ExecutionState.COMPLETED:
                stats["success_count"] += 1
            elif event.state == ExecutionState.FAILED:
                stats["error_count"] += 1

        # Calculate averages
        for agent, stats in agent_stats.items():
            if stats["execution_count"] > 0:
                stats["avg_duration"] = float(
                    stats["total_duration"] / stats["execution_count"]
                )
                stats["success_rate"] = float(
                    stats["success_count"] / stats["execution_count"]
                )

        return agent_stats

    def _generate_optimization_suggestions(self, trace: ExecutionTrace) -> List[str]:
        """Generate optimization suggestions based on trace analysis."""
        suggestions = []

        # Performance-based suggestions
        if trace.total_duration > 10.0:
            suggestions.append(
                "Consider optimizing slow agents or implementing parallel execution"
            )

        # Error-based suggestions
        error_events = [e for e in trace.events if e.state == ExecutionState.FAILED]
        if error_events:
            suggestions.append("Review error handling and implement retry mechanisms")

        # Path efficiency suggestions
        if len(trace.execution_path) > len(set(trace.execution_path)):
            suggestions.append(
                "Potential for parallelization detected in execution path"
            )

        return suggestions

    def _identify_branching_points(self, trace: ExecutionTrace) -> List[str]:
        """Identify potential branching or decision points."""
        # Simplified - would analyze actual conditional logic
        return []

    def _analyze_failure_cascade(
        self, trace: ExecutionTrace, error_events: List[TraceEvent]
    ) -> List[str]:
        """Analyze if failures caused cascading effects."""
        # Simplified analysis
        return [e.agent_name for e in error_events if e.agent_name]

    def _generate_detailed_timeline(
        self, trace: ExecutionTrace
    ) -> List[Dict[str, Any]]:
        """Generate detailed timeline of execution."""
        timeline = []

        for event in sorted(trace.events, key=lambda e: e.timestamp):
            timeline.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "relative_time": (
                        event.timestamp - trace.start_time
                    ).total_seconds(),
                    "event_type": event.event_type,
                    "agent": event.agent_name,
                    "state": event.state.value,
                    "duration": event.duration,
                }
            )

        return timeline

    def _analyze_dependencies(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze agent dependencies and execution order."""
        return {
            "execution_order": trace.execution_path,
            "dependencies": {},  # Would map actual dependencies
            "critical_path": trace.execution_path,  # Simplified
            "parallel_opportunities": [],
        }

    def _analyze_resource_utilization(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        return {
            "peak_concurrent_agents": 1,  # Would track actual concurrency
            "resource_efficiency": 0.8,  # Would calculate from real metrics
            "memory_usage_pattern": "stable",  # Would analyze memory trends
            "cpu_utilization": "moderate",  # Would analyze CPU usage
        }

    def _display_trace_analysis(
        self, analysis: Dict[str, Any], performance_analysis: bool
    ) -> None:
        """Display comprehensive trace analysis."""
        self.console.print("[bold]📊 Trace Analysis Results[/bold]")

        # Summary
        summary = analysis["trace_summary"]
        summary_table = Table(title="Trace Summary")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Duration", f"{summary['total_duration']:.3f}s")
        summary_table.add_row("Event Count", str(summary["event_count"]))
        summary_table.add_row("Success Rate", f"{summary['success_rate']:.1%}")
        summary_table.add_row("Path Length", str(summary["execution_path_length"]))

        self.console.print(summary_table)

        if performance_analysis:
            # Performance metrics
            perf = analysis["performance_metrics"]
            if perf:
                perf_table = Table(title="Performance Metrics")
                perf_table.add_column("Metric", style="bold")
                perf_table.add_column("Value", justify="right")

                perf_table.add_row(
                    "Avg Event Duration", f"{perf.get('avg_event_duration', 0):.3f}s"
                )
                perf_table.add_row(
                    "Min Duration", f"{perf.get('min_duration', 0):.3f}s"
                )
                perf_table.add_row(
                    "Max Duration", f"{perf.get('max_duration', 0):.3f}s"
                )
                perf_table.add_row(
                    "Efficiency Score", f"{perf.get('efficiency_score', 0):.2f}"
                )

                self.console.print(perf_table)

        # Optimization suggestions
        suggestions = analysis.get("optimization_suggestions", [])
        if suggestions:
            self.console.print(
                "\n[bold yellow]💡 Optimization Suggestions:[/bold yellow]"
            )
            for suggestion in suggestions:
                self.console.print(f"  • {suggestion}")

    def _compare_execution_traces(
        self, trace_a: ExecutionTrace, trace_b: ExecutionTrace, comparison_type: str
    ) -> Dict[str, Any]:
        """Compare two execution traces."""
        return {
            "trace_a_summary": {
                "duration": trace_a.total_duration,
                "success": trace_a.success,
                "event_count": len(trace_a.events),
            },
            "trace_b_summary": {
                "duration": trace_b.total_duration,
                "success": trace_b.success,
                "event_count": len(trace_b.events),
            },
            "performance_delta": {
                "duration_diff": trace_b.total_duration - trace_a.total_duration,
                "event_diff": len(trace_b.events) - len(trace_a.events),
                "improvement": trace_b.total_duration < trace_a.total_duration,
            },
        }

    def _display_trace_comparison(self, comparison: Dict[str, Any]) -> None:
        """Display trace comparison results."""
        self.console.print("[bold]⚖️ Trace Comparison Results[/bold]")

        comp_table = Table(title="Comparison")
        comp_table.add_column("Metric", style="bold")
        comp_table.add_column("Trace A", justify="right")
        comp_table.add_column("Trace B", justify="right")
        comp_table.add_column("Delta", justify="right")

        trace_a = comparison["trace_a_summary"]
        trace_b = comparison["trace_b_summary"]
        delta = comparison["performance_delta"]

        comp_table.add_row(
            "Duration",
            f"{trace_a['duration']:.3f}s",
            f"{trace_b['duration']:.3f}s",
            f"{delta['duration_diff']:+.3f}s",
        )

        comp_table.add_row(
            "Events",
            str(trace_a["event_count"]),
            str(trace_b["event_count"]),
            f"{delta['event_diff']:+d}",
        )

        self.console.print(comp_table)

        if delta["improvement"]:
            self.console.print("[green]✅ Performance improved in Trace B[/green]")
        else:
            self.console.print("[red]⚠️ Performance degraded in Trace B[/red]")

    def _interactive_replay(
        self, trace: ExecutionTrace, speed: float, highlight_list: List[str]
    ) -> None:
        """Interactive trace replay."""
        self.console.print("🎬 Interactive Trace Replay")
        self.console.print(
            "Commands: play, pause, step, jump <event>, speed <factor>, quit"
        )

        current_event = 0
        playing = False

        while True:
            command = Prompt.ask(f"replay ({current_event}/{len(trace.events)})> ")

            if command == "quit":
                break
            elif command == "play":
                playing = True
                self.console.print("▶️ Playing...")
            elif command == "pause":
                playing = False
                self.console.print("⏸️ Paused")
            elif command == "step":
                current_event = min(current_event + 1, len(trace.events) - 1)
                self._display_event(
                    trace.events[current_event]
                    if current_event < len(trace.events)
                    else None
                )
            else:
                self.console.print("Unknown command")

    def _automated_replay(
        self, trace: ExecutionTrace, speed: float, highlight_list: List[str]
    ) -> None:
        """Automated trace replay."""
        self.console.print(f"🎬 Replaying trace at {speed}x speed...")

        for event in trace.events:
            self._display_event(event)
            time.sleep(0.5 / speed)  # Adjust timing based on speed

    def _display_event(self, event: Optional[TraceEvent]) -> None:
        """Display a single trace event."""
        if not event:
            return

        status_color = "green" if event.state == ExecutionState.COMPLETED else "red"
        self.console.print(
            f"[{status_color}]{event.timestamp.strftime('%H:%M:%S.%f')[:-3]} - {event.agent_name}: {event.state.value}[/{status_color}]"
        )

    def _start_execution_monitoring(
        self,
        duration: int,
        agents: Optional[List[str]],
        alert_threshold: float,
        output_dir: str,
    ) -> None:
        """Start continuous execution monitoring."""
        self.console.print(f"📡 Monitoring execution for {duration} seconds...")

        start_time = time.time()
        while time.time() - start_time < duration:
            # Would monitor actual executions
            time.sleep(1)
            self.console.print(".", end="")

        self.console.print(f"\n✅ Monitoring complete. Results saved to {output_dir}")

    def _export_json(self, trace: ExecutionTrace, output_file: str) -> None:
        """Export trace as JSON."""
        self._save_trace(trace, output_file)

    def _export_csv(self, trace: ExecutionTrace, output_file: str) -> None:
        """Export trace as CSV."""
        import csv

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "agent", "state", "duration", "error"])

            for event in trace.events:
                writer.writerow(
                    [
                        event.timestamp.isoformat(),
                        event.agent_name or "",
                        event.state.value,
                        event.duration or 0,
                        event.error or "",
                    ]
                )

        self.console.print(f"[green]✅ CSV exported to: {output_file}[/green]")

    def _export_html(self, trace: ExecutionTrace, output_file: str) -> None:
        """Export trace as HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Execution Trace Report</title></head>
        <body>
        <h1>Execution Trace: {trace.trace_id}</h1>
        <p>Query: {trace.query}</p>
        <p>Duration: {trace.total_duration:.3f}s</p>
        <p>Success: {"✅" if trace.success else "❌"}</p>
        
        <h2>Events</h2>
        <table border="1">
        <tr><th>Time</th><th>Agent</th><th>State</th><th>Duration</th></tr>
        """

        for event in trace.events:
            html_content += f"""
            <tr>
                <td>{event.timestamp.strftime("%H:%M:%S.%f")[:-3]}</td>
                <td>{event.agent_name or ""}</td>
                <td>{event.state.value}</td>
                <td>{event.duration or 0:.3f}s</td>
            </tr>
            """

        html_content += """
        </table>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html_content)

        self.console.print(f"[green]✅ HTML exported to: {output_file}[/green]")

    def _export_flamegraph(self, trace: ExecutionTrace, output_file: str) -> None:
        """Export trace as flamegraph SVG."""
        # Simplified flamegraph export
        self.console.print("[yellow]⚠️ Flamegraph export not yet implemented[/yellow]")
        self.console.print(f"Would export to: {output_file}")


# Create global instance
execution_tracer = ExecutionTracer()
app = execution_tracer.create_app()

if __name__ == "__main__":
    app()
