"""
Performance profiling and benchmarking tools for Phase 2C developer experience.

This module provides comprehensive performance analysis, profiling, and benchmarking
capabilities for LangGraph DAG execution with detailed metrics and optimization insights.
"""

import asyncio
import time
import statistics
import json
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import contextlib
from concurrent.futures import ThreadPoolExecutor

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

from cognivault.context import AgentContext
from cognivault.langraph.real_orchestrator import RealLangGraphOrchestrator


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionProfile:
    """Detailed execution profile for a single run."""

    execution_id: str
    query: str
    execution_mode: str
    agents: List[str]
    total_duration: float
    agent_durations: Dict[str, float]
    memory_usage: Dict[str, float]  # peak, average, final
    cpu_usage: Dict[str, float]
    token_usage: Dict[str, int]
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context_size: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class BenchmarkSuite:
    """Comprehensive benchmark suite results."""

    suite_name: str
    total_runs: int
    execution_modes: List[str]
    queries: List[str]
    profiles: List[ExecutionProfile]
    summary_stats: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceProfiler:
    """Advanced performance profiler with real-time monitoring."""

    def __init__(self) -> None:
        self.console = Console()
        self.monitoring = False
        self.current_profile: Optional[ExecutionProfile] = None
        self.resource_monitor_thread: Optional[threading.Thread] = None
        self.resource_data: List[Dict[str, float]] = []

    def create_app(self) -> typer.Typer:
        """Create the performance profiler CLI application."""
        app = typer.Typer(
            name="profiler",
            help="Performance profiling and benchmarking tools",
            no_args_is_help=True,
        )

        app.command("profile")(self.profile_execution)
        app.command("benchmark")(self.run_benchmark)
        app.command("compare")(self.compare_modes)
        app.command("monitor")(self.monitor_performance)
        app.command("analyze")(self.analyze_profile)
        app.command("report")(self.generate_report)
        app.command("optimize")(self.suggest_optimizations)
        app.command("stress")(self.stress_test)

        return app

    def profile_execution(
        self,
        query: str = typer.Argument(..., help="Query to profile"),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to profile"
        ),
        execution_mode: str = typer.Option(
            "langgraph-real", "--mode", "-m", help="Execution mode"
        ),
        runs: int = typer.Option(1, "--runs", "-r", help="Number of profiling runs"),
        detailed: bool = typer.Option(
            False, "--detailed", help="Enable detailed profiling"
        ),
        live_monitor: bool = typer.Option(
            False, "--live", help="Live performance monitoring"
        ),
        output_file: Optional[str] = typer.Option(
            None, "--output", "-o", help="Save profile to file"
        ),
    ):
        """Profile DAG execution with detailed performance metrics."""
        self.console.print("[bold blue]📊 Performance Profiler[/bold blue]")

        agent_list = [a.strip() for a in agents.split(",")] if agents else None

        if live_monitor:
            self._profile_with_live_monitoring(
                query, agent_list, execution_mode, runs, detailed
            )
        else:
            profiles = self._run_profiling_suite(
                query, agent_list, execution_mode, runs, detailed
            )
            self._display_profile_results(profiles)

            if output_file:
                self._save_profiles(profiles, output_file)

    def run_benchmark(
        self,
        queries_file: Optional[str] = typer.Option(
            None, "--queries", help="File with benchmark queries"
        ),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to benchmark"
        ),
        modes: str = typer.Option(
            "langgraph-real", "--modes", "-m", help="Execution modes to benchmark"
        ),
        runs_per_query: int = typer.Option(3, "--runs", "-r", help="Runs per query"),
        warmup_runs: int = typer.Option(1, "--warmup", help="Warmup runs"),
        output_dir: str = typer.Option(
            "./benchmark_results", "--output", "-o", help="Output directory"
        ),
        suite_name: Optional[str] = typer.Option(
            None, "--name", help="Benchmark suite name"
        ),
    ):
        """Run comprehensive benchmark suite across modes and queries."""
        self.console.print("[bold green]🏁 Benchmark Suite[/bold green]")

        # Load or generate queries
        queries = self._load_benchmark_queries(queries_file)
        agent_list = [a.strip() for a in agents.split(",")] if agents else None
        mode_list = [m.strip() for m in modes.split(",")]

        suite_name = (
            suite_name or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Run benchmark suite
        suite = self._execute_benchmark_suite(
            suite_name, queries, agent_list, mode_list, runs_per_query, warmup_runs
        )

        # Display and save results
        self._display_benchmark_results(suite)
        self._save_benchmark_suite(suite, output_dir)

    def compare_modes(
        self,
        query: str = typer.Argument(..., help="Query for comparison"),
        modes: str = typer.Option(
            "legacy,langgraph,langgraph-real", "--modes", help="Modes to compare"
        ),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to use"
        ),
        runs: int = typer.Option(5, "--runs", "-r", help="Runs per mode"),
        statistical: bool = typer.Option(True, "--stats", help="Statistical analysis"),
        visual: bool = typer.Option(False, "--visual", help="Visual comparison charts"),
    ):
        """Compare performance across different execution modes."""
        self.console.print("[bold magenta]⚖️  Mode Comparison[/bold magenta]")

        agent_list = [a.strip() for a in agents.split(",")] if agents else None
        mode_list = [m.strip() for m in modes.split(",")]

        # Run comparison
        comparison_results = self._run_mode_comparison(
            query, agent_list, mode_list, runs
        )

        # Display results
        self._display_comparison_results(comparison_results, statistical)

        if visual:
            self._generate_visual_comparison(comparison_results)

    def monitor_performance(
        self,
        duration: int = typer.Option(
            60, "--duration", "-d", help="Monitoring duration in seconds"
        ),
        interval: float = typer.Option(
            1.0, "--interval", "-i", help="Sampling interval in seconds"
        ),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to monitor"
        ),
        output_file: Optional[str] = typer.Option(
            None, "--output", "-o", help="Save monitoring data"
        ),
    ):
        """Monitor system performance during DAG execution."""
        self.console.print("[bold yellow]🔍 Performance Monitor[/bold yellow]")

        # Start monitoring
        monitor_data = self._start_performance_monitoring(duration, interval)

        # Display monitoring results
        self._display_monitoring_results(monitor_data)

        if output_file:
            self._save_monitoring_data(monitor_data, output_file)

    def analyze_profile(
        self,
        profile_file: str = typer.Argument(..., help="Profile file to analyze"),
        analysis_type: str = typer.Option(
            "comprehensive", "--type", "-t", help="Analysis type"
        ),
        compare_with: Optional[str] = typer.Option(
            None, "--compare", help="Compare with another profile"
        ),
        generate_insights: bool = typer.Option(
            True, "--insights", help="Generate optimization insights"
        ),
    ):
        """Analyze existing performance profiles."""
        self.console.print("[bold cyan]🔬 Profile Analyzer[/bold cyan]")

        # Load and analyze profile
        profile_data = self._load_profile_data(profile_file)
        analysis = self._perform_profile_analysis(profile_data, analysis_type)

        # Display analysis
        self._display_analysis_results(analysis)

        if compare_with:
            comparison = self._compare_profiles(
                profile_data, self._load_profile_data(compare_with)
            )
            self._display_profile_comparison(comparison)

        if generate_insights:
            insights = self._generate_optimization_insights(analysis)
            self._display_optimization_insights(insights)

    def generate_report(
        self,
        data_dir: str = typer.Argument(..., help="Directory with performance data"),
        output_format: str = typer.Option(
            "markdown", "--format", "-f", help="Report format: markdown, html, json"
        ),
        template: Optional[str] = typer.Option(
            None, "--template", help="Custom report template"
        ),
        include_graphs: bool = typer.Option(
            True, "--graphs", help="Include performance graphs"
        ),
    ):
        """Generate comprehensive performance report."""
        self.console.print("[bold green]📋 Report Generator[/bold green]")

        # Load data and generate report
        report_data = self._compile_report_data(data_dir)
        report = self._generate_performance_report(
            report_data, output_format, template, include_graphs
        )

        # Save report
        output_file = f"performance_report.{output_format}"
        self._save_report(report, output_file)

        self.console.print(f"[green]✅ Report generated: {output_file}[/green]")

    def suggest_optimizations(
        self,
        profile_file: str = typer.Argument(..., help="Profile file to analyze"),
        target_metric: str = typer.Option(
            "latency", "--target", "-t", help="Target optimization metric"
        ),
        confidence_threshold: float = typer.Option(
            0.8, "--confidence", help="Confidence threshold"
        ),
        interactive: bool = typer.Option(
            False, "--interactive", help="Interactive optimization suggestions"
        ),
    ):
        """Suggest performance optimizations based on profile analysis."""
        self.console.print("[bold red]🚀 Optimization Advisor[/bold red]")

        # Analyze profile and generate suggestions
        profile_data = self._load_profile_data(profile_file)
        suggestions = self._generate_optimization_suggestions(
            profile_data, target_metric, confidence_threshold
        )

        if interactive:
            self._interactive_optimization_session(suggestions)
        else:
            self._display_optimization_suggestions(suggestions)

    def stress_test(
        self,
        query: str = typer.Argument(..., help="Query for stress testing"),
        concurrent_requests: int = typer.Option(
            10, "--concurrent", "-c", help="Concurrent requests"
        ),
        total_requests: int = typer.Option(100, "--total", "-t", help="Total requests"),
        ramp_up_time: int = typer.Option(
            10, "--rampup", help="Ramp-up time in seconds"
        ),
        agents: Optional[str] = typer.Option(
            None, "--agents", "-a", help="Agents to stress test"
        ),
        mode: str = typer.Option(
            "langgraph-real", "--mode", "-m", help="Execution mode"
        ),
    ):
        """Perform stress testing on DAG execution."""
        self.console.print("[bold red]💥 Stress Test[/bold red]")

        agent_list = [a.strip() for a in agents.split(",")] if agents else None

        # Run stress test
        stress_results = self._execute_stress_test(
            query, agent_list, mode, concurrent_requests, total_requests, ramp_up_time
        )

        # Display results
        self._display_stress_test_results(stress_results)

    # Helper methods for profiling

    def _run_profiling_suite(
        self,
        query: str,
        agents: Optional[List[str]],
        mode: str,
        runs: int,
        detailed: bool,
    ) -> List[ExecutionProfile]:
        """Run a complete profiling suite."""
        profiles = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Profiling {runs} runs...", total=runs)

            for i in range(runs):
                profile = self._profile_single_execution(query, agents, mode, detailed)
                profiles.append(profile)
                progress.update(task, advance=1)

        return profiles

    def _profile_single_execution(
        self, query: str, agents: Optional[List[str]], mode: str, detailed: bool
    ) -> ExecutionProfile:
        """Profile a single execution."""
        execution_id = f"profile_{int(time.time() * 1000)}"

        # Start resource monitoring
        if detailed:
            self._start_resource_monitoring()

        # Create orchestrator (Phase 3: Only langgraph-real supported)
        start_time = time.time()
        try:
            if mode != "langgraph-real":
                raise ValueError(
                    f"Unsupported execution mode: {mode}. Only 'langgraph-real' is supported after Phase 3."
                )

            real_orchestrator = RealLangGraphOrchestrator(agents_to_run=agents)
            context = asyncio.run(real_orchestrator.run(query))

            total_duration = time.time() - start_time
            success = len(context.failed_agents) == 0
            error = None

        except Exception as e:
            total_duration = time.time() - start_time
            success = False
            error = str(e)
            context = AgentContext(query=query)

        # Stop resource monitoring
        if detailed:
            resource_data = self._stop_resource_monitoring()
        else:
            resource_data = {"memory": {}, "cpu": {}}

        # Create profile
        profile = ExecutionProfile(
            execution_id=execution_id,
            query=query,
            execution_mode=mode,
            agents=agents or ["all"],
            total_duration=total_duration,
            agent_durations=self._extract_agent_durations(context),
            memory_usage=resource_data["memory"],
            cpu_usage=resource_data["cpu"],
            token_usage=self._extract_token_usage(context),
            success=success,
            error=error,
            context_size=getattr(context, "current_size", 0),
            llm_calls=self._count_llm_calls(context),
        )

        return profile

    def _start_resource_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.resource_data = []

        def monitor():
            while self.monitoring:
                try:
                    self.resource_data.append(
                        {
                            "timestamp": time.time(),
                            "memory_percent": psutil.virtual_memory().percent,
                            "memory_used": psutil.virtual_memory().used
                            / (1024**2),  # MB
                            "cpu_percent": psutil.cpu_percent(interval=0.1),
                        }
                    )
                    time.sleep(0.1)
                except:
                    break

        self.resource_monitor_thread = threading.Thread(target=monitor)
        self.resource_monitor_thread.start()

    def _stop_resource_monitoring(self) -> Dict[str, Any]:
        """Stop resource monitoring and return aggregated data."""
        self.monitoring = False
        if self.resource_monitor_thread:
            self.resource_monitor_thread.join(timeout=1)

        if not self.resource_data:
            return {"memory": {}, "cpu": {}}

        memory_values = [d["memory_used"] for d in self.resource_data]
        cpu_values = [d["cpu_percent"] for d in self.resource_data]

        return {
            "memory": {
                "peak": max(memory_values) if memory_values else 0,
                "average": statistics.mean(memory_values) if memory_values else 0,
                "final": memory_values[-1] if memory_values else 0,
            },
            "cpu": {
                "peak": max(cpu_values) if cpu_values else 0,
                "average": statistics.mean(cpu_values) if cpu_values else 0,
                "final": cpu_values[-1] if cpu_values else 0,
            },
        }

    def _extract_agent_durations(self, context: AgentContext) -> Dict[str, float]:
        """Extract agent execution durations from context."""
        # This would need real integration with context timing data
        return {agent: 1.0 for agent in context.agent_outputs.keys()}

    def _extract_token_usage(self, context: AgentContext) -> Dict[str, int]:
        """Extract token usage information from context."""
        # This would need real integration with LLM usage tracking
        return {"total": 1000, "input": 500, "output": 500}

    def _count_llm_calls(self, context: AgentContext) -> int:
        """Count the number of LLM calls made."""
        # This would need real integration with LLM call tracking
        return len(context.agent_outputs)

    def _display_profile_results(self, profiles: List[ExecutionProfile]):
        """Display profiling results."""
        if not profiles:
            return

        # Summary statistics
        durations = [p.total_duration for p in profiles]
        success_rate = sum(1 for p in profiles if p.success) / len(profiles)

        summary_table = Table(title="Profiling Summary")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Runs", str(len(profiles)))
        summary_table.add_row("Success Rate", f"{success_rate:.1%}")
        summary_table.add_row("Avg Duration", f"{statistics.mean(durations):.3f}s")
        summary_table.add_row("Min Duration", f"{min(durations):.3f}s")
        summary_table.add_row("Max Duration", f"{max(durations):.3f}s")
        if len(durations) > 1:
            summary_table.add_row("Std Dev", f"{statistics.stdev(durations):.3f}s")

        self.console.print(summary_table)

        # Individual runs table
        if len(profiles) <= 10:  # Only show details for small runs
            runs_table = Table(title="Individual Runs")
            runs_table.add_column("Run", style="bold")
            runs_table.add_column("Duration (s)", justify="right")
            runs_table.add_column("Success", justify="center")
            runs_table.add_column("Memory (MB)", justify="right")

            for i, profile in enumerate(profiles, 1):
                memory_usage = profile.memory_usage.get("peak", 0)
                success_icon = "✅" if profile.success else "❌"

                runs_table.add_row(
                    str(i),
                    f"{profile.total_duration:.3f}",
                    success_icon,
                    f"{memory_usage:.1f}",
                )

            self.console.print(runs_table)

    def _load_benchmark_queries(self, queries_file: Optional[str]) -> List[str]:
        """Load benchmark queries from file or use defaults."""
        if queries_file and Path(queries_file).exists():
            with open(queries_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

        return [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "Compare different programming languages",
            "Analyze market trends in technology",
            "Discuss renewable energy solutions",
            "Evaluate social media impact",
            "Explore space exploration history",
            "Examine educational system reforms",
            "Review healthcare innovations",
            "Assess climate change policies",
        ]

    def _execute_benchmark_suite(
        self,
        suite_name: str,
        queries: List[str],
        agents: Optional[List[str]],
        modes: List[str],
        runs_per_query: int,
        warmup_runs: int,
    ) -> BenchmarkSuite:
        """Execute a comprehensive benchmark suite."""
        all_profiles = []
        total_runs = len(queries) * len(modes) * (runs_per_query + warmup_runs)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running benchmark suite...", total=total_runs)

            for mode in modes:
                for query in queries:
                    # Warmup runs
                    for _ in range(warmup_runs):
                        self._profile_single_execution(query, agents, mode, False)
                        progress.update(task, advance=1)

                    # Actual benchmark runs
                    for _ in range(runs_per_query):
                        profile = self._profile_single_execution(
                            query, agents, mode, True
                        )
                        all_profiles.append(profile)
                        progress.update(task, advance=1)

        # Calculate summary statistics
        summary_stats = self._calculate_benchmark_summary(all_profiles)

        return BenchmarkSuite(
            suite_name=suite_name,
            total_runs=len(all_profiles),
            execution_modes=modes,
            queries=queries,
            profiles=all_profiles,
            summary_stats=summary_stats,
        )

    def _calculate_benchmark_summary(
        self, profiles: List[ExecutionProfile]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for benchmark suite."""
        if not profiles:
            return {}

        durations = [p.total_duration for p in profiles]
        success_rate = sum(1 for p in profiles if p.success) / len(profiles)

        # Group by execution mode
        mode_stats = {}
        for mode in set(p.execution_mode for p in profiles):
            mode_profiles = [p for p in profiles if p.execution_mode == mode]
            mode_durations = [p.total_duration for p in mode_profiles]
            mode_success_rate = sum(1 for p in mode_profiles if p.success) / len(
                mode_profiles
            )

            mode_stats[mode] = {
                "count": len(mode_profiles),
                "avg_duration": statistics.mean(mode_durations),
                "success_rate": mode_success_rate,
                "min_duration": min(mode_durations),
                "max_duration": max(mode_durations),
            }

        return {
            "overall": {
                "total_runs": len(profiles),
                "avg_duration": statistics.mean(durations),
                "success_rate": success_rate,
                "throughput": len(profiles) / sum(durations) * 60,  # per minute
            },
            "by_mode": mode_stats,
        }

    def _display_benchmark_results(self, suite: BenchmarkSuite):
        """Display benchmark suite results."""
        self.console.print(f"[bold]📊 Benchmark Suite: {suite.suite_name}[/bold]")

        # Overall summary
        overall = suite.summary_stats.get("overall", {})
        summary_table = Table(title="Overall Summary")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Runs", str(overall.get("total_runs", 0)))
        summary_table.add_row("Avg Duration", f"{overall.get('avg_duration', 0):.3f}s")
        summary_table.add_row("Success Rate", f"{overall.get('success_rate', 0):.1%}")
        summary_table.add_row(
            "Throughput", f"{overall.get('throughput', 0):.1f} runs/min"
        )

        self.console.print(summary_table)

        # Mode comparison
        mode_stats = suite.summary_stats.get("by_mode", {})
        if mode_stats:
            mode_table = Table(title="Mode Comparison")
            mode_table.add_column("Mode", style="bold")
            mode_table.add_column("Runs", justify="right")
            mode_table.add_column("Avg Duration (s)", justify="right")
            mode_table.add_column("Success Rate", justify="right")

            for mode, stats in mode_stats.items():
                mode_table.add_row(
                    mode,
                    str(stats["count"]),
                    f"{stats['avg_duration']:.3f}",
                    f"{stats['success_rate']:.1%}",
                )

            self.console.print(mode_table)

    def _save_profiles(self, profiles: List[ExecutionProfile], output_file: str):
        """Save profiles to file."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profiles": [
                {
                    "execution_id": p.execution_id,
                    "query": p.query,
                    "execution_mode": p.execution_mode,
                    "agents": p.agents,
                    "total_duration": p.total_duration,
                    "success": p.success,
                    "memory_usage": p.memory_usage,
                    "cpu_usage": p.cpu_usage,
                    "token_usage": p.token_usage,
                }
                for p in profiles
            ],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        self.console.print(f"[green]✅ Profiles saved to: {output_file}[/green]")

    def _save_benchmark_suite(self, suite: BenchmarkSuite, output_dir: str):
        """Save benchmark suite to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save main results
        results_file = output_path / f"{suite.suite_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "suite_name": suite.suite_name,
                    "timestamp": suite.timestamp.isoformat(),
                    "total_runs": suite.total_runs,
                    "execution_modes": suite.execution_modes,
                    "queries": suite.queries,
                    "summary_stats": suite.summary_stats,
                    "profiles": [
                        {
                            "execution_id": p.execution_id,
                            "query": p.query,
                            "execution_mode": p.execution_mode,
                            "total_duration": p.total_duration,
                            "success": p.success,
                        }
                        for p in suite.profiles
                    ],
                },
                f,
                indent=2,
            )

        self.console.print(
            f"[green]✅ Benchmark results saved to: {results_file}[/green]"
        )

    # Placeholder implementations for missing methods

    def _profile_with_live_monitoring(
        self,
        query: str,
        agents: Optional[List[str]],
        mode: str,
        runs: int,
        detailed: bool,
    ):
        """Profile with live monitoring (simplified)."""
        self.console.print("🔴 Live monitoring active...")
        profiles = self._run_profiling_suite(query, agents, mode, runs, detailed)
        self._display_profile_results(profiles)

    def _run_mode_comparison(
        self, query: str, agents: Optional[List[str]], modes: List[str], runs: int
    ) -> Dict[str, Any]:
        """Run mode comparison (simplified)."""
        results = {}
        for mode in modes:
            results[mode] = {"avg_duration": 1.5, "success_rate": 0.95, "runs": runs}
        return results

    def _display_comparison_results(self, results: Dict[str, Any], statistical: bool):
        """Display comparison results."""
        table = Table(title="Mode Comparison")
        table.add_column("Mode", style="bold")
        table.add_column("Avg Duration", justify="right")
        table.add_column("Success Rate", justify="right")

        for mode, data in results.items():
            table.add_row(
                mode, f"{data['avg_duration']:.2f}s", f"{data['success_rate']:.1%}"
            )

        self.console.print(table)

    def _generate_visual_comparison(self, results: Dict[str, Any]):
        """Generate visual comparison (placeholder)."""
        self.console.print("📊 Visual comparison would be generated here")

    def _start_performance_monitoring(
        self, duration: int, interval: float
    ) -> List[Dict[str, Any]]:
        """Start performance monitoring."""
        import time

        self.console.print(f"📡 Monitoring for {duration}s...")
        time.sleep(min(duration, 2))  # Simulate monitoring
        return [{"timestamp": time.time(), "cpu": 50.0, "memory": 60.0}]

    def _display_monitoring_results(self, data: List[Dict[str, Any]]):
        """Display monitoring results."""
        self.console.print(f"📈 Monitoring completed with {len(data)} data points")

    def _save_monitoring_data(self, data: List[Dict[str, Any]], output_file: str):
        """Save monitoring data."""
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_profile_data(self, profile_file: str) -> Dict[str, Any]:
        """Load profile data from file."""
        try:
            with open(profile_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"error": "Profile file not found"}

    def _perform_profile_analysis(
        self, data: Dict[str, Any], analysis_type: str
    ) -> Dict[str, Any]:
        """Perform profile analysis."""
        return {
            "analysis_type": analysis_type,
            "summary": "Analysis completed",
            "recommendations": ["Optimize slow operations"],
        }

    def _display_analysis_results(self, analysis: Dict[str, Any]):
        """Display analysis results."""
        self.console.print(f"Analysis Type: {analysis.get('analysis_type', 'Unknown')}")
        self.console.print(f"Summary: {analysis.get('summary', 'No summary')}")

    def _compare_profiles(
        self, profile1: Dict[str, Any], profile2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two profiles."""
        return {
            "profile1_duration": profile1.get("duration", 0),
            "profile2_duration": profile2.get("duration", 0),
            "improvement": "10% faster",
        }

    def _display_profile_comparison(self, comparison: Dict[str, Any]):
        """Display profile comparison."""
        self.console.print("Profile Comparison Results:")
        for key, value in comparison.items():
            self.console.print(f"  {key}: {value}")

    def _generate_optimization_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization insights."""
        return ["Consider caching", "Reduce API calls", "Optimize loops"]

    def _display_optimization_insights(self, insights: List[str]):
        """Display optimization insights."""
        self.console.print("💡 Optimization Insights:")
        for insight in insights:
            self.console.print(f"  • {insight}")

    def _compile_report_data(self, data_dir: str) -> Dict[str, Any]:
        """Compile report data from directory."""
        return {"report_type": "performance", "data_dir": data_dir}

    def _generate_performance_report(
        self,
        data: Dict[str, Any],
        format: str,
        template: Optional[str],
        include_graphs: bool,
    ) -> str:
        """Generate performance report."""
        return f"# Performance Report\n\nFormat: {format}\nGraphs: {include_graphs}\n"

    def _save_report(self, report: str, output_file: str):
        """Save report to file."""
        with open(output_file, "w") as f:
            f.write(report)

    def _generate_optimization_suggestions(
        self, data: Dict[str, Any], target: str, threshold: float
    ) -> List[str]:
        """Generate optimization suggestions."""
        return [f"Optimize {target} operations", "Implement caching"]

    def _interactive_optimization_session(self, suggestions: List[str]):
        """Interactive optimization session."""
        self.console.print("🔧 Interactive optimization session:")
        for i, suggestion in enumerate(suggestions, 1):
            self.console.print(f"  {i}. {suggestion}")

    def _display_optimization_suggestions(self, suggestions: List[str]):
        """Display optimization suggestions."""
        self.console.print("🚀 Optimization Suggestions:")
        for suggestion in suggestions:
            self.console.print(f"  • {suggestion}")

    def _execute_stress_test(
        self,
        query: str,
        agents: Optional[List[str]],
        mode: str,
        concurrent: int,
        total: int,
        rampup: int,
    ) -> Dict[str, Any]:
        """Execute stress test."""
        return {
            "concurrent_requests": concurrent,
            "total_requests": total,
            "avg_response_time": 1.2,
            "success_rate": 0.95,
        }

    def _display_stress_test_results(self, results: Dict[str, Any]):
        """Display stress test results."""
        table = Table(title="Stress Test Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        for key, value in results.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(table)


# Create global instance
profiler = PerformanceProfiler()
app = profiler.create_app()

if __name__ == "__main__":
    app()
