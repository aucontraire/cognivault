"""
Comprehensive tests for Performance Profiler module.

Tests the PerformanceProfiler class with profiling capabilities,
benchmarking, performance monitoring, and optimization suggestions.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from cognivault.diagnostics.profiler import (
    PerformanceProfiler,
    PerformanceMetric,
    ExecutionProfile,
    BenchmarkSuite,
)
from cognivault.context import AgentContext


class TestPerformanceMetric:
    """Test suite for PerformanceMetric dataclass."""

    def test_performance_metric_creation(self):
        """Test PerformanceMetric creation."""
        timestamp = datetime.now(timezone.utc)
        metric = PerformanceMetric(
            name="execution_time",
            value=1.5,
            unit="seconds",
            timestamp=timestamp,
            metadata={"agent": "refiner", "query_length": 100},
        )

        assert metric.name == "execution_time"
        assert metric.value == 1.5
        assert metric.unit == "seconds"
        assert metric.timestamp == timestamp
        assert metric.metadata["agent"] == "refiner"
        assert metric.metadata["query_length"] == 100

    def test_performance_metric_minimal(self):
        """Test PerformanceMetric with minimal fields."""
        timestamp = datetime.now(timezone.utc)
        metric = PerformanceMetric(
            name="cpu_usage", value=75.0, unit="percent", timestamp=timestamp
        )

        assert metric.name == "cpu_usage"
        assert metric.value == 75.0
        assert metric.unit == "percent"
        assert len(metric.metadata) == 0


class TestExecutionProfile:
    """Test suite for ExecutionProfile dataclass."""

    def test_execution_profile_creation(self):
        """Test ExecutionProfile creation with all fields."""
        timestamp = datetime.now(timezone.utc)
        profile = ExecutionProfile(
            execution_id="exec_123",
            query="Test query for profiling",
            execution_mode="langgraph-real",
            agents=["refiner", "critic"],
            total_duration=3.5,
            agent_durations={"refiner": 1.5, "critic": 2.0},
            memory_usage={"peak": 256.0, "average": 200.0, "final": 180.0},
            cpu_usage={"peak": 85.0, "average": 65.0, "final": 45.0},
            token_usage={"total": 1500, "input": 800, "output": 700},
            success=True,
            error=None,
            timestamp=timestamp,
            context_size=2048,
            llm_calls=3,
            cache_hits=2,
            cache_misses=1,
        )

        assert profile.execution_id == "exec_123"
        assert profile.query == "Test query for profiling"
        assert profile.execution_mode == "langgraph-real"
        assert len(profile.agents) == 2
        assert profile.total_duration == 3.5
        assert profile.agent_durations["refiner"] == 1.5
        assert profile.memory_usage["peak"] == 256.0
        assert profile.cpu_usage["average"] == 65.0
        assert profile.token_usage["total"] == 1500
        assert profile.success is True
        assert profile.context_size == 2048
        assert profile.llm_calls == 3

    def test_execution_profile_failed(self):
        """Test ExecutionProfile for failed execution."""
        profile = ExecutionProfile(
            execution_id="exec_failed",
            query="Failing query",
            execution_mode="legacy",
            agents=["refiner"],
            total_duration=0.5,
            agent_durations={},
            memory_usage={},
            cpu_usage={},
            token_usage={},
            success=False,
            error="Agent execution timeout",
        )

        assert profile.success is False
        assert profile.error == "Agent execution timeout"
        assert len(profile.agent_durations) == 0


class TestBenchmarkSuite:
    """Test suite for BenchmarkSuite dataclass."""

    def test_benchmark_suite_creation(self):
        """Test BenchmarkSuite creation."""
        timestamp = datetime.now(timezone.utc)
        profile1 = ExecutionProfile(
            execution_id="bench_1",
            query="Benchmark query 1",
            execution_mode="langgraph",
            agents=["refiner"],
            total_duration=1.0,
            agent_durations={"refiner": 1.0},
            memory_usage={},
            cpu_usage={},
            token_usage={},
            success=True,
        )

        suite = BenchmarkSuite(
            suite_name="Performance Benchmark Suite",
            total_runs=10,
            execution_modes=["legacy", "langgraph", "langgraph-real"],
            queries=["Query 1", "Query 2", "Query 3"],
            profiles=[profile1],
            summary_stats={"avg_duration": 1.5, "success_rate": 0.95},
            timestamp=timestamp,
        )

        assert suite.suite_name == "Performance Benchmark Suite"
        assert suite.total_runs == 10
        assert len(suite.execution_modes) == 3
        assert len(suite.queries) == 3
        assert len(suite.profiles) == 1
        assert suite.summary_stats["avg_duration"] == 1.5
        assert suite.timestamp == timestamp


class TestPerformanceProfiler:
    """Test suite for PerformanceProfiler class."""

    @pytest.fixture
    def profiler(self):
        """Create a PerformanceProfiler instance for testing."""
        return PerformanceProfiler()

    @pytest.fixture
    def sample_profile(self):
        """Create a sample execution profile for testing."""
        return ExecutionProfile(
            execution_id="sample_123",
            query="Sample profiling query",
            execution_mode="langgraph-real",
            agents=["refiner", "critic", "synthesis"],
            total_duration=4.2,
            agent_durations={"refiner": 1.2, "critic": 1.5, "synthesis": 1.5},
            memory_usage={"peak": 300.0, "average": 250.0, "final": 200.0},
            cpu_usage={"peak": 90.0, "average": 70.0, "final": 50.0},
            token_usage={"total": 2000, "input": 1000, "output": 1000},
            success=True,
            context_size=4096,
            llm_calls=5,
        )

    def test_initialization(self, profiler):
        """Test PerformanceProfiler initialization."""
        assert profiler.console is not None
        assert profiler.monitoring is False
        assert profiler.current_profile is None
        assert profiler.resource_monitor_thread is None
        assert len(profiler.resource_data) == 0

    def test_create_app(self, profiler):
        """Test CLI app creation."""
        app = profiler.create_app()
        assert app is not None
        assert app.info.name == "profiler"

    @patch("cognivault.diagnostics.profiler.LangGraphOrchestrator")
    def test_profile_execution_basic(self, mock_orchestrator, profiler):
        """Test basic execution profiling."""
        with patch.object(profiler, "_run_profiling_suite") as mock_suite:
            with patch.object(profiler, "_display_profile_results"):
                mock_profiles = [
                    ExecutionProfile(
                        execution_id="test_1",
                        query="test",
                        execution_mode="langgraph-real",
                        agents=["refiner"],
                        total_duration=1.0,
                        agent_durations={},
                        memory_usage={},
                        cpu_usage={},
                        token_usage={},
                        success=True,
                    )
                ]
                mock_suite.return_value = mock_profiles

                profiler.profile_execution(
                    query="test query",
                    agents="refiner",
                    execution_mode="langgraph-real",
                    runs=1,
                    detailed=False,
                    live_monitor=False,
                    output_file=None,
                )

                mock_suite.assert_called_once()

    def test_profile_execution_with_live_monitoring(self, profiler):
        """Test profiling with live monitoring."""
        with patch.object(profiler, "_profile_with_live_monitoring") as mock_live:
            profiler.profile_execution(
                query="live test",
                agents="refiner,critic",
                execution_mode="langgraph",
                runs=3,
                detailed=True,
                live_monitor=True,
                output_file=None,
            )

            mock_live.assert_called_once()

    def test_profile_execution_with_output_file(self, profiler, tmp_path):
        """Test profiling with output file."""
        output_file = tmp_path / "profile_output.json"

        with patch.object(profiler, "_run_profiling_suite") as mock_suite:
            with patch.object(profiler, "_display_profile_results"):
                with patch.object(profiler, "_save_profiles") as mock_save:
                    mock_profiles = [
                        ExecutionProfile(
                            execution_id="save_test",
                            query="save test",
                            execution_mode="legacy",
                            agents=["refiner"],
                            total_duration=1.0,
                            agent_durations={},
                            memory_usage={},
                            cpu_usage={},
                            token_usage={},
                            success=True,
                        )
                    ]
                    mock_suite.return_value = mock_profiles

                    profiler.profile_execution(
                        query="save test",
                        agents="refiner",
                        runs=1,
                        detailed=False,
                        live_monitor=False,
                        execution_mode="langgraph-real",
                        output_file=str(output_file),
                    )

                    mock_save.assert_called_once()

    def test_run_benchmark_basic(self, profiler):
        """Test basic benchmark execution."""
        with patch.object(profiler, "_load_benchmark_queries") as mock_load:
            with patch.object(profiler, "_execute_benchmark_suite") as mock_execute:
                with patch.object(profiler, "_display_benchmark_results"):
                    with patch.object(profiler, "_save_benchmark_suite"):
                        mock_load.return_value = ["Query 1", "Query 2"]
                        mock_suite = BenchmarkSuite(
                            suite_name="test_suite",
                            total_runs=6,
                            execution_modes=["langgraph"],
                            queries=["Query 1", "Query 2"],
                            profiles=[],
                            summary_stats={},
                        )
                        mock_execute.return_value = mock_suite

                        profiler.run_benchmark(
                            queries_file=None,
                            agents="refiner,critic",
                            modes="langgraph",
                            runs_per_query=3,
                            warmup_runs=0,
                        )

                        mock_execute.assert_called_once()

    def test_run_benchmark_with_queries_file(self, profiler, tmp_path):
        """Test benchmark with custom queries file."""
        queries_file = tmp_path / "queries.txt"
        queries_file.write_text("Custom query 1\nCustom query 2\n")

        with patch.object(profiler, "_execute_benchmark_suite") as mock_execute:
            with patch.object(profiler, "_display_benchmark_results"):
                with patch.object(profiler, "_save_benchmark_suite"):
                    mock_suite = BenchmarkSuite(
                        suite_name="custom_suite",
                        total_runs=4,
                        execution_modes=["langgraph-real"],
                        queries=["Custom query 1", "Custom query 2"],
                        profiles=[],
                        summary_stats={},
                    )
                    mock_execute.return_value = mock_suite

                    profiler.run_benchmark(
                        queries_file=str(queries_file),
                        agents="synthesis",
                        modes="langgraph-real",
                        runs_per_query=2,
                    )

    def test_compare_modes(self, profiler):
        """Test execution mode comparison."""
        with patch.object(profiler, "_run_mode_comparison") as mock_compare:
            with patch.object(profiler, "_display_comparison_results"):
                mock_compare.return_value = {
                    "legacy": {"avg_duration": 2.0, "success_rate": 0.9},
                    "langgraph": {"avg_duration": 1.5, "success_rate": 0.95},
                }

                profiler.compare_modes(
                    query="comparison test",
                    modes="legacy,langgraph",
                    agents="refiner,critic",
                    runs=5,
                    statistical=True,
                    visual=False,
                )

                mock_compare.assert_called_once()

    def test_compare_modes_with_visual(self, profiler):
        """Test mode comparison with visual output."""
        with patch.object(profiler, "_run_mode_comparison") as mock_compare:
            with patch.object(profiler, "_display_comparison_results"):
                with patch.object(
                    profiler, "_generate_visual_comparison"
                ) as mock_visual:
                    mock_compare.return_value = {"test": "data"}

                    profiler.compare_modes(
                        query="visual test",
                        modes="langgraph,langgraph-real",
                        agents="refiner,critic",
                        runs=3,
                        visual=True,
                    )

                    mock_visual.assert_called_once()

    def test_monitor_performance(self, profiler):
        """Test performance monitoring."""
        with patch.object(profiler, "_start_performance_monitoring") as mock_monitor:
            with patch.object(profiler, "_display_monitoring_results"):
                mock_monitor.return_value = [{"timestamp": time.time(), "cpu": 50.0}]

                profiler.monitor_performance(
                    duration=60, interval=1.0, agents="refiner,critic", output_file=None
                )

                mock_monitor.assert_called_once()

    def test_monitor_performance_with_output(self, profiler, tmp_path):
        """Test performance monitoring with output file."""
        output_file = tmp_path / "monitoring.json"

        with patch.object(profiler, "_start_performance_monitoring") as mock_monitor:
            with patch.object(profiler, "_display_monitoring_results"):
                with patch.object(profiler, "_save_monitoring_data") as mock_save:
                    mock_monitor.return_value = [{"data": "test"}]

                    profiler.monitor_performance(
                        duration=30, output_file=str(output_file)
                    )

                    mock_save.assert_called_once()

    def test_analyze_profile(self, profiler, tmp_path):
        """Test profile analysis."""
        profile_file = tmp_path / "profile.json"
        profile_file.write_text('{"test": "data"}')

        with patch.object(profiler, "_load_profile_data") as mock_load:
            with patch.object(profiler, "_perform_profile_analysis") as mock_analyze:
                with patch.object(profiler, "_display_analysis_results"):
                    with patch.object(profiler, "_generate_optimization_insights"):
                        mock_load.return_value = {"test": "data"}
                        mock_analyze.return_value = {"analysis": "complete"}

                        profiler.analyze_profile(
                            profile_file=str(profile_file),
                            analysis_type="comprehensive",
                            generate_insights=True,
                        )

                        mock_analyze.assert_called_once()

    def test_analyze_profile_with_comparison(self, profiler, tmp_path):
        """Test profile analysis with comparison."""
        profile_file = tmp_path / "profile1.json"
        compare_file = tmp_path / "profile2.json"
        profile_file.write_text('{"profile": 1}')
        compare_file.write_text('{"profile": 2}')

        with patch.object(profiler, "_load_profile_data") as mock_load:
            with patch.object(profiler, "_perform_profile_analysis") as mock_analyze:
                with patch.object(profiler, "_compare_profiles") as mock_compare:
                    with patch.object(profiler, "_display_analysis_results"):
                        with patch.object(profiler, "_display_profile_comparison"):
                            mock_load.side_effect = [{"profile": 1}, {"profile": 2}]
                            mock_analyze.return_value = {"analysis": "done"}
                            mock_compare.return_value = {"comparison": "complete"}

                            profiler.analyze_profile(
                                profile_file=str(profile_file),
                                compare_with=str(compare_file),
                            )

                            mock_compare.assert_called_once()

    def test_generate_report(self, profiler, tmp_path):
        """Test report generation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch.object(profiler, "_compile_report_data") as mock_compile:
            with patch.object(
                profiler, "_generate_performance_report"
            ) as mock_generate:
                with patch.object(profiler, "_save_report") as mock_save:
                    mock_compile.return_value = {"data": "compiled"}
                    mock_generate.return_value = "# Performance Report\nContent here"

                    profiler.generate_report(
                        data_dir=str(data_dir),
                        output_format="markdown",
                        include_graphs=True,
                    )

                    mock_generate.assert_called_once()
                    mock_save.assert_called_once()

    def test_suggest_optimizations(self, profiler, tmp_path):
        """Test optimization suggestions."""
        profile_file = tmp_path / "optimization_profile.json"
        profile_file.write_text('{"optimization": "test"}')

        with patch.object(profiler, "_load_profile_data") as mock_load:
            with patch.object(
                profiler, "_generate_optimization_suggestions"
            ) as mock_suggest:
                with patch.object(profiler, "_display_optimization_suggestions"):
                    mock_load.return_value = {"optimization": "test"}
                    mock_suggest.return_value = ["Optimize loops", "Cache results"]

                    profiler.suggest_optimizations(
                        profile_file=str(profile_file),
                        target_metric="latency",
                        confidence_threshold=0.8,
                    )

                    mock_suggest.assert_called_once()

    def test_suggest_optimizations_interactive(self, profiler, tmp_path):
        """Test interactive optimization suggestions."""
        profile_file = tmp_path / "interactive_profile.json"
        profile_file.write_text('{"interactive": "test"}')

        with patch.object(profiler, "_load_profile_data") as mock_load:
            with patch.object(
                profiler, "_generate_optimization_suggestions"
            ) as mock_suggest:
                with patch.object(
                    profiler, "_interactive_optimization_session"
                ) as mock_interactive:
                    mock_load.return_value = {"interactive": "test"}
                    mock_suggest.return_value = ["Interactive suggestion"]

                    profiler.suggest_optimizations(
                        profile_file=str(profile_file), interactive=True
                    )

                    mock_interactive.assert_called_once()

    def test_stress_test(self, profiler):
        """Test stress testing."""
        with patch.object(profiler, "_execute_stress_test") as mock_stress:
            with patch.object(profiler, "_display_stress_test_results"):
                mock_stress.return_value = {
                    "concurrent_requests": 10,
                    "total_requests": 100,
                    "avg_response_time": 1.2,
                    "success_rate": 0.95,
                }

                profiler.stress_test(
                    query="stress test query",
                    concurrent_requests=10,
                    total_requests=100,
                    ramp_up_time=5,
                    agents="refiner,critic",
                )

                mock_stress.assert_called_once()

    def test_run_profiling_suite(self, profiler):
        """Test running profiling suite."""
        with patch.object(profiler, "_profile_single_execution") as mock_single:
            mock_profile = ExecutionProfile(
                execution_id="suite_test",
                query="suite query",
                execution_mode="langgraph",
                agents=["refiner"],
                total_duration=1.0,
                agent_durations={},
                memory_usage={},
                cpu_usage={},
                token_usage={},
                success=True,
            )
            mock_single.return_value = mock_profile

            results = profiler._run_profiling_suite(
                query="suite query",
                agents=["refiner"],
                mode="langgraph",
                runs=3,
                detailed=False,
            )

            assert len(results) == 3
            assert all(isinstance(p, ExecutionProfile) for p in results)

    @patch("asyncio.run")
    def test_profile_single_execution_legacy(self, mock_asyncio, profiler):
        """Test single execution profiling in legacy mode."""
        mock_context = AgentContext(query="test")
        mock_asyncio.return_value = mock_context

        with patch.object(profiler, "_start_resource_monitoring"):
            with patch.object(profiler, "_stop_resource_monitoring") as mock_stop:
                mock_stop.return_value = {
                    "memory": {"peak": 100.0},
                    "cpu": {"peak": 50.0},
                }

                profile = profiler._profile_single_execution(
                    query="legacy test",
                    agents=["refiner"],
                    mode="legacy",
                    detailed=True,
                )

                assert isinstance(profile, ExecutionProfile)
                assert profile.execution_mode == "legacy"

    @patch("asyncio.run")
    def test_profile_single_execution_langgraph_real(self, mock_asyncio, profiler):
        """Test single execution profiling in langgraph-real mode."""
        mock_context = AgentContext(query="test")
        mock_asyncio.return_value = mock_context

        profile = profiler._profile_single_execution(
            query="real test",
            agents=["refiner", "critic"],
            mode="langgraph-real",
            detailed=False,
        )

        assert isinstance(profile, ExecutionProfile)
        assert profile.execution_mode == "langgraph-real"
        assert "refiner" in profile.agents or "critic" in profile.agents

    def test_start_stop_resource_monitoring(self, profiler):
        """Test resource monitoring start and stop."""
        # Start monitoring
        profiler._start_resource_monitoring()
        assert profiler.monitoring is True
        assert profiler.resource_monitor_thread is not None

        # Give it a moment to collect some data
        time.sleep(0.1)

        # Stop monitoring
        resource_data = profiler._stop_resource_monitoring()
        assert profiler.monitoring is False
        assert isinstance(resource_data, dict)
        assert "memory" in resource_data
        assert "cpu" in resource_data

    def test_extract_agent_durations(self, profiler):
        """Test extracting agent durations from context."""
        context = AgentContext(query="test")
        context.agent_outputs = {"refiner": "output1", "critic": "output2"}

        durations = profiler._extract_agent_durations(context)

        assert isinstance(durations, dict)
        assert "refiner" in durations
        assert "critic" in durations

    def test_extract_token_usage(self, profiler):
        """Test extracting token usage from context."""
        context = AgentContext(query="test")

        token_usage = profiler._extract_token_usage(context)

        assert isinstance(token_usage, dict)
        assert "total" in token_usage
        assert "input" in token_usage
        assert "output" in token_usage

    def test_count_llm_calls(self, profiler):
        """Test counting LLM calls from context."""
        context = AgentContext(query="test")
        context.agent_outputs = {"refiner": "output1", "critic": "output2"}

        llm_calls = profiler._count_llm_calls(context)

        assert isinstance(llm_calls, int)
        assert llm_calls >= 0

    def test_display_profile_results(self, profiler, sample_profile):
        """Test displaying profile results."""
        profiles = [sample_profile]

        with patch.object(profiler.console, "print") as mock_print:
            profiler._display_profile_results(profiles)
            mock_print.assert_called()

    def test_display_profile_results_empty(self, profiler):
        """Test displaying empty profile results."""
        profiler._display_profile_results([])
        # Should not raise any exceptions

    def test_load_benchmark_queries_from_file(self, profiler, tmp_path):
        """Test loading benchmark queries from file."""
        queries_file = tmp_path / "test_queries.txt"
        queries_file.write_text("Query 1\nQuery 2\nQuery 3\n")

        queries = profiler._load_benchmark_queries(str(queries_file))

        assert len(queries) == 3
        assert "Query 1" in queries
        assert "Query 2" in queries
        assert "Query 3" in queries

    def test_load_benchmark_queries_default(self, profiler):
        """Test loading default benchmark queries."""
        queries = profiler._load_benchmark_queries(None)

        assert isinstance(queries, list)
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)

    @patch("cognivault.diagnostics.profiler.LangGraphOrchestrator")
    def test_execute_benchmark_suite(self, mock_orchestrator, profiler):
        """Test executing benchmark suite."""
        # Create non-async mock orchestrator
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.run = Mock(return_value=AgentContext(query="test"))
        mock_orchestrator.return_value = mock_orchestrator_instance

        with patch.object(profiler, "_profile_single_execution") as mock_profile:
            mock_profile.return_value = ExecutionProfile(
                execution_id="bench_exec",
                query="bench query",
                execution_mode="langgraph",
                agents=["refiner"],
                total_duration=1.0,
                agent_durations={},
                memory_usage={},
                cpu_usage={},
                token_usage={},
                success=True,
            )

            suite = profiler._execute_benchmark_suite(
                suite_name="test_bench",
                queries=["Query 1", "Query 2"],
                agents=["refiner"],
                modes=["langgraph"],
                runs_per_query=2,
                warmup_runs=1,
            )

            assert isinstance(suite, BenchmarkSuite)
            assert suite.suite_name == "test_bench"
            assert len(suite.profiles) == 4  # 2 queries * 2 runs

    def test_calculate_benchmark_summary(self, profiler, sample_profile):
        """Test calculating benchmark summary statistics."""
        profiles = [sample_profile, sample_profile]  # Duplicate for testing

        summary = profiler._calculate_benchmark_summary(profiles)

        assert isinstance(summary, dict)
        assert "overall" in summary
        assert "by_mode" in summary
        assert "total_runs" in summary["overall"]
        assert "avg_duration" in summary["overall"]

    def test_calculate_benchmark_summary_empty(self, profiler):
        """Test calculating summary with empty profiles."""
        summary = profiler._calculate_benchmark_summary([])

        assert summary == {}

    def test_display_benchmark_results(self, profiler):
        """Test displaying benchmark results."""
        suite = BenchmarkSuite(
            suite_name="Display Test",
            total_runs=5,
            execution_modes=["langgraph"],
            queries=["Test query"],
            profiles=[],
            summary_stats={
                "overall": {
                    "total_runs": 5,
                    "avg_duration": 1.5,
                    "success_rate": 0.95,
                    "throughput": 40.0,
                },
                "by_mode": {
                    "langgraph": {"count": 5, "avg_duration": 1.5, "success_rate": 0.95}
                },
            },
        )

        with patch.object(profiler.console, "print") as mock_print:
            profiler._display_benchmark_results(suite)
            mock_print.assert_called()

    def test_save_profiles(self, profiler, sample_profile, tmp_path):
        """Test saving profiles to file."""
        output_file = tmp_path / "saved_profiles.json"
        profiles = [sample_profile]

        profiler._save_profiles(profiles, str(output_file))

        assert output_file.exists()

        with open(output_file, "r") as f:
            data = json.load(f)
            assert "timestamp" in data
            assert "profiles" in data
            assert len(data["profiles"]) == 1

    def test_save_benchmark_suite(self, profiler, tmp_path):
        """Test saving benchmark suite."""
        suite = BenchmarkSuite(
            suite_name="Save Test Suite",
            total_runs=3,
            execution_modes=["langgraph-real"],
            queries=["Save query"],
            profiles=[],
            summary_stats={"test": "data"},
        )

        output_dir = tmp_path / "benchmark_output"

        profiler._save_benchmark_suite(suite, str(output_dir))

        assert output_dir.exists()
        results_file = output_dir / f"{suite.suite_name}_results.json"
        assert results_file.exists()


# Integration tests
class TestPerformanceProfilerIntegration:
    """Integration tests for PerformanceProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create profiler for integration tests."""
        return PerformanceProfiler()

    def test_full_profiling_workflow(self, profiler):
        """Test complete profiling workflow."""
        with patch("cognivault.diagnostics.profiler.LangGraphOrchestrator"):
            with patch.object(profiler, "_run_profiling_suite") as mock_suite:
                mock_suite.return_value = [
                    ExecutionProfile(
                        execution_id="integration_test",
                        query="integration query",
                        execution_mode="langgraph-real",
                        agents=["refiner"],
                        total_duration=1.0,
                        agent_durations={},
                        memory_usage={},
                        cpu_usage={},
                        token_usage={},
                        success=True,
                    )
                ]

                # Should not raise exceptions
                profiler.profile_execution(
                    query="integration test",
                    agents="refiner",
                    runs=1,
                    detailed=False,
                    live_monitor=False,
                    execution_mode="langgraph-real",
                    output_file=None,
                )

    def test_cli_app_integration(self, profiler):
        """Test CLI app creation and commands."""
        app = profiler.create_app()

        # Test that app was created successfully
        assert app is not None
        assert app.info.name == "profiler"


# Performance tests
class TestPerformanceProfilerPerformance:
    """Performance tests for PerformanceProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create profiler for performance tests."""
        return PerformanceProfiler()

    def test_resource_monitoring_performance(self, profiler):
        """Test performance of resource monitoring."""
        start_time = time.time()

        profiler._start_resource_monitoring()
        time.sleep(0.1)  # Monitor for short time
        resource_data = profiler._stop_resource_monitoring()

        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 1.0
        assert isinstance(resource_data, dict)

    def test_benchmark_summary_performance(self, profiler):
        """Test performance of benchmark summary calculation."""
        # Create many profiles for performance testing
        profiles = []
        for i in range(100):
            profiles.append(
                ExecutionProfile(
                    execution_id=f"perf_test_{i}",
                    query=f"performance query {i}",
                    execution_mode="langgraph-real",
                    agents=["refiner"],
                    total_duration=float(i % 5 + 1),  # Vary duration
                    agent_durations={},
                    memory_usage={},
                    cpu_usage={},
                    token_usage={},
                    success=True,
                )
            )

        start_time = time.time()
        summary = profiler._calculate_benchmark_summary(profiles)
        end_time = time.time()

        # Should complete quickly even with many profiles
        assert (end_time - start_time) < 1.0
        assert isinstance(summary, dict)
        assert summary["overall"]["total_runs"] == 100
