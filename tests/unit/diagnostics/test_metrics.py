from typing import Any

"""
Tests for metrics collection functionality.

This module tests the MetricsCollector class, performance metrics,
and thread-safe metrics collection.
"""

import threading
from datetime import datetime, timedelta

from cognivault.diagnostics.metrics import (
    MetricsCollector,
    PerformanceMetrics,
    get_metrics_collector,
    reset_metrics_collector,
)


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.metrics = MetricsCollector()

    def test_counter_operations(self) -> None:
        """Test counter increment and retrieval."""
        # Test basic increment
        self.metrics.increment_counter("test_counter")
        assert self.metrics.get_counter("test_counter") == 1

        # Test increment with amount
        self.metrics.increment_counter("test_counter", 5)
        assert self.metrics.get_counter("test_counter") == 6

        # Test increment with labels
        self.metrics.increment_counter("labeled_counter", labels={"type": "test"})
        self.metrics.increment_counter("labeled_counter", labels={"type": "test"})
        assert self.metrics.get_counter("labeled_counter", labels={"type": "test"}) == 2

        # Different labels should create separate counters
        self.metrics.increment_counter("labeled_counter", labels={"type": "other"})
        assert (
            self.metrics.get_counter("labeled_counter", labels={"type": "other"}) == 1
        )
        assert self.metrics.get_counter("labeled_counter", labels={"type": "test"}) == 2

    def test_gauge_operations(self) -> None:
        """Test gauge set and retrieval."""
        # Test basic gauge setting
        self.metrics.set_gauge("test_gauge", 42.5)
        assert self.metrics.get_gauge("test_gauge") == 42.5

        # Test gauge overwriting
        self.metrics.set_gauge("test_gauge", 100.0)
        assert self.metrics.get_gauge("test_gauge") == 100.0

        # Test gauge with labels
        self.metrics.set_gauge("labeled_gauge", 10, labels={"service": "api"})
        self.metrics.set_gauge("labeled_gauge", 20, labels={"service": "db"})

        assert self.metrics.get_gauge("labeled_gauge", labels={"service": "api"}) == 10
        assert self.metrics.get_gauge("labeled_gauge", labels={"service": "db"}) == 20

    def test_histogram_operations(self) -> None:
        """Test histogram recording and retrieval."""
        # Record some values
        values = [10, 20, 30, 40, 50]
        for value in values:
            self.metrics.record_histogram("test_histogram", value)

        histogram = self.metrics.get_histogram("test_histogram")

        assert len(histogram) == 5
        assert histogram == values

        # Test histogram with labels
        self.metrics.record_histogram(
            "labeled_histogram", 100, labels={"method": "GET"}
        )
        self.metrics.record_histogram(
            "labeled_histogram", 200, labels={"method": "POST"}
        )

        get_histogram = self.metrics.get_histogram(
            "labeled_histogram", labels={"method": "GET"}
        )
        post_histogram = self.metrics.get_histogram(
            "labeled_histogram", labels={"method": "POST"}
        )

        assert get_histogram == [100]
        assert post_histogram == [200]

    def test_timing_operations(self) -> None:
        """Test timing recording and retrieval."""
        # Record some timings
        timings = [10.5, 20.0, 15.2]
        for timing in timings:
            self.metrics.record_timing("test_timing", timing)

        recorded_timings = self.metrics.get_timing("test_timing")

        assert len(recorded_timings) == 3
        assert recorded_timings == timings

        # Test timing with labels
        self.metrics.record_timing("labeled_timing", 50.0, labels={"operation": "read"})
        self.metrics.record_timing(
            "labeled_timing", 75.0, labels={"operation": "write"}
        )

        read_timings = self.metrics.get_timing(
            "labeled_timing", labels={"operation": "read"}
        )
        write_timings = self.metrics.get_timing(
            "labeled_timing", labels={"operation": "write"}
        )

        assert read_timings == [50.0]
        assert write_timings == [75.0]

    def test_nonexistent_metrics(self) -> None:
        """Test retrieving nonexistent metrics."""
        # Should return 0 for counters
        assert self.metrics.get_counter("nonexistent") == 0

        # Should return 0 for gauges
        assert self.metrics.get_gauge("nonexistent") == 0

        # Should return empty lists for histograms and timings
        assert self.metrics.get_histogram("nonexistent") == []
        assert self.metrics.get_timing("nonexistent") == []

    def test_agent_execution_recording(self) -> None:
        """Test agent execution recording."""
        # Record successful execution
        self.metrics.record_agent_execution("refiner", True, 150.5, tokens_used=100)

        # Check counters
        assert (
            self.metrics.get_counter(
                "agent_executions_total", labels={"agent": "refiner"}
            )
            == 1
        )
        assert (
            self.metrics.get_counter(
                "agent_executions_successful", labels={"agent": "refiner"}
            )
            == 1
        )
        assert (
            self.metrics.get_counter(
                "agent_executions_failed", labels={"agent": "refiner"}
            )
            == 0
        )

        # Check timings
        timings = self.metrics.get_timing(
            "agent_execution_duration", labels={"agent": "refiner"}
        )
        assert timings == [150.5]

        # Check token usage
        assert (
            self.metrics.get_counter("tokens_consumed", labels={"agent": "refiner"})
            == 100
        )

        # Record failed execution
        self.metrics.record_agent_execution(
            "refiner", False, 75.0, error_type="ValidationError"
        )

        # Check updated counters
        assert (
            self.metrics.get_counter(
                "agent_executions_total", labels={"agent": "refiner"}
            )
            == 2
        )
        assert (
            self.metrics.get_counter(
                "agent_executions_successful", labels={"agent": "refiner"}
            )
            == 1
        )
        assert (
            self.metrics.get_counter(
                "agent_executions_failed", labels={"agent": "refiner"}
            )
            == 1
        )

        # Check error type counter
        assert (
            self.metrics.get_counter(
                "agent_errors",
                labels={"agent": "refiner", "error_type": "ValidationError"},
            )
            == 1
        )

    def test_llm_call_recording(self) -> None:
        """Test LLM call recording."""
        # Record LLM call
        self.metrics.record_llm_call(
            "gpt-4", True, 250.0, tokens_used=150, tokens_generated=50
        )

        # Check counters
        assert (
            self.metrics.get_counter("llm_api_calls_total", labels={"model": "gpt-4"})
            == 1
        )
        assert (
            self.metrics.get_counter("llm_calls_successful", labels={"model": "gpt-4"})
            == 1
        )
        assert (
            self.metrics.get_counter("llm_calls_failed", labels={"model": "gpt-4"}) == 0
        )

        # Check timings
        timings = self.metrics.get_timing(
            "llm_call_duration", labels={"model": "gpt-4"}
        )
        assert timings == [250.0]

        # Check token usage
        assert (
            self.metrics.get_counter("llm_tokens_input", labels={"model": "gpt-4"})
            == 150
        )
        assert (
            self.metrics.get_counter("llm_tokens_output", labels={"model": "gpt-4"})
            == 50
        )

        # Record failed call
        self.metrics.record_llm_call("gpt-4", False, 100.0, error_type="RateLimitError")

        # Check updated counters
        assert (
            self.metrics.get_counter("llm_api_calls_total", labels={"model": "gpt-4"})
            == 2
        )
        assert (
            self.metrics.get_counter("llm_calls_successful", labels={"model": "gpt-4"})
            == 1
        )
        assert (
            self.metrics.get_counter("llm_calls_failed", labels={"model": "gpt-4"}) == 1
        )

        # Check error type counter
        assert (
            self.metrics.get_counter(
                "llm_errors", labels={"model": "gpt-4", "error_type": "RateLimitError"}
            )
            == 1
        )

    def test_pipeline_recording(self) -> None:
        """Test pipeline execution recording."""
        # Record pipeline execution
        self.metrics.record_pipeline_execution(
            "pipeline-123", True, 500.0, ["agent1", "agent2", "agent3"]
        )

        # Check counters
        assert self.metrics._get_counter_value("pipeline_executions_total") == 1
        assert self.metrics._get_counter_value("pipeline_executions_successful") == 1
        assert self.metrics._get_counter_value("pipeline_executions_failed") == 0

        # Check timings
        timings = self.metrics.get_timing("pipeline_execution_duration")
        assert timings == [500.0]

        # Check agent count gauge
        assert self.metrics.get_gauge("pipeline_agents_count") == 3

        # Record failed pipeline
        self.metrics.record_pipeline_execution(
            "pipeline-456", False, 200.0, ["agent1", "agent2"]
        )

        # Check updated counters
        assert self.metrics._get_counter_value("pipeline_executions_total") == 2
        assert self.metrics._get_counter_value("pipeline_executions_successful") == 1
        assert self.metrics._get_counter_value("pipeline_executions_failed") == 1

    def test_thread_safety(self) -> None:
        """Test that metrics collection is thread-safe."""
        results = []

        def worker(worker_id: int) -> None:
            """Worker function that increments counters."""
            for i in range(100):
                self.metrics.increment_counter("thread_test_counter")
                self.metrics.record_timing("thread_test_timing", worker_id * 10 + i)
            results.append(worker_id)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 5

        # Counter should be incremented 500 times (5 threads × 100 increments)
        assert self.metrics._get_counter_value("thread_test_counter") == 500

        # Timing should have 500 entries
        timings = self.metrics.get_timing("thread_test_timing")
        assert len(timings) == 500

    def test_get_all_metrics(self) -> None:
        """Test retrieving all metrics."""
        # Add some metrics
        self.metrics.increment_counter("test_counter", 5)
        self.metrics.set_gauge("test_gauge", 42.0)
        self.metrics.record_histogram("test_histogram", 10)
        self.metrics.record_timing("test_timing", 15.5)

        all_metrics = self.metrics.get_all_metrics()

        assert "counter_test_counter" in all_metrics
        assert "gauge_test_gauge" in all_metrics
        assert "histogram_test_histogram" in all_metrics
        assert "timer_test_timing" in all_metrics

        # Check specific metrics - actual structure returns list of metric entries
        assert all_metrics["counter_test_counter"][0]["value"] == 5
        assert all_metrics["gauge_test_gauge"][0]["value"] == 42.0
        assert all_metrics["histogram_test_histogram"][0]["value"] == 10
        assert all_metrics["timer_test_timing"][0]["value"] == 15.5

    def test_clear_metrics(self) -> None:
        """Test clearing all metrics."""
        # Add some metrics
        self.metrics.increment_counter("test_counter", 5)
        self.metrics.set_gauge("test_gauge", 42.0)
        self.metrics.record_histogram("test_histogram", 10)
        self.metrics.record_timing("test_timing", 15.5)

        # Verify metrics exist
        assert self.metrics._get_counter_value("test_counter") == 5
        assert self.metrics.get_gauge("test_gauge") == 42.0

        # Clear metrics
        self.metrics.clear_metrics()

        # Verify metrics are cleared
        assert self.metrics._get_counter_value("test_counter") == 0
        assert self.metrics.get_gauge("test_gauge") == 0
        assert self.metrics._get_histogram_values("test_histogram") == []
        assert self.metrics.get_timing("test_timing") == []


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self) -> None:
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            total_executions=4,
            successful_executions=3,
            failed_executions=1,
        )
        assert metrics.total_executions == 4
        assert metrics.successful_executions == 3
        assert metrics.failed_executions == 1

    def test_performance_metrics_to_dict(self) -> None:
        """Test PerformanceMetrics to_dict method."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)

        metrics = PerformanceMetrics(
            total_executions=4,
            successful_executions=3,
            failed_executions=1,
            collection_start=start_time,
            collection_end=end_time,
        )

        result = metrics.to_dict()

        assert result["execution"]["total"] == 4
        assert result["execution"]["successful"] == 3
        assert result["execution"]["failed"] == 1
        assert result["collection_period"]["start"] == start_time.isoformat()
        assert result["collection_period"]["end"] == end_time.isoformat()

    def test_performance_metrics_calculate_from_collector(self) -> None:
        """Test calculating performance metrics from collector."""
        collector = MetricsCollector()

        # Add some test data
        collector.record_agent_execution("refiner", True, 100.0, tokens_used=50)
        collector.record_agent_execution("critic", True, 150.0, tokens_used=75)
        collector.record_agent_execution("synthesis", False, 75.0, tokens_used=25)

        collector.record_llm_call(
            "gpt-4", True, 50.0, tokens_used=100, tokens_generated=25
        )
        collector.record_llm_call(
            "gpt-4", True, 75.0, tokens_used=150, tokens_generated=40
        )
        collector.record_llm_call(
            "gpt-4", False, 25.0, tokens_used=50, tokens_generated=0
        )

        collector.record_pipeline_execution(
            "pipeline-1", True, 400.0, ["refiner", "critic", "synthesis"]
        )

        start_time = datetime.now() - timedelta(seconds=30)
        end_time = datetime.now()

        metrics = PerformanceMetrics.calculate_from_collector(
            collector, start_time, end_time
        )

        assert metrics.collection_start == start_time
        assert metrics.collection_end == end_time
        assert metrics.total_executions == 3
        assert metrics.successful_executions == 2
        assert metrics.failed_executions == 1
        assert metrics.llm_api_calls == 3
        assert metrics.successful_llm_calls == 2
        assert metrics.failed_llm_calls == 1
        assert metrics.total_tokens_consumed == 450  # 50 + 75 + 25 + 100 + 150 + 50
        assert metrics.total_tokens_generated == 65  # 25 + 40 + 0
        assert metrics.average_agent_duration == 108.33  # (100 + 150 + 75) / 3
        assert metrics.average_llm_duration == 50.0  # (50 + 75 + 25) / 3
        assert metrics.pipeline_duration == 400.0


class TestMetricsCollectorSingleton:
    """Test global metrics collector functionality."""

    def teardown_method(self) -> None:
        """Reset metrics collector after each test."""
        reset_metrics_collector()

    def test_get_metrics_collector_singleton(self) -> None:
        """Test that get_metrics_collector returns the same instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2
        assert isinstance(collector1, MetricsCollector)

    def test_metrics_collector_persistence(self) -> None:
        """Test that metrics persist across calls."""
        collector1 = get_metrics_collector()
        collector1.increment_counter("test_counter", 5)

        collector2 = get_metrics_collector()
        assert collector2.get_counter("test_counter") == 5

    def test_reset_metrics_collector(self) -> None:
        """Test resetting the global metrics collector."""
        collector1 = get_metrics_collector()
        collector1.increment_counter("test_counter", 5)

        assert collector1.get_counter("test_counter") == 5

        reset_metrics_collector()

        collector2 = get_metrics_collector()
        assert collector2.get_counter("test_counter") == 0

        # Should be a new instance
        assert collector1 is not collector2

    def test_metrics_collector_thread_safety_global(self) -> None:
        """Test global metrics collector thread safety."""
        results = []

        def worker(worker_id: int) -> None:
            """Worker function that uses global collector."""
            collector = get_metrics_collector()
            for i in range(50):
                collector.increment_counter("global_thread_test")
            results.append(worker_id)

        # Start multiple threads
        threads = []
        for i in range(4):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 4

        # Counter should be incremented 200 times (4 threads × 50 increments)
        collector = get_metrics_collector()
        assert collector.get_counter("global_thread_test") == 200
