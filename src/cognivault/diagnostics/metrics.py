"""
Performance metrics collection and reporting system.

This module provides comprehensive metrics collection for CogniVault operations,
including execution timing, resource usage, token consumption, and success rates.
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Deque
import statistics

from pydantic import BaseModel, Field, ConfigDict


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class MetricEntry(BaseModel):
    """Individual metric entry with timestamp."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(..., description="Timestamp when metric was recorded")
    labels: Dict[str, str] = Field(
        default_factory=dict, description="Labels/tags associated with this metric"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


class PerformanceMetrics(BaseModel):
    """Aggregated performance metrics for analysis and reporting."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Execution metrics
    total_executions: int = Field(
        default=0, ge=0, description="Total number of executions"
    )
    successful_executions: int = Field(
        default=0, ge=0, description="Number of successful executions"
    )
    failed_executions: int = Field(
        default=0, ge=0, description="Number of failed executions"
    )
    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Success rate (0.0-1.0)"
    )

    # Timing metrics
    total_execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total execution time in milliseconds"
    )
    average_execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average execution time in milliseconds"
    )
    min_execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Minimum execution time in milliseconds"
    )
    max_execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Maximum execution time in milliseconds"
    )
    p50_execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="50th percentile execution time in milliseconds",
    )
    p95_execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="95th percentile execution time in milliseconds",
    )
    p99_execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="99th percentile execution time in milliseconds",
    )

    # Agent-specific metrics
    agent_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-agent performance metrics"
    )

    # Resource metrics
    peak_memory_usage_bytes: int = Field(
        default=0, ge=0, description="Peak memory usage in bytes"
    )
    total_tokens_consumed: int = Field(
        default=0, ge=0, description="Total tokens consumed"
    )
    total_tokens_generated: int = Field(
        default=0, ge=0, description="Total tokens generated"
    )
    llm_api_calls: int = Field(default=0, ge=0, description="Total LLM API calls made")

    # LLM metrics
    successful_llm_calls: int = Field(
        default=0, ge=0, description="Number of successful LLM calls"
    )
    failed_llm_calls: int = Field(
        default=0, ge=0, description="Number of failed LLM calls"
    )
    average_agent_duration: float = Field(
        default=0.0, ge=0.0, description="Average agent execution duration"
    )
    average_llm_duration: float = Field(
        default=0.0, ge=0.0, description="Average LLM call duration"
    )
    pipeline_duration: float = Field(
        default=0.0, ge=0.0, description="Pipeline execution duration"
    )

    # Error metrics
    error_breakdown: Dict[str, int] = Field(
        default_factory=dict, description="Breakdown of errors by type"
    )
    circuit_breaker_trips: int = Field(
        default=0, ge=0, description="Number of circuit breaker trips"
    )
    retry_attempts: int = Field(default=0, ge=0, description="Number of retry attempts")

    # Time window
    collection_start: Optional[datetime] = Field(
        None, description="Start of metrics collection period"
    )
    collection_end: Optional[datetime] = Field(
        None, description="End of metrics collection period"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "execution": {
                "total": self.total_executions,
                "successful": self.successful_executions,
                "failed": self.failed_executions,
                "success_rate": self.success_rate,
            },
            "timing_ms": {
                "total": self.total_execution_time_ms,
                "average": self.average_execution_time_ms,
                "min": self.min_execution_time_ms,
                "max": self.max_execution_time_ms,
                "p50": self.p50_execution_time_ms,
                "p95": self.p95_execution_time_ms,
                "p99": self.p99_execution_time_ms,
            },
            "agents": self.agent_metrics,
            "resources": {
                "peak_memory_bytes": self.peak_memory_usage_bytes,
                "total_tokens": self.total_tokens_consumed,
                "llm_api_calls": self.llm_api_calls,
            },
            "errors": {
                "breakdown": self.error_breakdown,
                "circuit_breaker_trips": self.circuit_breaker_trips,
                "retry_attempts": self.retry_attempts,
            },
            "collection_period": {
                "start": (
                    self.collection_start.isoformat() if self.collection_start else None
                ),
                "end": self.collection_end.isoformat() if self.collection_end else None,
            },
        }

    @classmethod
    def calculate_from_collector(
        cls, collector: "MetricsCollector", start_time: datetime, end_time: datetime
    ) -> "PerformanceMetrics":
        """Calculate performance metrics from a MetricsCollector."""
        # Calculate execution metrics
        total_agents = int(collector._get_counter_value("agent_executions_total"))
        successful_agents = int(
            collector._get_counter_value("agent_executions_successful")
        )
        failed_agents = int(collector._get_counter_value("agent_executions_failed"))

        # Calculate LLM metrics
        total_llm_calls = int(collector._get_counter_value("llm_api_calls_total"))
        successful_llm_calls = int(collector._get_counter_value("llm_calls_successful"))
        failed_llm_calls = int(collector._get_counter_value("llm_calls_failed"))

        # Calculate token metrics
        total_tokens_used = int(
            collector._get_counter_value("tokens_consumed")
            + collector._get_counter_value("llm_tokens_input")
        )
        total_tokens_generated = int(collector._get_counter_value("llm_tokens_output"))

        # Calculate timing metrics
        agent_timings = collector._get_histogram_values("agent_execution_duration")
        llm_timings = collector._get_histogram_values("llm_call_duration")
        pipeline_timings = collector._get_histogram_values(
            "pipeline_execution_duration"
        )

        average_agent_duration = (
            statistics.mean(agent_timings) if agent_timings else 0.0
        )
        average_llm_duration = statistics.mean(llm_timings) if llm_timings else 0.0
        pipeline_duration = (
            statistics.mean(pipeline_timings) if pipeline_timings else 0.0
        )

        # Calculate success rate
        success_rate = successful_agents / total_agents if total_agents > 0 else 0.0

        # Create metrics object with all fields
        metrics = cls(
            total_executions=total_agents,
            successful_executions=successful_agents,
            failed_executions=failed_agents,
            success_rate=success_rate,
            average_execution_time_ms=round(average_agent_duration, 2),
            total_tokens_consumed=total_tokens_used,
            total_tokens_generated=total_tokens_generated,
            llm_api_calls=total_llm_calls,
            successful_llm_calls=successful_llm_calls,
            failed_llm_calls=failed_llm_calls,
            average_agent_duration=round(average_agent_duration, 2),
            average_llm_duration=round(average_llm_duration, 2),
            pipeline_duration=round(pipeline_duration, 2),
            collection_start=start_time,
            collection_end=end_time,
        )

        return metrics


class MetricsCollector:
    """
    Thread-safe metrics collector for CogniVault performance monitoring.

    Collects and aggregates metrics including:
    - Execution timing and success rates
    - Agent-specific performance data
    - Resource usage statistics
    - Error rates and types
    - LLM API usage
    """

    def __init__(self, max_entries: int = 10000) -> None:
        """
        Initialize metrics collector.

        Parameters
        ----------
        max_entries : int
            Maximum number of metric entries to keep in memory
        """
        self.max_entries = max_entries
        self._metrics: Dict[str, Deque[MetricEntry]] = defaultdict(
            lambda: deque(maxlen=max_entries)
        )
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        self._collection_start = datetime.now()

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.

        Parameters
        ----------
        name : str
            Counter name
        value : float
            Value to increment by
        labels : Dict[str, str], optional
            Additional labels for the metric
        """
        with self._lock:
            key = f"counter_{name}"
            self._counters[key] += value

            entry = MetricEntry(
                value=value, timestamp=datetime.now(), labels=labels or {}
            )
            self._metrics[key].append(entry)

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric value.

        Parameters
        ----------
        name : str
            Gauge name
        value : float
            Current value
        labels : Dict[str, str], optional
            Additional labels for the metric
        """
        with self._lock:
            key = f"gauge_{name}"
            self._gauges[key] = value

            entry = MetricEntry(
                value=value, timestamp=datetime.now(), labels=labels or {}
            )
            self._metrics[key].append(entry)

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a histogram metric value.

        Parameters
        ----------
        name : str
            Histogram name
        value : float
            Value to record
        labels : Dict[str, str], optional
            Additional labels for the metric
        """
        with self._lock:
            key = f"histogram_{name}"

            entry = MetricEntry(
                value=value, timestamp=datetime.now(), labels=labels or {}
            )
            self._metrics[key].append(entry)

    def record_timing(
        self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a timing metric.

        Parameters
        ----------
        name : str
            Timer name
        duration_ms : float
            Duration in milliseconds
        labels : Dict[str, str], optional
            Additional labels for the metric
        """
        with self._lock:
            key = f"timer_{name}"

            entry = MetricEntry(
                value=duration_ms, timestamp=datetime.now(), labels=labels or {}
            )
            self._metrics[key].append(entry)

    def timing_context(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> "TimingContext":
        """
        Context manager for timing operations.

        Parameters
        ----------
        name : str
            Timer name
        labels : Dict[str, str], optional
            Additional labels for the metric

        Returns
        -------
        TimingContext
            Context manager that records timing
        """
        return TimingContext(self, name, labels)

    def record_agent_execution(
        self,
        agent_name: str,
        success: bool,
        duration_ms: float,
        tokens_used: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record agent execution metrics.

        Parameters
        ----------
        agent_name : str
            Name of the agent
        success : bool
            Whether execution was successful
        duration_ms : float
            Execution duration in milliseconds
        tokens_used : int
            Number of tokens consumed
        error_type : str, optional
            Type of error if execution failed
        """
        labels = {"agent": agent_name}

        # Record basic metrics
        self.increment_counter("agent_executions_total", labels=labels)
        self.record_timing("agent_execution_duration", duration_ms, labels=labels)

        if success:
            self.increment_counter("agent_executions_successful", labels=labels)
        else:
            self.increment_counter("agent_executions_failed", labels=labels)
            if error_type:
                error_labels = dict(labels, error_type=error_type)
                self.increment_counter("agent_errors", labels=error_labels)

        if tokens_used > 0:
            self.increment_counter("tokens_consumed", tokens_used, labels=labels)
            self.increment_counter("llm_api_calls", labels=labels)

    def record_pipeline_execution(
        self,
        pipeline_id: str,
        success: bool,
        duration_ms: float,
        agents_executed: List[str],
        total_tokens: int = 0,
    ) -> None:
        """
        Record full pipeline execution metrics.

        Parameters
        ----------
        pipeline_id : str
            Unique identifier for the pipeline execution
        success : bool
            Whether pipeline execution was successful
        duration_ms : float
            Total pipeline duration in milliseconds
        agents_executed : List[str]
            List of agents that were executed
        total_tokens : int
            Total tokens consumed across all agents
        """
        labels = {"pipeline_id": pipeline_id}

        self.increment_counter("pipeline_executions_total", labels=labels)
        self.record_timing("pipeline_execution_duration", duration_ms, labels=labels)

        if success:
            self.increment_counter("pipeline_executions_successful", labels=labels)
        else:
            self.increment_counter("pipeline_executions_failed", labels=labels)

        self.set_gauge("pipeline_agents_count", len(agents_executed), labels=labels)

        if total_tokens > 0:
            self.increment_counter("pipeline_tokens_total", total_tokens, labels=labels)

    def record_circuit_breaker_trip(self, agent_name: str) -> None:
        """Record circuit breaker activation."""
        labels = {"agent": agent_name}
        self.increment_counter("circuit_breaker_trips", labels=labels)

    def record_retry_attempt(self, agent_name: str, attempt_number: int) -> None:
        """Record retry attempt."""
        labels = {"agent": agent_name, "attempt": str(attempt_number)}
        self.increment_counter("retry_attempts", labels=labels)

    def record_llm_call(
        self,
        model: str,
        success: bool,
        duration_ms: float,
        tokens_used: int = 0,
        tokens_generated: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record LLM API call metrics.

        Parameters
        ----------
        model : str
            The LLM model used
        success : bool
            Whether the call was successful
        duration_ms : float
            Call duration in milliseconds
        tokens_used : int
            Input tokens consumed
        tokens_generated : int
            Output tokens generated
        error_type : str, optional
            Type of error if call failed
        """
        labels = {"model": model}

        self.increment_counter("llm_api_calls_total", labels=labels)
        self.record_timing("llm_call_duration", duration_ms, labels=labels)

        if success:
            self.increment_counter("llm_calls_successful", labels=labels)
        else:
            self.increment_counter("llm_calls_failed", labels=labels)
            if error_type:
                error_labels = dict(labels, error_type=error_type)
                self.increment_counter("llm_errors", labels=error_labels)

        if tokens_used > 0:
            self.increment_counter("llm_tokens_input", tokens_used, labels=labels)
        if tokens_generated > 0:
            self.increment_counter("llm_tokens_output", tokens_generated, labels=labels)

    def get_counter(
        self, metric_name: str, labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Get counter value with optional labels."""
        if labels is None:
            return self._get_counter_value(metric_name)

        # Filter by labels
        key = f"counter_{metric_name}"
        if key not in self._metrics:
            return 0.0

        total = 0.0
        for entry in self._metrics[key]:
            if entry.labels == (labels or {}):
                total += entry.value

        return total

    def get_gauge(
        self, metric_name: str, labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Get current gauge value with optional labels."""
        key = f"gauge_{metric_name}"
        if key not in self._metrics or not self._metrics[key]:
            return 0.0

        if labels is None:
            # Return the most recent gauge value regardless of labels
            return self._metrics[key][-1].value

        # Find the most recent gauge value with matching labels
        for entry in reversed(self._metrics[key]):
            if entry.labels == (labels or {}):
                return entry.value

        return 0.0

    def get_histogram(
        self, metric_name: str, labels: Optional[Dict[str, str]] = None
    ) -> List[float]:
        """Get histogram values with optional labels."""
        if labels is None:
            return self._get_histogram_values(metric_name)

        # Filter by labels
        key = f"histogram_{metric_name}"
        if key not in self._metrics:
            return []

        values = []
        for entry in self._metrics[key]:
            if entry.labels == (labels or {}):
                values.append(entry.value)

        return values

    def get_timing(
        self, metric_name: str, labels: Optional[Dict[str, str]] = None
    ) -> List[float]:
        """Get timing values with optional labels."""
        if labels is None:
            return self._get_histogram_values(metric_name)

        # Filter by labels
        key = f"timer_{metric_name}"
        if key not in self._metrics:
            return []

        values = []
        for entry in self._metrics[key]:
            if entry.labels == (labels or {}):
                values.append(entry.value)

        return values

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics data."""
        result = {}
        for key, entries in self._metrics.items():
            if entries:
                result[key] = [
                    {
                        "value": e.value,
                        "timestamp": e.timestamp.isoformat(),
                        "labels": e.labels,
                    }
                    for e in entries
                ]
        return result

    def get_metrics_summary(
        self, window_minutes: Optional[int] = None
    ) -> PerformanceMetrics:
        """
        Get aggregated metrics summary.

        Parameters
        ----------
        window_minutes : int, optional
            Time window in minutes to consider (None for all time)

        Returns
        -------
        PerformanceMetrics
            Aggregated performance metrics
        """
        with self._lock:
            cutoff_time = None
            if window_minutes:
                cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

            # Aggregate execution metrics
            total_executions = self._get_counter_value(
                "agent_executions_total", cutoff_time
            )
            successful_executions = self._get_counter_value(
                "agent_executions_successful", cutoff_time
            )
            failed_executions = self._get_counter_value(
                "agent_executions_failed", cutoff_time
            )

            success_rate = 0.0
            if total_executions > 0:
                success_rate = successful_executions / total_executions

            # Aggregate timing metrics
            timing_values = self._get_histogram_values(
                "timer_agent_execution_duration", cutoff_time
            )
            timing_stats = self._calculate_timing_stats(timing_values)

            # Agent-specific metrics
            agent_metrics = self._get_agent_specific_metrics(cutoff_time)

            # Resource metrics
            total_tokens = self._get_counter_value("tokens_consumed", cutoff_time)
            llm_calls = self._get_counter_value("llm_api_calls", cutoff_time)

            # Error metrics
            error_breakdown = self._get_error_breakdown(cutoff_time)
            circuit_breaker_trips = self._get_counter_value(
                "circuit_breaker_trips", cutoff_time
            )
            retry_attempts = self._get_counter_value("retry_attempts", cutoff_time)

            return PerformanceMetrics(
                total_executions=int(total_executions),
                successful_executions=int(successful_executions),
                failed_executions=int(failed_executions),
                success_rate=success_rate,
                total_execution_time_ms=timing_stats["total"],
                average_execution_time_ms=timing_stats["average"],
                min_execution_time_ms=timing_stats["min"],
                max_execution_time_ms=timing_stats["max"],
                p50_execution_time_ms=timing_stats["p50"],
                p95_execution_time_ms=timing_stats["p95"],
                p99_execution_time_ms=timing_stats["p99"],
                agent_metrics=agent_metrics,
                total_tokens_consumed=int(total_tokens),
                llm_api_calls=int(llm_calls),
                error_breakdown=error_breakdown,
                circuit_breaker_trips=int(circuit_breaker_trips),
                retry_attempts=int(retry_attempts),
                collection_start=self._collection_start,
                collection_end=datetime.now(),
            )

    def _get_counter_value(
        self, metric_name: str, cutoff_time: Optional[datetime] = None
    ) -> float:
        """Get aggregated counter value, optionally within time window."""
        key = f"counter_{metric_name}"
        if key not in self._metrics:
            return 0.0

        total = 0.0
        for entry in self._metrics[key]:
            if cutoff_time is None or entry.timestamp >= cutoff_time:
                total += entry.value

        return total

    def _get_histogram_values(
        self, metric_name: str, cutoff_time: Optional[datetime] = None
    ) -> List[float]:
        """Get histogram values, optionally within time window."""
        key = f"histogram_{metric_name}"
        if key not in self._metrics:
            key = f"timer_{metric_name.replace('histogram_', '')}"  # Try timer variant

        if key not in self._metrics:
            return []

        values = []
        for entry in self._metrics[key]:
            if cutoff_time is None or entry.timestamp >= cutoff_time:
                values.append(entry.value)

        return values

    def _calculate_timing_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate timing statistics from values."""
        if not values:
            return {
                "total": 0.0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_values = sorted(values)

        return {
            "total": sum(values),
            "average": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": statistics.median(sorted_values),
            "p95": (
                sorted_values[int(len(sorted_values) * 0.95)]
                if len(sorted_values) > 1
                else sorted_values[0]
            ),
            "p99": (
                sorted_values[int(len(sorted_values) * 0.99)]
                if len(sorted_values) > 1
                else sorted_values[0]
            ),
        }

    def _get_agent_specific_metrics(
        self, cutoff_time: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get metrics broken down by agent."""
        agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "tokens_consumed": 0,
            }
        )

        # Collect agent-specific data
        for key, entries in self._metrics.items():
            for entry in entries:
                if cutoff_time and entry.timestamp < cutoff_time:
                    continue

                agent = entry.labels.get("agent")
                if not agent:
                    continue

                if "agent_executions_total" in key:
                    agent_metrics[agent]["executions"] += entry.value
                elif "agent_executions_successful" in key:
                    agent_metrics[agent]["successes"] += entry.value
                elif "agent_executions_failed" in key:
                    agent_metrics[agent]["failures"] += entry.value
                elif "tokens_consumed" in key:
                    agent_metrics[agent]["tokens_consumed"] += entry.value

        # Calculate derived metrics
        for agent, metrics in agent_metrics.items():
            if metrics["executions"] > 0:
                metrics["success_rate"] = metrics["successes"] / metrics["executions"]

            # Get timing data for this agent
            timing_values = []
            key = "timer_agent_execution_duration"
            if key in self._metrics:
                for entry in self._metrics[key]:
                    if (
                        cutoff_time is None or entry.timestamp >= cutoff_time
                    ) and entry.labels.get("agent") == agent:
                        timing_values.append(entry.value)

            if timing_values:
                metrics["avg_duration_ms"] = statistics.mean(timing_values)

        return dict(agent_metrics)

    def _get_error_breakdown(
        self, cutoff_time: Optional[datetime] = None
    ) -> Dict[str, int]:
        """Get error breakdown by type."""
        error_breakdown: Dict[str, int] = defaultdict(int)

        key = "counter_agent_errors"
        if key in self._metrics:
            for entry in self._metrics[key]:
                if cutoff_time is None or entry.timestamp >= cutoff_time:
                    error_type = entry.labels.get("error_type", "unknown")
                    error_breakdown[error_type] += int(entry.value)

        return dict(error_breakdown)

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._collection_start = datetime.now()

    def clear_metrics(self) -> None:
        """Clear all metrics (alias for reset_metrics)."""
        self.reset_metrics()


class TimingContext:
    """Context manager for recording timing metrics."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TimingContext":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timing(self.name, duration_ms, self.labels)


# Global metrics collector instance
_global_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector."""
    global _global_metrics_collector
    _global_metrics_collector = None
