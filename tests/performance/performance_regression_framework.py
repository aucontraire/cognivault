"""
Performance Regression Detection Framework

This framework provides:
1. Automated performance baseline establishment
2. Regression detection with configurable thresholds
3. Performance trend analysis and alerting
4. Integration with CI/CD for continuous monitoring
5. Detailed reporting for performance investigation

Designed to catch issues like the 4x RefinerAgent regression (82s vs 15s expected) early.
"""

import json
import time
import statistics
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

from cognivault.observability import get_logger

logger = get_logger("performance.regression_framework")


@dataclass
class PerformanceBaseline:
    """Performance baseline for a specific operation."""

    operation_name: str
    mean_duration_ms: float
    median_duration_ms: float
    std_dev_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    sample_count: int
    baseline_date: str
    environment: str
    version: str

    def is_regression(self, duration_ms: float, threshold_factor: float = 2.0) -> bool:
        """Check if a duration represents a performance regression."""
        # Use median + 2*std_dev as the upper bound for normal performance
        upper_bound = self.median_duration_ms + (threshold_factor * self.std_dev_ms)
        return duration_ms > upper_bound

    def regression_severity(self, duration_ms: float) -> str:
        """Classify regression severity."""
        median = self.median_duration_ms

        if duration_ms > median * 4.0:
            return "critical"  # 4x slower (like the observed regression)
        elif duration_ms > median * 2.0:
            return "major"
        elif duration_ms > median * 1.5:
            return "moderate"
        else:
            return "minor"


@dataclass
class PerformanceMeasurement:
    """Individual performance measurement."""

    operation_name: str
    duration_ms: float
    timestamp: str
    metadata: Dict[str, Any]
    success: bool
    error_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PerformanceRegression:
    """Detected performance regression."""

    def __init__(
        self, measurement: PerformanceMeasurement, baseline: PerformanceBaseline
    ):
        self.measurement = measurement
        self.baseline = baseline
        self.detected_at = datetime.now().isoformat()
        self.severity = baseline.regression_severity(measurement.duration_ms)
        self.factor = measurement.duration_ms / baseline.median_duration_ms

    def get_alert_data(self) -> Dict[str, Any]:
        """Get structured alert data for notification systems."""
        return {
            "alert_type": "performance_regression",
            "severity": self.severity,
            "operation": self.measurement.operation_name,
            "current_duration_ms": self.measurement.duration_ms,
            "baseline_median_ms": self.baseline.median_duration_ms,
            "regression_factor": self.factor,
            "detected_at": self.detected_at,
            "metadata": self.measurement.metadata,
        }


class PerformanceDatabase:
    """Storage and retrieval for performance data."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("tests/performance/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.baselines_file = self.data_dir / "baselines.json"
        self.measurements_file = self.data_dir / "measurements.json"
        self.regressions_file = self.data_dir / "regressions.json"

    def save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save performance baseline."""
        baselines = self.load_baselines()
        baselines[baseline.operation_name] = asdict(baseline)

        with open(self.baselines_file, "w") as f:
            json.dump(baselines, f, indent=2)

        logger.info(
            f"Saved baseline for {baseline.operation_name}: {baseline.median_duration_ms:.1f}ms median"
        )

    def load_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Load performance baselines."""
        if not self.baselines_file.exists():
            return {}

        with open(self.baselines_file, "r") as f:
            data: Dict[str, Dict[str, Any]] = json.load(f)
            return data

    def get_baseline(self, operation_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline for an operation."""
        baselines = self.load_baselines()

        if operation_name not in baselines:
            return None

        data = baselines[operation_name]
        return PerformanceBaseline(**data)

    def save_measurement(self, measurement: PerformanceMeasurement) -> None:
        """Save performance measurement."""
        measurements = self.load_measurements()
        measurements.append(measurement.to_dict())

        # Keep only last 1000 measurements per operation to prevent unbounded growth
        operation_measurements = [
            m for m in measurements if m["operation_name"] == measurement.operation_name
        ]
        if len(operation_measurements) > 1000:
            # Remove oldest measurements for this operation
            measurements = [
                m
                for m in measurements
                if m["operation_name"] != measurement.operation_name
            ]
            measurements.extend(operation_measurements[-1000:])

        with open(self.measurements_file, "w") as f:
            json.dump(measurements, f, indent=2)

    def load_measurements(self) -> List[Dict[str, Any]]:
        """Load performance measurements."""
        if not self.measurements_file.exists():
            return []

        with open(self.measurements_file, "r") as f:
            data: List[Dict[str, Any]] = json.load(f)
            return data

    def get_recent_measurements(
        self, operation_name: str, hours: int = 24
    ) -> List[PerformanceMeasurement]:
        """Get recent measurements for an operation."""
        measurements = self.load_measurements()
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = []
        for m_data in measurements:
            if m_data["operation_name"] != operation_name:
                continue

            measurement_time = datetime.fromisoformat(m_data["timestamp"])
            if measurement_time >= cutoff_time:
                recent.append(PerformanceMeasurement(**m_data))

        return sorted(recent, key=lambda x: x.timestamp)

    def load_regressions(self) -> List[Dict[str, Any]]:
        """Load detected regressions."""
        if not self.regressions_file.exists():
            return []

        with open(self.regressions_file, "r") as f:
            data: List[Dict[str, Any]] = json.load(f)
            return data

    def save_regression(self, regression: PerformanceRegression) -> None:
        """Save detected regression."""
        regressions = self.load_regressions()
        regressions.append(regression.get_alert_data())

        with open(self.regressions_file, "w") as f:
            json.dump(regressions, f, indent=2)

        logger.error(
            f"REGRESSION DETECTED: {regression.measurement.operation_name} - {regression.factor:.1f}x slower than baseline"
        )


class PerformanceMonitor:
    """Main performance monitoring and regression detection system."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.db = PerformanceDatabase(data_dir)
        self.regression_threshold = 2.0  # 2x slower triggers regression
        self.baseline_min_samples = 10  # Minimum samples to establish baseline

    def record_measurement(
        self,
        operation_name: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        error_details: Optional[str] = None,
    ) -> Optional[PerformanceRegression]:
        """
        Record a performance measurement and check for regressions.

        Returns:
            PerformanceRegression if regression detected, None otherwise
        """
        measurement = PerformanceMeasurement(
            operation_name=operation_name,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            success=success,
            error_details=error_details,
        )

        # Save measurement
        self.db.save_measurement(measurement)

        # Check for regression against baseline
        baseline = self.db.get_baseline(operation_name)
        if baseline and baseline.is_regression(duration_ms, self.regression_threshold):
            regression = PerformanceRegression(measurement, baseline)
            self.db.save_regression(regression)
            return regression

        return None

    def establish_baseline(
        self,
        operation_name: str,
        measurements: List[float],
        environment: str = "test",
        version: str = "unknown",
    ) -> PerformanceBaseline:
        """Establish performance baseline from measurement samples."""

        if len(measurements) < self.baseline_min_samples:
            raise ValueError(
                f"Need at least {self.baseline_min_samples} samples, got {len(measurements)}"
            )

        # Calculate statistics
        mean_duration = statistics.mean(measurements)
        median_duration = statistics.median(measurements)
        std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0

        # Calculate percentiles
        sorted_measurements = sorted(measurements)
        p95_idx = int(0.95 * len(sorted_measurements))
        p99_idx = int(0.99 * len(sorted_measurements))

        p95_duration = sorted_measurements[p95_idx]
        p99_duration = sorted_measurements[p99_idx]

        baseline = PerformanceBaseline(
            operation_name=operation_name,
            mean_duration_ms=mean_duration,
            median_duration_ms=median_duration,
            std_dev_ms=std_dev,
            p95_duration_ms=p95_duration,
            p99_duration_ms=p99_duration,
            sample_count=len(measurements),
            baseline_date=datetime.now().isoformat(),
            environment=environment,
            version=version,
        )

        self.db.save_baseline(baseline)
        return baseline

    def analyze_trend(self, operation_name: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trend for an operation."""
        measurements = self.db.get_recent_measurements(operation_name, hours)

        if len(measurements) < 2:
            return {
                "trend": "insufficient_data",
                "measurement_count": len(measurements),
            }

        durations = [m.duration_ms for m in measurements if m.success]
        if not durations:
            return {"trend": "no_successful_measurements"}

        # Calculate trend metrics
        first_half = durations[: len(durations) // 2]
        second_half = durations[len(durations) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        trend_factor = second_avg / first_avg if first_avg > 0 else 1.0

        # Classify trend
        if trend_factor > 1.2:
            trend = "degrading"
        elif trend_factor < 0.8:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "trend_factor": trend_factor,
            "first_half_avg_ms": first_avg,
            "second_half_avg_ms": second_avg,
            "current_avg_ms": statistics.mean(durations),
            "measurement_count": len(measurements),
            "success_rate": len(durations) / len(measurements),
            "timespan_hours": hours,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        baselines = self.db.load_baselines()
        regressions = self.db.load_regressions()

        # Recent regressions (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_regressions = [
            r
            for r in regressions
            if datetime.fromisoformat(r["detected_at"]) >= recent_cutoff
        ]

        # Group by severity
        regression_by_severity: Dict[str, List[Dict[str, Any]]] = {}
        for r in recent_regressions:
            severity = r["severity"]
            if severity not in regression_by_severity:
                regression_by_severity[severity] = []
            regression_by_severity[severity].append(r)

        return {
            "total_baselines": len(baselines),
            "total_regressions": len(regressions),
            "recent_regressions_24h": len(recent_regressions),
            "regression_by_severity": {
                k: len(v) for k, v in regression_by_severity.items()
            },
            "monitored_operations": list(baselines.keys()),
        }


class PerformanceTestDecorator:
    """Decorator for automatic performance monitoring of test functions."""

    def __init__(
        self, monitor: PerformanceMonitor, operation_name: Optional[str] = None
    ):
        self.monitor = monitor
        self.operation_name = operation_name

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator implementation."""
        operation_name = self.operation_name or func.__name__

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                error_details = None
                success = True

                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    error_details = str(e)
                    success = False
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000

                    regression = self.monitor.record_measurement(
                        operation_name=operation_name,
                        duration_ms=duration_ms,
                        success=success,
                        metadata={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        },
                        error_details=error_details,
                    )

                    if regression:
                        logger.warning(
                            f"Performance regression detected in {operation_name}: {regression.factor:.1f}x slower"
                        )

                return result

            return async_wrapper
        else:

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                error_details = None
                success = True

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    error_details = str(e)
                    success = False
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000

                    regression = self.monitor.record_measurement(
                        operation_name=operation_name,
                        duration_ms=duration_ms,
                        success=success,
                        metadata={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        },
                        error_details=error_details,
                    )

                    if regression:
                        logger.warning(
                            f"Performance regression detected in {operation_name}: {regression.factor:.1f}x slower"
                        )

                return result

            return sync_wrapper


# Global monitor instance for easy access
_global_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def monitor_performance(
    operation_name: Optional[str] = None,
) -> PerformanceTestDecorator:
    """Decorator for automatic performance monitoring."""
    return PerformanceTestDecorator(get_performance_monitor(), operation_name)


# Example usage and integration helpers
class CogniVaultPerformanceTests:
    """Performance tests specifically for CogniVault operations."""

    def __init__(self) -> None:
        self.monitor = get_performance_monitor()

    @monitor_performance("refiner_agent_structured_output")
    async def test_refiner_agent_performance(self) -> Any:
        """Test RefinerAgent performance with monitoring."""
        from cognivault.services.langchain_service import LangChainService
        from cognivault.agents.models import RefinerOutput

        service = LangChainService(
            model="gpt-5", agent_name="refiner", use_pool=False, use_discovery=False
        )

        result = await service.get_structured_output(
            "Refine this question: 'What is AI?' Provide refined question and confidence.",
            RefinerOutput,
        )

        return result

    def establish_agent_baselines(self) -> None:
        """Establish performance baselines for all agents."""
        # This would be run during CI/CD to establish baselines

        # Example baseline data (would come from actual measurements)
        baseline_data = {
            "refiner_agent_structured_output": [
                12000.0,
                14000.0,
                13500.0,
                15000.0,
                11000.0,
                13000.0,
                14500.0,
                12500.0,
                13800.0,
                14200.0,
            ],  # ms
            "critic_agent_structured_output": [
                18000.0,
                20000.0,
                19500.0,
                21000.0,
                17000.0,
                19000.0,
                20500.0,
                18500.0,
                19800.0,
                20200.0,
            ],
            "historian_agent_structured_output": [
                25000.0,
                28000.0,
                26500.0,
                29000.0,
                24000.0,
                26000.0,
                28500.0,
                25500.0,
                27800.0,
                28200.0,
            ],
            "synthesis_agent_structured_output": [
                35000.0,
                38000.0,
                36500.0,
                39000.0,
                34000.0,
                36000.0,
                38500.0,
                35500.0,
                37800.0,
                38200.0,
            ],
        }

        for operation_name, measurements in baseline_data.items():
            baseline = self.monitor.establish_baseline(
                operation_name=operation_name,
                measurements=measurements,
                environment="ci",
                version="baseline_v1",
            )
            logger.info(
                f"Established baseline for {operation_name}: {baseline.median_duration_ms:.1f}ms median"
            )

    def check_for_regressions(self) -> List[Dict[str, Any]]:
        """Check for recent performance regressions."""
        summary = self.monitor.get_performance_summary()

        if summary["recent_regressions_24h"] > 0:
            logger.error(
                f"Found {summary['recent_regressions_24h']} performance regressions in last 24h"
            )

            # Get recent regressions
            regressions = self.monitor.db.load_regressions()
            recent_cutoff = datetime.now() - timedelta(hours=24)

            recent_regressions = [
                r
                for r in regressions
                if datetime.fromisoformat(r["detected_at"]) >= recent_cutoff
            ]

            return recent_regressions

        return []


if __name__ == "__main__":
    # Example usage for debugging
    async def example_usage() -> None:
        # Initialize performance testing
        perf_tests = CogniVaultPerformanceTests()

        # Establish baselines (run once)
        perf_tests.establish_agent_baselines()

        # Run performance test with monitoring
        try:
            result = await perf_tests.test_refiner_agent_performance()
            print(f"RefinerAgent test completed successfully")
        except Exception as e:
            print(f"RefinerAgent test failed: {e}")

        # Check for regressions
        regressions = perf_tests.check_for_regressions()
        if regressions:
            print(f"ALERT: {len(regressions)} performance regressions detected!")
            for regression in regressions:
                print(
                    f"  - {regression['operation']}: {regression['regression_factor']:.1f}x slower"
                )

    asyncio.run(example_usage())
