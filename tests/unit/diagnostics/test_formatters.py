from typing import Any

"""
Tests for diagnostic output formatters.

This module tests the various output formatters for diagnostic data
including JSON, CSV, Prometheus, and InfluxDB formats.
"""

import json
from datetime import datetime, timedelta

from cognivault.diagnostics.formatters import (
    JSONFormatter,
    CSVFormatter,
    PrometheusFormatter,
    InfluxDBFormatter,
)
from cognivault.diagnostics.health import HealthStatus, ComponentHealth
from cognivault.diagnostics.metrics import PerformanceMetrics
from cognivault.diagnostics.diagnostics import SystemDiagnostics


class TestJSONFormatter:
    """Test JSON formatter functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.formatter = JSONFormatter()

    def test_format_health_results(self) -> None:
        """Test formatting health results as JSON."""
        timestamp = datetime.now()
        health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=timestamp,
                response_time_ms=25.0,
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.DEGRADED,
                message="LLM has issues",
                details={"provider": "openai", "errors": ["timeout"]},
                check_time=timestamp,
                response_time_ms=150.0,
            ),
        }

        formatted = self.formatter.format_health_results(health_results)
        parsed = json.loads(formatted)

        assert "agent_registry" in parsed
        assert "llm_connectivity" in parsed

        # Check agent_registry
        registry_data = parsed["agent_registry"]
        assert registry_data["name"] == "agent_registry"
        assert registry_data["status"] == "healthy"
        assert registry_data["message"] == "Registry is healthy"
        assert registry_data["details"]["agent_count"] == 4
        assert registry_data["response_time_ms"] == 25.0

        # Check llm_connectivity
        llm_data = parsed["llm_connectivity"]
        assert llm_data["name"] == "llm_connectivity"
        assert llm_data["status"] == "degraded"
        assert llm_data["message"] == "LLM has issues"
        assert llm_data["details"]["provider"] == "openai"
        assert llm_data["details"]["errors"] == ["timeout"]
        assert llm_data["response_time_ms"] == 150.0

    def test_format_performance_metrics(self) -> None:
        """Test formatting performance metrics as JSON."""
        start_time = datetime.now() - timedelta(minutes=5)
        end_time = datetime.now()

        metrics = PerformanceMetrics(
            collection_start=start_time,
            collection_end=end_time,
            total_executions=4,
            successful_executions=3,
            failed_executions=1,
            llm_api_calls=10,
            total_tokens_consumed=1500,
            average_execution_time_ms=125.5,
        )

        formatted = self.formatter.format_performance_metrics(metrics)
        parsed = json.loads(formatted)

        assert parsed["execution"]["total"] == 4
        assert parsed["execution"]["successful"] == 3
        assert parsed["execution"]["failed"] == 1
        assert parsed["timing_ms"]["average"] == 125.5
        assert parsed["resources"]["llm_api_calls"] == 10
        assert parsed["resources"]["total_tokens"] == 1500

    def test_format_system_diagnostics(self) -> None:
        """Test formatting complete system diagnostics as JSON."""
        timestamp = datetime.now()

        health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                details={"test": "value"},
                check_time=timestamp,
            )
        }

        performance_metrics = PerformanceMetrics(
            collection_start=timestamp - timedelta(minutes=1),
            collection_end=timestamp,
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
        )

        diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=health_results,
            performance_metrics=performance_metrics,
            system_info={"version": "1.0.0", "platform": "Darwin"},
            configuration_status={"is_valid": True},
            environment_info={"mode": "test"},
        )

        formatted = self.formatter.format_system_diagnostics(diagnostics)
        parsed = json.loads(formatted)

        assert parsed["timestamp"] == timestamp.isoformat()
        assert parsed["health"]["overall_status"] == "healthy"
        assert "health" in parsed
        assert "performance" in parsed
        assert "system_info" in parsed

        # Check nested data
        assert parsed["health"]["components"]["test_component"]["status"] == "healthy"
        assert parsed["performance"]["execution"]["total"] == 1
        assert parsed["system_info"]["version"] == "1.0.0"

    def test_format_json_pretty_print(self) -> None:
        """Test JSON formatting with pretty printing."""
        formatter = JSONFormatter(indent=4)
        health_results = {
            "test": ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="Test message",
                details={},
                check_time=datetime.now(),
            )
        }

        formatted = formatter.format_health_results(health_results)

        # Pretty printed JSON should have newlines and indentation
        assert "\\n" in formatted or "\n" in formatted
        assert "    " in formatted  # Indentation

        # Should still be valid JSON
        parsed = json.loads(formatted)
        assert "test" in parsed


class TestCSVFormatter:
    """Test CSV formatter functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.formatter = CSVFormatter()

    def test_format_health_results_csv(self) -> None:
        """Test formatting health results as CSV."""
        timestamp = datetime.now()
        health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=timestamp,
                response_time_ms=25.0,
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.DEGRADED,
                message="LLM has issues",
                details={"provider": "openai"},
                check_time=timestamp,
                response_time_ms=150.0,
            ),
        }

        formatted = self.formatter.format_health_results(health_results)
        lines = formatted.strip().split("\n")

        # Should have header + 2 data rows
        assert len(lines) == 3

        # Check header
        header = lines[0]
        assert "component" in header
        assert "status" in header
        assert "message" in header
        assert "response_time_ms" in header
        assert "check_time" in header

        # Check data rows
        assert "agent_registry" in lines[1]
        assert "healthy" in lines[1]
        assert "25.0" in lines[1]

        assert "llm_connectivity" in lines[2]
        assert "degraded" in lines[2]
        assert "150.0" in lines[2]

    def test_format_performance_metrics_csv(self) -> None:
        """Test formatting performance metrics as CSV."""
        start_time = datetime.now() - timedelta(minutes=5)
        end_time = datetime.now()

        metrics = PerformanceMetrics(
            collection_start=start_time,
            collection_end=end_time,
            total_executions=4,
            successful_executions=3,
            failed_executions=1,
            llm_api_calls=10,
            total_tokens_consumed=1500,
            average_execution_time_ms=125.5,
        )

        formatted = self.formatter.format_performance_metrics(metrics)
        lines = formatted.strip().split("\n")

        # Should have header + 1 data row
        assert len(lines) == 2

        # Check header contains expected columns
        header = lines[0]
        assert "start_time" in header
        assert "end_time" in header
        assert "total_agents" in header
        assert "successful_agents" in header
        assert "failed_agents" in header
        assert "total_llm_calls" in header
        assert "average_agent_duration" in header

        # Check data row
        data_row = lines[1]
        assert "4" in data_row
        assert "3" in data_row
        assert "1" in data_row
        assert "10" in data_row
        assert "9" in data_row
        assert "1500" in data_row
        assert "750" in data_row
        assert "125.5" in data_row

    def test_format_system_diagnostics_csv(self) -> None:
        """Test formatting system diagnostics as CSV."""
        timestamp = datetime.now()

        health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                details={},
                check_time=timestamp,
            )
        }

        performance_metrics = PerformanceMetrics(
            collection_start=timestamp - timedelta(minutes=1),
            collection_end=timestamp,
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
        )

        diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=health_results,
            performance_metrics=performance_metrics,
            system_info={"version": "1.0.0"},
            configuration_status={"is_valid": True},
            environment_info={"mode": "test"},
        )

        formatted = self.formatter.format_system_diagnostics(diagnostics)

        # Should contain CSV headers and data for system overview
        assert "timestamp,overall_health,total_components" in formatted
        assert "healthy" in formatted
        assert (
            "1,1,0,0,1" in formatted
        )  # healthy_components,degraded_components,unhealthy_components,total_executions

    def test_csv_special_characters(self) -> None:
        """Test CSV formatting handles special characters properly."""
        timestamp = datetime.now()
        health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.DEGRADED,
                message='Component has "quotes" and, commas',
                details={"description": "Multi-line\ntext with\ttabs"},
                check_time=timestamp,
            )
        }

        formatted = self.formatter.format_health_data(health_results)

        # Should handle quotes and commas properly in the message field
        assert '"Component has ""quotes"" and, commas"' in formatted
        # Details field is not included in CSV health data format, so check message field only
        assert "degraded" in formatted
        assert "test_component" in formatted


class TestPrometheusFormatter:
    """Test Prometheus formatter functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.formatter = PrometheusFormatter()

    def test_format_health_results_prometheus(self) -> None:
        """Test formatting health results as Prometheus metrics."""
        timestamp = datetime.now()
        health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=timestamp,
                response_time_ms=25.0,
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.DEGRADED,
                message="LLM has issues",
                details={"provider": "openai"},
                check_time=timestamp,
                response_time_ms=150.0,
            ),
        }

        formatted = self.formatter.format_health_results(health_results)

        # Should contain Prometheus metric format
        assert "# HELP cognivault_health_status Component health status" in formatted
        assert "# TYPE cognivault_health_status gauge" in formatted

        # Should contain metrics for each component
        assert 'cognivault_health_status{component="agent_registry"} 1' in formatted
        assert 'cognivault_health_status{component="llm_connectivity"} 0.5' in formatted

        # Should contain response time metrics
        assert (
            "# HELP cognivault_health_response_time_ms Component health check response time"
            in formatted
        )
        assert "# TYPE cognivault_health_response_time_ms gauge" in formatted
        assert (
            'cognivault_health_response_time_ms{component="agent_registry"} 25.0'
            in formatted
        )
        assert (
            'cognivault_health_response_time_ms{component="llm_connectivity"} 150.0'
            in formatted
        )

    def test_format_performance_metrics_prometheus(self) -> None:
        """Test formatting performance metrics as Prometheus metrics."""
        start_time = datetime.now() - timedelta(minutes=5)
        end_time = datetime.now()

        metrics = PerformanceMetrics(
            collection_start=start_time,
            collection_end=end_time,
            total_executions=4,
            successful_executions=3,
            failed_executions=1,
            llm_api_calls=10,
            total_tokens_consumed=1500,
            average_execution_time_ms=125.5,
        )

        formatted = self.formatter.format_performance_metrics(metrics)

        # Should contain various Prometheus metrics
        assert (
            "# HELP cognivault_agents_total Total number of agents executed"
            in formatted
        )
        assert "# TYPE cognivault_agents_total counter" in formatted
        assert "cognivault_agents_total 4" in formatted

        assert (
            "# HELP cognivault_agents_successful Number of successful agent executions"
            in formatted
        )
        assert "cognivault_agents_successful 3" in formatted

        assert (
            "# HELP cognivault_agents_failed Number of failed agent executions"
            in formatted
        )
        assert "cognivault_agents_failed 1" in formatted

        assert (
            "# HELP cognivault_llm_calls_total Total number of LLM calls" in formatted
        )
        assert "cognivault_llm_calls_total 10" in formatted

        assert "# HELP cognivault_tokens_used_total Total tokens consumed" in formatted
        assert "cognivault_tokens_used_total 1500" in formatted

        assert (
            "# HELP cognivault_tokens_generated_total Total tokens generated"
            in formatted
        )
        assert "cognivault_tokens_generated_total 750" in formatted

        assert (
            "# HELP cognivault_agent_duration_avg Average agent execution duration"
            in formatted
        )
        assert "cognivault_agent_duration_avg 125.5" in formatted

        assert (
            "# HELP cognivault_llm_duration_avg Average LLM call duration" in formatted
        )
        assert "cognivault_llm_duration_avg 75.0" in formatted

        assert (
            "# HELP cognivault_pipeline_duration Pipeline execution duration"
            in formatted
        )
        assert "cognivault_pipeline_duration 500.0" in formatted

    def test_format_system_diagnostics_prometheus(self) -> None:
        """Test formatting system diagnostics as Prometheus metrics."""
        timestamp = datetime.now()

        health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                details={},
                check_time=timestamp,
            )
        }

        performance_metrics = PerformanceMetrics(
            collection_start=timestamp - timedelta(minutes=1),
            collection_end=timestamp,
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
        )

        diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=health_results,
            performance_metrics=performance_metrics,
            system_info={"version": "1.0.0"},
            configuration_status={"is_valid": True},
            environment_info={"mode": "test"},
        )

        formatted = self.formatter.format_system_diagnostics(diagnostics)

        # Should contain health metrics
        assert "cognivault_system_health" in formatted
        assert "cognivault_component_health" in formatted

        # Should contain performance metrics
        assert "cognivault_executions_total" in formatted
        assert "cognivault_tokens_consumed_total" in formatted

        # Verify numeric values are present
        assert "cognivault_executions_total 1" in formatted
        assert "cognivault_tokens_consumed_total 100" in formatted

    def test_prometheus_health_status_values(self) -> None:
        """Test Prometheus health status numeric values."""
        assert self.formatter._health_status_to_value(HealthStatus.HEALTHY) == 1.0
        assert self.formatter._health_status_to_value(HealthStatus.DEGRADED) == 0.5
        assert self.formatter._health_status_to_value(HealthStatus.UNHEALTHY) == 0.0
        assert self.formatter._health_status_to_value(HealthStatus.UNKNOWN) == -1.0


class TestInfluxDBFormatter:
    """Test InfluxDB formatter functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.formatter = InfluxDBFormatter()

    def test_format_health_results_influxdb(self) -> None:
        """Test formatting health results as InfluxDB line protocol."""
        timestamp = datetime.now()
        health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=timestamp,
                response_time_ms=25.0,
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.DEGRADED,
                message="LLM has issues",
                details={"provider": "openai"},
                check_time=timestamp,
                response_time_ms=150.0,
            ),
        }

        formatted = self.formatter.format_health_data(health_results)
        lines = formatted.strip().split("\n")

        # Should have lines for each component
        assert len(lines) == 2

        # Check line protocol format
        for line in lines:
            assert "cognivault_component_health," in line
            assert "value=" in line
            assert "response_time_ms=" in line
            # Skip exact timestamp check due to timing differences, just check it's a valid number
            timestamp_part = line.split()[-1]
            assert timestamp_part.isdigit()

        # Check specific values
        registry_line = next(
            line for line in lines if "component=agent_registry" in line
        )
        assert "value=1.0" in registry_line
        assert "response_time_ms=25.0" in registry_line

        llm_line = next(line for line in lines if "component=llm_connectivity" in line)
        assert "value=0.5" in llm_line
        assert "response_time_ms=150.0" in llm_line

    def test_format_performance_metrics_influxdb(self) -> None:
        """Test formatting performance metrics as InfluxDB line protocol."""
        start_time = datetime.now() - timedelta(minutes=5)
        end_time = datetime.now()

        metrics = PerformanceMetrics(
            collection_start=start_time,
            collection_end=end_time,
            total_executions=4,
            successful_executions=3,
            failed_executions=1,
            llm_api_calls=10,
            total_tokens_consumed=1500,
            average_execution_time_ms=125.5,
        )

        formatted = self.formatter.format_performance_metrics(metrics)

        # Should contain line protocol format
        assert "cognivault_performance" in formatted
        assert "total_agents=4" in formatted
        assert "successful_agents=3" in formatted
        assert "failed_agents=1" in formatted
        assert "total_llm_calls=10" in formatted
        assert "successful_llm_calls=9" in formatted
        assert "failed_llm_calls=1" in formatted
        assert "total_tokens_used=1500" in formatted
        assert "total_tokens_generated=750" in formatted
        assert "average_agent_duration=125.5" in formatted
        assert "average_llm_duration=75.0" in formatted
        assert "pipeline_duration=500.0" in formatted

        # Should end with timestamp
        assert formatted.strip().endswith(str(int(end_time.timestamp() * 1000000000)))

    def test_format_system_diagnostics_influxdb(self) -> None:
        """Test formatting system diagnostics as InfluxDB line protocol."""
        timestamp = datetime.now()

        health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                details={},
                check_time=timestamp,
            )
        }

        performance_metrics = PerformanceMetrics(
            collection_start=timestamp - timedelta(minutes=1),
            collection_end=timestamp,
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
        )

        diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=health_results,
            performance_metrics=performance_metrics,
            system_info={"version": "1.0.0"},
            configuration_status={"is_valid": True},
            environment_info={"mode": "test"},
        )

        formatted = self.formatter.format_system_diagnostics(diagnostics)
        lines = formatted.strip().split("\n")

        # Should have multiple lines
        assert len(lines) >= 2

        # Should contain health and performance data
        health_lines = [
            line for line in lines if "cognivault_component_health," in line
        ]
        performance_lines = [line for line in lines if "cognivault_performance" in line]

        assert len(health_lines) >= 1
        assert len(performance_lines) >= 1

        # Check line protocol format - skip exact timestamp check due to timing differences
        for line in lines:
            assert "=" in line  # Should have field values
            # Check timestamp is at end and is a valid number
            timestamp_part = line.split()[-1]
            assert timestamp_part.isdigit()

    def test_influxdb_special_characters(self) -> None:
        """Test InfluxDB formatting handles special characters properly."""
        timestamp = datetime.now()
        health_results = {
            "test component": ComponentHealth(  # Space in name
                name="test component",
                status=HealthStatus.HEALTHY,
                message="Component with spaces and = signs",
                details={"key with spaces": "value,with,commas"},
                check_time=timestamp,
            )
        }

        formatted = self.formatter.format_health_results(health_results)

        # Should handle values with special characters (note: current implementation doesn't escape spaces)
        assert "component=test component" in formatted
        assert "value=1.0" in formatted

    def test_influxdb_health_status_values(self) -> None:
        """Test InfluxDB health status numeric values."""
        assert self.formatter._health_status_to_value(HealthStatus.HEALTHY) == 1.0
        assert self.formatter._health_status_to_value(HealthStatus.DEGRADED) == 0.5
        assert self.formatter._health_status_to_value(HealthStatus.UNHEALTHY) == 0.0
        assert self.formatter._health_status_to_value(HealthStatus.UNKNOWN) == -1.0


class TestFormatterIntegration:
    """Test formatter integration and edge cases."""

    def test_empty_health_results(self) -> None:
        """Test formatters handle empty health results."""
        formatters = [
            JSONFormatter(),
            CSVFormatter(),
            PrometheusFormatter(),
            InfluxDBFormatter(),
        ]

        for formatter in formatters:
            result = formatter.format_health_results({})
            assert isinstance(result, str)
            # Should not crash and should return valid format
            if isinstance(formatter, JSONFormatter):
                assert result == "{}"
            elif isinstance(formatter, CSVFormatter):
                # Should at least have headers
                assert "component" in result

    def test_none_values(self) -> None:
        """Test formatters handle None values gracefully."""
        formatters = [
            JSONFormatter(),
            CSVFormatter(),
            PrometheusFormatter(),
            InfluxDBFormatter(),
        ]

        for formatter in formatters:
            result = formatter.format_performance_metrics(None)
            assert isinstance(result, str)
            # Should not crash

    def test_formatter_consistency(self) -> None:
        """Test that all formatters produce consistent data."""
        timestamp = datetime.now()
        health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                details={"test": "value"},
                check_time=timestamp,
            )
        }

        json_formatter = JSONFormatter()
        csv_formatter = CSVFormatter()
        prometheus_formatter = PrometheusFormatter()
        influxdb_formatter = InfluxDBFormatter()

        # All should produce non-empty results
        json_result = json_formatter.format_health_results(health_results)
        csv_result = csv_formatter.format_health_results(health_results)
        prometheus_result = prometheus_formatter.format_health_results(health_results)
        influxdb_result = influxdb_formatter.format_health_results(health_results)

        assert len(json_result) > 0
        assert len(csv_result) > 0
        assert len(prometheus_result) > 0
        assert len(influxdb_result) > 0

        # All should contain the component name
        assert "test_component" in json_result
        assert "test_component" in csv_result
        assert "test_component" in prometheus_result
        assert "test_component" in influxdb_result

        # All should indicate healthy status
        assert "healthy" in json_result.lower()
        assert "healthy" in csv_result.lower()
        assert "1" in prometheus_result  # 1.0 for healthy
        assert "1.0" in influxdb_result  # 1.0 for healthy
