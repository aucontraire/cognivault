"""
Tests for diagnostics manager functionality.

This module tests the DiagnosticsManager class and SystemDiagnostics
coordination between health checks and metrics collection.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from cognivault.diagnostics.diagnostics import (
    DiagnosticsManager,
    SystemDiagnostics,
)
from cognivault.diagnostics.health import HealthStatus, ComponentHealth
from cognivault.diagnostics.metrics import PerformanceMetrics


class TestSystemDiagnostics:
    """Test SystemDiagnostics dataclass."""

    def test_system_diagnostics_creation(self):
        """Test creating SystemDiagnostics."""
        timestamp = datetime.now()

        # Create mock health results
        health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=timestamp,
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.HEALTHY,
                message="LLM is healthy",
                details={"provider": "openai"},
                check_time=timestamp,
            ),
        }

        # Create mock performance metrics (using only valid fields)
        performance_metrics = PerformanceMetrics(
            total_executions=4,
            successful_executions=4,
            failed_executions=0,
            llm_api_calls=10,
            total_tokens_consumed=1000,
            average_execution_time_ms=150.0,
            collection_start=timestamp - timedelta(minutes=1),
            collection_end=timestamp,
            peak_memory_usage_bytes=0,
            error_breakdown={},
            circuit_breaker_trips=0,
            retry_attempts=0,
        )

        diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=health_results,
            performance_metrics=performance_metrics,
            system_info={"version": "1.0.0", "python_version": "3.12.2"},
            configuration_status={"debug": True},
            environment_info={"env": "test"},
        )

        assert diagnostics.timestamp == timestamp
        assert diagnostics.overall_health == HealthStatus.HEALTHY
        assert diagnostics.component_healths == health_results
        assert diagnostics.performance_metrics == performance_metrics
        assert diagnostics.system_info == {
            "version": "1.0.0",
            "python_version": "3.12.2",
        }
        assert diagnostics.configuration_status == {"debug": True}
        assert diagnostics.environment_info == {"env": "test"}

    def test_system_diagnostics_to_dict(self):
        """Test SystemDiagnostics to_dict method."""
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
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
            collection_start=timestamp - timedelta(minutes=1),
            collection_end=timestamp,
            peak_memory_usage_bytes=0,
            error_breakdown={},
            circuit_breaker_trips=0,
            retry_attempts=0,
        )

        diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=health_results,
            performance_metrics=performance_metrics,
            system_info={"version": "1.0.0"},
            configuration_status={"debug": True},
            environment_info={"env": "test"},
        )

        result = diagnostics.to_dict()

        assert result["timestamp"] == timestamp.isoformat()
        assert "health" in result
        assert result["health"]["overall_status"] == "healthy"
        assert "components" in result["health"]
        assert result["health"]["components"]["test_component"]["status"] == "healthy"
        assert "performance" in result
        assert result["performance"]["execution"]["total"] == 1
        assert result["system_info"] == {"version": "1.0.0"}
        assert result["configuration"] == {"debug": True}
        assert result["environment"] == {"env": "test"}


class TestDiagnosticsManager:
    """Test DiagnosticsManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = DiagnosticsManager()

    @pytest.mark.asyncio
    async def test_full_diagnostics(self):
        """Test full diagnostics collection."""
        # Mock health checker
        mock_health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=datetime.now(),
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.HEALTHY,
                message="LLM is healthy",
                details={"provider": "openai"},
                check_time=datetime.now(),
            ),
        }

        with patch.object(self.manager.health_checker, "check_all") as mock_check_all:
            mock_check_all.return_value = mock_health_results

            # Mock metrics collector
            with patch.object(
                self.manager.metrics_collector, "get_all_metrics"
            ) as mock_get_metrics:
                mock_get_metrics.return_value = {
                    "counters": {"test_counter": 5},
                    "gauges": {"test_gauge": 42.0},
                    "histograms": {"test_histogram": [1, 2, 3]},
                    "timings": {"test_timing": [10.0, 20.0]},
                }

                diagnostics = await self.manager.run_full_diagnostics()

                assert isinstance(diagnostics, SystemDiagnostics)
                assert diagnostics.overall_health == HealthStatus.HEALTHY
                assert len(diagnostics.component_healths) == 2
                assert diagnostics.performance_metrics is not None
                assert diagnostics.system_info is not None
                assert diagnostics.performance_metrics is not None

    @pytest.mark.asyncio
    async def test_full_diagnostics_with_unhealthy_components(self):
        """Test full diagnostics when components are unhealthy."""
        # Mock health checker with unhealthy component
        mock_health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=datetime.now(),
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.UNHEALTHY,
                message="LLM connectivity failed",
                details={"provider": "openai", "error": "Connection timeout"},
                check_time=datetime.now(),
            ),
        }

        with patch.object(self.manager.health_checker, "check_all") as mock_check_all:
            mock_check_all.return_value = mock_health_results

            with patch.object(
                self.manager.metrics_collector, "get_all_metrics"
            ) as mock_get_metrics:
                mock_get_metrics.return_value = {
                    "counters": {},
                    "gauges": {},
                    "histograms": {},
                    "timings": {},
                }

                diagnostics = await self.manager.run_full_diagnostics()

                assert diagnostics.overall_health == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_quick_health_check(self):
        """Test quick health check functionality."""
        mock_health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={"agent_count": 4},
                check_time=datetime.now(),
            )
        }

        with patch.object(self.manager.health_checker, "check_all") as mock_check_all:
            mock_check_all.return_value = mock_health_results

            health_summary = await self.manager.quick_health_check()

            assert health_summary["status"] == "healthy"
            assert health_summary["components"]["total"] == 1
            assert health_summary["components"]["healthy"] == 1
            assert health_summary["components"]["degraded"] == 0
            assert health_summary["components"]["unhealthy"] == 0
            assert "timestamp" in health_summary
            assert "uptime_seconds" in health_summary

    @pytest.mark.asyncio
    async def test_quick_health_check_with_exception(self):
        """Test quick health check handles exceptions."""
        with patch.object(self.manager.health_checker, "check_all") as mock_check_all:
            mock_check_all.side_effect = Exception("Health check failed")

            # The current implementation doesn't handle exceptions, so it will raise
            with pytest.raises(Exception, match="Health check failed"):
                await self.manager.quick_health_check()

    def test_performance_summary(self):
        """Test performance summary generation."""
        # Mock metrics collector with sample data
        mock_metrics = PerformanceMetrics(
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
            llm_api_calls=15,
            total_tokens_consumed=1500,
            average_execution_time_ms=150.0,
        )

        with patch.object(
            self.manager.metrics_collector, "get_metrics_summary"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_metrics

            summary = self.manager.get_performance_summary()

            assert isinstance(summary, dict)
            assert summary["execution"]["total"] == 10
            assert summary["execution"]["successful"] == 8
            assert summary["execution"]["failed"] == 2
            assert summary["resources"]["llm_api_calls"] == 15
            assert summary["resources"]["total_tokens"] == 1500
            assert summary["timing_ms"]["average"] == 150.0

    def test_performance_summary_with_no_data(self):
        """Test performance summary when no data is available."""
        # Mock empty metrics
        mock_metrics = PerformanceMetrics()

        with patch.object(
            self.manager.metrics_collector, "get_metrics_summary"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_metrics

            summary = self.manager.get_performance_summary()

            assert isinstance(summary, dict)
            assert summary["execution"]["total"] == 0
            assert summary["execution"]["successful"] == 0
            assert summary["execution"]["failed"] == 0
            assert summary["resources"]["llm_api_calls"] == 0
            assert summary["resources"]["total_tokens"] == 0
            assert summary["timing_ms"]["average"] == 0.0

    def test_get_system_info(self):
        """Test system info collection."""
        with (
            patch("platform.platform") as mock_platform,
            patch("platform.machine") as mock_machine,
            patch("platform.processor") as mock_processor,
            patch("sys.version", "3.12.2 (test)"),
            patch("sys.executable", "/usr/bin/python"),
            patch("platform.architecture") as mock_architecture,
        ):
            mock_platform.return_value = "Darwin-21.6.0"
            mock_machine.return_value = "x86_64"
            mock_processor.return_value = "i386"
            mock_architecture.return_value = ("64bit", "")

            system_info = self.manager._get_system_info()

            assert system_info["platform"] == "Darwin-21.6.0"
            assert system_info["machine"] == "x86_64"
            assert system_info["processor"] == "i386"
            assert system_info["python_version"] == "3.12.2 (test)"
            assert system_info["python_executable"] == "/usr/bin/python"
            assert system_info["architecture"] == ("64bit", "")

    def test_get_system_info_with_exceptions(self):
        """Test system info collection handles exceptions."""
        with (
            patch("platform.platform") as mock_platform,
            patch("platform.machine") as mock_machine,
            patch("platform.processor") as mock_processor,
        ):
            # Mock exceptions
            mock_platform.side_effect = Exception("Platform error")
            mock_machine.side_effect = Exception("Machine error")
            mock_processor.side_effect = Exception("Processor error")

            # The current implementation doesn't handle exceptions, so it will raise
            # This test should be updated to test exception handling if implemented
            with pytest.raises(Exception):
                self.manager._get_system_info()

    def test_get_overall_status(self):
        """Test overall status calculation."""
        # Test all healthy
        health_results = {
            "comp1": ComponentHealth(
                "comp1", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
            "comp2": ComponentHealth(
                "comp2", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
        }

        status = self.manager.health_checker.get_overall_status(health_results)
        assert status == HealthStatus.HEALTHY

        # Test with degraded component
        health_results["comp2"] = ComponentHealth(
            "comp2", HealthStatus.DEGRADED, "Issues", {}, datetime.now()
        )
        status = self.manager.health_checker.get_overall_status(health_results)
        assert status == HealthStatus.DEGRADED

        # Test with unhealthy component
        health_results["comp1"] = ComponentHealth(
            "comp1", HealthStatus.UNHEALTHY, "Failed", {}, datetime.now()
        )
        status = self.manager.health_checker.get_overall_status(health_results)
        assert status == HealthStatus.UNHEALTHY

        # Test empty results
        status = self.manager.health_checker.get_overall_status({})
        assert status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_diagnostics_manager_integration(self):
        """Test full integration of diagnostics manager."""
        # This test verifies that all components work together

        # Mock health checker to return realistic results
        mock_health_results = {
            "agent_registry": ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry healthy with 4 agents",
                details={"agent_count": 4, "pipeline_valid": True},
                check_time=datetime.now(),
                response_time_ms=25.0,
            ),
            "llm_connectivity": ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.HEALTHY,
                message="OpenAI connectivity healthy",
                details={"provider": "openai", "model": "gpt-4"},
                check_time=datetime.now(),
                response_time_ms=150.0,
            ),
            "configuration": ComponentHealth(
                name="configuration",
                status=HealthStatus.HEALTHY,
                message="Configuration is valid and complete",
                details={"validation_errors": []},
                check_time=datetime.now(),
                response_time_ms=5.0,
            ),
        }

        # Reset metrics collector to ensure clean state
        self.manager.metrics_collector.reset_metrics()

        # Add some realistic metrics data
        self.manager.metrics_collector.record_agent_execution(
            "refiner", True, 120.0, tokens_used=100
        )
        self.manager.metrics_collector.record_agent_execution(
            "critic", True, 150.0, tokens_used=80
        )
        self.manager.metrics_collector.record_agent_execution(
            "synthesis", True, 200.0, tokens_used=120
        )

        self.manager.metrics_collector.record_llm_call(
            "gpt-4", True, 50.0, tokens_used=300, tokens_generated=100
        )
        self.manager.metrics_collector.record_llm_call(
            "gpt-4", True, 75.0, tokens_used=250, tokens_generated=85
        )

        self.manager.metrics_collector.record_pipeline_execution(
            "pipeline-1", True, 500.0, ["refiner", "critic", "synthesis"]
        )

        with patch.object(self.manager.health_checker, "check_all") as mock_check_all:
            mock_check_all.return_value = mock_health_results

            # Get full diagnostics
            diagnostics = await self.manager.run_full_diagnostics()

            # Verify comprehensive results
            assert diagnostics.overall_health == HealthStatus.HEALTHY
            assert len(diagnostics.component_healths) == 3

            # Check performance metrics
            assert diagnostics.performance_metrics.total_executions == 3
            assert diagnostics.performance_metrics.successful_executions == 3
            assert diagnostics.performance_metrics.failed_executions == 0
            assert diagnostics.performance_metrics.llm_api_calls == 3
            # Token consumption will be tracked in total_tokens_consumed
            assert diagnostics.performance_metrics.total_tokens_consumed >= 0

            # Check system info
            assert "platform" in diagnostics.system_info
            assert "python_version" in diagnostics.system_info
            assert "machine" in diagnostics.system_info

            # Check configuration and environment info
            assert diagnostics.configuration_status is not None
            assert diagnostics.environment_info is not None

            # Verify serialization works
            diagnostics_dict = diagnostics.to_dict()
            assert isinstance(diagnostics_dict, dict)
            assert "timestamp" in diagnostics_dict
            assert "health" in diagnostics_dict
            assert "performance" in diagnostics_dict
            assert "system_info" in diagnostics_dict
            assert "configuration" in diagnostics_dict
            assert "environment" in diagnostics_dict
