"""
Tests for diagnostic CLI commands.

This module tests the CLI interface for diagnostics including
health checks, metrics, and various output formats.
"""

import json
import tempfile
from datetime import datetime
from unittest.mock import patch, AsyncMock
from typer.testing import CliRunner

from cognivault.diagnostics.cli import app, diagnostics_cli
from cognivault.diagnostics.health import HealthStatus, ComponentHealth
from cognivault.diagnostics.metrics import PerformanceMetrics
from cognivault.diagnostics.diagnostics import SystemDiagnostics


class TestDiagnosticsCLI:
    """Test diagnostics CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_health_command_basic(self):
        """Test basic health command."""

        # Mock the diagnostics instance and resource scheduler to prevent async task creation
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch.object(
                diagnostics_cli.diagnostics,
                "quick_health_check",
                new_callable=AsyncMock,
            ) as mock_health_check,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler
            # quick_health_check returns a dict with status, timestamp, components, uptime_seconds
            mock_health_check.return_value = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "total": 2,
                    "healthy": 2,
                    "degraded": 0,
                    "unhealthy": 0,
                    "unknown": 0,
                },
                "uptime_seconds": 0.0,
            }

            # Use standalone_mode=False to prevent typer from handling exits
            result = self.runner.invoke(app, ["health"], standalone_mode=False)

            assert result.exit_code == 0
            assert "CogniVault Health Check" in result.stdout
            assert "Status: HEALTHY" in result.stdout
            assert "Healthy" in result.stdout

    def test_health_command_with_unhealthy_components(self):
        """Test health command with unhealthy components."""
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

        from cognivault.diagnostics.cli import diagnostics_cli

        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch.object(
                diagnostics_cli.diagnostics,
                "quick_health_check",
                new_callable=AsyncMock,
            ) as mock_quick_check,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler
            # quick_health_check returns a dict with status, timestamp, components, uptime_seconds
            mock_quick_check.return_value = {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "total": 2,
                    "healthy": 1,
                    "degraded": 0,
                    "unhealthy": 1,
                    "unknown": 0,
                },
                "uptime_seconds": 0.0,
            }

            result = self.runner.invoke(app, ["health"], catch_exceptions=False)

            assert result.exit_code == 2  # Should exit with error code for unhealthy
            assert "Status: UNHEALTHY" in result.stdout

    def test_health_command_json_format(self):
        """Test health command with JSON output format."""
        # Mock the diagnostics instance and resource scheduler to prevent async task creation
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch.object(
                diagnostics_cli.diagnostics,
                "quick_health_check",
                new_callable=AsyncMock,
            ) as mock_health_check,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler
            # quick_health_check returns a dict with status, timestamp, components, uptime_seconds
            mock_health_check_result = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "total": 1,
                    "healthy": 1,
                    "degraded": 0,
                    "unhealthy": 0,
                    "unknown": 0,
                },
                "uptime_seconds": 0.0,
            }
            mock_health_check.return_value = mock_health_check_result

            result = self.runner.invoke(app, ["health", "--format", "json"])

            assert result.exit_code == 0

            # Should be valid JSON that matches the quick_health_check return
            parsed = json.loads(result.stdout)
            assert parsed["status"] == "healthy"
            assert "timestamp" in parsed
            assert "components" in parsed
            assert parsed["components"]["total"] == 1
            assert parsed["components"]["healthy"] == 1

    def test_status_command(self):
        """Test status command (alias for health)."""
        mock_health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                details={},
                check_time=datetime.now(),
            )
        }

        mock_performance_metrics = PerformanceMetrics(
            collection_start=datetime.now(),
            collection_end=datetime.now(),
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
        )

        mock_diagnostics = SystemDiagnostics(
            timestamp=datetime.now(),
            overall_health=HealthStatus.HEALTHY,
            component_healths=mock_health_results,
            performance_metrics=mock_performance_metrics,
            system_info={"version": "1.0.0", "platform": "Darwin"},
            configuration_status={},
            environment_info={},
        )

        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch(
                "cognivault.diagnostics.cli.DiagnosticsManager"
            ) as mock_manager_class,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler

            mock_manager = AsyncMock()
            # status command calls run_full_diagnostics
            mock_manager.run_full_diagnostics.return_value = mock_diagnostics
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert "CogniVault System Status" in result.stdout

    def test_metrics_command_basic(self):
        """Test basic metrics command."""
        mock_performance_metrics = PerformanceMetrics(
            collection_start=datetime.now(),
            collection_end=datetime.now(),
            total_executions=4,
            successful_executions=3,
            failed_executions=1,
            llm_api_calls=10,
            total_tokens_consumed=1500,
            average_execution_time_ms=125.5,
        )

        with patch(
            "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.get_performance_summary"
        ) as mock_get_performance:
            mock_get_performance.return_value = mock_performance_metrics.to_dict()

            result = self.runner.invoke(app, ["metrics"])

            assert result.exit_code == 0
            assert "Performance Metrics" in result.stdout
            assert "Execution Summary" in result.stdout
            assert "Total Executions" in result.stdout
            assert "Successful" in result.stdout
            assert "Failed" in result.stdout

    def test_metrics_command_json_format(self):
        """Test metrics command with JSON output format."""
        mock_performance_metrics = PerformanceMetrics(
            collection_start=datetime.now(),
            collection_end=datetime.now(),
            total_executions=2,
            successful_executions=2,
            failed_executions=0,
            llm_api_calls=5,
            total_tokens_consumed=500,
            average_execution_time_ms=100.0,
        )

        with patch(
            "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.get_performance_summary"
        ) as mock_get_performance:
            mock_get_performance.return_value = mock_performance_metrics.to_dict()

            result = self.runner.invoke(app, ["metrics", "--format", "json"])

            assert result.exit_code == 0

            # Should be valid JSON with the to_dict() structure
            parsed = json.loads(result.stdout)
            assert parsed["execution"]["total"] == 2
            assert parsed["execution"]["successful"] == 2
            assert parsed["execution"]["failed"] == 0
            assert parsed["resources"]["llm_api_calls"] == 5
            assert parsed["resources"]["total_tokens"] == 500

    def test_metrics_command_with_window_filter(self):
        """Test metrics command with time window filter."""
        mock_performance_summary = {
            "execution": {
                "total": 1,
                "successful": 1,
                "failed": 0,
                "success_rate": 1.0,
            },
            "timing_ms": {
                "average": 80.0,
                "min": 80.0,
                "max": 80.0,
                "p50": 80.0,
                "p95": 80.0,
                "p99": 80.0,
            },
            "agents": {},
            "errors": {"breakdown": {}},
        }

        with patch(
            "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.get_performance_summary"
        ) as mock_get_performance:
            mock_get_performance.return_value = mock_performance_summary

            result = self.runner.invoke(app, ["metrics", "--window", "60"])

            assert result.exit_code == 0
            assert "Performance Metrics" in result.stdout
            assert "Time Window: Last 60 minutes" in result.stdout

    def test_agents_command(self):
        """Test agents command."""
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch(
                "cognivault.diagnostics.cli.DiagnosticsManager"
            ) as mock_manager_class,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler
            mock_manager = AsyncMock()
            mock_agent_status = {
                "timestamp": datetime.now().isoformat(),
                "total_agents": 3,
                "agents": {
                    "refiner": {
                        "name": "refiner",
                        "description": "Refines and improves user queries",
                        "requires_llm": True,
                        "is_critical": True,
                        "failure_strategy": "fail_fast",
                        "dependencies": [],
                        "health_check": True,
                        "metrics": {
                            "executions": 5,
                            "successes": 4,
                            "failures": 1,
                            "success_rate": 0.8,
                            "avg_duration_ms": 110.0,
                            "tokens_consumed": 500,
                        },
                    },
                    "critic": {
                        "name": "critic",
                        "description": "Critical analysis agent",
                        "requires_llm": True,
                        "is_critical": True,
                        "failure_strategy": "fail_fast",
                        "dependencies": [],
                        "health_check": True,
                        "metrics": {
                            "executions": 3,
                            "successes": 3,
                            "failures": 0,
                            "success_rate": 1.0,
                            "avg_duration_ms": 150.0,
                            "tokens_consumed": 300,
                        },
                    },
                },
            }
            mock_manager.get_agent_status.return_value = mock_agent_status
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["agents"])

            assert result.exit_code == 0
            assert "Agent Status" in result.stdout
            assert "refiner" in result.stdout
            assert "critic" in result.stdout

    def test_config_command(self):
        """Test config command."""
        mock_config_report = {
            "timestamp": datetime.now().isoformat(),
            "environment": "development",
            "validation": {
                "is_valid": True,
                "error_count": 0,
                "errors": [],
            },
            "configuration": {
                "execution": {
                    "timeout_seconds": 30,
                    "max_retries": 3,
                    "critic_enabled": True,
                    "default_agents": ["refiner", "critic"],
                },
                "models": {
                    "default_provider": "openai",
                    "max_tokens_per_request": 4000,
                    "temperature": 0.7,
                },
                "files": {
                    "notes_directory": "/tmp/notes",
                    "logs_directory": "/tmp/logs",
                    "max_file_size": 10485760,
                },
            },
            "recommendations": [],
        }

        with patch.object(
            diagnostics_cli.diagnostics,
            "get_configuration_report",
            return_value=mock_config_report,
        ):
            result = self.runner.invoke(app, ["config"])

            assert result.exit_code == 0
            assert "Configuration Report" in result.stdout
            assert "Timeout: 30s" in result.stdout
            assert "Max Retries: 3" in result.stdout
            assert "Provider: openai" in result.stdout

    def test_config_command_json_format(self):
        """Test config command with JSON output format."""
        mock_config_report = {
            "timestamp": datetime.now().isoformat(),
            "environment": "development",
            "validation": {"is_valid": True, "error_count": 0, "errors": []},
            "configuration": {
                "execution": {"timeout_seconds": 30},
                "models": {"default_provider": "openai"},
            },
            "recommendations": [],
        }

        with patch.object(
            diagnostics_cli.diagnostics,
            "get_configuration_report",
            return_value=mock_config_report,
        ):
            result = self.runner.invoke(app, ["config", "--json"])

            assert result.exit_code == 0

            # Should be valid JSON
            parsed = json.loads(result.stdout)
            assert parsed["configuration"]["execution"]["timeout_seconds"] == 30
            assert parsed["configuration"]["models"]["default_provider"] == "openai"

    def test_full_command(self):
        """Test full diagnostics command."""
        timestamp = datetime.now()

        mock_health_results = {
            "test_component": ComponentHealth(
                name="test_component",
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                details={},
                check_time=timestamp,
            )
        }

        mock_performance_metrics = PerformanceMetrics(
            collection_start=timestamp,
            collection_end=timestamp,
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
        )

        mock_diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=mock_health_results,
            performance_metrics=mock_performance_metrics,
            system_info={"version": "1.0.0", "platform": "Darwin"},
            configuration_status={},
            environment_info={},
        )

        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch(
                "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.run_full_diagnostics"
            ) as mock_run_full,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler

            mock_run_full.return_value = mock_diagnostics

            result = self.runner.invoke(app, ["full"])

            assert result.exit_code == 0
            assert "Complete System Diagnostics" in result.stdout
            assert "Overall Status: ✅ HEALTHY" in result.stdout
            assert "Component Health" in result.stdout
            assert "Performance Metrics" in result.stdout
            assert "System Information" in result.stdout

    def test_full_command_with_output_file(self):
        """Test full diagnostics command with output file."""
        timestamp = datetime.now()

        mock_diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths={},
            performance_metrics=PerformanceMetrics(
                collection_start=timestamp,
                collection_end=timestamp,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                llm_api_calls=0,
                total_tokens_consumed=0,
                average_execution_time_ms=0.0,
            ),
            system_info={},
            configuration_status={},
            environment_info={},
        )

        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch(
                "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.run_full_diagnostics"
            ) as mock_run_full,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler

            mock_run_full.return_value = mock_diagnostics

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                result = self.runner.invoke(
                    app, ["full", "--format", "json", "--output", temp_path]
                )

                assert result.exit_code == 0
                assert f"Diagnostics saved to: \n{temp_path}" in result.stdout

                # Check that file was written
                with open(temp_path, "r") as f:
                    content = f.read()
                    parsed = json.loads(content)
                    assert "health" in parsed
                    assert parsed["health"]["overall_status"] == "healthy"

            finally:
                import os

                os.unlink(temp_path)

    def test_prometheus_format(self):
        """Test Prometheus output format."""
        mock_performance_metrics = PerformanceMetrics(
            collection_start=datetime.now(),
            collection_end=datetime.now(),
            total_executions=2,
            successful_executions=2,
            failed_executions=0,
            llm_api_calls=5,
            total_tokens_consumed=500,
            average_execution_time_ms=100.0,
        )

        with patch(
            "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.metrics_collector.get_metrics_summary"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_performance_metrics

            result = self.runner.invoke(app, ["metrics", "--format", "prometheus"])

            assert result.exit_code == 0
            assert "# HELP cognivault_agents_total" in result.stdout
            assert "# TYPE cognivault_agents_total counter" in result.stdout
            assert "cognivault_agents_total 2" in result.stdout

    def test_influxdb_format(self):
        """Test InfluxDB output format."""
        mock_performance_metrics = PerformanceMetrics(
            collection_start=datetime.now(),
            collection_end=datetime.now(),
            total_executions=2,
            successful_executions=2,
            failed_executions=0,
            llm_api_calls=5,
            total_tokens_consumed=500,
            average_execution_time_ms=100.0,
        )

        with patch(
            "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.metrics_collector.get_metrics_summary"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_performance_metrics

            result = self.runner.invoke(app, ["metrics", "--format", "influxdb"])

            assert result.exit_code == 0
            assert "cognivault_performance" in result.stdout
            assert "total_agents=2" in result.stdout
            assert "successful_agents=2" in result.stdout

    def test_csv_format(self):
        """Test CSV output format."""
        mock_performance_metrics = PerformanceMetrics(
            collection_start=datetime.now(),
            collection_end=datetime.now(),
            total_executions=2,
            successful_executions=2,
            failed_executions=0,
            llm_api_calls=5,
            total_tokens_consumed=500,
            average_execution_time_ms=100.0,
        )

        with patch(
            "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.metrics_collector.get_metrics_summary"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_performance_metrics

            result = self.runner.invoke(app, ["metrics", "--format", "csv"])

            assert result.exit_code == 0
            assert "start_time,end_time,total_agents" in result.stdout
            # Check the actual data row: total_agents=2, successful_agents=2, failed_agents=0, total_llm_calls=5
            assert "2,2,0,0.0000,5" in result.stdout

    def test_error_handling(self):
        """Test CLI error handling."""
        # Mock the diagnostics instance and resource scheduler to prevent async task creation
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch.object(
                diagnostics_cli.diagnostics,
                "quick_health_check",
                new_callable=AsyncMock,
            ) as mock_health_check,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler
            mock_health_check.side_effect = Exception("Test error")

            result = self.runner.invoke(app, ["health"])

            # CLI raises exception when health check fails - this is the current behavior
            assert result.exit_code != 0  # Should fail
            # Spinner shows before the exception occurs
            assert "Running health checks" in result.stdout

    def test_invalid_format(self):
        """Test handling of invalid output format."""
        result = self.runner.invoke(app, ["health", "--format", "invalid"])

        # Should fail with invalid choice
        assert result.exit_code != 0

    def test_health_output_format(self):
        """Test health command output format."""
        # Mock the diagnostics instance and resource scheduler to prevent async task creation
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch.object(
                diagnostics_cli.diagnostics,
                "quick_health_check",
                new_callable=AsyncMock,
            ) as mock_health_check,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler
            # quick_health_check returns a dict with status, timestamp, components, uptime_seconds
            mock_health_check.return_value = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "total": 1,
                    "healthy": 1,
                    "degraded": 0,
                    "unhealthy": 0,
                    "unknown": 0,
                },
                "uptime_seconds": 0.0,
            }

            result = self.runner.invoke(app, ["health"])

            assert result.exit_code == 0
            assert "CogniVault Health Check" in result.stdout
            assert "Status: HEALTHY" in result.stdout
            assert "Component Summary" in result.stdout

    def test_health_command_quiet_mode(self):
        """Test health command in quiet mode."""
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch.object(
                diagnostics_cli.diagnostics,
                "quick_health_check",
                new_callable=AsyncMock,
            ) as mock_health_check,
        ):
            # Mock ResourceScheduler to prevent background task creation
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler

            mock_health_check.return_value = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {"total": 1, "healthy": 1},
                "uptime_seconds": 0.0,
            }

            result = self.runner.invoke(app, ["health", "--quiet"])

            # Should exit cleanly with minimal output
            assert result.exit_code == 0

    def test_metrics_command_agents_only(self):
        """Test metrics command with agents-only flag."""
        mock_performance_summary = {
            "execution": {
                "total": 10,
                "successful": 9,
                "failed": 1,
                "success_rate": 0.9,
            },
            "timing_ms": {
                "average": 150.0,
                "min": 100.0,
                "max": 200.0,
                "p50": 150.0,
                "p95": 180.0,
                "p99": 195.0,
            },
            "agents": {
                "refiner": {
                    "executions": 5,
                    "success_rate": 0.8,
                    "avg_duration_ms": 120.0,
                    "tokens_consumed": 1000,
                },
                "critic": {
                    "executions": 5,
                    "success_rate": 1.0,
                    "avg_duration_ms": 180.0,
                    "tokens_consumed": 800,
                },
            },
            "errors": {"breakdown": {}},
        }

        with patch(
            "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.get_performance_summary"
        ) as mock_get_performance:
            mock_get_performance.return_value = mock_performance_summary

            result = self.runner.invoke(app, ["metrics", "--agents"])

            assert result.exit_code == 0
            assert "Performance Metrics" in result.stdout
            assert "Agent Metrics" in result.stdout
            assert "refiner" in result.stdout
            assert "critic" in result.stdout

    def test_agents_command_specific_agent(self):
        """Test agents command for specific agent."""
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch(
                "cognivault.diagnostics.cli.DiagnosticsManager"
            ) as mock_manager_class,
        ):
            # Mock ResourceScheduler
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler

            mock_manager = AsyncMock()
            mock_agent_status = {
                "timestamp": datetime.now().isoformat(),
                "total_agents": 1,
                "agents": {
                    "refiner": {
                        "name": "refiner",
                        "description": "Refines and improves user queries",
                        "requires_llm": True,
                        "is_critical": True,
                        "failure_strategy": "fail_fast",
                        "dependencies": [],
                        "health_check": True,
                        "metrics": {
                            "executions": 10,
                            "success_rate": 0.9,
                            "avg_duration_ms": 110.0,
                            "tokens_consumed": 500,
                        },
                    }
                },
            }
            mock_manager.get_agent_status.return_value = mock_agent_status
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["agents", "--agent", "refiner"])

            assert result.exit_code == 0
            assert "refiner Agent Details" in result.stdout
            assert "Refines and improves user queries" in result.stdout

    def test_agents_command_json_output(self):
        """Test agents command with JSON output."""
        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch(
                "cognivault.diagnostics.cli.DiagnosticsManager"
            ) as mock_manager_class,
        ):
            # Mock ResourceScheduler
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler

            mock_manager = AsyncMock()
            mock_agent_status = {
                "timestamp": datetime.now().isoformat(),
                "total_agents": 2,
                "agents": {
                    "refiner": {"health_check": True},
                    "critic": {"health_check": False},
                },
            }
            mock_manager.get_agent_status.return_value = mock_agent_status
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["agents", "--json"])

            assert result.exit_code == 0
            # The CLI returns real agent data, just check structure
            assert '"total_agents"' in result.stdout
            assert '"agents"' in result.stdout
            assert '"refiner"' in result.stdout

    def test_config_command_validate_only(self):
        """Test config command with validate only flag."""
        mock_config_report = {
            "timestamp": datetime.now().isoformat(),
            "environment": "development",
            "validation": {
                "is_valid": False,
                "error_count": 1,
                "errors": ["Missing API key"],
            },
            "configuration": {},
            "recommendations": [],
        }

        with patch.object(
            diagnostics_cli.diagnostics,
            "get_configuration_report",
            return_value=mock_config_report,
        ):
            result = self.runner.invoke(app, ["config", "--validate"])

            assert result.exit_code == 0
            assert "Configuration has 1 errors" in result.stdout
            assert "Missing API key" in result.stdout

    def test_full_command_with_window_filter(self):
        """Test full diagnostics command with time window."""
        timestamp = datetime.now()

        mock_diagnostics = SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths={},
            performance_metrics=PerformanceMetrics(
                collection_start=timestamp,
                collection_end=timestamp,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                llm_api_calls=0,
                total_tokens_consumed=0,
                average_execution_time_ms=0.0,
            ),
            system_info={},
            configuration_status={},
            environment_info={},
        )

        with (
            patch(
                "cognivault.dependencies.resource_scheduler.ResourceScheduler"
            ) as mock_scheduler_class,
            patch(
                "cognivault.diagnostics.cli.diagnostics_cli.diagnostics.run_full_diagnostics"
            ) as mock_run_full,
        ):
            # Mock ResourceScheduler
            mock_scheduler = AsyncMock()
            mock_scheduler.request_resources = AsyncMock(return_value=[])
            mock_scheduler.release_resources = AsyncMock(return_value=True)
            mock_scheduler._scheduler_running = False
            mock_scheduler_class.return_value = mock_scheduler

            mock_run_full.return_value = mock_diagnostics

            result = self.runner.invoke(app, ["full", "--window", "30"])

            assert result.exit_code == 0
            assert "Complete System Diagnostics" in result.stdout
