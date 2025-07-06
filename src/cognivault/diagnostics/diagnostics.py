"""
Main diagnostics manager for CogniVault system monitoring.

This module coordinates health checks, metrics collection, and system diagnostics
to provide comprehensive observability for CogniVault operations.
"""

import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from .health import HealthChecker, ComponentHealth, HealthStatus
from .metrics import PerformanceMetrics, get_metrics_collector
from cognivault.config.app_config import get_config


@dataclass
class SystemDiagnostics:
    """Complete system diagnostic information."""

    # System information
    timestamp: datetime
    system_info: Dict[str, Any]

    # Health status
    overall_health: HealthStatus
    component_healths: Dict[str, ComponentHealth]

    # Performance metrics
    performance_metrics: PerformanceMetrics

    # Configuration status
    configuration_status: Dict[str, Any]

    # Environment information
    environment_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_info": self.system_info,
            "health": {
                "overall_status": self.overall_health.value,
                "components": {
                    name: health.to_dict()
                    for name, health in self.component_healths.items()
                },
            },
            "performance": self.performance_metrics.to_dict(),
            "configuration": self.configuration_status,
            "environment": self.environment_info,
        }


class DiagnosticsManager:
    """
    Central diagnostics manager for CogniVault system monitoring.

    Coordinates health checks, metrics collection, and comprehensive
    system diagnostics to provide observability into CogniVault operations.
    """

    def __init__(self):
        self.health_checker = HealthChecker()
        self.metrics_collector = get_metrics_collector()
        self.config = get_config()

    async def run_full_diagnostics(
        self, metrics_window_minutes: Optional[int] = None
    ) -> SystemDiagnostics:
        """
        Run comprehensive system diagnostics.

        Parameters
        ----------
        metrics_window_minutes : int, optional
            Time window for metrics aggregation (None for all time)

        Returns
        -------
        SystemDiagnostics
            Complete diagnostic information
        """
        # Run health checks
        component_healths = await self.health_checker.check_all()
        overall_health = self.health_checker.get_overall_status(component_healths)

        # Collect performance metrics
        performance_metrics = self.metrics_collector.get_metrics_summary(
            metrics_window_minutes
        )

        # Gather system information
        system_info = self._get_system_info()

        # Get configuration status
        config_status = self._get_configuration_status()

        # Get environment information
        env_info = self._get_environment_info()

        return SystemDiagnostics(
            timestamp=datetime.now(),
            system_info=system_info,
            overall_health=overall_health,
            component_healths=component_healths,
            performance_metrics=performance_metrics,
            configuration_status=config_status,
            environment_info=env_info,
        )

    async def quick_health_check(self) -> Dict[str, Any]:
        """
        Run a quick health check suitable for monitoring endpoints.

        Returns
        -------
        Dict[str, Any]
            Basic health status information
        """
        component_healths = await self.health_checker.check_all()
        overall_health = self.health_checker.get_overall_status(component_healths)

        # Count components by status
        status_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
        for health in component_healths.values():
            status_counts[health.status.value] += 1

        return {
            "status": overall_health.value,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "total": len(component_healths),
                "healthy": status_counts["healthy"],
                "degraded": status_counts["degraded"],
                "unhealthy": status_counts["unhealthy"],
                "unknown": status_counts["unknown"],
            },
            "uptime_seconds": self._get_uptime_seconds(),
        }

    def get_performance_summary(
        self, window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics summary.

        Parameters
        ----------
        window_minutes : int, optional
            Time window for metrics aggregation

        Returns
        -------
        Dict[str, Any]
            Performance metrics summary
        """
        metrics = self.metrics_collector.get_metrics_summary(window_minutes)

        # Add derived metrics
        summary = metrics.to_dict()
        summary["derived"] = {
            "avg_tokens_per_execution": 0.0,
            "executions_per_minute": 0.0,
            "error_rate": 0.0,
        }

        if metrics.total_executions > 0:
            summary["derived"]["avg_tokens_per_execution"] = (
                metrics.total_tokens_consumed / metrics.total_executions
            )
            summary["derived"]["error_rate"] = (
                metrics.failed_executions / metrics.total_executions
            )

            # Calculate executions per minute
            if metrics.collection_start and metrics.collection_end:
                duration_minutes = (
                    metrics.collection_end - metrics.collection_start
                ).total_seconds() / 60
                if duration_minutes > 0:
                    summary["derived"]["executions_per_minute"] = (
                        metrics.total_executions / duration_minutes
                    )

        return summary

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status and statistics.

        Returns
        -------
        Dict[str, Any]
            Agent status information
        """
        from cognivault.agents.registry import get_agent_registry

        registry = get_agent_registry()
        available_agents = registry.get_available_agents()

        # Get agent-specific metrics
        metrics = self.metrics_collector.get_metrics_summary()
        agent_metrics = metrics.agent_metrics

        agent_status = {}
        for agent_name in available_agents:
            # Get agent metadata
            try:
                metadata = registry.get_agent_info(agent_name)

                agent_status[agent_name] = {
                    "name": agent_name,
                    "description": metadata.description,
                    "requires_llm": metadata.requires_llm,
                    "is_critical": metadata.is_critical,
                    "failure_strategy": metadata.failure_strategy.value,
                    "dependencies": metadata.dependencies,
                    "health_check": registry.check_health(agent_name),
                    "metrics": agent_metrics.get(
                        agent_name,
                        {
                            "executions": 0,
                            "successes": 0,
                            "failures": 0,
                            "success_rate": 0.0,
                            "avg_duration_ms": 0.0,
                            "tokens_consumed": 0,
                        },
                    ),
                }
            except Exception as e:
                agent_status[agent_name] = {
                    "name": agent_name,
                    "error": f"Failed to get agent info: {str(e)}",
                    "health_check": False,
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(available_agents),
            "agents": agent_status,
        }

    def get_configuration_report(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration report.

        Returns
        -------
        Dict[str, Any]
            Configuration status and validation report
        """
        config_status = self._get_configuration_status()

        # Add validation report
        validation_errors = self.config.validate()

        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment.value,
            "validation": {
                "is_valid": len(validation_errors) == 0,
                "error_count": len(validation_errors),
                "errors": validation_errors,
            },
            "configuration": config_status,
            "recommendations": self._get_configuration_recommendations(),
        }

        return report

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "architecture": platform.architecture(),
        }

    def _get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status."""
        return {
            "environment": self.config.environment.value,
            "debug_mode": self.config.debug_mode,
            "log_level": self.config.log_level.value,
            "execution": {
                "max_retries": self.config.execution.max_retries,
                "timeout_seconds": self.config.execution.timeout_seconds,
                "critic_enabled": self.config.execution.critic_enabled,
                "default_agents": self.config.execution.default_agents,
            },
            "models": {
                "default_provider": self.config.models.default_provider,
                "max_tokens_per_request": self.config.models.max_tokens_per_request,
                "temperature": self.config.models.temperature,
            },
            "files": {
                "notes_directory": self.config.files.notes_directory,
                "logs_directory": self.config.files.logs_directory,
                "max_file_size": self.config.files.max_file_size,
            },
        }

    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        import os

        # Get relevant environment variables
        env_vars = {}
        for key in os.environ:
            if key.startswith("COGNIVAULT_") or key.startswith("OPENAI_"):
                # Mask sensitive values
                if (
                    "key" in key.lower()
                    or "secret" in key.lower()
                    or "token" in key.lower()
                ):
                    env_vars[key] = "***MASKED***"
                else:
                    env_vars[key] = os.environ[key]

        return {
            "environment_variables": env_vars,
            "working_directory": os.getcwd(),
            "path_exists": {
                "notes_dir": os.path.exists(self.config.files.notes_directory),
                "logs_dir": os.path.exists(self.config.files.logs_directory),
            },
        }

    def _get_uptime_seconds(self) -> float:
        """Get approximate uptime in seconds."""
        # Use metrics collector start time as proxy for uptime
        if hasattr(self.metrics_collector, "_collection_start"):
            return (
                datetime.now() - self.metrics_collector._collection_start
            ).total_seconds()
        return 0.0

    def _get_configuration_recommendations(self) -> List[str]:
        """Get configuration recommendations."""
        recommendations = []

        # Check timeout settings
        if self.config.execution.timeout_seconds < 30:
            recommendations.append(
                "Consider increasing timeout_seconds for production use (current: {})".format(
                    self.config.execution.timeout_seconds
                )
            )

        # Check retry settings
        if self.config.execution.max_retries > 5:
            recommendations.append(
                "High retry count may cause delays (current: {})".format(
                    self.config.execution.max_retries
                )
            )

        # Check file size limits
        if self.config.files.max_file_size > 50 * 1024 * 1024:  # 50MB
            recommendations.append(
                "Large file size limit may cause memory issues (current: {} MB)".format(
                    self.config.files.max_file_size // (1024 * 1024)
                )
            )

        # Check directory existence
        import os

        if not os.path.exists(self.config.files.notes_directory):
            recommendations.append(
                f"Create notes directory: {self.config.files.notes_directory}"
            )

        if not os.path.exists(self.config.files.logs_directory):
            recommendations.append(
                f"Create logs directory: {self.config.files.logs_directory}"
            )

        return recommendations
