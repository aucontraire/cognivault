"""
Diagnostic Health Factory Functions for Testing.

Provides factory functions for creating diagnostic health objects with sensible defaults
to eliminate parameter unfilled warnings in tests and improve maintainability.

This implements factory patterns for ComponentHealth, PerformanceMetrics, and SystemDiagnostics
objects used in diagnostic formatters and health monitoring tests.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from cognivault.diagnostics.health import HealthStatus, ComponentHealth
from cognivault.diagnostics.metrics import PerformanceMetrics
from cognivault.diagnostics.diagnostics import SystemDiagnostics


class ComponentHealthFactory:
    """Factory for creating ComponentHealth instances with sensible defaults."""

    @staticmethod
    def basic_healthy_component(
        name: str = "test_component",
        message: str = "Component is healthy",
        **overrides: Any,
    ) -> ComponentHealth:
        """Create basic healthy component with minimal required parameters."""
        # Extract specific parameters from overrides to avoid duplicates
        check_time = overrides.pop("check_time", datetime.now(timezone.utc))
        status = overrides.pop("status", HealthStatus.HEALTHY)
        details = overrides.pop("details", {})

        return ComponentHealth(
            name=name,
            status=status,
            message=message,
            details=details,
            check_time=check_time,
            **overrides,
        )

    @staticmethod
    def healthy_component_with_details(
        name: str = "test_component",
        message: str = "Component is healthy",
        details: Optional[Dict[str, Any]] = None,
        response_time_ms: float = 25.0,
        **overrides: Any,
    ) -> ComponentHealth:
        """Create healthy component with details and response time."""
        if details is None:
            details = {"test": "value"}

        # Extract specific parameters from overrides to avoid duplicates
        check_time = overrides.pop("check_time", datetime.now(timezone.utc))
        status = overrides.pop("status", HealthStatus.HEALTHY)

        return ComponentHealth(
            name=name,
            status=status,
            message=message,
            details=details,
            check_time=check_time,
            response_time_ms=response_time_ms,
            **overrides,
        )

    @staticmethod
    def degraded_component(
        name: str = "degraded_component",
        message: str = "Component has issues",
        details: Optional[Dict[str, Any]] = None,
        response_time_ms: float = 150.0,
        **overrides: Any,
    ) -> ComponentHealth:
        """Create degraded component with issues."""
        if details is None:
            details = {"provider": "openai", "errors": ["timeout"]}

        return ComponentHealth(
            name=name,
            status=HealthStatus.DEGRADED,
            message=message,
            details=details,
            check_time=datetime.now(timezone.utc),
            response_time_ms=response_time_ms,
            **overrides,
        )

    @staticmethod
    def unhealthy_component(
        name: str = "unhealthy_component",
        message: str = "Component failed",
        details: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> ComponentHealth:
        """Create unhealthy component with failure details."""
        if details is None:
            details = {"error": "Connection failed"}

        return ComponentHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=message,
            details=details,
            check_time=datetime.now(timezone.utc),
            **overrides,
        )

    @staticmethod
    def agent_registry_health(
        status: HealthStatus = HealthStatus.HEALTHY,
        agent_count: int = 4,
        response_time_ms: float = 25.0,
        **overrides: Any,
    ) -> ComponentHealth:
        """Create agent registry health component."""
        message = (
            f"Registry is healthy with {agent_count} agents"
            if status == HealthStatus.HEALTHY
            else "Registry has issues"
        )

        # Extract specific parameters from overrides to avoid duplicates
        check_time = overrides.pop("check_time", datetime.now(timezone.utc))

        return ComponentHealth(
            name="agent_registry",
            status=status,
            message=message,
            details={"agent_count": agent_count},
            check_time=check_time,
            response_time_ms=response_time_ms,
            **overrides,
        )

    @staticmethod
    def llm_connectivity_health(
        status: HealthStatus = HealthStatus.HEALTHY,
        provider: str = "openai",
        response_time_ms: float = 150.0,
        **overrides: Any,
    ) -> ComponentHealth:
        """Create LLM connectivity health component."""
        if status == HealthStatus.HEALTHY:
            message = "LLM connectivity healthy"
            details: Dict[str, Any] = {"provider": provider}
        else:
            message = "LLM has issues"
            details = {"provider": provider, "errors": ["timeout"]}

        # Extract specific parameters from overrides to avoid duplicates
        check_time = overrides.pop("check_time", datetime.now(timezone.utc))
        # Override details if provided in overrides
        details = overrides.pop("details", details)

        return ComponentHealth(
            name="llm_connectivity",
            status=status,
            message=message,
            details=details,
            check_time=check_time,
            response_time_ms=response_time_ms,
            **overrides,
        )

    @staticmethod
    def component_with_special_characters(
        name: str = "test component",  # Space in name
        status: HealthStatus = HealthStatus.HEALTHY,
        message: str = 'Component has "quotes" and, commas',
        **overrides: Any,
    ) -> ComponentHealth:
        """Create component with special characters for CSV/formatting testing."""
        # Extract specific parameters from overrides to avoid duplicates
        check_time = overrides.pop("check_time", datetime.now(timezone.utc))
        details = overrides.pop(
            "details", {"description": "Multi-line\ntext with\ttabs"}
        )

        return ComponentHealth(
            name=name,
            status=status,
            message=message,
            details=details,
            check_time=check_time,
            **overrides,
        )

    @staticmethod
    def with_current_timestamp(
        name: str = "test_component",
        status: HealthStatus = HealthStatus.HEALTHY,
        message: str = "Component is healthy",
        **overrides: Any,
    ) -> ComponentHealth:
        """Create component with current timestamp for time-sensitive tests."""
        return ComponentHealth(
            name=name,
            status=status,
            message=message,
            check_time=datetime.now(timezone.utc),
            **overrides,
        )


class PerformanceMetricsFactory:
    """Factory for creating PerformanceMetrics instances with sensible defaults."""

    @staticmethod
    def basic_metrics(
        total_executions: int = 4,
        successful_executions: int = 3,
        failed_executions: int = 1,
        **overrides: Any,
    ) -> PerformanceMetrics:
        """Create basic performance metrics with standard values."""
        start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        end_time = datetime.now(timezone.utc)

        # Extract specific parameters from overrides to avoid duplicates
        collection_start = overrides.pop("collection_start", start_time)
        collection_end = overrides.pop("collection_end", end_time)

        return PerformanceMetrics(
            collection_start=collection_start,
            collection_end=collection_end,
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            llm_api_calls=10,
            total_tokens_consumed=1500,
            average_execution_time_ms=125.5,
            **overrides,
        )

    @staticmethod
    def minimal_metrics(**overrides: Any) -> PerformanceMetrics:
        """Create minimal performance metrics for simple tests."""
        timestamp = datetime.now(timezone.utc)

        # Extract specific parameters from overrides to avoid duplicates
        collection_start = overrides.pop(
            "collection_start", timestamp - timedelta(minutes=1)
        )
        collection_end = overrides.pop("collection_end", timestamp)

        return PerformanceMetrics(
            collection_start=collection_start,
            collection_end=collection_end,
            total_executions=1,
            successful_executions=1,
            failed_executions=0,
            llm_api_calls=1,
            total_tokens_consumed=100,
            average_execution_time_ms=100.0,
            **overrides,
        )

    @staticmethod
    def with_time_range(
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **overrides: Any,
    ) -> PerformanceMetrics:
        """Create performance metrics with specific time range."""
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        # Extract specific parameters from overrides to avoid duplicates
        total_executions = overrides.pop("total_executions", 4)
        successful_executions = overrides.pop("successful_executions", 3)
        failed_executions = overrides.pop("failed_executions", 1)
        llm_api_calls = overrides.pop("llm_api_calls", 10)
        total_tokens_consumed = overrides.pop("total_tokens_consumed", 1500)
        average_execution_time_ms = overrides.pop("average_execution_time_ms", 125.5)

        return PerformanceMetrics(
            collection_start=start_time,
            collection_end=end_time,
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            llm_api_calls=llm_api_calls,
            total_tokens_consumed=total_tokens_consumed,
            average_execution_time_ms=average_execution_time_ms,
            **overrides,
        )


class SystemDiagnosticsFactory:
    """Factory for creating SystemDiagnostics instances with sensible defaults."""

    @staticmethod
    def basic_diagnostics(
        overall_health: HealthStatus = HealthStatus.HEALTHY,
        timestamp: Optional[datetime] = None,
        **overrides: Any,
    ) -> SystemDiagnostics:
        """Create basic system diagnostics with healthy components."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Extract specific parameters from overrides to avoid duplicates
        component_healths = overrides.pop(
            "component_healths",
            {
                "test_component": ComponentHealthFactory.basic_healthy_component(
                    name="test_component",
                    message="Component is healthy",
                    check_time=timestamp,
                )
            },
        )

        performance_metrics = overrides.pop(
            "performance_metrics", PerformanceMetricsFactory.minimal_metrics()
        )
        system_info = overrides.pop(
            "system_info", {"version": "1.0.0", "platform": "Darwin"}
        )
        configuration_status = overrides.pop("configuration_status", {"is_valid": True})
        environment_info = overrides.pop("environment_info", {"mode": "test"})

        return SystemDiagnostics(
            timestamp=timestamp,
            overall_health=overall_health,
            component_healths=component_healths,
            performance_metrics=performance_metrics,
            system_info=system_info,
            configuration_status=configuration_status,
            environment_info=environment_info,
            **overrides,
        )

    @staticmethod
    def comprehensive_diagnostics(
        timestamp: Optional[datetime] = None,
        **overrides: Any,
    ) -> SystemDiagnostics:
        """Create comprehensive system diagnostics with multiple components."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        component_healths = {
            "agent_registry": ComponentHealthFactory.agent_registry_health(
                check_time=timestamp
            ),
            "llm_connectivity": ComponentHealthFactory.llm_connectivity_health(
                check_time=timestamp
            ),
            "test_component": ComponentHealthFactory.basic_healthy_component(
                check_time=timestamp
            ),
        }

        performance_metrics = PerformanceMetricsFactory.basic_metrics()

        return SystemDiagnostics(
            timestamp=timestamp,
            overall_health=HealthStatus.HEALTHY,
            component_healths=component_healths,
            performance_metrics=performance_metrics,
            system_info={"version": "1.0.0", "platform": "Darwin"},
            configuration_status={"is_valid": True},
            environment_info={"mode": "test"},
            **overrides,
        )


class DiagnosticHealthTestPatterns:
    """Common test patterns for diagnostic health objects."""

    @staticmethod
    def create_health_results_dict(
        component_count: int = 2,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, ComponentHealth]:
        """Create a dictionary of component health results for testing."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        results = {}
        if component_count >= 1:
            results["agent_registry"] = ComponentHealthFactory.agent_registry_health(
                check_time=timestamp
            )
        if component_count >= 2:
            results["llm_connectivity"] = (
                ComponentHealthFactory.llm_connectivity_health(
                    status=HealthStatus.DEGRADED, check_time=timestamp
                )
            )
        if component_count >= 3:
            results["test_component"] = ComponentHealthFactory.basic_healthy_component(
                check_time=timestamp
            )

        return results

    @staticmethod
    def create_empty_health_results() -> Dict[str, ComponentHealth]:
        """Create empty health results for edge case testing."""
        return {}

    @staticmethod
    def create_mixed_health_results(
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, ComponentHealth]:
        """Create health results with mixed status for comprehensive testing."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return {
            "healthy_component": ComponentHealthFactory.basic_healthy_component(
                name="healthy_component", check_time=timestamp
            ),
            "degraded_component": ComponentHealthFactory.degraded_component(
                name="degraded_component", check_time=timestamp
            ),
            "unhealthy_component": ComponentHealthFactory.unhealthy_component(
                name="unhealthy_component", check_time=timestamp
            ),
        }
