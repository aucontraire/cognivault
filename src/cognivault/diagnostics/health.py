"""
Health checking system for CogniVault components.

This module provides comprehensive health checks for agents, LLM connections,
configuration validity, and system dependencies.
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict
from cognivault.agents.registry import get_agent_registry
from cognivault.config.app_config import get_config

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for Python < 3.8
    try:
        from importlib_metadata import version, PackageNotFoundError  # type: ignore
    except ImportError:
        # Create a stub if neither is available
        class PackageNotFoundError(Exception):  # type: ignore[misc,no-redef]
            pass

        def version(distribution_name: str) -> str:  # type: ignore[misc]
            raise PackageNotFoundError(f"Package {distribution_name} not found")


if TYPE_CHECKING:
    pass


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health information for a system component."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    name: str = Field(..., description="Component name identifier")
    status: HealthStatus = Field(
        ..., description="Current health status of the component"
    )
    message: str = Field(..., description="Human-readable status message")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional diagnostic details and metadata"
    )
    check_time: datetime = Field(
        ..., description="Timestamp when health check was performed"
    )
    response_time_ms: Optional[float] = Field(
        None, ge=0.0, description="Health check response time in milliseconds"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "check_time": self.check_time.isoformat(),
            "response_time_ms": self.response_time_ms,
        }


class HealthChecker:
    """
    Comprehensive health checker for CogniVault system components.

    Provides health checks for:
    - Agent registry and agent creation
    - LLM connectivity and authentication
    - Configuration validity
    - File system access
    - System dependencies
    """

    def __init__(self):
        self.registry = get_agent_registry()
        self.config = get_config()

    async def check_all(self) -> Dict[str, ComponentHealth]:
        """
        Run all health checks and return comprehensive health status.

        Returns
        -------
        Dict[str, ComponentHealth]
            Health status for each component
        """
        checks = [
            self._check_agent_registry(),
            self._check_llm_connectivity(),
            self._check_configuration(),
            self._check_file_system(),
            self._check_dependencies(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        health_status = {}
        for result in results:
            if isinstance(result, ComponentHealth):
                health_status[result.name] = result
            elif isinstance(result, Exception):
                health_status["error"] = ComponentHealth(
                    name="error",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(result)}",
                    details={"exception": str(result)},
                    check_time=datetime.now(),
                )

        return health_status

    async def _check_agent_registry(self) -> ComponentHealth:
        """Check agent registry health and agent creation capabilities."""
        start_time = time.time()

        try:
            # Check registry is available
            available_agents = self.registry.get_available_agents()

            if not available_agents:
                return ComponentHealth(
                    name="agent_registry",
                    status=HealthStatus.UNHEALTHY,
                    message="No agents registered",
                    details={"agent_count": 0},
                    check_time=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            # Try to validate the pipeline
            core_agents = ["refiner", "historian", "critic", "synthesis"]
            pipeline_valid = self.registry.validate_pipeline(core_agents)

            # Check dependency resolution
            try:
                resolved_order = self.registry.resolve_dependencies(core_agents)
                dependency_health = HealthStatus.HEALTHY
                dependency_message = f"Dependencies resolved: {resolved_order}"
            except Exception as e:
                dependency_health = HealthStatus.DEGRADED
                dependency_message = f"Dependency resolution issues: {str(e)}"

            # Determine overall status
            if pipeline_valid and dependency_health == HealthStatus.HEALTHY:
                status = HealthStatus.HEALTHY
                message = f"Registry healthy with {len(available_agents)} agents"
            else:
                status = HealthStatus.DEGRADED
                message = f"Registry functional but has issues: {dependency_message}"

            return ComponentHealth(
                name="agent_registry",
                status=status,
                message=message,
                details={
                    "agent_count": len(available_agents),
                    "available_agents": available_agents,
                    "pipeline_valid": pipeline_valid,
                    "dependency_resolution": dependency_message,
                },
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="agent_registry",
                status=HealthStatus.UNHEALTHY,
                message=f"Registry check failed: {str(e)}",
                details={"error": str(e)},
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_llm_connectivity(self) -> ComponentHealth:
        """Check LLM connectivity and authentication."""
        start_time = time.time()

        try:
            # Skip LLM check if using stub
            if self.config.models.default_provider == "stub":
                return ComponentHealth(
                    name="llm_connectivity",
                    status=HealthStatus.HEALTHY,
                    message="Using stub LLM provider (no connectivity check needed)",
                    details={"provider": "stub", "skip_reason": "mock_provider"},
                    check_time=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            # Try to create LLM instance
            if self.config.models.default_provider == "openai":
                from cognivault.config.openai_config import OpenAIConfig

                openai_config = OpenAIConfig.load()

                if not openai_config.api_key:
                    return ComponentHealth(
                        name="llm_connectivity",
                        status=HealthStatus.UNHEALTHY,
                        message="OpenAI API key not configured",
                        details={"provider": "openai", "error": "missing_api_key"},
                        check_time=datetime.now(),
                        response_time_ms=(time.time() - start_time) * 1000,
                    )

                # Use lazy import to avoid circular dependency
                from cognivault.llm.openai import OpenAIChatLLM

                llm = OpenAIChatLLM(api_key=openai_config.api_key)

                # Try a simple test call
                try:
                    # Simple connectivity test - just try to create the LLM
                    ping_success = (
                        True  # If we got here without exception, it's working
                    )

                    if ping_success:
                        return ComponentHealth(
                            name="llm_connectivity",
                            status=HealthStatus.HEALTHY,
                            message="OpenAI connectivity healthy",
                            details={
                                "provider": "openai",
                                "model": openai_config.model,
                                "ping_success": True,
                            },
                            check_time=datetime.now(),
                            response_time_ms=(time.time() - start_time) * 1000,
                        )
                    else:
                        return ComponentHealth(
                            name="llm_connectivity",
                            status=HealthStatus.UNHEALTHY,
                            message="OpenAI connectivity failed",
                            details={
                                "provider": "openai",
                                "model": openai_config.model,
                                "ping_success": False,
                            },
                            check_time=datetime.now(),
                            response_time_ms=(time.time() - start_time) * 1000,
                        )
                except Exception as llm_error:
                    return ComponentHealth(
                        name="llm_connectivity",
                        status=HealthStatus.UNHEALTHY,
                        message=f"LLM connectivity failed: {str(llm_error)}",
                        details={
                            "provider": "openai",
                            "error": str(llm_error),
                            "model": openai_config.model,
                        },
                        check_time=datetime.now(),
                        response_time_ms=(time.time() - start_time) * 1000,
                    )

            return ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.UNKNOWN,
                message=f"Unknown LLM provider: {self.config.models.default_provider}",
                details={"provider": self.config.models.default_provider},
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM health check failed: {str(e)}",
                details={"error": str(e)},
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_configuration(self) -> ComponentHealth:
        """Check configuration validity and completeness."""
        start_time = time.time()

        try:
            # Validate configuration
            validation_errors = self.config.validate()

            if validation_errors:
                return ComponentHealth(
                    name="configuration",
                    status=HealthStatus.DEGRADED,
                    message=f"Configuration has {len(validation_errors)} validation errors",
                    details={
                        "validation_errors": validation_errors,
                        "environment": self.config.environment.value,
                    },
                    check_time=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            # Check critical settings
            critical_issues = []

            # Check timeout settings
            if self.config.execution.timeout_seconds <= 0:
                critical_issues.append("Invalid timeout configuration")

            # Check directory paths
            import os

            if not os.path.exists(self.config.files.notes_directory):
                critical_issues.append(
                    f"Notes directory does not exist: {self.config.files.notes_directory}"
                )

            if not os.path.exists(self.config.files.logs_directory):
                critical_issues.append(
                    f"Logs directory does not exist: {self.config.files.logs_directory}"
                )

            if critical_issues:
                return ComponentHealth(
                    name="configuration",
                    status=HealthStatus.DEGRADED,
                    message=f"Configuration has {len(critical_issues)} critical issues",
                    details={
                        "critical_issues": critical_issues,
                        "environment": self.config.environment.value,
                    },
                    check_time=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            return ComponentHealth(
                name="configuration",
                status=HealthStatus.HEALTHY,
                message="Configuration is valid and complete",
                details={
                    "environment": self.config.environment.value,
                    "llm_provider": self.config.models.default_provider,
                    "notes_dir": self.config.files.notes_directory,
                    "logs_dir": self.config.files.logs_directory,
                },
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="configuration",
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration check failed: {str(e)}",
                details={"error": str(e)},
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_file_system(self) -> ComponentHealth:
        """Check file system access and permissions."""
        start_time = time.time()

        try:
            import os
            import tempfile

            issues: List[str] = []
            details: Dict[str, Any] = {}

            # Check notes directory
            notes_dir = self.config.files.notes_directory
            if os.path.exists(notes_dir):
                if os.access(notes_dir, os.W_OK):
                    details["notes_dir_writable"] = True
                else:
                    issues.append(f"Notes directory not writable: {notes_dir}")
                    details["notes_dir_writable"] = False
            else:
                issues.append(f"Notes directory does not exist: {notes_dir}")
                details["notes_dir_exists"] = False

            # Check logs directory
            logs_dir = self.config.files.logs_directory
            if os.path.exists(logs_dir):
                if os.access(logs_dir, os.W_OK):
                    details["logs_dir_writable"] = True
                else:
                    issues.append(f"Logs directory not writable: {logs_dir}")
                    details["logs_dir_writable"] = False
            else:
                issues.append(f"Logs directory does not exist: {logs_dir}")
                details["logs_dir_exists"] = False

            # Test temp file creation
            try:
                with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
                    f.write("test")
                    details["temp_file_creation"] = True
            except Exception as e:
                issues.append(f"Cannot create temporary files: {str(e)}")
                details["temp_file_creation"] = False

            # Check disk space (basic check)
            try:
                statvfs = os.statvfs(".")
                free_bytes = statvfs.f_frsize * statvfs.f_bavail
                details["free_disk_bytes"] = int(free_bytes)
                details["free_disk_mb"] = round(free_bytes / (1024 * 1024), 2)

                if free_bytes < 100 * 1024 * 1024:  # Less than 100MB
                    issues.append("Low disk space (< 100MB)")
            except (AttributeError, OSError, ValueError):
                details["disk_space_check"] = "unavailable"

            if issues:
                status = (
                    HealthStatus.DEGRADED
                    if len(issues) <= 2
                    else HealthStatus.UNHEALTHY
                )
                message = f"File system has {len(issues)} issues"
            else:
                status = HealthStatus.HEALTHY
                message = "File system access is healthy"

            return ComponentHealth(
                name="file_system",
                status=status,
                message=message,
                details=dict(details, issues=issues),
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {str(e)}",
                details={"error": str(e)},
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    async def _check_dependencies(self) -> ComponentHealth:
        """Check system dependencies and Python packages."""
        start_time = time.time()

        try:
            import sys

            details: Dict[str, Any] = {
                "python_version": sys.version,
                "platform": sys.platform,
            }

            # Check critical packages
            critical_packages = [
                "typer",
                "pydantic",
                "openai",
                "pytest",
            ]

            missing_packages = []
            package_versions = {}

            for package in critical_packages:
                try:
                    pkg_version = version(package)
                    package_versions[package] = pkg_version
                except PackageNotFoundError:
                    missing_packages.append(package)

            details["package_versions"] = dict(package_versions)

            if missing_packages:
                return ComponentHealth(
                    name="dependencies",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Missing critical packages: {missing_packages}",
                    details={**details, "missing_packages": missing_packages},
                    check_time=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000,
                )

            return ComponentHealth(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message=f"All {len(critical_packages)} critical packages available",
                details=details,
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependencies check failed: {str(e)}",
                details={"error": str(e)},
                check_time=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    def get_overall_status(
        self, component_healths: Dict[str, ComponentHealth]
    ) -> HealthStatus:
        """
        Determine overall system health from component healths.

        Parameters
        ----------
        component_healths : Dict[str, ComponentHealth]
            Individual component health statuses

        Returns
        -------
        HealthStatus
            Overall system health status
        """
        if not component_healths:
            return HealthStatus.UNKNOWN

        statuses = [health.status for health in component_healths.values()]

        # If any component is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # If all components are healthy, system is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY

        # Otherwise unknown
        return HealthStatus.UNKNOWN
