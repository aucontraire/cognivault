"""
Tests for health checking functionality.

This module tests the health checking system including component health,
health status enumeration, and the HealthChecker class.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

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


from cognivault.diagnostics.health import (
    HealthStatus,
    ComponentHealth,
    HealthChecker,
)


class TestHealthStatus:
    """Test HealthStatus enumeration."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_health_status_comparison(self):
        """Test health status can be compared."""
        # Test basic equality
        assert HealthStatus.HEALTHY == HealthStatus.HEALTHY
        assert HealthStatus.HEALTHY != HealthStatus.DEGRADED

        # Test with string values
        assert HealthStatus.HEALTHY.value == "healthy"


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_component_health_creation(self):
        """Test creating ComponentHealth instance."""
        check_time = datetime.now()
        details = {"test": "value", "count": 42}

        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="Component is healthy",
            details=details,
            check_time=check_time,
            response_time_ms=150.5,
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "Component is healthy"
        assert health.details == details
        assert health.check_time == check_time
        assert health.response_time_ms == 150.5

    def test_component_health_to_dict(self):
        """Test ComponentHealth serialization to dictionary."""
        check_time = datetime.now()
        details = {"test": "value"}

        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.DEGRADED,
            message="Component has issues",
            details=details,
            check_time=check_time,
            response_time_ms=250.0,
        )

        result = health.to_dict()

        assert result["name"] == "test_component"
        assert result["status"] == "degraded"
        assert result["message"] == "Component has issues"
        assert result["details"] == details
        assert result["check_time"] == check_time.isoformat()
        assert result["response_time_ms"] == 250.0

    def test_component_health_without_response_time(self):
        """Test ComponentHealth without response time."""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="Test message",
            details={},
            check_time=datetime.now(),
        )

        result = health.to_dict()
        assert result["response_time_ms"] is None


class TestHealthChecker:
    """Test HealthChecker functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.health_checker = HealthChecker()

    @pytest.mark.asyncio
    async def test_check_all_basic(self):
        """Test basic check_all functionality."""
        with (
            patch.object(self.health_checker, "_check_agent_registry") as mock_registry,
            patch.object(self.health_checker, "_check_llm_connectivity") as mock_llm,
            patch.object(self.health_checker, "_check_configuration") as mock_config,
            patch.object(self.health_checker, "_check_file_system") as mock_fs,
            patch.object(self.health_checker, "_check_dependencies") as mock_deps,
        ):
            # Mock all health checks to return healthy
            mock_registry.return_value = ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                message="Registry is healthy",
                details={},
                check_time=datetime.now(),
            )
            mock_llm.return_value = ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.HEALTHY,
                message="LLM is healthy",
                details={},
                check_time=datetime.now(),
            )
            mock_config.return_value = ComponentHealth(
                name="configuration",
                status=HealthStatus.HEALTHY,
                message="Config is healthy",
                details={},
                check_time=datetime.now(),
            )
            mock_fs.return_value = ComponentHealth(
                name="file_system",
                status=HealthStatus.HEALTHY,
                message="File system is healthy",
                details={},
                check_time=datetime.now(),
            )
            mock_deps.return_value = ComponentHealth(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="Dependencies are healthy",
                details={},
                check_time=datetime.now(),
            )

            result = await self.health_checker.check_all()

            assert len(result) == 5
            assert "agent_registry" in result
            assert "llm_connectivity" in result
            assert "configuration" in result
            assert "file_system" in result
            assert "dependencies" in result

            # All should be healthy
            for component_health in result.values():
                assert component_health.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_all_with_exception(self):
        """Test check_all handles exceptions."""
        with (
            patch.object(self.health_checker, "_check_agent_registry") as mock_registry,
            patch.object(self.health_checker, "_check_llm_connectivity") as mock_llm,
            patch.object(self.health_checker, "_check_configuration") as mock_config,
            patch.object(self.health_checker, "_check_file_system") as mock_fs,
            patch.object(self.health_checker, "_check_dependencies") as mock_deps,
        ):
            # Make one check raise an exception
            mock_registry.side_effect = RuntimeError("Registry check failed")

            # Others return healthy
            mock_llm.return_value = ComponentHealth(
                name="llm_connectivity",
                status=HealthStatus.HEALTHY,
                message="LLM is healthy",
                details={},
                check_time=datetime.now(),
            )
            mock_config.return_value = ComponentHealth(
                name="configuration",
                status=HealthStatus.HEALTHY,
                message="Config is healthy",
                details={},
                check_time=datetime.now(),
            )
            mock_fs.return_value = ComponentHealth(
                name="file_system",
                status=HealthStatus.HEALTHY,
                message="File system is healthy",
                details={},
                check_time=datetime.now(),
            )
            mock_deps.return_value = ComponentHealth(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="Dependencies are healthy",
                details={},
                check_time=datetime.now(),
            )

            result = await self.health_checker.check_all()

            # Should still return results for successful checks plus error
            assert "error" in result
            assert result["error"].status == HealthStatus.UNHEALTHY
            assert "Registry check failed" in result["error"].message

    @pytest.mark.asyncio
    async def test_check_agent_registry_healthy(self):
        """Test agent registry health check when healthy."""
        with (
            patch.object(
                self.health_checker.registry, "get_available_agents"
            ) as mock_agents,
            patch.object(
                self.health_checker.registry, "validate_pipeline"
            ) as mock_validate,
            patch.object(
                self.health_checker.registry, "resolve_dependencies"
            ) as mock_resolve,
        ):
            mock_agents.return_value = ["refiner", "critic", "historian", "synthesis"]
            mock_validate.return_value = True
            mock_resolve.return_value = ["refiner", "critic", "historian", "synthesis"]

            result = await self.health_checker._check_agent_registry()

            assert result.name == "agent_registry"
            assert result.status == HealthStatus.HEALTHY
            assert "Registry healthy with 4 agents" in result.message
            assert result.details["agent_count"] == 4
            assert result.details["pipeline_valid"] is True

    @pytest.mark.asyncio
    async def test_check_agent_registry_no_agents(self):
        """Test agent registry health check when no agents are registered."""
        with patch.object(
            self.health_checker.registry, "get_available_agents"
        ) as mock_agents:
            mock_agents.return_value = []

            result = await self.health_checker._check_agent_registry()

            assert result.name == "agent_registry"
            assert result.status == HealthStatus.UNHEALTHY
            assert "No agents registered" in result.message
            assert result.details["agent_count"] == 0

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_stub_provider(self):
        """Test LLM connectivity check with stub provider."""
        with patch.object(
            self.health_checker.config.models, "default_provider", "stub"
        ):
            result = await self.health_checker._check_llm_connectivity()

            assert result.name == "llm_connectivity"
            assert result.status == HealthStatus.HEALTHY
            assert "Using stub LLM provider" in result.message
            assert result.details["provider"] == "stub"

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_openai_no_api_key(self):
        """Test LLM connectivity check with OpenAI but no API key."""
        with (
            patch.object(
                self.health_checker.config.models, "default_provider", "openai"
            ),
            patch("cognivault.config.openai_config.OpenAIConfig") as mock_config_class,
        ):
            mock_config = MagicMock()
            mock_config.api_key = None
            mock_config_class.load.return_value = mock_config

            result = await self.health_checker._check_llm_connectivity()

            assert result.name == "llm_connectivity"
            assert result.status == HealthStatus.UNHEALTHY
            assert "OpenAI API key not configured" in result.message

    @pytest.mark.asyncio
    async def test_check_configuration_valid(self):
        """Test configuration check when configuration is valid."""
        with (
            patch.object(self.health_checker.config, "validate") as mock_validate,
            patch.object(self.health_checker.config.execution, "timeout_seconds", 30),
            patch("os.path.exists") as mock_exists,
        ):
            mock_validate.return_value = []  # No validation errors
            mock_exists.return_value = True  # Directories exist

            result = await self.health_checker._check_configuration()

            assert result.name == "configuration"
            assert result.status == HealthStatus.HEALTHY
            assert "Configuration is valid and complete" in result.message

    @pytest.mark.asyncio
    async def test_check_configuration_validation_errors(self):
        """Test configuration check with validation errors."""
        with patch.object(self.health_checker.config, "validate") as mock_validate:
            mock_validate.return_value = ["Invalid timeout", "Missing directory"]

            result = await self.health_checker._check_configuration()

            assert result.name == "configuration"
            assert result.status == HealthStatus.DEGRADED
            assert "Configuration has 2 validation errors" in result.message
            assert result.details["validation_errors"] == [
                "Invalid timeout",
                "Missing directory",
            ]

    @pytest.mark.asyncio
    async def test_check_file_system_healthy(self):
        """Test file system check when everything is healthy."""
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.access") as mock_access,
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.statvfs") as mock_statvfs,
        ):
            # Mock file system checks
            mock_exists.return_value = True
            mock_access.return_value = True
            mock_temp.return_value.__enter__.return_value = MagicMock()

            # Mock disk space check
            mock_stat = MagicMock()
            mock_stat.f_frsize = 4096
            mock_stat.f_bavail = 1000000  # ~4GB free
            mock_statvfs.return_value = mock_stat

            result = await self.health_checker._check_file_system()

            assert result.name == "file_system"
            assert result.status == HealthStatus.HEALTHY
            assert "File system access is healthy" in result.message
            assert result.details["notes_dir_writable"] is True
            assert result.details["logs_dir_writable"] is True
            assert result.details["temp_file_creation"] is True

    @pytest.mark.asyncio
    async def test_check_file_system_missing_directories(self):
        """Test file system check when directories are missing."""
        with (
            patch("os.path.exists") as mock_exists,
            patch("tempfile.NamedTemporaryFile") as mock_temp,
        ):
            mock_exists.return_value = False  # Directories don't exist
            mock_temp.return_value.__enter__.return_value = MagicMock()

            result = await self.health_checker._check_file_system()

            assert result.name == "file_system"
            assert result.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
            assert len(result.details["issues"]) > 0

    @pytest.mark.asyncio
    async def test_check_dependencies_all_present(self):
        """Test dependencies check when all packages are present."""
        with patch("cognivault.diagnostics.health.version") as mock_version:
            # Mock successful package lookups
            mock_version.return_value = "1.0.0"

            result = await self.health_checker._check_dependencies()

            assert result.name == "dependencies"
            assert result.status == HealthStatus.HEALTHY
            assert "All 4 critical packages available" in result.message

    @pytest.mark.asyncio
    async def test_check_dependencies_missing_packages(self):
        """Test dependencies check when packages are missing."""
        with patch("cognivault.diagnostics.health.version") as mock_version:
            # Mock missing package
            mock_version.side_effect = PackageNotFoundError("Package not found")

            result = await self.health_checker._check_dependencies()

            assert result.name == "dependencies"
            assert result.status == HealthStatus.UNHEALTHY
            assert "Missing critical packages" in result.message

    def test_get_overall_status_all_healthy(self):
        """Test overall status when all components are healthy."""
        components = {
            "comp1": ComponentHealth(
                "comp1", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
            "comp2": ComponentHealth(
                "comp2", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
        }

        status = self.health_checker.get_overall_status(components)

        assert status == HealthStatus.HEALTHY

    def test_get_overall_status_some_degraded(self):
        """Test overall status when some components are degraded."""
        components = {
            "comp1": ComponentHealth(
                "comp1", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
            "comp2": ComponentHealth(
                "comp2", HealthStatus.DEGRADED, "Issues", {}, datetime.now()
            ),
        }

        status = self.health_checker.get_overall_status(components)

        assert status == HealthStatus.DEGRADED

    def test_get_overall_status_some_unhealthy(self):
        """Test overall status when some components are unhealthy."""
        components = {
            "comp1": ComponentHealth(
                "comp1", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
            "comp2": ComponentHealth(
                "comp2", HealthStatus.UNHEALTHY, "Failed", {}, datetime.now()
            ),
        }

        status = self.health_checker.get_overall_status(components)

        assert status == HealthStatus.UNHEALTHY

    def test_get_overall_status_empty(self):
        """Test overall status when no components are provided."""
        components = {}

        status = self.health_checker.get_overall_status(components)

        assert status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_openai_success(self):
        """Test LLM connectivity check with successful OpenAI connection."""

        with (
            patch.object(
                self.health_checker.config.models, "default_provider", "openai"
            ),
            patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm_class,
            patch("cognivault.config.openai_config.OpenAIConfig") as mock_config_class,
        ):
            # Mock successful OpenAI configuration
            mock_config = MagicMock()
            mock_config.api_key = "test-key"
            mock_config.model = "gpt-4"
            mock_config_class.load.return_value = mock_config

            # Mock successful LLM creation (no exception means success)
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            result = await self.health_checker._check_llm_connectivity()

            assert result.name == "llm_connectivity"
            assert result.status == HealthStatus.HEALTHY
            assert "OpenAI connectivity healthy" in result.message
            assert result.details["provider"] == "openai"
            assert result.details["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_openai_failure(self):
        """Test LLM connectivity check with failed OpenAI connection."""

        with (
            patch.object(
                self.health_checker.config.models, "default_provider", "openai"
            ),
            patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm_class,
        ):
            # Mock failed LLM creation - raise exception when creating LLM
            mock_llm_class.side_effect = Exception("Connection failed")

            result = await self.health_checker._check_llm_connectivity()

            assert result.name == "llm_connectivity"
            assert result.status == HealthStatus.UNHEALTHY
            assert "LLM health check failed" in result.message

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_exception(self):
        """Test LLM connectivity check with exception."""
        with (
            patch.object(
                self.health_checker.config.models, "default_provider", "openai"
            ),
            patch("cognivault.llm.openai.OpenAIChatLLM") as mock_llm_class,
        ):
            # Mock exception during LLM creation
            mock_llm_class.side_effect = Exception("Connection failed")

            result = await self.health_checker._check_llm_connectivity()

            assert result.name == "llm_connectivity"
            assert result.status == HealthStatus.UNHEALTHY
            assert "LLM health check failed" in result.message

    @pytest.mark.asyncio
    async def test_check_file_system_low_disk_space(self):
        """Test file system check with low disk space."""
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.access") as mock_access,
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.statvfs") as mock_statvfs,
        ):
            # Mock basic file system checks as healthy
            mock_exists.return_value = True
            mock_access.return_value = True
            mock_temp.return_value.__enter__.return_value = MagicMock()

            # Mock low disk space (< 100MB)
            mock_stat = MagicMock()
            mock_stat.f_frsize = 4096
            mock_stat.f_bavail = 10000  # ~40MB free
            mock_statvfs.return_value = mock_stat

            result = await self.health_checker._check_file_system()

            assert result.name == "file_system"
            assert result.status == HealthStatus.DEGRADED
            assert "File system has 1 issues" in result.message
            assert "Low disk space (< 100MB)" in result.details["issues"]

    @pytest.mark.asyncio
    async def test_check_file_system_temp_file_creation_failure(self):
        """Test file system check when temp file creation fails."""
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.access") as mock_access,
            patch("tempfile.NamedTemporaryFile") as mock_temp,
        ):
            # Mock directory checks as healthy
            mock_exists.return_value = True
            mock_access.return_value = True

            # Mock temp file creation failure
            mock_temp.side_effect = OSError("Permission denied")

            result = await self.health_checker._check_file_system()

            assert result.name == "file_system"
            assert result.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
            assert "temp_file_creation" in result.details
            assert result.details["temp_file_creation"] is False

    @pytest.mark.asyncio
    async def test_check_dependencies_version_conflict(self):
        """Test dependencies check with version conflicts."""

        with patch("cognivault.diagnostics.health.version") as mock_version:
            # Mock package with old version
            mock_version.return_value = "0.1.0"  # Old version

            result = await self.health_checker._check_dependencies()

            assert result.name == "dependencies"
            # Should still be healthy as long as packages are available
            assert result.status == HealthStatus.HEALTHY
            assert "package_versions" in result.details

    def test_get_overall_status_priority_order(self):
        """Test overall status priority (unhealthy > degraded > healthy)."""
        # Test all combinations to ensure unhealthy takes precedence
        components = {
            "healthy": ComponentHealth(
                "healthy", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
            "degraded": ComponentHealth(
                "degraded", HealthStatus.DEGRADED, "Issues", {}, datetime.now()
            ),
            "unhealthy": ComponentHealth(
                "unhealthy", HealthStatus.UNHEALTHY, "Failed", {}, datetime.now()
            ),
        }

        status = self.health_checker.get_overall_status(components)
        assert status == HealthStatus.UNHEALTHY

        # Test degraded + healthy
        components_degraded = {
            "healthy": ComponentHealth(
                "healthy", HealthStatus.HEALTHY, "OK", {}, datetime.now()
            ),
            "degraded": ComponentHealth(
                "degraded", HealthStatus.DEGRADED, "Issues", {}, datetime.now()
            ),
        }

        status = self.health_checker.get_overall_status(components_degraded)
        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_agent_registry_validation_failure(self):
        """Test agent registry health check when pipeline validation fails."""
        with (
            patch.object(
                self.health_checker.registry, "get_available_agents"
            ) as mock_agents,
            patch.object(
                self.health_checker.registry, "validate_pipeline"
            ) as mock_validate,
        ):
            mock_agents.return_value = ["refiner", "critic"]
            mock_validate.side_effect = Exception("Pipeline validation failed")

            result = await self.health_checker._check_agent_registry()

            assert result.name == "agent_registry"
            assert result.status == HealthStatus.UNHEALTHY
            assert "Registry check failed" in result.message
            assert "Pipeline validation failed" in result.details["error"]

    @pytest.mark.asyncio
    async def test_check_configuration_missing_directories(self):
        """Test configuration check when directories are missing."""
        with (
            patch.object(self.health_checker.config, "validate") as mock_validate,
            patch("os.path.exists") as mock_exists,
        ):
            mock_validate.return_value = []  # No validation errors
            mock_exists.return_value = False  # Directories don't exist

            result = await self.health_checker._check_configuration()

            assert result.name == "configuration"
            assert result.status == HealthStatus.DEGRADED
            assert "Configuration has 2 critical issues" in result.message
            assert "does not exist" in str(result.details["critical_issues"])

    @pytest.mark.asyncio
    async def test_component_health_response_time_tracking(self):
        """Test that component health tracks response times."""
        import time

        start_time = time.time()

        # Mock a quick health check
        with patch.object(
            self.health_checker.registry, "get_available_agents"
        ) as mock_agents:
            mock_agents.return_value = ["refiner"]

            result = await self.health_checker._check_agent_registry()

            # Should have response time recorded
            assert result.response_time_ms is not None
            assert result.response_time_ms >= 0
            assert isinstance(result.response_time_ms, float)
