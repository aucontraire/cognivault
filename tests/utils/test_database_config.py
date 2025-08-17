"""
Test Database Configuration Management for CogniVault Testing.

This module provides centralized database configuration management for all test environments,
following the same patterns as the main application config but with test-specific defaults
and environment isolation.

This centralizes the previously scattered database URLs and provides consistent configuration
across unit tests, integration tests, and infrastructure tests.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from cognivault.database.config import DatabaseConfig
from cognivault.observability import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TestDatabaseEnvironment:
    """Immutable test database environment configuration."""

    # Default test database URLs for different environments
    LOCAL_TEST_URL = (
        "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5432/cognivault"
    )
    DOCKER_TEST_URL = "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5435/cognivault"  # Docker container port
    CI_TEST_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"

    # Environment-specific configurations
    ENVIRONMENTS = {
        "local": {
            "database_url": LOCAL_TEST_URL,
            "pool_size": 5,
            "max_overflow": 10,
            "ssl_require": False,
        },
        "docker": {
            "database_url": DOCKER_TEST_URL,
            "pool_size": 3,
            "max_overflow": 5,
            "ssl_require": False,
        },
        "ci": {
            "database_url": CI_TEST_URL,
            "pool_size": 2,
            "max_overflow": 3,
            "ssl_require": False,
        },
    }


class TestDatabaseConfigFactory:
    """
    Factory for creating test database configurations with environment-specific defaults.

    Provides centralized configuration management for all test environments while maintaining
    compatibility with existing test patterns and fixtures.
    """

    @classmethod
    def get_test_database_url(cls, environment: Optional[str] = None) -> str:
        """
        Get database URL for specified test environment.

        Args:
            environment: Test environment ('local', 'docker', 'ci') or None for auto-detection

        Returns:
            Database URL string configured for the specified environment
        """
        if environment is None:
            environment = cls._detect_test_environment()

        env_config = TestDatabaseEnvironment.ENVIRONMENTS.get(environment)
        if not env_config:
            logger.warning(
                f"Unknown test environment '{environment}', using local default"
            )
            return TestDatabaseEnvironment.LOCAL_TEST_URL

        # Check for environment variable override
        env_var_url = os.getenv("TEST_DATABASE_URL")
        if env_var_url:
            logger.info(
                f"Using TEST_DATABASE_URL override: {cls._mask_credentials(env_var_url)}"
            )
            return env_var_url

        database_url = env_config["database_url"]
        return str(database_url)  # Ensure string type for mypy

    @classmethod
    def create_test_database_config(
        cls, environment: Optional[str] = None, **overrides: Any
    ) -> DatabaseConfig:
        """
        Create DatabaseConfig instance optimized for testing.

        Args:
            environment: Test environment ('local', 'docker', 'ci') or None for auto-detection
            **overrides: Additional configuration overrides

        Returns:
            DatabaseConfig instance configured for testing
        """
        if environment is None:
            environment = cls._detect_test_environment()

        env_config = TestDatabaseEnvironment.ENVIRONMENTS.get(
            environment, TestDatabaseEnvironment.ENVIRONMENTS["local"]
        )

        # Test-specific defaults
        test_defaults = {
            "database_url": cls.get_test_database_url(environment),
            "echo_sql": os.getenv("TEST_DB_ECHO_SQL", "false").lower() == "true",
            "pool_size": int(
                os.getenv("TEST_DB_POOL_SIZE", str(env_config["pool_size"]))
            ),
            "max_overflow": int(
                os.getenv("TEST_DB_MAX_OVERFLOW", str(env_config["max_overflow"]))
            ),
            "pool_timeout": int(
                os.getenv("TEST_DB_POOL_TIMEOUT", "10")
            ),  # Shorter for tests
            "pool_recycle": int(
                os.getenv("TEST_DB_POOL_RECYCLE", "1800")
            ),  # 30 min for tests
            "pool_pre_ping": os.getenv("TEST_DB_POOL_PRE_PING", "true").lower()
            == "true",
            "connection_timeout": int(os.getenv("TEST_DB_CONNECTION_TIMEOUT", "30")),
            "command_timeout": int(os.getenv("TEST_DB_COMMAND_TIMEOUT", "60")),
            "ssl_require": os.getenv("TEST_DB_SSL_REQUIRE", "false").lower() == "true",
            "vector_dimensions": int(os.getenv("TEST_VECTOR_DIMENSIONS", "1536")),
        }

        # Apply overrides
        config_params = {**test_defaults, **overrides}

        return DatabaseConfig(**config_params)

    @classmethod
    def create_test_environment_variables(
        cls, environment: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create environment variables dict for test database configuration.

        Useful for test fixtures that need to set environment variables.

        Args:
            environment: Test environment or None for auto-detection

        Returns:
            Dictionary of environment variables for database configuration
        """
        config = cls.create_test_database_config(environment)

        return {
            "DATABASE_URL": config.database_url,
            "TESTING": "true",
            "DB_POOL_SIZE": str(config.pool_size),
            "DB_MAX_OVERFLOW": str(config.max_overflow),
            "DB_POOL_TIMEOUT": str(config.pool_timeout),
            "DB_CONNECTION_TIMEOUT": str(config.connection_timeout),
            "DB_COMMAND_TIMEOUT": str(config.command_timeout),
            "DB_SSL_REQUIRE": str(config.ssl_require).lower(),
            "DB_ECHO_SQL": str(config.echo_sql).lower(),
        }

    @classmethod
    def _detect_test_environment(cls) -> str:
        """
        Auto-detect the current test environment.

        Returns:
            Environment name ('local', 'docker', 'ci')
        """
        # Check for CI environment
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            return "ci"

        # Check if Docker database is available (port 5435)
        docker_url = TestDatabaseEnvironment.DOCKER_TEST_URL
        if cls._is_database_available(docker_url):
            return "docker"

        # Default to local
        return "local"

    @classmethod
    def _is_database_available(cls, database_url: str) -> bool:
        """
        Check if database at URL is available (basic connectivity check).

        Args:
            database_url: Database URL to check

        Returns:
            True if database appears to be available
        """
        try:
            parsed = urlparse(database_url)
            import socket

            host = parsed.hostname or "localhost"
            port = parsed.port or 5432

            # Quick socket check
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.error, socket.timeout):
            return False

    @classmethod
    def _mask_credentials(cls, url: str) -> str:
        """Mask credentials in URL for safe logging."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                return url.replace(parsed.password, "***")
        except Exception:
            pass
        return url


# Convenience functions for backward compatibility and ease of use
def get_test_database_url(environment: Optional[str] = None) -> str:
    """Get test database URL for specified environment."""
    return TestDatabaseConfigFactory.get_test_database_url(environment)


def create_test_database_config(
    environment: Optional[str] = None, **overrides: Any
) -> DatabaseConfig:
    """Create test database configuration."""
    return TestDatabaseConfigFactory.create_test_database_config(
        environment, **overrides
    )


def get_test_env_vars(environment: Optional[str] = None) -> Dict[str, str]:
    """Get test environment variables for database configuration."""
    return TestDatabaseConfigFactory.create_test_environment_variables(environment)


# Legacy constants for backward compatibility (deprecated - use factory methods)
TEST_DATABASE_URL = TestDatabaseConfigFactory.get_test_database_url("local")
INTEGRATION_DATABASE_URL = TestDatabaseConfigFactory.get_test_database_url("local")
DOCKER_DATABASE_URL = TestDatabaseConfigFactory.get_test_database_url("docker")
