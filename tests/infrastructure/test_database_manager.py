"""
Database test infrastructure manager for handling container lifecycle and connection management.

This module provides utilities for:
1. Starting/stopping test database containers
2. Managing database connection health
3. Providing container-aware test fixtures
4. Handling connection failures gracefully
"""

import asyncio
import logging
import os
import subprocess
import time
from typing import Any, Optional

import pytest

from tests.utils.test_database_config import get_test_database_url

logger = logging.getLogger(__name__)


class DatabaseTestManager:
    """Manages database container lifecycle and connection health for tests."""

    def __init__(self) -> None:
        self.container_running = False
        self.local_db_available = False
        self.preferred_db_url: Optional[str] = None

    def check_local_database(self) -> bool:
        """Check if local PostgreSQL is running and accessible."""
        try:
            result = subprocess.run(
                [
                    "psql",
                    "-h",
                    "localhost",
                    "-p",
                    "5432",
                    "-U",
                    "cognivault",
                    "-d",
                    "cognivault",
                    "-c",
                    "SELECT 1;",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                env={**os.environ, "PGPASSWORD": "cognivault_dev"},
            )

            if result.returncode == 0:
                logger.info("Local PostgreSQL database is accessible")
                self.local_db_available = True
                self.preferred_db_url = get_test_database_url("local")
                return True
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            logger.warning(f"Local PostgreSQL not accessible: {e}")

        self.local_db_available = False
        return False

    def check_container_database(self) -> bool:
        """Check if test database container is running and accessible."""
        try:
            # Check if Docker is available
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=postgres",
                    "--format",
                    "table {{.Names}}\\t{{.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if (
                result.returncode == 0
                and "postgres" in result.stdout
                and "Up" in result.stdout
            ):
                logger.info("Test database container is running")
                self.container_running = True
                self.preferred_db_url = get_test_database_url("docker")
                return True
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            logger.warning(f"Container database check failed: {e}")

        self.container_running = False
        return False

    def start_container_database(self) -> bool:
        """Start test database container if not running."""
        try:
            logger.info("Starting test database container...")
            result = subprocess.run(
                ["make", "db-test-start"],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes timeout
                cwd=os.path.dirname(os.path.dirname(__file__)),  # Project root
            )

            if result.returncode == 0:
                logger.info("Test database container started successfully")
                self.container_running = True
                self.preferred_db_url = get_test_database_url("docker")
                return True
            else:
                logger.error(f"Failed to start container: {result.stderr}")
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            logger.error(f"Container startup failed: {e}")

        return False

    def stop_container_database(self) -> bool:
        """Stop test database container."""
        try:
            logger.info("Stopping test database container...")
            result = subprocess.run(
                ["make", "db-test-stop"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(os.path.dirname(__file__)),  # Project root
            )

            if result.returncode == 0:
                logger.info("Test database container stopped successfully")
                self.container_running = False
                return True
            else:
                logger.warning(f"Container stop had issues: {result.stderr}")
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            logger.warning(f"Container stop failed: {e}")

        return False

    def get_best_database_url(self) -> str:
        """Get the best available database URL for testing."""
        # Strategy: Try local first (faster), then container
        if self.check_local_database():
            assert self.preferred_db_url is not None
            return self.preferred_db_url

        if self.check_container_database() or self.start_container_database():
            assert self.preferred_db_url is not None
            return self.preferred_db_url

        # Fallback to local even if check failed (maybe it will work)
        logger.warning("No database available, falling back to local PostgreSQL")
        return get_test_database_url("local")

    def ensure_database_available(self) -> tuple[str, bool]:
        """
        Ensure a database is available for testing.

        Returns:
            Tuple of (database_url, is_ready)
        """
        database_url = self.get_best_database_url()

        # Give database a moment to be fully ready
        if self.container_running:
            logger.info("Waiting for container database to be fully ready...")
            time.sleep(5)

        return database_url, True

    def cleanup(self) -> None:
        """Clean up test database resources."""
        if (
            self.container_running
            and os.environ.get("KEEP_TEST_CONTAINER", "false").lower() != "true"
        ):
            self.stop_container_database()


# Global instance
_db_manager = DatabaseTestManager()


def get_db_manager() -> DatabaseTestManager:
    """Get the global database test manager instance."""
    return _db_manager


def pytest_configure(config: Any) -> None:
    """Configure pytest with database test support."""
    config.addinivalue_line("markers", "database: mark test as requiring database")


def pytest_runtest_setup(item: Any) -> None:
    """Set up database for tests that need it."""
    if item.get_closest_marker("database"):
        db_manager = get_db_manager()
        database_url, is_ready = db_manager.ensure_database_available()

        if not is_ready:
            pytest.skip("Database not available for testing")

        # Set environment variable for the test
        os.environ["DATABASE_URL"] = database_url
        os.environ["TESTING"] = "true"


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Clean up database resources after test session."""
    db_manager = get_db_manager()
    db_manager.cleanup()


@pytest.fixture(scope="session")
def database_test_manager() -> DatabaseTestManager:
    """Provide database test manager as a fixture."""
    return get_db_manager()


@pytest.fixture(scope="session", autouse=True)
def setup_database_environment() -> Any:
    """Automatically set up database environment for test session."""
    db_manager = get_db_manager()
    database_url, is_ready = db_manager.ensure_database_available()

    if not is_ready:
        pytest.skip("No database available for testing")

    # Set environment variables
    original_vars = {}
    test_vars = {
        "DATABASE_URL": database_url,
        "TESTING": "true",
        "DB_POOL_SIZE": "5",
        "DB_MAX_OVERFLOW": "10",
        "DB_POOL_TIMEOUT": "30",
        "DB_CONNECTION_TIMEOUT": "30",
        "DB_COMMAND_TIMEOUT": "60",
        "DB_SSL_REQUIRE": "false",
    }

    # Backup and set test environment
    for key, value in test_vars.items():
        original_vars[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original environment
    for key, original_value in original_vars.items():
        if original_value is not None:
            os.environ[key] = original_value
        elif key in os.environ:
            del os.environ[key]
