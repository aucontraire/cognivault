"""
Integration test configuration and fixtures.

This module provides function-scoped database fixtures for integration tests to prevent
AsyncIO event loop conflicts, applying the same successful patterns used in unit tests.
"""

import pytest
import os
from typing import AsyncGenerator
import logging

from cognivault.database import get_database_session, init_database, RepositoryFactory
from cognivault.database.connection import close_database
from sqlalchemy.ext.asyncio import AsyncSession
from tests.utils.test_database_config import get_test_env_vars


@pytest.fixture(scope="function", autouse=True)
async def setup_integration_database() -> AsyncGenerator[None, None]:
    """Setup integration test database configuration and initialize database (function-scoped).

    Function-scoped to prevent AsyncIO event loop conflicts, applying the same successful
    pattern from unit tests. Each integration test gets its own database initialization
    in the current event loop.
    """
    # Store original environment values
    original_values = {}
    # Get centralized integration test environment configuration
    integration_env_vars = get_test_env_vars(
        environment=None
    )  # Auto-detect environment

    # Store original values and set integration test values
    for key, value in integration_env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    # Initialize database with error handling
    try:
        await init_database()
        logging.info("Database initialized successfully for integration test function")
    except Exception as e:
        logging.error(f"Failed to initialize integration test database: {e}")
        # Don't fail test setup - individual tests will handle this
        pass

    yield

    # Clean up database connections for this integration test
    try:
        await close_database()
        logging.debug("Database connections closed for integration test function")
    except Exception as e:
        logging.warning(f"Error during integration database cleanup: {e}")

    # Restore original environment variables
    for key, original_value in original_values.items():
        if original_value is not None:
            os.environ[key] = original_value
        elif key in os.environ:
            del os.environ[key]


@pytest.fixture(scope="function")
async def integration_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for integration testing (function-scoped to prevent event loop conflicts).

    Each integration test gets its own database session created in the current event loop.
    This prevents the "Task got Future attached to a different loop" error that occurs
    when session-scoped fixtures create sessions in one event loop but tests run in another.
    """
    async with get_database_session() as session:
        try:
            yield session
        finally:
            # Session cleanup happens automatically via context manager
            pass


@pytest.fixture(scope="function")
async def integration_repository_factory(
    integration_db_session: AsyncSession,
) -> RepositoryFactory:
    """Get repository factory with integration test database session (function-scoped).

    Function-scoped to match integration_db_session fixture and prevent event loop conflicts.
    Each integration test gets its own repository factory instance.
    """
    return RepositoryFactory(integration_db_session)
