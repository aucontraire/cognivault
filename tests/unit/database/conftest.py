"""
Database test configuration and fixtures.
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator

from cognivault.database import get_database_session, init_database, RepositoryFactory


# Test database configuration
TEST_DATABASE_URL = (
    "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5435/cognivault"
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_database():
    """Setup test database configuration and initialize database."""
    # Set test database URL
    original_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL

    # Initialize database
    await init_database()

    yield

    # Restore original database URL if it existed
    if original_db_url:
        os.environ["DATABASE_URL"] = original_db_url
    elif "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]


@pytest.fixture
async def db_session():
    """Get database session for testing."""
    async with get_database_session() as session:
        yield session


@pytest.fixture
async def repository_factory(db_session) -> RepositoryFactory:
    """Get repository factory with test database session."""
    return RepositoryFactory(db_session)
