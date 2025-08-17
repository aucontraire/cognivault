#!/usr/bin/env python3
"""
Debug script to test database connection and identify issues.
"""

import asyncio
import os
import logging
from sqlalchemy import text

from tests.utils.test_database_config import get_test_database_url, get_test_env_vars

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_local_connection() -> bool:
    """Test connection to local PostgreSQL."""
    from cognivault.database.connection import get_database_engine, init_database

    # Use centralized test database configuration
    test_env_vars = get_test_env_vars("local")
    for key, value in test_env_vars.items():
        os.environ[key] = value

    try:
        logger.info("Testing local PostgreSQL connection...")

        # Test database initialization
        await init_database()
        logger.info("✅ Database initialization successful")

        # Test engine creation
        engine = get_database_engine()
        logger.info("✅ Database engine created")

        # Test basic query
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 as test"))
            scalar_value = result.scalar()
            assert scalar_value is not None
            if scalar_value == 1:
                logger.info("✅ Basic query test passed")
            else:
                logger.error(f"❌ Basic query returned {scalar_value}, expected 1")

        # Test session creation
        from cognivault.database.connection import get_database_session

        async with get_database_session() as session:
            result_session = await session.execute(text("SELECT 2 as test"))
            scalar_session_value = result_session.scalar()
            assert scalar_session_value is not None
            if scalar_session_value == 2:
                logger.info("✅ Session test passed")
            else:
                logger.error(
                    f"❌ Session query returned {scalar_session_value}, expected 2"
                )

        logger.info("🎉 All connection tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False


async def test_container_connection() -> bool:
    """Test connection to Docker container PostgreSQL."""
    from cognivault.database.connection import (
        get_database_engine,
        init_database,
        close_database,
    )

    # Reset global state
    await close_database()

    # Use Docker container PostgreSQL
    os.environ["DATABASE_URL"] = get_test_database_url("docker")
    os.environ["TESTING"] = "true"
    os.environ["DB_SSL_REQUIRE"] = "false"
    os.environ["DB_POOL_SIZE"] = "5"

    try:
        logger.info("Testing Docker container PostgreSQL connection...")

        # Test database initialization with timeout
        await asyncio.wait_for(init_database(), timeout=30)
        logger.info("✅ Container database initialization successful")

        # Test engine creation
        engine = get_database_engine()
        logger.info("✅ Container database engine created")

        # Test basic query
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 as test"))
            value = result.scalar()
            if value == 1:
                logger.info("✅ Container basic query test passed")
            else:
                logger.error(f"❌ Container basic query returned {value}, expected 1")

        logger.info("🎉 Container connection tests passed!")
        return True

    except asyncio.TimeoutError:
        logger.error("❌ Container connection timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Container connection test failed: {e}")
        return False


async def main() -> None:
    """Run all connection tests."""
    logger.info("🔧 Starting database connection tests...")

    # Test 1: Local PostgreSQL
    local_success = await test_local_connection()

    logger.info("-" * 50)

    # Test 2: Container PostgreSQL
    container_success = await test_container_connection()

    # Summary
    logger.info("=" * 50)
    logger.info("📊 Test Results:")
    logger.info(f"Local PostgreSQL: {'✅ PASS' if local_success else '❌ FAIL'}")
    logger.info(
        f"Container PostgreSQL: {'✅ PASS' if container_success else '❌ FAIL'}"
    )

    if local_success or container_success:
        logger.info("🎉 At least one database connection is working!")
    else:
        logger.error("💥 All database connections failed!")


if __name__ == "__main__":
    asyncio.run(main())
