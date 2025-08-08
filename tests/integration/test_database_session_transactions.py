"""
Debug script to test session creation and transaction handling.
"""

import pytest
import asyncio
import os
import logging
from sqlalchemy import text

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_session_creation() -> bool:
    """Test session creation directly."""
    # Set environment
    os.environ["DATABASE_URL"] = (
        "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5432/cognivault"
    )
    os.environ["TESTING"] = "true"
    os.environ["DB_SSL_REQUIRE"] = "false"

    from cognivault.database.connection import get_database_session, init_database

    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database initialized")

        # Test simple session
        async with get_database_session() as session:
            result = await session.execute(text("SELECT 1 as test"))
            value = result.scalar()
            logger.info(f"✅ Simple session test: {value}")

        # Test nested transaction (savepoint)
        async with get_database_session() as session:
            savepoint = await session.begin_nested()
            try:
                result = await session.execute(text("SELECT 2 as test"))
                value = result.scalar()
                logger.info(f"✅ Savepoint session test: {value}")
            finally:
                await savepoint.rollback()

        logger.info("🎉 All session tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Session test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_session_creation())
    if success:
        print("✅ Session tests passed")
    else:
        print("❌ Session tests failed")
