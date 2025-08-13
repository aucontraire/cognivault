"""
Debug script to test session creation and transaction handling.
"""

import pytest
import asyncio
import os
import logging
from sqlalchemy import text

from tests.utils.test_database_config import get_test_env_vars

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_session_creation() -> bool:
    """Test session creation directly."""
    # Set environment using centralized test configuration
    test_env_vars = get_test_env_vars("local")
    for key, value in test_env_vars.items():
        os.environ[key] = value

    from cognivault.database.connection import get_database_session, init_database

    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database initialized")

        # Test simple session
        async with get_database_session() as session:
            session_result1 = await session.execute(text("SELECT 1 as test"))
            scalar_session1_value = session_result1.scalar()
            logger.info(f"✅ Simple session test: {scalar_session1_value}")

        # Test nested transaction (savepoint)
        async with get_database_session() as session:
            savepoint = await session.begin_nested()
            try:
                session_result2 = await session.execute(text("SELECT 2 as test"))
                scalar_session2_value = session_result2.scalar()
                logger.info(f"✅ Savepoint session test: {scalar_session2_value}")
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
