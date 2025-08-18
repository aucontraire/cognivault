"""
Centralized async test utilities.

Provides properly typed async context managers and utilities for testing,
eliminating code duplication and ensuring type safety.
"""

from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from types import TracebackType


class AsyncSessionWrapper:
    """Test utility for wrapping database sessions in async context manager.

    Provides a clean async context manager interface for database sessions
    in test scenarios, ensuring proper type annotations and consistent behavior.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize wrapper with database session.

        Args:
            session: The AsyncSession to wrap
        """
        self.session = session

    async def __aenter__(self) -> AsyncSession:
        """Enter async context manager.

        Returns:
            The wrapped AsyncSession instance
        """
        return self.session

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit async context manager.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        pass


def create_mock_session_factory(session: AsyncSession) -> Any:
    """Create a properly typed mock session factory.

    Args:
        session: The AsyncSession to return from the factory

    Returns:
        A callable that returns an AsyncSessionWrapper
    """

    def mock_session_factory() -> AsyncSessionWrapper:
        return AsyncSessionWrapper(session)

    return mock_session_factory
