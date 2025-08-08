"""
Database session factory for centralized session management.

Provides a centralized session factory pattern for database operations
with proper lifecycle management and resource cleanup.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import (
    Any,
    TypeVar,
)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from cognivault.observability import get_logger

from .connection import get_database_engine, get_session_factory
from .repositories import RepositoryFactory

logger = get_logger(__name__)

T = TypeVar("T")


class DatabaseSessionFactory:
    """
    Centralized database session factory for repository management.

    Provides session lifecycle management, repository factory integration,
    and resource cleanup for database operations.
    """

    def __init__(self) -> None:
        self._session_factory = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Initialize the session factory and database connection.

        This should be called once at application startup.
        """
        if self._is_initialized:
            logger.debug("Session factory already initialized")
            return

        try:
            # Initialize database engine and validate connection
            get_database_engine()  # Validate engine creation
            self._session_factory = get_session_factory()

            # Test the connection
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))

            self._is_initialized = True
            logger.info("Database session factory initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize session factory: {e}")
            raise

    async def shutdown(self) -> None:
        """
        Shutdown the session factory and cleanup resources.

        This should be called once at application shutdown.
        """
        if not self._is_initialized:
            return

        try:
            from .connection import close_database

            await close_database()
            self._is_initialized = False
            logger.info("Database session factory shutdown completed")

        except Exception as e:
            logger.error(f"Error during session factory shutdown: {e}")
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic cleanup.

        Yields:
            AsyncSession: Database session with automatic transaction management

        Raises:
            RuntimeError: If session factory is not initialized
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Session factory not initialized. Call initialize() first."
            )

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def get_repository_factory(self) -> AsyncGenerator[RepositoryFactory, None]:
        """
        Get a repository factory with managed session.

        Yields:
            RepositoryFactory: Repository factory with managed database session

        Raises:
            RuntimeError: If session factory is not initialized
        """
        async with self.get_session() as session:
            factory = RepositoryFactory(session)
            yield factory

    async def execute_with_session(
        self, operation: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute an operation with a managed database session.

        Args:
            operation: Async callable that takes a session as first argument
            *args: Additional positional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If session factory is not initialized
        """
        async with self.get_session() as session:
            return await operation(session, *args, **kwargs)

    async def execute_with_repositories(
        self, operation: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute an operation with managed repositories.

        Args:
            operation: Async callable that takes a RepositoryFactory as first argument
            *args: Additional positional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If session factory is not initialized
        """
        async with self.get_repository_factory() as repo_factory:
            return await operation(repo_factory, *args, **kwargs)

    @property
    def is_initialized(self) -> bool:
        """Check if the session factory is initialized."""
        return self._is_initialized

    async def health_check(self) -> dict:
        """
        Perform a health check on the session factory.

        Returns:
            Dictionary with health status information
        """
        if not self._is_initialized:
            return {"status": "unhealthy", "error": "Session factory not initialized"}

        try:
            # Test session creation and basic query
            start_time = asyncio.get_event_loop().time()

            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1 as test"))
                assert result.scalar() == 1

            response_time = asyncio.get_event_loop().time() - start_time

            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "initialized": True,
            }

        except Exception as e:
            logger.error(f"Session factory health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._is_initialized,
            }


# Global session factory instance
_session_factory: DatabaseSessionFactory | None = None


def get_database_session_factory() -> DatabaseSessionFactory:
    """
    Get the global database session factory instance.

    Returns:
        DatabaseSessionFactory: Global session factory instance
    """
    global _session_factory

    if _session_factory is None:
        _session_factory = DatabaseSessionFactory()

    return _session_factory


async def initialize_database_session_factory() -> None:
    """Initialize the global database session factory."""
    factory = get_database_session_factory()
    await factory.initialize()


async def shutdown_database_session_factory() -> None:
    """Shutdown the global database session factory."""
    global _session_factory

    if _session_factory:
        await _session_factory.shutdown()
        _session_factory = None
