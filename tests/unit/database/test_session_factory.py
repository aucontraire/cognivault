"""
Unit tests for database session factory.

Tests the DatabaseSessionFactory class and session management functionality
with proper lifecycle management and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from cognivault.database.session_factory import (
    DatabaseSessionFactory,
    get_database_session_factory,
    initialize_database_session_factory,
    shutdown_database_session_factory,
)


class TestDatabaseSessionFactory:
    """Test suite for DatabaseSessionFactory class."""

    @pytest.fixture
    def session_factory(self):
        """Create a fresh session factory for testing."""
        return DatabaseSessionFactory()

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        return session

    @pytest.fixture
    def mock_session_factory_func(self, mock_session):
        """Create a mock session factory function."""
        factory_func = MagicMock()
        factory_func.return_value = mock_session
        return factory_func

    def test_session_factory_initial_state(self, session_factory):
        """Test initial state of session factory."""
        assert not session_factory.is_initialized
        assert session_factory._session_factory is None

    @patch("cognivault.database.session_factory.get_database_engine")
    @patch("cognivault.database.session_factory.get_session_factory")
    async def test_initialize_success(
        self,
        mock_get_session_factory,
        mock_get_engine,
        session_factory,
        mock_session_factory_func,
        mock_session,
    ):
        """Test successful initialization of session factory."""
        # Setup mocks
        mock_get_engine.return_value = MagicMock()
        mock_get_session_factory.return_value = mock_session_factory_func

        # Mock successful test query
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result

        # Mock the async context manager for session creation
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Initialize
        await session_factory.initialize()

        assert session_factory.is_initialized
        assert session_factory._session_factory is not None
        mock_get_engine.assert_called_once()
        mock_get_session_factory.assert_called_once()
        # The execute call happens during the test query in initialize
        mock_session.execute.assert_called_once()

    @patch("cognivault.database.session_factory.get_database_engine")
    async def test_initialize_failure(self, mock_get_engine, session_factory):
        """Test initialization failure handling."""
        # Mock engine creation failure
        mock_get_engine.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            await session_factory.initialize()

        assert not session_factory.is_initialized
        assert session_factory._session_factory is None

    async def test_initialize_already_initialized(self, session_factory):
        """Test that initialize is idempotent."""
        # Mark as already initialized
        session_factory._is_initialized = True

        await session_factory.initialize()

        # Should remain initialized without changes
        assert session_factory.is_initialized

    @patch("cognivault.database.connection.close_database")
    async def test_shutdown(self, mock_close_db, session_factory):
        """Test shutdown functionality."""
        # Mark as initialized
        session_factory._is_initialized = True

        await session_factory.shutdown()

        assert not session_factory.is_initialized
        mock_close_db.assert_called_once()

    async def test_shutdown_not_initialized(self, session_factory):
        """Test shutdown when not initialized."""
        # Should not raise error
        await session_factory.shutdown()
        assert not session_factory.is_initialized

    async def test_get_session_not_initialized(self, session_factory):
        """Test getting session when not initialized raises error."""
        with pytest.raises(RuntimeError, match="Session factory not initialized"):
            async with session_factory.get_session():
                pass

    async def test_get_session_success(
        self, session_factory, mock_session_factory_func, mock_session
    ):
        """Test successful session creation and cleanup."""
        session_factory._is_initialized = True
        session_factory._session_factory = mock_session_factory_func

        async with session_factory.get_session() as session:
            assert session is mock_session

        # Verify session lifecycle
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    async def test_get_session_with_exception(
        self, session_factory, mock_session_factory_func, mock_session
    ):
        """Test session rollback on exception."""
        session_factory._is_initialized = True
        session_factory._session_factory = mock_session_factory_func

        with pytest.raises(ValueError, match="Test error"):
            async with session_factory.get_session():
                raise ValueError("Test error")

        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()

    @patch("cognivault.database.session_factory.RepositoryFactory")
    async def test_get_repository_factory(
        self,
        mock_repo_factory_class,
        session_factory,
        mock_session_factory_func,
        mock_session,
    ):
        """Test getting repository factory with managed session."""
        session_factory._is_initialized = True
        session_factory._session_factory = mock_session_factory_func

        mock_repo_factory = MagicMock()
        mock_repo_factory_class.return_value = mock_repo_factory

        async with session_factory.get_repository_factory() as repo_factory:
            assert repo_factory is mock_repo_factory

        # Verify repository factory was created with session
        mock_repo_factory_class.assert_called_once_with(mock_session)
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    async def test_execute_with_session(
        self, session_factory, mock_session_factory_func, mock_session
    ):
        """Test executing operation with managed session."""
        session_factory._is_initialized = True
        session_factory._session_factory = mock_session_factory_func

        async def test_operation(session, arg1, arg2):
            assert session is mock_session
            assert arg1 == "test_arg"
            assert arg2 == "test_kwarg"
            return "operation_result"

        result = await session_factory.execute_with_session(
            test_operation, "test_arg", arg2="test_kwarg"
        )

        assert result == "operation_result"
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch("cognivault.database.session_factory.RepositoryFactory")
    async def test_execute_with_repositories(
        self,
        mock_repo_factory_class,
        session_factory,
        mock_session_factory_func,
        mock_session,
    ):
        """Test executing operation with managed repository factory."""
        session_factory._is_initialized = True
        session_factory._session_factory = mock_session_factory_func

        mock_repo_factory = MagicMock()
        mock_repo_factory_class.return_value = mock_repo_factory

        async def test_operation(repo_factory, arg1):
            assert repo_factory is mock_repo_factory
            assert arg1 == "test_arg"
            return "repo_operation_result"

        result = await session_factory.execute_with_repositories(
            test_operation, "test_arg"
        )

        assert result == "repo_operation_result"
        mock_repo_factory_class.assert_called_once_with(mock_session)

    async def test_health_check_not_initialized(self, session_factory):
        """Test health check when not initialized."""
        health = await session_factory.health_check()

        assert health["status"] == "unhealthy"
        assert "not initialized" in health["error"]

    async def test_health_check_success(
        self, session_factory, mock_session_factory_func, mock_session
    ):
        """Test successful health check."""
        session_factory._is_initialized = True
        session_factory._session_factory = mock_session_factory_func

        # Mock successful query
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        health = await session_factory.health_check()

        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert "response_time_ms" in health
        # Verify execute was called (with TextClause object)
        mock_session.execute.assert_called_once()

    async def test_health_check_failure(
        self, session_factory, mock_session_factory_func, mock_session
    ):
        """Test health check failure."""
        session_factory._is_initialized = True
        session_factory._session_factory = mock_session_factory_func

        # Mock query failure
        mock_session.execute.side_effect = Exception("Database error")

        health = await session_factory.health_check()

        assert health["status"] == "unhealthy"
        assert "Database error" in health["error"]
        assert health["initialized"] is True


class TestGlobalSessionFactory:
    """Test suite for global session factory management."""

    def teardown_method(self):
        """Reset global session factory after each test."""
        # Reset global state
        from cognivault.database import session_factory

        session_factory._session_factory = None

    def test_get_database_session_factory_singleton(self):
        """Test that global session factory is a singleton."""
        factory1 = get_database_session_factory()
        factory2 = get_database_session_factory()

        assert factory1 is factory2
        assert isinstance(factory1, DatabaseSessionFactory)

    @patch("cognivault.database.session_factory.get_database_session_factory")
    async def test_initialize_database_session_factory(self, mock_get_factory):
        """Test global session factory initialization."""
        mock_factory = AsyncMock()
        mock_get_factory.return_value = mock_factory

        await initialize_database_session_factory()

        mock_get_factory.assert_called_once()
        mock_factory.initialize.assert_called_once()

    @patch("cognivault.database.session_factory.get_database_session_factory")
    async def test_shutdown_database_session_factory(self, mock_get_factory):
        """Test global session factory shutdown."""
        mock_factory = AsyncMock()

        # Setup global factory
        from cognivault.database import session_factory

        session_factory._session_factory = mock_factory

        await shutdown_database_session_factory()

        mock_factory.shutdown.assert_called_once()
        # Global factory should be reset to None
        assert session_factory._session_factory is None

    async def test_shutdown_database_session_factory_none(self):
        """Test shutdown when global factory is None."""
        # Should not raise error
        await shutdown_database_session_factory()


class TestSessionFactoryIntegration:
    """Integration tests for session factory with repositories."""

    @pytest.fixture
    async def initialized_session_factory(self):
        """Create and initialize a session factory for integration tests."""
        factory = DatabaseSessionFactory()

        # Mock initialization
        with (
            patch("cognivault.database.session_factory.get_database_engine"),
            patch(
                "cognivault.database.session_factory.get_session_factory"
            ) as mock_get_factory,
        ):
            mock_session = AsyncMock()
            mock_session.execute.return_value = MagicMock()
            mock_session_factory_func = MagicMock(return_value=mock_session)
            mock_get_factory.return_value = mock_session_factory_func

            await factory.initialize()

            yield factory, mock_session

        await factory.shutdown()

    @patch("cognivault.database.session_factory.RepositoryFactory")
    async def test_repository_operations_integration(
        self, mock_repo_factory_class, initialized_session_factory
    ):
        """Test integration between session factory and repositories."""
        factory, mock_session = initialized_session_factory

        # Mock repository factory
        mock_repo_factory = MagicMock()
        mock_repo_factory.topics.create_topic = AsyncMock(return_value="created_topic")
        mock_repo_factory_class.return_value = mock_repo_factory

        async def create_topic_operation(repo_factory, name):
            return await repo_factory.topics.create_topic(name=name)

        result = await factory.execute_with_repositories(
            create_topic_operation, "Test Topic"
        )

        assert result == "created_topic"
        mock_repo_factory.topics.create_topic.assert_called_once_with(name="Test Topic")

    async def test_multiple_concurrent_sessions(self, initialized_session_factory):
        """Test that multiple concurrent sessions work correctly."""
        factory, mock_session = initialized_session_factory

        async def concurrent_operation(session_num):
            async with factory.get_session() as session:
                return f"session_{session_num}_result"

        # Run multiple concurrent operations
        tasks = [concurrent_operation(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(f"session_{i}_result" in results for i in range(5))

        # Verify session was created and cleaned up for each operation
        assert mock_session.commit.call_count == 5
        assert mock_session.close.call_count == 5
