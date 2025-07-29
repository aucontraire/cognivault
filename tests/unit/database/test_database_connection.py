"""
Unit tests for database connection management.

Tests database engine creation, connection pooling, health checks,
and schema validation functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from cognivault.database.connection import (
    get_database_engine,
    get_session_factory,
    get_database_session,
    get_connection_pool_status,
    health_check,
    validate_database_schema,
    init_database,
    close_database,
)


class TestDatabaseEngine:
    """Test suite for database engine management."""

    def setup_method(self):
        """Reset global engine state before each test."""
        from cognivault.database import connection

        connection._database_engine = None
        connection._session_factory = None

    def teardown_method(self):
        """Reset global engine state after each test."""
        from cognivault.database import connection

        connection._database_engine = None
        connection._session_factory = None

    @patch("cognivault.database.connection.get_database_config")
    @patch("cognivault.database.connection.create_async_engine")
    @patch("cognivault.database.connection.async_sessionmaker")
    def test_get_database_engine_creation(
        self, mock_sessionmaker, mock_create_engine, mock_get_config
    ):
        """Test database engine creation with proper configuration."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.validate.return_value = None
        mock_config.get_engine_kwargs.return_value = {
            "echo": False,
            "pool_size": 20,
            "future": True,
            "connect_args": {"command_timeout": 60},
        }
        mock_config.get_connection_info.return_value = {
            "hostname": "testhost",
            "port": 5432,
            "database": "testdb",
        }
        mock_config.pool_size = 20
        mock_config.max_overflow = 30
        mock_get_config.return_value = mock_config

        # Mock engine and session factory
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_sessionmaker.return_value = mock_session_factory

        # Test engine creation
        engine = get_database_engine()

        assert engine is mock_engine
        mock_config.validate.assert_called_once()
        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()

    @patch("cognivault.database.connection.get_database_config")
    @patch("cognivault.database.connection.create_async_engine")
    def test_get_database_engine_singleton(self, mock_create_engine, mock_get_config):
        """Test that database engine is a singleton."""
        mock_config = MagicMock()
        mock_config.validate.return_value = None
        mock_config.get_engine_kwargs.return_value = {"echo": False}
        mock_config.get_connection_info.return_value = {
            "hostname": "test",
            "port": 5432,
            "database": "testdb",
        }
        mock_config.pool_size = 20
        mock_config.max_overflow = 30
        mock_get_config.return_value = mock_config

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine1 = get_database_engine()
        engine2 = get_database_engine()

        assert engine1 is engine2
        # Should only create engine once
        assert mock_create_engine.call_count == 1

    @patch("cognivault.database.connection.get_database_engine")
    def test_get_session_factory(self, mock_get_engine):
        """Test getting session factory."""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        # Mock that session factory was created during engine creation
        from cognivault.database import connection

        mock_session_factory = MagicMock()
        connection._session_factory = mock_session_factory

        session_factory = get_session_factory()

        assert session_factory is mock_session_factory


class TestDatabaseSession:
    """Test suite for database session management."""

    @patch("cognivault.database.connection.get_session_factory")
    async def test_get_database_session_success(self, mock_get_factory):
        """Test successful database session creation and cleanup."""
        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_factory.return_value = mock_session_factory

        async with get_database_session() as session:
            assert session is mock_session

        # Verify session factory was called
        mock_get_factory.assert_called_once()

    @patch("cognivault.database.connection.get_session_factory")
    async def test_get_database_session_with_exception(self, mock_get_factory):
        """Test database session rollback on exception."""
        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_factory.return_value = mock_session_factory

        with pytest.raises(ValueError, match="Test error"):
            async with get_database_session() as session:
                raise ValueError("Test error")

        # Session context manager should handle rollback
        mock_get_factory.assert_called_once()


class TestConnectionPoolStatus:
    """Test suite for connection pool status monitoring."""

    @patch("cognivault.database.connection.get_database_engine")
    async def test_get_connection_pool_status_success(self, mock_get_engine):
        """Test successful connection pool status retrieval."""
        # Mock pool with realistic values
        mock_pool = MagicMock()
        mock_pool.__class__.__name__ = "QueuePool"
        mock_pool.size.return_value = 20
        mock_pool.checkedin.return_value = 15
        mock_pool.checkedout.return_value = 3
        mock_pool.overflow.return_value = 2
        mock_pool.invalid.return_value = 0
        mock_pool._max_overflow = 30

        mock_engine = MagicMock()
        mock_engine.pool = mock_pool
        mock_get_engine.return_value = mock_engine

        status = await get_connection_pool_status()

        assert status["pool_type"] == "QueuePool"
        assert status["size"] == 20
        assert status["checked_in"] == 15
        assert status["checked_out"] == 3
        assert status["total_connections"] == 18  # 15 + 3
        assert status["max_connections"] == 50  # 20 + 30
        assert status["utilization_percent"] == 36.0  # (18/50) * 100
        assert status["status"] == "healthy"  # < 80%

    @patch("cognivault.database.connection.get_database_engine")
    async def test_get_connection_pool_status_high_utilization(self, mock_get_engine):
        """Test connection pool status with high utilization."""
        mock_pool = MagicMock()
        mock_pool.__class__.__name__ = "QueuePool"
        mock_pool.size.return_value = 20
        mock_pool.checkedin.return_value = 5
        mock_pool.checkedout.return_value = 40  # High usage
        mock_pool.overflow.return_value = 20
        mock_pool.invalid.return_value = 0
        mock_pool._max_overflow = 30

        mock_engine = MagicMock()
        mock_engine.pool = mock_pool
        mock_get_engine.return_value = mock_engine

        status = await get_connection_pool_status()

        assert status["total_connections"] == 45  # 5 + 40
        assert status["utilization_percent"] == 90.0  # (45/50) * 100
        assert status["status"] == "warning"  # 80% <= x < 95%

    @patch("cognivault.database.connection.get_database_engine")
    async def test_get_connection_pool_status_critical(self, mock_get_engine):
        """Test connection pool status with critical utilization."""
        mock_pool = MagicMock()
        mock_pool.__class__.__name__ = "QueuePool"
        mock_pool.size.return_value = 20
        mock_pool.checkedin.return_value = 2
        mock_pool.checkedout.return_value = 46  # Critical usage
        mock_pool.overflow.return_value = 30
        mock_pool.invalid.return_value = 0
        mock_pool._max_overflow = 30

        mock_engine = MagicMock()
        mock_engine.pool = mock_pool
        mock_get_engine.return_value = mock_engine

        status = await get_connection_pool_status()

        assert status["utilization_percent"] == 96.0  # >= 95%
        assert status["status"] == "critical"

    @patch("cognivault.database.connection.get_database_engine")
    async def test_get_connection_pool_status_error(self, mock_get_engine):
        """Test connection pool status error handling."""
        mock_get_engine.side_effect = Exception("Engine error")

        status = await get_connection_pool_status()

        assert status["pool_type"] == "unknown"
        assert status["status"] == "error"
        assert "Engine error" in status["error"]


class TestHealthCheck:
    """Test suite for database health check functionality."""

    @patch("cognivault.database.connection.get_database_config")
    @patch("cognivault.database.connection.get_database_engine")
    @patch("cognivault.database.connection.get_connection_pool_status")
    @patch("asyncio.get_event_loop")
    async def test_health_check_success(
        self, mock_get_loop, mock_pool_status, mock_get_engine, mock_get_config
    ):
        """Test successful health check."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.connection_timeout = 10
        mock_config.get_connection_info.return_value = {
            "hostname": "test",
            "port": 5432,
            "database": "testdb",
        }
        mock_get_config.return_value = mock_config

        # Mock event loop for timing
        mock_loop = MagicMock()
        mock_loop.time.side_effect = [
            0.0,
            0.1,
            0.1,
        ]  # start and end times (extra for error handling)
        mock_get_loop.return_value = mock_loop

        # Mock database connection and queries
        mock_conn = AsyncMock()

        # Mock basic connectivity test - SELECT 1 as test
        mock_result1 = MagicMock()
        mock_result1.scalar.return_value = 1

        # Mock pgvector vector creation test - SELECT '[1,2,3]'::vector
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = "[1,2,3]"  # vector creation success

        # Mock pgvector distance test - SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector
        mock_result3 = MagicMock()
        mock_result3.scalar.return_value = 1.0  # vector distance

        # Mock database version - SELECT version()
        mock_result4 = MagicMock()
        mock_result4.scalar.return_value = "PostgreSQL 15.0"

        mock_conn.execute.side_effect = [
            mock_result1,
            mock_result2,
            mock_result3,
            mock_result4,
        ]

        mock_engine = MagicMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_engine.return_value = mock_engine

        # Mock pool status
        mock_pool_status.return_value = {"status": "healthy"}

        health = await health_check()

        assert health["status"] == "healthy"
        assert health["pgvector_available"] is True
        assert health["database_version"] == "PostgreSQL 15.0"
        assert "response_time_ms" in health
        assert "pool_status" in health
        assert "connection_info" in health

    @patch("cognivault.database.connection.get_database_config")
    @patch("cognivault.database.connection.get_database_engine")
    async def test_health_check_pgvector_unavailable(
        self, mock_get_engine, mock_get_config
    ):
        """Test health check when pgvector is unavailable."""
        mock_config = MagicMock()
        mock_config.connection_timeout = 10
        mock_config.get_connection_info.return_value = {"hostname": "test"}
        mock_get_config.return_value = mock_config

        mock_conn = AsyncMock()

        # Mock basic connectivity success
        mock_result1 = MagicMock()
        mock_result1.scalar.return_value = 1

        # Mock database version
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = "PostgreSQL 15.0"

        # Mock pgvector test failure
        mock_conn.execute.side_effect = [
            mock_result1,  # Basic test passes
            Exception("pgvector not available"),  # pgvector fails
            mock_result2,  # Version check passes
        ]

        mock_engine = MagicMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_engine.return_value = mock_engine

        with patch(
            "cognivault.database.connection.get_connection_pool_status"
        ) as mock_pool:
            mock_pool.return_value = {"status": "healthy"}

            health = await health_check()

        assert health["status"] == "healthy"
        assert health["pgvector_available"] is False

    @patch("cognivault.database.connection.get_database_config")
    @patch("cognivault.database.connection.get_database_engine")
    @patch("asyncio.timeout")
    async def test_health_check_timeout(
        self, mock_timeout, mock_get_engine, mock_get_config
    ):
        """Test health check timeout handling."""
        mock_config = MagicMock()
        mock_config.connection_timeout = 1  # Short timeout
        mock_get_config.return_value = mock_config

        # Mock timeout exception
        mock_timeout.side_effect = asyncio.TimeoutError()

        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        health = await health_check()

        assert health["status"] == "unhealthy"
        assert "timeout" in health["error"]
        assert health["response_time_ms"] == 1000  # timeout * 1000
        assert health["pgvector_available"] is False

    @patch("cognivault.database.connection.get_database_config")
    @patch("cognivault.database.connection.get_database_engine")
    async def test_health_check_connection_error(
        self, mock_get_engine, mock_get_config
    ):
        """Test health check connection error handling."""
        mock_config = MagicMock()
        mock_config.connection_timeout = 10
        mock_get_config.return_value = mock_config

        mock_get_engine.side_effect = Exception("Connection refused")

        health = await health_check()

        assert health["status"] == "unhealthy"
        assert "Connection refused" in health["error"]
        assert health["pgvector_available"] is False


class TestSchemaValidation:
    """Test suite for database schema validation."""

    @patch("cognivault.database.connection.get_database_engine")
    async def test_validate_database_schema_complete(self, mock_get_engine):
        """Test schema validation with complete schema."""
        mock_conn = AsyncMock()

        # Mock table existence checks (all tables exist)
        table_results = [MagicMock() for _ in range(5)]  # 5 required tables
        for result in table_results:
            result.scalar.return_value = 1

        # Mock pgvector extension check
        pgvector_result = MagicMock()
        pgvector_result.scalar.return_value = 1

        # Mock migration version check
        migration_result = MagicMock()
        migration_result.scalar.return_value = "abc123def456"

        mock_conn.execute.side_effect = table_results + [
            pgvector_result,
            migration_result,
        ]

        mock_engine = MagicMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_engine.return_value = mock_engine

        validation = await validate_database_schema()

        assert validation["schema_valid"] is True
        assert validation["status"] == "valid"
        assert len(validation["existing_tables"]) == 5
        assert len(validation["missing_tables"]) == 0
        assert validation["pgvector_installed"] is True
        assert validation["migration_version"] == "abc123def456"

    @patch("cognivault.database.connection.get_database_engine")
    async def test_validate_database_schema_missing_tables(self, mock_get_engine):
        """Test schema validation with missing tables."""
        mock_conn = AsyncMock()

        # Mock some tables missing (only 3 out of 5 exist)
        table_results = [
            MagicMock(),  # topics exists
            MagicMock(),  # questions exists
            MagicMock(),  # wiki_entries exists
            MagicMock(),  # api_keys missing
            MagicMock(),  # semantic_links missing
        ]
        table_results[0].scalar.return_value = 1  # topics
        table_results[1].scalar.return_value = 1  # questions
        table_results[2].scalar.return_value = 1  # wiki_entries
        table_results[3].scalar.return_value = None  # api_keys missing
        table_results[4].scalar.return_value = None  # semantic_links missing

        # Mock pgvector extension check
        pgvector_result = MagicMock()
        pgvector_result.scalar.return_value = 1

        mock_conn.execute.side_effect = table_results + [pgvector_result]

        mock_engine = MagicMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_engine.return_value = mock_engine

        validation = await validate_database_schema()

        assert validation["schema_valid"] is False
        assert validation["status"] == "invalid"
        assert len(validation["existing_tables"]) == 3
        assert set(validation["missing_tables"]) == {"api_keys", "semantic_links"}
        assert validation["pgvector_installed"] is True

    @patch("cognivault.database.connection.get_database_engine")
    async def test_validate_database_schema_error(self, mock_get_engine):
        """Test schema validation error handling."""
        mock_get_engine.side_effect = Exception("Database connection failed")

        validation = await validate_database_schema()

        assert validation["schema_valid"] is False
        assert validation["status"] == "error"
        assert "Database connection failed" in validation["error"]


class TestDatabaseInitialization:
    """Test suite for database initialization and cleanup."""

    @patch("cognivault.database.connection.get_database_engine")
    async def test_init_database_success(self, mock_get_engine):
        """Test successful database initialization."""
        mock_conn = AsyncMock()

        # Mock successful connectivity test
        basic_result = MagicMock()
        basic_result.scalar.return_value = 1

        # Mock pgvector extension check
        pgvector_result = MagicMock()
        pgvector_result.scalar.return_value = 1

        mock_conn.execute.side_effect = [basic_result, pgvector_result, MagicMock()]

        mock_engine = MagicMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_engine.return_value = mock_engine

        # Should not raise exception
        await init_database()

    @patch("cognivault.database.connection.get_database_engine")
    async def test_init_database_missing_pgvector(self, mock_get_engine):
        """Test database initialization with missing pgvector."""
        mock_conn = AsyncMock()

        # Mock successful connectivity test
        basic_result = MagicMock()
        basic_result.scalar.return_value = 1

        # Mock missing pgvector extension
        pgvector_result = MagicMock()
        pgvector_result.scalar.return_value = None

        mock_conn.execute.side_effect = [basic_result, pgvector_result]

        mock_engine = MagicMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_get_engine.return_value = mock_engine

        with pytest.raises(RuntimeError, match="pgvector extension"):
            await init_database()

    async def test_close_database(self):
        """Test database connection cleanup."""
        # Set up global state
        from cognivault.database import connection

        mock_engine = AsyncMock()
        connection._database_engine = mock_engine
        connection._session_factory = MagicMock()

        await close_database()

        # Verify cleanup
        mock_engine.dispose.assert_called_once()
        assert connection._database_engine is None
        assert connection._session_factory is None
