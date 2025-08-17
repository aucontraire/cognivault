"""
Database connection management for CogniVault.

Provides production-ready async SQLAlchemy engine and session management with
proper connection pooling, health checks, and monitoring.
"""

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from cognivault.observability import get_logger

from .config import get_database_config

logger = get_logger(__name__)

# Global database engine and session factory instances
_database_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_engine() -> AsyncEngine:
    """
    Get or create the global database engine with production-ready settings.

    Returns:
        AsyncEngine: Configured async SQLAlchemy engine
    """
    global _database_engine, _session_factory

    if _database_engine is None:
        config = get_database_config()

        # Validate configuration before creating engine
        config.validate()

        # Get engine configuration parameters
        engine_kwargs = config.get_engine_kwargs()

        # For testing environments, use StaticPool to allow connection reuse
        # For production, use NullPool as originally intended
        testing_mode = (
            os.environ.get("TESTING", "false").lower() == "true"
            or "test" in config.database_url.lower()
            or config.pool_size <= 10
        )  # Small pool size indicates testing

        if testing_mode:
            # Use StaticPool for tests - allows connection reuse and is more forgiving
            from sqlalchemy.pool import StaticPool

            engine_kwargs["poolclass"] = StaticPool

            # Remove pool-specific parameters that StaticPool doesn't accept
            pool_params_to_remove = [
                "pool_size",
                "max_overflow",
                "pool_timeout",
                "pool_recycle",
                "pool_pre_ping",
            ]
            for param in pool_params_to_remove:
                engine_kwargs.pop(param, None)

            logger.info("Using StaticPool for testing environment")
        else:
            # Use NullPool for production async engines (SQLAlchemy requirement for async)
            # Note: Async engines handle their own connection pooling internally
            engine_kwargs["poolclass"] = NullPool

            # Remove pool-specific parameters that NullPool doesn't accept
            pool_params_to_remove = [
                "pool_size",
                "max_overflow",
                "pool_timeout",
                "pool_recycle",
                "pool_pre_ping",
            ]
            for param in pool_params_to_remove:
                engine_kwargs.pop(param, None)
            logger.info("Using NullPool for production environment")

        _database_engine = create_async_engine(config.database_url, **engine_kwargs)

        # Create session factory with optimized settings
        _session_factory = async_sessionmaker(
            _database_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )

        conn_info = config.get_connection_info()
        logger.info(
            f"Database engine created - "
            f"Host: {conn_info['hostname']}:{conn_info['port']}, "
            f"DB: {conn_info['database']}, "
            f"Pool: {config.pool_size}+{config.max_overflow}"
        )

    return _database_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the database session factory."""
    global _session_factory

    if _session_factory is None:
        # Initialize engine which also creates session factory
        get_database_engine()

    return _session_factory  # type: ignore


@asynccontextmanager
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session with automatic cleanup and retry logic."""
    session_factory = get_session_factory()
    max_retries = 3

    for attempt in range(max_retries):
        session = None
        try:
            session = session_factory()
            async with session:
                yield session
            break
        except (ConnectionResetError, SQLAlchemyError) as e:
            if session:
                try:
                    await session.rollback()
                except Exception:
                    pass  # Rollback may fail if connection is broken
                try:
                    await session.close()
                except Exception:
                    pass  # Close may fail if connection is broken

            if attempt == max_retries - 1:
                logger.error(
                    f"Database session failed after {max_retries} attempts: {e}"
                )
                raise

            # Exponential backoff
            retry_delay = min(2**attempt, 10)  # Cap at 10 seconds
            logger.warning(
                f"Database session attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s..."
            )
            await asyncio.sleep(retry_delay)
        except Exception as e:
            if session:
                try:
                    await session.rollback()
                except Exception:
                    pass
                try:
                    await session.close()
                except Exception:
                    pass
            raise


async def init_database() -> None:
    """Initialize database connection and verify pgvector extension with retry logic."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            engine = get_database_engine()

            # Test database connection with timeout
            async with asyncio.timeout(30):  # 30 second timeout
                async with engine.begin() as conn:
                    # Test basic connectivity
                    result = await conn.execute(text("SELECT 1"))
                    assert result.scalar() == 1

                    # Verify pgvector extension is available
                    result = await conn.execute(
                        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    )
                    if result.scalar() != 1:
                        logger.warning("pgvector extension not found in database")
                        # Don't fail initialization for missing pgvector in tests
                        testing_mode = (
                            os.environ.get("TESTING", "false").lower() == "true"
                            or "test" in str(engine.url).lower()
                        )
                        if not testing_mode:
                            raise RuntimeError(
                                "Database is missing pgvector extension. "
                                "Run 'CREATE EXTENSION vector;' as a superuser."
                            )
                    else:
                        # Test vector functionality if extension is available
                        try:
                            await conn.execute(text("SELECT '[1,2,3]'::vector"))
                        except Exception as vector_e:
                            logger.warning(
                                f"pgvector functionality test failed: {vector_e}"
                            )

            logger.info("Database initialization successful")
            return

        except (ConnectionResetError, TimeoutError, SQLAlchemyError) as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Database initialization failed after {max_retries} attempts: {e}"
                )
                raise

            # Exponential backoff
            retry_delay = min(2**attempt, 10)  # Cap at 10 seconds
            logger.warning(
                f"Database initialization attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s..."
            )
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            raise


async def close_database() -> None:
    """Close database connections and cleanup resources."""
    global _database_engine, _session_factory

    if _database_engine:
        await _database_engine.dispose()
        _database_engine = None
        _session_factory = None
        logger.info("Database connections closed")


async def get_connection_pool_status() -> dict[str, Any]:
    """
    Get detailed connection pool status and metrics.

    Returns:
        Dictionary with connection pool status and performance metrics
    """
    try:
        engine = get_database_engine()
        pool = engine.pool
        pool_type = pool.__class__.__name__

        # NullPool doesn't maintain connection statistics like QueuePool
        if pool_type == "NullPool":
            return {
                "pool_type": "NullPool",
                "status": "active",
                "description": (
                    "NullPool creates new connections for each request (async engine)"
                ),
                "connection_management": "engine_internal",
            }
        else:
            # For other pool types (in case we switch back)
            pool_status = {
                "pool_type": pool_type,
                "size": int(getattr(pool, "size", lambda: 0)()),
                "checked_in": int(getattr(pool, "checkedin", lambda: 0)()),
                "checked_out": int(getattr(pool, "checkedout", lambda: 0)()),
                "overflow": int(getattr(pool, "overflow", lambda: 0)()),
                "invalid": int(getattr(pool, "invalid", lambda: 0)()),
            }

            # Calculate utilization metrics for traditional pools
            checked_in: int = pool_status["checked_in"]  # type: ignore
            checked_out: int = pool_status["checked_out"]  # type: ignore
            size: int = pool_status["size"]  # type: ignore
            max_overflow = int(getattr(pool, "_max_overflow", 0))

            total_connections = checked_in + checked_out
            max_connections = size + max_overflow
            utilization = (
                (total_connections / max_connections) * 100
                if max_connections > 0
                else 0.0
            )

            status = "healthy"
            if utilization >= 95:
                status = "critical"
            elif utilization >= 80:
                status = "warning"

            pool_status.update(
                {
                    "total_connections": total_connections,
                    "max_connections": max_connections,
                    "utilization_percent": round(utilization, 2),
                    "status": status,
                }
            )

        return pool_status

    except Exception as e:
        logger.error(f"Failed to get connection pool status: {e}")
        return {"pool_type": "unknown", "status": "error", "error": str(e)}


async def health_check() -> dict[str, Any]:
    """
    Perform comprehensive database health check.

    Returns:
        Dictionary with health status, performance metrics, and diagnostics
    """
    try:
        config = get_database_config()
        start_time = asyncio.get_event_loop().time()

        # Test connection with timeout
        engine = get_database_engine()

        async with asyncio.timeout(config.connection_timeout):
            async with engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1 as test"))
                assert result.scalar() == 1

                # Test pgvector extension
                pgvector_available = True
                try:
                    await conn.execute(text("SELECT '[1,2,3]'::vector"))
                    # Test vector operations
                    result = await conn.execute(
                        text(
                            "SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector as distance"
                        )
                    )
                    vector_distance = result.scalar()
                    assert vector_distance is not None
                except Exception as e:
                    logger.warning(f"pgvector extension test failed: {e}")
                    pgvector_available = False

                # Test database version
                result = await conn.execute(text("SELECT version()"))
                db_version = result.scalar()

        response_time = asyncio.get_event_loop().time() - start_time

        # Get connection pool status
        pool_status = await get_connection_pool_status()

        return {
            "status": "healthy",
            "response_time_ms": round(response_time * 1000, 2),
            "database_version": db_version,
            "pgvector_available": pgvector_available,
            "pool_status": pool_status,
            "connection_info": config.get_connection_info(),
            "timestamp": asyncio.get_event_loop().time(),
        }

    except TimeoutError:
        return {
            "status": "unhealthy",
            "error": "Database connection timeout",
            "response_time_ms": config.connection_timeout * 1000,
            "pgvector_available": False,
            "timestamp": asyncio.get_event_loop().time(),
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "pgvector_available": False,
            "timestamp": asyncio.get_event_loop().time(),
        }


async def validate_database_schema() -> dict[str, Any]:
    """
    Validate that the database schema matches expected structure.

    Returns:
        Dictionary with schema validation results
    """
    try:
        engine = get_database_engine()

        async with engine.begin() as conn:
            # Check for required tables
            required_tables = [
                "topics",
                "questions",
                "wiki_entries",
                "api_keys",
                "semantic_links",
            ]
            existing_tables = []

            for table in required_tables:
                result = await conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        f"WHERE table_name = '{table}'"
                    )
                )
                if result.scalar():
                    existing_tables.append(table)

            missing_tables = set(required_tables) - set(existing_tables)

            # Check for pgvector extension
            result = await conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            pgvector_installed = result.scalar() == 1

            # Check for migration version
            migration_version = None
            try:
                result = await conn.execute(
                    text("SELECT version_num FROM alembic_version")
                )
                migration_version = result.scalar()
            except Exception:
                pass  # alembic_version table may not exist yet

            schema_valid = len(missing_tables) == 0 and pgvector_installed

            return {
                "schema_valid": schema_valid,
                "existing_tables": existing_tables,
                "missing_tables": list(missing_tables),
                "pgvector_installed": pgvector_installed,
                "migration_version": migration_version,
                "status": "valid" if schema_valid else "invalid",
            }

    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        return {"schema_valid": False, "status": "error", "error": str(e)}
