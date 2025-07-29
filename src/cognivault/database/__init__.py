"""
Database integration module for CogniVault.

Provides PostgreSQL + pgvector integration with async SQLAlchemy,
production-ready connection pooling, centralized session management,
Alembic migrations, and vector similarity search capabilities.
"""

from .config import DatabaseConfig, get_database_config
from .connection import (
    close_database,
    get_connection_pool_status,
    get_database_engine,
    get_database_session,
    health_check,
    init_database,
    validate_database_schema,
)
from .models import Base
from .repositories import RepositoryFactory
from .session_factory import (
    DatabaseSessionFactory,
    get_database_session_factory,
    initialize_database_session_factory,
    shutdown_database_session_factory,
)

__all__ = [
    # Configuration
    "DatabaseConfig",
    "get_database_config",
    # Connection management
    "get_database_engine",
    "get_database_session",
    "get_connection_pool_status",
    "init_database",
    "close_database",
    "health_check",
    "validate_database_schema",
    # Session factory
    "DatabaseSessionFactory",
    "get_database_session_factory",
    "initialize_database_session_factory",
    "shutdown_database_session_factory",
    # Models and repositories
    "Base",
    "RepositoryFactory",
]
