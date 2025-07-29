"""
Database configuration for CogniVault PostgreSQL + pgvector integration.

Provides production-ready database configuration with connection pooling,
SSL support, and comprehensive environment-based settings.
"""

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from cognivault.observability import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """
    Production-ready PostgreSQL database configuration.

    Provides comprehensive settings for connection pooling, SSL,
    timeouts, and monitoring for production deployment.
    """

    # Connection parameters
    database_url: str
    echo_sql: bool = False

    # Connection pool settings (production-ready)
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True

    # Vector search parameters
    vector_dimensions: int = 1536  # text-embedding-3-large dimensions

    # Connection timeouts
    connection_timeout: int = 10
    command_timeout: int = 60

    # SSL/Security settings
    ssl_require: bool = True
    ssl_ca_file: str | None = None
    ssl_cert_file: str | None = None
    ssl_key_file: str | None = None

    # Migration settings
    alembic_config_path: str = "alembic.ini"
    migration_directory: str = "migrations"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        Create database configuration from environment variables.

        Environment variables:
        - DATABASE_URL: PostgreSQL connection string (required)
        - DB_ECHO_SQL: Enable SQL query logging (default: False)
        - DB_POOL_SIZE: Connection pool size (default: 20)
        - DB_MAX_OVERFLOW: Max pool overflow (default: 30)
        - DB_POOL_TIMEOUT: Pool checkout timeout (default: 30s)
        - DB_POOL_RECYCLE: Connection recycle time (default: 3600s)
        - DB_POOL_PRE_PING: Enable connection health checks (default: True)
        - DB_CONNECTION_TIMEOUT: Connection timeout (default: 10s)
        - DB_COMMAND_TIMEOUT: Query timeout (default: 60s)
        - DB_SSL_REQUIRE: Require SSL connection (default: True)
        - DB_SSL_CA_FILE: SSL CA certificate file path
        - DB_SSL_CERT_FILE: SSL certificate file path
        - DB_SSL_KEY_FILE: SSL private key file path
        - VECTOR_DIMENSIONS: Vector embedding dimensions (default: 1536)

        Returns:
            DatabaseConfig instance with environment-based settings

        Raises:
            ValueError: If DATABASE_URL is missing or invalid
        """
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            # Provide default for development, but warn
            database_url = (
                "postgresql+asyncpg://postgres:postgres@localhost:5432/cognivault"
            )
            logger.warning(
                "DATABASE_URL not set, using development default: postgresql+asyncpg://localhost:5432/cognivault"
            )

        # Validate database URL format
        parsed = urlparse(database_url)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(
                f"Invalid DATABASE_URL format: {database_url}. "
                "Must include scheme and hostname."
            )

        # Ensure asyncpg driver is specified for async support
        if not parsed.scheme.startswith("postgresql+asyncpg"):
            logger.warning(
                f"DATABASE_URL uses {parsed.scheme} driver. "
                "Recommend postgresql+asyncpg for async support."
            )

        return cls(
            database_url=database_url,
            echo_sql=os.getenv("DB_ECHO_SQL", "false").lower() == "true",
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            pool_pre_ping=os.getenv("DB_POOL_PRE_PING", "true").lower() == "true",
            vector_dimensions=int(os.getenv("VECTOR_DIMENSIONS", "1536")),
            connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT", "10")),
            command_timeout=int(os.getenv("DB_COMMAND_TIMEOUT", "60")),
            ssl_require=os.getenv("DB_SSL_REQUIRE", "false").lower()
            == "true",  # Default false for dev
            ssl_ca_file=os.getenv("DB_SSL_CA_FILE"),
            ssl_cert_file=os.getenv("DB_SSL_CERT_FILE"),
            ssl_key_file=os.getenv("DB_SSL_KEY_FILE"),
            alembic_config_path=os.getenv("ALEMBIC_CONFIG", "alembic.ini"),
            migration_directory=os.getenv("MIGRATION_DIR", "migrations"),
        )

    def get_sync_url(self) -> str:
        """Get synchronous database URL for Alembic migrations."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")

    def get_engine_kwargs(self) -> dict[str, Any]:
        """
        Get SQLAlchemy engine configuration parameters.

        Returns:
            Dictionary of engine configuration parameters for async engine
        """
        engine_kwargs: dict[str, Any] = {
            "echo": self.echo_sql,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "future": True,  # Use SQLAlchemy 2.0 style
        }

        # Add connection arguments for asyncpg
        connect_args: dict[str, Any] = {
            "command_timeout": self.command_timeout,
            "server_settings": {
                "application_name": "cognivault",
            },
        }

        # Add SSL configuration if required
        if self.ssl_require:
            ssl_context = self._build_ssl_context()
            if ssl_context:
                connect_args["ssl"] = ssl_context

        engine_kwargs["connect_args"] = connect_args

        return engine_kwargs

    def _build_ssl_context(self) -> dict[str, Any] | None:
        """
        Build SSL context for database connections.

        Returns:
            SSL context dictionary or None if SSL files not provided
        """
        ssl_config = {}

        if self.ssl_ca_file and os.path.exists(self.ssl_ca_file):
            ssl_config["ca"] = self.ssl_ca_file

        if self.ssl_cert_file and os.path.exists(self.ssl_cert_file):
            ssl_config["cert"] = self.ssl_cert_file

        if self.ssl_key_file and os.path.exists(self.ssl_key_file):
            ssl_config["key"] = self.ssl_key_file

        # Return SSL context if we have at least CA file
        return ssl_config if ssl_config else None

    def validate(self) -> None:
        """
        Validate database configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate database URL format
        try:
            parsed = urlparse(self.database_url)
            if not parsed.scheme.startswith("postgresql"):
                raise ValueError(f"Invalid database scheme: {parsed.scheme}")
            if not parsed.hostname:
                raise ValueError("Database hostname is required")
        except Exception as e:
            raise ValueError(f"Invalid database URL: {e}")

        # Validate pool settings
        if self.pool_size < 1:
            raise ValueError("Pool size must be at least 1")
        if self.max_overflow < 0:
            raise ValueError("Max overflow cannot be negative")
        if self.pool_timeout <= 0:
            raise ValueError("Pool timeout must be positive")

        # Validate timeout settings
        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")
        if self.command_timeout <= 0:
            raise ValueError("Command timeout must be positive")

        # Validate SSL files if provided
        ssl_files = [
            ("SSL CA file", self.ssl_ca_file),
            ("SSL certificate file", self.ssl_cert_file),
            ("SSL key file", self.ssl_key_file),
        ]

        for file_desc, file_path in ssl_files:
            if file_path and not os.path.exists(file_path):
                raise ValueError(f"{file_desc} not found: {file_path}")

        logger.debug("Database configuration validation passed")

    def mask_credentials(self) -> str:
        """Get database URL with masked credentials for logging."""
        parsed = urlparse(self.database_url)
        if parsed.password:
            masked_url = self.database_url.replace(parsed.password, "***")
            return masked_url
        return self.database_url

    def get_connection_info(self) -> dict[str, Any]:
        """
        Get sanitized connection information for logging/monitoring.

        Returns:
            Dictionary with connection info (sensitive data redacted)
        """
        parsed = urlparse(self.database_url)

        return {
            "scheme": parsed.scheme,
            "hostname": parsed.hostname or "unknown",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") if parsed.path else None,
            "username": parsed.username or "unknown",
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "ssl_enabled": self.ssl_require,
        }

    def __repr__(self) -> str:
        """String representation with sensitive data redacted."""
        conn_info = self.get_connection_info()
        return (
            f"DatabaseConfig("
            f"host={conn_info['hostname']}:{conn_info['port']}, "
            f"db={conn_info['database']}, "
            f"pool_size={self.pool_size}, "
            f"ssl={self.ssl_require})"
        )


# Global database configuration instance
_database_config: DatabaseConfig | None = None


def get_database_config() -> DatabaseConfig:
    """Get or create the global database configuration."""
    global _database_config

    if _database_config is None:
        _database_config = DatabaseConfig.from_env()
        logger.info(
            f"Database configuration loaded: {_database_config.mask_credentials()}"
        )

    return _database_config
