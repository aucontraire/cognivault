"""
Unit tests for database configuration management.

Tests the DatabaseConfig class and environment-based configuration loading
with validation and error handling.
"""

import os
import pytest
from typing import Any
from unittest.mock import patch, MagicMock
from urllib.parse import urlparse

from cognivault.database.config import DatabaseConfig, get_database_config


class TestDatabaseConfig:
    """Test suite for DatabaseConfig class."""

    def test_database_config_creation(self) -> None:
        """Test creating DatabaseConfig with explicit parameters."""
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://test:test@localhost:5432/testdb",
            echo_sql=True,
            pool_size=10,
            max_overflow=20,
            ssl_require=False,
        )

        assert (
            config.database_url
            == "postgresql+asyncpg://test:test@localhost:5432/testdb"
        )
        assert config.echo_sql is True
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.ssl_require is False

    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql+asyncpg://user:pass@testhost:5432/testdb",
            "DB_ECHO_SQL": "true",
            "DB_POOL_SIZE": "15",
            "DB_MAX_OVERFLOW": "25",
            "DB_SSL_REQUIRE": "true",
            "VECTOR_DIMENSIONS": "768",
        },
    )
    def test_from_env_with_environment_variables(self) -> None:
        """Test loading configuration from environment variables."""
        config = DatabaseConfig.from_env()

        assert (
            config.database_url == "postgresql+asyncpg://user:pass@testhost:5432/testdb"
        )
        assert config.echo_sql is True
        assert config.pool_size == 15
        assert config.max_overflow == 25
        assert config.ssl_require is True
        assert config.vector_dimensions == 768

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_defaults(self) -> None:
        """Test loading configuration with default values when env vars not set."""
        config = DatabaseConfig.from_env()

        # Should use development default URL
        assert "postgresql+asyncpg" in config.database_url
        assert "localhost:5432/cognivault" in config.database_url
        assert config.echo_sql is False
        assert config.pool_size == 20  # default
        assert config.max_overflow == 30  # default
        assert config.ssl_require is False  # default for dev

    def test_get_sync_url(self) -> None:
        """Test converting async URL to sync URL for migrations."""
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@host:5432/db"
        )

        sync_url = config.get_sync_url()
        assert sync_url == "postgresql://user:pass@host:5432/db"

    def test_get_engine_kwargs(self) -> None:
        """Test getting engine configuration parameters."""
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://test:test@localhost:5432/testdb",
            echo_sql=True,
            pool_size=10,
            max_overflow=20,
            pool_timeout=15,
            command_timeout=30,
        )

        kwargs = config.get_engine_kwargs()

        assert kwargs["echo"] is True
        assert kwargs["pool_size"] == 10
        assert kwargs["max_overflow"] == 20
        assert kwargs["pool_timeout"] == 15
        assert kwargs["future"] is True

        # Check connect_args
        connect_args = kwargs["connect_args"]
        assert connect_args["command_timeout"] == 30
        assert connect_args["server_settings"]["application_name"] == "cognivault"

    def test_validate_valid_config(self) -> None:
        """Test validation passes for valid configuration."""
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            connection_timeout=10,
            command_timeout=60,
        )

        # Should not raise any exception
        config.validate()

    def test_validate_invalid_database_url(self) -> None:
        """Test validation fails for invalid database URL."""
        config = DatabaseConfig(database_url="invalid-url")

        with pytest.raises(ValueError, match="Invalid database URL"):
            config.validate()

    def test_validate_invalid_pool_settings(self) -> None:
        """Test validation fails for invalid pool settings."""
        # Test invalid pool size
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db", pool_size=0
        )
        with pytest.raises(ValueError, match="Pool size must be at least 1"):
            config.validate()

        # Test negative max overflow
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
            max_overflow=-1,
        )
        with pytest.raises(ValueError, match="Max overflow cannot be negative"):
            config.validate()

    def test_validate_invalid_timeouts(self) -> None:
        """Test validation fails for invalid timeout settings."""
        # Test invalid connection timeout
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
            connection_timeout=0,
        )
        with pytest.raises(ValueError, match="Connection timeout must be positive"):
            config.validate()

        # Test invalid command timeout
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
            command_timeout=-1,
        )
        with pytest.raises(ValueError, match="Command timeout must be positive"):
            config.validate()

    def test_get_connection_info(self) -> None:
        """Test getting sanitized connection information."""
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://testuser:secret@testhost:5433/testdb",
            pool_size=15,
            max_overflow=25,
            ssl_require=True,
        )

        conn_info = config.get_connection_info()

        assert conn_info["scheme"] == "postgresql+asyncpg"
        assert conn_info["hostname"] == "testhost"
        assert conn_info["port"] == 5433
        assert conn_info["database"] == "testdb"
        assert conn_info["username"] == "testuser"
        # Password should not be included
        assert "password" not in conn_info
        assert conn_info["pool_size"] == 15
        assert conn_info["max_overflow"] == 25
        assert conn_info["ssl_enabled"] is True

    def test_mask_credentials(self) -> None:
        """Test masking credentials in database URL."""
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:secretpass@host:5432/db"
        )

        masked_url = config.mask_credentials()
        assert "secretpass" not in masked_url
        assert "***" in masked_url
        assert "user" in masked_url
        assert "host:5432/db" in masked_url

    def test_mask_credentials_no_password(self) -> None:
        """Test masking credentials when no password present."""
        config = DatabaseConfig(database_url="postgresql+asyncpg://user@host:5432/db")

        masked_url = config.mask_credentials()
        # Should return original URL when no password
        assert masked_url == config.database_url

    @patch("os.path.exists")
    def test_ssl_context_with_files(self, mock_exists: MagicMock) -> None:
        """Test SSL context creation when SSL files exist."""
        mock_exists.return_value = True

        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@host:5432/db",
            ssl_ca_file="/path/to/ca.crt",
            ssl_cert_file="/path/to/client.crt",
            ssl_key_file="/path/to/client.key",
        )

        ssl_context = config._build_ssl_context()

        assert ssl_context["ca"] == "/path/to/ca.crt"
        assert ssl_context["cert"] == "/path/to/client.crt"
        assert ssl_context["key"] == "/path/to/client.key"

    @patch("os.path.exists")
    def test_ssl_context_missing_files(self, mock_exists: MagicMock) -> None:
        """Test SSL context when SSL files don't exist."""
        mock_exists.return_value = False

        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@host:5432/db",
            ssl_ca_file="/nonexistent/ca.crt",
        )

        ssl_context = config._build_ssl_context()
        assert ssl_context is None

    @patch("os.path.exists")
    def test_validate_missing_ssl_files(self, mock_exists: MagicMock) -> None:
        """Test validation fails when SSL files don't exist."""
        mock_exists.return_value = False

        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@host:5432/db",
            ssl_ca_file="/nonexistent/ca.crt",
        )

        with pytest.raises(ValueError, match="SSL CA file not found"):
            config.validate()

    def test_repr(self) -> None:
        """Test string representation of config."""
        config = DatabaseConfig(
            database_url="postgresql+asyncpg://user:pass@testhost:5433/testdb",
            pool_size=10,
            ssl_require=True,
        )

        repr_str = repr(config)

        assert "DatabaseConfig" in repr_str
        assert "testhost:5433" in repr_str
        assert "testdb" in repr_str
        assert "pool_size=10" in repr_str
        assert "ssl=True" in repr_str
        # Password should not appear in repr
        assert "pass" not in repr_str


class TestGlobalDatabaseConfig:
    """Test suite for global database configuration management."""

    def setup_method(self) -> None:
        """Reset global config before each test."""
        # Reset the global config variable
        from cognivault.database import config

        config._database_config = None

    def teardown_method(self) -> None:
        """Reset global config after each test."""
        # Reset the global config variable
        from cognivault.database import config

        config._database_config = None

    @patch.dict(
        os.environ,
        {"DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/testdb"},
    )
    def test_get_database_config_creates_singleton(self) -> None:
        """Test that get_database_config creates a singleton instance."""
        config1 = get_database_config()
        config2 = get_database_config()

        # Should return the same instance
        assert config1 is config2
        assert (
            config1.database_url
            == "postgresql+asyncpg://test:test@localhost:5432/testdb"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_get_database_config_with_defaults(self) -> None:
        """Test getting config with default values."""
        config = get_database_config()

        # Should create config with development defaults
        assert "postgresql+asyncpg" in config.database_url
        assert "localhost:5432/cognivault" in config.database_url
        assert config.pool_size == 20
