"""Tests for the application configuration system."""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cognivault.config.app_config import (
    ApplicationConfig,
    ExecutionConfig,
    FileConfig,
    ModelConfig,
    DevelopmentConfig,
    Environment,
    LogLevel,
    get_config,
    set_config,
    reset_config,
    load_config_from_file,
)


class TestExecutionConfig:
    """Test suite for ExecutionConfig."""

    def test_default_values(self):
        """Test that ExecutionConfig has correct default values."""
        config = ExecutionConfig()

        assert config.max_retries == 3
        assert config.timeout_seconds == 10
        assert config.retry_delay_seconds == 1.0
        assert config.enable_simulation_delay is False
        assert config.simulation_delay_seconds == 0.1
        assert config.default_agents == ["refiner", "historian", "synthesis"]
        assert config.critic_enabled is True


class TestFileConfig:
    """Test suite for FileConfig."""

    def test_default_values(self):
        """Test that FileConfig has correct default values."""
        config = FileConfig()

        assert config.notes_directory == "./src/cognivault/notes"
        assert config.logs_directory == "./src/cognivault/logs"
        assert config.question_truncate_length == 40
        assert config.hash_length == 6
        assert config.filename_separator == "_"
        assert config.max_file_size == 10 * 1024 * 1024  # 10MB
        assert config.max_note_files == 1000


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_default_values(self):
        """Test that ModelConfig has correct default values."""
        config = ModelConfig()

        assert config.default_provider == "openai"
        assert config.default_model == "gpt-4"
        assert config.max_tokens_per_request == 4096
        assert config.temperature == 0.7
        assert config.mock_tokens_used == 10
        assert config.mock_response_truncate_length == 50


class TestDevelopmentConfig:
    """Test suite for DevelopmentConfig."""

    def test_default_values(self):
        """Test that DevelopmentConfig has correct default values."""
        config = DevelopmentConfig()

        assert config.test_timeout_multiplier == 1.5
        assert config.test_simulation_enabled is True
        assert len(config.mock_history_entries) == 3
        assert "Mexico had a third party" in config.mock_history_entries[0]
        assert config.prompt_min_length == 2000
        assert config.prompt_max_length == 8000


class TestApplicationConfig:
    """Test suite for ApplicationConfig."""

    def test_default_values(self):
        """Test that ApplicationConfig has correct default values."""
        config = ApplicationConfig()

        assert config.environment == Environment.DEVELOPMENT
        assert config.log_level == LogLevel.INFO
        assert config.debug_mode is False
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.files, FileConfig)
        assert isinstance(config.models, ModelConfig)
        assert isinstance(config.testing, DevelopmentConfig)

    def test_from_env_basic(self):
        """Test creating configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_ENV": "production",
                "COGNIVAULT_LOG_LEVEL": "ERROR",
                "COGNIVAULT_DEBUG": "true",
                "COGNIVAULT_MAX_RETRIES": "5",
                "COGNIVAULT_TIMEOUT_SECONDS": "20",
            },
        ):
            config = ApplicationConfig.from_env()

            assert config.environment == Environment.PRODUCTION
            assert config.log_level == LogLevel.ERROR
            assert config.debug_mode is True
            assert config.execution.max_retries == 5
            assert config.execution.timeout_seconds == 20

    def test_from_env_file_config(self):
        """Test environment variables for file configuration."""
        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_NOTES_DIR": "/custom/notes",
                "COGNIVAULT_LOGS_DIR": "/custom/logs",
                "COGNIVAULT_QUESTION_TRUNCATE": "50",
                "COGNIVAULT_HASH_LENGTH": "8",
                "COGNIVAULT_MAX_FILE_SIZE": "20971520",  # 20MB
            },
        ):
            config = ApplicationConfig.from_env()

            assert config.files.notes_directory == "/custom/notes"
            assert config.files.logs_directory == "/custom/logs"
            assert config.files.question_truncate_length == 50
            assert config.files.hash_length == 8
            assert config.files.max_file_size == 20 * 1024 * 1024

    def test_from_env_model_config(self):
        """Test environment variables for model configuration."""
        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_LLM": "anthropic",
                "OPENAI_MODEL": "gpt-3.5-turbo",
                "COGNIVAULT_MAX_TOKENS": "2048",
                "COGNIVAULT_TEMPERATURE": "0.9",
            },
        ):
            config = ApplicationConfig.from_env()

            assert config.models.default_provider == "anthropic"
            assert config.models.default_model == "gpt-3.5-turbo"
            assert config.models.max_tokens_per_request == 2048
            assert config.models.temperature == 0.9

    def test_from_env_execution_config(self):
        """Test environment variables for execution configuration."""
        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_SIMULATION_DELAY": "true",
                "COGNIVAULT_SIMULATION_DELAY_SECONDS": "0.5",
                "COGNIVAULT_RETRY_DELAY": "2.0",
                "COGNIVAULT_CRITIC_ENABLED": "false",
            },
        ):
            config = ApplicationConfig.from_env()

            assert config.execution.enable_simulation_delay is True
            assert config.execution.simulation_delay_seconds == 0.5
            assert config.execution.retry_delay_seconds == 2.0
            assert config.execution.critic_enabled is False

    def test_from_env_invalid_values(self):
        """Test that invalid environment values fall back to defaults."""
        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_ENV": "invalid_env",
                "COGNIVAULT_LOG_LEVEL": "invalid_level",
            },
        ):
            config = ApplicationConfig.from_env()

            # Should fall back to defaults for invalid values
            assert config.environment == Environment.DEVELOPMENT
            assert config.log_level == LogLevel.INFO

    def test_from_file(self):
        """Test loading configuration from a JSON file."""
        config_data = {
            "environment": "testing",
            "log_level": "DEBUG",
            "debug_mode": True,
            "execution": {
                "max_retries": 2,
                "timeout_seconds": 15,
                "retry_delay_seconds": 0.5,
                "enable_simulation_delay": True,
                "simulation_delay_seconds": 0.2,
                "critic_enabled": False,
            },
            "files": {
                "notes_directory": "/test/notes",
                "question_truncate_length": 30,
                "hash_length": 4,
            },
            "models": {
                "default_provider": "stub",
                "default_model": "test-model",
                "temperature": 0.5,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = ApplicationConfig.from_file(config_file)

            assert config.environment == Environment.TESTING
            assert config.log_level == LogLevel.DEBUG
            assert config.debug_mode is True
            assert config.execution.max_retries == 2
            assert config.execution.timeout_seconds == 15
            assert config.execution.retry_delay_seconds == 0.5
            assert config.execution.enable_simulation_delay is True
            assert config.execution.simulation_delay_seconds == 0.2
            assert config.execution.critic_enabled is False
            assert config.files.notes_directory == "/test/notes"
            assert config.files.question_truncate_length == 30
            assert config.files.hash_length == 4
            assert config.models.default_provider == "stub"
            assert config.models.default_model == "test-model"
            assert config.models.temperature == 0.5
        finally:
            os.unlink(config_file)

    def test_from_file_not_found(self):
        """Test that from_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            ApplicationConfig.from_file("/nonexistent/config.json")

    def test_from_file_invalid_json(self):
        """Test that from_file raises JSONDecodeError for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            config_file = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                ApplicationConfig.from_file(config_file)
        finally:
            os.unlink(config_file)

    def test_save_to_file(self):
        """Test saving configuration to a JSON file."""
        config = ApplicationConfig()
        config.environment = Environment.TESTING
        config.log_level = LogLevel.DEBUG
        config.debug_mode = True
        config.execution.max_retries = 5
        config.files.notes_directory = "/test/notes"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = f.name

        try:
            config.save_to_file(config_file)

            # Load the saved file and verify contents
            with open(config_file, "r") as f:
                saved_data = json.load(f)

            assert saved_data["environment"] == "testing"
            assert saved_data["log_level"] == "DEBUG"
            assert saved_data["debug_mode"] is True
            assert saved_data["execution"]["max_retries"] == 5
            assert saved_data["files"]["notes_directory"] == "/test/notes"
        finally:
            os.unlink(config_file)

    def test_get_timeout_for_environment(self):
        """Test timeout adjustment for different environments."""
        config = ApplicationConfig()
        config.execution.timeout_seconds = 10
        config.testing.test_timeout_multiplier = 2.0

        # Development environment uses base timeout
        config.environment = Environment.DEVELOPMENT
        assert config.get_timeout_for_environment() == 10

        # Production environment uses base timeout
        config.environment = Environment.PRODUCTION
        assert config.get_timeout_for_environment() == 10

        # Testing environment uses multiplied timeout
        config.environment = Environment.TESTING
        assert config.get_timeout_for_environment() == 20

    def test_ensure_directories_exist(self):
        """Test that ensure_directories_exist creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ApplicationConfig()
            config.files.notes_directory = os.path.join(temp_dir, "notes")
            config.files.logs_directory = os.path.join(temp_dir, "logs")

            # Directories shouldn't exist initially
            assert not os.path.exists(config.files.notes_directory)
            assert not os.path.exists(config.files.logs_directory)

            # Call ensure_directories_exist
            config.ensure_directories_exist()

            # Directories should now exist
            assert os.path.exists(config.files.notes_directory)
            assert os.path.exists(config.files.logs_directory)

    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        config = ApplicationConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_execution_config(self):
        """Test validation of invalid execution configuration."""
        config = ApplicationConfig()
        config.execution.max_retries = -1
        config.execution.timeout_seconds = 0
        config.execution.retry_delay_seconds = -0.5
        config.execution.simulation_delay_seconds = -0.1

        errors = config.validate()
        assert "max_retries must be non-negative" in errors
        assert "timeout_seconds must be positive" in errors
        assert "retry_delay_seconds must be non-negative" in errors
        assert "simulation_delay_seconds must be non-negative" in errors

    def test_validate_invalid_file_config(self):
        """Test validation of invalid file configuration."""
        config = ApplicationConfig()
        config.files.question_truncate_length = 0
        config.files.hash_length = -1
        config.files.max_file_size = 0
        config.files.max_note_files = -10

        errors = config.validate()
        assert "question_truncate_length must be positive" in errors
        assert "hash_length must be positive" in errors
        assert "max_file_size must be positive" in errors
        assert "max_note_files must be positive" in errors

    def test_validate_invalid_model_config(self):
        """Test validation of invalid model configuration."""
        config = ApplicationConfig()
        config.models.max_tokens_per_request = 0
        config.models.temperature = -0.1

        errors = config.validate()
        assert "max_tokens_per_request must be positive" in errors
        assert "temperature must be between 0 and 2" in errors

        # Test temperature upper bound
        config.models.temperature = 2.1
        errors = config.validate()
        assert "temperature must be between 0 and 2" in errors

    def test_validate_invalid_testing_config(self):
        """Test validation of invalid testing configuration."""
        config = ApplicationConfig()
        config.testing.test_timeout_multiplier = 0
        config.testing.prompt_min_length = -1
        config.testing.prompt_max_length = 100

        errors = config.validate()
        assert "test_timeout_multiplier must be positive" in errors
        assert "prompt_min_length must be non-negative" in errors

        # Test case where max < min separately
        config.testing.prompt_min_length = 200  # max < min
        errors = config.validate()
        assert "prompt_max_length must be greater than prompt_min_length" in errors


class TestGlobalConfiguration:
    """Test suite for global configuration functions."""

    def setup_method(self):
        """Reset global configuration before each test."""
        reset_config()

    def teardown_method(self):
        """Reset global configuration after each test."""
        reset_config()

    def test_get_config_default(self):
        """Test that get_config returns a default configuration."""
        config = get_config()
        assert isinstance(config, ApplicationConfig)
        assert config.environment == Environment.DEVELOPMENT

    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        """Test setting a custom global configuration."""
        custom_config = ApplicationConfig()
        custom_config.environment = Environment.PRODUCTION
        custom_config.debug_mode = True

        set_config(custom_config)

        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.environment == Environment.PRODUCTION
        assert retrieved_config.debug_mode is True

    def test_reset_config(self):
        """Test resetting the global configuration."""
        # Set a custom config
        custom_config = ApplicationConfig()
        custom_config.environment = Environment.PRODUCTION
        set_config(custom_config)

        # Verify it's set
        assert get_config().environment == Environment.PRODUCTION

        # Reset and verify it creates a new default instance
        reset_config()
        new_config = get_config()
        assert new_config is not custom_config
        assert new_config.environment == Environment.DEVELOPMENT

    def test_load_config_from_file(self):
        """Test loading configuration from file and setting as global."""
        config_data = {
            "environment": "production",
            "debug_mode": True,
            "execution": {
                "max_retries": 5,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            loaded_config = load_config_from_file(config_file)

            # Verify the loaded config
            assert loaded_config.environment == Environment.PRODUCTION
            assert loaded_config.debug_mode is True
            assert loaded_config.execution.max_retries == 5

            # Verify it's set as global
            global_config = get_config()
            assert global_config is loaded_config
        finally:
            os.unlink(config_file)


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_environment_based_configuration(self):
        """Test that configuration adapts correctly to different environments."""
        config = ApplicationConfig()

        # Test development environment
        config.environment = Environment.DEVELOPMENT
        config.execution.timeout_seconds = 10
        assert config.get_timeout_for_environment() == 10

        # Test testing environment with multiplier
        config.environment = Environment.TESTING
        config.testing.test_timeout_multiplier = 1.5
        assert config.get_timeout_for_environment() == 15

        # Test production environment
        config.environment = Environment.PRODUCTION
        assert config.get_timeout_for_environment() == 10

    def test_configuration_persistence_roundtrip(self):
        """Test that configuration can be saved and loaded without data loss."""
        original_config = ApplicationConfig()
        original_config.environment = Environment.TESTING
        original_config.log_level = LogLevel.DEBUG
        original_config.debug_mode = True
        original_config.execution.max_retries = 7
        original_config.execution.enable_simulation_delay = True
        original_config.files.question_truncate_length = 25
        original_config.models.temperature = 0.3

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = f.name

        try:
            # Save and reload
            original_config.save_to_file(config_file)
            loaded_config = ApplicationConfig.from_file(config_file)

            # Verify all settings are preserved
            assert loaded_config.environment == Environment.TESTING
            assert loaded_config.log_level == LogLevel.DEBUG
            assert loaded_config.debug_mode is True
            assert loaded_config.execution.max_retries == 7
            assert loaded_config.execution.enable_simulation_delay is True
            assert loaded_config.files.question_truncate_length == 25
            assert loaded_config.models.temperature == 0.3
        finally:
            os.unlink(config_file)

    def test_environment_variable_precedence(self):
        """Test that environment variables take precedence over defaults."""
        # Test with environment variables set
        with patch.dict(
            os.environ,
            {
                "COGNIVAULT_ENV": "production",
                "COGNIVAULT_MAX_RETRIES": "15",
                "COGNIVAULT_NOTES_DIR": "/production/notes",
            },
        ):
            config = ApplicationConfig.from_env()
            assert config.environment == Environment.PRODUCTION
            assert config.execution.max_retries == 15
            assert config.files.notes_directory == "/production/notes"

        # Test that removing environment variables reverts to defaults
        config_default = ApplicationConfig.from_env()
        assert config_default.environment == Environment.DEVELOPMENT
        assert config_default.execution.max_retries == 3
        assert config_default.files.notes_directory == "./src/cognivault/notes"
