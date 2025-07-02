"""
Application Configuration Management for CogniVault.

This module provides centralized configuration management for all application
constants, timeouts, paths, and operational parameters previously scattered
throughout the codebase as magic numbers.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from enum import Enum


class LogLevel(Enum):
    """Enumeration for log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Environment(Enum):
    """Enumeration for deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class ExecutionConfig:
    """Configuration for agent execution and orchestration."""

    # Timeout and retry settings
    max_retries: int = 3
    timeout_seconds: int = 10
    retry_delay_seconds: float = 1.0

    # Agent execution settings
    enable_simulation_delay: bool = False
    simulation_delay_seconds: float = 0.1

    # Default agent pipeline
    default_agents: List[str] = field(
        default_factory=lambda: ["refiner", "historian", "synthesis"]
    )
    critic_enabled: bool = True


@dataclass
class FileConfig:
    """Configuration for file handling and storage."""

    # Output directories
    notes_directory: str = "./src/cognivault/notes"
    logs_directory: str = "./src/cognivault/logs"

    # Filename generation
    question_truncate_length: int = 40
    hash_length: int = 6
    filename_separator: str = "_"

    # File size limits (in bytes)
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_note_files: int = 1000


@dataclass
class ModelConfig:
    """Configuration for LLM models and providers."""

    # Default model settings
    default_provider: str = "openai"
    default_model: str = "gpt-4"

    # Token limits and processing
    max_tokens_per_request: int = 4096
    temperature: float = 0.7

    # Stub/Mock settings for testing
    mock_tokens_used: int = 10
    mock_response_truncate_length: int = 50


@dataclass
class TestConfig:
    """Configuration for testing and development."""

    # Test timeouts and delays
    test_timeout_multiplier: float = 1.5  # Multiply base timeout for tests
    test_simulation_enabled: bool = True

    # Test data generation
    mock_history_entries: List[str] = field(
        default_factory=lambda: [
            "Note from 2024-10-15: Mexico had a third party win the presidency.",
            "Note from 2024-11-05: Discussion on judiciary reforms in Mexico.",
            "Note from 2024-12-01: Analysis of democratic institutions and their evolution.",
        ]
    )

    # Coverage and quality thresholds
    prompt_min_length: int = 2000
    prompt_max_length: int = 8000


@dataclass
class ApplicationConfig:
    """
    Main application configuration containing all subsystem configurations.

    This class serves as the central point for all configuration management,
    consolidating previously scattered magic numbers and constants.
    """

    # Environment settings
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False

    # Subsystem configurations
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    files: FileConfig = field(default_factory=FileConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    testing: TestConfig = field(default_factory=TestConfig)

    @classmethod
    def from_env(cls) -> "ApplicationConfig":
        """
        Create configuration from environment variables.

        Returns
        -------
        ApplicationConfig
            Configuration instance populated from environment variables
        """
        config = cls()

        # Environment settings
        env_name = os.getenv("COGNIVAULT_ENV", "development").lower()
        try:
            config.environment = Environment(env_name)
        except ValueError:
            config.environment = Environment.DEVELOPMENT

        log_level_name = os.getenv("COGNIVAULT_LOG_LEVEL", "INFO").upper()
        try:
            config.log_level = LogLevel(log_level_name)
        except ValueError:
            config.log_level = LogLevel.INFO

        config.debug_mode = os.getenv("COGNIVAULT_DEBUG", "false").lower() == "true"

        # Execution configuration
        config.execution.max_retries = int(os.getenv("COGNIVAULT_MAX_RETRIES", "3"))
        config.execution.timeout_seconds = int(
            os.getenv("COGNIVAULT_TIMEOUT_SECONDS", "10")
        )
        config.execution.retry_delay_seconds = float(
            os.getenv("COGNIVAULT_RETRY_DELAY", "1.0")
        )
        config.execution.enable_simulation_delay = (
            os.getenv("COGNIVAULT_SIMULATION_DELAY", "false").lower() == "true"
        )
        config.execution.simulation_delay_seconds = float(
            os.getenv("COGNIVAULT_SIMULATION_DELAY_SECONDS", "0.1")
        )
        config.execution.critic_enabled = (
            os.getenv("COGNIVAULT_CRITIC_ENABLED", "true").lower() == "true"
        )

        # File configuration
        config.files.notes_directory = os.getenv(
            "COGNIVAULT_NOTES_DIR", "./src/cognivault/notes"
        )
        config.files.logs_directory = os.getenv(
            "COGNIVAULT_LOGS_DIR", "./src/cognivault/logs"
        )
        config.files.question_truncate_length = int(
            os.getenv("COGNIVAULT_QUESTION_TRUNCATE", "40")
        )
        config.files.hash_length = int(os.getenv("COGNIVAULT_HASH_LENGTH", "6"))
        config.files.max_file_size = int(
            os.getenv("COGNIVAULT_MAX_FILE_SIZE", str(10 * 1024 * 1024))
        )
        config.files.max_note_files = int(
            os.getenv("COGNIVAULT_MAX_NOTE_FILES", "1000")
        )

        # Model configuration
        config.models.default_provider = os.getenv("COGNIVAULT_LLM", "openai")
        config.models.default_model = os.getenv("OPENAI_MODEL", "gpt-4")
        config.models.max_tokens_per_request = int(
            os.getenv("COGNIVAULT_MAX_TOKENS", "4096")
        )
        config.models.temperature = float(os.getenv("COGNIVAULT_TEMPERATURE", "0.7"))

        # Testing configuration
        config.testing.test_timeout_multiplier = float(
            os.getenv("COGNIVAULT_TEST_TIMEOUT_MULTIPLIER", "1.5")
        )
        config.testing.test_simulation_enabled = (
            os.getenv("COGNIVAULT_TEST_SIMULATION", "true").lower() == "true"
        )

        return config

    @classmethod
    def from_file(cls, config_path: str) -> "ApplicationConfig":
        """
        Create configuration from a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to the JSON configuration file

        Returns
        -------
        ApplicationConfig
            Configuration instance populated from file

        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist
        json.JSONDecodeError
            If the configuration file is not valid JSON
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config = cls()
        config._update_from_dict(config_data)
        return config

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        if "environment" in data:
            self.environment = Environment(data["environment"])
        if "log_level" in data:
            self.log_level = LogLevel(data["log_level"])
        if "debug_mode" in data:
            self.debug_mode = data["debug_mode"]

        if "execution" in data:
            exec_data = data["execution"]
            for key, value in exec_data.items():
                if hasattr(self.execution, key):
                    setattr(self.execution, key, value)

        if "files" in data:
            file_data = data["files"]
            for key, value in file_data.items():
                if hasattr(self.files, key):
                    setattr(self.files, key, value)

        if "models" in data:
            model_data = data["models"]
            for key, value in model_data.items():
                if hasattr(self.models, key):
                    setattr(self.models, key, value)

        if "testing" in data:
            test_data = data["testing"]
            for key, value in test_data.items():
                if hasattr(self.testing, key):
                    setattr(self.testing, key, value)

    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to a JSON file.

        Parameters
        ----------
        config_path : str
            Path where to save the configuration file
        """
        config_data = {
            "environment": self.environment.value,
            "log_level": self.log_level.value,
            "debug_mode": self.debug_mode,
            "execution": {
                "max_retries": self.execution.max_retries,
                "timeout_seconds": self.execution.timeout_seconds,
                "retry_delay_seconds": self.execution.retry_delay_seconds,
                "enable_simulation_delay": self.execution.enable_simulation_delay,
                "simulation_delay_seconds": self.execution.simulation_delay_seconds,
                "default_agents": self.execution.default_agents,
                "critic_enabled": self.execution.critic_enabled,
            },
            "files": {
                "notes_directory": self.files.notes_directory,
                "logs_directory": self.files.logs_directory,
                "question_truncate_length": self.files.question_truncate_length,
                "hash_length": self.files.hash_length,
                "filename_separator": self.files.filename_separator,
                "max_file_size": self.files.max_file_size,
                "max_note_files": self.files.max_note_files,
            },
            "models": {
                "default_provider": self.models.default_provider,
                "default_model": self.models.default_model,
                "max_tokens_per_request": self.models.max_tokens_per_request,
                "temperature": self.models.temperature,
                "mock_tokens_used": self.models.mock_tokens_used,
                "mock_response_truncate_length": self.models.mock_response_truncate_length,
            },
            "testing": {
                "test_timeout_multiplier": self.testing.test_timeout_multiplier,
                "test_simulation_enabled": self.testing.test_simulation_enabled,
                "mock_history_entries": self.testing.mock_history_entries,
                "prompt_min_length": self.testing.prompt_min_length,
                "prompt_max_length": self.testing.prompt_max_length,
            },
        }

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    def get_timeout_for_environment(self) -> int:
        """
        Get timeout value adjusted for current environment.

        Returns
        -------
        int
            Timeout in seconds, potentially adjusted for testing
        """
        base_timeout = self.execution.timeout_seconds
        if self.environment == Environment.TESTING:
            return int(base_timeout * self.testing.test_timeout_multiplier)
        return base_timeout

    def ensure_directories_exist(self) -> None:
        """Create configured directories if they don't exist."""
        Path(self.files.notes_directory).mkdir(parents=True, exist_ok=True)
        Path(self.files.logs_directory).mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """
        Validate configuration values and return any errors.

        Returns
        -------
        List[str]
            List of validation error messages, empty if valid
        """
        errors = []

        # Validate execution config
        if self.execution.max_retries < 0:
            errors.append("max_retries must be non-negative")
        if self.execution.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        if self.execution.retry_delay_seconds < 0:
            errors.append("retry_delay_seconds must be non-negative")
        if self.execution.simulation_delay_seconds < 0:
            errors.append("simulation_delay_seconds must be non-negative")

        # Validate file config
        if self.files.question_truncate_length <= 0:
            errors.append("question_truncate_length must be positive")
        if self.files.hash_length <= 0:
            errors.append("hash_length must be positive")
        if self.files.max_file_size <= 0:
            errors.append("max_file_size must be positive")
        if self.files.max_note_files <= 0:
            errors.append("max_note_files must be positive")

        # Validate model config
        if self.models.max_tokens_per_request <= 0:
            errors.append("max_tokens_per_request must be positive")
        if not 0 <= self.models.temperature <= 2:
            errors.append("temperature must be between 0 and 2")

        # Validate testing config
        if self.testing.test_timeout_multiplier <= 0:
            errors.append("test_timeout_multiplier must be positive")
        if self.testing.prompt_min_length < 0:
            errors.append("prompt_min_length must be non-negative")
        if self.testing.prompt_max_length <= self.testing.prompt_min_length:
            errors.append("prompt_max_length must be greater than prompt_min_length")

        return errors


# Global configuration instance
_global_config: Optional[ApplicationConfig] = None


def get_config() -> ApplicationConfig:
    """
    Get the global application configuration instance.

    Returns
    -------
    ApplicationConfig
        The global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ApplicationConfig.from_env()
    return _global_config


def set_config(config: ApplicationConfig) -> None:
    """
    Set the global application configuration instance.

    Parameters
    ----------
    config : ApplicationConfig
        The configuration instance to set as global
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to default values."""
    global _global_config
    _global_config = None


def load_config_from_file(config_path: str) -> ApplicationConfig:
    """
    Load configuration from file and set it as global.

    Parameters
    ----------
    config_path : str
        Path to the configuration file

    Returns
    -------
    ApplicationConfig
        The loaded configuration instance
    """
    config = ApplicationConfig.from_file(config_path)
    set_config(config)
    return config
