"""
Application Configuration Management for CogniVault.

This module provides centralized configuration management for all application
constants, timeouts, paths, and operational parameters previously scattered
throughout the codebase as magic numbers.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


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


class ExecutionConfig(BaseModel):
    """Configuration for agent execution and orchestration."""

    # Timeout and retry settings
    max_retries: int = Field(
        3,
        description="Maximum number of retry attempts for agent execution",
        ge=0,
        le=100,  # More flexible for tests
        json_schema_extra={"example": 3},
    )
    timeout_seconds: int = Field(
        10,
        description="Timeout in seconds for agent execution",
        gt=0,
        le=300,
        json_schema_extra={"example": 10},
    )
    retry_delay_seconds: float = Field(
        1.0,
        description="Delay in seconds between retry attempts",
        ge=0.0,
        le=60.0,
        json_schema_extra={"example": 1.0},
    )

    # Agent execution settings
    enable_simulation_delay: bool = Field(
        False,
        description="Whether to enable artificial delays for simulation",
        json_schema_extra={"example": False},
    )
    simulation_delay_seconds: float = Field(
        0.1,
        description="Delay in seconds for simulation mode",
        ge=0.0,
        le=10.0,
        json_schema_extra={"example": 0.1},
    )

    # Default agent pipeline
    default_agents: List[str] = Field(
        default_factory=lambda: ["refiner", "historian", "synthesis"],
        description="Default list of agents to execute in the pipeline",
        min_length=1,
        json_schema_extra={"example": ["refiner", "historian", "synthesis"]},
    )
    critic_enabled: bool = Field(
        True,
        description="Whether the critic agent is enabled",
        json_schema_extra={"example": True},
    )

    @field_validator("default_agents")
    @classmethod
    def validate_default_agents(cls, v: List[str]) -> List[str]:
        """Validate default agents list."""
        if not v:
            raise ValueError("default_agents cannot be empty")
        for agent in v:
            if not agent.strip():
                raise ValueError("Agent names cannot be empty or whitespace")
        return [agent.strip() for agent in v]

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=False,  # Allow tests to set invalid values
    )


class FileConfig(BaseModel):
    """Configuration for file handling and storage."""

    # Output directories
    notes_directory: str = Field(
        "./src/cognivault/notes",
        description="Directory path for storing note files",
        min_length=1,
        json_schema_extra={"example": "./src/cognivault/notes"},
    )
    logs_directory: str = Field(
        "./src/cognivault/logs",
        description="Directory path for storing log files",
        min_length=1,
        json_schema_extra={"example": "./src/cognivault/logs"},
    )

    # Filename generation
    question_truncate_length: int = Field(
        40,
        description="Maximum length for question text in filenames",
        gt=0,
        le=200,
        json_schema_extra={"example": 40},
    )
    hash_length: int = Field(
        6,
        description="Length of hash suffix for filename uniqueness",
        gt=0,
        le=32,
        json_schema_extra={"example": 6},
    )
    filename_separator: str = Field(
        "_",
        description="Character used to separate filename components",
        min_length=1,
        max_length=5,
        json_schema_extra={"example": "_"},
    )

    # File size limits (in bytes)
    max_file_size: int = Field(
        10 * 1024 * 1024,  # 10MB
        description="Maximum file size in bytes",
        gt=0,
        le=1024 * 1024 * 1024,  # 1GB limit
        json_schema_extra={"example": 10485760},
    )
    max_note_files: int = Field(
        1000,
        description="Maximum number of note files to maintain",
        gt=0,
        le=100000,
        json_schema_extra={"example": 1000},
    )

    @field_validator("notes_directory", "logs_directory")
    @classmethod
    def validate_directory_paths(cls, v: str) -> str:
        """Validate directory path format."""
        if not v.strip():
            raise ValueError("Directory path cannot be empty")
        # Allow relative and absolute paths
        return v.strip()

    @field_validator("filename_separator")
    @classmethod
    def validate_filename_separator(cls, v: str) -> str:
        """Validate filename separator character."""
        if not v.strip():
            raise ValueError("Filename separator cannot be empty")
        # Ensure it's filesystem-safe
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        if any(char in v for char in invalid_chars):
            raise ValueError(
                f"Filename separator cannot contain: {', '.join(invalid_chars)}"
            )
        return v.strip()

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=False,  # Allow tests to set invalid values
    )


class ModelConfig(BaseModel):
    """Configuration for LLM models and providers."""

    # Default model settings
    default_provider: str = Field(
        "openai",
        description="Default LLM provider to use",
        min_length=1,
        json_schema_extra={"example": "openai"},
    )
    default_model: str = Field(
        "gpt-4",
        description="Default model name to use",
        min_length=1,
        json_schema_extra={"example": "gpt-4"},
    )

    # Token limits and processing
    max_tokens_per_request: int = Field(
        4096,
        description="Maximum tokens per API request",
        gt=0,
        le=128000,  # Current max for most models
        json_schema_extra={"example": 4096},
    )
    temperature: float = Field(
        0.7,
        description="Temperature setting for model creativity",
        ge=0.0,
        le=2.0,
        json_schema_extra={"example": 0.7},
    )

    # Stub/Mock settings for testing
    mock_tokens_used: int = Field(
        10,
        description="Mock token count for testing",
        ge=0,
        le=10000,
        json_schema_extra={"example": 10},
    )
    mock_response_truncate_length: int = Field(
        50,
        description="Length to truncate mock responses for testing",
        gt=0,
        le=1000,
        json_schema_extra={"example": 50},
    )

    @field_validator("default_provider", "default_model")
    @classmethod
    def validate_string_fields(cls, v: str) -> str:
        """Validate string fields are not empty."""
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v.strip()

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=False,  # Allow tests to set invalid values
    )


class DevelopmentConfig(BaseModel):
    """Configuration for testing and development."""

    # Test timeouts and delays
    test_timeout_multiplier: float = Field(
        1.5,  # Multiply base timeout for tests
        description="Multiplier for timeouts in testing environment",
        gt=0.0,
        le=10.0,
        json_schema_extra={"example": 1.5},
    )
    test_simulation_enabled: bool = Field(
        True,
        description="Whether test simulation mode is enabled",
        json_schema_extra={"example": True},
    )

    # Test data generation
    mock_history_entries: List[str] = Field(
        default_factory=lambda: [
            "Note from 2024-10-15: Mexico had a third party win the presidency.",
            "Note from 2024-11-05: Discussion on judiciary reforms in Mexico.",
            "Note from 2024-12-01: Analysis of democratic institutions and their evolution.",
        ],
        description="Mock history entries for testing",
        min_length=0,
        json_schema_extra={"example": ["Sample note 1", "Sample note 2"]},
    )

    # Coverage and quality thresholds
    prompt_min_length: int = Field(
        2000,
        description="Minimum prompt length for quality testing",
        ge=0,
        le=50000,
        json_schema_extra={"example": 2000},
    )
    prompt_max_length: int = Field(
        8000,
        description="Maximum prompt length for quality testing",
        gt=0,
        le=100000,
        json_schema_extra={"example": 8000},
    )

    # Context management settings
    max_context_size_bytes: int = Field(
        1024 * 1024,  # 1MB default
        description="Maximum context size in bytes",
        gt=0,
        le=100 * 1024 * 1024,  # 100MB limit
        json_schema_extra={"example": 1048576},
    )
    max_snapshots: int = Field(
        5,
        description="Maximum number of context snapshots to maintain",
        gt=0,
        le=100,
        json_schema_extra={"example": 5},
    )
    enable_context_compression: bool = Field(
        True,
        description="Whether context compression is enabled",
        json_schema_extra={"example": True},
    )
    context_compression_threshold: float = Field(
        0.8,  # Compress when 80% of max size
        description="Threshold for triggering context compression (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.8},
    )

    @field_validator("mock_history_entries")
    @classmethod
    def validate_mock_history_entries(cls, v: List[str]) -> List[str]:
        """Validate mock history entries."""
        return [entry.strip() for entry in v if entry.strip()]

    @field_validator("prompt_max_length")
    @classmethod
    def validate_prompt_max_length(cls, v: int, info) -> int:
        """Validate that max length is greater than min length."""
        if hasattr(info, "data") and "prompt_min_length" in info.data:
            if v <= info.data["prompt_min_length"]:
                raise ValueError(
                    "prompt_max_length must be greater than prompt_min_length"
                )
        return v

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=False,  # Allow tests to set invalid values
    )


class ApplicationConfig(BaseModel):
    """
    Main application configuration containing all subsystem configurations.

    This class serves as the central point for all configuration management,
    consolidating previously scattered magic numbers and constants.
    """

    # Environment settings
    environment: Environment = Field(
        Environment.DEVELOPMENT,
        description="Deployment environment (development, testing, production)",
        json_schema_extra={"example": "development"},
    )
    log_level: LogLevel = Field(
        LogLevel.INFO,
        description="Logging level for the application",
        json_schema_extra={"example": "INFO"},
    )
    debug_mode: bool = Field(
        False,
        description="Whether debug mode is enabled",
        json_schema_extra={"example": False},
    )

    # Subsystem configurations
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Configuration for agent execution and orchestration",
    )
    files: FileConfig = Field(
        default_factory=FileConfig,
        description="Configuration for file handling and storage",
    )
    models: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Configuration for LLM models and providers",
    )
    testing: DevelopmentConfig = Field(
        default_factory=DevelopmentConfig,
        description="Configuration for testing and development",
    )

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

        # Context management configuration
        config.testing.max_context_size_bytes = int(
            os.getenv("COGNIVAULT_MAX_CONTEXT_SIZE_BYTES", str(1024 * 1024))
        )
        config.testing.max_snapshots = int(os.getenv("COGNIVAULT_MAX_SNAPSHOTS", "5"))
        config.testing.enable_context_compression = (
            os.getenv("COGNIVAULT_ENABLE_CONTEXT_COMPRESSION", "true").lower() == "true"
        )
        config.testing.context_compression_threshold = float(
            os.getenv("COGNIVAULT_CONTEXT_COMPRESSION_THRESHOLD", "0.8")
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
                "max_context_size_bytes": self.testing.max_context_size_bytes,
                "max_snapshots": self.testing.max_snapshots,
                "enable_context_compression": self.testing.enable_context_compression,
                "context_compression_threshold": self.testing.context_compression_threshold,
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

    def validate_configuration(self) -> List[str]:
        """
        Validate configuration values and return any errors.

        Kept for backward compatibility with the original dataclass validation.
        This method replicates the original validation logic for tests.

        Returns
        -------
        List[str]
            List of validation error messages, empty if valid
        """
        errors = []

        # Validate execution config (replicating original logic)
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

    def validate_config(self) -> List[str]:
        """
        Alias for validate_configuration() method.

        Returns
        -------
        List[str]
            List of validation error messages, empty if valid
        """
        return self.validate_configuration()

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=False,  # Allow tests to set invalid values
    )


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
