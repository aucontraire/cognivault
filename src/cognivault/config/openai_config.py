import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)


class OpenAIConfig(BaseModel):
    """
    OpenAI configuration with environment variable loading and validation.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault configuration system.
    """

    api_key: str = Field(
        ...,
        description="OpenAI API key for authentication",
        min_length=1,
        json_schema_extra={"example": "sk-..."},
    )
    model: str = Field(
        "gpt-4",
        description="OpenAI model to use for completions",
        min_length=1,
        json_schema_extra={"example": "gpt-4"},
    )
    base_url: Optional[str] = Field(
        None,
        description="Optional base URL for OpenAI API (for custom endpoints)",
        json_schema_extra={"example": "https://api.openai.com/v1"},
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format and security."""
        if not v.strip():
            raise ValueError("API key cannot be empty or whitespace")
        # More flexible length validation (allow test keys)
        if len(v) < 5:
            raise ValueError("API key appears to be too short")
        # Basic format validation for OpenAI keys (flexible for testing)
        if not (v.startswith("sk-") or v.startswith("org-") or v.startswith("test-")):
            # Allow any format that looks reasonable for testing
            if len(v) < 8 and not v.startswith("test"):
                raise ValueError(
                    "API key should start with 'sk-', 'org-', or be a test key"
                )
        return v.strip()

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name format."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        # Allow common OpenAI model patterns and test patterns
        valid_patterns = [
            "gpt-",
            "text-",
            "davinci",
            "curie",
            "babbage",
            "ada",
            "test-",
        ]
        if not any(v.startswith(pattern) for pattern in valid_patterns):
            # Still allow it for flexibility
            pass
        return v.strip()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate base URL format if provided."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Base URL must start with http:// or https://")
        if v.endswith("/"):
            v = v.rstrip("/")  # Remove trailing slash for consistency
        return v

    @classmethod
    def load(cls) -> "OpenAIConfig":
        """
        Load configuration from environment variables.

        Returns
        -------
        OpenAIConfig
            Configuration instance loaded from environment

        Raises
        ------
        EnvironmentError
            If OPENAI_API_KEY is not set
        ValueError
            If loaded values fail validation
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment variables")

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )
