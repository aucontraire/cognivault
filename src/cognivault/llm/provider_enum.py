from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    STUB = "stub"


class LLMModel(str, Enum):
    """Supported LLM models with type safety."""

    # OpenAI models
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # Future models (for extensibility)
    CLAUDE_OPUS = "claude-3-opus"
    CLAUDE_SONNET = "claude-3-sonnet"
    CLAUDE_HAIKU = "claude-3-haiku"
    MISTRAL_7B = "mistral-7b"
    LLAMA_3 = "llama-3"

    # Stub/testing
    STUB = "stub"
    LOCAL_CUSTOM = "local-custom"
