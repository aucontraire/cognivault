from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    STUB = "stub"
