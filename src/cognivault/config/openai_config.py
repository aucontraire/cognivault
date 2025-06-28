import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4"
    base_url: Optional[str] = None

    @classmethod
    def load(cls) -> "OpenAIConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment variables")

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
