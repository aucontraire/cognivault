from cognivault.llm.llm_interface import LLMInterface
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.stub import StubLLM
from cognivault.config.openai_config import OpenAIConfig
from cognivault.llm.provider_enum import LLMProvider

import os
from typing import Optional


class LLMFactory:
    @staticmethod
    def create(llm_name: Optional[LLMProvider] = None) -> LLMInterface:
        llm_name = llm_name or LLMProvider(
            os.getenv("COGNIVAULT_LLM", "openai").lower()
        )

        if llm_name == LLMProvider.OPENAI:
            config = OpenAIConfig.load()
            return OpenAIChatLLM(
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url,
            )
        elif llm_name == LLMProvider.STUB:
            return StubLLM()
        else:
            raise ValueError(f"Unsupported LLM type: {llm_name}")
