"""
Structured LLM calls using Pydantic AI for validated responses.

This module provides wrappers around the existing LLM interface that use
Pydantic AI to ensure structured, validated responses from language models.
"""

import asyncio
import time
from typing import Optional, Type, TypeVar, Dict, Any, Callable, Union
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .llm_interface import LLMInterface, LLMResponse
from cognivault.observability import get_logger, get_observability_context
from cognivault.exceptions import LLMError, LLMValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredLLMResponse(BaseModel):
    """Wrapper for structured LLM responses with metadata."""

    content: BaseModel
    raw_response: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    processing_time_ms: float
    model_used: str
    validation_success: bool = True
    validation_errors: Optional[list[str]] = None


class PydanticAIWrapper:
    """
    Wrapper around existing LLM interface that provides structured responses.

    This class uses Pydantic AI to ensure that LLM responses conform to
    specified Pydantic models, enabling structured data extraction.
    """

    def __init__(self, llm_interface: LLMInterface) -> None:
        """
        Initialize the Pydantic AI wrapper.

        Args:
            llm_interface: The underlying LLM interface to wrap
        """
        self.llm_interface = llm_interface
        self.logger = get_logger("llm.structured")

        # Extract model info for Pydantic AI
        if hasattr(llm_interface, "model") and llm_interface.model:
            self.model = OpenAIModel(llm_interface.model)
        else:
            self.model = OpenAIModel("gpt-4")  # Default fallback

        # Cache agents to avoid recreation
        self._agent_cache: Dict[str, Agent] = {}

    def _get_or_create_agent(
        self, response_model: Type[T], system_prompt: Optional[str] = None
    ) -> Agent:
        """Get or create a cached Pydantic AI agent for the given model."""
        cache_key = f"{response_model.__name__}_{hash(system_prompt or '')}"

        if cache_key not in self._agent_cache:
            # Pydantic AI Agent expects system_prompt to be str | Sequence[str], not Optional[str]
            agent_system_prompt = system_prompt if system_prompt is not None else ""
            self._agent_cache[cache_key] = Agent(
                model=self.model,
                result_type=response_model,
                system_prompt=agent_system_prompt,
            )

        return self._agent_cache[cache_key]

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_prompt: Optional[str] = None,
        on_log: Optional[Callable[[str], None]] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> StructuredLLMResponse:
        """
        Generate a structured response using Pydantic AI.

        Args:
            prompt: The user prompt to send to the LLM
            response_model: Pydantic model class for the expected response structure
            system_prompt: Optional system prompt for context
            on_log: Optional logging callback
            max_retries: Number of retries for validation failures
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            StructuredLLMResponse containing the validated response and metadata

        Raises:
            LLMError: If the LLM call fails
            LLMValidationError: If validation fails after all retries
        """
        start_time = time.time()
        obs_context = get_observability_context()

        if on_log:
            on_log(
                f"[StructuredLLM] Generating structured response for {response_model.__name__}"
            )
            on_log(f"[StructuredLLM] Prompt: {prompt[:100]}...")

        # Get or create the Pydantic AI agent
        agent = self._get_or_create_agent(response_model, system_prompt)

        validation_errors = []

        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(
                    f"Structured LLM call attempt {attempt + 1}/{max_retries + 1}",
                    model=str(self.model),
                    response_model=response_model.__name__,
                    prompt_length=len(prompt),
                    agent_name=obs_context.agent_name if obs_context else None,
                )

                # Run the Pydantic AI agent
                result = await agent.run(prompt)

                processing_time_ms = (time.time() - start_time) * 1000

                # Extract metadata if available
                tokens_used = None
                if hasattr(result, "usage"):
                    usage = getattr(result, "usage")
                    if callable(usage):
                        # If usage is a function, call it to get the actual usage data
                        try:
                            usage_data = usage()
                            tokens_used = (
                                usage_data.get("total_tokens")
                                if isinstance(usage_data, dict)
                                else None
                            )
                        except Exception:
                            tokens_used = None
                    elif isinstance(usage, dict):
                        # If usage is already a dict, get total_tokens directly
                        tokens_used = usage.get("total_tokens")
                    # Otherwise, leave tokens_used as None
                cost_usd = (
                    self._estimate_cost(tokens_used, str(self.model))
                    if tokens_used
                    else None
                )

                if on_log:
                    on_log(
                        f"[StructuredLLM] Successfully generated {response_model.__name__} in {processing_time_ms:.2f}ms"
                    )

                self.logger.info(
                    f"Structured LLM call successful",
                    model=str(self.model),
                    response_model=response_model.__name__,
                    processing_time_ms=processing_time_ms,
                    tokens_used=tokens_used,
                    agent_name=obs_context.agent_name if obs_context else None,
                )

                return StructuredLLMResponse(
                    content=result.data,
                    raw_response=getattr(result, "raw_response", None),
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    processing_time_ms=processing_time_ms,
                    model_used=str(self.model),
                    validation_success=True,
                )

            except Exception as e:
                error_msg = str(e)
                validation_errors.append(f"Attempt {attempt + 1}: {error_msg}")

                self.logger.warning(
                    f"Structured LLM call attempt {attempt + 1} failed: {error_msg}",
                    model=str(self.model),
                    response_model=response_model.__name__,
                    agent_name=obs_context.agent_name if obs_context else None,
                )

                if on_log:
                    on_log(f"[StructuredLLM] Attempt {attempt + 1} failed: {error_msg}")

                # If this is the last attempt, raise the error
                if attempt == max_retries:
                    processing_time_ms = (time.time() - start_time) * 1000

                    # Check if it's a validation error or an LLM error
                    if (
                        "validation" in error_msg.lower()
                        or "parse" in error_msg.lower()
                    ):
                        raise LLMValidationError(
                            message=f"Failed to validate {response_model.__name__} after {max_retries + 1} attempts",
                            model_name=str(self.model),
                            validation_errors=validation_errors,
                            context={
                                "response_model": response_model.__name__,
                                "prompt_length": len(prompt),
                                "processing_time_ms": processing_time_ms,
                            },
                        )
                    else:
                        # Re-raise as LLM error
                        raise LLMError(
                            message=f"Structured LLM call failed: {error_msg}",
                            llm_provider="openai",
                            context={"attempts_made": attempt + 1},
                        )

                # Wait a bit before retrying
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # This should never be reached due to exception handling above, but added for MyPy
        raise LLMError(
            message="Structured LLM call completed without success or exception",
            llm_provider="openai",
            context={"max_retries": max_retries},
        )

    def generate_structured_sync(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_prompt: Optional[str] = None,
        on_log: Optional[Callable[[str], None]] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> StructuredLLMResponse:
        """
        Synchronous version of generate_structured.

        This is a convenience wrapper around the async version for use in
        synchronous contexts.
        """
        return asyncio.run(
            self.generate_structured(
                prompt=prompt,
                response_model=response_model,
                system_prompt=system_prompt,
                on_log=on_log,
                max_retries=max_retries,
                **kwargs,
            )
        )

    def _estimate_cost(self, tokens: Optional[int], model: str) -> Optional[float]:
        """
        Estimate the cost of an LLM call based on tokens and model.

        Args:
            tokens: Number of tokens used
            model: Model name

        Returns:
            Estimated cost in USD, or None if cannot estimate
        """
        if not tokens:
            return None

        # Simple cost estimation - in production, this would use more accurate rates
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
        }

        # Get base model name (remove any suffixes)
        base_model = model.lower()
        for known_model in cost_per_1k_tokens:
            if known_model in base_model:
                return (tokens / 1000) * cost_per_1k_tokens[known_model]

        # Default fallback
        return (tokens / 1000) * 0.01

    def clear_cache(self) -> None:
        """Clear the agent cache."""
        self._agent_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent cache."""
        return {
            "cached_agents": len(self._agent_cache),
            "cache_keys": list(self._agent_cache.keys()),
        }


class StructuredLLMFactory:
    """Factory for creating structured LLM wrappers."""

    @staticmethod
    def create_from_llm(llm_interface: LLMInterface) -> PydanticAIWrapper:
        """Create a structured LLM wrapper from an existing LLM interface."""
        return PydanticAIWrapper(llm_interface)

    @staticmethod
    def create_openai_structured(
        api_key: str, model: str = "gpt-4", base_url: Optional[str] = None
    ) -> PydanticAIWrapper:
        """Create a structured LLM wrapper directly from OpenAI parameters."""
        # Import here to avoid circular imports
        from .openai import OpenAIChatLLM

        llm = OpenAIChatLLM(api_key=api_key, model=model, base_url=base_url)
        return PydanticAIWrapper(llm)
