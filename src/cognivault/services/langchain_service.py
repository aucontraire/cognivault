"""
LangChain service implementation with structured output support.

This service implements the patterns from the LangChain structured output article:
- Direct with_structured_output() usage
- Provider-specific method selection
- Fallback to PydanticOutputParser
- Rich Pydantic validation
"""

import json
import asyncio
from typing import Type, TypeVar, Optional, Dict, Any, Union, cast, List
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from cognivault.observability import get_logger, get_observability_context
from cognivault.exceptions import LLMError, LLMValidationError


T = TypeVar("T", bound=BaseModel)


class StructuredOutputResult:
    """Wrapper for structured output with debugging info."""

    def __init__(
        self,
        parsed: BaseModel,
        raw: Optional[str] = None,
        method_used: Optional[str] = None,
        fallback_used: bool = False,
        processing_time_ms: Optional[float] = None,
    ):
        self.parsed = parsed
        self.raw = raw
        self.method_used = method_used
        self.fallback_used = fallback_used
        self.processing_time_ms = processing_time_ms


class LangChainService:
    """
    LangChain service with structured output capabilities.

    Implements patterns from the LangChain structured output article:
    - Uses with_structured_output() as primary method
    - Provider-specific optimizations (json_schema, function_calling, etc.)
    - Fallback to PydanticOutputParser for models that don't support structured output
    - Rich validation and error handling
    """

    # Provider-specific method mapping from article
    PROVIDER_METHODS = {
        "gpt-5": "json_schema",  # GPT-5 has full json_schema support
        "gpt-4o": "json_schema",
        "gpt-4": "function_calling",  # GPT-4 does NOT support json_schema
        "gpt-3.5": "function_calling",
        "claude-3": "function_calling",
        "claude-2": "function_calling",
        "gemini": "json_mode",
        "llama": "json_mode",
        "mistral": "json_mode",
    }

    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize LangChain service with model."""
        self.model_name = model
        self.logger = get_logger("services.langchain")

        # Create appropriate LLM instance based on provider
        self.llm = self._create_llm_instance(model, temperature, api_key, base_url)

        # Metrics tracking
        self.metrics = {
            "total_calls": 0,
            "successful_structured": 0,
            "fallback_used": 0,
            "validation_failures": 0,
        }

    def _create_llm_instance(
        self,
        model: str,
        temperature: float,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseChatModel:
        """Create appropriate LLM instance based on model name."""
        model_lower = model.lower()

        if "gpt" in model_lower or "o1" in model_lower:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
            )
        elif "claude" in model_lower:
            # Note: Would need anthropic API key configuration
            if api_key is None:
                api_key = "test-key-for-mock"  # Allow mock testing without real API key
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        else:
            # Default to OpenAI for unknown models
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
            )

    def _get_structured_output_method(self, model: str) -> Optional[str]:
        """Get the optimal structured output method for the model."""
        model_lower = model.lower()

        for model_prefix, method in self.PROVIDER_METHODS.items():
            if model_prefix in model_lower:
                return method

        # Default fallback
        return "json_mode"

    async def get_structured_output(
        self,
        prompt: str,
        output_class: Type[T],
        *,
        include_raw: bool = False,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
    ) -> Union[T, StructuredOutputResult]:
        """
        Get structured output using LangChain's with_structured_output().

        Implements the exact pattern from the article:
        1. Try native structured output with provider-specific method
        2. Fallback to PydanticOutputParser if native fails
        3. Return parsed result or full result with debugging info

        Args:
            prompt: User prompt
            output_class: Pydantic model class for output structure
            include_raw: Whether to include raw response for debugging
            max_retries: Number of retry attempts
            system_prompt: Optional system prompt

        Returns:
            Either the parsed Pydantic model or StructuredOutputResult with debug info
        """
        import time

        start_time = time.time()

        self.metrics["total_calls"] += 1
        obs_context = get_observability_context()

        # Build messages
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))

        # Try native structured output first (article's primary approach)
        for attempt in range(max_retries):
            try:
                result = await self._try_native_structured_output(
                    messages, output_class, include_raw
                )

                processing_time_ms = (time.time() - start_time) * 1000

                self.metrics["successful_structured"] += 1

                self.logger.info(
                    "Structured output successful",
                    model=self.model_name,
                    output_class=output_class.__name__,
                    method="native",
                    attempt=attempt + 1,
                    processing_time_ms=processing_time_ms,
                    agent_name=obs_context.agent_name if obs_context else None,
                )

                if include_raw:
                    return StructuredOutputResult(
                        parsed=result["parsed"] if isinstance(result, dict) else result,
                        raw=result.get("raw") if isinstance(result, dict) else None,
                        method_used=self._get_structured_output_method(self.model_name),
                        fallback_used=False,
                        processing_time_ms=processing_time_ms,
                    )

                return result["parsed"] if isinstance(result, dict) else result

            except Exception as e:
                self.logger.warning(
                    f"Native structured output attempt {attempt + 1} failed",
                    error=str(e),
                    model=self.model_name,
                    output_class=output_class.__name__,
                )

                if attempt == max_retries - 1:
                    # Last attempt failed, try fallback
                    break

                # Wait before retry
                await asyncio.sleep(0.5 * (attempt + 1))

        # Fallback to PydanticOutputParser (article's fallback strategy)
        try:
            result = await self._fallback_to_parser(messages, output_class, include_raw)

            processing_time_ms = (time.time() - start_time) * 1000

            self.metrics["fallback_used"] += 1

            self.logger.info(
                "Fallback parser successful",
                model=self.model_name,
                output_class=output_class.__name__,
                method="parser",
                processing_time_ms=processing_time_ms,
            )

            if include_raw:
                return StructuredOutputResult(
                    parsed=result,
                    raw=getattr(result, "_raw_response", None),
                    method_used="parser",
                    fallback_used=True,
                    processing_time_ms=processing_time_ms,
                )

            return result

        except Exception as e:
            self.metrics["validation_failures"] += 1
            processing_time_ms = (time.time() - start_time) * 1000

            raise LLMValidationError(
                message=f"Failed to get structured output for {output_class.__name__} after all attempts",
                model_name=self.model_name,
                validation_errors=[str(e)],
                context={
                    "output_class": output_class.__name__,
                    "max_retries": max_retries,
                    "processing_time_ms": processing_time_ms,
                    "fallback_attempted": True,
                },
            )

    async def _try_native_structured_output(
        self,
        messages: List[tuple[str, str]],
        output_class: Type[T],
        include_raw: bool = False,
    ) -> Union[T, Dict[str, Any]]:
        """Try native structured output with provider-specific method."""

        # Get provider-specific method (key insight from article)
        method = self._get_structured_output_method(self.model_name)

        # Create structured LLM (article's core pattern)
        try:
            structured_llm = self.llm.with_structured_output(
                output_class,
                method=method,
                include_raw=include_raw,
            )

            # Invoke with messages (article's pattern)
            result = await structured_llm.ainvoke(messages)

            return cast(Union[T, Dict[str, Any]], result)

        except AttributeError as e:
            # Model doesn't support with_structured_output
            raise LLMError(
                message=f"Model {self.model_name} doesn't support with_structured_output: {e}",
                llm_provider=(
                    self.model_name.split("-")[0]
                    if "-" in self.model_name
                    else "unknown"
                ),
                context={"method": method},
            )

    async def _fallback_to_parser(
        self,
        messages: List[tuple[str, str]],
        output_class: Type[T],
        include_raw: bool = False,
    ) -> T:
        """Fallback to PydanticOutputParser (article's fallback strategy)."""

        # Create parser (article's pattern)
        parser: PydanticOutputParser[T] = PydanticOutputParser(
            pydantic_object=output_class
        )

        # Enhance prompt with format instructions (key from article)
        original_prompt = messages[-1][1]  # Get human message
        enhanced_prompt = f"{original_prompt}\n\n{parser.get_format_instructions()}"

        # Replace the human message with enhanced version
        enhanced_messages = messages[:-1] + [("human", enhanced_prompt)]

        # Get raw response
        response = await self.llm.ainvoke(enhanced_messages)

        # Parse response content (article's pattern)
        try:
            # Ensure content is a string for parser
            content_str = str(response.content) if response.content is not None else ""
            result = parser.parse(content_str)

            # Store raw response for debugging if requested
            if include_raw:
                setattr(result, "_raw_response", content_str)

            return result

        except Exception as e:
            raise LLMValidationError(
                message=f"Failed to parse response into {output_class.__name__}",
                model_name=self.model_name,
                validation_errors=[str(e)],
                context={
                    "raw_response": content_str[:500],  # Truncated for logging
                    "parser_instructions": parser.get_format_instructions(),
                },
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get service usage statistics."""
        total = self.metrics["total_calls"]
        return {
            "total_calls": total,
            "success_rate": (
                self.metrics["successful_structured"] / total if total > 0 else 0.0
            ),
            "fallback_rate": (
                self.metrics["fallback_used"] / total if total > 0 else 0.0
            ),
            "validation_failure_rate": (
                self.metrics["validation_failures"] / total if total > 0 else 0.0
            ),
            "metrics": self.metrics,
        }

    def clear_cache(self) -> None:
        """Reset service metrics."""
        self.metrics = {
            "total_calls": 0,
            "successful_structured": 0,
            "fallback_used": 0,
            "validation_failures": 0,
        }
