"""
LangChain service implementation with structured output support.

This service implements the patterns from the LangChain structured output article:
- Direct with_structured_output() usage
- Provider-specific method selection
- Fallback to PydanticOutputParser
- Rich Pydantic validation
- Dynamic model discovery and selection
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
from cognivault.services.model_discovery_service import (
    get_model_discovery_service,
    ModelDiscoveryService,
    ModelCategory,
    ModelSpeed,
)

# Import pool for eliminating redundancy
try:
    from cognivault.services.llm_pool import LLMServicePool

    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False


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
    - Dynamic model discovery and intelligent selection
    """

    # Provider-specific method mapping from article
    PROVIDER_METHODS = {
        "gpt-5": "json_schema",  # GPT-5 has full json_schema support
        "gpt-5-mini": "json_schema",  # GPT-5 variants have json_schema
        "gpt-5-nano": "json_schema",
        "gpt-4o": "json_schema",  # GPT-4o supports json_schema
        "gpt-4o-mini": "json_schema",  # GPT-4o-mini supports json_schema
        "gpt-4-turbo": "json_schema",  # GPT-4-turbo supports json_schema
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
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_name: Optional[str] = None,
        use_discovery: bool = True,
        use_pool: bool = True,  # NEW: Use pool when available
    ):
        """
        Initialize LangChain service with model.

        Args:
            model: Model name (if None, uses discovery service or pool)
            temperature: Model temperature
            api_key: OpenAI API key
            base_url: Optional base URL for API
            agent_name: Agent name for model selection (e.g., "refiner")
            use_discovery: Whether to use model discovery service
            use_pool: Whether to use the LLMServicePool (eliminates redundancy)
        """
        self.logger = get_logger("services.langchain")
        self.agent_name = agent_name
        self.use_discovery = use_discovery
        self.use_pool = use_pool and POOL_AVAILABLE
        self.api_key = api_key
        self.temperature = temperature

        # Type declarations for pool client
        self._pool_client: Optional[BaseChatModel] = None
        self._use_pool_client = False
        self.llm: Optional[BaseChatModel] = None

        # Try to use pool first (eliminates redundancy)
        if self.use_pool and self.agent_name and not model:
            try:
                self.logger.info(f"Using LLMServicePool for {self.agent_name}")
                pool = LLMServicePool.get_instance()

                # Use async-safe method to get client (will be called later)
                self._use_pool_client = True
                self.model_name = (
                    "pooled"  # Temporary, will be set when client is created
                )

            except Exception as e:
                self.logger.warning(f"Could not use LLMServicePool: {e}, falling back")
                self._use_pool_client = False
        else:
            self._use_pool_client = False

        # Fallback to original logic if not using pool
        if not self._use_pool_client:
            # Initialize model discovery service if enabled
            self.discovery_service: Optional[ModelDiscoveryService] = None
            if use_discovery:
                try:
                    self.discovery_service = get_model_discovery_service(
                        api_key=api_key
                    )
                except Exception as e:
                    self.logger.warning(f"Could not initialize discovery service: {e}")

            # Determine model to use
            self.model_name = model or self._select_best_model()
            self.logger.info(
                f"Using model: {self.model_name} for agent: {agent_name or 'default'}"
            )

            # Create appropriate LLM instance based on provider
            self.llm = self._create_llm_instance(
                self.model_name, temperature, api_key, base_url
            )

        # Metrics tracking with proper typing
        self.metrics: Dict[str, Union[int, str]] = {
            "total_calls": 0,
            "successful_structured": 0,
            "fallback_used": 0,
            "validation_failures": 0,
            "model_selected": (
                self.model_name if hasattr(self, "model_name") else "unknown"
            ),
        }

    async def _ensure_pooled_client(self) -> None:
        """Ensure pooled client is initialized (async-safe)."""
        if self._use_pool_client and not self._pool_client:
            try:
                pool = LLMServicePool.get_instance()
                if self.agent_name:  # Type guard
                    (
                        self._pool_client,
                        self.model_name,
                    ) = await pool.get_optimal_client_for_agent(
                        self.agent_name, self.temperature
                    )
                    self.llm = self._pool_client
                else:
                    raise ValueError("agent_name is required for pooled client")
                self.logger.info(
                    f"Initialized pooled client: {self.model_name} for {self.agent_name}"
                )
            except Exception as e:
                self.logger.error(f"Failed to get pooled client: {e}")
                # Fallback to traditional client creation
                self._use_pool_client = False
                self.model_name = self._select_best_model()
                self.llm = self._create_llm_instance(
                    self.model_name, self.temperature, self.api_key, None
                )

    def _select_best_model(self) -> str:
        """Select the best available model using discovery service."""
        if not self.discovery_service or not self.agent_name:
            # Default fallback
            return "gpt-4o"

        try:
            # Check if we're already in an async context
            import asyncio

            try:
                # Try to get the current running loop
                loop = asyncio.get_running_loop()
                # We're in an async context, can't use run_until_complete
                # Use sync fallback for now
                self.logger.info(
                    f"In async context, using fallback model selection for '{self.agent_name}'"
                )
            except RuntimeError:
                # No running loop, we can create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    best_model = loop.run_until_complete(
                        self.discovery_service.get_best_model_for_agent(self.agent_name)
                    )
                    if best_model:
                        self.logger.info(
                            f"Discovery service selected '{best_model}' for agent '{self.agent_name}'"
                        )
                        return best_model
                finally:
                    loop.close()
        except Exception as e:
            self.logger.warning(f"Model discovery failed: {e}")

        # Improved fallbacks based on agent type - use models that actually exist
        agent_fallbacks = {
            "refiner": "gpt-4o-mini",  # Fast model for refinement
            "historian": "gpt-4o",  # Balanced model for search
            "critic": "gpt-4o-mini",  # Fast model for critique
            "synthesis": "gpt-4o",  # Strong model for synthesis
        }
        fallback = agent_fallbacks.get(self.agent_name, "gpt-4o")
        self.logger.info(
            f"Using fallback model '{fallback}' for agent '{self.agent_name}'"
        )
        return fallback

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

        # Ensure pooled client is ready if we're using pool
        await self._ensure_pooled_client()

        # Ensure we have an LLM instance
        if self.llm is None:
            raise ValueError("LLM instance not initialized")

        self.metrics["total_calls"] = int(self.metrics["total_calls"]) + 1
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

                self.metrics["successful_structured"] = (
                    int(self.metrics["successful_structured"]) + 1
                )

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

            self.metrics["fallback_used"] = int(self.metrics["fallback_used"]) + 1

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
            self.metrics["validation_failures"] = (
                int(self.metrics["validation_failures"]) + 1
            )
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
            assert self.llm is not None  # Type assertion - already checked above
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
        assert self.llm is not None  # Type assertion - already checked above
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
        total = int(self.metrics["total_calls"])
        successful = int(self.metrics["successful_structured"])
        fallback = int(self.metrics["fallback_used"])
        failures = int(self.metrics["validation_failures"])

        return {
            "total_calls": total,
            "success_rate": (successful / total if total > 0 else 0.0),
            "fallback_rate": (fallback / total if total > 0 else 0.0),
            "validation_failure_rate": (failures / total if total > 0 else 0.0),
            "metrics": self.metrics,
        }

    def clear_cache(self) -> None:
        """Reset service metrics."""
        self.metrics = {
            "total_calls": 0,
            "successful_structured": 0,
            "fallback_used": 0,
            "validation_failures": 0,
            "model_selected": (
                self.model_name if hasattr(self, "model_name") else "unknown"
            ),
        }
