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

    # Provider-specific method mapping - SIMPLIFIED for base models
    PROVIDER_METHODS = {
        "gpt-5": "json_schema",  # Base GPT-5 has full json_schema support
        "gpt-5-mini": "json_schema",  # Mini is still a base model, keep it
        "gpt-5-nano": "json_schema",  # Nano is still a base model, keep it
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

        # SIMPLIFIED fallbacks - prefer base GPT-5 when available
        # User wisdom: "just use the model: 'gpt-5'"
        agent_fallbacks = {
            "refiner": "gpt-5-nano",  # Keep nano for ultra-fast requirement
            "historian": "gpt-5",  # Base GPT-5 for historian
            "critic": "gpt-5",  # Base GPT-5 for critic
            "synthesis": "gpt-5",  # Base GPT-5 for synthesis
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
            # Build kwargs for ChatOpenAI
            kwargs: Dict[str, Any] = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
            }

            # CRITICAL FIX: GPT-5 models only support temperature=1 (default)
            # Exclude temperature parameter for GPT-5 to avoid API constraint errors
            if "gpt-5" not in model_lower:
                kwargs["temperature"] = temperature
            else:
                self.logger.info(
                    f"Excluding temperature parameter for {model} (GPT-5 only supports default temperature=1)"
                )

            # CRITICAL FIX: GPT-5 models require max_completion_tokens instead of max_tokens
            # Transform parameter for GPT-5 models to avoid "Unsupported parameter" errors
            if "max_tokens" in kwargs and "gpt-5" in model_lower:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                self.logger.info(
                    f"Transformed max_tokens → max_completion_tokens for {model} (GPT-5 parameter requirement)"
                )

            # CRITICAL FIX: Removing output_version for now as it breaks the endpoint
            # The native OpenAI parse() method will handle GPT-5 structured outputs
            # if "gpt-5" in model_lower:
            #     kwargs["output_version"] = "responses/v1"  # This causes /v1/responses endpoint issue

            return ChatOpenAI(**kwargs)  # Uses kwargs from if-branch above
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
            openai_kwargs: Dict[str, Any] = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
            }

            # CRITICAL FIX: GPT-5 models only support temperature=1 (default)
            # Exclude temperature parameter for GPT-5 to avoid API constraint errors
            if "gpt-5" not in model_lower:
                openai_kwargs["temperature"] = temperature
            else:
                self.logger.info(
                    f"Excluding temperature parameter for {model} (GPT-5 only supports default temperature=1)"
                )

            # CRITICAL FIX: GPT-5 models require max_completion_tokens instead of max_tokens
            # Transform parameter for GPT-5 models to avoid "Unsupported parameter" errors
            if "max_tokens" in openai_kwargs and "gpt-5" in model_lower:
                openai_kwargs["max_completion_tokens"] = openai_kwargs.pop("max_tokens")
                self.logger.info(
                    f"Transformed max_tokens → max_completion_tokens for {model} (GPT-5 parameter requirement)"
                )

            # CRITICAL FIX: Removing output_version for now as it breaks the endpoint
            # The native OpenAI parse() method will handle GPT-5 structured outputs
            # if "gpt-5" in model_lower:
            #     openai_kwargs["output_version"] = "responses/v1"  # This causes /v1/responses endpoint issue

            return ChatOpenAI(**openai_kwargs)

    def _get_structured_output_method(self, model: str) -> Optional[str]:
        """Get the optimal structured output method for the model.

        ENHANCED: Handles model variants with different capabilities:
        - Base GPT-5 models: json_schema
        - Timestamped GPT-5 (2025-08-07): function_calling
        - Chat variants: json_mode (no structured output)
        """
        model_lower = model.lower()

        # CRITICAL FIX: Handle timestamped GPT-5 versions that require function_calling
        if "gpt-5" in model_lower and "2025-08" in model_lower:
            self.logger.warning(
                f"Model {model} is a timestamped GPT-5 variant requiring function_calling method"
            )
            return "function_calling"

        # Handle chat variants that don't support structured outputs
        if "-chat" in model_lower:
            self.logger.warning(
                f"Model {model} is a chat variant with limited structured output support, using json_mode"
            )
            return "json_mode"

        # For base GPT-5 models, use json_schema
        if model_lower in ["gpt-5", "gpt-5-nano", "gpt-5-mini"]:
            self.logger.info(f"Using json_schema for base model {model}")
            return "json_schema"

        # Standard lookup for other models
        for model_prefix, method in self.PROVIDER_METHODS.items():
            if model_prefix in model_lower:
                return method

        # Default fallback for unknown models
        self.logger.info(
            f"Unknown model {model}, using json_mode as fallback. "
            "Consider using base GPT-5 model for best results."
        )
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
                    messages, output_class, include_raw, attempt, start_time
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
                error_context = getattr(e, "context", {})
                error_type = error_context.get("error_type", "unknown")
                fallback_recommended = error_context.get("fallback_recommended", True)

                self.logger.warning(
                    f"Native structured output attempt {attempt + 1} failed",
                    error=str(e),
                    error_type=error_type,
                    model=self.model_name,
                    output_class=output_class.__name__,
                )

                # Smart fallback decision based on error type
                if error_type == "quota_exceeded":
                    # Don't retry on quota errors - fail fast
                    self.logger.error(
                        "API quota exceeded - failing immediately without retries"
                    )
                    raise e
                elif error_type == "schema_validation":
                    # Schema errors won't be fixed by retries, skip to fallback
                    self.logger.info(
                        "Schema validation error detected - skipping retries, going to fallback parser"
                    )
                    break
                elif not fallback_recommended:
                    # Error analysis suggests fallback won't help
                    raise e

                if attempt == max_retries - 1:
                    # Last attempt failed, try fallback
                    break

                # Progressive backoff with jitter for rate limiting
                base_delay = 0.5 * (attempt + 1)
                jitter = 0.1 * attempt  # Add small jitter to avoid thundering herd
                await asyncio.sleep(base_delay + jitter)

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
        attempt: int = 0,
        start_time: Optional[float] = None,
    ) -> Union[T, Dict[str, Any]]:
        """Try native structured output with provider-specific method.

        Args:
            messages: Chat messages
            output_class: Expected output structure
            include_raw: Whether to include raw response
            attempt: Current retry attempt number for timeout adjustment
            start_time: Start time for timeout budget calculation
        """
        import time

        if start_time is None:
            start_time = time.time()

        # CRITICAL FIX: OpenAI beta.chat.completions.parse has bugs returning None
        # Testing shows LangChain's json_schema method works reliably for GPT-5
        # Skip the buggy native parse API and use proven LangChain implementation
        if False and "gpt-5" in self.model_name.lower():  # Disabled - beta API is buggy
            try:
                return await self._try_native_openai_parse(
                    messages, output_class, include_raw
                )
            except Exception as e:
                self.logger.warning(
                    f"Native OpenAI parse failed for GPT-5: {e}, trying LangChain"
                )
                # Fall through to LangChain attempt

        # Get provider-specific method (key insight from article)
        method = self._get_structured_output_method(self.model_name)

        # ENHANCEMENT: Try alternative methods if primary fails
        methods_to_try = [method]

        # Add fallback methods based on primary method
        if method == "json_schema" and "gpt-5" in self.model_name.lower():
            methods_to_try.append(
                "function_calling"
            )  # Fallback for problematic GPT-5 variants
        if method != "json_mode":
            methods_to_try.append("json_mode")  # Ultimate fallback

        last_error = None
        for method_attempt in methods_to_try:
            try:
                assert self.llm is not None  # Type assertion - already checked above

                self.logger.debug(
                    f"Attempting structured output with method={method_attempt}"
                )

                # Add timeout protection to prevent hanging
                try:
                    structured_llm = self.llm.with_structured_output(
                        output_class,
                        method=method_attempt,
                        include_raw=include_raw,
                    )

                    # Dynamic timeout calculation to prevent cascade failures
                    # Calculate remaining time budget based on total elapsed time
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    # Reserve time for fallback parser (5s) and buffer (2s)
                    reserved_time = 7.0
                    max_agent_timeout = 30.0  # Agent timeout constraint

                    remaining_budget = max_agent_timeout - elapsed_time - reserved_time

                    # Calculate optimal timeout for this attempt
                    if remaining_budget <= 0:
                        # Out of time budget, skip to fallback immediately
                        self.logger.warning(
                            f"Time budget exhausted ({elapsed_time:.1f}s elapsed), skipping to fallback"
                        )
                        raise asyncio.TimeoutError("Time budget exhausted")

                    # Progressive timeout with budget constraints
                    base_timeouts = [8.0, 6.0, 4.0]  # Reduced base timeouts
                    attempt_timeout = min(
                        base_timeouts[min(attempt, len(base_timeouts) - 1)],
                        remaining_budget,
                    )

                    self.logger.debug(
                        f"Attempt {attempt + 1}: {attempt_timeout:.1f}s timeout "
                        f"(elapsed: {elapsed_time:.1f}s, budget: {remaining_budget:.1f}s)"
                    )

                    result = await asyncio.wait_for(
                        structured_llm.ainvoke(messages),
                        timeout=attempt_timeout,  # Progressive reduction: 10s, 8s, 5s
                    )

                    # Success! Log and return
                    self.logger.info(
                        f"Structured output succeeded with method={method_attempt} for {self.model_name}"
                    )

                    # No additional processing needed - LangChain handles defaults properly

                    return cast(Union[T, Dict[str, Any]], result)

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Structured output timed out after {attempt_timeout}s with method={method_attempt} for {self.model_name}"
                    )
                    last_error = (
                        f"Timeout after {attempt_timeout}s with method={method_attempt}"
                    )
                    continue  # Try next method

            except (AttributeError, ValueError, Warning) as e:
                # Method not supported or schema issue - try next method
                self.logger.warning(
                    f"Method {method_attempt} failed for {self.model_name}: {e}"
                )
                last_error = str(e)
                continue  # Try next method

        # All methods failed - raise final error
        if last_error:
            raise LLMError(
                message=f"All structured output methods failed for {self.model_name}. Last error: {last_error}",
                llm_provider=(
                    self.model_name.split("-")[0]
                    if "-" in self.model_name
                    else "unknown"
                ),
                context={"methods_tried": methods_to_try, "last_error": last_error},
            )

        # Shouldn't reach here, but handle gracefully
        raise LLMError(
            message="Unexpected error in structured output",
            llm_provider="unknown",
            context={},
        )

    async def _try_native_openai_parse(
        self,
        messages: List[tuple[str, str]],
        output_class: Type[T],
        include_raw: bool = False,
    ) -> Union[T, Dict[str, Any]]:
        """
        Use OpenAI's native parse() API for GPT-5 models.

        This is the PROVEN approach from user testing that works with GPT-5.
        LangChain's with_structured_output() breaks with output_version parameter,
        but native OpenAI API works perfectly.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            self.logger.warning("OpenAI library not available for native parse")
            raise

        # Initialize native OpenAI client
        client = AsyncOpenAI(api_key=self.api_key)

        # Convert LangChain message format to OpenAI format
        openai_messages = []
        for role, content in messages:
            if role == "system":
                openai_messages.append({"role": "system", "content": content})
            elif role == "human" or role == "user":
                openai_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                openai_messages.append({"role": "assistant", "content": content})
            else:
                # Skip unknown roles
                self.logger.warning(f"Unknown message role: {role}")

        try:
            # Prepare OpenAI-compatible schema
            openai_schema = self._prepare_schema_for_openai(output_class)

            # Build kwargs for parse call with transformed schema
            parse_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": openai_messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_class.__name__,
                        "schema": openai_schema,
                        "strict": True,  # Enable strict mode for deterministic output
                    },
                },
            }

            # CRITICAL FIX: GPT-5 models only support temperature=1 (default)
            # Exclude temperature parameter for GPT-5 to avoid API constraint errors
            if "gpt-5" not in self.model_name.lower():
                parse_kwargs["temperature"] = self.temperature
            else:
                self.logger.info(
                    f"Excluding temperature from native parse for {self.model_name} (GPT-5 only supports default temperature=1)"
                )

            # Use OpenAI's beta parse API for structured outputs
            completion = await client.beta.chat.completions.parse(**parse_kwargs)

            # Extract parsed result
            parsed_result = completion.choices[0].message.parsed

            if parsed_result is None:
                raise LLMError(
                    message=f"Native OpenAI parse returned None for {self.model_name}",
                    llm_provider="openai",
                    context={"model": self.model_name},
                )

            # Parse the result back into the Pydantic model
            # Since we forced all fields to be required, OpenAI will always return them
            # The model's default values will handle the semantic optionality
            if not isinstance(parsed_result, output_class):
                # If OpenAI returns a dict, instantiate the Pydantic model
                if isinstance(parsed_result, dict):
                    parsed_result = output_class(**parsed_result)
                else:
                    # Already a proper instance from the beta.parse API
                    pass

            if include_raw:
                # Return with raw content for debugging
                raw_content = completion.choices[0].message.content
                return {"parsed": parsed_result, "raw": raw_content}

            return cast(T, parsed_result)

        except Exception as e:
            error_message = str(e).lower()

            # Enhanced error classification for better fallback decisions
            is_schema_error = any(
                phrase in error_message
                for phrase in [
                    "invalid schema",
                    "required is required",
                    "missing",
                    "additional keywords",
                    "$ref",
                    "additionalproperties",
                ]
            )

            is_quota_error = any(
                phrase in error_message
                for phrase in [
                    "quota exceeded",
                    "rate limit",
                    "insufficient credits",
                    "billing",
                ]
            )

            is_timeout_error = any(
                phrase in error_message
                for phrase in ["timeout", "connection", "network"]
            )

            # Log different error types with appropriate levels
            if is_quota_error:
                self.logger.error(
                    f"OpenAI quota/billing error for {self.model_name}: {e}"
                )
            elif is_schema_error:
                self.logger.warning(
                    f"OpenAI schema validation error for {self.model_name}: {e}"
                )
            elif is_timeout_error:
                self.logger.warning(f"OpenAI timeout error for {self.model_name}: {e}")
            else:
                self.logger.error(
                    f"Native OpenAI parse failed for {self.model_name}: {e}"
                )

            # Provide context-aware error information for fallback decisions
            raise LLMError(
                message=f"Native OpenAI parse failed for {self.model_name}",
                llm_provider="openai",
                context={
                    "error": str(e),
                    "model": self.model_name,
                    "method": "native_parse",
                    "error_type": (
                        "schema_validation"
                        if is_schema_error
                        else (
                            "quota_exceeded"
                            if is_quota_error
                            else "timeout" if is_timeout_error else "unknown"
                        )
                    ),
                    "fallback_recommended": not is_quota_error,  # Don't fallback on quota errors
                    "schema_fix_needed": is_schema_error,
                },
            )

    def _prepare_schema_for_openai(
        self, model_class: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Prepare Pydantic model schema for OpenAI's structured output API.

        CORRECTED REQUIREMENTS based on OpenAI beta.parse API testing:
        1. ALL properties MUST be in required array (OpenAI requirement)
        2. Optional/default fields should use anyOf with null type for nullable values
        3. Dict fields need explicit additionalProperties: false in all definitions
        4. $ref fields CANNOT have additional keywords like 'description'
        5. All unsupported constraints must be removed (maxLength, minLength, format, etc.)
        6. Nested models need same treatment as root model

        Args:
            model_class: The Pydantic model class to generate schema for

        Returns:
            Dict containing the OpenAI-compatible JSON schema
        """
        import copy
        from typing import get_origin, get_args

        # Generate the standard Pydantic JSON schema
        schema = model_class.model_json_schema()

        # Create a deep copy to avoid modifying the original
        fixed_schema = copy.deepcopy(schema)

        if "properties" not in fixed_schema:
            return fixed_schema

        # Get model fields to understand which are actually required
        model_fields = model_class.model_fields

        # Based on OpenAI error: "required is required to be supplied and to be an array including every key in properties"
        # OpenAI requires ALL properties to be in required array, but optional fields should be nullable
        actual_required = list(fixed_schema["properties"].keys())

        for field_name, field_info in model_fields.items():
            # Import PydanticUndefined for correct detection
            from pydantic_core import PydanticUndefined

            # Check if field has a default value or default_factory
            # In Pydantic v2, PydanticUndefined means "no default value"
            has_default_value = field_info.default is not PydanticUndefined
            has_default_factory = field_info.default_factory is not None
            has_any_default = has_default_value or has_default_factory

            # Check if field is Optional (Union with None)
            is_optional = get_origin(field_info.annotation) is Union and type(
                None
            ) in get_args(field_info.annotation)

            # For fields with defaults or Optional fields, ensure proper schema setup
            if field_name in fixed_schema["properties"]:
                prop_def = fixed_schema["properties"][field_name]

                # For fields with defaults or Optional fields, make them nullable in schema
                if (has_any_default or is_optional) and isinstance(prop_def, dict):
                    if "type" in prop_def and not isinstance(
                        prop_def.get("anyOf"), list
                    ):
                        # Convert to union type with null to indicate it can accept null values
                        original_type = prop_def.copy()
                        # DON'T remove the type - OpenAI requires it
                        # Clean unsupported constraints from the original type
                        unsupported_keys = [
                            "maxLength",
                            "minLength",
                            "format",
                            "pattern",
                            "maxItems",
                            "minItems",
                            "maximum",
                            "minimum",
                        ]
                        for key in unsupported_keys:
                            original_type.pop(key, None)
                        prop_def.clear()
                        prop_def["anyOf"] = [original_type, {"type": "null"}]

                # Clean up unsupported constraints
                if isinstance(prop_def, dict):
                    # Remove constraints that OpenAI doesn't support
                    unsupported_keys = [
                        "maxLength",
                        "minLength",
                        "format",
                        "pattern",
                        "maxItems",
                        "minItems",
                        "maximum",
                        "minimum",
                    ]
                    for key in unsupported_keys:
                        prop_def.pop(key, None)

                self.logger.debug(
                    f"Field {field_name}: has_any_default={has_any_default}, is_optional={is_optional}, "
                    f"in_required=yes (all fields required by OpenAI), nullable={'yes' if (has_any_default or is_optional) else 'no'}"
                )

        # Set ALL properties as required (OpenAI requirement)
        fixed_schema["required"] = actual_required

        self.logger.info(
            f"OpenAI schema correction for {model_class.__name__}: "
            f"{len(actual_required)} required fields (all properties): {actual_required}"
        )

        # Remove descriptions from $ref fields (OpenAI requirement)
        for prop_name, prop_def in fixed_schema["properties"].items():
            if isinstance(prop_def, dict) and "$ref" in prop_def:
                # Keep ONLY the $ref key, remove description or any other keys
                fixed_schema["properties"][prop_name] = {"$ref": prop_def["$ref"]}
                self.logger.debug(
                    f"Cleaned $ref field {prop_name} for OpenAI compatibility"
                )

        # Ensure additionalProperties is false (OpenAI requirement)
        fixed_schema["additionalProperties"] = False

        # Handle nested model definitions in $defs
        # CRITICAL FIX: Apply same ALL-fields-required rule to nested models
        if "$defs" in fixed_schema:
            for def_name, def_schema in fixed_schema["$defs"].items():
                if "properties" in def_schema and isinstance(
                    def_schema["properties"], dict
                ):
                    # Apply the same ALL-properties-required rule to nested models
                    nested_properties = def_schema["properties"]
                    nested_required = list(nested_properties.keys())
                    def_schema["required"] = nested_required
                    def_schema["additionalProperties"] = False
                    
                    # CRITICAL FIX: Dict fields need special handling
                    # If this is a dict-type field, ensure it has proper schema structure
                    if def_schema.get("type") == "object" and not nested_properties:
                        # Empty object schema - this is likely a Dict field
                        def_schema["additionalProperties"] = False
                        # Don't require any fields for generic Dict types
                        def_schema["required"] = []

                    # Handle nullable fields in nested models too
                    for nested_prop_name, nested_prop_def in nested_properties.items():
                        if isinstance(nested_prop_def, dict):
                            # Check if this is an Optional field (anyOf with null)
                            is_nullable = isinstance(
                                nested_prop_def.get("anyOf"), list
                            ) and any(
                                item.get("type") == "null"
                                for item in nested_prop_def.get("anyOf", [])
                            )

                            # For Optional fields that aren't already nullable, make them nullable
                            if not is_nullable and "type" in nested_prop_def:
                                # Check if this looks like an Optional field (has default: null)
                                if nested_prop_def.get("default") is None:
                                    original_type = nested_prop_def.copy()
                                    original_type.pop(
                                        "default", None
                                    )  # Remove default from type definition
                                    # DON'T remove the type - OpenAI requires it
                                    # Clean unsupported constraints from the original type
                                    unsupported_keys = [
                                        "maxLength",
                                        "minLength",
                                        "format",
                                        "pattern",
                                        "maxItems",
                                        "minItems",
                                        "maximum",
                                        "minimum",
                                    ]
                                    for key in unsupported_keys:
                                        original_type.pop(key, None)
                                    nested_prop_def.clear()
                                    nested_prop_def["anyOf"] = [
                                        original_type,
                                        {"type": "null"},
                                    ]
                                    nested_prop_def["default"] = (
                                        None  # Preserve default at top level
                                    )

                            # Clean unsupported constraints
                            unsupported_keys = [
                                "maxLength",
                                "minLength",
                                "format",
                                "pattern",
                                "maxItems",
                                "minItems",
                                "maximum",
                                "minimum",
                            ]
                            for key in unsupported_keys:
                                nested_prop_def.pop(key, None)
                                
                            # CRITICAL FIX: Handle Dict fields in nested models
                            # If this is a dict field (type: object with additionalProperties)
                            if (
                                nested_prop_def.get("type") == "object" 
                                and "properties" not in nested_prop_def
                            ):
                                # This is a Dict field - ensure additionalProperties is explicitly false
                                nested_prop_def["additionalProperties"] = False

                    self.logger.debug(
                        f"Fixed nested model {def_name}: {len(nested_required)} required fields (all properties): {nested_required}"
                    )

        # Recursively clean up $ref fields and unsupported constraints throughout the schema
        def clean_refs_recursive(obj: Any) -> Any:
            """Recursively clean $ref fields and unsupported constraints throughout the schema."""
            if isinstance(obj, dict):
                if "$ref" in obj and len(obj) > 1:
                    # Keep only the $ref key
                    return {"$ref": obj["$ref"]}
                else:
                    # Clean unsupported constraints and recursively process
                    cleaned = {}
                    unsupported_keys = [
                        "maxLength",
                        "minLength",
                        "format",
                        "pattern",
                        "maxItems",
                        "minItems",
                        "maximum",
                        "minimum",
                    ]
                    for k, v in obj.items():
                        if k not in unsupported_keys:
                            cleaned[k] = clean_refs_recursive(v)
                    
                    # CRITICAL FIX: Ensure all object types have additionalProperties: false
                    if cleaned.get("type") == "object" and "additionalProperties" not in cleaned:
                        cleaned["additionalProperties"] = False
                    
                    # CRITICAL FIX: For Dict-like objects without properties, still need additionalProperties: false
                    if (
                        cleaned.get("type") == "object" 
                        and "properties" not in cleaned
                        and "$ref" not in cleaned
                    ):
                        cleaned["additionalProperties"] = False
                        
                    return cleaned
            elif isinstance(obj, list):
                return [clean_refs_recursive(item) for item in obj]
            else:
                return obj

        # Apply recursive cleaning
        fixed_schema = clean_refs_recursive(fixed_schema)

        self.logger.debug(
            f"OpenAI schema finalized for {model_class.__name__}: "
            f"required={actual_required}, additionalProperties=false, refs cleaned"
        )

        return cast(Dict[str, Any], fixed_schema)

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
