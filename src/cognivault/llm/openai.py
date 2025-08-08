import openai
import time
from typing import Any, Iterator, Optional, Callable, Union, cast, List
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from .llm_interface import LLMInterface, LLMResponse
from cognivault.exceptions import (
    LLMQuotaError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMContextLimitError,
    LLMModelNotFoundError,
    LLMServerError,
    LLMError,
)
from cognivault.observability import get_logger, get_observability_context


class OpenAIChatLLM(LLMInterface):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpenAIChatLLM.

        Parameters
        ----------
        api_key : str
            Your OpenAI API key.
        model : str, optional
            The model name to use (default is "gpt-4").
        base_url : str, optional
            Optional base URL for custom endpoints.
        """
        self.api_key = api_key
        self.model: Optional[str] = model
        self.base_url: Optional[str] = base_url
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[str]]:
        """
        Generate a response from the OpenAI LLM.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM.
        system_prompt : str, optional
            System prompt to provide context/instructions to the LLM.
        stream : bool, optional
            Whether to yield partial output tokens (default is False).
        on_log : Callable[[str], None], optional
            Logging hook to trace internal decisions.
        **kwargs : dict
            Additional OpenAI API parameters such as temperature, max_tokens, etc.

        Returns
        -------
        LLMResponse or Iterator[str]
            The structured response or a stream of tokens.
        """
        # Initialize observability
        logger = get_logger("llm.openai")

        # Lazy import to avoid circular dependency
        try:
            from cognivault.diagnostics.metrics import get_metrics_collector

            metrics = get_metrics_collector()
        except ImportError:
            metrics = None

        llm_start_time = time.time()

        # Get observability context for correlation
        obs_context = get_observability_context()

        if on_log:
            on_log(f"[OpenAIChatLLM] Prompt: {prompt}")
            if system_prompt:
                on_log(f"[OpenAIChatLLM] System Prompt: {system_prompt}")
            on_log(f"[OpenAIChatLLM] Stream: {stream}, Model: {self.model}")

        # Build messages array with optional system prompt
        messages: List[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(
                f"Making LLM call to {self.model}",
                model=self.model,
                prompt_length=len(prompt),
                system_prompt_length=len(system_prompt) if system_prompt else 0,
                stream=stream,
                agent_name=obs_context.agent_name if obs_context else None,
            )

            assert isinstance(self.model, str), "model must be a string"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
        except openai.APIError as e:
            # Record failed LLM call metrics
            llm_duration_ms = (time.time() - llm_start_time) * 1000

            logger.error(
                f"LLM API call failed: {str(e)}",
                model=self.model,
                duration_ms=llm_duration_ms,
                error_type=type(e).__name__,
                agent_name=obs_context.agent_name if obs_context else None,
            )

            if metrics:
                metrics.increment_counter(
                    "llm_api_calls_failed",
                    labels={
                        "model": self.model or "unknown",
                        "agent": (
                            obs_context.agent_name
                            if obs_context and obs_context.agent_name
                            else "unknown"
                        ),
                        "error_type": type(e).__name__,
                    },
                )

            if on_log:
                on_log(f"[OpenAIChatLLM][error] {str(e)}")

            # Convert OpenAI API errors to our structured exception hierarchy
            self._handle_openai_error(e)
        except Exception as e:
            # Record unexpected error metrics
            llm_duration_ms = (time.time() - llm_start_time) * 1000

            logger.error(
                f"Unexpected LLM error: {str(e)}",
                model=self.model,
                duration_ms=llm_duration_ms,
                error_type=type(e).__name__,
                agent_name=obs_context.agent_name if obs_context else None,
            )

            if metrics:
                metrics.increment_counter(
                    "llm_api_calls_failed",
                    labels={
                        "model": self.model or "unknown",
                        "agent": (
                            obs_context.agent_name
                            if obs_context and obs_context.agent_name
                            else "unknown"
                        ),
                        "error_type": "unexpected_error",
                    },
                )

            if on_log:
                on_log(f"[OpenAIChatLLM][unexpected_error] {str(e)}")

            # Handle non-OpenAI exceptions
            raise LLMError(
                message=f"Unexpected error during OpenAI API call: {str(e)}",
                llm_provider="openai",
                error_code="unexpected_error",
                cause=e,
            )

        if stream:

            def token_generator() -> None:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    if on_log and content:
                        on_log(f"[OpenAIChatLLM][streaming] {content}")
                    yield content

            return token_generator()

        else:
            response = cast(ChatCompletion, response)
            if (
                not hasattr(response, "choices")
                or not isinstance(response.choices, list)
                or not response.choices
            ):
                raise ValueError("Missing or invalid 'choices' in response")
            choice = response.choices[0]
            if not hasattr(choice, "message") or not hasattr(choice.message, "content"):
                raise ValueError("Missing 'message.content' in the first choice")

            if not hasattr(response, "usage") or not hasattr(
                response.usage, "total_tokens"
            ):
                raise ValueError("Missing 'usage.total_tokens' in response")

            text = choice.message.content or ""

            # Extract detailed token usage information
            tokens_used = response.usage.total_tokens if response.usage else None
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            finish_reason = choice.finish_reason

            # Calculate LLM call duration and record metrics
            llm_duration_ms = (time.time() - llm_start_time) * 1000

            # Record LLM call metrics
            logger.log_llm_call(
                model=self.model or "unknown",
                tokens_used=tokens_used or 0,
                duration_ms=llm_duration_ms,
                prompt_length=len(prompt),
                response_length=len(text),
                finish_reason=finish_reason,
                agent_name=obs_context.agent_name if obs_context else None,
            )

            if metrics:
                metrics.record_timing(
                    "llm_call_duration",
                    llm_duration_ms,
                    labels={
                        "model": self.model or "unknown",
                        "agent": (
                            obs_context.agent_name
                            if obs_context and obs_context.agent_name
                            else "unknown"
                        ),
                    },
                )

            if tokens_used and metrics:
                metrics.increment_counter(
                    "llm_tokens_consumed",
                    tokens_used,
                    labels={
                        "model": self.model or "unknown",
                        "agent": (
                            obs_context.agent_name
                            if obs_context and obs_context.agent_name
                            else "unknown"
                        ),
                    },
                )

            if metrics:
                metrics.increment_counter(
                    "llm_api_calls_successful",
                    labels={
                        "model": self.model or "unknown",
                        "agent": (
                            obs_context.agent_name
                            if obs_context and obs_context.agent_name
                            else "unknown"
                        ),
                    },
                )

            return LLMResponse(
                text=text,
                tokens_used=tokens_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_name=self.model,
                finish_reason=finish_reason,
            )

    def _handle_openai_error(self, error: openai.APIError) -> None:
        """
        Convert OpenAI API errors to structured CogniVault exceptions.

        Parameters
        ----------
        error : openai.APIError
            The OpenAI API error to convert

        Raises
        ------
        LLMError
            Appropriate CogniVault LLM exception based on the error type
        """
        error_message = str(error)
        error_code = getattr(error, "code", None)
        status_code = getattr(error, "status_code", None)

        # Handle quota/billing errors
        if (
            error_code == "insufficient_quota"
            or "quota" in error_message.lower()
            or "billing" in error_message.lower()
        ):
            raise LLMQuotaError(
                llm_provider="openai", quota_type="api_credits", cause=error
            )

        # Handle authentication errors
        if (
            error_code == "invalid_api_key"
            or status_code == 401
            or "authentication" in error_message.lower()
            or "api key" in error_message.lower()
        ):
            raise LLMAuthError(
                llm_provider="openai", auth_issue="invalid_api_key", cause=error
            )

        # Handle rate limiting errors
        if (
            error_code == "rate_limit_exceeded"
            or status_code == 429
            or "rate limit" in error_message.lower()
        ):
            # Try to extract retry-after header
            retry_after = None
            if hasattr(error, "response") and error.response:
                retry_after_header = error.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass

            raise LLMRateLimitError(
                llm_provider="openai",
                rate_limit_type="requests_per_minute",
                retry_after_seconds=retry_after,
                cause=error,
            )

        # Handle context length errors
        if (
            error_code == "context_length_exceeded"
            or "context length" in error_message.lower()
            or "token limit" in error_message.lower()
        ):
            # Try to extract token counts from error message
            token_count = None
            max_tokens = None

            # Parse error message for token info (OpenAI usually provides this)
            import re

            token_match = re.search(
                r"(\d+)\s*(?:tokens?).*?(?:maximum|limit).*?(\d+)",
                error_message.lower(),
            )
            if token_match:
                try:
                    token_count = int(token_match.group(1))
                    max_tokens = int(token_match.group(2))
                except ValueError:
                    pass

            raise LLMContextLimitError(
                llm_provider="openai",
                model_name=self.model or "unknown",
                token_count=token_count or 0,
                max_tokens=max_tokens or 0,
                cause=error,
            )

        # Handle model not found errors
        if (
            error_code == "model_not_found"
            or status_code == 404
            or "model" in error_message.lower()
            and "not found" in error_message.lower()
        ):
            raise LLMModelNotFoundError(
                llm_provider="openai", model_name=self.model or "unknown", cause=error
            )

        # Handle server errors (5xx)
        if status_code and 500 <= status_code < 600:
            raise LLMServerError(
                llm_provider="openai",
                http_status=status_code,
                error_details=error_message,
                cause=error,
            )

        # Handle timeout errors
        if "timeout" in error_message.lower() or isinstance(
            error, openai.APITimeoutError
        ):
            raise LLMTimeoutError(
                llm_provider="openai",
                timeout_seconds=30.0,  # Default timeout, could be extracted from config
                timeout_type="api_request",
                cause=error,
            )

        # Default fallback for unknown OpenAI errors
        raise LLMError(
            message=f"OpenAI API error: {error_message}",
            llm_provider="openai",
            error_code=error_code or "unknown_api_error",
            context={"status_code": status_code, "error_type": type(error).__name__},
            cause=error,
        )
