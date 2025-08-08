import pytest
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMResponse


@pytest.fixture
def mock_openai_chat_completion() -> Any:
    with patch("cognivault.llm.openai.openai.OpenAI") as mock_openai_client:
        instance: MagicMock = MagicMock()
        mock_openai_client.return_value = instance
        yield instance.chat.completions


def test_generate_non_streaming(mock_openai_chat_completion: Any) -> None:
    # Create mocks that simulate OpenAI's response structure
    message: MagicMock = MagicMock()
    message.content = "Hello!"

    choice: MagicMock = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage: MagicMock = MagicMock()
    usage.total_tokens = 10

    mock_response: MagicMock = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_openai_chat_completion.create.return_value = mock_response

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    response = llm.generate(prompt="Hi", stream=False)

    assert isinstance(response, LLMResponse)
    assert response.text == "Hello!"
    assert response.tokens_used == 10
    assert response.finish_reason == "stop"


def test_generate_invalid_response_structure(mock_openai_chat_completion: Any) -> None:
    # Missing 'choices'
    mock_response: MagicMock = MagicMock()
    type(mock_response).choices = PropertyMock(side_effect=AttributeError("choices"))
    mock_response.usage: MagicMock = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_openai_chat_completion.create.return_value = mock_response

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(ValueError, match="Missing or invalid 'choices' in response"):
        llm.generate(prompt="Hi", stream=False)

    # Missing 'message.content'
    mock_response: MagicMock = MagicMock()
    choice: MagicMock = MagicMock()
    delattr(choice, "message")
    mock_response.choices = [choice]
    mock_response.usage = MagicMock(total_tokens=10)
    mock_openai_chat_completion.create.return_value = mock_response

    with pytest.raises(
        ValueError, match="Missing 'message.content' in the first choice"
    ):
        llm.generate(prompt="Hi", stream=False)

    # Missing 'usage.total_tokens'
    mock_response: MagicMock = MagicMock()
    message: MagicMock = MagicMock()
    message.content = "Hello!"
    choice: MagicMock = MagicMock()
    choice.message = message
    mock_response.choices = [choice]
    delattr(mock_response, "usage")
    mock_openai_chat_completion.create.return_value = mock_response

    with pytest.raises(ValueError, match="Missing 'usage.total_tokens' in response"):
        llm.generate(prompt="Hi", stream=False)


def test_generate_streaming(mock_openai_chat_completion: Any) -> None:
    def _mk_chunk(text: str) -> MagicMock:
        delta = MagicMock(content=text)
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice])

    stream_chunks = [_mk_chunk("Hel"), _mk_chunk("lo"), _mk_chunk("!")]
    mock_openai_chat_completion.create.return_value = iter(stream_chunks)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    output = "".join(llm.generate(prompt="Hi", stream=True))

    assert output == "Hello!"


def test_generate_with_logging_hook(mock_openai_chat_completion: Any) -> None:
    # Set up proper mock structure
    message: MagicMock = MagicMock()
    message.content = "Logged"

    choice: MagicMock = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage: MagicMock = MagicMock()
    usage.total_tokens = 5

    mock_response: MagicMock = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_openai_chat_completion.create.return_value = mock_response

    logs = []

    def log_hook(msg: Any) -> None:
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    _ = llm.generate(prompt="Log this", stream=False, on_log=log_hook)

    assert any("Prompt: Log this" in log for log in logs)
    assert any("Model: gpt-4" in log for log in logs)


def test_base_url_sets_api_base(mock_openai_chat_completion: Any) -> None:
    with patch("cognivault.llm.openai.openai.OpenAI") as mock_openai_client:
        OpenAIChatLLM(
            api_key="test-key", model="gpt-4", base_url="https://custom.openai.com"
        )
        mock_openai_client.assert_called_with(
            api_key="test-key", base_url="https://custom.openai.com"
        )


def test_streaming_with_logging_hook(mock_openai_chat_completion: Any) -> None:
    def _mk_chunk(text: str) -> MagicMock:
        delta = MagicMock(content=text)
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice])

    stream_chunks = [_mk_chunk("Test")]
    mock_openai_chat_completion.create.return_value = iter(stream_chunks)

    logs = []

    def log_hook(msg: Any) -> None:
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    _ = "".join(llm.generate(prompt="Log stream", stream=True, on_log=log_hook))

    assert any("[OpenAIChatLLM][streaming] Test" in log for log in logs)


def test_generate_with_system_prompt_logging(mock_openai_chat_completion: Any) -> None:
    # Set up proper mock structure
    message: MagicMock = MagicMock()
    message.content = "System response"

    choice: MagicMock = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage: MagicMock = MagicMock()
    usage.total_tokens = 15

    mock_response: MagicMock = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_openai_chat_completion.create.return_value = mock_response

    logs = []

    def log_hook(msg: Any) -> None:
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    _ = llm.generate(
        prompt="User prompt",
        system_prompt="You are a helpful assistant",
        stream=False,
        on_log=log_hook,
    )

    # Verify system prompt is logged
    assert any("System Prompt: You are a helpful assistant" in log for log in logs)
    assert any("Prompt: User prompt" in log for log in logs)


def test_generate_api_error_with_logging(mock_openai_chat_completion: Any) -> None:
    from openai import APIError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_openai_chat_completion.create.side_effect = APIError(
        "Mock API error", request=dummy_request, body="{}"
    )

    logs = []

    def log_hook(msg: Any) -> None:
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMError, match="Mock API error"):
        llm.generate(prompt="Trigger API failure", stream=False, on_log=log_hook)

    assert any("Mock API error" in log for log in logs)
    assert any("[OpenAIChatLLM][error]" in log for log in logs)


def test_unexpected_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of unexpected (non-OpenAI) exceptions."""
    from cognivault.exceptions.llm_errors import LLMError

    # Simulate an unexpected error (not an OpenAI APIError)
    mock_openai_chat_completion.create.side_effect = ValueError("Unexpected error")

    logs = []

    def log_hook(msg: Any) -> None:
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMError) as exc_info:
        llm.generate(prompt="Test", stream=False, on_log=log_hook)

    assert "Unexpected error during OpenAI API call" in str(exc_info.value)
    assert exc_info.value.error_code == "unexpected_error"
    assert exc_info.value.llm_provider == "openai"
    # Check logging of unexpected error
    assert any("[OpenAIChatLLM][unexpected_error]" in log for log in logs)


def test_quota_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of quota/billing errors."""
    from openai import APIError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMQuotaError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    # Test quota exceeded error
    quota_error = APIError(
        "You have exceeded your quota", request=dummy_request, body="{}"
    )
    mock_openai_chat_completion.create.side_effect = quota_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMQuotaError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.quota_type == "api_credits"


def test_auth_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of authentication errors."""
    from openai import APIError
    from httpx import Request, Response
    from cognivault.exceptions.llm_errors import LLMAuthError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    # Create a mock response with 401 status
    response = Response(status_code=401, request=dummy_request)
    auth_error = APIError("Invalid API key", request=dummy_request, body="{}")
    auth_error.response = response

    mock_openai_chat_completion.create.side_effect = auth_error

    llm = OpenAIChatLLM(api_key="invalid-key", model="gpt-4")

    with pytest.raises(LLMAuthError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.auth_issue == "invalid_api_key"


def test_rate_limit_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of rate limit errors."""
    from openai import APIError
    from httpx import Request, Response, Headers
    from cognivault.exceptions.llm_errors import LLMRateLimitError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    # Create response with rate limit headers
    headers = Headers({"retry-after": "60"})
    response = Response(status_code=429, request=dummy_request, headers=headers)
    rate_error = APIError("Rate limit exceeded", request=dummy_request, body="{}")
    rate_error.response = response

    mock_openai_chat_completion.create.side_effect = rate_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMRateLimitError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.rate_limit_type == "requests_per_minute"
    assert exc_info.value.retry_after_seconds == 60.0


def test_rate_limit_error_without_retry_after(mock_openai_chat_completion: Any) -> None:
    """Test rate limit error handling without retry-after header."""
    from openai import APIError
    from httpx import Request, Response
    from cognivault.exceptions.llm_errors import LLMRateLimitError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    response = Response(status_code=429, request=dummy_request)
    rate_error = APIError("Rate limit exceeded", request=dummy_request, body="{}")
    rate_error.response = response

    mock_openai_chat_completion.create.side_effect = rate_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMRateLimitError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert exc_info.value.retry_after_seconds is None


def test_context_limit_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of context length limit errors."""
    from openai import APIError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMContextLimitError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    # Error message with "token limit" pattern (easier to match)
    context_error = APIError(
        "Request exceeds token limit", request=dummy_request, body="{}"
    )

    mock_openai_chat_completion.create.side_effect = context_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMContextLimitError) as exc_info:
        llm.generate(prompt="Very long prompt...", stream=False)

    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.model_name == "gpt-4"
    # When regex parsing fails, defaults to 0
    assert exc_info.value.token_count == 0
    assert exc_info.value.max_tokens == 0


def test_context_limit_error_without_token_parsing(
    mock_openai_chat_completion: Any,
) -> None:
    """Test context limit error without parseable token counts."""
    from openai import APIError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMContextLimitError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    # Error message without parseable token counts
    context_error = APIError(
        "Context length limit exceeded", request=dummy_request, body="{}"
    )

    mock_openai_chat_completion.create.side_effect = context_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMContextLimitError) as exc_info:
        llm.generate(prompt="Long prompt", stream=False)

    assert exc_info.value.token_count == 0  # Default when parsing fails
    assert exc_info.value.max_tokens == 0


def test_context_limit_error_with_regex_but_int_parsing_failure() -> None:
    """Test context limit error when regex succeeds but int() conversion fails.

    This tests the ValueError exception handler on lines 353-354 in openai.py.
    We directly test the _handle_openai_error method to trigger this edge case.
    """
    from openai import APIError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMContextLimitError
    from unittest.mock import MagicMock, PropertyMock, patch

    # Create an LLM instance
    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    # Create a mock API error that would match context limit conditions
    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    context_error = APIError(
        "Request exceeds token limit",
        request=dummy_request,
        body="{}",
    )

    # Patch the regex search to return a match that will cause int() to fail
    with patch("re.search") as mock_search:
        # Create a mock match object where group() returns values that int() can't parse
        mock_match: MagicMock = MagicMock()
        # Make group(1) return an empty string (which int() can't parse)
        mock_match.group.side_effect = lambda x: "" if x == 1 else "4096"
        mock_search.return_value = mock_match

        with pytest.raises(LLMContextLimitError) as exc_info:
            llm._handle_openai_error(context_error)

    # When int() parsing fails, should default to 0
    assert exc_info.value.token_count == 0
    assert exc_info.value.max_tokens == 0
    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.model_name == "gpt-4"


def test_model_not_found_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of model not found errors."""
    from openai import APIError
    from httpx import Request, Response
    from cognivault.exceptions.llm_errors import LLMModelNotFoundError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    response = Response(status_code=404, request=dummy_request)

    model_error = APIError(
        "Model 'nonexistent-model' not found", request=dummy_request, body="{}"
    )
    model_error.response = response

    mock_openai_chat_completion.create.side_effect = model_error

    llm = OpenAIChatLLM(api_key="test-key", model="nonexistent-model")

    with pytest.raises(LLMModelNotFoundError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.model_name == "nonexistent-model"


def test_server_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of server errors (500+)."""
    from openai import APIError
    from httpx import Request, Response
    from cognivault.exceptions.llm_errors import LLMServerError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    response = Response(status_code=500, request=dummy_request)

    server_error = APIError("Internal server error", request=dummy_request, body="{}")
    server_error.response = response
    # Set the status_code attribute that the error handler looks for
    server_error.status_code = 500

    mock_openai_chat_completion.create.side_effect = server_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMServerError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.http_status == 500


def test_generic_llm_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of generic API errors that don't match specific patterns."""
    from openai import APIError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    generic_error = APIError("Some other API error", request=dummy_request, body="{}")

    mock_openai_chat_completion.create.side_effect = generic_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert "Some other API error" in str(exc_info.value)
    assert exc_info.value.llm_provider == "openai"
    assert (
        exc_info.value.error_code == "unknown_api_error"
    )  # This is the actual error code used


def test_rate_limit_retry_after_invalid_value(mock_openai_chat_completion: Any) -> None:
    """Test rate limit error handling with invalid retry-after header value."""
    from openai import APIError
    from httpx import Request, Response, Headers
    from cognivault.exceptions.llm_errors import LLMRateLimitError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    # Create response with invalid retry-after header
    headers = Headers({"retry-after": "invalid-value"})
    response = Response(status_code=429, request=dummy_request, headers=headers)
    rate_error = APIError("Rate limit exceeded", request=dummy_request, body="{}")
    rate_error.response = response

    mock_openai_chat_completion.create.side_effect = rate_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMRateLimitError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    # Should handle invalid retry-after value gracefully
    assert exc_info.value.retry_after_seconds is None


def test_context_limit_with_valid_token_parsing(
    mock_openai_chat_completion: Any,
) -> None:
    """Test context limit error with successful token parsing."""
    from openai import APIError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMContextLimitError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")

    # Error message that contains "token limit" and matches the regex pattern correctly
    context_error = APIError(
        "Request exceeds token limit: has 5000 tokens but maximum is 4096",
        request=dummy_request,
        body="{}",
    )

    mock_openai_chat_completion.create.side_effect = context_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMContextLimitError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    # Should successfully parse token counts
    assert exc_info.value.token_count == 5000
    assert exc_info.value.max_tokens == 4096


def test_timeout_error_handling(mock_openai_chat_completion: Any) -> None:
    """Test handling of timeout errors."""
    from openai import APITimeoutError
    from httpx import Request
    from cognivault.exceptions.llm_errors import LLMTimeoutError

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    timeout_error = APITimeoutError(request=dummy_request)

    mock_openai_chat_completion.create.side_effect = timeout_error

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(LLMTimeoutError) as exc_info:
        llm.generate(prompt="Test", stream=False)

    assert exc_info.value.llm_provider == "openai"
    assert exc_info.value.timeout_seconds == 30.0
    assert exc_info.value.timeout_type == "api_request"
