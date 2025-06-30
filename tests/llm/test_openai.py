import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from unittest.mock import patch
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMResponse


@pytest.fixture
def mock_openai_chat_completion():
    with patch("cognivault.llm.openai.openai.OpenAI") as mock_openai_client:
        instance = MagicMock()
        mock_openai_client.return_value = instance
        yield instance.chat.completions


def test_generate_non_streaming(mock_openai_chat_completion):
    # Create mocks that simulate OpenAI's response structure
    message = MagicMock()
    message.content = "Hello!"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.total_tokens = 10

    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_openai_chat_completion.create.return_value = mock_response

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    response = llm.generate(prompt="Hi", stream=False)

    assert isinstance(response, LLMResponse)
    assert response.text == "Hello!"
    assert response.tokens_used == 10
    assert response.finish_reason == "stop"


def test_generate_invalid_response_structure(mock_openai_chat_completion):
    # Missing 'choices'
    mock_response = MagicMock()
    type(mock_response).choices = PropertyMock(side_effect=AttributeError("choices"))
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_openai_chat_completion.create.return_value = mock_response

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(ValueError, match="Missing or invalid 'choices' in response"):
        llm.generate(prompt="Hi", stream=False)

    # Missing 'message.content'
    mock_response = MagicMock()
    choice = MagicMock()
    delattr(choice, "message")
    mock_response.choices = [choice]
    mock_response.usage = MagicMock(total_tokens=10)
    mock_openai_chat_completion.create.return_value = mock_response

    with pytest.raises(
        ValueError, match="Missing 'message.content' in the first choice"
    ):
        llm.generate(prompt="Hi", stream=False)

    # Missing 'usage.total_tokens'
    mock_response = MagicMock()
    message = MagicMock()
    message.content = "Hello!"
    choice = MagicMock()
    choice.message = message
    mock_response.choices = [choice]
    delattr(mock_response, "usage")
    mock_openai_chat_completion.create.return_value = mock_response

    with pytest.raises(ValueError, match="Missing 'usage.total_tokens' in response"):
        llm.generate(prompt="Hi", stream=False)


def test_generate_streaming(mock_openai_chat_completion):
    def _mk_chunk(text: str):
        delta = MagicMock(content=text)
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice])

    stream_chunks = [_mk_chunk("Hel"), _mk_chunk("lo"), _mk_chunk("!")]
    mock_openai_chat_completion.create.return_value = iter(stream_chunks)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    output = "".join(llm.generate(prompt="Hi", stream=True))

    assert output == "Hello!"


def test_generate_with_logging_hook(mock_openai_chat_completion):
    # Set up proper mock structure
    message = MagicMock()
    message.content = "Logged"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.total_tokens = 5

    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_openai_chat_completion.create.return_value = mock_response

    logs = []

    def log_hook(msg):
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    _ = llm.generate(prompt="Log this", stream=False, on_log=log_hook)

    assert any("Prompt: Log this" in log for log in logs)
    assert any("Model: gpt-4" in log for log in logs)


def test_base_url_sets_api_base(mock_openai_chat_completion):
    with patch("cognivault.llm.openai.openai.OpenAI") as mock_openai_client:
        llm = OpenAIChatLLM(
            api_key="test-key", model="gpt-4", base_url="https://custom.openai.com"
        )
        mock_openai_client.assert_called_with(
            api_key="test-key", base_url="https://custom.openai.com"
        )


def test_streaming_with_logging_hook(mock_openai_chat_completion):
    def _mk_chunk(text: str):
        delta = MagicMock(content=text)
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice])

    stream_chunks = [_mk_chunk("Test")]
    mock_openai_chat_completion.create.return_value = iter(stream_chunks)

    logs = []

    def log_hook(msg):
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")
    _ = "".join(llm.generate(prompt="Log stream", stream=True, on_log=log_hook))

    assert any("[OpenAIChatLLM][streaming] Test" in log for log in logs)


def test_generate_with_system_prompt_logging(mock_openai_chat_completion):
    # Set up proper mock structure
    message = MagicMock()
    message.content = "System response"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_openai_chat_completion.create.return_value = mock_response

    logs = []

    def log_hook(msg):
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


def test_generate_api_error_with_logging(mock_openai_chat_completion):
    from openai import APIError

    from httpx import Request

    dummy_request = Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_openai_chat_completion.create.side_effect = APIError(
        "Mock API error", request=dummy_request, body="{}"
    )

    logs = []

    def log_hook(msg):
        logs.append(msg)

    llm = OpenAIChatLLM(api_key="test-key", model="gpt-4")

    with pytest.raises(APIError):
        llm.generate(prompt="Trigger API failure", stream=False, on_log=log_hook)

    assert any("Mock API error" in log for log in logs)
    assert any("[OpenAIChatLLM][error]" in log for log in logs)
