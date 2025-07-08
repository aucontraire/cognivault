import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from io import StringIO

from cognivault.agents.refiner.main import run_refiner, parse_args, main


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.RefinerAgent")
async def test_run_refiner_returns_expected_output(mock_agent_class):
    mock_agent = AsyncMock()
    mock_agent.name = "Refiner"

    async def mock_run(context):
        context.add_agent_output("Refiner", "Mocked refiner output")

    mock_agent.run.side_effect = mock_run
    mock_agent_class.return_value = mock_agent

    query = "What causes revolutions?"
    result, debug_info = await run_refiner(query)

    assert result == "Mocked refiner output"
    assert debug_info is None  # No debug mode
    mock_agent.run.assert_awaited_once()


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.RefinerAgent")
async def test_run_refiner_debug_mode(mock_agent_class):
    mock_agent = AsyncMock()
    mock_agent.name = "Refiner"

    async def mock_run(context):
        context.add_agent_output("Refiner", "Refined query: Test refined output")

    mock_agent.run.side_effect = mock_run
    mock_agent_class.return_value = mock_agent

    query = "Test query"
    result, debug_info = await run_refiner(query, debug=True)

    assert result == "Refined query: Test refined output"
    assert debug_info is not None
    assert debug_info["original_query"] == "Test query"
    assert debug_info["system_prompt_used"] is True
    assert "raw_output" in debug_info
    mock_agent.run.assert_awaited_once()


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.RefinerAgent")
@patch("cognivault.agents.refiner.main.LLMFactory.create")
@patch(
    "cognivault.agents.refiner.main.OpenAIConfig.load",
    side_effect=Exception("Config error"),
)
async def test_run_refiner_debug_mode_config_exception(
    mock_config_load, mock_llm_factory, mock_agent_class
):
    """Test run_refiner debug mode when OpenAI config loading fails."""
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm_factory.return_value = mock_llm

    mock_agent = AsyncMock()
    mock_agent.name = "Refiner"

    async def mock_run(context):
        context.add_agent_output("Refiner", "Refined query: Test output")

    mock_agent.run.side_effect = mock_run
    mock_agent_class.return_value = mock_agent

    query = "Test query"
    result, debug_info = await run_refiner(query, debug=True)

    assert result == "Refined query: Test output"
    assert debug_info is not None
    assert debug_info["model"] == "stub-llm"  # Fallback when config fails
    assert debug_info["original_query"] == "Test query"
    assert debug_info["system_prompt_used"] is True


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.RefinerAgent")
async def test_run_refiner_debug_mode_unchanged_output(mock_agent_class):
    """Test run_refiner debug mode with unchanged output."""
    mock_agent = AsyncMock()
    mock_agent.name = "Refiner"

    async def mock_run(context):
        context.add_agent_output("Refiner", "[Unchanged] Original query")

    mock_agent.run.side_effect = mock_run
    mock_agent_class.return_value = mock_agent

    query = "Original query"
    result, debug_info = await run_refiner(query, debug=True)

    assert result == "[Unchanged] Original query"
    assert debug_info is not None
    assert debug_info["refined_query"] == "[Unchanged] Original query"


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.run_refiner")
@patch("sys.argv", ["main.py", "--query", "Refined query: test output"])
async def test_main_strips_refined_query_prefix(mock_run_refiner):
    """Test main function strips 'Refined query:' prefix from output."""
    mock_run_refiner.return_value = ("Refined query: clean output", None)

    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        await main()

    output = captured_output.getvalue()
    assert "clean output" in output
    assert "Refined query: clean output" not in output  # Prefix should be stripped


@patch("sys.argv", ["main.py", "--query", "test query"])
def test_parse_args_with_query():
    """Test argument parsing with query flag."""
    args = parse_args()
    assert args.query == "test query"
    assert args.debug is False


@patch("sys.argv", ["main.py", "--query", "test", "--debug"])
def test_parse_args_with_debug():
    """Test argument parsing with debug flag."""
    args = parse_args()
    assert args.query == "test"
    assert args.debug is True


@patch("sys.argv", ["main.py", "-q", "short flag", "-d"])
def test_parse_args_short_flags():
    """Test argument parsing with short flags."""
    args = parse_args()
    assert args.query == "short flag"
    assert args.debug is True


@patch("sys.argv", ["main.py"])
def test_parse_args_no_arguments():
    """Test argument parsing with no arguments (interactive mode)."""
    args = parse_args()
    assert args.query is None
    assert args.debug is False


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.run_refiner")
@patch("sys.argv", ["main.py", "--query", "test query"])
async def test_main_with_query_flag(mock_run_refiner):
    """Test main function with query flag."""
    mock_run_refiner.return_value = ("Refined output", None)

    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        await main()

    output = captured_output.getvalue()
    assert "🧠 Refiner Output:" in output
    assert "Refined output" in output
    mock_run_refiner.assert_called_once_with("test query", debug=False)


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.run_refiner")
@patch("sys.argv", ["main.py", "--query", "test", "--debug"])
async def test_main_with_debug_flag(mock_run_refiner):
    """Test main function with debug flag."""
    mock_run_refiner.return_value = ("Debug output", {"model": "gpt-4"})

    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        await main()

    output = captured_output.getvalue()
    assert "🧠 Refiner Output:" in output
    assert "Debug output" in output
    mock_run_refiner.assert_called_once_with("test", debug=True)


@pytest.mark.asyncio
@patch("builtins.input", return_value="interactive query")
@patch("cognivault.agents.refiner.main.run_refiner")
@patch("sys.argv", ["main.py"])
async def test_main_interactive_mode(mock_run_refiner, mock_input):
    """Test main function in interactive mode."""
    mock_run_refiner.return_value = ("Interactive output", None)

    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        await main()

    output = captured_output.getvalue()
    assert "🧠 Refiner Output:" in output
    assert "Interactive output" in output
    mock_run_refiner.assert_called_once_with("interactive query", debug=False)


@pytest.mark.asyncio
@patch("builtins.input", return_value="")
@patch("sys.argv", ["main.py"])
@patch("sys.exit")
@patch(
    "cognivault.agents.refiner.main.run_refiner", return_value=("", None)
)  # Mock run_refiner with return value
async def test_main_empty_query_exits(mock_run_refiner, mock_exit, mock_input):
    """Test main function exits when empty query provided."""
    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        await main()

    output = captured_output.getvalue()
    assert "❌ Error: No query provided" in output
    # Check that sys.exit was called with 1
    assert mock_exit.call_count >= 1
    exit_calls = [call for call in mock_exit.call_args_list if call[0] == (1,)]
    assert len(exit_calls) >= 1, (
        f"Expected at least one call with exit code 1, got: {mock_exit.call_args_list}"
    )


@pytest.mark.asyncio
@patch("builtins.input", side_effect=KeyboardInterrupt())
@patch("sys.argv", ["main.py"])
@patch("sys.exit", side_effect=SystemExit)
async def test_main_keyboard_interrupt_in_input(mock_exit, mock_input):
    """Test main function handles KeyboardInterrupt during input."""
    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        with pytest.raises(SystemExit):
            await main()

    output = captured_output.getvalue()
    assert "Exiting..." in output
    mock_exit.assert_called_once_with(0)


@pytest.mark.asyncio
@patch("builtins.input", side_effect=EOFError())
@patch("sys.argv", ["main.py"])
@patch("sys.exit", side_effect=SystemExit)
async def test_main_eof_error_in_input(mock_exit, mock_input):
    """Test main function handles EOFError during input."""
    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        with pytest.raises(SystemExit):
            await main()

    output = captured_output.getvalue()
    assert "Exiting..." in output
    mock_exit.assert_called_once_with(0)


@pytest.mark.asyncio
@patch(
    "cognivault.agents.refiner.main.run_refiner", side_effect=Exception("Test error")
)
@patch("sys.argv", ["main.py", "--query", "test"])
@patch("sys.exit")
async def test_main_handles_exception(mock_exit, mock_run_refiner):
    """Test main function handles exceptions."""
    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        await main()

    output = captured_output.getvalue()
    assert "❌ Error: Test error" in output
    mock_exit.assert_called_once_with(1)


@pytest.mark.asyncio
@patch(
    "cognivault.agents.refiner.main.run_refiner", side_effect=Exception("Test error")
)
@patch("sys.argv", ["main.py", "--query", "test", "--debug"])
@patch("sys.exit")
async def test_main_handles_exception_with_debug(mock_exit, mock_run_refiner):
    """Test main function handles exceptions in debug mode."""
    # Capture stdout and stderr
    captured_output = StringIO()
    captured_error = StringIO()
    with patch("sys.stdout", captured_output), patch("sys.stderr", captured_error):
        await main()

    output = captured_output.getvalue()
    assert "❌ Error: Test error" in output
    mock_exit.assert_called_once_with(1)


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.RefinerAgent")
async def test_run_refiner_debug_mode_plain_output(mock_agent_class):
    """Test run_refiner debug mode with output that doesn't match prefixes."""
    mock_agent = AsyncMock()
    mock_agent.name = "Refiner"

    async def mock_run(context):
        context.add_agent_output("Refiner", "Plain output without prefixes")

    mock_agent.run.side_effect = mock_run
    mock_agent_class.return_value = mock_agent

    query = "Test query"
    result, debug_info = await run_refiner(query, debug=True)

    assert result == "Plain output without prefixes"
    assert debug_info is not None
    assert debug_info["refined_query"] == "Plain output without prefixes"


@pytest.mark.asyncio
@patch("cognivault.agents.refiner.main.run_refiner", side_effect=KeyboardInterrupt())
@patch("sys.argv", ["main.py", "--query", "test"])
@patch("sys.exit", side_effect=SystemExit)
async def test_main_handles_keyboard_interrupt_during_execution(
    mock_exit, mock_run_refiner
):
    """Test main function handles KeyboardInterrupt during refiner execution."""
    # Capture stdout
    captured_output = StringIO()
    with patch("sys.stdout", captured_output):
        with pytest.raises(SystemExit):
            await main()

    output = captured_output.getvalue()
    assert "❌ Interrupted by user" in output
    mock_exit.assert_called_once_with(1)
