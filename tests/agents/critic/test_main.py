import pytest
import asyncio
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from cognivault.agents.critic import main as critic_main


@pytest.mark.asyncio
async def test_run_critic_basic(monkeypatch):
    """Test basic run_critic functionality."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLMFactory:
        @staticmethod
        def create():
            return MagicMock()

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)

    result, debug_info = await critic_main.run_critic("test input")
    assert result == "Mocked Critic Output"
    assert debug_info is None


@pytest.mark.asyncio
async def test_run_critic_with_debug(monkeypatch, capsys):
    """Test run_critic with debug mode enabled."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLMFactory:
        @staticmethod
        def create():
            return MagicMock()

    class MockOpenAIConfig:
        @staticmethod
        def load():
            config = MagicMock()
            config.model = "gpt-4"
            return config

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)
    monkeypatch.setattr("cognivault.agents.critic.main.OpenAIConfig", MockOpenAIConfig)

    result, debug_info = await critic_main.run_critic("test input", debug=True)

    assert result == "Mocked Critic Output"
    assert debug_info is not None
    assert debug_info["model"] == "gpt-4"
    assert debug_info["original_query"] == "test input"
    assert debug_info["system_prompt_used"] is True

    # Check debug output was printed
    captured = capsys.readouterr()
    assert "[DEBUG] Input query: 'test input'" in captured.out
    assert "[DEBUG] Model: gpt-4" in captured.out
    assert (
        "[DEBUG] Simulated Refiner output for critique: 'Refined query: test input'"
        in captured.out
    )
    assert "[DEBUG] Running CriticAgent..." in captured.out
    assert "[DEBUG] Raw agent output: Mocked Critic Output" in captured.out
    assert "[DEBUG] Processing complete" in captured.out


@pytest.mark.asyncio
async def test_run_critic_debug_openai_config_exception(monkeypatch, capsys):
    """Test run_critic debug mode when OpenAIConfig.load() fails."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLM:
        model_name = "stub-llm"

    class MockLLMFactory:
        @staticmethod
        def create():
            return MockLLM()

    class MockOpenAIConfig:
        @staticmethod
        def load():
            raise Exception("Config not found")

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)
    monkeypatch.setattr("cognivault.agents.critic.main.OpenAIConfig", MockOpenAIConfig)

    result, debug_info = await critic_main.run_critic("test input", debug=True)

    assert result == "Mocked Critic Output"
    assert debug_info is not None
    assert debug_info["model"] == "stub-llm"

    # Check debug output shows fallback model
    captured = capsys.readouterr()
    assert "[DEBUG] Model: stub-llm" in captured.out


@pytest.mark.asyncio
async def test_run_critic_debug_no_model_name(monkeypatch, capsys):
    """Test run_critic debug mode when LLM has no model_name attribute."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLMWithoutModelName:
        pass  # No model_name attribute

    class MockLLMFactory:
        @staticmethod
        def create():
            return MockLLMWithoutModelName()

    class MockOpenAIConfig:
        @staticmethod
        def load():
            raise Exception("Config not found")

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)
    monkeypatch.setattr("cognivault.agents.critic.main.OpenAIConfig", MockOpenAIConfig)

    result, debug_info = await critic_main.run_critic("test input", debug=True)

    assert result == "Mocked Critic Output"
    assert debug_info is not None
    assert debug_info["model"] == "unknown"

    # Check debug output shows unknown model
    captured = capsys.readouterr()
    assert "[DEBUG] Model: unknown" in captured.out


@pytest.mark.asyncio
async def test_run_critic_no_output(monkeypatch):
    """Test run_critic when agent returns no output."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return None  # No output

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLMFactory:
        @staticmethod
        def create():
            return MagicMock()

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)

    result, debug_info = await critic_main.run_critic("test input")
    assert result == "[No output]"
    assert debug_info is None


def test_parse_args_with_query():
    """Test argument parsing with query provided."""
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["main.py", "--query", "test query", "--debug"]
        args = critic_main.parse_args()
        assert args.query == "test query"
        assert args.debug is True
    finally:
        sys.argv = original_argv


def test_parse_args_without_query():
    """Test argument parsing without query (interactive mode)."""
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["main.py"]
        args = critic_main.parse_args()
        assert args.query is None
        assert args.debug is False
    finally:
        sys.argv = original_argv


def test_parse_args_short_flags():
    """Test argument parsing with short flags."""
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["main.py", "-q", "test query", "-d"]
        args = critic_main.parse_args()
        assert args.query == "test query"
        assert args.debug is True
    finally:
        sys.argv = original_argv


@pytest.mark.asyncio
async def test_main_with_query(monkeypatch, capsys):
    """Test main function with query provided."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLMFactory:
        @staticmethod
        def create():
            return MagicMock()

    def mock_parse_args():
        args = MagicMock()
        args.query = "test query"
        args.debug = False
        return args

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)
    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)

    await critic_main.main()

    captured = capsys.readouterr()
    assert "🤔 Critic Output:" in captured.out
    assert "Mocked Critic Output" in captured.out


@pytest.mark.asyncio
async def test_main_interactive_mode(monkeypatch, capsys):
    """Test main function in interactive mode."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLMFactory:
        @staticmethod
        def create():
            return MagicMock()

    def mock_parse_args():
        args = MagicMock()
        args.query = None
        args.debug = False
        return args

    def mock_input(prompt):
        return "interactive query"

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)
    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)
    monkeypatch.setattr("builtins.input", mock_input)

    await critic_main.main()

    captured = capsys.readouterr()
    assert "🤔 Critic Output:" in captured.out
    assert "Mocked Critic Output" in captured.out


@pytest.mark.asyncio
async def test_main_keyboard_interrupt_interactive(monkeypatch, capsys):
    """Test main function handling KeyboardInterrupt in interactive mode."""

    def mock_parse_args():
        args = MagicMock()
        args.query = None
        args.debug = False
        return args

    def mock_input(prompt):
        raise KeyboardInterrupt()

    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)
    monkeypatch.setattr("builtins.input", mock_input)

    with pytest.raises(SystemExit) as exc_info:
        await critic_main.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Exiting..." in captured.out


@pytest.mark.asyncio
async def test_main_eoferror_interactive(monkeypatch, capsys):
    """Test main function handling EOFError in interactive mode."""

    def mock_parse_args():
        args = MagicMock()
        args.query = None
        args.debug = False
        return args

    def mock_input(prompt):
        raise EOFError()

    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)
    monkeypatch.setattr("builtins.input", mock_input)

    with pytest.raises(SystemExit) as exc_info:
        await critic_main.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Exiting..." in captured.out


@pytest.mark.asyncio
async def test_main_empty_query(monkeypatch, capsys):
    """Test main function with empty query."""

    def mock_parse_args():
        args = MagicMock()
        args.query = "   "  # Empty/whitespace query
        args.debug = False
        return args

    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)

    with pytest.raises(SystemExit) as exc_info:
        await critic_main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "❌ Error: No query provided" in captured.out


@pytest.mark.asyncio
async def test_main_keyboard_interrupt_during_execution(monkeypatch, capsys):
    """Test main function handling KeyboardInterrupt during execution."""

    def mock_parse_args():
        args = MagicMock()
        args.query = "test query"
        args.debug = False
        return args

    async def mock_run_critic(query, debug=False):
        raise KeyboardInterrupt()

    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)
    monkeypatch.setattr("cognivault.agents.critic.main.run_critic", mock_run_critic)

    with pytest.raises(SystemExit) as exc_info:
        await critic_main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "❌ Interrupted by user" in captured.out


@pytest.mark.asyncio
async def test_main_generic_exception(monkeypatch, capsys):
    """Test main function handling generic exception."""

    def mock_parse_args():
        args = MagicMock()
        args.query = "test query"
        args.debug = False
        return args

    async def mock_run_critic(query, debug=False):
        raise Exception("Test error")

    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)
    monkeypatch.setattr("cognivault.agents.critic.main.run_critic", mock_run_critic)

    with pytest.raises(SystemExit) as exc_info:
        await critic_main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "❌ Error: Test error" in captured.out


@pytest.mark.asyncio
async def test_main_generic_exception_with_debug(monkeypatch, capsys):
    """Test main function handling generic exception with debug mode."""

    def mock_parse_args():
        args = MagicMock()
        args.query = "test query"
        args.debug = True
        return args

    async def mock_run_critic(query, debug=False):
        raise Exception("Test error")

    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)
    monkeypatch.setattr("cognivault.agents.critic.main.run_critic", mock_run_critic)

    with pytest.raises(SystemExit) as exc_info:
        await critic_main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "❌ Error: Test error" in captured.out
    # Debug mode should show traceback, but we can't easily test the exact output


@pytest.mark.asyncio
async def test_main_with_debug_spacing(monkeypatch, capsys):
    """Test main function with debug mode shows proper spacing."""

    class MockContext:
        def __init__(self, query):
            self.query = query
            self.agent_outputs = {}

        def add_agent_output(self, agent_name, output):
            self.agent_outputs[agent_name] = output

        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return context

    class MockLLMFactory:
        @staticmethod
        def create():
            return MagicMock()

    def mock_parse_args():
        args = MagicMock()
        args.query = "test query"
        args.debug = True
        return args

    class MockOpenAIConfig:
        @staticmethod
        def load():
            config = MagicMock()
            config.model = "gpt-4"
            return config

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)
    monkeypatch.setattr("cognivault.agents.critic.main.AgentContext", MockContext)
    monkeypatch.setattr("cognivault.agents.critic.main.parse_args", mock_parse_args)
    monkeypatch.setattr("cognivault.agents.critic.main.OpenAIConfig", MockOpenAIConfig)

    await critic_main.main()

    captured = capsys.readouterr()
    # Check for debug output followed by spacing and then main output
    assert "[DEBUG]" in captured.out
    assert "🤔 Critic Output:" in captured.out
    assert "Mocked Critic Output" in captured.out
