import pytest
from unittest.mock import AsyncMock, patch
import asyncio
import logging
from cognivault.orchestrator import AgentOrchestrator
from cognivault.orchestrator import BaseAgent, AgentContext
from cognivault.config.app_config import (
    ApplicationConfig,
    Environment,
    set_config,
    reset_config,
)


class TimeoutAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Timeout")

    async def run(self, context: AgentContext):
        # Use configuration for timeout calculation
        from cognivault.config.app_config import get_config

        config = get_config()
        timeout_seconds = config.get_timeout_for_environment()
        await asyncio.sleep(timeout_seconds + 5)


class SilentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Silent")

    async def run(self, context: AgentContext):
        context.add_agent_output(self.name, "Silent success")
        # Deliberately no context.log_trace()


@patch("cognivault.agents.refiner.agent.RefinerAgent.run", new_callable=AsyncMock)
def test_orchestrator_with_critic_enabled(mock_refiner_run):
    # Simulate Refiner adding output to context
    async def mock_run(context):
        context.add_agent_output(
            "Refiner", "Refined insight: Democracy is gradually improving in Mexico."
        )

    mock_refiner_run.side_effect = mock_run

    orchestrator = AgentOrchestrator(critic_enabled=True)
    context = asyncio.run(orchestrator.run("Is democracy in Mexico improving?"))

    assert "Refiner" in context.agent_outputs
    assert "Critic" in context.agent_outputs
    assert "Historian" in context.agent_outputs
    assert "Synthesis" in context.agent_outputs


@patch("cognivault.agents.refiner.agent.RefinerAgent.run", new_callable=AsyncMock)
def test_orchestrator_without_critic(mock_generate):
    async def fake_run(context):
        context.add_agent_output("Refiner", "Mocked refined output")
        context.log_trace(
            "Refiner",
            input_data="What changed after the elections?",
            output_data="Mocked refined output",
        )

    mock_generate.side_effect = fake_run

    orchestrator = AgentOrchestrator(critic_enabled=False)
    context = asyncio.run(orchestrator.run("What changed after the elections?"))

    assert "Refiner" in context.agent_outputs
    assert "Historian" in context.agent_outputs
    assert "Critic" not in context.agent_outputs
    assert "Synthesis" in context.agent_outputs

    for agent_name in ["Refiner", "Historian", "Synthesis"]:
        assert agent_name in context.agent_trace
        assert isinstance(context.agent_trace[agent_name], list)
        assert len(context.agent_trace[agent_name]) >= 1
        assert "timestamp" in context.agent_trace[agent_name][0]
        assert "input" in context.agent_trace[agent_name][0]
        assert "output" in context.agent_trace[agent_name][0]

    assert context.final_synthesis is not None
    assert "Critic" not in context.final_synthesis


@patch("cognivault.agents.refiner.agent.RefinerAgent.run", new_callable=AsyncMock)
def test_orchestrator_with_empty_query(mock_refiner_run):
    async def mock_run(context):
        context.add_agent_output("Refiner", "[Refined] Empty query handled")
        context.log_trace(
            "Refiner", input_data="", output_data="[Refined] Empty query handled"
        )

    mock_refiner_run.side_effect = mock_run

    orchestrator = AgentOrchestrator()
    context = asyncio.run(orchestrator.run("   "))

    assert "Refiner" in context.agent_outputs
    assert context.agent_outputs["Refiner"]
    # Check agent_trace for Refiner
    assert "Refiner" in context.agent_trace
    assert isinstance(context.agent_trace["Refiner"], list)
    assert len(context.agent_trace["Refiner"]) >= 1
    assert "timestamp" in context.agent_trace["Refiner"][0]
    assert "input" in context.agent_trace["Refiner"][0]
    assert "output" in context.agent_trace["Refiner"][0]
    assert context.final_synthesis is not None


@patch("cognivault.agents.refiner.agent.RefinerAgent.run", new_callable=AsyncMock)
def test_orchestrator_with_only_refiner(mock_refiner_run):
    async def mock_run(context):
        context.add_agent_output("Refiner", "[Refined] Mocked single agent execution")
        context.log_trace(
            "Refiner",
            input_data="Test single agent execution",
            output_data="[Refined] Mocked single agent execution",
        )

    mock_refiner_run.side_effect = mock_run

    orchestrator = AgentOrchestrator(agents_to_run=["refiner"])
    context = asyncio.run(orchestrator.run("Test single agent execution"))

    # Only Refiner should be present
    assert list(context.agent_outputs.keys()) == ["Refiner"]
    assert "Refiner" in context.agent_outputs
    assert "Historian" not in context.agent_outputs
    assert "Critic" not in context.agent_outputs
    assert "Synthesis" not in context.agent_outputs

    # Check agent_trace for Refiner
    assert "Refiner" in context.agent_trace
    assert isinstance(context.agent_trace["Refiner"], list)
    assert len(context.agent_trace["Refiner"]) >= 1
    assert "timestamp" in context.agent_trace["Refiner"][0]
    assert "input" in context.agent_trace["Refiner"][0]
    assert "output" in context.agent_trace["Refiner"][0]

    assert context.final_synthesis is None  # Synthesis should not have run


def test_orchestrator_with_only_critic():
    orchestrator = AgentOrchestrator(agents_to_run=["critic"])
    context = asyncio.run(orchestrator.run("Test critic agent in isolation"))

    assert list(context.agent_outputs.keys()) == ["Critic"]
    assert "Critic" in context.agent_outputs
    assert (
        "No refined output available from RefinerAgent to critique."
        in context.agent_outputs["Critic"]
    )
    # Check agent_trace for Critic
    assert "Critic" in context.agent_trace
    assert isinstance(context.agent_trace["Critic"], list)
    assert len(context.agent_trace["Critic"]) >= 1
    assert "timestamp" in context.agent_trace["Critic"][0]
    assert "input" in context.agent_trace["Critic"][0]
    assert "output" in context.agent_trace["Critic"][0]
    assert context.final_synthesis is None


def test_orchestrator_with_only_historian():
    orchestrator = AgentOrchestrator(agents_to_run=["historian"])
    context = asyncio.run(orchestrator.run("Test historian agent in isolation"))

    assert list(context.agent_outputs.keys()) == ["Historian"]
    assert "Historian" in context.agent_outputs
    assert "Note from" in context.agent_outputs["Historian"]
    # Check agent_trace for Historian
    assert "Historian" in context.agent_trace
    assert isinstance(context.agent_trace["Historian"], list)
    assert len(context.agent_trace["Historian"]) >= 1
    assert "timestamp" in context.agent_trace["Historian"][0]
    assert "input" in context.agent_trace["Historian"][0]
    assert "output" in context.agent_trace["Historian"][0]
    assert context.final_synthesis is None


def test_orchestrator_with_only_synthesis():
    orchestrator = AgentOrchestrator(agents_to_run=["synthesis"])
    context = asyncio.run(orchestrator.run("Test synthesis agent in isolation"))

    assert list(context.agent_outputs.keys()) == ["Synthesis"]
    assert "Synthesis" in context.agent_outputs
    assert context.agent_outputs["Synthesis"].strip() == ""
    # Check agent_trace for Synthesis
    assert "Synthesis" in context.agent_trace
    assert isinstance(context.agent_trace["Synthesis"], list)
    assert len(context.agent_trace["Synthesis"]) >= 1
    assert "timestamp" in context.agent_trace["Synthesis"][0]
    assert "input" in context.agent_trace["Synthesis"][0]
    assert "output" in context.agent_trace["Synthesis"][0]
    assert context.final_synthesis == ""


def test_orchestrator_with_invalid_only_name():
    orchestrator = AgentOrchestrator(agents_to_run=["fakeagent"])
    context = asyncio.run(orchestrator.run("Test invalid agent name"))

    assert context.agent_outputs == {}
    assert context.final_synthesis is None


@patch("cognivault.agents.refiner.agent.RefinerAgent.run", new_callable=AsyncMock)
def test_orchestrator_with_mixed_valid_and_invalid_agents(mock_refiner_run, capfd):
    async def mock_run(context):
        context.add_agent_output("Refiner", "Handled valid agent")
        context.log_trace(
            "Refiner",
            input_data="Mix valid and invalid agents",
            output_data="Handled valid agent",
        )

    mock_refiner_run.side_effect = mock_run

    orchestrator = AgentOrchestrator(agents_to_run=["refiner", "fakeagent"])
    context = asyncio.run(orchestrator.run("Mix valid and invalid agents"))

    # Should run only the valid agent
    assert "Refiner" in context.agent_outputs
    assert "fakeagent" not in context.agent_outputs
    # Check agent_trace for Refiner
    assert "Refiner" in context.agent_trace
    assert isinstance(context.agent_trace["Refiner"], list)
    assert len(context.agent_trace["Refiner"]) >= 1
    assert "timestamp" in context.agent_trace["Refiner"][0]
    assert "input" in context.agent_trace["Refiner"][0]
    assert "output" in context.agent_trace["Refiner"][0]
    out, _ = capfd.readouterr()
    assert "[DEBUG] Unknown agent name: fakeagent" in out


def test_orchestrator_with_timeout_agent():
    orchestrator = AgentOrchestrator(agents_to_run=[])
    orchestrator.agents = [TimeoutAgent()]
    context = asyncio.run(orchestrator.run("This will timeout"))
    assert "Timeout" not in context.agent_outputs
    assert context.agent_trace["Timeout"][0]["output"]["success"] is True
    assert context.final_synthesis is None


# New test: retry logic with a flaky agent
class FlakyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Flaky")
        self.run_count = 0

    async def run(self, context: AgentContext):
        self.run_count += 1
        if self.run_count < 2:
            raise Exception("Temporary failure")
        context.add_agent_output(self.name, "Recovered")
        context.log_trace(
            self.name, input_data="Handle retries", output_data="Recovered"
        )


def test_orchestrator_retries_flaky_agent():
    FlakyAgent.run_count = 0
    orchestrator = AgentOrchestrator(agents_to_run=[])
    orchestrator.agents = [FlakyAgent()]
    context = asyncio.run(orchestrator.run("Handle retries"))
    assert "Flaky" in context.agent_outputs
    assert context.agent_outputs["Flaky"] == "Recovered"
    # Check agent_trace for Flaky
    assert "Flaky" in context.agent_trace
    assert isinstance(context.agent_trace["Flaky"], list)
    assert len(context.agent_trace["Flaky"]) >= 1
    assert "timestamp" in context.agent_trace["Flaky"][0]
    assert "input" in context.agent_trace["Flaky"][0]
    assert "output" in context.agent_trace["Flaky"][0]


def test_orchestrator_skips_log_trace_if_handled_externally(caplog):
    caplog.set_level(logging.DEBUG)

    orchestrator = AgentOrchestrator(agents_to_run=[])
    orchestrator.agents = [SilentAgent()]
    context = asyncio.run(orchestrator.run("Test skipping log_trace"))

    assert "Silent" in context.agent_outputs
    assert context.agent_outputs["Silent"] == "Silent success"
    assert context.agent_trace["Silent"][0]["output"]["success"] is True

    # This is the exact line we're trying to hit
    assert any(
        "Logged trace for agent 'Silent'" in message for message in caplog.messages
    )


def test_orchestrator_core_agent_creation_failure():
    """Test orchestrator handles core agent creation failures."""
    from unittest.mock import Mock, patch

    # Mock the registry to fail when creating a core agent
    mock_registry = Mock()
    mock_registry.create_agent.side_effect = ValueError("Mock agent creation failed")
    # Mock dependency resolution to return default agents
    mock_registry.resolve_dependencies.return_value = [
        "refiner",
        "historian",
        "synthesis",
    ]

    with patch(
        "cognivault.orchestrator.get_agent_registry", return_value=mock_registry
    ):
        # This should raise the ValueError when trying to create core agents
        with pytest.raises(ValueError, match="Mock agent creation failed"):
            AgentOrchestrator()


class TestOrchestratorConfiguration:
    """Test suite for orchestrator configuration integration."""

    def setup_method(self):
        """Reset global configuration before each test."""
        reset_config()

    def teardown_method(self):
        """Reset global configuration after each test."""
        reset_config()

    def test_orchestrator_uses_config_defaults(self):
        """Test that orchestrator uses configuration defaults."""
        # Set up custom configuration
        config = ApplicationConfig()
        config.execution.critic_enabled = False
        config.execution.default_agents = ["refiner", "synthesis"]
        set_config(config)

        # Create orchestrator without explicit parameters
        orchestrator = AgentOrchestrator()

        # Should use configuration defaults
        assert orchestrator.critic_enabled is False
        agent_names = [agent.name for agent in orchestrator.agents]
        assert "Refiner" in agent_names
        assert "Synthesis" in agent_names
        assert "Critic" not in agent_names

    def test_orchestrator_explicit_params_override_config(self):
        """Test that explicit parameters override configuration."""
        # Set up custom configuration
        config = ApplicationConfig()
        config.execution.critic_enabled = False
        set_config(config)

        # Create orchestrator with explicit critic_enabled=True
        orchestrator = AgentOrchestrator(critic_enabled=True)

        # Should use explicit parameter, not configuration
        assert orchestrator.critic_enabled is True
        agent_names = [agent.name for agent in orchestrator.agents]
        assert "Critic" in agent_names

    def test_orchestrator_uses_config_timeout_settings(self):
        """Test that orchestrator uses configuration timeout settings."""
        config = ApplicationConfig()
        config.execution.max_retries = 5
        config.execution.timeout_seconds = 20
        config.execution.retry_delay_seconds = 2.0
        config.environment = Environment.TESTING
        config.testing.test_timeout_multiplier = 2.0
        set_config(config)

        orchestrator = AgentOrchestrator(agents_to_run=[])

        # Create a test agent that tracks configuration usage
        class ConfigTestAgent(BaseAgent):
            def __init__(self):
                super().__init__("ConfigTest")
                self.config_values = {}

            async def run(self, context: AgentContext):
                from cognivault.config.app_config import get_config

                config = get_config()
                self.config_values = {
                    "max_retries": config.execution.max_retries,
                    "timeout": config.get_timeout_for_environment(),
                    "retry_delay": config.execution.retry_delay_seconds,
                }
                context.add_agent_output(self.name, "Test completed")

        test_agent = ConfigTestAgent()
        orchestrator.agents = [test_agent]

        # Run the orchestrator (it will access configuration internally)
        asyncio.run(orchestrator.run("Test configuration usage"))

        # Verify configuration was accessed correctly
        assert test_agent.config_values["max_retries"] == 5
        assert (
            test_agent.config_values["timeout"] == 40
        )  # 20 * 2.0 multiplier for testing
        assert test_agent.config_values["retry_delay"] == 2.0

    def test_orchestrator_dynamic_agent_pipeline(self):
        """Test orchestrator with dynamically configured agent pipeline."""
        config = ApplicationConfig()
        config.execution.default_agents = ["historian", "refiner", "synthesis"]
        config.execution.critic_enabled = True
        set_config(config)

        orchestrator = AgentOrchestrator()

        # Should follow configured pipeline order with critic inserted appropriately
        agent_names = [agent.name for agent in orchestrator.agents]

        # Historian should be first (as configured)
        assert agent_names[0] == "Historian"
        # Refiner should be second
        assert agent_names[1] == "Refiner"
        # Synthesis should be third (configured order, no dependencies)
        assert agent_names[2] == "Synthesis"
        # Critic should be last (depends on refiner, so comes after dependency resolution)
        assert agent_names[3] == "Critic"

    def test_orchestrator_critic_enabled_no_refiner_in_pipeline(self):
        """Test critic handling when refiner is not in the default agent pipeline."""
        config = ApplicationConfig()
        config.execution.default_agents = ["historian", "synthesis"]  # No refiner
        config.execution.critic_enabled = True
        set_config(config)

        orchestrator = AgentOrchestrator()

        # When refiner is not in pipeline but critic is enabled,
        # critic should be appended to the end
        agent_names = [agent.name for agent in orchestrator.agents]

        assert "Historian" in agent_names
        assert "Synthesis" in agent_names
        assert "Critic" in agent_names  # Should be appended
        assert "Refiner" not in agent_names

    def test_orchestrator_remove_critic_from_pipeline(self):
        """Test removing critic from pipeline when critic_enabled=False."""
        config = ApplicationConfig()
        config.execution.default_agents = [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]
        config.execution.critic_enabled = False  # Explicitly disable critic
        set_config(config)

        orchestrator = AgentOrchestrator()

        # Critic should be removed from the pipeline
        agent_names = [agent.name for agent in orchestrator.agents]

        assert "Refiner" in agent_names
        assert "Historian" in agent_names
        assert "Synthesis" in agent_names
        assert "Critic" not in agent_names  # Should be removed
