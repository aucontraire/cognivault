import pytest
import asyncio
import logging
from cognivault.orchestrator import AgentOrchestrator
from cognivault.orchestrator import BaseAgent, AgentContext

TIMEOUT_SECONDS = 10  # Should match the orchestrator's timeout, adjust if needed


class TimeoutAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Timeout")

    async def run(self, context: AgentContext):
        await asyncio.sleep(TIMEOUT_SECONDS + 5)


class SilentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Silent")

    async def run(self, context: AgentContext):
        context.add_agent_output(self.name, "Silent success")
        # Deliberately no context.log_trace()


def test_orchestrator_with_critic_enabled():
    orchestrator = AgentOrchestrator(critic_enabled=True)
    context = asyncio.run(orchestrator.run("Is democracy in Mexico improving?"))

    # All agents should run
    assert "Refiner" in context.agent_outputs
    assert "Historian" in context.agent_outputs
    assert "Critic" in context.agent_outputs
    assert "Synthesis" in context.agent_outputs

    # Check agent_trace for all agents
    for agent_name in ["Refiner", "Historian", "Critic", "Synthesis"]:
        assert agent_name in context.agent_trace
        assert isinstance(context.agent_trace[agent_name], list)
        assert len(context.agent_trace[agent_name]) >= 1
        assert "timestamp" in context.agent_trace[agent_name][0]
        assert "input" in context.agent_trace[agent_name][0]
        assert "output" in context.agent_trace[agent_name][0]

    assert context.final_synthesis is not None
    assert "Refiner" in context.final_synthesis or context.final_synthesis == ""
    if context.final_synthesis:
        assert "Historian" in context.final_synthesis
    assert "Critic" in context.agent_outputs


def test_orchestrator_without_critic():
    orchestrator = AgentOrchestrator(critic_enabled=False)
    context = asyncio.run(orchestrator.run("What changed after the elections?"))

    assert "Refiner" in context.agent_outputs
    assert "Historian" in context.agent_outputs
    assert "Critic" not in context.agent_outputs
    assert "Synthesis" in context.agent_outputs

    # Check agent_trace for present agents
    for agent_name in ["Refiner", "Historian", "Synthesis"]:
        assert agent_name in context.agent_trace
        assert isinstance(context.agent_trace[agent_name], list)
        assert len(context.agent_trace[agent_name]) >= 1
        assert "timestamp" in context.agent_trace[agent_name][0]
        assert "input" in context.agent_trace[agent_name][0]
        assert "output" in context.agent_trace[agent_name][0]

    assert context.final_synthesis is not None
    assert "Critic" not in context.final_synthesis


def test_orchestrator_with_empty_query():
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


def test_orchestrator_with_only_refiner():
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
    assert "No refined output found" in context.agent_outputs["Critic"]
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


# New test: mixed valid and invalid agent names
def test_orchestrator_with_mixed_valid_and_invalid_agents(capfd):
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
    assert "Timeout" not in context.agent_trace
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
    assert "Silent" not in context.agent_trace

    # This is the exact line we're trying to hit
    assert any(
        "Skipping log_trace because agent 'Silent'" in message
        for message in caplog.messages
    )
