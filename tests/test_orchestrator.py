import pytest
from cognivault.orchestrator import AgentOrchestrator
from cognivault.context import AgentContext


def test_orchestrator_with_critic_enabled():
    orchestrator = AgentOrchestrator(critic_enabled=True)
    context = orchestrator.run("Is democracy in Mexico improving?")

    # All agents should run
    assert "Refiner" in context.agent_outputs
    assert "Historian" in context.agent_outputs
    assert "Critic" in context.agent_outputs
    assert "Synthesis" in context.agent_outputs

    assert context.final_synthesis is not None
    assert "Refiner" in context.final_synthesis
    assert "Historian" in context.final_synthesis
    assert "Critic" in context.final_synthesis


def test_orchestrator_without_critic():
    orchestrator = AgentOrchestrator(critic_enabled=False)
    context = orchestrator.run("What changed after the elections?")

    assert "Refiner" in context.agent_outputs
    assert "Historian" in context.agent_outputs
    assert "Critic" not in context.agent_outputs
    assert "Synthesis" in context.agent_outputs

    assert context.final_synthesis is not None
    assert "Critic" not in context.final_synthesis


def test_orchestrator_with_empty_query():
    orchestrator = AgentOrchestrator()
    context = orchestrator.run("   ")

    assert "Refiner" in context.agent_outputs
    assert context.agent_outputs["Refiner"]
    assert context.final_synthesis is not None


def test_orchestrator_with_only_refiner():
    orchestrator = AgentOrchestrator(agents_to_run=["refiner"])
    context = orchestrator.run("Test single agent execution")

    # Only Refiner should be present
    assert list(context.agent_outputs.keys()) == ["Refiner"]
    assert "Refiner" in context.agent_outputs
    assert "Historian" not in context.agent_outputs
    assert "Critic" not in context.agent_outputs
    assert "Synthesis" not in context.agent_outputs

    assert context.final_synthesis is None  # Synthesis should not have run


def test_orchestrator_with_only_critic():
    orchestrator = AgentOrchestrator(agents_to_run=["critic"])
    context = orchestrator.run("Test critic agent in isolation")

    assert list(context.agent_outputs.keys()) == ["Critic"]
    assert "Critic" in context.agent_outputs
    assert "No refined output found" in context.agent_outputs["Critic"]
    assert context.final_synthesis is None


def test_orchestrator_with_only_historian():
    orchestrator = AgentOrchestrator(agents_to_run=["historian"])
    context = orchestrator.run("Test historian agent in isolation")

    assert list(context.agent_outputs.keys()) == ["Historian"]
    assert "Historian" in context.agent_outputs
    assert "Note from" in context.agent_outputs["Historian"]
    assert context.final_synthesis is None


def test_orchestrator_with_only_synthesis():
    orchestrator = AgentOrchestrator(agents_to_run=["synthesis"])
    context = orchestrator.run("Test synthesis agent in isolation")

    assert list(context.agent_outputs.keys()) == ["Synthesis"]
    assert "Synthesis" in context.agent_outputs
    assert context.agent_outputs["Synthesis"].strip() == ""
    assert context.final_synthesis == ""


def test_orchestrator_with_invalid_only_name():
    orchestrator = AgentOrchestrator(agents_to_run=["fakeagent"])
    context = orchestrator.run("Test invalid agent name")

    assert context.agent_outputs == {}
    assert context.final_synthesis is None
