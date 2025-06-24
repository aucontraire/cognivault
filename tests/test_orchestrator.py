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
