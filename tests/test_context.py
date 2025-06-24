import pytest
from cognivault.context import AgentContext


def test_add_and_get_agent_output():
    context = AgentContext(query="What is democracy?")
    context.add_agent_output("Refiner", "Structured explanation of democracy.")

    assert "Refiner" in context.agent_outputs
    assert context.get_output("Refiner") == "Structured explanation of democracy."


def test_get_output_returns_none_for_missing_agent():
    context = AgentContext(query="History of voting rights?")
    assert context.get_output("Historian") is None


def test_retrieved_notes_are_optional():
    context = AgentContext(query="Constitutional reforms", retrieved_notes=None)
    assert context.retrieved_notes is None or isinstance(context.retrieved_notes, list)


def test_user_config_and_final_synthesis_defaults():
    context = AgentContext(query="Impact of social media on elections")
    assert context.user_config == {}
    assert context.final_synthesis is None
