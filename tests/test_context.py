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


def test_update_and_get_user_config():
    context = AgentContext(query="AI alignment")
    context.update_user_config({"verbosity": "high", "style": "explanatory"})

    assert context.user_config["verbosity"] == "high"
    assert context.get_user_config("style") == "explanatory"
    assert (
        context.get_user_config("nonexistent", default="default_val") == "default_val"
    )


def test_set_and_get_final_synthesis():
    context = AgentContext(query="Ethics in AI")
    context.set_final_synthesis("AI ethics involves balancing risks and benefits.")

    assert (
        context.get_final_synthesis()
        == "AI ethics involves balancing risks and benefits."
    )


def test_log_trace():
    context = AgentContext(query="History of AI")
    context.log_trace(
        "Historian", input_data="Query: History", output_data="AI history summary"
    )

    assert "Historian" in context.agent_trace
    assert isinstance(context.agent_trace["Historian"], list)
    assert context.agent_trace["Historian"][0]["input"] == "Query: History"
    assert "timestamp" in context.agent_trace["Historian"][0]
