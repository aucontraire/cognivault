import pytest
from cognivault.agents.historian.main import run_historian


@pytest.mark.asyncio
async def test_run_historian(monkeypatch):
    class DummyHistorianAgent:
        name = "Historian"

        async def run(self, context):
            context.add_agent_output("Historian", "Test Output")
            return context

    monkeypatch.setattr(
        "cognivault.agents.historian.main.HistorianAgent", DummyHistorianAgent
    )

    result = await run_historian("What happened in 2024?")
    assert result == "Test Output"
