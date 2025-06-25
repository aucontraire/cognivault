import pytest
from cognivault.agents.critic import main as critic_main


@pytest.mark.asyncio
async def test_run_critic(monkeypatch):
    class MockCriticAgent:
        def __init__(self):
            self.name = "Critic"

        async def run(self, context):
            class Result:
                agent_outputs = {"Critic": "Mocked Critic Output"}

            return Result()

    # Patch the CriticAgent class in the correct module
    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)

    result = await critic_main.run_critic("test input")
    assert result == "Mocked Critic Output"
