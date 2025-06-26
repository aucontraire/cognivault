import pytest
from cognivault.agents.critic import main as critic_main


@pytest.mark.asyncio
async def test_run_critic(monkeypatch):
    class MockContext:
        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self):
            self.name = "Critic"

        async def run(self, context):
            return MockContext()

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)

    result = await critic_main.run_critic("test input")
    assert result == "Mocked Critic Output"
