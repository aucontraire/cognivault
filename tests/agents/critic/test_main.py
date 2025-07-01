import pytest
from unittest.mock import MagicMock
from cognivault.agents.critic import main as critic_main


@pytest.mark.asyncio
async def test_run_critic(monkeypatch):
    class MockContext:
        def get_output(self, agent_name):
            return "Mocked Critic Output"

    class MockCriticAgent:
        def __init__(self, llm):
            self.name = "Critic"
            self.llm = llm

        async def run(self, context):
            return MockContext()

    class MockLLMFactory:
        @staticmethod
        def create():
            return MagicMock()

    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)
    monkeypatch.setattr("cognivault.agents.critic.main.LLMFactory", MockLLMFactory)

    result = await critic_main.run_critic("test input")
    assert result == "Mocked Critic Output"
