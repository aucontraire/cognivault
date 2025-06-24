import pytest
from cognivault.agents.critic import main as critic_main


def test_run_critic(monkeypatch):
    class MockCriticAgent:
        def run(self, context):
            class Result:
                agent_outputs = {"Critic": "Mocked Critic Output"}

            return Result()

    # Patch the CriticAgent class in the correct module
    monkeypatch.setattr("cognivault.agents.critic.main.CriticAgent", MockCriticAgent)

    result = critic_main.run_critic("test input")
    assert result == "Mocked Critic Output"
