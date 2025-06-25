import pytest
from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


@pytest.mark.anyio
async def test_base_agent_run_concrete():
    class ConcreteAgent(BaseAgent):
        async def run(self, context: AgentContext) -> AgentContext:
            context.agent_outputs[self.name] = "mocked"
            return context

    agent = ConcreteAgent(name="TestAgent")
    context = AgentContext(query="test")
    result = await agent.run(context)

    assert isinstance(result, AgentContext)
    assert result.agent_outputs["TestAgent"] == "mocked"


def test_base_agent_run_abstract_error():
    with pytest.raises(TypeError):
        BaseAgent(name="Test")  # Directly instantiating BaseAgent should fail
