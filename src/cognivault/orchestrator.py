from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext


class AgentOrchestrator:
    def __init__(self, critic_enabled: bool = True):
        self.critic_enabled = critic_enabled
        self.agents = [
            RefinerAgent(),
            HistorianAgent(),
            CriticAgent() if self.critic_enabled else None,
            SynthesisAgent(),
        ]

    def run(self, query: str) -> AgentContext:
        context = AgentContext(query=query)

        for agent in self.agents:
            if agent is not None:
                context = agent.run(context)
                if isinstance(agent, SynthesisAgent):
                    context.final_synthesis = context.get_output(agent.name)

        return context
