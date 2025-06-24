

from typing import Optional

from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext


class AgentOrchestrator:
    def __init__(self, critic_enabled: bool = True, only: Optional[str] = None):
        self.critic_enabled = critic_enabled
        self.only = only

        self.agents: list[BaseAgent] = []
        if only is None or only == "refiner":
            self.agents.append(RefinerAgent())
        if only is None or only == "historian":
            self.agents.append(HistorianAgent())
        if (only is None or only == "critic") and self.critic_enabled:
            self.agents.append(CriticAgent())
        if only is None or only == "synthesis":
            self.agents.append(SynthesisAgent())

    def run(self, query: str) -> AgentContext:
        context = AgentContext(query=query)

        for agent in self.agents:
            if agent is not None:
                context = agent.run(context)
                if isinstance(agent, SynthesisAgent):
                    context.final_synthesis = context.get_output(agent.name)

        return context
