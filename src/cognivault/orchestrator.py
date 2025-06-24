from typing import Optional

from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext


class AgentOrchestrator:
    def __init__(self, critic_enabled: bool = True, agents_to_run: Optional[list[str]] = None):
        self.critic_enabled = critic_enabled
        self.agents_to_run = [a.lower() for a in agents_to_run] if agents_to_run else None
        self.agents = []

        if self.agents_to_run:
            for agent_name in self.agents_to_run:
                if agent_name == "refiner":
                    self.agents.append(RefinerAgent())
                elif agent_name == "historian":
                    self.agents.append(HistorianAgent())
                elif agent_name == "synthesis":
                    self.agents.append(SynthesisAgent())
                elif agent_name == "critic":
                    self.agents.append(CriticAgent())
                else:
                    print(f"[DEBUG] Unknown agent name: {agent_name}")
        else:
            self.agents = [RefinerAgent(), HistorianAgent()]
            if self.critic_enabled:
                self.agents.append(CriticAgent())
            self.agents.append(SynthesisAgent())

    def run(self, query: str) -> AgentContext:
        context = AgentContext(query=query)

        for agent in self.agents:
            if agent is not None:
                context = agent.run(context)
                if isinstance(agent, SynthesisAgent):
                    context.final_synthesis = context.get_output(agent.name)

        return context
