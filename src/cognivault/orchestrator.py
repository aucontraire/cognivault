import logging
from cognivault.config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from typing import Optional

from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.synthesis.agent import SynthesisAgent
from cognivault.context import AgentContext


class AgentOrchestrator:
    def __init__(
        self, critic_enabled: bool = True, agents_to_run: Optional[list[str]] = None
    ):
        self.critic_enabled = critic_enabled
        self.agents_to_run = (
            [a.lower() for a in agents_to_run] if agents_to_run else None
        )
        logger.debug(
            f"Initializing AgentOrchestrator with agents_to_run={self.agents_to_run} and critic_enabled={self.critic_enabled}"
        )
        self.agents: list[BaseAgent] = []

        if self.agents_to_run:
            logger.debug("Custom agent list specified.")
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
                logger.debug(f"Added agent: {agent_name}")
        else:
            self.agents = [RefinerAgent(), HistorianAgent()]
            if self.critic_enabled:
                self.agents.append(CriticAgent())
            self.agents.append(SynthesisAgent())
            logger.debug(
                f"Default agent order: {[agent.__class__.__name__ for agent in self.agents]}"
            )

    def run(self, query: str) -> AgentContext:
        logger.info(f"Running orchestrator with query: {query}")
        context = AgentContext(query=query)

        for agent in self.agents:
            if agent is not None:
                logger.info(f"Running agent: {agent.name}")
                context = agent.run(context)
                logger.info(f"Completed agent: {agent.name}")
                if isinstance(agent, SynthesisAgent):
                    logger.debug(f"Setting final_synthesis from {agent.name}")
                    context.final_synthesis = context.get_output(agent.name)

        return context
