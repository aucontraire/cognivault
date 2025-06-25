from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext

import logging

logger = logging.getLogger(__name__)


class RefinerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Refiner")

    def run(self, context: AgentContext) -> AgentContext:
        query = context.query.strip()
        logger.info(f"Running {self.name}Agent with query: {query}")
        refined_output = f"[Refined Note] Based on: '{query}'\n\nThis is a structured draft created by the Refiner agent."
        logger.debug(f"{self.name}Agent output: {refined_output}")

        context.add_agent_output(self.name, refined_output)
        return context
