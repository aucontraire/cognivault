from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
import logging


class CriticAgent(BaseAgent):
    logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__("Critic")

    def run(self, context: AgentContext) -> AgentContext:
        self.logger.debug(
            f"Running CriticAgent with context.agent_outputs: {context.agent_outputs}"
        )
        refined_output = context.agent_outputs.get("Refiner", "")
        if not refined_output:
            critique = "No refined output found to critique."
            self.logger.warning("No refined output found to critique.")
        else:
            critique = f"[Critique] The refined note may lack depth or miss opposing perspectives."
            self.logger.debug(f"Generated critique: {critique}")

        context.add_agent_output(self.name, critique)
        self.logger.debug(f"Updated context with CriticAgent output: {critique}")
        return context
