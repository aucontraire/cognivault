from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
import logging
import asyncio


class CriticAgent(BaseAgent):
    """Agent responsible for critiquing the output of the RefinerAgent.

    The CriticAgent evaluates the output provided by the RefinerAgent and
    adds constructive critique or feedback to the context.

    Attributes
    ----------
    name : str
        The name of the agent, used for identification in the context.
    logger : logging.Logger
        Logger instance used to emit internal debug and warning messages.
    """

    logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__("Critic")

    async def run(self, context: AgentContext) -> AgentContext:
        """Run the CriticAgent's logic to provide feedback on refined notes.

        Parameters
        ----------
        context : AgentContext
            The shared context containing outputs from other agents.

        Returns
        -------
        AgentContext
            The updated context including this agent's critique output.
        """
        await asyncio.sleep(0.1)
        self.logger.debug(
            f"[{self.name}] Running {self.name} with context.agent_outputs: {context.agent_outputs}"
        )
        refined_output = context.agent_outputs.get("Refiner", "")
        if not refined_output:
            critique = "No refined output found to critique."
            self.logger.warning(f"[{self.name}] No refined output found to critique.")
        else:
            critique = "[Critique] The refined note may lack depth or miss opposing perspectives."
            self.logger.debug(f"[{self.name}] Generated critique: {critique}")

        context.add_agent_output(self.name, critique)
        self.logger.debug(
            f"[{self.name}] Updated context with {self.name} output: {critique}"
        )
        return context
