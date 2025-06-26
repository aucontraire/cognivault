from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext

import logging
import asyncio

logger = logging.getLogger(__name__)


class RefinerAgent(BaseAgent):
    """
    Agent responsible for refining a user query into a structured note.

    This agent generates a draft based on the user's input, simulating the
    behavior of a note-taking assistant that structures ideas into a clear format.
    """

    def __init__(self):
        super().__init__("Refiner")

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Execute the refinement process on the provided agent context.

        Parameters
        ----------
        context : AgentContext
            The current shared context containing the user query and past agent outputs.

        Returns
        -------
        AgentContext
            The updated context with the refined note added under the agent's name.
        """
        await asyncio.sleep(0.1)  # Simulate asynchronous work
        query = context.query.strip()
        logger.info(f"[{self.name}] Running agent with query: {query}")
        refined_output = f"[Refined Note] Based on: '{query}'\n\nThis is a structured draft created by the Refiner agent."
        logger.debug(f"[{self.name}] Output: {refined_output}")

        context.add_agent_output(self.name, refined_output)
        context.log_trace(self.name, input_data=query, output_data=refined_output)
        return context
