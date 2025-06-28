from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface

import logging
import asyncio

logger = logging.getLogger(__name__)


class RefinerAgent(BaseAgent):
    """
    Agent responsible for refining a user query into a structured note.

    This agent generates a draft based on the user's input, simulating the
    behavior of a note-taking assistant that structures ideas into a clear format.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__("Refiner")
        self.llm: LLMInterface = llm

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

        response = self.llm.generate(query)

        if not hasattr(response, "text"):
            raise ValueError("LLMResponse missing 'text' field")

        text = response.text

        refined_output = f"[Refined Note] Based on: '{query}'\n\n{text}"
        logger.debug(f"[{self.name}] Output: {refined_output}")

        context.add_agent_output(self.name, refined_output)
        context.log_trace(self.name, input_data=query, output_data=refined_output)
        return context
