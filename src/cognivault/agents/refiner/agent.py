from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from .prompts import REFINER_SYSTEM_PROMPT

import logging
import asyncio

logger = logging.getLogger(__name__)


class RefinerAgent(BaseAgent):
    """
    Agent responsible for transforming raw user queries into structured, clarified prompts.

    The RefinerAgent acts as the first stage in the CogniVault cognitive pipeline,
    detecting ambiguity and vagueness in user queries and rephrasing them for
    improved clarity and structure. It preserves the original intent while ensuring
    downstream agents can process the query effectively.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__("Refiner")
        self.llm: LLMInterface = llm

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Execute the refinement process on the provided agent context.

        Transforms the raw user query into a structured, clarified prompt using
        the RefinerAgent system prompt to guide the LLM behavior.

        Parameters
        ----------
        context : AgentContext
            The current shared context containing the user query and past agent outputs.

        Returns
        -------
        AgentContext
            The updated context with the refined query added under the agent's name.
        """
        await asyncio.sleep(0.1)  # Simulate asynchronous work
        query = context.query.strip()
        logger.info(f"[{self.name}] Processing query: {query}")

        # Generate refined query using system prompt
        response = self.llm.generate(prompt=query, system_prompt=REFINER_SYSTEM_PROMPT)

        if not hasattr(response, "text"):
            raise ValueError("LLMResponse missing 'text' field")

        refined_query = response.text.strip()

        # Format output to show the refinement
        if refined_query.startswith("[Unchanged]"):
            refined_output = refined_query
        else:
            refined_output = f"Refined query: {refined_query}"

        logger.debug(f"[{self.name}] Output: {refined_output}")

        context.add_agent_output(self.name, refined_output)
        context.log_trace(self.name, input_data=query, output_data=refined_output)
        return context
