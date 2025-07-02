import logging
import asyncio

logger = logging.getLogger(__name__)

from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
from cognivault.config.app_config import get_config


class HistorianAgent(BaseAgent):
    """
    Agent that retrieves historical context or notes relevant to a given query.

    Parameters
    ----------
    name : str
        The name of the agent. Defaults to "Historian".
    """

    logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__("Historian")

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Executes the Historian agent to fetch historical data related to the query.

        Parameters
        ----------
        context : AgentContext
            The current context object containing the user query and accumulated outputs.

        Returns
        -------
        AgentContext
            The updated context object with the Historian's output and retrieved notes.
        """
        query = context.query.strip()
        self.logger.info(f"[{self.name}] Received query: {query}")

        # Use configurable simulation delay if enabled
        config = get_config()
        if config.execution.enable_simulation_delay:
            await asyncio.sleep(config.execution.simulation_delay_seconds)

        # Use configurable mock history data
        mock_history = config.testing.mock_history_entries

        retrieved_text = "\n".join(mock_history)
        context.retrieved_notes = mock_history
        context.add_agent_output(self.name, retrieved_text)
        self.logger.info(f"[{self.name}] Retrieved notes: {retrieved_text}")
        context.log_trace(self.name, input_data=query, output_data=retrieved_text)

        return context
