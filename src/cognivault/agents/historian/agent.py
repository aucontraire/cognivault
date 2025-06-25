import logging

logger = logging.getLogger(__name__)

from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


class HistorianAgent(BaseAgent):
    def __init__(self):
        super().__init__("Historian")

    def run(self, context: AgentContext) -> AgentContext:
        query = context.query.strip()
        logger.info(f"[{self.name}] Received query: {query}")

        # For now, simulate retrieval of past notes or history
        mock_history = [
            "Note from 2024-10-15: Mexico had a third party win the presidency.",
            "Note from 2024-11-05: Discussion on judiciary reforms in Mexico.",
        ]

        retrieved_text = "\n".join(mock_history)
        context.retrieved_notes = mock_history
        context.add_agent_output(self.name, retrieved_text)
        logger.info(f"[{self.name}] Retrieved notes: {retrieved_text}")

        return context
