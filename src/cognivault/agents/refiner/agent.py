from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


class RefinerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Refiner")

    def run(self, context: AgentContext) -> AgentContext:
        query = context.query.strip()
        refined_output = f"[Refined Note] Based on: '{query}'\n\nThis is a structured draft created by the Refiner agent."

        context.add_agent_output(self.name, refined_output)
        return context
