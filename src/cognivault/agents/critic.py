from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__("Critic")

    def run(self, context: AgentContext) -> AgentContext:
        refined_output = context.agent_outputs.get("Refiner", "")
        if not refined_output:
            critique = "No refined output found to critique."
        else:
            # Placeholder critique logic
            critique = f"[Critique] The refined note may lack depth or miss opposing perspectives."

        context.add_agent_output(self.name, critique)
        return context
