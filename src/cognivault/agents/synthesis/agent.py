from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Synthesis")

    def run(self, context: AgentContext) -> AgentContext:
        outputs = context.agent_outputs
        combined = []

        for agent, output in outputs.items():
            combined.append(f"### From {agent}:\n{output.strip()}\n")

        synthesized_note = "\n".join(combined)
        context.add_agent_output(self.name, synthesized_note)
        return context
