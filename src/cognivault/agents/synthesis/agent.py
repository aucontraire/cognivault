import logging

logger = logging.getLogger(__name__)

from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Synthesis")

    def run(self, context: AgentContext) -> AgentContext:
        outputs = context.agent_outputs
        logger.info(
            "Running SynthesisAgent with agent_outputs: %s", list(outputs.keys())
        )
        combined = []

        for agent, output in outputs.items():
            combined.append(f"### From {agent}:\n{output.strip()}\n")

        synthesized_note = "\n".join(combined)
        logger.debug("Synthesized note:\n%s", synthesized_note)
        context.add_agent_output(self.name, synthesized_note)
        return context
