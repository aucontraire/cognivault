import logging

logger = logging.getLogger(__name__)

from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


class SynthesisAgent(BaseAgent):
    """
    Agent responsible for synthesizing outputs from multiple agents into a single summary.

    Attributes
    ----------
    name : str
        The name of the agent, set to 'Synthesis'.
    """

    def __init__(self):
        super().__init__("Synthesis")

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Asynchronously synthesizes the outputs from all agents and adds the synthesized
        result to the agent context.

        Parameters
        ----------
        context : AgentContext
            The context containing outputs from other agents.

        Returns
        -------
        AgentContext
            The updated context with the synthesized note added.
        """
        outputs = context.agent_outputs
        logger.info(
            f"[{self.name}] Running agent with agent_outputs: {list(outputs.keys())}"
        )
        combined = []

        for agent, output in outputs.items():
            combined.append(f"### From {agent}:\n{output.strip()}\n")

        synthesized_note = "\n".join(combined)
        logger.info(f"[{self.name}] Synthesized note:\n{synthesized_note}")
        context.add_agent_output(self.name, synthesized_note)
        context.log_trace(self.name, input_data=outputs, output_data=synthesized_note)
        return context
