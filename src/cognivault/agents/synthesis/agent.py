import logging
from typing import Dict, Any, List

from cognivault.agents.base_agent import (
    BaseAgent,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
)
from cognivault.context import AgentContext

logger = logging.getLogger(__name__)


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

    def define_node_metadata(self) -> Dict[str, Any]:
        """
        Define LangGraph-specific metadata for the Synthesis agent.

        Returns
        -------
        Dict[str, Any]
            Node metadata including type, dependencies, schemas, and routing logic
        """
        return {
            "node_type": NodeType.AGGREGATOR,
            "dependencies": ["refiner", "critic", "historian"],  # Waits for all agents
            "description": "Synthesizes outputs from all agents into a comprehensive summary",
            "inputs": [
                NodeInputSchema(
                    name="context",
                    description="Agent context containing all agent outputs to synthesize",
                    required=True,
                    type_hint="AgentContext",
                )
            ],
            "outputs": [
                NodeOutputSchema(
                    name="context",
                    description="Final context with synthesized summary of all agent outputs",
                    type_hint="AgentContext",
                )
            ],
            "tags": ["synthesis", "agent", "aggregator", "terminator", "final"],
        }
