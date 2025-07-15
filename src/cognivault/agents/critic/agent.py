from cognivault.agents.base_agent import (
    BaseAgent,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
)
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from cognivault.agents.critic.prompts import CRITIC_SYSTEM_PROMPT
import logging
from typing import Dict, Any


class CriticAgent(BaseAgent):
    """Agent responsible for critiquing the output of the RefinerAgent.

    The CriticAgent evaluates the output provided by the RefinerAgent and
    adds constructive critique or feedback to the context using an LLM.

    Attributes
    ----------
    name : str
        The name of the agent, used for identification in the context.
    llm : LLMInterface
        The language model interface used to generate critiques.
    logger : logging.Logger
        Logger instance used to emit internal debug and warning messages.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, llm: LLMInterface):
        """Initialize the CriticAgent with an LLM interface.

        Parameters
        ----------
        llm : LLMInterface
            The language model interface to use for generating critiques.
        """
        super().__init__("Critic")
        self.llm = llm

    async def run(self, context: AgentContext) -> AgentContext:
        """Run the CriticAgent's logic to provide feedback on refined queries.

        Parameters
        ----------
        context : AgentContext
            The shared context containing outputs from other agents.

        Returns
        -------
        AgentContext
            The updated context including this agent's critique output.
        """
        self.logger.info(f"[{self.name}] Processing query: {context.query}")

        refined_output = context.agent_outputs.get("Refiner", "")
        if not refined_output:
            critique = "No refined output available from RefinerAgent to critique."
            self.logger.warning(
                f"[{self.name}] No refined output available to critique."
            )
        else:
            self.logger.debug(
                f"[{self.name}] Analyzing refined query: {refined_output}"
            )

            # Use LLM to generate critique with system prompt
            response = self.llm.generate(
                prompt=refined_output, system_prompt=CRITIC_SYSTEM_PROMPT
            )
            if hasattr(response, "text"):
                critique = response.text.strip()
            else:
                # Handle streaming response (shouldn't happen with current usage)
                critique = "Error: received streaming response instead of text response"
            self.logger.debug(f"[{self.name}] Generated critique: {critique}")

        context.add_agent_output(self.name, critique)
        context.log_trace(self.name, input_data=refined_output, output_data=critique)
        self.logger.debug(
            f"[{self.name}] Updated context with {self.name} output: {critique}"
        )
        return context

    def define_node_metadata(self) -> Dict[str, Any]:
        """
        Define LangGraph-specific metadata for the Critic agent.

        Returns
        -------
        Dict[str, Any]
            Node metadata including type, dependencies, schemas, and routing logic
        """
        return {
            "node_type": NodeType.PROCESSOR,
            "dependencies": ["refiner"],  # Depends on Refiner output
            "description": "Evaluates and critiques refined queries for quality and clarity",
            "inputs": [
                NodeInputSchema(
                    name="context",
                    description="Agent context containing refined query to critique",
                    required=True,
                    type_hint="AgentContext",
                )
            ],
            "outputs": [
                NodeOutputSchema(
                    name="context",
                    description="Updated context with critique feedback added",
                    type_hint="AgentContext",
                )
            ],
            "tags": ["critic", "agent", "processor", "evaluator"],
        }
