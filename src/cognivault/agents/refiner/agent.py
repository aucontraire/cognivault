from cognivault.agents.base_agent import (
    BaseAgent,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
)
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from cognivault.config.app_config import get_config
from .prompts import REFINER_SYSTEM_PROMPT

# Configuration system imports
from typing import Optional
from cognivault.config.agent_configs import RefinerConfig
from cognivault.workflows.prompt_composer import PromptComposer

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RefinerAgent(BaseAgent):
    """
    Agent responsible for transforming raw user queries into structured, clarified prompts.

    The RefinerAgent acts as the first stage in the CogniVault cognitive pipeline,
    detecting ambiguity and vagueness in user queries and rephrasing them for
    improved clarity and structure. It preserves the original intent while ensuring
    downstream agents can process the query effectively.
    """

    def __init__(self, llm: LLMInterface, config: Optional[RefinerConfig] = None):
        """
        Initialize the RefinerAgent with LLM interface and optional configuration.

        Parameters
        ----------
        llm : LLMInterface
            The language model interface for generating responses
        config : Optional[RefinerConfig]
            Configuration for agent behavior. If None, uses default configuration.
            Maintains backward compatibility - existing code continues to work.
        """
        super().__init__("refiner")
        self.llm: LLMInterface = llm

        # Configuration system - backward compatible
        self.config = config if config is not None else RefinerConfig()
        self._prompt_composer = PromptComposer()
        self._composed_prompt = None

        # Compose the prompt on initialization for performance
        self._update_composed_prompt()

    def _update_composed_prompt(self):
        """Update the composed prompt based on current configuration."""
        try:
            self._composed_prompt = self._prompt_composer.compose_refiner_prompt(
                self.config
            )
            logger.debug(
                f"[{self.name}] Prompt composed with config: {self.config.refinement_level}"
            )
        except Exception as e:
            logger.warning(
                f"[{self.name}] Failed to compose prompt, using default: {e}"
            )
            self._composed_prompt = None

    def _get_system_prompt(self) -> str:
        """Get the system prompt, using composed prompt if available, otherwise default."""
        if self._composed_prompt and self._prompt_composer.validate_composition(
            self._composed_prompt
        ):
            return self._composed_prompt.system_prompt
        else:
            # Fallback to default prompt for backward compatibility
            logger.debug(f"[{self.name}] Using default system prompt (fallback)")
            return REFINER_SYSTEM_PROMPT

    def update_config(self, config: RefinerConfig):
        """
        Update the agent configuration and recompose prompts.

        Parameters
        ----------
        config : RefinerConfig
            New configuration to apply
        """
        self.config = config
        self._update_composed_prompt()
        logger.info(
            f"[{self.name}] Configuration updated: {config.refinement_level} refinement"
        )

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
        # Use configurable simulation delay if enabled
        config = get_config()
        if config.execution.enable_simulation_delay:
            await asyncio.sleep(config.execution.simulation_delay_seconds)
        query = context.query.strip()
        logger.info(f"[{self.name}] Processing query: {query}")

        # Generate refined query using configured system prompt
        system_prompt = self._get_system_prompt()
        response = self.llm.generate(prompt=query, system_prompt=system_prompt)

        if not hasattr(response, "text"):
            raise ValueError("LLMResponse missing 'text' field")

        refined_query = response.text.strip()

        # Format output to show the refinement
        if refined_query.startswith("[Unchanged]"):
            refined_output = refined_query
        else:
            refined_output = f"Refined query: {refined_query}"

        logger.debug(f"[{self.name}] Output: {refined_output}")

        # Add agent output
        context.add_agent_output(self.name, refined_output)

        # Record token usage if available from LLM response
        if hasattr(response, "tokens_used") and response.tokens_used is not None:
            # Use detailed token breakdown if available, otherwise fall back to total
            input_tokens = getattr(response, "input_tokens", None) or 0
            output_tokens = getattr(response, "output_tokens", None) or 0
            total_tokens = response.tokens_used

            # Ensure consistency: if we have detailed breakdown, use it for total
            if input_tokens and output_tokens:
                total_tokens = input_tokens + output_tokens

            context.add_agent_token_usage(
                agent_name=self.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            logger.debug(
                f"[{self.name}] Token usage recorded - "
                f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
            )
        else:
            logger.debug(
                f"[{self.name}] No token usage information available from LLM response"
            )

        context.log_trace(self.name, input_data=query, output_data=refined_output)
        return context

    def define_node_metadata(self) -> Dict[str, Any]:
        """
        Define LangGraph-specific metadata for the Refiner agent.

        Returns
        -------
        Dict[str, Any]
            Node metadata including type, dependencies, schemas, and routing logic
        """
        return {
            "node_type": NodeType.PROCESSOR,
            "dependencies": [],  # Entry point - no dependencies
            "description": "Transforms raw user queries into structured, clarified prompts",
            "inputs": [
                NodeInputSchema(
                    name="context",
                    description="Agent context containing raw user query to refine",
                    required=True,
                    type_hint="AgentContext",
                )
            ],
            "outputs": [
                NodeOutputSchema(
                    name="context",
                    description="Updated context with refined query added",
                    type_hint="AgentContext",
                )
            ],
            "tags": ["refiner", "agent", "processor", "entry_point"],
        }
