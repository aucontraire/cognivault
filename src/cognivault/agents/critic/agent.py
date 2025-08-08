from cognivault.agents.base_agent import (
    BaseAgent,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
)
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface
from cognivault.agents.critic.prompts import CRITIC_SYSTEM_PROMPT

# Configuration system imports
from typing import Optional, cast
from cognivault.config.agent_configs import CriticConfig
from cognivault.workflows.prompt_composer import PromptComposer

# Structured LLM integration
from cognivault.llm.structured import StructuredLLMFactory
from cognivault.agents.models import CriticOutput, ProcessingMode, ConfidenceLevel

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
    config : CriticConfig
        Configuration for agent behavior and prompt composition.
    logger : logging.Logger
        Logger instance used to emit internal debug and warning messages.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self, llm: LLMInterface, config: Optional[CriticConfig] = None
    ) -> None:
        """Initialize the CriticAgent with an LLM interface and optional configuration.

        Parameters
        ----------
        llm : LLMInterface
            The language model interface to use for generating critiques.
        config : Optional[CriticConfig]
            Configuration for agent behavior. If None, uses default configuration.
            Maintains backward compatibility - existing code continues to work.
        """
        super().__init__("critic")
        self.llm = llm

        # Configuration system - backward compatible
        self.config = config if config is not None else CriticConfig()
        self._prompt_composer = PromptComposer()
        self._composed_prompt = None

        # Initialize structured LLM wrapper for Pydantic AI integration
        self._structured_llm = StructuredLLMFactory.create_from_llm(llm)

        # Compose the prompt on initialization for performance
        self._update_composed_prompt()

    def _update_composed_prompt(self) -> None:
        """Update the composed prompt based on current configuration."""
        try:
            self._composed_prompt = self._prompt_composer.compose_critic_prompt(
                self.config
            )
            self.logger.debug(
                f"[{self.name}] Prompt composed with config: {self.config.analysis_depth}"
            )
        except Exception as e:
            self.logger.warning(
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
            self.logger.debug(f"[{self.name}] Using default system prompt (fallback)")
            return CRITIC_SYSTEM_PROMPT

    def update_config(self, config: CriticConfig) -> None:
        """
        Update the agent configuration and recompose prompts.

        Parameters
        ----------
        config : CriticConfig
            New configuration to apply
        """
        self.config = config
        self._update_composed_prompt()
        self.logger.info(
            f"[{self.name}] Configuration updated: {config.analysis_depth} analysis"
        )

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

        refined_output = context.agent_outputs.get("refiner", "")
        if not refined_output:
            critique = "No refined output available from RefinerAgent to critique."
            self.logger.warning(
                f"[{self.name}] No refined output available to critique."
            )
        else:
            self.logger.debug(
                f"[{self.name}] Analyzing refined query: {refined_output}"
            )

            # Use LLM to generate critique with configurable system prompt
            system_prompt = self._get_system_prompt()
            response = self.llm.generate(
                prompt=refined_output, system_prompt=system_prompt
            )
            if hasattr(response, "text"):
                critique = response.text.strip()

                # Record token usage if available from LLM response
                if (
                    hasattr(response, "tokens_used")
                    and response.tokens_used is not None
                ):
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
                    self.logger.debug(
                        f"[{self.name}] Token usage recorded - "
                        f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
                    )
                else:
                    self.logger.debug(
                        f"[{self.name}] No token usage information available from LLM response"
                    )
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

    async def run_structured(self, context: AgentContext) -> AgentContext:
        """
        Enhanced version of run() that uses structured Pydantic AI outputs.

        This method demonstrates the integration of Pydantic AI for structured
        responses while maintaining backward compatibility with the existing
        run() method.

        Parameters
        ----------
        context : AgentContext
            The shared context containing outputs from other agents.

        Returns
        -------
        AgentContext
            The updated context including structured critique output.
        """
        import time

        start_time = time.time()

        self.logger.info(
            f"[{self.name}] Processing query with structured output: {context.query}"
        )

        refined_output = context.agent_outputs.get("refiner", "")

        if not refined_output:
            # Create structured fallback response
            critique_output = CriticOutput(
                agent_name=self.name,
                processing_mode=ProcessingMode.PASSIVE,
                confidence=ConfidenceLevel.HIGH,
                processing_time_ms=0,
                critique_summary="No refined output available from RefinerAgent to critique.",
                issues_detected=0,
                no_issues_found=False,
            )
            self.logger.warning(
                f"[{self.name}] No refined output available to critique."
            )
        else:
            try:
                self.logger.debug(
                    f"[{self.name}] Analyzing refined query with structured LLM: {refined_output}"
                )

                # Enhanced system prompt for structured output
                structured_system_prompt = (
                    self._get_system_prompt()
                    + """

For your response, provide a structured analysis in JSON format with the following fields:
- assumptions: List of implicit assumptions identified
- logical_gaps: List of logical gaps or under-specified concepts  
- biases: List of bias types (temporal, cultural, methodological, scale)
- bias_details: Object mapping bias types to explanations
- alternate_framings: List of suggested alternate framings
- critique_summary: Overall critique summary
- issues_detected: Number of issues found
- confidence: Confidence level (high, medium, low)
- processing_mode: Processing mode used (active, passive)
"""
                )

                # Use structured LLM to get validated response
                structured_response = await self._structured_llm.generate_structured(
                    prompt=refined_output,
                    response_model=CriticOutput,
                    system_prompt=structured_system_prompt,
                    on_log=lambda msg: self.logger.debug(f"[{self.name}] {msg}"),
                )

                # Extract the validated Pydantic model
                # MyPy doesn't know the specific type, but we know it's CriticOutput based on response_model
                critique_output = cast(CriticOutput, structured_response.content)
                critique_output.processing_time_ms = (
                    structured_response.processing_time_ms
                )

                self.logger.info(
                    f"[{self.name}] Generated structured critique with {critique_output.issues_detected} issues "
                    f"(confidence: {critique_output.confidence})"
                )

            except Exception as e:
                # Fallback to unstructured response if structured fails
                self.logger.warning(
                    f"[{self.name}] Structured LLM failed, falling back to basic response: {e}"
                )

                # Use original LLM method as fallback
                system_prompt = self._get_system_prompt()
                response = self.llm.generate(
                    prompt=refined_output, system_prompt=system_prompt
                )

                if hasattr(response, "text"):
                    critique_text = response.text.strip()
                else:
                    critique_text = (
                        "Error: received streaming response instead of text response"
                    )

                # Create a basic structured response from unstructured output
                critique_output = CriticOutput(
                    agent_name=self.name,
                    processing_mode=ProcessingMode.ACTIVE,
                    confidence=ConfidenceLevel.MEDIUM,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    critique_summary=critique_text,
                    issues_detected=(
                        1 if critique_text != "No significant critique needed." else 0
                    ),
                    no_issues_found=critique_text.startswith("Query is well-scoped"),
                )

        # Store both structured and unstructured outputs for compatibility
        context.add_agent_output(self.name, critique_output.critique_summary)

        # Store structured output in execution state
        if "structured_outputs" not in context.execution_state:
            context.execution_state["structured_outputs"] = {}
        context.execution_state["structured_outputs"][self.name] = critique_output

        # Enhanced logging with structured data
        context.log_trace(
            self.name,
            input_data=refined_output,
            output_data={
                "critique_summary": critique_output.critique_summary,
                "issues_detected": critique_output.issues_detected,
                "confidence": critique_output.confidence,
                "processing_mode": critique_output.processing_mode,
                "processing_time_ms": critique_output.processing_time_ms,
            },
        )

        self.logger.debug(
            f"[{self.name}] Updated context with structured output: "
            f"{critique_output.issues_detected} issues, {critique_output.confidence} confidence"
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
