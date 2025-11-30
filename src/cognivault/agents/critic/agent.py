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
from typing import Optional, cast, Type
from cognivault.config.agent_configs import CriticConfig
from cognivault.workflows.prompt_composer import PromptComposer, ComposedPrompt

# Structured output integration using LangChain service pattern
from cognivault.services.langchain_service import LangChainService
from cognivault.agents.models import CriticOutput, ProcessingMode, ConfidenceLevel

import asyncio
import logging
from typing import Dict, Any
from cognivault.config.app_config import get_config


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
        # Configuration system - backward compatible
        # All config classes have sensible defaults via Pydantic Field definitions
        self.config = config if config is not None else CriticConfig()

        # Pass timeout from config to BaseAgent
        super().__init__(
            "critic", timeout_seconds=self.config.execution_config.timeout_seconds
        )

        self.llm = llm
        self._prompt_composer = PromptComposer()
        self._composed_prompt: Optional[ComposedPrompt] = None

        # Initialize LangChain service for structured output (following RefinerAgent pattern)
        self.structured_service: Optional[LangChainService] = None
        self._setup_structured_service()

        # Compose the prompt on initialization for performance
        self._update_composed_prompt()

    def _setup_structured_service(self) -> None:
        """Initialize the LangChain service for structured output support."""
        try:
            # Get API key from LLM interface
            api_key = getattr(self.llm, "api_key", None)

            # Let discovery service choose the best model for CriticAgent
            self.structured_service = LangChainService(
                model=None,  # Let discovery service choose
                api_key=api_key,
                temperature=0.0,  # Use deterministic output for critiques
                agent_name="critic",  # Enable agent-specific model selection
                use_discovery=True,  # Enable model discovery
            )

            # Log the selected model
            selected_model = self.structured_service.model_name
            self.logger.info(
                f"[{self.name}] Structured output service initialized with model: {selected_model}"
            )
        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Failed to initialize structured service: {e}. "
                f"Will use traditional LLM interface only."
            )
            self.structured_service = None

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
        """
        Execute the critique process on the provided agent context.

        Analyzes the refined query from RefinerAgent and provides constructive
        critique using structured output when available for improved consistency
        and content pollution prevention.

        Parameters
        ----------
        context : AgentContext
            The shared context containing outputs from other agents.

        Returns
        -------
        AgentContext
            The updated context including this agent's critique output.
        """
        # Use configurable simulation delay if enabled
        config = get_config()
        if config.execution.enable_simulation_delay:
            await asyncio.sleep(config.execution.simulation_delay_seconds)

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

            system_prompt = self._get_system_prompt()

            # Try structured output first, fallback to traditional method
            if self.structured_service:
                try:
                    critique = await self._run_structured(
                        refined_output, system_prompt, context
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[{self.name}] Structured output failed, falling back to traditional: {e}"
                    )
                    critique = await self._run_traditional(
                        refined_output, system_prompt, context
                    )
            else:
                critique = await self._run_traditional(
                    refined_output, system_prompt, context
                )

        self.logger.debug(f"[{self.name}] Generated critique: {critique}")

        # Add agent output
        context.add_agent_output(self.name, critique)
        context.log_trace(self.name, input_data=refined_output, output_data=critique)

        return context

    async def _run_structured(
        self, refined_output: str, system_prompt: str, context: AgentContext
    ) -> str:
        """Run with structured output using LangChain service."""
        import time

        start_time = time.time()

        if not self.structured_service:
            raise ValueError("Structured service not available")

        try:
            # Build the critique prompt
            prompt = f"Refined query to critique: {refined_output}\n\nPlease provide a comprehensive critique according to the system instructions."

            # Get structured output
            result = await self.structured_service.get_structured_output(
                prompt=prompt,
                output_class=CriticOutput,
                system_prompt=system_prompt,
                max_retries=3,
            )

            # Handle both CriticOutput and StructuredOutputResult types
            if isinstance(result, CriticOutput):
                structured_result = result
            else:
                # It's a StructuredOutputResult, extract the parsed result
                from cognivault.services.langchain_service import StructuredOutputResult

                if isinstance(result, StructuredOutputResult):
                    parsed_result = result.parsed
                    if not isinstance(parsed_result, CriticOutput):
                        raise ValueError(
                            f"Expected CriticOutput, got {type(parsed_result)}"
                        )
                    structured_result = parsed_result
                else:
                    raise ValueError(f"Unexpected result type: {type(result)}")

            # SERVER-SIDE PROCESSING TIME INJECTION
            # CRITICAL FIX: LLMs cannot accurately measure their own processing time
            # We calculate actual execution time server-side and inject it into the model
            processing_time_ms = (time.time() - start_time) * 1000

            # Inject server-calculated processing time if LLM returned None
            if structured_result.processing_time_ms is None:
                # Use model_copy to create new instance with updated processing_time_ms
                structured_result = structured_result.model_copy(
                    update={"processing_time_ms": processing_time_ms}
                )
                self.logger.info(
                    f"[{self.name}] Injected server-calculated processing_time_ms: {processing_time_ms:.1f}ms"
                )

            # Store structured output in execution_state for future use
            # This follows a pattern where structured outputs can be accessed later
            if "structured_outputs" not in context.execution_state:
                context.execution_state["structured_outputs"] = {}
            context.execution_state["structured_outputs"][self.name] = (
                structured_result.model_dump()
            )

            # Record token usage - for structured output, we need to record some usage
            # Since structured output doesn't directly expose token usage from LangChain,
            # we'll check if we can get it from the underlying LLM response or estimate
            token_usage_recorded = False

            # Try to get token usage from LangChain service metrics
            if hasattr(self.structured_service, "get_metrics"):
                metrics = self.structured_service.get_metrics()
                self.logger.debug(f"[{self.name}] Service metrics: {metrics}")

            # For testing scenarios where mock LLMs are used, we need to ensure token usage is recorded
            # Check if this is a fallback scenario where traditional LLM was used
            if hasattr(structured_result, "_token_usage"):
                # If the structured result carries token usage info, use it
                usage = structured_result._token_usage
                context.add_agent_token_usage(
                    agent_name=self.name,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
                token_usage_recorded = True
                self.logger.debug(
                    f"[{self.name}] Token usage from structured result: {usage}"
                )

            if not token_usage_recorded:
                # For structured output without explicit token usage, record minimal usage
                # This ensures event emission doesn't fail
                context.add_agent_token_usage(
                    agent_name=self.name,
                    input_tokens=0,  # LangChain structured output doesn't expose detailed tokens
                    output_tokens=0,
                    total_tokens=0,
                )
                self.logger.debug(
                    f"[{self.name}] Recorded zero token usage for structured output (token details not available)"
                )

            self.logger.info(
                f"[{self.name}] Structured output successful - "
                f"processing_time: {processing_time_ms:.1f}ms, "
                f"confidence: {structured_result.confidence}, "
                f"issues_detected: {structured_result.issues_detected}"
            )

            # Return the critique summary directly for backward compatibility
            return structured_result.critique_summary

        except Exception as e:
            self.logger.error(f"[{self.name}] Structured output processing failed: {e}")
            raise

    async def _run_traditional(
        self, refined_output: str, system_prompt: str, context: AgentContext
    ) -> str:
        """Fallback to traditional LLM interface."""
        self.logger.info(f"[{self.name}] Using traditional LLM interface")

        response = self.llm.generate(prompt=refined_output, system_prompt=system_prompt)

        if not hasattr(response, "text"):
            # For backward compatibility with tests, return error message instead of raising
            self.logger.error(f"[{self.name}] LLM response missing 'text' field")
            # Still record minimal token usage to avoid downstream issues
            context.add_agent_token_usage(
                agent_name=self.name,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
            return "Error: received streaming response instead of text response"

        # Record token usage from traditional LLM response
        input_tokens = getattr(response, "input_tokens", None) or 0
        output_tokens = getattr(response, "output_tokens", None) or 0
        total_tokens = getattr(response, "tokens_used", None) or 0

        context.add_agent_token_usage(
            agent_name=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        self.logger.debug(
            f"[{self.name}] Traditional LLM token usage - "
            f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
        )

        critique = response.text.strip()

        # Return the critique as-is for backward compatibility
        return critique

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
