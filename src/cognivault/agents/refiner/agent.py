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

# Structured output imports
from cognivault.agents.models import RefinerOutput, ProcessingMode, ConfidenceLevel
from cognivault.services.langchain_service import LangChainService

# Configuration system imports
from typing import Optional, Union
from cognivault.config.agent_configs import RefinerConfig
from cognivault.workflows.prompt_composer import PromptComposer, ComposedPrompt

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

    def __init__(
        self, llm: LLMInterface, config: Optional[RefinerConfig] = None
    ) -> None:
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

        # Initialize structured output service
        self.structured_service: Optional[LangChainService] = None
        self._setup_structured_service()

        # Configuration system - backward compatible
        # All config classes have sensible defaults via Pydantic Field definitions
        self.config = config if config is not None else RefinerConfig()
        self._prompt_composer = PromptComposer()
        self._composed_prompt: Optional[ComposedPrompt]

        # Compose the prompt on initialization for performance
        self._update_composed_prompt()

    def _setup_structured_service(self) -> None:
        """Setup LangChain structured output service with model discovery."""
        try:
            # Get API key from LLM interface
            api_key = getattr(self.llm, "api_key", None)

            # Create LangChain service with model discovery for RefinerAgent
            # This will automatically select the best model (preferring gpt-5-nano/mini)
            self.structured_service = LangChainService(
                model=None,  # Let discovery service choose
                api_key=api_key,
                temperature=0.1,  # Low temperature for consistent refinement
                agent_name="refiner",  # Critical: enables agent-specific model selection
                use_discovery=True,  # Enable model discovery
            )

            # Log the selected model
            selected_model = self.structured_service.model_name
            logger.info(
                f"[{self.name}] Structured output service initialized with discovered model: {selected_model}"
            )

            # Special handling for ultra-fast models
            if "nano" in selected_model.lower() or "mini" in selected_model.lower():
                logger.info(
                    f"[{self.name}] Using optimized fast model '{selected_model}' for improved performance"
                )
        except Exception as e:
            logger.warning(
                f"[{self.name}] Could not initialize structured output service: {e}"
            )
            self.structured_service = None

    def _update_composed_prompt(self) -> None:
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

    def update_config(self, config: RefinerConfig) -> None:
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
        the RefinerAgent system prompt to guide the LLM behavior. Uses structured
        output when available for improved consistency and content pollution prevention.

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

        system_prompt = self._get_system_prompt()

        # Try structured output first, fallback to traditional method
        if self.structured_service:
            try:
                refined_output = await self._run_structured(
                    query, system_prompt, context
                )
            except Exception as e:
                logger.warning(
                    f"[{self.name}] Structured output failed, falling back to traditional: {e}"
                )
                refined_output = await self._run_traditional(
                    query, system_prompt, context
                )
        else:
            refined_output = await self._run_traditional(query, system_prompt, context)

        logger.debug(f"[{self.name}] Output: {refined_output}")

        # Add agent output
        context.add_agent_output(self.name, refined_output)

        context.log_trace(self.name, input_data=query, output_data=refined_output)
        return context

    async def _run_structured(
        self, query: str, system_prompt: str, context: AgentContext
    ) -> str:
        """Run with structured output using LangChain service."""
        import time

        start_time = time.time()

        if not self.structured_service:
            raise ValueError("Structured service not available")

        try:
            # Build the refinement prompt
            prompt = f"Original query: {query}\n\nPlease refine this query according to the system instructions."

            # Get structured output
            result = await self.structured_service.get_structured_output(
                prompt=prompt,
                output_class=RefinerOutput,
                system_prompt=system_prompt,
                max_retries=3,
            )

            # Handle both RefinerOutput and StructuredOutputResult types
            if isinstance(result, RefinerOutput):
                structured_result = result
            else:
                # It's a StructuredOutputResult, extract the parsed result
                from cognivault.services.langchain_service import StructuredOutputResult

                if isinstance(result, StructuredOutputResult):
                    parsed_result = result.parsed
                    if not isinstance(parsed_result, RefinerOutput):
                        raise ValueError(
                            f"Expected RefinerOutput, got {type(parsed_result)}"
                        )
                    structured_result = parsed_result
                else:
                    raise ValueError(f"Unexpected result type: {type(result)}")

            processing_time_ms = (time.time() - start_time) * 1000

            # Store structured metadata if available
            if hasattr(context, "execution_metadata"):
                context.execution_metadata["agent_outputs"] = (
                    context.execution_metadata.get("agent_outputs", {})
                )
                context.execution_metadata["agent_outputs"][self.name] = (
                    structured_result.dict()
                )

            # Record token usage - for structured output, we need to record some usage
            # Since structured output doesn't directly expose token usage from LangChain,
            # we'll check if we can get it from the underlying LLM response or estimate
            token_usage_recorded = False

            # Try to get token usage from LangChain service metrics
            if hasattr(self.structured_service, "get_metrics"):
                metrics = self.structured_service.get_metrics()
                logger.debug(f"[{self.name}] Service metrics: {metrics}")

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
                logger.debug(
                    f"[{self.name}] Token usage from structured result: {usage}"
                )

            # If we're in a testing scenario and structured output failed,
            # the fallback to traditional would have been used
            # In that case, token usage should already be recorded by _run_traditional

            if not token_usage_recorded:
                # For structured output without explicit token usage, record minimal usage
                # This ensures event emission doesn't fail
                context.add_agent_token_usage(
                    agent_name=self.name,
                    input_tokens=0,  # LangChain structured output doesn't expose detailed tokens
                    output_tokens=0,
                    total_tokens=0,
                )
                logger.debug(
                    f"[{self.name}] Recorded zero token usage for structured output (token details not available)"
                )

            logger.info(
                f"[{self.name}] Structured output successful - "
                f"processing_time: {processing_time_ms:.1f}ms, "
                f"confidence: {structured_result.confidence}, "
                f"changes_made: {len(structured_result.changes_made)}"
            )

            # Format output based on refinement results
            if structured_result.was_unchanged:
                return "[Unchanged] " + structured_result.refined_query
            else:
                return f"Refined query: {structured_result.refined_query}"

        except Exception as e:
            logger.error(f"[{self.name}] Structured output processing failed: {e}")
            raise

    async def _run_traditional(
        self, query: str, system_prompt: str, context: AgentContext
    ) -> str:
        """Fallback to traditional LLM interface."""
        logger.info(f"[{self.name}] Using traditional LLM interface")

        response = self.llm.generate(prompt=query, system_prompt=system_prompt)

        if not hasattr(response, "text"):
            raise ValueError("LLMResponse missing 'text' field")

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

        logger.debug(
            f"[{self.name}] Traditional LLM token usage - "
            f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
        )

        refined_query = response.text.strip()

        # Format output to show the refinement
        if refined_query.startswith("[Unchanged]"):
            return refined_query
        else:
            return f"Refined query: {refined_query}"

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
