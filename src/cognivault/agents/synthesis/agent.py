import logging
from typing import Dict, Any, Optional, Union

from cognivault.agents.base_agent import (
    BaseAgent,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
)
from cognivault.context import AgentContext
from cognivault.llm.llm_interface import LLMInterface

# Configuration system imports
from cognivault.config.agent_configs import SynthesisConfig
from cognivault.workflows.prompt_composer import PromptComposer, ComposedPrompt

# Structured output integration using LangChain service pattern
from cognivault.services.langchain_service import LangChainService
from cognivault.agents.models import SynthesisOutput, SynthesisTheme, ConfidenceLevel

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Enhanced agent for sophisticated synthesis of multi-agent outputs with LLM-powered
    thematic analysis, conflict resolution, and comprehensive knowledge integration.

    Performs deep analysis of agent outputs to identify themes, resolve conflicts,
    extract meta-insights, and produce coherent, wiki-ready synthesis.

    Parameters
    ----------
    llm : LLMInterface, optional
        LLM interface for synthesis analysis. If None, uses default OpenAI setup.
    config : Optional[SynthesisConfig], optional
        Configuration for agent behavior. If None, uses default configuration.
        Maintains backward compatibility - existing code continues to work.

    Attributes
    ----------
    config : SynthesisConfig
        Configuration for agent behavior and prompt composition.
    """

    def __init__(
        self,
        llm: Optional[Union[LLMInterface, str]] = "default",
        config: Optional[SynthesisConfig] = None,
    ) -> None:
        # Configuration system - backward compatible
        # All config classes have sensible defaults via Pydantic Field definitions
        self.config = config if config is not None else SynthesisConfig()

        # Pass timeout from config to BaseAgent
        super().__init__(
            "synthesis",
            timeout_seconds=self.config.execution_config.timeout_seconds
        )

        self._prompt_composer = PromptComposer()
        self._composed_prompt: Optional[ComposedPrompt]

        # Use sentinel value to distinguish between None (explicit) and default
        if llm == "default":
            self.llm: Optional[LLMInterface] = self._create_default_llm()
        else:
            # Type guard: ensure llm is either None or LLMInterface
            if llm is None:
                self.llm = None
            elif hasattr(llm, "generate"):
                self.llm = llm  # type: ignore[assignment]
            else:
                self.llm = None

        # Initialize LangChain service for structured output (following RefinerAgent pattern)
        self.structured_service: Optional[LangChainService] = None
        self._setup_structured_service()

        # Compose the prompt on initialization for performance
        self._update_composed_prompt()

    def _create_default_llm(self) -> Optional[LLMInterface]:
        """Create default LLM interface using OpenAI configuration."""
        try:
            # Import here to avoid circular imports
            from cognivault.llm.openai import OpenAIChatLLM
            from cognivault.config.openai_config import OpenAIConfig

            openai_config = OpenAIConfig.load()
            return OpenAIChatLLM(
                api_key=openai_config.api_key,
                model=openai_config.model,
                base_url=openai_config.base_url,
            )
        except Exception as e:
            logger.warning(
                f"Failed to create OpenAI LLM: {e}. Using fallback synthesis."
            )
            return None

    def _setup_structured_service(self) -> None:
        """Initialize the LangChain service for structured output support."""
        try:
            # Only set up if we have an LLM
            if self.llm is None:
                logger.info(
                    f"[{self.name}] No LLM available, structured service disabled"
                )
                self.structured_service = None
                return

            # Get API key from LLM interface
            api_key = getattr(self.llm, "api_key", None)

            # Let discovery service choose the best model for SynthesisAgent
            self.structured_service = LangChainService(
                model=None,  # Let discovery service choose
                api_key=api_key,
                temperature=0.2,  # Use low temperature for consistent synthesis
                agent_name="synthesis",  # Enable agent-specific model selection
                use_discovery=True,  # Enable model discovery
            )

            # Log the selected model
            selected_model = self.structured_service.model_name
            logger.info(
                f"[{self.name}] Structured output service initialized with discovered model: {selected_model}"
            )
        except Exception as e:
            logger.warning(
                f"[{self.name}] Failed to initialize structured service: {e}. "
                f"Will use traditional LLM interface only."
            )
            self.structured_service = None

    def _update_composed_prompt(self) -> None:
        """Update the composed prompt based on current configuration."""
        try:
            self._composed_prompt = self._prompt_composer.compose_synthesis_prompt(
                self.config
            )
            logger.debug(
                f"[{self.name}] Prompt composed with config: {self.config.synthesis_strategy}"
            )
        except Exception as e:
            logger.warning(
                f"[{self.name}] Failed to compose prompt, using default: {e}"
            )
            self._composed_prompt = None

    def _get_system_prompt(self) -> str:
        """Get the system prompt, using composed prompt if available, otherwise fallback."""
        if self._composed_prompt and self._prompt_composer.validate_composition(
            self._composed_prompt
        ):
            return self._composed_prompt.system_prompt
        else:
            # Fallback to embedded prompt for backward compatibility
            logger.debug(f"[{self.name}] Using default system prompt (fallback)")
            return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for fallback compatibility."""
        try:
            from cognivault.agents.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT

            return SYNTHESIS_SYSTEM_PROMPT
        except ImportError:
            # Fallback to basic embedded prompt
            return """As a synthesis agent, analyze and integrate multiple agent outputs to create comprehensive, coherent responses that combine all perspectives."""

    def update_config(self, config: SynthesisConfig) -> None:
        """
        Update the agent configuration and recompose prompts.

        Parameters
        ----------
        config : SynthesisConfig
            New configuration to apply
        """
        self.config = config
        self._update_composed_prompt()
        logger.info(
            f"[{self.name}] Configuration updated: {config.synthesis_strategy} strategy"
        )

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Execute enhanced synthesis with sophisticated LLM-powered analysis.

        Performs thematic analysis, conflict resolution, and comprehensive synthesis
        of all agent outputs into a coherent, wiki-ready final result.

        Parameters
        ----------
        context : AgentContext
            The context containing outputs from other agents.

        Returns
        -------
        AgentContext
            The updated context with the sophisticated synthesis result.
        """
        outputs = context.agent_outputs
        query = context.query.strip()
        logger.info(f"[{self.name}] Running synthesis for query: {query}")
        logger.info(f"[{self.name}] Processing outputs from: {list(outputs.keys())}")

        # Track execution start
        context.start_agent_execution(self.name)

        try:
            # Try structured output first if available, otherwise use traditional approach
            if self.structured_service:
                try:
                    final_synthesis = await self._run_structured(
                        query, outputs, context
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.name}] Structured output failed, falling back to traditional: {e}"
                    )
                    # Fall back to traditional synthesis
                    analysis = await self._analyze_agent_outputs(
                        query, outputs, context
                    )
                    if self.llm:
                        synthesis_result = await self._llm_powered_synthesis(
                            query, outputs, analysis, context
                        )
                    else:
                        synthesis_result = await self._fallback_synthesis(
                            query, outputs, context
                        )
                    final_synthesis = await self._format_final_output(
                        query, synthesis_result, analysis
                    )
            else:
                # Traditional synthesis path (backward compatible)
                analysis = await self._analyze_agent_outputs(query, outputs, context)

                if self.llm:
                    synthesis_result = await self._llm_powered_synthesis(
                        query, outputs, analysis, context
                    )
                else:
                    synthesis_result = await self._fallback_synthesis(
                        query, outputs, context
                    )

                final_synthesis = await self._format_final_output(
                    query, synthesis_result, analysis
                )

            # Step 4: Update context
            context.add_agent_output(self.name, final_synthesis)
            context.set_final_synthesis(final_synthesis)

            # Log successful execution
            logger.info(
                f"[{self.name}] Generated synthesis: {len(final_synthesis)} characters"
            )
            context.log_trace(
                self.name, input_data=outputs, output_data=final_synthesis
            )
            context.complete_agent_execution(self.name, success=True)

            return context

        except Exception as e:
            # Handle failures gracefully
            logger.error(f"[{self.name}] Error during synthesis: {e}")

            # Fall back to basic concatenation
            fallback_output = await self._create_emergency_fallback(query, outputs)
            context.add_agent_output(self.name, fallback_output)
            context.set_final_synthesis(fallback_output)
            context.complete_agent_execution(self.name, success=False)
            context.log_trace(self.name, input_data=outputs, output_data=str(e))

            return context

    async def _analyze_agent_outputs(
        self, query: str, outputs: Dict[str, Any], context: AgentContext
    ) -> Dict[str, Any]:
        """Analyze agent outputs to identify themes, conflicts, and synthesis opportunities."""
        analysis: Dict[str, Any] = {
            "themes": [],
            "conflicts": [],
            "complementary_insights": [],
            "gaps": [],
            "confidence_levels": {},
            "key_topics": [],
            "meta_insights": [],
        }

        if not self.llm:
            # Basic analysis without LLM
            analysis["themes"] = ["synthesis", "multi-agent", "integration"]
            analysis["key_topics"] = [query.split()[0] if query.split() else "general"]
            return analysis

        try:
            # Build analysis prompt
            analysis_prompt = self._build_analysis_prompt(query, outputs)

            # Get LLM analysis
            llm_response = self.llm.generate(analysis_prompt)
            response_text = (
                llm_response.text
                if hasattr(llm_response, "text")
                else str(llm_response)
            )

            # Track token usage for analysis (first LLM call)
            if (
                hasattr(llm_response, "tokens_used")
                and llm_response.tokens_used is not None
            ):
                # Use detailed token breakdown if available
                input_tokens = getattr(llm_response, "input_tokens", None) or 0
                output_tokens = getattr(llm_response, "output_tokens", None) or 0
                total_tokens = llm_response.tokens_used

                # For synthesis, we accumulate token usage across multiple LLM calls
                existing_usage = context.get_agent_token_usage(self.name)

                context.add_agent_token_usage(
                    agent_name=self.name,
                    input_tokens=existing_usage["input_tokens"] + input_tokens,
                    output_tokens=existing_usage["output_tokens"] + output_tokens,
                    total_tokens=existing_usage["total_tokens"] + total_tokens,
                )

                logger.debug(
                    f"[{self.name}] Analysis token usage - "
                    f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
                )

            # Parse analysis response
            analysis = self._parse_analysis_response(response_text)

            logger.debug(f"[{self.name}] Completed thematic analysis")
            return analysis

        except Exception as e:
            logger.error(f"[{self.name}] Analysis failed: {e}")
            return analysis  # Return default analysis

    async def _run_structured(
        self, query: str, outputs: Dict[str, Any], context: AgentContext
    ) -> str:
        """
        Run with structured output using LangChain service.

        This method performs the entire synthesis process using structured output,
        ensuring content pollution prevention and consistent formatting.
        """
        import time

        start_time = time.time()

        if not self.structured_service:
            raise ValueError("Structured service not available")

        try:
            # Step 1: Analyze agent outputs (this remains the same)
            analysis = await self._analyze_agent_outputs(query, outputs, context)

            # Step 2: Create structured synthesis prompt
            system_prompt = self._get_system_prompt()

            # Format agent outputs for the prompt
            outputs_text = "\n\n".join(
                [
                    f"### {agent.upper()} OUTPUT:\n{str(output)}"
                    for agent, output in outputs.items()
                    if agent != self.name
                ]
            )

            # Extract contributing agents
            contributing_agents = [
                agent for agent in outputs.keys() if agent != self.name
            ]

            # Build comprehensive prompt for structured output
            prompt = f"""Original Query: {query}

Agent Outputs:
{outputs_text}

Contributing Agents: {", ".join(contributing_agents)}
Themes Identified: {", ".join(analysis.get("themes", [])[:5])}
Key Topics: {", ".join(analysis.get("key_topics", [])[:10])}
Conflicts Found: {", ".join(analysis.get("conflicts", ["None"])[:5])}
Complementary Insights: {", ".join(analysis.get("complementary_insights", [])[:10])}
Knowledge Gaps: {", ".join(analysis.get("gaps", [])[:8])}
Meta Insights: {", ".join(analysis.get("meta_insights", [])[:5])}

Please provide a comprehensive synthesis according to the system instructions.
Focus on the synthesized content only - do not describe your synthesis process.
The word_count field should reflect the actual word count of your final_synthesis field.
The contributing_agents field should list: {", ".join(contributing_agents)}"""

            # Get structured output
            from cognivault.services.langchain_service import StructuredOutputResult

            result = await self.structured_service.get_structured_output(
                prompt=prompt,
                output_class=SynthesisOutput,
                system_prompt=system_prompt,
                max_retries=3,
            )

            # Handle both SynthesisOutput and StructuredOutputResult types
            if isinstance(result, SynthesisOutput):
                structured_result = result
            else:
                # It's a StructuredOutputResult, extract the parsed result
                if isinstance(result, StructuredOutputResult):
                    parsed_result = result.parsed
                    if not isinstance(parsed_result, SynthesisOutput):
                        raise ValueError(
                            f"Expected SynthesisOutput, got {type(parsed_result)}"
                        )
                    structured_result = parsed_result
                else:
                    raise ValueError(f"Unexpected result type: {type(result)}")

            processing_time_ms = (time.time() - start_time) * 1000

            # Store structured output in execution_state for future use
            if "structured_outputs" not in context.execution_state:
                context.execution_state["structured_outputs"] = {}
            context.execution_state["structured_outputs"][self.name] = (
                structured_result.model_dump()
            )

            # Record token usage - for structured output, we record minimal usage
            # since LangChain doesn't expose detailed token counts
            context.add_agent_token_usage(
                agent_name=self.name,
                input_tokens=0,  # LangChain structured output doesn't expose detailed tokens
                output_tokens=0,
                total_tokens=0,
            )

            logger.info(
                f"[{self.name}] Structured output successful - "
                f"processing_time: {processing_time_ms:.1f}ms, "
                f"themes: {len(structured_result.key_themes)}, "
                f"word_count: {structured_result.word_count}"
            )

            # Format the final output with metadata
            return await self._format_structured_output(
                query, structured_result, analysis
            )

        except Exception as e:
            logger.error(f"[{self.name}] Structured output processing failed: {e}")
            raise

    async def _format_structured_output(
        self, query: str, structured_result: SynthesisOutput, analysis: Dict[str, Any]
    ) -> str:
        """Format the structured output into the final synthesis text."""
        formatted_parts = [
            f"# Comprehensive Analysis: {query}\n",
        ]

        # Add topic summary
        if structured_result.topics_extracted:
            formatted_parts.append(
                f"**Key Topics:** {', '.join(structured_result.topics_extracted[:5])}\n"
            )

        # Add theme overview
        if structured_result.key_themes:
            theme_names = [
                theme.theme_name for theme in structured_result.key_themes[:3]
            ]
            formatted_parts.append(f"**Primary Themes:** {', '.join(theme_names)}\n")

        # Add the main synthesis content
        formatted_parts.append("## Synthesis\n")
        formatted_parts.append(structured_result.final_synthesis)

        # Add meta-insights if available
        if structured_result.meta_insights:
            formatted_parts.append("\n## Meta-Insights\n")
            for insight in structured_result.meta_insights[:3]:
                formatted_parts.append(f"- {insight}")

        return "\n".join(formatted_parts)

    async def _llm_powered_synthesis(
        self,
        query: str,
        outputs: Dict[str, Any],
        analysis: Dict[str, Any],
        context: AgentContext,
    ) -> str:
        """Perform sophisticated LLM-powered synthesis."""
        try:
            # Build comprehensive synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(query, outputs, analysis)

            # Get LLM synthesis
            if not self.llm:
                return await self._fallback_synthesis(query, outputs, context)

            llm_response = self.llm.generate(synthesis_prompt)
            synthesis_text = (
                llm_response.text
                if hasattr(llm_response, "text")
                else str(llm_response)
            )

            # Track token usage for synthesis (second LLM call - accumulate)
            if (
                hasattr(llm_response, "tokens_used")
                and llm_response.tokens_used is not None
            ):
                # Use detailed token breakdown if available
                input_tokens = getattr(llm_response, "input_tokens", None) or 0
                output_tokens = getattr(llm_response, "output_tokens", None) or 0
                total_tokens = llm_response.tokens_used

                # Accumulate with existing usage from analysis call
                existing_usage = context.get_agent_token_usage(self.name)

                context.add_agent_token_usage(
                    agent_name=self.name,
                    input_tokens=existing_usage["input_tokens"] + input_tokens,
                    output_tokens=existing_usage["output_tokens"] + output_tokens,
                    total_tokens=existing_usage["total_tokens"] + total_tokens,
                )

                logger.debug(
                    f"[{self.name}] Synthesis token usage - "
                    f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}. "
                    f"Total accumulated: {existing_usage['total_tokens'] + total_tokens}"
                )

            logger.debug(
                f"[{self.name}] LLM synthesis completed: {len(synthesis_text)} characters"
            )
            return synthesis_text

        except Exception as e:
            logger.error(f"[{self.name}] LLM synthesis failed: {e}")
            return await self._fallback_synthesis(query, outputs, context)

    async def _fallback_synthesis(
        self, query: str, outputs: Dict[str, Any], context: AgentContext
    ) -> str:
        """Create fallback synthesis when LLM is unavailable."""
        logger.info(f"[{self.name}] Using fallback synthesis")

        synthesis_parts = [
            f"# Synthesis for: {query}\n",
            "## Integrated Analysis\n",
            "The following synthesis combines insights from multiple analytical agents:\n",
        ]

        # Add each agent's contribution
        for agent_name, output in outputs.items():
            if agent_name != self.name:  # Don't include our own output
                synthesis_parts.append(f"### {agent_name} Analysis")
                synthesis_parts.append(str(output).strip())
                synthesis_parts.append("")

        # Add basic conclusion
        synthesis_parts.extend(
            [
                "## Summary",
                f"This synthesis integrates perspectives from {len(outputs)} agents to provide comprehensive analysis of: {query}",
            ]
        )

        return "\n".join(synthesis_parts)

    async def _format_final_output(
        self, query: str, synthesis_result: str, analysis: Dict[str, Any]
    ) -> str:
        """Format the final synthesis output with metadata and structure."""
        # Extract key elements for formatting
        themes = analysis.get("themes", [])
        key_topics = analysis.get("key_topics", [])

        formatted_parts = [
            f"# Comprehensive Analysis: {query}\n",
        ]

        # Add topic summary if available
        if key_topics:
            formatted_parts.append(f"**Key Topics:** {', '.join(key_topics[:5])}\n")

        # Add theme overview if available
        if themes:
            formatted_parts.append(f"**Primary Themes:** {', '.join(themes[:3])}\n")

        # Add the main synthesis content
        formatted_parts.append("## Synthesis\n")
        formatted_parts.append(synthesis_result)

        # Add meta-insights if available
        meta_insights = analysis.get("meta_insights", [])
        if meta_insights:
            formatted_parts.append("\n## Meta-Insights\n")
            for insight in meta_insights[:3]:
                formatted_parts.append(f"- {insight}")

        return "\n".join(formatted_parts)

    async def _create_emergency_fallback(
        self, query: str, outputs: Dict[str, Any]
    ) -> str:
        """Create emergency fallback when all other synthesis methods fail."""
        logger.warning(f"[{self.name}] Using emergency fallback synthesis")

        fallback_parts = [
            f"# Emergency Synthesis: {query}\n",
            "## Agent Outputs\n",
            "*Note: This is a basic concatenation due to synthesis system failure.*\n",
        ]

        for agent, output in outputs.items():
            if agent != self.name:
                fallback_parts.append(f"### {agent}")
                fallback_parts.append(
                    str(output)[:500] + "..." if len(str(output)) > 500 else str(output)
                )
                fallback_parts.append("")

        return "\n".join(fallback_parts)

    def _build_analysis_prompt(self, query: str, outputs: Dict[str, Any]) -> str:
        """Build prompt for thematic analysis of agent outputs."""
        outputs_text = "\n\n".join(
            [
                f"### {agent.upper()} OUTPUT:\n{str(output)}"
                for agent, output in outputs.items()
                if agent != self.name
            ]
        )

        # Try to use composed prompt from PromptComposer first
        if self._composed_prompt:
            analysis_template = self._composed_prompt.get_template("analysis_prompt")
            if analysis_template:
                try:
                    formatted_prompt: str = analysis_template.format(
                        query=query, outputs_text=outputs_text
                    )
                    return formatted_prompt
                except Exception as e:
                    logger.debug(
                        f"[{self.name}] Failed to use composed analysis prompt: {e}"
                    )

        # Try to load prompt template from prompts.py
        try:
            from cognivault.agents.synthesis.prompts import (
                SYNTHESIS_ANALYSIS_PROMPT_TEMPLATE,
            )

            return SYNTHESIS_ANALYSIS_PROMPT_TEMPLATE.format(
                query=query, outputs_text=outputs_text
            )
        except ImportError:
            # Fallback to embedded prompt
            return f"""As an expert analyst, perform thematic analysis of multiple agent outputs for synthesis.

ORIGINAL QUERY: {query}

AGENT OUTPUTS:
{outputs_text}

Analyze the outputs and provide a structured analysis in the following format:

THEMES: [list 3-5 main themes across all outputs]
CONFLICTS: [identify any contradictions or disagreements between agents]
COMPLEMENTARY: [highlight insights that build on each other]
GAPS: [note any important aspects not covered]
TOPICS: [extract 5-10 key topics/concepts mentioned]
META_INSIGHTS: [provide 2-3 higher-level insights about the analysis process itself]

Provide your analysis in the exact format above."""

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM analysis response into structured data."""
        analysis: Dict[str, Any] = {
            "themes": [],
            "conflicts": [],
            "complementary_insights": [],
            "gaps": [],
            "key_topics": [],
            "meta_insights": [],
        }

        try:
            lines = response_text.strip().split("\n")
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith("THEMES:"):
                    current_section = "themes"
                    content = line.replace("THEMES:", "").strip()
                    if content:
                        analysis["themes"] = [t.strip() for t in content.split(",")]
                elif line.startswith("CONFLICTS:"):
                    current_section = "conflicts"
                    content = line.replace("CONFLICTS:", "").strip()
                    if content:
                        analysis["conflicts"] = [c.strip() for c in content.split(",")]
                elif line.startswith("COMPLEMENTARY:"):
                    current_section = "complementary_insights"
                    content = line.replace("COMPLEMENTARY:", "").strip()
                    if content:
                        analysis["complementary_insights"] = [
                            c.strip() for c in content.split(",")
                        ]
                elif line.startswith("GAPS:"):
                    current_section = "gaps"
                    content = line.replace("GAPS:", "").strip()
                    if content:
                        analysis["gaps"] = [g.strip() for g in content.split(",")]
                elif line.startswith("TOPICS:"):
                    current_section = "key_topics"
                    content = line.replace("TOPICS:", "").strip()
                    if content:
                        analysis["key_topics"] = [t.strip() for t in content.split(",")]
                elif line.startswith("META_INSIGHTS:"):
                    current_section = "meta_insights"
                    content = line.replace("META_INSIGHTS:", "").strip()
                    if content:
                        analysis["meta_insights"] = [
                            m.strip() for m in content.split(",")
                        ]
                elif (
                    line
                    and current_section
                    and not line.startswith(
                        tuple(
                            [
                                "THEMES:",
                                "CONFLICTS:",
                                "COMPLEMENTARY:",
                                "GAPS:",
                                "TOPICS:",
                                "META_INSIGHTS:",
                            ]
                        )
                    )
                ):
                    # Continue parsing multi-line content
                    if current_section in analysis:
                        if isinstance(analysis[current_section], list):
                            analysis[current_section].append(line)

            return analysis

        except Exception as e:
            logger.error(f"[{self.name}] Failed to parse analysis response: {e}")
            return analysis

    def _build_synthesis_prompt(
        self, query: str, outputs: Dict[str, Any], analysis: Dict[str, Any]
    ) -> str:
        """Build comprehensive synthesis prompt."""
        outputs_text = "\n\n".join(
            [
                f"### {agent.upper()}:\n{str(output)}"
                for agent, output in outputs.items()
                if agent != self.name
            ]
        )

        themes_text = ", ".join(analysis.get("themes", []))
        conflicts_text = ", ".join(analysis.get("conflicts", ["None identified"]))
        topics_text = ", ".join(analysis.get("key_topics", []))

        # Try to use composed prompt from PromptComposer first
        if self._composed_prompt:
            synthesis_template = self._composed_prompt.get_template("synthesis_prompt")
            if synthesis_template:
                try:
                    formatted_prompt: str = synthesis_template.format(
                        query=query,
                        themes_text=themes_text,
                        topics_text=topics_text,
                        conflicts_text=conflicts_text,
                        outputs_text=outputs_text,
                    )
                    return formatted_prompt
                except Exception as e:
                    logger.debug(
                        f"[{self.name}] Failed to use composed synthesis prompt: {e}"
                    )

        # Try to load prompt template from prompts.py
        try:
            from cognivault.agents.synthesis.prompts import (
                SYNTHESIS_COMPOSITION_PROMPT_TEMPLATE,
            )

            return SYNTHESIS_COMPOSITION_PROMPT_TEMPLATE.format(
                query=query,
                themes_text=themes_text,
                topics_text=topics_text,
                conflicts_text=conflicts_text,
                outputs_text=outputs_text,
            )
        except ImportError:
            # Fallback to embedded prompt
            return f"""As a knowledge synthesis expert, create a comprehensive, wiki-ready synthesis of multiple expert analyses.

ORIGINAL QUERY: {query}

IDENTIFIED THEMES: {themes_text}
KEY TOPICS: {topics_text}
CONFLICTS TO RESOLVE: {conflicts_text}

EXPERT ANALYSES:
{outputs_text}

Create a sophisticated synthesis that:
1. Integrates all perspectives into a coherent narrative
2. Resolves any conflicts or contradictions intelligently
3. Highlights emergent insights from combining analyses
4. Provides a definitive, comprehensive answer to the original query
5. Uses clear, wiki-style formatting with appropriate headers
6. Includes nuanced conclusions that acknowledge complexity

COMPREHENSIVE SYNTHESIS:"""

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
