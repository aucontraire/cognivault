import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from cognivault.agents.base_agent import (
    BaseAgent,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
)
from cognivault.context import AgentContext
from cognivault.config.app_config import get_config
from cognivault.llm.llm_interface import LLMInterface
from cognivault.agents.historian.search import SearchFactory, SearchResult

# Configuration system imports
from cognivault.config.agent_configs import HistorianConfig
from cognivault.workflows.prompt_composer import PromptComposer

logger = logging.getLogger(__name__)


class HistorianAgent(BaseAgent):
    """
    Enhanced agent that retrieves and analyzes historical context using intelligent search
    and LLM-powered relevance analysis.

    Combines multiple search strategies with sophisticated relevance filtering to provide
    contextually appropriate historical information that informs current queries.

    Parameters
    ----------
    llm : LLMInterface, optional
        LLM interface for relevance analysis. If None, uses default OpenAI setup.
    search_type : str, optional
        Type of search strategy to use. Defaults to "hybrid".
    config : Optional[HistorianConfig], optional
        Configuration for agent behavior. If None, uses default configuration.
        Maintains backward compatibility - existing code continues to work.

    Attributes
    ----------
    config : HistorianConfig
        Configuration for agent behavior and prompt composition.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        llm: Optional[Union[LLMInterface, str]] = "default",
        search_type: str = "hybrid",
        config: Optional[HistorianConfig] = None,
    ):
        super().__init__("historian")

        # Configuration system - backward compatible
        self.config = config if config is not None else HistorianConfig()
        self._prompt_composer = PromptComposer()
        self._composed_prompt = None

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
        self.search_engine = SearchFactory.create_search(search_type)
        self.search_type = search_type

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
            logger.warning(f"Failed to create OpenAI LLM: {e}. Using mock LLM.")
            return None

    def _update_composed_prompt(self):
        """Update the composed prompt based on current configuration."""
        try:
            self._composed_prompt = self._prompt_composer.compose_historian_prompt(
                self.config
            )
            self.logger.debug(
                f"[{self.name}] Prompt composed with config: {self.config.search_depth}"
            )
        except Exception as e:
            self.logger.warning(
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
            self.logger.debug(f"[{self.name}] Using default system prompt (fallback)")
            return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for fallback compatibility."""
        try:
            from cognivault.agents.historian.prompts import HISTORIAN_SYSTEM_PROMPT

            return HISTORIAN_SYSTEM_PROMPT
        except ImportError:
            # Fallback to basic embedded prompt
            return """As a historian agent, analyze queries and provide relevant historical context using available search results and historical information."""

    def update_config(self, config: HistorianConfig):
        """
        Update the agent configuration and recompose prompts.

        Parameters
        ----------
        config : HistorianConfig
            New configuration to apply
        """
        self.config = config
        self._update_composed_prompt()
        self.logger.info(
            f"[{self.name}] Configuration updated: {config.search_depth} search depth"
        )

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Executes the enhanced Historian agent with intelligent search and LLM analysis.

        Parameters
        ----------
        context : AgentContext
            The current context object containing the user query and accumulated outputs.

        Returns
        -------
        AgentContext
            The updated context object with the Historian's output and retrieved notes.
        """
        query = context.query.strip()
        self.logger.info(f"[{self.name}] Received query: {query}")

        # Use configurable simulation delay if enabled
        config = get_config()
        if config.execution.enable_simulation_delay:
            await asyncio.sleep(config.execution.simulation_delay_seconds)

        # Track execution start
        context.start_agent_execution(self.name)

        try:
            # Step 1: Search for relevant historical content
            search_results = await self._search_historical_content(query, context)

            # Step 2: Analyze and filter results with LLM
            filtered_results = await self._analyze_relevance(
                query, search_results, context
            )

            # Step 3: Synthesize findings into contextual summary
            historical_summary = await self._synthesize_historical_context(
                query, filtered_results, context
            )

            # Step 4: Update context with results
            context.retrieved_notes = [result.filepath for result in filtered_results]
            context.add_agent_output(self.name, historical_summary)

            # Log successful execution
            self.logger.info(
                f"[{self.name}] Found {len(filtered_results)} relevant historical notes"
            )
            context.log_trace(
                self.name, input_data=query, output_data=historical_summary
            )
            context.complete_agent_execution(self.name, success=True)

            return context

        except Exception as e:
            # Handle failures gracefully
            self.logger.error(f"[{self.name}] Error during execution: {e}")

            # Fall back to mock data if configured
            if config.testing.mock_history_entries:
                fallback_output = await self._create_fallback_output(
                    query, config.testing.mock_history_entries
                )
                context.add_agent_output(self.name, fallback_output)
                context.retrieved_notes = config.testing.mock_history_entries
                context.complete_agent_execution(self.name, success=True)
                self.logger.info(f"[{self.name}] Used fallback mock data")
            else:
                # No historical context available
                no_context_output = await self._create_no_context_output(query)
                context.add_agent_output(self.name, no_context_output)
                context.complete_agent_execution(self.name, success=False)

            context.log_trace(self.name, input_data=query, output_data=str(e))
            return context

    async def _search_historical_content(
        self, query: str, context: AgentContext
    ) -> List[SearchResult]:
        """Search for relevant historical content using resilient search processing."""
        try:
            # Import resilient processor
            from cognivault.agents.historian.resilient_search import (
                ResilientSearchProcessor,
            )

            # Use configured search limit
            config = get_config()
            search_limit = getattr(config.testing, "historian_search_limit", 10)

            # Create resilient processor with LLM for title generation
            processor = ResilientSearchProcessor(llm_client=self.llm)

            # Use resilient search processing
            (
                search_results,
                validation_report,
            ) = await processor.process_search_with_recovery(
                self.search_engine, query, limit=search_limit
            )

            self.logger.debug(
                f"[{self.name}] Found {len(search_results)} search results "
                f"({validation_report.recovered_validations} recovered)"
            )

            # Log validation issues for monitoring
            if validation_report.failed_validations > 0:
                self.logger.warning(
                    f"[{self.name}] {validation_report.failed_validations} documents failed validation, "
                    f"{validation_report.recovered_validations} recovered"
                )

                # Log data quality insights
                for insight in validation_report.data_quality_insights:
                    self.logger.info(f"[{self.name}] Data quality insight: {insight}")

            return search_results

        except Exception as e:
            self.logger.error(f"[{self.name}] Search failed: {e}")
            return []

    async def _analyze_relevance(
        self, query: str, search_results: List[SearchResult], context: AgentContext
    ) -> List[SearchResult]:
        """Use LLM to analyze relevance and filter search results."""
        if not search_results:
            return []

        # If no LLM available, return top results based on search scores
        if not self.llm:
            return search_results[:5]  # Return top 5 results

        try:
            # Prepare relevance analysis prompt
            relevance_prompt = self._build_relevance_prompt(query, search_results)

            # Get LLM analysis
            llm_response = self.llm.generate(relevance_prompt)

            # Parse response to get relevant result indices
            response_text = (
                llm_response.text
                if hasattr(llm_response, "text")
                else str(llm_response)
            )

            # Track token usage for relevance analysis
            if (
                hasattr(llm_response, "tokens_used")
                and llm_response.tokens_used is not None
            ):
                # Use detailed token breakdown if available
                input_tokens = getattr(llm_response, "input_tokens", None) or 0
                output_tokens = getattr(llm_response, "output_tokens", None) or 0
                total_tokens = llm_response.tokens_used

                # For historian, we accumulate token usage across multiple LLM calls
                existing_usage = context.get_agent_token_usage(self.name)

                context.add_agent_token_usage(
                    agent_name=self.name,
                    input_tokens=existing_usage["input_tokens"] + input_tokens,
                    output_tokens=existing_usage["output_tokens"] + output_tokens,
                    total_tokens=existing_usage["total_tokens"] + total_tokens,
                )

                self.logger.debug(
                    f"[{self.name}] Relevance analysis token usage - "
                    f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
                )

            relevant_indices = self._parse_relevance_response(response_text)

            # Filter results based on LLM analysis
            filtered_results = [
                search_results[i] for i in relevant_indices if i < len(search_results)
            ]

            self.logger.debug(
                f"[{self.name}] LLM filtered {len(search_results)} to {len(filtered_results)} results"
            )
            return filtered_results

        except Exception as e:
            self.logger.error(f"[{self.name}] LLM relevance analysis failed: {e}")
            # Fall back to top search results
            return search_results[:5]

    async def _synthesize_historical_context(
        self, query: str, filtered_results: List[SearchResult], context: AgentContext
    ) -> str:
        """Synthesize historical findings into a contextual summary."""
        if not filtered_results:
            return f"No relevant historical context found for: {query}"

        # If no LLM available, create basic summary
        if not self.llm:
            return self._create_basic_summary(query, filtered_results)

        try:
            # Prepare synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(query, filtered_results)

            # Get LLM synthesis
            llm_response = self.llm.generate(synthesis_prompt)
            historical_summary = (
                llm_response.text
                if hasattr(llm_response, "text")
                else str(llm_response)
            )

            # Track token usage for synthesis (accumulate with previous usage)
            if (
                hasattr(llm_response, "tokens_used")
                and llm_response.tokens_used is not None
            ):
                # Use detailed token breakdown if available
                input_tokens = getattr(llm_response, "input_tokens", None) or 0
                output_tokens = getattr(llm_response, "output_tokens", None) or 0
                total_tokens = llm_response.tokens_used

                # Accumulate with existing usage from relevance analysis
                existing_usage = context.get_agent_token_usage(self.name)

                context.add_agent_token_usage(
                    agent_name=self.name,
                    input_tokens=existing_usage["input_tokens"] + input_tokens,
                    output_tokens=existing_usage["output_tokens"] + output_tokens,
                    total_tokens=existing_usage["total_tokens"] + total_tokens,
                )

                self.logger.debug(
                    f"[{self.name}] Synthesis token usage - "
                    f"input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
                )

            self.logger.debug(
                f"[{self.name}] Generated historical summary: {len(historical_summary)} characters"
            )
            return historical_summary

        except Exception as e:
            self.logger.error(f"[{self.name}] LLM synthesis failed: {e}")
            # Fall back to basic summary
            return self._create_basic_summary(query, filtered_results)

    def _build_relevance_prompt(
        self, query: str, search_results: List[SearchResult]
    ) -> str:
        """Build prompt for LLM relevance analysis."""
        results_text = "\n".join(
            [
                f"[{i}] TITLE: {result.title}\n    EXCERPT: {result.excerpt}\n    TOPICS: {', '.join(result.topics)}\n    MATCH: {result.match_type} ({result.relevance_score:.2f})"
                for i, result in enumerate(search_results)
            ]
        )

        # Try to load prompt template from prompts.py
        try:
            from cognivault.agents.historian.prompts import (
                HISTORIAN_RELEVANCE_PROMPT_TEMPLATE,
            )

            return HISTORIAN_RELEVANCE_PROMPT_TEMPLATE.format(
                query=query, results_text=results_text
            )
        except ImportError:
            # Fallback to embedded prompt
            return f"""As a historian analyzing relevance, determine which historical notes are most relevant to the current query.

QUERY: {query}

HISTORICAL NOTES:
{results_text}

Instructions:
1. Analyze each note for relevance to the query
2. Consider topic overlap, content similarity, and contextual connections
3. Respond with ONLY the indices (0-based) of relevant notes, separated by commas
4. Include maximum 5 most relevant notes
5. If no notes are relevant, respond with "NONE"

RELEVANT INDICES:"""

    def _parse_relevance_response(self, llm_response: str) -> List[int]:
        """Parse LLM response to extract relevant result indices."""
        try:
            response_clean = llm_response.strip().upper()

            if response_clean == "NONE":
                return []

            # Extract numbers from response
            import re

            numbers = re.findall(r"\d+", response_clean)
            indices = [int(num) for num in numbers]

            # Limit to maximum 5 results
            return indices[:5]

        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to parse relevance response: {e}")
            return list(range(min(5, len(llm_response))))  # Default to first 5

    def _build_synthesis_prompt(
        self, query: str, filtered_results: List[SearchResult]
    ) -> str:
        """Build prompt for LLM historical context synthesis."""
        results_context = "\n\n".join(
            [
                f"### {result.title} ({result.date})\n{result.excerpt}\nTopics: {', '.join(result.topics)}\nSource: {result.filename}"
                for result in filtered_results
            ]
        )

        # Try to load prompt template from prompts.py
        try:
            from cognivault.agents.historian.prompts import (
                HISTORIAN_SYNTHESIS_PROMPT_TEMPLATE,
            )

            return HISTORIAN_SYNTHESIS_PROMPT_TEMPLATE.format(
                query=query, results_context=results_context
            )
        except ImportError:
            # Fallback to embedded prompt
            return f"""As a historian, synthesize the following historical context to inform the current query.

CURRENT QUERY: {query}

RELEVANT HISTORICAL CONTEXT:
{results_context}

Instructions:
1. Synthesize the historical information into a coherent narrative
2. Highlight patterns, themes, and connections relevant to the current query
3. Provide context that would inform understanding of the current question
4. Be concise but comprehensive (2-3 paragraphs maximum)
5. Include specific references to the historical sources when relevant

HISTORICAL SYNTHESIS:"""

    def _create_basic_summary(
        self, query: str, filtered_results: List[SearchResult]
    ) -> str:
        """Create a basic summary when LLM is not available."""
        if not filtered_results:
            return f"No relevant historical context found for: {query}"

        summary_parts = [
            f"Found {len(filtered_results)} relevant historical notes for: {query}\n"
        ]

        for result in filtered_results:
            summary_parts.append(f"• {result.title} ({result.date})")
            summary_parts.append(f"  Topics: {', '.join(result.topics)}")
            summary_parts.append(
                f"  Match: {result.match_type} (score: {result.relevance_score:.2f})"
            )
            summary_parts.append(f"  Excerpt: {result.excerpt[:100]}...")
            summary_parts.append("")

        return "\n".join(summary_parts)

    async def _create_fallback_output(self, query: str, mock_history: List[str]) -> str:
        """Create fallback output using mock history data."""
        return f"Historical context for: {query}\n\nUsing fallback data:\n" + "\n".join(
            mock_history
        )

    async def _create_no_context_output(self, query: str) -> str:
        """Create output when no historical context is available."""
        return f"No historical context available for: {query}\n\nThis appears to be a new topic or the notes directory is empty."

    def define_node_metadata(self) -> Dict[str, Any]:
        """
        Define LangGraph-specific metadata for the Historian agent.

        Returns
        -------
        Dict[str, Any]
            Node metadata including type, dependencies, schemas, and routing logic
        """
        return {
            "node_type": NodeType.PROCESSOR,
            "dependencies": [],  # Independent - can run in parallel with other entry agents
            "description": "Retrieves historical context and relevant notes for the given query",
            "inputs": [
                NodeInputSchema(
                    name="context",
                    description="Agent context containing query for historical context retrieval",
                    required=True,
                    type_hint="AgentContext",
                )
            ],
            "outputs": [
                NodeOutputSchema(
                    name="context",
                    description="Updated context with historical notes and retrieved information",
                    type_hint="AgentContext",
                )
            ],
            "tags": ["historian", "agent", "processor", "independent", "parallel"],
        }
