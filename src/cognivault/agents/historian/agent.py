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

# Database repository imports
from cognivault.database.session_factory import DatabaseSessionFactory
from cognivault.database.repositories import RepositoryFactory

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

        # Database components for hybrid search
        self._db_session_factory: Optional[DatabaseSessionFactory] = None
        self._repository_factory: Optional[RepositoryFactory] = None

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

    async def _ensure_database_connection(self) -> Optional[DatabaseSessionFactory]:
        """Ensure database connection is established and return session factory."""
        if self._db_session_factory is not None:
            return self._db_session_factory

        try:
            # Initialize database session factory if not already done
            self._db_session_factory = DatabaseSessionFactory()

            # Add timeout for database initialization
            import asyncio

            await asyncio.wait_for(
                self._db_session_factory.initialize(),
                timeout=self.config.search_timeout_seconds,
            )

            self.logger.debug(f"[{self.name}] Database connection initialized")
            return self._db_session_factory

        except asyncio.TimeoutError:
            self.logger.warning(
                f"[{self.name}] Database initialization timed out after {self.config.search_timeout_seconds}s"
            )
            return None
        except Exception as e:
            self.logger.warning(f"[{self.name}] Failed to initialize database: {e}")
            return None

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
        """Search for relevant historical content using hybrid file + database search."""
        all_results: List[SearchResult] = []

        try:
            # Use configured search limit
            config = get_config()
            search_limit = getattr(config.testing, "historian_search_limit", 10)

            # Check if hybrid search is enabled using agent config first, then fallback to testing config
            enable_hybrid_search = self.config.hybrid_search_enabled or getattr(
                config.testing, "enable_hybrid_search", True
            )

            if not enable_hybrid_search:
                # Legacy mode: file-only search for backward compatibility
                return await self._search_file_content(query, search_limit)

            # Calculate split between file and database search using configurable ratio
            file_ratio = self.config.hybrid_search_file_ratio
            file_limit = max(1, int(search_limit * file_ratio))
            db_limit = max(1, search_limit - file_limit)

            # Step 1: File-based search using existing resilient processor
            file_results = await self._search_file_content(query, file_limit)
            all_results.extend(file_results)

            # Step 2: Database search for additional content
            db_results = await self._search_database_content(query, db_limit)
            all_results.extend(db_results)

            # Step 3: Remove duplicates and rank by relevance
            deduplicated_results = self._deduplicate_search_results(all_results)

            # Limit to search_limit and rank by relevance score
            final_results = sorted(
                deduplicated_results, key=lambda r: r.relevance_score, reverse=True
            )[:search_limit]

            self.logger.debug(
                f"[{self.name}] Hybrid search: {len(file_results)} file + {len(db_results)} db = "
                f"{len(final_results)} total results (after deduplication)"
            )

            return final_results

        except Exception as e:
            self.logger.error(f"[{self.name}] Hybrid search failed: {e}")
            # Fallback to file-only search
            return await self._search_file_content(query, search_limit)

    async def _search_file_content(self, query: str, limit: int) -> List[SearchResult]:
        """Search file-based content using existing resilient processor."""
        try:
            # Import resilient processor
            from cognivault.agents.historian.resilient_search import (
                ResilientSearchProcessor,
            )

            # Create resilient processor with LLM for title generation
            processor = ResilientSearchProcessor(llm_client=self.llm)

            # Use resilient search processing
            (
                search_results,
                validation_report,
            ) = await processor.process_search_with_recovery(
                self.search_engine, query, limit=limit
            )

            self.logger.debug(
                f"[{self.name}] File search: {len(search_results)} results "
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
            self.logger.error(f"[{self.name}] File search failed: {e}")
            return []

    async def _search_database_content(
        self, query: str, limit: int
    ) -> List[SearchResult]:
        """Search database content using repository pattern."""
        try:
            # Ensure database connection
            session_factory = await self._ensure_database_connection()
            if session_factory is None:
                self.logger.debug(
                    f"[{self.name}] Database not available, skipping database search"
                )
                return []

            # Use repository factory context manager
            async with session_factory.get_repository_factory() as repo_factory:
                # Get historian document repository
                doc_repo = repo_factory.historian_documents
                analytics_repo = repo_factory.historian_search_analytics

                # Perform fulltext search
                import time

                start_time = time.time()

                documents = await doc_repo.fulltext_search(query, limit=limit)

                execution_time_ms = int((time.time() - start_time) * 1000)

                # Log search analytics
                await analytics_repo.log_search(
                    search_query=query,
                    search_type="database_fulltext",
                    results_count=len(documents),
                    execution_time_ms=execution_time_ms,
                    search_metadata={"limit": limit, "agent": "historian"},
                )

                # Convert database documents to SearchResult format
                search_results = []
                for doc in documents:
                    # Create metadata with topics and other info
                    metadata = {
                        "topics": (
                            list(doc.document_metadata.get("topics", []))
                            if doc.document_metadata
                            else []
                        ),
                        "word_count": (
                            doc.word_count if hasattr(doc, "word_count") else 0
                        ),
                        "database_id": str(doc.id),
                        "source": "database",
                    }

                    # Create SearchResult compatible with existing code
                    search_result = SearchResult(
                        title=doc.title,
                        excerpt=(
                            doc.content[:200] + "..."
                            if len(doc.content) > 200
                            else doc.content
                        ),
                        filepath=doc.source_path or f"db_doc_{doc.id}",
                        filename=f"document_{doc.id}",
                        date=(
                            doc.created_at.strftime("%Y-%m-%d")
                            if doc.created_at
                            else "unknown"
                        ),
                        match_type="content",
                        relevance_score=0.8 + self.config.database_relevance_boost,
                        metadata=metadata,
                    )
                    search_results.append(search_result)

                self.logger.debug(
                    f"[{self.name}] Database search: {len(search_results)} results in {execution_time_ms}ms"
                )
                return search_results

        except Exception as e:
            self.logger.error(f"[{self.name}] Database search failed: {e}")
            return []

    def _deduplicate_search_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Remove duplicate search results based on configurable similarity threshold."""
        if not results:
            return []

        deduplicated: List[SearchResult] = []

        for result in results:
            is_duplicate = False

            # Check against all existing results for similarity
            for existing in deduplicated:
                similarity = self._calculate_result_similarity(result, existing)
                if similarity >= self.config.deduplication_threshold:
                    is_duplicate = True
                    self.logger.debug(
                        f"[{self.name}] Found duplicate (similarity: {similarity:.2f}): "
                        f"'{result.title}' vs '{existing.title}'"
                    )
                    break

            if not is_duplicate:
                deduplicated.append(result)

        self.logger.debug(
            f"[{self.name}] Deduplicated {len(results)} to {len(deduplicated)} results "
            f"(threshold: {self.config.deduplication_threshold})"
        )
        return deduplicated

    def _calculate_result_similarity(
        self, result1: SearchResult, result2: SearchResult
    ) -> float:
        """Calculate similarity between two search results."""
        # Title similarity (weighted 40%)
        title_similarity = self._text_similarity(
            result1.title.lower(), result2.title.lower()
        )

        # Excerpt similarity (weighted 60%)
        excerpt_similarity = self._text_similarity(
            result1.excerpt.lower(), result2.excerpt.lower()
        )

        # Combined weighted similarity
        return 0.4 * title_similarity + 0.6 * excerpt_similarity

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using character overlap."""
        if not text1 or not text2:
            return 0.0

        # Exact match
        if text1 == text2:
            return 1.0

        # Character set similarity (Jaccard similarity)
        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

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
