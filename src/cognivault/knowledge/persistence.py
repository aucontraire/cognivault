"""Knowledge persistence service — the US1 write path.

Persists each completed pipeline run's refined question, its >=-threshold topics,
and the synthesis output to the existing ``questions`` / ``topics`` / ``wiki_entries``
tables. The public entry point ``persist_run`` NEVER raises: any failure is caught,
logged, evented, and returned as a ``PersistedRunResult`` with ``skipped_reason``
(FR-001 / FR-003 / FR-013). Persistence is skipped entirely when the database is
unavailable, so a run behaves exactly as before.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from cognivault.database.repositories.factory import RepositoryFactory
from cognivault.database.session_factory import (
    DatabaseSessionFactory,
    get_database_session_factory,
)
from cognivault.events.emitter import emit_agent_execution_completed
from cognivault.knowledge.config import KnowledgePersistenceConfig
from cognivault.knowledge.embedding import EmbeddingError, EmbeddingService
from cognivault.observability import get_logger
from cognivault.store.topic_manager import TopicManager, TopicSuggestion

logger = get_logger(__name__)


class PersistedRunResult(BaseModel):
    """Outcome of a ``persist_run`` call (data-model §3)."""

    question_id: Optional[UUID] = None
    topic_ids: List[UUID] = Field(default_factory=list)
    primary_topic_id: Optional[UUID] = None
    wiki_entry_id: Optional[UUID] = None
    embeddings_written: int = 0
    embeddings_deferred: int = 0
    suggestions_filtered: int = 0
    skipped_reason: Optional[str] = None


class KnowledgePersistenceService:
    """Coordinates the pipeline→repository write path, using existing repositories."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        topic_manager: TopicManager,
        config: KnowledgePersistenceConfig,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self._session_factory = session_factory
        self._topic_manager = topic_manager
        self._config = config
        self._embedding_service = embedding_service

    @classmethod
    def from_defaults(cls) -> "KnowledgePersistenceService":
        """Build with the process-wide session factory, a default TopicManager, and
        ``KnowledgePersistenceConfig.from_env`` — cheap and side-effect-free (no DB
        connection until ``persist_run``). The embedding service is optional: with no
        API key configured, topics persist with null embeddings (recoverable via backfill).
        """
        embedding_service: Optional[EmbeddingService] = None
        try:
            embedding_service = EmbeddingService.from_env()
        except Exception:
            logger.info(
                "Embedding service unavailable (no OPENAI_API_KEY?); "
                "topics will persist without embeddings"
            )
        return cls(
            session_factory=get_database_session_factory(),
            topic_manager=TopicManager(),
            config=KnowledgePersistenceConfig.from_env(),
            embedding_service=embedding_service,
        )

    async def persist_run(
        self,
        *,
        refined_query: str,
        agent_outputs: Dict[str, Any],
        structured_outputs: Dict[str, Any],
        correlation_id: Optional[str],
        execution_id: Optional[str],
        nodes_executed: Optional[List[str]] = None,
    ) -> PersistedRunResult:
        """Persist a run's knowledge. Never raises (FR-003)."""
        try:
            return await self._persist(
                refined_query=refined_query,
                agent_outputs=agent_outputs,
                structured_outputs=structured_outputs,
                correlation_id=correlation_id,
                execution_id=execution_id,
                nodes_executed=nodes_executed,
            )
        except Exception as exc:  # graceful degradation — must not fail the run
            logger.warning(
                f"Knowledge persistence failed (non-blocking); run unaffected: "
                f"{type(exc).__name__}: {exc}"
            )
            return PersistedRunResult(skipped_reason="error")

    async def _persist(
        self,
        *,
        refined_query: str,
        agent_outputs: Dict[str, Any],
        structured_outputs: Dict[str, Any],
        correlation_id: Optional[str],
        execution_id: Optional[str],
        nodes_executed: Optional[List[str]],
    ) -> PersistedRunResult:
        if not self._config.persistence_enabled:
            return PersistedRunResult(skipped_reason="disabled")

        # Gate on database availability (FR-003). initialize() is idempotent and
        # raises if the DB is unreachable → skip gracefully.
        try:
            await self._session_factory.initialize()
        except Exception:
            logger.info("Knowledge persistence skipped: database unavailable")
            return PersistedRunResult(skipped_reason="no_database")

        # Topic suggestions + confidence gate (FR-011).
        analysis = await self._topic_manager.analyze_and_suggest_topics(
            query=refined_query, agent_outputs=agent_outputs
        )
        threshold = self._config.topic_min_confidence
        qualifying = [s for s in analysis.suggested_topics if s.confidence >= threshold]
        filtered = [s for s in analysis.suggested_topics if s.confidence < threshold]

        synthesis_content = self._extract_synthesis(structured_outputs, agent_outputs)

        async with self._session_factory.get_repository_factory() as repo:
            # Idempotency (FR-013): a run already recorded is a no-op.
            if correlation_id is not None:
                existing = await repo.questions.get_by_correlation_id(correlation_id)
                if existing is not None:
                    return PersistedRunResult(
                        question_id=existing.id, skipped_reason="already_persisted"
                    )

            # Get-or-create qualifying topics (deduped, concurrency-safe).
            topic_ids: List[UUID] = []
            primary_id: Optional[UUID] = None
            best_confidence = -1.0
            new_topics: List[tuple[UUID, str]] = []  # (id, embedding input text)
            for suggestion in qualifying:
                topic, created = await repo.topics.get_or_create_by_canonical_name(
                    suggestion.topic, description=suggestion.reasoning
                )
                topic_ids.append(topic.id)
                if created:
                    text = suggestion.topic
                    if suggestion.reasoning:
                        text = f"{suggestion.topic} — {suggestion.reasoning}"
                    new_topics.append((topic.id, text))
                if suggestion.confidence > best_confidence:
                    best_confidence = suggestion.confidence
                    primary_id = topic.id

            # Embed newly created topics (FR-004); failures leave null for backfill.
            embeddings_written, embeddings_deferred = await self._embed_new_topics(
                repo, new_topics
            )

            # Persist the question (linked to the primary topic).
            question = await repo.questions.create_question(
                query=refined_query,
                topic_id=primary_id,
                correlation_id=correlation_id,
                execution_id=execution_id,
                nodes_executed=nodes_executed,
                execution_metadata=structured_outputs or None,
            )

            # Persist synthesis as a WikiEntry only when a primary topic exists and
            # there is content (zero-topic runs persist the question only).
            wiki_id: Optional[UUID] = None
            if primary_id is not None and synthesis_content:
                related = [tid for tid in topic_ids if tid != primary_id]
                wiki = await repo.wiki.create_wiki_entry(
                    topic_id=primary_id,
                    content=synthesis_content,
                    question_id=question.id,
                    sources=[question.id],
                    related_topics=related or None,
                )
                wiki_id = wiki.id

            result = PersistedRunResult(
                question_id=question.id,
                topic_ids=topic_ids,
                primary_topic_id=primary_id,
                wiki_entry_id=wiki_id,
                suggestions_filtered=len(filtered),
                embeddings_written=embeddings_written,
                embeddings_deferred=embeddings_deferred,
            )

        await self._emit_event(result, filtered, correlation_id, execution_id)
        return result

    async def _embed_new_topics(
        self, repo: RepositoryFactory, new_topics: List[tuple[UUID, str]]
    ) -> tuple[int, int]:
        """Embed newly created topics, capped per run. Returns (written, deferred).

        On embedding failure the topics are left with null embeddings, recoverable via
        the backfill command (FR-004). Topics beyond the per-run cap are also deferred.
        """
        if self._embedding_service is None or not new_topics:
            return 0, len(new_topics)
        capped = new_topics[: self._config.max_embedding_calls_per_run]
        deferred = len(new_topics) - len(capped)
        try:
            results = await self._embedding_service.embed_batch(
                [text for _id, text in capped]
            )
        except EmbeddingError as exc:
            logger.warning(
                f"Embedding failed; {len(new_topics)} topics left null for backfill: {exc}"
            )
            return 0, len(new_topics)
        written = 0
        for (topic_id, _text), embedding in zip(capped, results):
            if await repo.topics.update_embedding(topic_id, embedding.vector):
                written += 1
            else:
                deferred += 1
        return written, deferred

    @staticmethod
    def _extract_synthesis(
        structured_outputs: Dict[str, Any], agent_outputs: Dict[str, Any]
    ) -> str:
        synthesis = structured_outputs.get("synthesis")
        if isinstance(synthesis, dict):
            content = synthesis.get("final_synthesis") or synthesis.get(
                "final_analysis"
            )
            if isinstance(content, str) and content.strip():
                return content
        fallback = agent_outputs.get("synthesis")
        return fallback if isinstance(fallback, str) else ""

    async def _emit_event(
        self,
        result: PersistedRunResult,
        filtered: List[TopicSuggestion],
        correlation_id: Optional[str],
        execution_id: Optional[str],
    ) -> None:
        """Emit a persistence-completed event with counts + gate telemetry (FR-008)."""
        try:
            await emit_agent_execution_completed(
                workflow_id=execution_id or correlation_id or "unknown",
                agent_name="knowledge_persistence",
                success=result.skipped_reason is None,
                output_context={},
                correlation_id=correlation_id,
                metadata={
                    "topics_persisted": len(result.topic_ids),
                    "wiki_entry_written": result.wiki_entry_id is not None,
                    "embeddings_written": result.embeddings_written,
                    "embeddings_deferred": result.embeddings_deferred,
                    "suggestions_filtered": result.suggestions_filtered,
                    # Confidence scores of filtered suggestions, to tune the gate (FR-011).
                    "filtered_confidences": [s.confidence for s in filtered],
                    "skipped_reason": result.skipped_reason,
                },
            )
        except Exception as exc:
            logger.debug(f"Knowledge persistence event emission failed: {exc}")
