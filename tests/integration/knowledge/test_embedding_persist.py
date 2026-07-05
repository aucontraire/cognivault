"""Integration tests for embed-on-persist + backfill (T021, FR-004/FR-005, SC-002).

Embeddings are MOCKED here (a fake embedder) — no real API calls — so the
outage→null→backfill recovery path is exercised deterministically. Requires the test
database (NullPool via the knowledge conftest). Uses distinct queries + cleanup.
"""

import uuid
from typing import List

import pytest
from sqlalchemy import text

from cognivault.cli import knowledge_commands
from cognivault.database.connection import get_database_session
from cognivault.database.session_factory import get_database_session_factory
from cognivault.knowledge.config import KnowledgePersistenceConfig
from cognivault.knowledge.embedding import (
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
)
from cognivault.knowledge.persistence import KnowledgePersistenceService
from cognivault.store.topic_manager import TopicManager


class _FakeEmbedder:
    """Stand-in for EmbeddingService: returns fixed-width vectors, or fails on demand."""

    def __init__(self, *, fail: bool = False, dimensions: int = 1536) -> None:
        self._fail = fail
        self._dims = dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        if self._fail:
            raise EmbeddingError("simulated outage")
        return EmbeddingResult(vector=[0.01] * self._dims, model="fake", input_tokens=1)

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        if self._fail:
            raise EmbeddingError("simulated outage")
        return [
            EmbeddingResult(vector=[0.01] * self._dims, model="fake", input_tokens=1)
            for _ in texts
        ]


def _service(embedder: object) -> KnowledgePersistenceService:
    return KnowledgePersistenceService(
        session_factory=get_database_session_factory(),
        topic_manager=TopicManager(),
        config=KnowledgePersistenceConfig(
            persistence_enabled=True, topic_min_confidence=0.0
        ),
        embedding_service=embedder,  # type: ignore[arg-type]
    )


async def _embedding_null_count(topic_ids: List[uuid.UUID]) -> int:
    async with get_database_session() as session:
        result = await session.execute(
            text(
                "SELECT count(*) FROM topics WHERE id = ANY(:ids) AND embedding IS NULL"
            ),
            {"ids": topic_ids},
        )
        return int(result.scalar() or 0)


async def _cleanup(correlation_id: str, topic_ids: List[uuid.UUID]) -> None:
    async with get_database_session() as session:
        await session.execute(
            text(
                "DELETE FROM wiki_entries WHERE question_id IN "
                "(SELECT id FROM questions WHERE correlation_id = :c)"
            ),
            {"c": correlation_id},
        )
        await session.execute(
            text("DELETE FROM questions WHERE correlation_id = :c"),
            {"c": correlation_id},
        )
        if topic_ids:
            await session.execute(
                text("DELETE FROM topics WHERE id = ANY(:ids)"), {"ids": topic_ids}
            )
        await session.commit()


@pytest.mark.asyncio
async def test_new_topics_embedded_on_persist() -> None:
    query = "quantum entanglement superposition decoherence tunneling"
    cid = "kp-emb-" + uuid.uuid4().hex
    result = await _service(_FakeEmbedder()).persist_run(
        refined_query=query,
        agent_outputs={"refiner": query},
        structured_outputs={"synthesis": {"final_synthesis": "..."}},
        correlation_id=cid,
        execution_id="e",
        nodes_executed=["refiner"],
    )
    try:
        assert result.skipped_reason is None
        assert result.embeddings_written > 0
        assert result.embeddings_deferred == 0
        # Every persisted topic now has a non-null embedding (SC-002).
        assert await _embedding_null_count(result.topic_ids) == 0
    finally:
        await _cleanup(cid, result.topic_ids)


@pytest.mark.asyncio
async def test_outage_leaves_null_then_backfill_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "photosynthesis chloroplast mitochondria ribosome enzyme"
    cid = "kp-emb-" + uuid.uuid4().hex
    # 1) Provider outage → topics persisted with NULL embeddings, counted as deferred.
    result = await _service(_FakeEmbedder(fail=True)).persist_run(
        refined_query=query,
        agent_outputs={"refiner": query},
        structured_outputs={"synthesis": {"final_synthesis": "..."}},
        correlation_id=cid,
        execution_id="e",
        nodes_executed=["refiner"],
    )
    try:
        assert result.embeddings_written == 0
        assert result.embeddings_deferred > 0
        assert await _embedding_null_count(result.topic_ids) == len(result.topic_ids)

        # 2) Backfill uses EmbeddingService.from_env internally → patch it to the fake.
        monkeypatch.setattr(EmbeddingService, "from_env", lambda: _FakeEmbedder())

        # --dry-run must write nothing: the global null count is unchanged.
        before = await _global_null_count()
        await knowledge_commands._backfill(limit=None, dry_run=True)
        assert await _global_null_count() == before

        # Real backfill recovers the previously-null topics.
        await knowledge_commands._backfill(limit=None, dry_run=False)
        assert await _embedding_null_count(result.topic_ids) == 0

        # Idempotent: a second run finds nothing left to do for these topics.
        await knowledge_commands._backfill(limit=None, dry_run=False)
        assert await _embedding_null_count(result.topic_ids) == 0
    finally:
        await _cleanup(cid, result.topic_ids)


async def _global_null_count() -> int:
    async with get_database_session() as session:
        result = await session.execute(
            text("SELECT count(*) FROM topics WHERE embedding IS NULL")
        )
        return int(result.scalar() or 0)
