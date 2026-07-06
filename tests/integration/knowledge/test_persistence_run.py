"""Integration test: pipeline run persists question + topic + wiki, reuses topics,
and leaves semantic_links untouched (T015, US1 scenarios 1-2, SC-001, FR-009).

Requires a configured database. Uses a 0.0 confidence gate so keyword-derived topic
suggestions qualify (exercising the full topic + wiki path deterministically), unique
correlation ids, and cleans up the questions/wiki it creates.
"""

import uuid

import pytest
from sqlalchemy import text

from cognivault.database.connection import get_database_session
from cognivault.database.session_factory import get_database_session_factory
from cognivault.knowledge.config import KnowledgePersistenceConfig
from cognivault.knowledge.persistence import KnowledgePersistenceService
from cognivault.store.topic_manager import TopicManager


def _service() -> KnowledgePersistenceService:
    return KnowledgePersistenceService(
        session_factory=get_database_session_factory(),
        topic_manager=TopicManager(),
        config=KnowledgePersistenceConfig(
            persistence_enabled=True, topic_min_confidence=0.0
        ),
    )


async def _count_semantic_links() -> int:
    async with get_database_session() as session:
        return int(
            (
                await session.execute(text("SELECT count(*) FROM semantic_links"))
            ).scalar()
            or 0
        )


async def _cleanup(correlation_id: str) -> None:
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
        await session.commit()


@pytest.mark.asyncio
async def test_run_persists_question_topic_wiki_and_reuses_topics() -> None:
    svc = _service()
    query = "Tradeoffs of event-driven architecture and message queues"
    outputs = {
        "refiner": query,
        "synthesis": "Event-driven systems decouple producers.",
    }
    structured = {
        "synthesis": {
            "final_synthesis": "Event-driven architecture decouples components, "
            "improving scalability at the cost of operational complexity."
        }
    }
    cid1 = "kp-int-" + uuid.uuid4().hex
    cid2 = "kp-int-" + uuid.uuid4().hex

    links_before = await _count_semantic_links()
    try:
        r1 = await svc.persist_run(
            refined_query=query,
            agent_outputs=outputs,
            structured_outputs=structured,
            correlation_id=cid1,
            execution_id="e1",
            nodes_executed=["refiner", "synthesis"],
        )
        # A related run (same query) must reuse the shared topics (SC-005 dedup).
        r2 = await svc.persist_run(
            refined_query=query,
            agent_outputs=outputs,
            structured_outputs=structured,
            correlation_id=cid2,
            execution_id="e2",
            nodes_executed=["refiner", "synthesis"],
        )

        assert r1.skipped_reason is None and r2.skipped_reason is None
        assert r1.question_id is not None and r1.primary_topic_id is not None
        assert r1.wiki_entry_id is not None  # SC-001: synthesis persisted + linked
        # Topic reuse: identical content resolves to the same primary topic row.
        assert r1.primary_topic_id == r2.primary_topic_id

        async with get_database_session() as session:
            wiki_count = (
                await session.execute(
                    text("SELECT count(*) FROM wiki_entries WHERE question_id = :q"),
                    {"q": r1.question_id},
                )
            ).scalar()
            assert wiki_count == 1

        # FR-009: this feature does not write semantic_links.
        assert await _count_semantic_links() == links_before
    finally:
        await _cleanup(cid1)
        await _cleanup(cid2)
