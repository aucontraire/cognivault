"""Performance: knowledge-persistence step overhead (T031, SC-004).

SC-004 budgets persistence + embedding at ≤10% of the median end-to-end run time. The
4-agent pipeline is LLM-bound (multiple seconds per run), so the check that matters is
that the persistence STEP itself stays small in absolute terms. This measures the median
persist_run latency over 20 runs against a real database (embeddings mocked, so the number
reflects the DB write path, not network variance) and asserts it stays under a budget that
is comfortably <10% of any realistic seconds-long pipeline run. The measured median is
printed for the checkpoint report.
"""

import statistics
import time
import uuid
from typing import List

import pytest
from sqlalchemy import text

from cognivault.database.connection import get_database_session
from cognivault.database.session_factory import get_database_session_factory
from cognivault.knowledge.config import KnowledgePersistenceConfig
from cognivault.knowledge.embedding import EmbeddingResult
from cognivault.knowledge.persistence import KnowledgePersistenceService
from cognivault.store.topic_manager import TopicManager

# Budget for the persistence step. Sized against the CONTAINERIZED test DB (5440), where
# round-trips are ~2x a native Postgres (~324ms median / ~414ms p95 on Docker vs ~137ms
# native). 600ms keeps healthy margin over that while staying well under 10% of the
# multi-second LLM pipeline runtime (SC-004) — a persist step ballooning past 600ms is a
# real regression worth catching.
OVERHEAD_BUDGET_MS = 600.0
SAMPLE_RUNS = 20


class _FakeEmbedder:
    async def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[0.01] * 1536, model="fake", input_tokens=1)

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        return [
            EmbeddingResult(vector=[0.01] * 1536, model="fake", input_tokens=1)
            for _ in texts
        ]


@pytest.mark.asyncio
async def test_persistence_step_overhead_within_budget() -> None:
    svc = KnowledgePersistenceService(
        session_factory=get_database_session_factory(),
        topic_manager=TopicManager(),
        config=KnowledgePersistenceConfig(
            persistence_enabled=True, topic_min_confidence=0.0
        ),
        embedding_service=_FakeEmbedder(),  # type: ignore[arg-type]
    )

    correlations: List[str] = []
    topic_ids: List[uuid.UUID] = []
    durations_ms: List[float] = []
    try:
        for i in range(SAMPLE_RUNS):
            cid = f"perf-{uuid.uuid4().hex}"
            correlations.append(cid)
            query = f"performance sample query number {i} {uuid.uuid4().hex[:6]}"
            start = time.perf_counter()
            result = await svc.persist_run(
                refined_query=query,
                agent_outputs={"refiner": query},
                structured_outputs={"synthesis": {"final_synthesis": "x"}},
                correlation_id=cid,
                execution_id="e",
                nodes_executed=["refiner"],
            )
            durations_ms.append((time.perf_counter() - start) * 1000)
            topic_ids.extend(result.topic_ids)

        median_ms = statistics.median(durations_ms)
        p95_ms = sorted(durations_ms)[int(len(durations_ms) * 0.95) - 1]
        print(
            f"\nPersistence step latency over {SAMPLE_RUNS} runs: "
            f"median={median_ms:.1f}ms p95={p95_ms:.1f}ms "
            f"(budget {OVERHEAD_BUDGET_MS:.0f}ms)"
        )
        assert (
            median_ms < OVERHEAD_BUDGET_MS
        ), f"persistence step median {median_ms:.1f}ms exceeds budget"
    finally:
        async with get_database_session() as session:
            for cid in correlations:
                await session.execute(
                    text(
                        "DELETE FROM wiki_entries WHERE question_id IN "
                        "(SELECT id FROM questions WHERE correlation_id = :c)"
                    ),
                    {"c": cid},
                )
                await session.execute(
                    text("DELETE FROM questions WHERE correlation_id = :c"), {"c": cid}
                )
            if topic_ids:
                await session.execute(
                    text("DELETE FROM topics WHERE id = ANY(:ids)"), {"ids": topic_ids}
                )
            await session.commit()
