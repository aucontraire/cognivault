"""Fixture eval: semantic retrieval surfaces synonym-phrased content (T027, SC-003,
US3 scenario 1).

Seeds topics with REAL embeddings, then queries the Historian's SemanticSearch with
synonym phrasings that share no keywords with the topic names, and asserts the seeded
content is retrieved in the top-5 for ≥80% of queries. A focused version of SC-003's
≥20-query benchmark. Skipped without a real key so keyless environments stay green.
"""

import os
import uuid
from typing import List

import pytest
from sqlalchemy import text

from cognivault.agents.historian.search import SemanticSearch
from cognivault.database.connection import get_database_session
from cognivault.database.session_factory import get_database_session_factory
from cognivault.knowledge.embedding import EmbeddingService

# Capture the real key at import (before the per-test dummy-key override).
_REAL_KEY = os.getenv("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (_REAL_KEY and _REAL_KEY.startswith("sk-")),
    reason="requires a real OPENAI_API_KEY (sk-...) for real embeddings",
)

# Synonym phrasings for "backpropagation" that share no keyword with the topic name.
_SYNONYM_QUERIES = [
    "how do neural networks learn from their mistakes",
    "adjusting model weights using gradient descent during training",
    "the method for updating layer parameters when training deep models",
]


@pytest.mark.asyncio
async def test_semantic_retrieval_surfaces_synonym_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _REAL_KEY is not None
    real_service = EmbeddingService(
        api_key=_REAL_KEY, model="text-embedding-3-small", dimensions=1536
    )
    # Make SemanticSearch (which calls from_env internally) use the real key.
    monkeypatch.setattr(EmbeddingService, "from_env", lambda: real_service)

    marker = uuid.uuid4().hex[:8]
    seeds = {
        f"backpropagation {marker}": "trains neural networks by propagating errors backward through layers",
        f"photosynthesis {marker}": "how plants convert sunlight into chemical energy",
    }

    factory = get_database_session_factory()
    await factory.initialize()
    created: List[uuid.UUID] = []
    seeded_id = None
    try:
        async with factory.get_repository_factory() as repo:
            for name, description in seeds.items():
                topic, _ = await repo.topics.get_or_create_by_canonical_name(
                    name, description=description
                )
                embedding = await real_service.embed(f"{name} — {description}")
                await repo.topics.update_embedding(topic.id, embedding.vector)
                created.append(topic.id)
                if "backpropagation" in name:
                    seeded_id = topic.id

        hits = 0
        for query in _SYNONYM_QUERIES:
            results = await SemanticSearch(similarity_threshold=0.0).search(
                query, limit=5
            )
            top_topic_ids = [r.metadata.get("topic_id") for r in results]
            if str(seeded_id) in top_topic_ids:
                hits += 1

        hit_rate = hits / len(_SYNONYM_QUERIES)
        assert hit_rate >= 0.8, (
            f"semantic retrieval surfaced the seeded topic in only {hits}/"
            f"{len(_SYNONYM_QUERIES)} synonym queries (need >=80%)"
        )
    finally:
        async with get_database_session() as session:
            if created:
                await session.execute(
                    text("DELETE FROM topics WHERE id = ANY(:ids)"), {"ids": created}
                )
                await session.commit()
