"""Integration test: semantic similarity ranks related topics above unrelated (T022,
US2 scenario 3).

Uses REAL embeddings (a few cents) so the ranking assertion is meaningful, and is
SKIPPED when OPENAI_API_KEY is absent — keeping the suite green in keyless environments
(the same DB-optional / provider-optional discipline used everywhere else). Requires the
test database (NullPool via the knowledge conftest).
"""

import os
import uuid
from typing import List

import pytest
from sqlalchemy import text

from cognivault.database.connection import get_database_session
from cognivault.database.session_factory import get_database_session_factory
from cognivault.knowledge.embedding import EmbeddingService

# Capture the real key at import time — per-test conftest fixtures overwrite
# OPENAI_API_KEY with a dummy value, so from_env() would get the dummy. Only run when a
# genuine key (sk-...) is present, keeping keyless/CI environments green.
_REAL_KEY = os.getenv("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (_REAL_KEY and _REAL_KEY.startswith("sk-")),
    reason="requires a real OPENAI_API_KEY (sk-...) for real embeddings",
)


@pytest.mark.asyncio
async def test_similarity_ranks_related_above_unrelated() -> None:
    assert _REAL_KEY is not None  # guaranteed by skipif
    marker = uuid.uuid4().hex[:8]
    service = EmbeddingService(
        api_key=_REAL_KEY, model="text-embedding-3-small", dimensions=1536
    )
    concepts = {
        f"backpropagation {marker}": "gradient-based training of neural networks",
        f"neural networks {marker}": "layered artificial neurons for learning",
        f"photosynthesis {marker}": "plants converting sunlight into chemical energy",
    }

    factory = get_database_session_factory()
    await factory.initialize()
    created_ids: List[uuid.UUID] = []
    try:
        async with factory.get_repository_factory() as repo:
            for name, description in concepts.items():
                topic, _ = await repo.topics.get_or_create_by_canonical_name(
                    name, description=description
                )
                embedding = await service.embed(f"{name} — {description}")
                await repo.topics.update_embedding(topic.id, embedding.vector)
                created_ids.append(topic.id)

        # Query with a concept semantically close to the ML topics, far from biology.
        query = await service.embed(
            "training deep neural networks with gradient descent"
        )
        async with factory.get_repository_factory() as repo:
            neighbors = await repo.topics.find_similar_by_embedding(
                query.vector, limit=10, similarity_threshold=0.0
            )

        ranked = [t.name for t, _sim in neighbors if marker in t.name]
        assert ranked, "seeded topics should appear among the neighbors"

        def _index(substr: str) -> int:
            return next(
                (i for i, name in enumerate(ranked) if substr in name), len(ranked)
            )

        ml_rank = min(_index("backpropagation"), _index("neural"))
        bio_rank = _index("photosynthesis")
        assert ml_rank < bio_rank, (
            f"related topics should outrank photosynthesis: {ranked}"
        )
    finally:
        async with get_database_session() as session:
            if created_ids:
                await session.execute(
                    text("DELETE FROM topics WHERE id = ANY(:ids)"),
                    {"ids": created_ids},
                )
                await session.commit()
