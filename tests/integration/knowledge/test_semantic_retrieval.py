"""Fixture eval: semantic retrieval surfaces synonym-phrased content (T027, SC-003,
US3 scenario 1).

Seeds several distinct topics with REAL embeddings, then queries the Historian's
SemanticSearch with synonym phrasings (aiming for minimal keyword overlap with the topic
name/description), and asserts each query's target topic appears in the top-5 for ≥80% of
queries. With many seeded topics, "top-5" is discriminating (a target must out-rank other
concepts), so the hit rate is a meaningful signal — a low rate would justify the
document-level-embedding follow-up. Skipped without a real key (keyless CI stays green).
"""

import os
import uuid
from typing import List, Tuple

import pytest
from sqlalchemy import text

from cognivault.agents.historian.search import SemanticSearch
from cognivault.database.connection import get_database_session
from cognivault.database.session_factory import get_database_session_factory
from cognivault.knowledge.embedding import EmbeddingService

_REAL_KEY = os.getenv("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (_REAL_KEY and _REAL_KEY.startswith("sk-")),
    reason="requires a real OPENAI_API_KEY (sk-...) for real embeddings",
)

# (concept name, description, [synonym queries]) — queries avoid the concept name and
# minimize overlap with the description, so retrieval is semantic rather than lexical.
_CONCEPTS: List[Tuple[str, str, List[str]]] = [
    (
        "backpropagation",
        "the algorithm that trains multilayer neural networks by propagating errors backward",
        [
            "how a model adjusts its weights after prediction mistakes",
            "iteratively tuning connection strengths in a deep learning system",
        ],
    ),
    (
        "photosynthesis",
        "the process where plants convert sunlight into chemical energy",
        [
            "how green leaves turn solar radiation into stored fuel",
            "flora producing food from daylight inside chloroplasts",
        ],
    ),
    (
        "blockchain consensus",
        "distributed agreement on ledger state without a central authority",
        [
            "how decentralized nodes agree on transaction ordering",
            "establishing trust between untrusted peers without a middleman",
        ],
    ),
    (
        "mitochondria",
        "the organelle that generates ATP through cellular respiration",
        [
            "where a living cell produces its energy currency",
            "the powerhouse structure driving aerobic metabolism",
        ],
    ),
    (
        "supply and demand",
        "the market mechanism where price balances scarcity against buyer interest",
        [
            "how prices settle when purchasers and vendors interact",
            "what drives cost when a scarce good is widely wanted",
        ],
    ),
    (
        "plate tectonics",
        "movement of Earth's lithospheric slabs causing quakes and mountain ranges",
        [
            "why continents drift and tremors shake the ground",
            "the slow shifting of the planet's crustal segments",
        ],
    ),
    (
        "immune response",
        "the body defending against pathogens via antibodies and white cells",
        [
            "how an organism fights off an infection",
            "biological defenses that neutralize invading microbes",
        ],
    ),
    (
        "compound interest",
        "growth where earned interest itself earns further interest over time",
        [
            "how savings snowball when returns are reinvested",
            "money accelerating because gains generate more gains",
        ],
    ),
]


@pytest.mark.asyncio
async def test_semantic_retrieval_surfaces_synonym_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _REAL_KEY is not None
    real_service = EmbeddingService(
        api_key=_REAL_KEY, model="text-embedding-3-small", dimensions=1536
    )
    # SemanticSearch calls EmbeddingService.from_env internally → give it the real key.
    monkeypatch.setattr(EmbeddingService, "from_env", lambda: real_service)

    marker = uuid.uuid4().hex[:8]
    factory = get_database_session_factory()
    await factory.initialize()
    created: List[uuid.UUID] = []
    target_ids: dict[str, uuid.UUID] = {}  # query -> expected topic id
    try:
        async with factory.get_repository_factory() as repo:
            for name, description, queries in _CONCEPTS:
                topic, _ = await repo.topics.get_or_create_by_canonical_name(
                    f"{name} {marker}", description=description
                )
                embedding = await real_service.embed(f"{name} — {description}")
                await repo.topics.update_embedding(topic.id, embedding.vector)
                created.append(topic.id)
                for query in queries:
                    target_ids[query] = topic.id

        hits = 0
        misses: List[str] = []
        for query, expected_id in target_ids.items():
            results = await SemanticSearch(similarity_threshold=0.0).search(
                query, limit=5
            )
            top_ids = [r.metadata.get("topic_id") for r in results]
            if str(expected_id) in top_ids:
                hits += 1
            else:
                misses.append(query)

        total = len(target_ids)
        hit_rate = hits / total
        # Emit the concrete number so the checkpoint can report the empirical SC-003 result.
        print(f"\nSC-003 semantic retrieval hit rate: {hits}/{total} = {hit_rate:.0%}")
        if misses:
            print("  misses:", misses)
        assert hit_rate >= 0.8, (
            f"semantic retrieval top-5 hit rate {hits}/{total} < 80%; misses: {misses}"
        )
    finally:
        async with get_database_session() as session:
            if created:
                await session.execute(
                    text("DELETE FROM topics WHERE id = ANY(:ids)"), {"ids": created}
                )
                await session.commit()
