"""Integration test: graceful degradation when the database is unavailable
(T016, US1 scenario 3, SC-004 error clause).

Uses a session factory whose initialization fails, proving that persist_run degrades
to a skip (no exception) — so the surrounding pipeline run is unaffected. DB-independent
by construction (no real connection needed).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognivault.knowledge.config import KnowledgePersistenceConfig
from cognivault.knowledge.persistence import KnowledgePersistenceService


@pytest.mark.asyncio
async def test_database_unavailable_degrades_without_error() -> None:
    session_factory = MagicMock()
    session_factory.initialize = AsyncMock(side_effect=OSError("connection refused"))
    svc = KnowledgePersistenceService(
        session_factory=session_factory,
        topic_manager=MagicMock(),
        config=KnowledgePersistenceConfig(persistence_enabled=True),
    )

    result = await svc.persist_run(
        refined_query="anything",
        agent_outputs={},
        structured_outputs={},
        correlation_id="corr-x",
        execution_id="exec-x",
        nodes_executed=["refiner"],
    )

    # Skipped gracefully; no exception propagated → the run adds 0 user-visible errors.
    assert result.skipped_reason == "no_database"
    assert result.question_id is None
