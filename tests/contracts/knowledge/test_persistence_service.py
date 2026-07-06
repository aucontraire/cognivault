"""Contract tests for KnowledgePersistenceService (T014, FR-003/FR-013).

These verify the never-raises + skipped_reason guarantees WITHOUT a database, using
fakes — so they run in any environment (no Docker/Postgres required).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognivault.knowledge.config import KnowledgePersistenceConfig
from cognivault.knowledge.persistence import (
    KnowledgePersistenceService,
    PersistedRunResult,
)


def _service(
    session_factory: object, topic_manager: object, **config: object
) -> KnowledgePersistenceService:
    return KnowledgePersistenceService(
        session_factory=session_factory,  # type: ignore[arg-type]
        topic_manager=topic_manager,  # type: ignore[arg-type]
        config=KnowledgePersistenceConfig(**config),
    )


async def _run(svc: KnowledgePersistenceService) -> PersistedRunResult:
    return await svc.persist_run(
        refined_query="q",
        agent_outputs={},
        structured_outputs={},
        correlation_id="corr-1",
        execution_id="exec-1",
        nodes_executed=["refiner"],
    )


@pytest.mark.asyncio
async def test_disabled_skips_without_touching_database() -> None:
    session_factory = MagicMock()
    session_factory.initialize = AsyncMock()
    svc = _service(session_factory, MagicMock(), persistence_enabled=False)
    result = await _run(svc)
    assert result.skipped_reason == "disabled"
    session_factory.initialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_database_skips_gracefully() -> None:
    session_factory = MagicMock()
    session_factory.initialize = AsyncMock(side_effect=RuntimeError("db unreachable"))
    svc = _service(session_factory, MagicMock(), persistence_enabled=True)
    result = await _run(svc)
    assert result.skipped_reason == "no_database"


@pytest.mark.asyncio
async def test_internal_error_never_raises() -> None:
    session_factory = MagicMock()
    session_factory.initialize = AsyncMock(return_value=None)
    topic_manager = MagicMock()
    topic_manager.analyze_and_suggest_topics = AsyncMock(side_effect=ValueError("boom"))
    svc = _service(session_factory, topic_manager, persistence_enabled=True)
    result = await _run(svc)  # must NOT raise
    assert result.skipped_reason == "error"
