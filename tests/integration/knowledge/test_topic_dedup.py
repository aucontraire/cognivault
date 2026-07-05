"""Integration tests for concurrency-safe topic dedup (T008, FR-002/FR-012, SC-005).

Requires a configured database. The knowledge-package conftest forces NullPool, so each
session gets its own connection — enabling a real 5-concurrent-writer test. Uses unique
per-test names so it never touches pre-existing rows, and cleans up.
"""

import asyncio
import uuid

import pytest
from sqlalchemy import text

from cognivault.database.connection import get_database_session
from cognivault.database.repositories.topic_repository import (
    TopicRepository,
    canonicalize,
)


async def _count_by_canonical(canonical: str) -> int:
    async with get_database_session() as session:
        result = await session.execute(
            text("SELECT count(*) FROM topics WHERE canonical_name = :cn"),
            {"cn": canonical},
        )
        return int(result.scalar() or 0)


async def _delete_by_canonical(canonical: str) -> None:
    async with get_database_session() as session:
        await session.execute(
            text("DELETE FROM topics WHERE canonical_name = :cn"), {"cn": canonical}
        )
        await session.commit()


@pytest.mark.asyncio
async def test_get_or_create_dedup_is_case_and_whitespace_insensitive() -> None:
    name = f"Dedup Seq {uuid.uuid4().hex}"
    canonical = canonicalize(name)
    try:
        async with get_database_session() as session:
            repo = TopicRepository(session)
            topic1, created1 = await repo.get_or_create_by_canonical_name(name)
            # A case/whitespace variant maps to the same canonical name → same row.
            topic2, created2 = await repo.get_or_create_by_canonical_name(
                f"   {name.upper()}   "
            )
            assert created1 is True
            assert created2 is False
            assert topic1.id == topic2.id
    finally:
        await _delete_by_canonical(canonical)


@pytest.mark.asyncio
async def test_concurrent_writers_resolve_to_one_row() -> None:
    """SC-005: >=5 concurrent writers of the same canonical name -> exactly 1 row."""
    name = f"Dedup Conc {uuid.uuid4().hex}"
    canonical = canonicalize(name)

    async def writer() -> uuid.UUID:
        async with get_database_session() as session:
            repo = TopicRepository(session)
            topic, _ = await repo.get_or_create_by_canonical_name(name)
            return topic.id

    try:
        results = await asyncio.gather(
            *[writer() for _ in range(5)], return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"writers raised: {errors}"
        # Exactly one topic row (DB-level uniqueness held under real concurrency).
        assert await _count_by_canonical(canonical) == 1
        # Every writer resolved to that single row.
        assert len({r for r in results if isinstance(r, uuid.UUID)}) == 1
    finally:
        await _delete_by_canonical(canonical)
