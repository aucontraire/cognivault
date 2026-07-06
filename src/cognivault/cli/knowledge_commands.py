"""CLI commands for knowledge-base maintenance (FR-005).

`cognivault knowledge backfill-embeddings` generates embeddings for topics whose
embedding is NULL — the recovery path for topics persisted while the embedding
provider was unavailable, or beyond a run's per-run cap. Idempotent: each run only
touches null-embedding rows, so re-running is safe and converges. `--dry-run` reports
the count and writes nothing (and makes no provider calls).

Unlike the pipeline write path, this command is user-invoked, so it surfaces errors
rather than degrading silently.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import typer

from cognivault.database.session_factory import get_database_session_factory
from cognivault.knowledge.embedding import EmbeddingError, EmbeddingService
from cognivault.observability import get_logger

logger = get_logger(__name__)

knowledge_app = typer.Typer(help="Knowledge-base maintenance commands")


@knowledge_app.command("backfill-embeddings")
def backfill_embeddings(
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Maximum number of topics to process"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report the count only; generate and write nothing"
    ),
) -> None:
    """Embed topics whose embedding is NULL (idempotent; --dry-run writes nothing)."""
    asyncio.run(_backfill(limit=limit, dry_run=dry_run))


async def _backfill(*, limit: Optional[int], dry_run: bool) -> None:
    factory = get_database_session_factory()
    await factory.initialize()

    async with factory.get_repository_factory() as repo:
        topics = await repo.topics.get_topics_without_embeddings(limit)
        typer.echo(f"Topics with null embeddings: {len(topics)}")

        if dry_run:
            typer.echo("--dry-run: no embeddings generated, no writes performed.")
            return
        if not topics:
            return

        service = EmbeddingService.from_env()  # raises if OPENAI_API_KEY is unset
        written = 0
        failed = 0
        for topic in topics:
            text = topic.name
            if topic.description:
                text = f"{topic.name} — {topic.description}"
            try:
                result = await service.embed(text)
            except EmbeddingError as exc:
                typer.echo(f"  ! embed failed for '{topic.name}': {exc}")
                failed += 1
                continue
            if await repo.topics.update_embedding(topic.id, result.vector):
                written += 1

        typer.echo(f"Backfilled {written} embeddings ({failed} failed).")
