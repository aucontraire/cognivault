"""
Wiki repository with knowledge synthesis and versioning support.
"""

from uuid import UUID

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cognivault.database.models import WikiEntry
from cognivault.observability import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class WikiRepository(BaseRepository[WikiEntry]):
    """
    Repository for WikiEntry model with knowledge versioning and synthesis.

    Provides wiki-specific operations including version management,
    knowledge evolution tracking, and multi-source synthesis.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, WikiEntry)

    async def create_wiki_entry(
        self,
        topic_id: UUID,
        content: str,
        question_id: UUID | None = None,
        version: int = 1,
        supersedes: UUID | None = None,
        sources: list[UUID] | None = None,
        related_topics: list[UUID] | None = None,
    ) -> WikiEntry:
        """
        Create a new wiki entry with versioning and source tracking.

        Args:
            topic_id: Associated topic UUID
            content: Wiki entry content
            question_id: Source question UUID
            version: Version number
            supersedes: Previous wiki entry UUID this supersedes
            sources: List of contributing question UUIDs
            related_topics: List of related topic UUIDs

        Returns:
            Created wiki entry instance
        """
        return await self.create(
            topic_id=topic_id,
            content=content,
            question_id=question_id,
            version=version,
            supersedes=supersedes,
            sources=sources,
            related_topics=related_topics,
        )

    async def get_latest_for_topic(self, topic_id: UUID) -> WikiEntry | None:
        """
        Get the latest version of wiki entry for a topic.

        Args:
            topic_id: Topic UUID

        Returns:
            Latest wiki entry or None if not found
        """
        try:
            stmt = (
                select(WikiEntry)
                .where(WikiEntry.topic_id == topic_id)
                .order_by(desc(WikiEntry.version))
                .limit(1)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get latest wiki entry for topic {topic_id}: {e}")
            raise

    async def get_all_versions_for_topic(
        self, topic_id: UUID, limit: int | None = None
    ) -> list[WikiEntry]:
        """
        Get all versions of wiki entries for a topic.

        Args:
            topic_id: Topic UUID
            limit: Maximum number of versions to return

        Returns:
            List of wiki entries ordered by version (newest first)
        """
        try:
            stmt = (
                select(WikiEntry)
                .where(WikiEntry.topic_id == topic_id)
                .order_by(desc(WikiEntry.version))
            )

            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get wiki versions for topic {topic_id}: {e}")
            raise

    async def get_by_version(self, topic_id: UUID, version: int) -> WikiEntry | None:
        """
        Get specific version of wiki entry for a topic.

        Args:
            topic_id: Topic UUID
            version: Version number

        Returns:
            Wiki entry or None if not found
        """
        try:
            stmt = select(WikiEntry).where(
                and_(WikiEntry.topic_id == topic_id, WikiEntry.version == version)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(
                f"Failed to get wiki entry version {version} for topic {topic_id}: {e}"
            )
            raise

    async def get_with_relationships(self, wiki_id: UUID) -> WikiEntry | None:
        """
        Get wiki entry with topic and source question loaded.

        Args:
            wiki_id: Wiki entry UUID

        Returns:
            Wiki entry with relationships loaded
        """
        try:
            stmt = (
                select(WikiEntry)
                .options(
                    selectinload(WikiEntry.topic),
                    selectinload(WikiEntry.source_question),
                )
                .where(WikiEntry.id == wiki_id)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get wiki entry with relationships {wiki_id}: {e}")
            raise

    async def get_by_source_question(self, question_id: UUID) -> list[WikiEntry]:
        """
        Get wiki entries that originated from a specific question.

        Args:
            question_id: Source question UUID

        Returns:
            List of wiki entries derived from the question
        """
        try:
            stmt = (
                select(WikiEntry)
                .where(WikiEntry.question_id == question_id)
                .order_by(desc(WikiEntry.created_at))
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                f"Failed to get wiki entries by source question {question_id}: {e}"
            )
            raise

    async def create_new_version(
        self,
        current_entry_id: UUID,
        new_content: str,
        additional_sources: list[UUID] | None = None,
        additional_related_topics: list[UUID] | None = None,
    ) -> WikiEntry | None:
        """
        Create a new version of an existing wiki entry.

        Args:
            current_entry_id: Current wiki entry UUID to supersede
            new_content: Updated content
            additional_sources: New sources to add
            additional_related_topics: New related topics to add

        Returns:
            New wiki entry version or None if current entry not found
        """
        try:
            # Get current entry
            current_entry = await self.get_by_id(current_entry_id)
            if not current_entry:
                return None

            # Merge sources and related topics
            sources = list(current_entry.sources or [])
            if additional_sources:
                sources.extend(additional_sources)
                sources = list(set(sources))  # Remove duplicates

            related_topics = list(current_entry.related_topics or [])
            if additional_related_topics:
                related_topics.extend(additional_related_topics)
                related_topics = list(set(related_topics))  # Remove duplicates

            # Create new version - cast to avoid Column type issues
            from typing import cast, Any

            new_version = await self.create_wiki_entry(
                topic_id=current_entry.topic_id,
                content=new_content,
                question_id=cast(UUID | None, current_entry.question_id),
                version=cast(int, current_entry.version) + 1,
                supersedes=current_entry_id,
                sources=sources,
                related_topics=related_topics,
            )

            logger.debug(
                f"Created new wiki version {new_version.version} "
                f"for topic {current_entry.topic_id}"
            )
            return new_version

        except Exception as e:
            logger.error(
                f"Failed to create new wiki version for {current_entry_id}: {e}"
            )
            raise

    async def get_evolution_chain(self, wiki_id: UUID) -> list[WikiEntry]:
        """
        Get the complete evolution chain for a wiki entry.

        Args:
            wiki_id: Wiki entry UUID (can be any version in the chain)

        Returns:
            List of wiki entries in evolution order (oldest to newest)
        """
        try:
            # First, find the root of the chain
            current_entry = await self.get_by_id(wiki_id)
            if not current_entry:
                return []

            # Traverse backwards to find the root
            from typing import cast

            root_entry = current_entry
            while root_entry.supersedes:
                supersedes_id = cast(UUID, root_entry.supersedes)
                parent_entry = await self.get_by_id(supersedes_id)
                if not parent_entry:
                    break
                root_entry = parent_entry

            if not root_entry:
                return [current_entry]

            # Now traverse forward to build the chain
            chain = [root_entry]

            # Find all entries that supersede entries in our chain
            stmt = (
                select(WikiEntry)
                .where(WikiEntry.topic_id == root_entry.topic_id)
                .order_by(WikiEntry.version)
            )
            result = await self.session.execute(stmt)
            all_versions = list(result.scalars().all())

            # Build the evolution chain
            for version in all_versions:
                if version.id != root_entry.id and version.supersedes:
                    # Check if this version supersedes any version in our chain
                    if any(v.id == version.supersedes for v in chain):
                        chain.append(version)

            return chain

        except Exception as e:
            logger.error(f"Failed to get evolution chain for wiki {wiki_id}: {e}")
            raise

    async def search_content(
        self, search_query: str, limit: int = 20, latest_only: bool = True
    ) -> list[WikiEntry]:
        """
        Search wiki entries by content.

        Args:
            search_query: Search terms
            limit: Maximum number of results
            latest_only: If True, only return latest versions

        Returns:
            List of matching wiki entries
        """
        try:
            stmt = (
                select(WikiEntry)
                .where(WikiEntry.content.ilike(f"%{search_query}%"))
                .order_by(desc(WikiEntry.created_at))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            entries = list(result.scalars().all())

            if latest_only:
                # Filter to only latest versions per topic
                topic_latest: dict[UUID, WikiEntry] = {}
                for entry in entries:
                    from typing import cast

                    entry_topic_id = cast(UUID, entry.topic_id)
                    entry_version = cast(int, entry.version)
                    if (
                        entry_topic_id not in topic_latest
                        or entry_version > topic_latest[entry_topic_id].version
                    ):
                        topic_latest[entry_topic_id] = entry

                entries = list(topic_latest.values())

            return entries

        except Exception as e:
            logger.error(f"Failed to search wiki content for {search_query}: {e}")
            raise

    async def get_multi_topic_entries(
        self, limit: int | None = None
    ) -> list[WikiEntry]:
        """
        Get wiki entries that reference multiple topics.

        Args:
            limit: Maximum number of results

        Returns:
            List of wiki entries with multiple topic relationships
        """
        try:
            # Using array length function to find entries with multiple related topics
            from sqlalchemy import func

            stmt = (
                select(WikiEntry)
                .where(
                    and_(
                        WikiEntry.related_topics.is_not(None),
                        func.array_length(WikiEntry.related_topics, 1) > 0,
                    )
                )
                .order_by(desc(WikiEntry.created_at))
            )

            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            entries = list(result.scalars().all())

            # Filter to entries that actually have multiple topics
            # (including the main topic)
            # related_topics contains additional topics beyond the main topic_id
            multi_topic_entries = [
                entry
                for entry in entries
                if entry.related_topics and len(entry.related_topics) >= 1
            ]

            return multi_topic_entries

        except Exception as e:
            logger.error(f"Failed to get multi-topic wiki entries: {e}")
            raise
