"""
Historian document repository with hybrid search capabilities.

Provides document storage, full-text search, content deduplication,
and analytics for the historian agent search system.
"""

import hashlib
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import TSVECTOR

from cognivault.database.models import HistorianDocument
from cognivault.observability import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class HistorianDocumentRepository(BaseRepository[HistorianDocument]):
    """
    Repository for HistorianDocument model with hybrid search capabilities.

    Provides document-specific operations including full-text search,
    content deduplication via hashing, and search analytics integration.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, HistorianDocument)

    async def create_document(
        self,
        title: str,
        content: str,
        source_path: str | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> HistorianDocument:
        """
        Create a new document with automatic deduplication.

        Args:
            title: Document title (max 500 chars for search validation)
            content: Full document content
            source_path: Original file path or URL
            document_metadata: Flexible document metadata

        Returns:
            Created document instance

        Raises:
            Exception: If document with same content hash already exists
        """
        # Truncate title if too long for search validation
        if len(title) > 500:
            logger.warning(f"Title truncated from {len(title)} to 500 characters")
            title = title[:497] + "..."

        # Generate content hash for deduplication
        content_hash = self._generate_content_hash(content)

        # Calculate content statistics
        word_count = len(content.split())
        char_count = len(content)

        return await self.create(
            title=title,
            content=content,
            source_path=source_path,
            content_hash=content_hash,
            word_count=word_count,
            char_count=char_count,
            document_metadata=document_metadata or {},
        )

    async def find_by_content_hash(self, content_hash: str) -> HistorianDocument | None:
        """
        Find document by content hash for deduplication.

        Args:
            content_hash: SHA-256 hash of content

        Returns:
            Existing document or None if not found
        """
        try:
            stmt = select(HistorianDocument).where(
                HistorianDocument.content_hash == content_hash
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to find document by content hash {content_hash}: {e}")
            raise

    async def get_or_create_document(
        self,
        title: str,
        content: str,
        source_path: str | None = None,
        document_metadata: dict[str, Any] | None = None,
    ) -> tuple[HistorianDocument, bool]:
        """
        Get existing document or create new one (deduplication).

        Args:
            title: Document title
            content: Full document content
            source_path: Original file path or URL
            document_metadata: Flexible document metadata

        Returns:
            Tuple of (document, created) where created is True if new document
        """
        try:
            content_hash = self._generate_content_hash(content)
            existing_doc = await self.find_by_content_hash(content_hash)

            if existing_doc:
                # Update last_accessed_at timestamp
                from uuid import UUID

                doc_id = UUID(str(existing_doc.id))
                await self.update_last_accessed(doc_id)
                logger.debug(f"Found existing document: {doc_id}")
                return existing_doc, False

            # Create new document
            new_doc = await self.create_document(
                title=title,
                content=content,
                source_path=source_path,
                document_metadata=document_metadata,
            )
            logger.debug(f"Created new document: {UUID(str(new_doc.id))}")
            return new_doc, True

        except Exception as e:
            logger.error(f"Failed to get or create document: {e}")
            raise

    async def fulltext_search(
        self,
        query: str,
        limit: int = 20,
        offset: int | None = None,
    ) -> list[HistorianDocument]:
        """
        Perform full-text search using PostgreSQL's built-in search capabilities.

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching documents ordered by relevance
        """
        try:
            # Use PostgreSQL's to_tsquery for better search
            search_query = self._prepare_fulltext_query(query)

            stmt = (
                select(HistorianDocument)
                .where(text("search_vector @@ to_tsquery('english', :query)"))
                .order_by(
                    desc(text("ts_rank(search_vector, to_tsquery('english', :query))"))
                )
                .limit(limit)
            )

            if offset:
                stmt = stmt.offset(offset)

            result = await self.session.execute(stmt, {"query": search_query})
            documents = list(result.scalars().all())

            logger.debug(
                f"Fulltext search for '{query}' returned {len(documents)} results"
            )
            return documents

        except Exception as e:
            logger.error(f"Failed to perform fulltext search for '{query}': {e}")
            raise

    async def search_by_title(
        self, title_query: str, exact_match: bool = False, limit: int = 20
    ) -> list[HistorianDocument]:
        """
        Search documents by title.

        Args:
            title_query: Title search query
            exact_match: If True, use exact match; otherwise use ILIKE pattern
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        try:
            if exact_match:
                stmt = (
                    select(HistorianDocument)
                    .where(HistorianDocument.title == title_query)
                    .order_by(desc(HistorianDocument.created_at))
                    .limit(limit)
                )
            else:
                stmt = (
                    select(HistorianDocument)
                    .where(HistorianDocument.title.ilike(f"%{title_query}%"))
                    .order_by(desc(HistorianDocument.created_at))
                    .limit(limit)
                )

            result = await self.session.execute(stmt)
            documents = list(result.scalars().all())

            logger.debug(
                f"Title search for '{title_query}' returned {len(documents)} results"
            )
            return documents

        except Exception as e:
            logger.error(f"Failed to search by title '{title_query}': {e}")
            raise

    async def search_by_metadata(
        self, metadata_query: dict[str, Any], limit: int = 20
    ) -> list[HistorianDocument]:
        """
        Search documents by metadata using JSONB queries.

        Args:
            metadata_query: Dictionary of metadata key-value pairs to match
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        try:
            stmt = select(HistorianDocument).limit(limit)

            # Build JSONB query conditions
            for key, value in metadata_query.items():
                if isinstance(value, str):
                    # String values use ->> operator
                    stmt = stmt.where(
                        HistorianDocument.document_metadata[key].astext == value
                    )
                else:
                    # Other types use -> operator
                    stmt = stmt.where(HistorianDocument.document_metadata[key] == value)

            stmt = stmt.order_by(desc(HistorianDocument.created_at))

            result = await self.session.execute(stmt)
            documents = list(result.scalars().all())

            logger.debug(f"Metadata search returned {len(documents)} results")
            return documents

        except Exception as e:
            logger.error(f"Failed to search by metadata {metadata_query}: {e}")
            raise

    async def get_recent_documents(
        self, limit: int = 50, offset: int | None = None
    ) -> list[HistorianDocument]:
        """
        Get recent documents ordered by creation time.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of recent documents
        """
        try:
            stmt = (
                select(HistorianDocument)
                .order_by(desc(HistorianDocument.created_at))
                .limit(limit)
            )

            if offset:
                stmt = stmt.offset(offset)

            result = await self.session.execute(stmt)
            documents = list(result.scalars().all())

            logger.debug(f"Retrieved {len(documents)} recent documents")
            return documents

        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
            raise

    async def get_content_statistics(self) -> dict[str, Any]:
        """
        Get content statistics for analytics.

        Returns:
            Dictionary with content statistics
        """
        try:
            from sqlalchemy import func

            stmt = select(
                func.count().label("total_documents"),
                func.avg(HistorianDocument.word_count).label("avg_word_count"),
                func.max(HistorianDocument.word_count).label("max_word_count"),
                func.avg(HistorianDocument.char_count).label("avg_char_count"),
                func.max(HistorianDocument.char_count).label("max_char_count"),
                func.count(HistorianDocument.source_path).label(
                    "documents_with_source"
                ),
            )

            result = await self.session.execute(stmt)
            stats_row = result.first()

            if stats_row is None:
                return {
                    "total_documents": 0,
                    "avg_word_count": 0.0,
                    "max_word_count": 0,
                    "avg_char_count": 0.0,
                    "max_char_count": 0,
                    "documents_with_source": 0,
                }

            stats = {
                "total_documents": stats_row.total_documents or 0,
                "avg_word_count": float(stats_row.avg_word_count or 0),
                "max_word_count": stats_row.max_word_count or 0,
                "avg_char_count": float(stats_row.avg_char_count or 0),
                "max_char_count": stats_row.max_char_count or 0,
                "documents_with_source": stats_row.documents_with_source or 0,
            }

            logger.debug(
                f"Generated content statistics: {stats['total_documents']} documents"
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to get content statistics: {e}")
            raise

    async def update_last_accessed(self, document_id: UUID) -> bool:
        """
        Update the last_accessed_at timestamp for analytics.

        Args:
            document_id: Document UUID

        Returns:
            True if update was successful
        """
        try:
            stmt = (
                update(HistorianDocument)
                .where(HistorianDocument.id == document_id)
                .values(last_accessed_at=func.now())
            )

            result = await self.session.execute(stmt)
            success = result.rowcount > 0

            if success:
                await self.session.commit()
                logger.debug(f"Updated last_accessed_at for document: {document_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update last_accessed_at for {document_id}: {e}")
            raise

    async def cleanup_old_documents(
        self, days_old: int = 30, dry_run: bool = True
    ) -> dict[str, Any]:
        """
        Cleanup old documents based on last access time.

        Args:
            days_old: Documents older than this will be considered for cleanup
            dry_run: If True, only count documents without deleting

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            from sqlalchemy import and_

            cutoff_date = func.now() - text(f"INTERVAL '{days_old} days'")

            # Find old documents
            stmt = select(HistorianDocument).where(
                and_(
                    HistorianDocument.last_accessed_at.is_not(None),
                    HistorianDocument.last_accessed_at < cutoff_date,
                )
            )

            result = await self.session.execute(stmt)
            old_documents = list(result.scalars().all())

            if dry_run:
                logger.info(
                    f"Dry run: Found {len(old_documents)} documents older than {days_old} days"
                )
                return {
                    "documents_found": len(old_documents),
                    "documents_deleted": 0,
                    "dry_run": True,
                }

            # Delete old documents
            deleted_count = 0
            for doc in old_documents:
                from uuid import UUID

                await self.delete(UUID(str(doc.id)))
                deleted_count += 1

            logger.info(
                f"Cleanup completed: deleted {deleted_count} documents older than {days_old} days"
            )
            return {
                "documents_found": len(old_documents),
                "documents_deleted": deleted_count,
                "dry_run": False,
            }

        except Exception as e:
            logger.error(f"Failed to cleanup old documents: {e}")
            raise

    def _generate_content_hash(self, content: str) -> str:
        """
        Generate SHA-256 hash of content for deduplication.

        Args:
            content: Document content

        Returns:
            SHA-256 hash as hex string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _prepare_fulltext_query(self, query: str) -> str:
        """
        Prepare query for PostgreSQL full-text search.

        Args:
            query: Raw search query

        Returns:
            Prepared query for to_tsquery
        """
        # Simple query preparation - split words and join with &
        # For more advanced query parsing, this could be enhanced
        words = query.strip().split()
        if not words:
            return ""

        # Join words with AND operator for PostgreSQL tsquery
        # Escape special characters and handle quoted phrases
        prepared_words = []
        for word in words:
            # Remove special characters that might break tsquery
            clean_word = "".join(c for c in word if c.isalnum() or c in "'-")
            if clean_word:
                prepared_words.append(clean_word)

        return " & ".join(prepared_words) if prepared_words else ""
