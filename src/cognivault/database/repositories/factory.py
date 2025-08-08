"""
Repository factory for creating repository instances.

Provides centralized repository management with session lifecycle
and transaction consistency across all repository operations.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from cognivault.observability import get_logger

from .api_key_repository import APIKeyRepository
from .historian_document_repository import HistorianDocumentRepository
from .historian_search_analytics_repository import HistorianSearchAnalyticsRepository
from .question_repository import QuestionRepository
from .topic_repository import TopicRepository
from .wiki_repository import WikiRepository

logger = get_logger(__name__)


class RepositoryFactory:
    """
    Factory for creating repository instances with a shared database session.

    Provides a centralized way to create all repositories with the same
    session for transaction consistency and proper resource management.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize factory with database session.

        Args:
            session: Async SQLAlchemy session for database operations
        """
        self.session = session
        self._repositories: dict = {}
        logger.debug("Repository factory initialized with new session")

    @property
    def topics(self) -> TopicRepository:
        """Get topic repository instance (cached)."""
        if "topics" not in self._repositories:
            self._repositories["topics"] = TopicRepository(self.session)
        return self._repositories["topics"]

    @property
    def questions(self) -> QuestionRepository:
        """Get question repository instance (cached)."""
        if "questions" not in self._repositories:
            self._repositories["questions"] = QuestionRepository(self.session)
        return self._repositories["questions"]

    @property
    def wiki(self) -> WikiRepository:
        """Get wiki repository instance (cached)."""
        if "wiki" not in self._repositories:
            self._repositories["wiki"] = WikiRepository(self.session)
        return self._repositories["wiki"]

    @property
    def api_keys(self) -> APIKeyRepository:
        """Get API key repository instance (cached)."""
        if "api_keys" not in self._repositories:
            self._repositories["api_keys"] = APIKeyRepository(self.session)
        return self._repositories["api_keys"]

    @property
    def historian_documents(self) -> HistorianDocumentRepository:
        """Get historian document repository instance (cached)."""
        if "historian_documents" not in self._repositories:
            self._repositories["historian_documents"] = HistorianDocumentRepository(
                self.session
            )
        return self._repositories["historian_documents"]

    @property
    def historian_search_analytics(self) -> HistorianSearchAnalyticsRepository:
        """Get historian search analytics repository instance (cached)."""
        if "historian_search_analytics" not in self._repositories:
            self._repositories["historian_search_analytics"] = (
                HistorianSearchAnalyticsRepository(self.session)
            )
        return self._repositories["historian_search_analytics"]

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()
        logger.debug("Repository factory session committed")

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()
        logger.debug("Repository factory session rolled back")

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        await self.session.close()
        self._repositories.clear()
        logger.debug("Repository factory session closed")

    def clear_cache(self) -> None:
        """Clear cached repository instances."""
        self._repositories.clear()
        logger.debug("Repository factory cache cleared")
