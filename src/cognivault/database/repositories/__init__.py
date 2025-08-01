"""
Repository pattern implementation for CogniVault database operations.

Provides abstractions for data access with standardized CRUD operations,
query methods, and transaction management.
"""

from .api_key_repository import APIKeyRepository
from .base import BaseRepository
from .factory import RepositoryFactory
from .historian_document_repository import HistorianDocumentRepository
from .historian_search_analytics_repository import HistorianSearchAnalyticsRepository
from .question_repository import QuestionRepository
from .topic_repository import TopicRepository
from .wiki_repository import WikiRepository

__all__ = [
    "BaseRepository",
    "TopicRepository",
    "QuestionRepository",
    "WikiRepository",
    "APIKeyRepository",
    "HistorianDocumentRepository",
    "HistorianSearchAnalyticsRepository",
    "RepositoryFactory",
]
