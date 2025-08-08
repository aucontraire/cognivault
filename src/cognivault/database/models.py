"""
Database models for CogniVault PostgreSQL + pgvector integration.

GraphRAG-ready schema with hierarchical topics, vector embeddings,
and semantic relationship tracking.
"""

import uuid
from datetime import datetime
from typing import Optional
from uuid import UUID as UUID_TYPE

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from typing import List
from sqlalchemy import (
    ARRAY,
    UUID,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Topic(Base):
    """
    Topics table with hierarchical support and vector embeddings.

    Designed for GraphRAG readiness with parent-child relationships
    and semantic similarity search via pgvector.
    """

    __tablename__ = "topics"

    # Primary identification
    id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Hierarchical topic organization (GraphRAG prep)
    parent_topic_id: Mapped[Optional[UUID_TYPE]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("topics.id"), nullable=True
    )
    parent = relationship("Topic", remote_side=[id], backref="children")

    # Vector embedding for semantic similarity (text-embedding-3-large)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1536), nullable=True)

    # Metadata and timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Performance indexes
    __table_args__ = (
        Index("idx_topics_name", "name"),
        Index("idx_topics_parent", "parent_topic_id"),
        Index(
            "idx_topics_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Question(Base):
    """
    Questions table with graph-friendly relationships and execution metadata.

    Stores workflow queries with DAG execution paths and semantic relationships
    for future GraphRAG integration.
    """

    __tablename__ = "questions"

    # Primary identification
    id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query: Mapped[Optional[str]] = mapped_column(Text, nullable=False)

    # Topic and semantic relationships
    topic_id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), ForeignKey("topics.id"), nullable=True
    )
    topic = relationship("Topic", backref="questions")

    # Future GraphRAG edge: IS_SIMILAR_TO relationship
    similar_to: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), ForeignKey("questions.id"), nullable=True
    )
    similar_question = relationship(
        "Question", remote_side=[id], backref="similar_questions"
    )

    # Workflow execution tracking
    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True
    )
    execution_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # For efficient filtering
    nodes_executed: Mapped[Optional[list]] = mapped_column(
        ARRAY(String), nullable=True
    )  # DAG execution path

    # Rich metadata storage (workflow results, agent outputs, performance)
    execution_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Performance indexes
    __table_args__ = (
        Index("idx_questions_correlation", "correlation_id"),
        Index("idx_questions_execution", "execution_id"),
        Index("idx_questions_topic", "topic_id"),
        Index("idx_questions_similar", "similar_to"),
        Index(
            "idx_questions_execution_metadata",
            "execution_metadata",
            postgresql_using="gin",
        ),
        Index("idx_questions_created", "created_at"),
    )


class WikiEntry(Base):
    """
    Wiki entries table with versioning and knowledge lineage tracking.

    Supports knowledge evolution with version history and multi-source
    synthesis for collaborative knowledge building.
    """

    __tablename__ = "wiki_entries"

    # Primary identification
    id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Topic and source relationships
    topic_id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), ForeignKey("topics.id"), nullable=False
    )
    topic = relationship("Topic", backref="wiki_entries")

    question_id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), ForeignKey("questions.id"), nullable=True
    )
    source_question = relationship("Question", backref="wiki_entries")

    # Content and versioning
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=False)
    version = Column(Integer, default=1)

    # Knowledge evolution tracking
    supersedes = Column(
        UUID(as_uuid=True), ForeignKey("wiki_entries.id"), nullable=True
    )
    superseded_entry = relationship(
        "WikiEntry", remote_side=[id], backref="superseding_entries"
    )

    # Multi-source synthesis
    sources = Column(ARRAY(UUID), nullable=True)  # type: ignore  # Contributing question IDs
    related_topics = Column(ARRAY(UUID), nullable=True)  # type: ignore  # Multi-topic relationships

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Performance indexes
    __table_args__ = (
        Index("idx_wiki_topic_version", "topic_id", "version"),
        Index("idx_wiki_source_question", "question_id"),
        Index("idx_wiki_supersedes", "supersedes"),
        Index("idx_wiki_created", "created_at"),
    )


class APIKey(Base):
    """
    API keys table for authentication and usage tracking.

    Supports rate limiting, quotas, and usage analytics for
    production API access control.
    """

    __tablename__ = "api_keys"

    # Primary identification
    id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    # Metadata
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Usage control
    rate_limit = Column(Integer, default=100)  # requests per minute
    daily_quota = Column(Integer, default=1000)
    usage_count = Column(Integer, default=0)

    # Status and lifecycle
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Performance indexes
    __table_args__ = (
        Index("idx_api_keys_hash", "key_hash"),
        Index(
            "idx_api_keys_active",
            "is_active",
            postgresql_where=Column("is_active"),
        ),
        Index("idx_api_keys_expires", "expires_at"),
    )


class SemanticLink(Base):
    """
    Semantic relationships table for future GraphRAG integration.

    Tracks relationships between entities (topics, questions, wiki entries)
    with weighted connections for knowledge graph construction.
    """

    __tablename__ = "semantic_links"

    # Primary identification
    id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Source entity
    from_entity_type = Column(
        String(50), nullable=False
    )  # 'topic', 'question', 'wiki_entry'
    from_entity_id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )

    # Target entity
    to_entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    to_entity_id: Mapped[UUID_TYPE] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Relationship metadata
    relation = Column(
        String(100), nullable=False
    )  # 'refines', 'related_to', 'derived_from'
    weight = Column(Float, default=1.0)  # Relationship strength

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Performance indexes for graph traversal
    __table_args__ = (
        Index("idx_semantic_from", "from_entity_type", "from_entity_id"),
        Index("idx_semantic_to", "to_entity_type", "to_entity_id"),
        Index("idx_semantic_relation", "relation"),
        Index("idx_semantic_weight", "weight"),
    )


class HistorianDocument(Base):
    """
    Historian documents table with hybrid search capabilities.

    Supports full-text search, content deduplication, and analytics
    for enhanced document storage and retrieval in the historian system.
    """

    __tablename__ = "historian_documents"

    # Primary identification
    id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(
        String(500), nullable=False
    )  # Max 500 chars for search validation
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=False)

    # Document metadata and organization
    source_path: Mapped[str] = mapped_column(
        String(1000), nullable=True
    )  # Original file path or URL
    content_hash = Column(
        String(64), nullable=False, unique=True
    )  # SHA-256 for deduplication

    # Content analytics
    word_count = Column(Integer, nullable=False, default=0)
    char_count = Column(Integer, nullable=False, default=0)

    # Flexible metadata as JSONB (using custom attribute name to avoid SQLAlchemy conflict)
    document_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Performance indexes
    __table_args__ = (
        Index("idx_historian_documents_title", "title"),
        Index("idx_historian_documents_content_hash", "content_hash", unique=True),
        Index("idx_historian_documents_created_at", "created_at"),
        Index("idx_historian_documents_word_count", "word_count"),
        Index(
            "idx_historian_documents_metadata",
            "metadata",
            postgresql_using="gin",
        ),
        Index("idx_historian_documents_title_created", "title", "created_at"),
    )


class HistorianSearchAnalytics(Base):
    """
    Analytics for historian search operations.

    Tracks search queries, performance metrics, and usage patterns
    for monitoring and optimization of the historian search system.
    """

    __tablename__ = "historian_search_analytics"

    # Primary identification
    id: Mapped[UUID_TYPE] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Search query details
    search_query: Mapped[Optional[str]] = mapped_column(
        Text, nullable=False
    )  # Original search query
    search_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # fulltext, semantic, hybrid

    # Performance and results tracking
    results_count = Column(Integer, nullable=False, default=0)
    execution_time_ms = Column(Integer, nullable=True)  # Query execution time

    # User and session tracking
    user_session_id: Mapped[str] = mapped_column(String(100), nullable=True)

    # Flexible search metadata (using custom attribute name to avoid SQLAlchemy conflict)
    search_metadata = Column("search_metadata", JSONB, nullable=False, default=dict)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Performance indexes for analytics queries
    __table_args__ = (
        Index("idx_historian_search_analytics_created_at", "created_at"),
        Index("idx_historian_search_analytics_search_type", "search_type"),
        Index("idx_historian_search_analytics_execution_time", "execution_time_ms"),
    )


# Convenience type definitions for application use
TopicModel = Topic
QuestionModel = Question
WikiEntryModel = WikiEntry
APIKeyModel = APIKey
SemanticLinkModel = SemanticLink
HistorianDocumentModel = HistorianDocument
HistorianSearchAnalyticsModel = HistorianSearchAnalytics
