"""Add historian documents schema with hybrid search capabilities

Revision ID: 9e9217e312f7
Revises: d23cfa5ca74f
Create Date: 2025-08-01 10:53:30.630378

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "9e9217e312f7"
down_revision: Union[str, Sequence[str], None] = "d23cfa5ca74f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create historian documents schema with hybrid search capabilities."""

    # Create historian_documents table
    op.create_table(
        "historian_documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "title",
            sa.String(500),
            nullable=False,
            comment="Document title (max 500 chars for search validation)",
        ),
        sa.Column("content", sa.Text, nullable=False, comment="Full document content"),
        sa.Column(
            "source_path",
            sa.String(1000),
            nullable=True,
            comment="Original file path or URL",
        ),
        sa.Column(
            "content_hash",
            sa.String(64),
            nullable=False,
            unique=True,
            comment="SHA-256 hash for deduplication",
        ),
        sa.Column(
            "word_count",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
            comment="Word count for analytics",
        ),
        sa.Column(
            "char_count",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
            comment="Character count for validation",
        ),
        # Structured metadata as JSONB for flexibility
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Flexible metadata: topics, tags, domain, etc.",
        ),
        # Generated columns for search optimization
        sa.Column(
            "excerpt",
            sa.String(1000),
            sa.Computed(
                "CASE WHEN char_length(content) <= 1000 THEN content ELSE left(content, 997) || '...' END"
            ),
            comment="Auto-generated excerpt (max 1000 chars)",
        ),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR,
            sa.Computed(
                "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))"
            ),
            comment="Full-text search vector",
        ),
        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_accessed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="For analytics and caching",
        ),
        comment="Historian documents with hybrid search capabilities",
    )

    # Create indexes for performance
    # GIN index for full-text search
    op.create_index(
        "idx_historian_documents_search_vector",
        "historian_documents",
        ["search_vector"],
        postgresql_using="gin",
    )

    # BTREE indexes for common queries
    op.create_index("idx_historian_documents_title", "historian_documents", ["title"])
    op.create_index(
        "idx_historian_documents_content_hash",
        "historian_documents",
        ["content_hash"],
        unique=True,
    )
    op.create_index(
        "idx_historian_documents_created_at", "historian_documents", ["created_at"]
    )
    op.create_index(
        "idx_historian_documents_word_count", "historian_documents", ["word_count"]
    )

    # GIN index for JSONB metadata queries
    op.create_index(
        "idx_historian_documents_metadata",
        "historian_documents",
        ["metadata"],
        postgresql_using="gin",
    )

    # Composite index for common search patterns
    op.create_index(
        "idx_historian_documents_title_created",
        "historian_documents",
        ["title", "created_at"],
    )

    # Create search analytics table for monitoring
    op.create_table(
        "historian_search_analytics",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "search_query", sa.Text, nullable=False, comment="Original search query"
        ),
        sa.Column(
            "search_type",
            sa.String(50),
            nullable=False,
            comment="Search type: fulltext, semantic, hybrid",
        ),
        sa.Column(
            "results_count", sa.Integer, nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "execution_time_ms",
            sa.Integer,
            nullable=True,
            comment="Query execution time in milliseconds",
        ),
        sa.Column(
            "user_session_id",
            sa.String(100),
            nullable=True,
            comment="User session for analytics",
        ),
        sa.Column(
            "search_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Search parameters, filters, ranking info",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        comment="Analytics for historian search operations",
    )

    # Indexes for analytics queries
    op.create_index(
        "idx_historian_search_analytics_created_at",
        "historian_search_analytics",
        ["created_at"],
    )
    op.create_index(
        "idx_historian_search_analytics_search_type",
        "historian_search_analytics",
        ["search_type"],
    )
    op.create_index(
        "idx_historian_search_analytics_execution_time",
        "historian_search_analytics",
        ["execution_time_ms"],
    )

    # Create trigger to update updated_at timestamp
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_historian_documents_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        CREATE TRIGGER trigger_historian_documents_updated_at
            BEFORE UPDATE ON historian_documents
            FOR EACH ROW
            EXECUTE FUNCTION update_historian_documents_updated_at();
    """
    )


def downgrade() -> None:
    """Remove historian documents schema."""

    # Drop trigger and function
    op.execute(
        "DROP TRIGGER IF EXISTS trigger_historian_documents_updated_at ON historian_documents;"
    )
    op.execute("DROP FUNCTION IF EXISTS update_historian_documents_updated_at();")

    # Drop tables (indexes will be dropped automatically)
    op.drop_table("historian_search_analytics")
    op.drop_table("historian_documents")
