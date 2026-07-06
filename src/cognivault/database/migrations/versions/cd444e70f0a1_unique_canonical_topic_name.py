"""unique canonical topic name

Revision ID: cd444e70f0a1
Revises: 9e9217e312f7
Create Date: 2026-07-05 07:28:46.708963

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "cd444e70f0a1"
down_revision: Union[str, Sequence[str], None] = "9e9217e312f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add topics.canonical_name (dedup key) + unique constraint (FR-002 / FR-012).

    Data-preserving: add nullable, backfill from name using the same canonicalization
    as ``TopicRepository.canonicalize`` (strip → lower → collapse whitespace, ≤255),
    defensively disambiguate any pre-existing collisions by appending the row id, then
    enforce NOT NULL + UNIQUE.
    """
    op.add_column(
        "topics", sa.Column("canonical_name", sa.String(length=255), nullable=True)
    )
    op.execute(
        "UPDATE topics SET canonical_name = "
        "left(lower(regexp_replace(btrim(name), '[[:space:]]+', ' ', 'g')), 255)"
    )
    # Defensive: keep every row even if legacy names collide on the canonical key.
    op.execute(
        """
        WITH ranked AS (
            SELECT id, row_number() OVER (
                PARTITION BY canonical_name ORDER BY created_at, id
            ) AS rn
            FROM topics
        )
        UPDATE topics t
        SET canonical_name = left(t.canonical_name || '-' || t.id::text, 255)
        FROM ranked r
        WHERE t.id = r.id AND r.rn > 1
        """
    )
    op.alter_column("topics", "canonical_name", nullable=False)
    op.create_unique_constraint(
        "uq_topics_canonical_name", "topics", ["canonical_name"]
    )


def downgrade() -> None:
    """Drop the unique constraint and the canonical_name column."""
    op.drop_constraint("uq_topics_canonical_name", "topics", type_="unique")
    op.drop_column("topics", "canonical_name")
