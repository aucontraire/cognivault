"""Configuration for knowledge persistence & semantic retrieval.

Reads the ``KNOWLEDGE_*`` environment variables following the same ``from_env``
pattern as ``config/agent_configs.py``. All fields are validated at construction
time; a dimension mismatch fails fast (FR-014) so mis-sized vectors are never
written to the ``Topic.embedding`` column.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, model_validator

# The pgvector column width in database/models.py (Topic.embedding = Vector(1536)).
# KNOWLEDGE_EMBEDDING_DIMENSIONS MUST equal this or startup fails (FR-014).
EMBEDDING_COLUMN_DIM = 1536


class KnowledgePersistenceConfig(BaseModel):
    """Typed configuration for the knowledge write-path and embedding generation."""

    persistence_enabled: bool = Field(
        False,
        description="Explicit opt-in for the write path (FR-001). Default off so that a "
        "merely-reachable database (e.g. a dev DB during tests) never triggers writes; "
        "set KNOWLEDGE_PERSISTENCE_ENABLED=true to enable, as the quickstart does.",
    )
    topic_min_confidence: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum TopicManager suggestion confidence to persist a topic "
        "(FR-011). False-positive topics are permanent, so this biases toward precision.",
    )
    embedding_model: str = Field(
        "text-embedding-3-small",
        description="OpenAI-compatible embedding model (FR-004).",
    )
    embedding_dimensions: int = Field(
        EMBEDDING_COLUMN_DIM,
        ge=1,
        description="Embedding width; MUST equal the Topic.embedding column (FR-014).",
    )
    max_embedding_calls_per_run: int = Field(
        8,
        ge=1,
        description="Per-run cap on embedding API calls for cost control.",
    )

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "KnowledgePersistenceConfig":
        if self.embedding_dimensions != EMBEDDING_COLUMN_DIM:
            raise ValueError(
                f"KNOWLEDGE_EMBEDDING_DIMENSIONS={self.embedding_dimensions} does not "
                f"match the Topic.embedding column width ({EMBEDDING_COLUMN_DIM}). "
                "Refusing to start rather than write mis-sized vectors (FR-014)."
            )
        return self

    @classmethod
    def from_env(cls, prefix: str = "KNOWLEDGE") -> "KnowledgePersistenceConfig":
        """Build config from ``{prefix}_*`` environment variables (documented defaults)."""
        config: dict[str, object] = {}
        if env_val := os.getenv(f"{prefix}_PERSISTENCE_ENABLED"):
            config["persistence_enabled"] = env_val.lower() == "true"
        if env_val := os.getenv(f"{prefix}_TOPIC_MIN_CONFIDENCE"):
            config["topic_min_confidence"] = float(env_val)
        if env_val := os.getenv(f"{prefix}_EMBEDDING_MODEL"):
            config["embedding_model"] = env_val
        if env_val := os.getenv(f"{prefix}_EMBEDDING_DIMENSIONS"):
            config["embedding_dimensions"] = int(env_val)
        if env_val := os.getenv(f"{prefix}_MAX_EMBEDDING_CALLS_PER_RUN"):
            config["max_embedding_calls_per_run"] = int(env_val)
        return cls(**config)
