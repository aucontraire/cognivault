"""Unit tests for KnowledgePersistenceConfig (T004, FR-011/FR-014)."""

import pytest

from cognivault.knowledge.config import (
    EMBEDDING_COLUMN_DIM,
    KnowledgePersistenceConfig,
)


def test_defaults_match_spec() -> None:
    c = KnowledgePersistenceConfig()
    assert c.persistence_enabled is False  # opt-in; see config docstring
    assert c.topic_min_confidence == 0.7
    assert c.embedding_model == "text-embedding-3-small"
    assert c.embedding_dimensions == EMBEDDING_COLUMN_DIM == 1536
    assert c.max_embedding_calls_per_run == 8


def test_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_PERSISTENCE_ENABLED", "true")
    monkeypatch.setenv("KNOWLEDGE_TOPIC_MIN_CONFIDENCE", "0.9")
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_MODEL", "custom-embed")
    monkeypatch.setenv("KNOWLEDGE_MAX_EMBEDDING_CALLS_PER_RUN", "3")
    c = KnowledgePersistenceConfig.from_env()
    assert c.persistence_enabled is True
    assert c.topic_min_confidence == 0.9
    assert c.embedding_model == "custom-embed"
    assert c.max_embedding_calls_per_run == 3


def test_from_env_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "KNOWLEDGE_PERSISTENCE_ENABLED",
        "KNOWLEDGE_TOPIC_MIN_CONFIDENCE",
        "KNOWLEDGE_EMBEDDING_MODEL",
        "KNOWLEDGE_EMBEDDING_DIMENSIONS",
        "KNOWLEDGE_MAX_EMBEDDING_CALLS_PER_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    c = KnowledgePersistenceConfig.from_env()
    assert c.embedding_dimensions == EMBEDDING_COLUMN_DIM
    assert c.topic_min_confidence == 0.7


def test_dimension_mismatch_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-014: dimensions != column width must fail at construction."""
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_DIMENSIONS", "3072")
    with pytest.raises(ValueError, match="does not"):
        KnowledgePersistenceConfig.from_env()


def test_confidence_out_of_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        KnowledgePersistenceConfig(topic_min_confidence=1.5)
    with pytest.raises(ValueError):
        KnowledgePersistenceConfig(topic_min_confidence=-0.1)
