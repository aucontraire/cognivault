"""Contract tests for EmbeddingService (T018, FR-004).

Mocks the OpenAI client, so these run without a real API key or network.
"""

from typing import List
from unittest.mock import AsyncMock

import pytest

from cognivault.knowledge.embedding import (
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
)


class _FakeItem:
    def __init__(self, embedding: List[float]) -> None:
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, vectors: List[List[float]]) -> None:
        self.data = [_FakeItem(v) for v in vectors]


def _service(dimensions: int = 1536) -> EmbeddingService:
    return EmbeddingService(api_key="test-key", dimensions=dimensions)


def _mock_create(
    svc: EmbeddingService, monkeypatch: pytest.MonkeyPatch, vectors: List[List[float]]
) -> None:
    monkeypatch.setattr(
        svc._client.embeddings,
        "create",
        AsyncMock(return_value=_FakeResponse(vectors)),
    )


@pytest.mark.asyncio
async def test_embed_returns_correct_width_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _service(dimensions=1536)
    _mock_create(svc, monkeypatch, [[0.01] * 1536])
    result = await svc.embed("event-driven architecture")
    assert isinstance(result, EmbeddingResult)
    assert len(result.vector) == 1536
    assert result.model == "text-embedding-3-small"
    assert result.input_tokens > 0


@pytest.mark.asyncio
async def test_wrong_width_vector_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _service(dimensions=1536)
    _mock_create(svc, monkeypatch, [[0.01] * 100])  # wrong width
    with pytest.raises(EmbeddingError, match="width"):
        await svc.embed("something")


@pytest.mark.asyncio
async def test_provider_error_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _service()
    monkeypatch.setattr(
        svc._client.embeddings,
        "create",
        AsyncMock(side_effect=RuntimeError("connection reset")),
    )
    with pytest.raises(EmbeddingError):
        await svc.embed("something")


@pytest.mark.asyncio
async def test_empty_batch_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _service()
    # No API call should be needed for an empty batch.
    monkeypatch.setattr(
        svc._client.embeddings,
        "create",
        AsyncMock(side_effect=AssertionError("should not be called")),
    )
    assert await svc.embed_batch([]) == []


@pytest.mark.asyncio
async def test_batch_returns_one_result_per_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _service(dimensions=1536)
    _mock_create(svc, monkeypatch, [[0.01] * 1536, [0.02] * 1536])
    results = await svc.embed_batch(["a", "b"])
    assert len(results) == 2
    assert all(len(r.vector) == 1536 for r in results)
