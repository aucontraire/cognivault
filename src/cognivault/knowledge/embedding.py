"""Embedding generation for knowledge persistence (US2).

Wraps the OpenAI embeddings endpoint, reusing the existing client/config pattern
(``OpenAIConfig.load`` for credentials, ``KNOWLEDGE_EMBEDDING_*`` for model/dims).
It is the sole producer of vectors for the ``Topic.embedding`` column.

Guarantees (contracts/embedding-service.md):
- every returned vector has exactly ``dimensions`` floats (== the DB column width);
  a wrong-width vector is a hard error, never a silent write;
- provider/transport failures raise ``EmbeddingError`` (callers decide degradation);
- each call emits an event consistent with the event system (FR-008).
"""

from __future__ import annotations

import time
from typing import List, Optional

import openai
import tiktoken
from pydantic import BaseModel, Field

from cognivault.config.openai_config import OpenAIConfig
from cognivault.events.emitter import emit_agent_execution_completed
from cognivault.knowledge.config import KnowledgePersistenceConfig
from cognivault.observability import get_logger

logger = get_logger(__name__)

# text-embedding-3-* accept up to 8191 input tokens.
MAX_INPUT_TOKENS = 8191


class EmbeddingError(Exception):
    """Raised when embedding generation fails (provider error or wrong-width vector)."""


class EmbeddingResult(BaseModel):
    """A single embedding plus accounting metadata (data-model §3)."""

    vector: List[float]
    model: str
    input_tokens: int = Field(ge=0)


class EmbeddingService:
    """Generates topic embeddings via the OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: Optional[str] = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def from_env(cls) -> "EmbeddingService":
        """Credentials from ``OpenAIConfig.load()``; model/dimensions from KNOWLEDGE_*."""
        openai_config = OpenAIConfig.load()
        knowledge_config = KnowledgePersistenceConfig.from_env()
        return cls(
            api_key=openai_config.api_key,
            model=knowledge_config.embedding_model,
            dimensions=knowledge_config.embedding_dimensions,
            base_url=openai_config.base_url,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _truncate(self, text: str) -> tuple[str, int]:
        """Deterministically truncate to MAX_INPUT_TOKENS; return (text, token_count)."""
        tokens = self._encoding.encode(text)
        if len(tokens) > MAX_INPUT_TOKENS:
            tokens = tokens[:MAX_INPUT_TOKENS]
            text = self._encoding.decode(tokens)
        return text, len(tokens)

    async def embed(self, text: str) -> EmbeddingResult:
        """Return one embedding. Raises EmbeddingError on provider failure or wrong width."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts in a single API call. Raises EmbeddingError on failure."""
        if not texts:
            return []
        prepared: List[str] = []
        token_counts: List[int] = []
        for text in texts:
            truncated, count = self._truncate(text)
            prepared.append(truncated)
            token_counts.append(count)

        start = time.time()
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=prepared, dimensions=self._dimensions
            )
        except Exception as exc:  # provider/transport failure
            raise EmbeddingError(
                f"Embedding request failed ({type(exc).__name__}): {exc}"
            ) from exc
        duration_ms = (time.time() - start) * 1000

        if len(response.data) != len(prepared):
            raise EmbeddingError(
                f"Embedding provider returned {len(response.data)} vectors for "
                f"{len(prepared)} inputs"
            )

        results: List[EmbeddingResult] = []
        for item, count in zip(response.data, token_counts):
            vector = list(item.embedding)
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"Embedding width {len(vector)} != expected {self._dimensions}; "
                    "refusing to return a mis-sized vector"
                )
            results.append(
                EmbeddingResult(vector=vector, model=self._model, input_tokens=count)
            )

        await self._emit_event(len(results), sum(token_counts), duration_ms)
        return results

    async def _emit_event(
        self, count: int, input_tokens: int, duration_ms: float
    ) -> None:
        try:
            await emit_agent_execution_completed(
                workflow_id="embedding",
                agent_name="embedding",
                success=True,
                output_context={},
                execution_time_ms=duration_ms,
                metadata={
                    "model": self._model,
                    "dimensions": self._dimensions,
                    "embeddings": count,
                    "input_tokens": input_tokens,
                },
            )
        except Exception as exc:
            logger.debug(f"Embedding event emission failed: {exc}")
