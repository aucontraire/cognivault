"""Off-parity golden contract (T026, FR-007, US3 scenario 3).

The safety contract for the whole feature: with HISTORIAN_SEMANTIC_SEARCH_ENABLED=false
(the default), the Historian's hybrid search must behave EXACTLY as before —
byte-for-byte identical results, same ordering. This test proves it two ways:

1. With the flag off, the semantic code path is never invoked (spy asserts not-awaited),
   so behavior is provably unchanged.
2. With the flag on but the semantic contribution empty, results are identical to the
   flag-off baseline — semantic retrieval is purely additive.

No database or network required — file/db search are stubbed with fixed results.
"""

from typing import List
from unittest.mock import AsyncMock

import pytest

from cognivault.agents.historian import search as search_mod
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.historian.search import SearchResult
from cognivault.config.agent_configs import HistorianConfig
from cognivault.context import AgentContext


def _result(name: str, score: float) -> SearchResult:
    return SearchResult(
        filepath=f"/notes/{name}.md",
        filename=f"{name}.md",
        title=name,
        date="2024-01-01",
        relevance_score=score,
        match_type="keyword",
        excerpt=f"excerpt for {name}",
        metadata={},
    )


_FILE_RESULTS = [_result("file-a", 9.0), _result("file-b", 5.0)]
_DB_RESULTS = [_result("db-a", 7.0)]


@pytest.fixture(autouse=True)
def _force_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the hybrid (file+db) path so the flag-gated semantic block is reached."""

    class _Testing:
        enable_hybrid_search = True
        historian_search_limit = 10

    class _AppConfig:
        testing = _Testing()

    monkeypatch.setattr(
        "cognivault.agents.historian.agent.get_config", lambda: _AppConfig()
    )


def _agent(semantic_enabled: bool) -> HistorianAgent:
    agent = HistorianAgent(
        llm=None,
        config=HistorianConfig(
            hybrid_search_enabled=True, semantic_search_enabled=semantic_enabled
        ),
    )

    async def fake_file(query: str, limit: int) -> List[SearchResult]:
        return list(_FILE_RESULTS)

    async def fake_db(query: str, limit: int) -> List[SearchResult]:
        return list(_DB_RESULTS)

    agent._search_file_content = fake_file  # type: ignore[method-assign]
    agent._search_database_content = fake_db  # type: ignore[method-assign]
    return agent


@pytest.mark.asyncio
async def test_semantic_off_never_invokes_semantic_and_matches_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = AgentContext(query="q")
    spy = AsyncMock(return_value=[])
    monkeypatch.setattr(search_mod.SemanticSearch, "search", spy)

    # Flag OFF (default): semantic search must never be invoked.
    results_off = await _agent(False)._search_historical_content("q", ctx)
    spy.assert_not_awaited()

    # Flag ON but empty semantic contribution: byte-for-byte identical to baseline.
    results_on = await _agent(True)._search_historical_content("q", ctx)
    spy.assert_awaited()
    assert results_on == results_off  # identical content AND ordering


@pytest.mark.asyncio
async def test_semantic_on_with_results_changes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = AgentContext(query="q")
    # A high-scoring semantic hit must change the result set (proves the flag has effect).
    semantic_hit = _result("semantic-topic", 9.9)
    monkeypatch.setattr(
        search_mod.SemanticSearch, "search", AsyncMock(return_value=[semantic_hit])
    )

    results_off = await _agent(False)._search_historical_content("q", ctx)
    results_on = await _agent(True)._search_historical_content("q", ctx)

    assert results_on != results_off
    assert any(r.title == "semantic-topic" for r in results_on)
    assert not any(r.title == "semantic-topic" for r in results_off)
