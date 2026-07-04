# Binding Decisions Register (Distilled from 2025 Archives)

**Created**: 2026-07-04, as part of closing branch `docs/consolidate-internal-documentation`.
**Source**: Distilled from ~45 archived root-level planning/analysis documents; originals at `.internal-docs/archive/root-cleanup-2026-07/` (see its `INDEX.md`). Companion: `LESSONS_LEARNED.md` (same directory).

Decisions below were actually made and remain binding unless explicitly revisited. Candidates for formal ADRs if the ADR practice is picked up.

---

## Structured outputs & schema

- **D-01 GPT-5 API parameters**: use `max_completion_tokens` (not `max_tokens`) and the `native_parse` method for GPT-5 models. (Also canonized in `.claude/reference/openai-compatibility-reference.md`.)
- **D-02 Schema nullability rule**: `default_factory` fields are required-and-non-nullable in OpenAI schemas; only `Optional[T]`/`default=None` fields get `anyOf` null. Enforced by the OpenAI-schema test suite (21 tests) + pre-commit hook.
- **D-03 `bias_details` shape**: `List[BiasDetail]` with a `BiasType` enum, replacing `Dict[str, str]`. Rationale: strict-mode compatibility (`additionalProperties: false`), type safety, eliminates hallucinated bias categories, and empirically captured 100% of data vs the Dict's 40–60% loss (no duplicate keys). Trade-off accepted: no arbitrary custom bias strings.
- **D-04 Server-side metadata injection**: `processing_time_ms` and `timestamp` are populated server-side around the LLM call, never requested from the model.
- **D-05 Validation philosophy**: critical validation (types, required fields) fails hard; advisory validation (string lengths, count consistency) warns only. Item length limits raised 150→250 chars; mechanical count tolerance removed.

## Persistence & state

- **D-06 Store models, not strings**: agent outputs persist as full Pydantic `model_dump()` JSONB (with GIN index), not `str(output)`; API models widened to `Dict[str, Any]`. (Implementation still incomplete in some agent paths — see LESSONS_LEARNED open threads.)
- **D-07 LangGraph parallel-state merging**: `structured_outputs` uses a custom merge reducer (`Annotated[Dict[str, Any], merge_structured_outputs]`), each agent writing its own key; all four node wrappers extract with a `.get("structured_outputs", {})` fallback.
- **D-08 WebSocket compatibility**: dual-format events — `agent_outputs` stays string-typed for legacy clients; full structured data rides in `metadata.structured_outputs`.
- **D-09 Historian relevance safeguard**: when LLM filtering rejects all results but search found some, keep top-N by search score (`minimum_results_threshold`, default 3).
- **D-10 Export frontmatter**: wiki/markdown export extracts agent metadata from structured outputs into frontmatter (helper converting Pydantic outputs to `AgentExecutionResult`); summaries prefer refiner's `refined_query`, then synthesis themes, then truncated synthesis text.

## Documentation & repo governance

- **D-11 MkDocs Material** is the documentation system (mkdocstrings for API docs), adopted from echomine's patterns; GitHub Actions workflows for test/docs/release/security were recommended and drafted (adoption still pending — workflows sit untracked in `.github/workflows/`).
- **D-12 Two-tier docs structure**: public, user-facing docs in `docs/` (MkDocs tree); internal working docs, investigations, and archives in `.internal-docs/` (gitignored). Migration executed on this branch: 46 public files, 130 internal files, 218 references updated, 0 broken links.
- **D-13 Artifact lifecycle policy**: investigation notes are written in `.internal-docs/` from the start (never repo root); at branch close, durable outcomes are distilled into this register / LESSONS_LEARNED / ADRs, and the raw material is archived under `.internal-docs/archive/`. Debug scripts live in `scripts/` or are deleted, never at root.

---

## Preserved design proposals (not decisions — roadmap candidates)

These two designs were fully worked out but never implemented; summaries preserved so archiving doesn't lose the ideas. Full documents in the archive's `design-proposals/`.

- **P-01 Web search integration (Tavily) as Historian fallback** — add web search as a third tier *inside* the Historian's existing hybrid search (file → database → web), not as a fifth agent. Reuses the hybrid-search ratio/config pattern; Tavily API as provider. Rationale: keeps the 4-agent topology stable while fixing the empty-context problem for novel queries. Natural companion to spec `002-knowledge-persistence-semantic-retrieval`.
- **P-02 Title generation UX polish** — the title-generation system works; remaining work is minor (log-level correction plus clearer context messaging, ~1–2h). Low priority.

## Superseded / rejected along the way

- Dict-shaped `bias_details` (rejected by D-03).
- LLM-generated timing/timestamp metadata (rejected by D-04).
- Hard-fail advisory validation and ±2 count tolerance (rejected by D-05).
- Retrying deterministic validation failures (rejected — classify errors before retry; see LESSONS_LEARNED).
