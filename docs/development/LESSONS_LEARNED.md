# Lessons Learned (Distilled from 2025 Debugging Archives)

**Created**: 2026-07-04, as part of closing branch `docs/consolidate-internal-documentation`.
**Source**: Distilled from ~45 root-level analysis/report documents (Sep–Dec 2025, primarily GPT-5 structured-output integration work), now archived at `.internal-docs/archive/root-cleanup-2026-07/`. See the `INDEX.md` there for the full file-by-file manifest.

These are the hard-won, still-relevant operational lessons. Full investigation narratives live in the archive.

---

## OpenAI structured-output schema rules (GPT-5)

1. **All properties must appear in the `required` array**, including fields with `default_factory` — but only fields typed `Optional[T]` or `default=None` may use the `anyOf: [..., null]` nullable pattern. Marking a `default_factory` field nullable caused a **100% failure rate** (`timestamp` returned None in 28/28 attempts): OpenAI reads "required but nullable" as "return null."
2. **Check `.default` and `.default_factory` separately** against `PydanticUndefined` when transforming schemas; Pydantic's generated JSON schema does not natively meet OpenAI strict-mode requirements.
3. **`Dict[str, str]` fields with dynamic keys violate `additionalProperties: false`.** This drove the `bias_details` redesign from Dict to `List[BiasDetail]` with a `BiasType` enum (see DECISIONS.md). Also: OpenAI returns `null` for Dict-typed fields even when the schema says otherwise, and Pydantic then rejects it.
4. **Nested models need recursive schema handling** — `$defs` entries must get their own correct `required` arrays (the `HistoricalReference.source_id` failure). This was identified but never definitively verified as fixed; treat as an open thread.
5. **Metadata fields must never be in the LLM schema.** `processing_time_ms` and `timestamp` are server-side concerns: inject them around the LLM call (in `LangChainService.get_structured_output()`), before Pydantic instantiation. An LLM cannot know wall-clock timing.

## Timeout cascades are usually validation failures in disguise

The infamous "Critic timeout" pattern: schema passes → OpenAI responds → Pydantic validation fails → retry → same deterministic failure → 3 retries × 60s → workflow timeout. Key sub-lessons:

- **Schema/validation errors are not retryable.** Retrying a deterministic validation failure just burns the time budget. Classify errors before retrying.
- **A time-budget check that logs but doesn't `break` is not a budget** (the exhaustion check in `langchain_service.py` warned and kept looping).
- **Don't impose human-scale limits on LLM output.** 150-char item limits failed 60–70% of the time against GPT-5's natural ~175-char sentences (max observed 265). Mechanical count tolerances (±2 between `issues_detected` and detail-item counts) conflict with the model's semantic grouping (22 issues, 27 items). Resolution: split *critical* validation (types, required fields → hard fail) from *advisory* validation (lengths, counts → warnings). Limits were raised (150→250) and count tolerance removed.
- **Retries can get worse, not better**: later attempts produced longer output that violated tight constraints harder.

## Retrieval and persistence gotchas

- **LLM relevance filtering needs a floor.** The Historian's LLM filter sometimes rejected *every* search result despite relevant documents existing. Safeguard: when the filter returns zero but search returned results, keep the top-N by search score (`minimum_results_threshold`, default 3), logged at WARNING.
- **`str(output)` is where metadata goes to die.** The persistence chain was correct (JSONB column, GIN index, 13 query methods) but agents returned formatted strings, and the orchestrator stringified outputs (~line 463), so `execution_metadata->'agent_outputs'->X` stored `"string"` not `"object"` — losing confidence, timing, everything. Rule: pass Pydantic models end-to-end; `model_dump()` only at the serialization boundary. (Partially fixed; see open threads.)
- **The same loss pattern hit markdown export**: `MarkdownExporter.export()` never received `agent_results`, so frontmatter metadata was `{}` and every summary was the hardcoded fallback. Rich structured outputs existed but were never extracted.

## Process lessons

- **Deterministic tests before LLM debugging.** The multi-expert analysis (SUB_AGENT_ANALYSIS_SYNTHESIS) converged on schema generation as root cause only after weeks; a direct OpenAI-schema-compatibility test suite (later built: 21 tests + pre-commit hook) would have caught it immediately. Mock-passing unit tests validated nothing about real API strictness.
- **Tests without implementation are debt in disguise**: a 295-line test file for `_ensure_default_factory_fields` was written for a method that was never implemented.
- **Debug scripts and analysis docs need a home policy, or they colonize the repo root.** That is why this file exists. Investigation notes → `.internal-docs/`; binding outcomes → this directory or ADRs; everything else gets archived or deleted at branch close.

## Open threads carried forward (unresolved as of archiving)

- Native `client.beta.chat.completions.parse()` intermittently returns None (all agents, GPT-5), silently falling back to LangChain `with_structured_output()` — never root-caused.
- Nested-model (`$defs`) required-array generation not conclusively verified fixed.
- Agents still return strings rather than Pydantic models in some paths (persistence fix incomplete end-to-end).
- Model selection: agents reportedly defaulted to GPT-5 regardless of configuration — never root-caused.
- Intermittent WebSocket drops (~20%) during long runs.
- Historian schema expects UUID source ids; code generates filename-style strings (`Aristotle_DeCaelo_BookII`) — latent mismatch.
- MyPy test-suite debt: 276 errors across 12 test files (51% missing annotations); phased fix plan estimated 17–20h (see archived MYPY_TEST_ERROR_CLUSTER_ANALYSIS.md).
