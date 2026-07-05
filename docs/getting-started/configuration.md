# Configuration

Configure CogniVault for your needs.

## Environment Variables

### Knowledge Persistence & Semantic Retrieval

Persistence and semantic search are **opt-in** and default off; a merely-reachable database
never triggers writes.

| Variable | Default | Purpose |
|----------|---------|---------|
| `KNOWLEDGE_PERSISTENCE_ENABLED` | `false` | Persist each run's question, topics, and synthesis to Postgres (explicit opt-in). |
| `KNOWLEDGE_TOPIC_MIN_CONFIDENCE` | `0.7` | Minimum confidence for a suggested topic to be persisted. |
| `KNOWLEDGE_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model for topic vectors. |
| `KNOWLEDGE_EMBEDDING_DIMENSIONS` | `1536` | Embedding width; must equal the `topics.embedding` column dimension. |
| `KNOWLEDGE_MAX_EMBEDDING_CALLS_PER_RUN` | `8` | Cap on new-topic embeddings generated per run (excess deferred to backfill). |
| `HISTORIAN_SEMANTIC_SEARCH_ENABLED` | `false` | Blend topic-embedding semantic retrieval into the Historian's hybrid search. |
| `HISTORIAN_SEMANTIC_SEARCH_WEIGHT` | `0.5` | Share of the result budget given to semantic hits when enabled. |

Persistence and semantic search both require `OPENAI_API_KEY` for embeddings and a
PostgreSQL database with the pgvector schema. All paths degrade gracefully: with the flags
off (or the key/database absent) the pipeline behaves exactly as before. Backfill missing
embeddings with `cognivault knowledge backfill-embeddings [--dry-run] [--limit N]`. See the
[Historian Hybrid Search guide](../agents/historian-hybrid-search.md#semantic-search-optional-tier)
for the semantic tier.

Other environment variables — content to be added.

## Configuration Files

Content to be added.

## Advanced Configuration

Content to be added.
