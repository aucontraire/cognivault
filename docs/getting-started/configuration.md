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

### Database Connection

Both the dev and test databases run as Docker containers (`docker-compose.dev.yml`). Point
the app at any database with `DATABASE_URL` (and tests with `TEST_DATABASE_URL`) for full
control; otherwise these `POSTGRES_*` vars set the defaults for the containers **and** the
app, so they stay in sync:

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_DEV_PORT` | `5441` | Host port for the dev database container. |
| `POSTGRES_TEST_PORT` | `5440` | Host port for the test database container (tests refuse any other local DB). |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `cognivault` / `cognivault_dev` / `cognivault` | Credentials + name shared by the containers and the app default. |

Override a port when 5440/5441 are taken, e.g.
`POSTGRES_DEV_PORT=6441 docker compose -f docker-compose.dev.yml up postgres-dev -d`, and
keep the same value in the shell that runs the app/tests so they resolve to the same port.

Other environment variables — content to be added.

## Configuration Files

Content to be added.

## Advanced Configuration

Content to be added.
