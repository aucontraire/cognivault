# Database Exploration Guide

This guide shows you how to explore and query the CogniVault database to examine Pydantic AI integration data, agent performance metrics, and structured outputs.

## Quick Start

### Automated Database Exploration

```bash
# Quick database overview with recent data
make db-explore

# Check database status and tables
make db-status
```

### Direct Database Connection

```bash
# Connect to database (PostgreSQL 17)
PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" psql -d cognivault

# Or use the alias for convenience
alias psql-cv='PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" psql -d cognivault'
psql-cv
```

## Database Schema Overview

The CogniVault database contains several key tables:

```sql
-- List all tables
\dt

-- Table descriptions:
-- questions: User queries with execution metadata (JSONB)
-- topics: Hierarchical topic organization with vector embeddings
-- wiki_entries: Knowledge base entries with versioning
-- api_keys: API access control and usage tracking
-- semantic_links: Graph relationships between entities
-- alembic_version: Database migration tracking
```

## Exploring Pydantic AI Integration Data

### Basic Question Data

```sql
-- Recent questions with basic info
SELECT 
    id, 
    query, 
    correlation_id, 
    created_at 
FROM questions 
ORDER BY created_at DESC 
LIMIT 5;

-- Questions with execution metadata
SELECT 
    query,
    correlation_id,
    nodes_executed,
    created_at
FROM questions 
WHERE execution_metadata IS NOT NULL
ORDER BY created_at DESC;
```

### Structured Agent Outputs

```sql
-- Extract critic agent structured data
SELECT 
    execution_metadata->'agent_outputs'->'critic'->>'confidence' as confidence,
    execution_metadata->'agent_outputs'->'critic'->>'issues_detected' as issues,
    execution_metadata->'agent_outputs'->'critic'->>'processing_mode' as mode
FROM questions 
WHERE execution_metadata->'agent_outputs' ? 'critic';

-- Get full critique summary
SELECT 
    query,
    execution_metadata->'agent_outputs'->'critic'->>'critique_summary' as critique
FROM questions 
WHERE execution_metadata->'agent_outputs' ? 'critic';

-- Pretty-print complete structured output
SELECT 
    query,
    jsonb_pretty(execution_metadata->'agent_outputs'->'critic') as structured_output
FROM questions 
WHERE correlation_id = 'your-correlation-id-here';
```

### Agent Performance Analytics

```sql
-- Agent processing times and confidence levels
SELECT 
    execution_metadata->'agent_outputs'->'critic'->>'confidence' as confidence,
    execution_metadata->'agent_outputs'->'critic'->>'processing_time_ms' as processing_time,
    execution_metadata->'agent_outputs'->'critic'->>'processing_mode' as mode,
    created_at
FROM questions 
WHERE execution_metadata->'agent_outputs' ? 'critic'
ORDER BY created_at DESC;

-- Overall execution performance
SELECT 
    execution_metadata->>'total_execution_time_ms' as total_time,
    execution_metadata->>'total_tokens_used' as tokens,
    execution_metadata->>'total_cost_usd' as cost,
    execution_metadata->>'model_used' as model,
    execution_metadata->>'success' as success
FROM questions 
ORDER BY created_at DESC;

-- Confidence distribution
SELECT 
    execution_metadata->'agent_outputs'->'critic'->>'confidence' as confidence_level,
    COUNT(*) as count
FROM questions 
WHERE execution_metadata->'agent_outputs' ? 'critic'
GROUP BY confidence_level;
```

### Advanced JSONB Queries

```sql
-- Find questions with specific issues detected
SELECT 
    query,
    execution_metadata->'agent_outputs'->'critic'->>'issues_detected' as issues
FROM questions 
WHERE (execution_metadata->'agent_outputs'->'critic'->>'issues_detected')::int > 3;

-- Extract assumptions and biases
SELECT 
    query,
    execution_metadata->'agent_outputs'->'critic'->'assumptions' as assumptions,
    execution_metadata->'agent_outputs'->'critic'->'biases' as biases
FROM questions 
WHERE execution_metadata->'agent_outputs' ? 'critic';

-- Search critique summaries
SELECT 
    query,
    execution_metadata->'agent_outputs'->'critic'->>'critique_summary' as critique
FROM questions 
WHERE execution_metadata->'agent_outputs'->'critic'->>'critique_summary' ILIKE '%assumption%';

-- Performance benchmarking queries
SELECT 
    AVG((execution_metadata->'agent_outputs'->'critic'->>'processing_time_ms')::float) as avg_processing_time,
    MIN((execution_metadata->'agent_outputs'->'critic'->>'processing_time_ms')::float) as min_processing_time,
    MAX((execution_metadata->'agent_outputs'->'critic'->>'processing_time_ms')::float) as max_processing_time,
    COUNT(*) as total_executions
FROM questions 
WHERE execution_metadata->'agent_outputs' ? 'critic';
```

## Repository Helper Methods

CogniVault provides repository methods for common queries. These can be tested programmatically:

```python
from cognivault.database.repositories.question_repository import QuestionRepository

# Get questions by agent confidence
confidence_questions = await repo.get_questions_by_agent_confidence(
    agent_name="critic",
    confidence_level="high",
    limit=10
)

# Get questions with issues detected
issues_questions = await repo.get_questions_with_issues(
    min_issues=3,
    agent_name="critic"
)

# Get agent performance statistics
performance_stats = await repo.get_agent_performance_stats("critic")

# Get questions with structured outputs
structured_questions = await repo.get_questions_with_structured_outputs(limit=10)

# Get execution time statistics
time_stats = await repo.get_execution_time_statistics()
```

## Example Data Structure

Here's what a complete structured agent output looks like in the database:

```json
{
    "biases": ["temporal"],
    "timestamp": "2025-07-28T23:30:09.386639+00:00",
    "agent_name": "CriticAgent",
    "confidence": "high",
    "assumptions": [
        "Assumes that AI can fully perform all tasks done by human workers",
        "Assumes that the transition to AI will be complete within 5 years"
    ],
    "bias_details": {},
    "logical_gaps": [
        "Doesn't define what is meant by 'completely replace'",
        "Doesn't specify which industries or job types are being referred to"
    ],
    "issues_detected": 4,
    "no_issues_found": false,
    "processing_mode": "active",
    "critique_summary": "The query assumes that AI is capable of completely replacing human workers...",
    "alternate_framings": [
        "Will AI significantly affect job availability within the next 5 years?",
        "Which jobs are most at risk of being automated within the next 5 years?"
    ],
    "processing_time_ms": 8582.122802734375
}
```

## Useful Commands Reference

### Database Management
```bash
# Setup database with migrations
make db-setup

# Reset database completely
make db-reset

# Check database status
make db-status

# Explore recent data
make db-explore
```

### Direct PostgreSQL Access
```bash
# Connect to database
PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" psql -d cognivault

# Run single query
PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" psql -d cognivault -c "SELECT COUNT(*) FROM questions;"

# Create convenience alias
alias psql-cv='PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" psql -d cognivault'
```

### Testing Integration
```bash
# Run complete Pydantic AI integration test
make test-pydantic-ai

# Run all integration tests
make test-integration
```

## Troubleshooting

### Common Issues

**Database Connection Issues:**
```bash
# Check PostgreSQL is running
make db-status

# Restart PostgreSQL 17
brew services restart postgresql@17

# Verify database exists
make db-explore
```

**No Data Found:**
```bash
# Run a test to generate data
make test-pydantic-ai

# Check if tables exist
PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH" psql -d cognivault -c "\dt"
```

**JSONB Query Errors:**
```sql
-- Check if execution_metadata exists
SELECT COUNT(*) FROM questions WHERE execution_metadata IS NOT NULL;

-- Check agent outputs structure
SELECT jsonb_object_keys(execution_metadata->'agent_outputs') 
FROM questions 
WHERE execution_metadata->'agent_outputs' IS NOT NULL;
```

## Performance Considerations

### Indexing
The database includes optimized indexes for JSONB queries:
- GIN index on `execution_metadata` for fast JSONB queries
- B-tree indexes on correlation_id, execution_id, created_at
- Vector indexes for pgvector similarity search

### Query Optimization
```sql
-- Use JSONB operators for better performance
WHERE execution_metadata->'agent_outputs' ? 'critic'  -- Fast
WHERE execution_metadata->'agent_outputs'->'critic' IS NOT NULL  -- Slower

-- Use correlation_id for specific question lookups
WHERE correlation_id = 'specific-id'  -- Indexed, very fast

-- Limit results for large datasets
ORDER BY created_at DESC LIMIT 100
```

## Next Steps

- **API Integration**: Use these queries in FastAPI endpoints for dashboard views
- **Analytics Dashboard**: Build visualization tools using the performance data
- **Monitoring**: Set up alerts based on execution time and error rates
- **Data Export**: Create scripts to export structured data for analysis

For more information on the repository layer, see the [Database Architecture Documentation](ARCHITECTURE.md).