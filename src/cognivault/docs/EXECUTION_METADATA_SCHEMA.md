# Execution Metadata Schema Documentation

This document describes the structured execution metadata schema used in CogniVault's PostgreSQL JSONB `execution_metadata` field, powered by Pydantic AI for validated agent outputs.

> **📡 Real-time Events**: For real-time monitoring of workflow execution, see [WebSocket Event Streams Documentation](./WEBSOCKET_EVENT_STREAMS.md) which provides dual-level event granularity from both orchestration and agent execution sources.

## Overview

The `execution_metadata` field in the `questions` table stores structured information about workflow executions using Pydantic models for validation and consistency. This approach provides:

- **Type Safety**: Pydantic validation ensures consistent data structures
- **Agent Flexibility**: New agents can define their own output schemas
- **Queryable Data**: Structured JSONB enables efficient PostgreSQL queries
- **Observability**: Rich metadata for monitoring and analytics

## Schema Structure

### ExecutionMetadata (Root Schema)

```python
{
  "execution_id": "exec_123abc",
  "correlation_id": "corr_456def", 
  "total_execution_time_ms": 3500.0,
  "nodes_executed": ["refiner", "critic", "historian", "synthesis"],
  "parallel_execution": false,
  
  "agent_outputs": {
    "refiner": { ... },      # RefinerOutput model
    "critic": { ... },       # CriticOutput model  
    "historian": { ... },    # HistorianOutput model
    "synthesis": { ... }     # SynthesisOutput model
  },
  
  "total_tokens_used": 2500,
  "total_cost_usd": 0.05,
  "model_used": "gpt-4",
  "errors_encountered": [],
  "retries_attempted": 0,
  "workflow_version": "1.0",
  "success": true
}
```

## Agent Output Models

### BaseAgentOutput (Common Fields)

All agent outputs inherit from `BaseAgentOutput`:

```python
{
  "agent_name": "agent_name",
  "processing_mode": "active|passive|fallback", 
  "confidence": "high|medium|low",
  "processing_time_ms": 1200.5,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### RefinerOutput

```python
{
  "agent_name": "refiner",
  "processing_mode": "active",
  "confidence": "high",
  "processing_time_ms": 800.2,
  "timestamp": "2025-01-15T10:30:00Z",
  
  # Refiner-specific fields
  "refined_query": "What are the potential impacts of AI on employment?",
  "original_query": "AI jobs",
  "changes_made": [
    "Clarified scope to include positive and negative impacts",
    "Specified domain as employment sector"
  ],
  "was_unchanged": false,
  "fallback_used": false, 
  "ambiguities_resolved": ["Unclear scope of 'AI jobs'"]
}
```

### CriticOutput

```python
{
  "agent_name": "critic",
  "processing_mode": "active",
  "confidence": "medium", 
  "processing_time_ms": 950.1,
  "timestamp": "2025-01-15T10:30:01Z",
  
  # Critic-specific fields
  "assumptions": ["Assumes AI adoption will continue at current pace"],
  "logical_gaps": ["No timeframe specified for impact assessment"],
  "biases": ["temporal", "scale"],
  "bias_details": {
    "temporal": "Assumes current AI trajectory will continue",
    "scale": "Focuses on individual job impact vs systemic changes"
  },
  "alternate_framings": [
    "Consider both displacement and job creation",
    "Examine sector-specific impacts separately"
  ],
  "critique_summary": "Query needs timeframe and scope clarification",
  "issues_detected": 3,
  "no_issues_found": false
}
```

### HistorianOutput

```python
{
  "agent_name": "historian", 
  "processing_mode": "active",
  "confidence": "high",
  "processing_time_ms": 1200.8,
  "timestamp": "2025-01-15T10:30:02Z",
  
  # Historian-specific fields
  "relevant_sources": [
    {
      "source_id": "uuid-123",
      "title": "Industrial Revolution Employment Patterns", 
      "relevance_score": 0.85,
      "content_snippet": "Historical analysis shows..."
    }
  ],
  "historical_synthesis": "Historical precedent shows technology adoption cycles...",
  "themes_identified": ["Technology adoption cycles", "Labor market adaptation"],
  "time_periods_covered": ["Industrial Revolution", "Digital Revolution"],
  "contextual_connections": ["Similar resistance patterns", "Adaptation timeframes"],
  "sources_searched": 25,
  "relevant_sources_found": 8,
  "no_relevant_context": false
}
```

### SynthesisOutput

```python
{
  "agent_name": "synthesis",
  "processing_mode": "active", 
  "confidence": "high",
  "processing_time_ms": 1500.3,
  "timestamp": "2025-01-15T10:30:03Z",
  
  # Synthesis-specific fields
  "final_synthesis": "# AI and Employment Impact\n\nArtificial intelligence...",
  "key_themes": [
    {
      "theme_name": "Job Displacement vs Creation", 
      "description": "Balance between job losses and new opportunities",
      "supporting_agents": ["refiner", "critic", "historian"],
      "confidence": "high"
    }
  ],
  "conflicts_resolved": ["Disagreement on timeline between agents"],
  "complementary_insights": ["Historical context supports current trends"],
  "knowledge_gaps": ["Limited data on emerging job categories"],
  "meta_insights": ["Pattern recognition across historical cycles"],
  "contributing_agents": ["refiner", "critic", "historian"], 
  "word_count": 580,
  "topics_extracted": ["artificial intelligence", "employment", "automation"]
}
```

## PostgreSQL Query Examples

### Find Questions by Agent Confidence Level

```sql
SELECT id, query, execution_metadata->'agent_outputs'->'critic'->>'confidence' as critic_confidence
FROM questions 
WHERE execution_metadata->'agent_outputs'->'critic'->>'confidence' = 'high';
```

### Find Questions with Specific Issues Detected

```sql
SELECT id, query, (execution_metadata->'agent_outputs'->'critic'->>'issues_detected')::int as issues
FROM questions
WHERE (execution_metadata->'agent_outputs'->'critic'->>'issues_detected')::int > 2;
```

### Average Processing Time by Agent

```sql
SELECT 
  'refiner' as agent,
  AVG((execution_metadata->'agent_outputs'->'refiner'->>'processing_time_ms')::float) as avg_time_ms
FROM questions WHERE execution_metadata->'agent_outputs' ? 'refiner'

UNION ALL

SELECT 
  'critic' as agent, 
  AVG((execution_metadata->'agent_outputs'->'critic'->>'processing_time_ms')::float) as avg_time_ms
FROM questions WHERE execution_metadata->'agent_outputs' ? 'critic';
```

### Find Questions by Processing Mode

```sql
SELECT id, query
FROM questions
WHERE execution_metadata->'agent_outputs'->'refiner'->>'processing_mode' = 'fallback';
```

### Agent Performance Analytics

```sql
SELECT
  COUNT(*) as total_executions,
  AVG((execution_metadata->>'total_execution_time_ms')::float) as avg_total_time,
  AVG((execution_metadata->>'total_tokens_used')::int) as avg_tokens,
  AVG((execution_metadata->>'total_cost_usd')::float) as avg_cost
FROM questions 
WHERE execution_metadata->>'success' = 'true';
```

## Repository Helper Methods

The `QuestionRepository` provides helper methods for common queries:

```python
# Get questions by agent confidence
questions = await repo.get_questions_by_agent_confidence("critic", "high")

# Get agent performance statistics  
stats = await repo.get_agent_performance_stats("refiner")

# Find questions with processing issues
problematic = await repo.get_questions_with_issues(min_issues=3)

# Get execution time analytics
time_stats = await repo.get_execution_time_statistics()
```

## Migration and Backwards Compatibility

The structured schema is designed to be backwards compatible:

1. **Existing Data**: Old unstructured metadata continues to work
2. **Gradual Migration**: Agents can be updated one at a time
3. **Fallback Support**: Structured agents fallback to unstructured if needed
4. **Mixed Workflows**: Structured and unstructured agents can coexist

## Usage Examples

### Using Structured Agents

```python
# Use structured critic agent
critic = CriticAgent(llm, config)
context = await critic.run_structured(context)

# Access structured output
structured_output = context.structured_outputs["critic"]
print(f"Issues detected: {structured_output.issues_detected}")
print(f"Confidence: {structured_output.confidence}")
```

### Storing in Database

```python
# The ExecutionMetadata is automatically serialized to JSONB
execution_metadata = ExecutionMetadata(
    execution_id="exec_123",
    total_execution_time_ms=3500.0,
    nodes_executed=["refiner", "critic"],
    agent_outputs={
        "critic": critic_output  # Pydantic model
    },
    success=True
)

# Store in database
question = await repo.create_question(
    query="What is AI?",
    execution_metadata=execution_metadata.model_dump()  # Convert to dict
)
```

## Benefits

1. **Type Safety**: Compile-time validation of agent outputs
2. **Consistency**: Standardized structure across all agents  
3. **Queryability**: Efficient PostgreSQL queries on structured data
4. **Observability**: Rich metadata for monitoring and debugging
5. **Analytics**: Easy extraction of performance metrics
6. **Extensibility**: New agents can define custom output schemas
7. **Backwards Compatibility**: Gradual migration from unstructured data

This structured approach provides a robust foundation for agent swapping, analytics, and observability while maintaining the flexibility needed for evolving agent architectures.