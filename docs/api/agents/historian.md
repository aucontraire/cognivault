# Historian Agent

The Historian Agent retrieves relevant historical context from CogniVault's knowledge base to inform the current analysis.

## Overview

The Historian Agent:

- Performs hybrid search combining semantic similarity and keyword matching
- Retrieves relevant prior analyses and insights
- Applies LLM-based relevance filtering as a safeguard
- Generates contextual summaries of historical knowledge
- Manages search analytics and document persistence

This agent uses PostgreSQL with pgvector for efficient semantic search and includes configurable hybrid search capabilities.

## API Reference

::: cognivault.agents.historian.agent.HistorianAgent
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - execute
