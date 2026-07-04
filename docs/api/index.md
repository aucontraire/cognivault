# API Reference

Complete API documentation for CogniVault, auto-generated from Python docstrings.

## Overview

CogniVault provides a comprehensive API for building multi-agent cognitive workflows with LangGraph orchestration, persistent storage, and event-driven observability.

## Core Components

### Agents
CogniVault's cognitive agents process user queries through a multi-stage pipeline:

- **[BaseAgent](agents/base.md)** - Abstract base class for all cognitive agents
- **[RefinerAgent](agents/refiner.md)** - Query refinement and clarification
- **[HistorianAgent](agents/historian.md)** - Historical context retrieval
- **[CriticAgent](agents/critic.md)** - Critical analysis and perspective generation
- **[SynthesisAgent](agents/synthesis.md)** - Multi-perspective synthesis

### Orchestration
LangGraph-based workflow orchestration:

- **[Orchestrator](orchestration/orchestrator.md)** - Main workflow orchestrator
- **[State Schemas](orchestration/state.md)** - TypedDict state definitions for workflow

### Database
Persistent storage with PostgreSQL and pgvector:

- **[Repositories](database/repository.md)** - Database access layer
- **[Wiki Adapter](database/wiki.md)** - Markdown export functionality

### Services
Core platform services:

- **[LangChain Service](services/langchain.md)** - LLM integration and structured output
- **[Event System](services/events.md)** - Event-driven observability

## Usage

All classes are accessible through their respective modules. For example:

```python
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.orchestration.orchestrator import LangGraphOrchestrator
from cognivault.services.langchain_service import LangChainService
```

## Auto-Generated Documentation

All API documentation is automatically generated from Python docstrings using mkdocstrings. To contribute to the documentation, update the docstrings in the source code.
