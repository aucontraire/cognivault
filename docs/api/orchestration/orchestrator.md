# Orchestrator

The LangGraph-based orchestrator coordinates the execution of CogniVault's multi-agent cognitive pipeline.

## Overview

The `LangGraphOrchestrator` class:

- Orchestrates the 4-agent cognitive pipeline (Refiner → Historian → Critic → Synthesis)
- Manages workflow state transitions using TypedDict schemas
- Provides event-driven observability through EventEmitter integration
- Handles error recovery and timeout management
- Supports configurable agent behaviors via YAML configuration

The orchestrator builds a directed acyclic graph (DAG) using LangGraph's StateGraph and executes agents in the optimal order based on data dependencies.

## Key Features

- **LangGraph Integration**: Uses LangGraph's StateGraph for workflow orchestration
- **State Management**: TypedDict-based state schemas for type safety and clarity
- **Event Emission**: Comprehensive event emission for workflow observability
- **Error Handling**: Graceful error recovery with configurable retry logic
- **Async Execution**: Pure asyncio architecture for concurrent agent execution

## API Reference

::: cognivault.orchestration.orchestrator.LangGraphOrchestrator
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - run
          - build_graph
