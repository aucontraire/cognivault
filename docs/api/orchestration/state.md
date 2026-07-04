# Workflow State Schemas

TypedDict-based state schemas that define the data structures passed between agents in CogniVault's workflow orchestration.

## Overview

CogniVault uses TypedDict classes to define strongly-typed state schemas for each agent and the overall workflow. This approach provides:

- **Type Safety**: Static type checking with MyPy for state transitions
- **Clear Contracts**: Explicit definition of data flowing between agents
- **LangGraph Compatibility**: Native integration with LangGraph's StateGraph
- **Documentation**: Self-documenting state structures with type annotations

## State Schema Hierarchy

Each agent has its own state schema (e.g., `RefinerState`, `CriticState`, `HistorianState`, `SynthesisState`) that defines the specific inputs and outputs for that agent. These are aggregated into the `CogniVaultState` schema that represents the complete workflow state.

## Key State Schemas

### Agent-Specific States

- **RefinerState**: Query refinement inputs and outputs
- **CriticState**: Critical analysis perspectives and biases
- **HistorianState**: Historical context retrieval results
- **SynthesisState**: Final synthesis outputs and themes

### Workflow State

- **CogniVaultState**: Complete workflow state aggregating all agent states
- **ExecutionMetadata**: Workflow execution tracking and correlation
- **CogniVaultContext**: Runtime context and configuration

## API Reference

### Refiner State

::: cognivault.orchestration.state_schemas.RefinerState
    options:
        show_source: true
        heading_level: 4

### Critic State

::: cognivault.orchestration.state_schemas.CriticState
    options:
        show_source: true
        heading_level: 4

### Historian State

::: cognivault.orchestration.state_schemas.HistorianState
    options:
        show_source: true
        heading_level: 4

### Synthesis State

::: cognivault.orchestration.state_schemas.SynthesisState
    options:
        show_source: true
        heading_level: 4

### Complete Workflow State

::: cognivault.orchestration.state_schemas.CogniVaultState
    options:
        show_source: true
        heading_level: 4

### Execution Metadata

::: cognivault.orchestration.state_schemas.ExecutionMetadata
    options:
        show_source: true
        heading_level: 4

### Workflow Context

::: cognivault.orchestration.state_schemas.CogniVaultContext
    options:
        show_source: true
        heading_level: 4
