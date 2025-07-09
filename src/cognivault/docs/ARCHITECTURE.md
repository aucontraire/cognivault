# 🧠 CogniVault Architecture Overview

This document provides a high-level overview of the architectural design of the CogniVault system. It outlines the key components, design principles, and extension points to guide future development.

---

## 🌐 System Overview

CogniVault is a modular, agent-driven platform designed to process, refine, critique, and synthesize user input into structured knowledge. It emphasizes:

- **Agent-based modularity**
- **LLM backend flexibility**
- **CLI + programmatic usage**
- **Structured trace logging**
- **Testable + extensible design**

---

## 🧩 Key Components

### 1. **Dual Execution Orchestrators**
CogniVault provides two orchestration modes for maximum flexibility:

**AgentOrchestrator (Legacy)**
- Sequential/parallel execution with dependency management
- Retry logic, timeouts, and comprehensive tracing
- Backward compatibility with existing workflows

**RealLangGraphOrchestrator (Phase 2.1)**
- DAG-based execution with true parallel processing
- Concurrent state updates using Annotated types with operator.add reducers
- Advanced error recovery and circuit breaker patterns
- Parallel execution flow: Refiner → [Critic, Historian] → Synthesis

### 2. **Enhanced Agent System**
Four-agent pipeline with sophisticated LLM-powered capabilities:

- `RefinerAgent`: Query refinement and structured analysis using LLM
- `CriticAgent`: Critical evaluation and improvement suggestions
- `HistorianAgent`: **[Phase 2.1]** Intelligent historical context retrieval with multi-strategy search
- `SynthesisAgent`: Advanced thematic analysis and conflict resolution for wiki-ready output

**Key Phase 2.1 Enhancements:**
- Parallel execution of Critic and Historian agents for improved performance
- LangGraph-compatible concurrent state updates with partial state returns
- Circuit breaker patterns for enhanced resilience
- Comprehensive metadata tracking for all agent outputs

### 3. **Enhanced Context Management**
**AgentContext** - A Pydantic-based container with advanced state management:
- The user query and execution metadata
- Agent outputs with both legacy and LangGraph-compatible formats
- Config state and comprehensive tracing
- Size monitoring and compression for large contexts
- Snapshot/rollback capabilities for complex workflows

**CogniVaultState** - **[Phase 2.1]** TypedDict for LangGraph integration:
- Type-safe state schemas with mypy compliance
- Concurrent state updates using Annotated types with operator.add reducers
- Partial state returns for optimized LangGraph execution
- Comprehensive agent output schemas (RefinerOutput, CriticOutput, HistorianOutput, SynthesisOutput)

### 4. **LLM Interface**
A strategy abstraction defined by `LLMInterface`, implemented by:
- `OpenAIChatLLM`: Uses OpenAI’s chat API
- `StubLLM`: Returns canned outputs for testing

### 5. **Config**
- `OpenAIConfig`: Loads credentials and model info from `.env`
- `logging_config.py`: Standardized logger setup

### 6. **CLI**
The entrypoint for executing agent workflows via the command line.
Supports flags for agent selection, log level, and Markdown export.

### 7. **Production LangGraph Integration**
**[Phase 2.1]** Complete LangGraph 0.5.1 integration with real StateGraph orchestration:

**Core LangGraph Components:**
- `RealLangGraphOrchestrator`: Production StateGraph execution with memory checkpointing
- `CogniVaultState`: Type-safe TypedDict schemas with comprehensive agent output definitions
- `AgentContextStateBridge`: Seamless conversion between AgentContext and LangGraph state
- Node wrapper functions with circuit breaker patterns and comprehensive metrics

**Advanced Features:**
- **Concurrent State Updates**: Solved LangGraph INVALID_CONCURRENT_GRAPH_UPDATE using Annotated types with operator.add reducers
- **Partial State Returns**: Optimized node execution with partial state updates for parallel processing
- **DAG Visualization**: Mermaid diagram generation showing parallel execution structure
- **Performance Comparison**: Statistical benchmarking between legacy and LangGraph execution modes

**Phase 2.1 Execution Flow:**
```
START → Refiner → [Critic, Historian] → Synthesis → END
```
- Refiner processes the initial query
- Critic and Historian execute in parallel with concurrent state updates
- Synthesis integrates all outputs for final analysis

---

## 🏗️ Planned Abstractions

### 🔄 `LLMFactory` (Planned)
To support runtime selection of LLM providers, an `LLMFactory` will encapsulate logic for instantiating an `LLMInterface` based on env or user config.

### 📖 `ARCHITECTURE.md`
This document! Used to track high-level system boundaries and design intent.

---

## 📦 Directory Structure

```
cognivault/
├── agents/           # Agent implementations and entrypoints
├── config/           # Logging and LLM config
├── context.py        # AgentContext data container
├── llm/              # LLMInterface + concrete implementations
├── orchestrator.py   # Orchestration engine
├── cli.py            # Command-line entrypoint
├── langraph/         # LangGraph compatibility layer
│   ├── graph_builder.py  # DAG construction and validation
│   └── routing.py        # Graph execution and routing
├── store/            # Markdown export + future persistence
├── retrieval/        # Embedding + search layer (stub)
tests/
```

---

## ✅ Design Principles

- **Inversion of Control** via `AgentOrchestrator`
- **Loose coupling** between agents and LLMs
- **Extensibility** for new agents or output formats
- **Testability** through mocks and StubLLM
- **Environment-based configuration** for LLMs

---

## 📦 Dependencies

### Core Dependencies
- **Python**: 3.12+ (Development environment)
- **Pydantic**: v2.x (Data validation and serialization)
- **OpenAI**: 1.92.3+ (LLM API integration)
- **Typer**: CLI framework with rich output support

### LangGraph Integration
- **LangGraph**: 0.5.1 (Pinned for Phase 1 stability)
- **Purpose**: Real LangGraph DAG execution for production-ready orchestration
- **Compatibility**: Python >=3.9, Pydantic v2, async/await support
- **Status**: Phase 1 integration with `RealLangGraphOrchestrator`

#### LangGraph Version Rationale
LangGraph 0.5.1 was chosen for Phase 1 development because:
- **Stability**: Version 0.5.x is described as "Getting-Ready-for-1.0" with refined APIs
- **Feature Complete**: Includes StateGraph, conditional routing, and memory checkpointing
- **Python 3.12 Compatible**: Supports our development environment requirements
- **Pydantic v2 Support**: Compatible with our existing data validation system
- **Active Development**: Team is preparing for 1.0 release, indicating continued support

#### Key LangGraph Features Used
- **StateGraph**: Core DAG orchestration with typed state management
- **Conditional Routing**: Dynamic agent execution based on context state
- **Memory Checkpointing**: Persistent state for long-running workflows
- **Async Support**: Full async/await compatibility for CogniVault agents
- **Error Handling**: Robust error propagation and recovery mechanisms

---

## 📌 Status

- ✅ Complete agent system with Refiner, Critic, Historian, and Synthesis agents
- ✅ Full LLM abstraction with OpenAI and stub implementations
- ✅ Enterprise-grade error handling and agent resilience
- ✅ Advanced context management with snapshots and compression
- ✅ Comprehensive observability and diagnostics system
- ✅ **LangGraph Compatibility Layer** - Complete DAG-ready architecture
- ✅ **[Phase 2.1]** Production LangGraph Integration with StateGraph orchestration
- ✅ **[Phase 2.1]** Historian agent implementation with parallel execution
- ✅ **[Phase 2.1]** Concurrent state updates with Annotated types and operator.add reducers
- ✅ **[Phase 2.1]** Comprehensive test coverage (>90%) for all LangGraph components
- 🔜 Web frontend and storage persistence layer

---

## ✨ Future Ideas

- **LangGraph Migration**: Seamless transition from current orchestrator to LangGraph DAG execution
- **Advanced Node Types**: DECISION and conditional routing nodes for complex workflows
- **Multi-LLM Blending**: Parallel execution with different LLM providers
- **Plugin Architecture**: Community agent ecosystem with dynamic loading
- **Graph Visualization**: DAG visualization and execution flow debugging
- **Conditional Execution**: Smart routing based on agent outputs and context state
- **GitHub Copilot-style inline annotation**: IDE integration for agent suggestions

## 🔗 LangGraph Architecture Details

### Node Interface Design

All agents implement the `invoke(state, config)` interface required by LangGraph:

```python
async def invoke(self, state: AgentContext, config: Optional[Dict[str, Any]] = None) -> AgentContext:
    # LangGraph-compatible execution with configuration support
    return await self.run_with_retry(state, step_id=config.get("step_id"))
```

### Node Metadata System

Each agent provides comprehensive metadata for graph construction:

- **Node Type**: PROCESSOR, DECISION, TERMINATOR, AGGREGATOR
- **Input Schema**: Required and optional inputs with type hints
- **Output Schema**: Expected outputs with descriptions
- **Dependencies**: List of required upstream agents
- **Tags**: Categorization for filtering and organization

### Graph Builder Features

- **Dependency Resolution**: Automatic topological sorting using Kahn's algorithm
- **Cycle Detection**: DFS-based cycle detection prevents infinite loops
- **Edge Validation**: Comprehensive validation of all edge endpoints
- **Graph Serialization**: Dictionary representation for LangGraph integration

### Execution Engine

The GraphExecutor provides:

- **Dependency-Aware Execution**: Proper ordering based on agent dependencies
- **Parallel Processing**: Independent agents can execute concurrently
- **Error Isolation**: Agent failures don't cascade to other agents
- **Execution Metadata**: Complete trace of execution order and visited nodes

This architecture provides a solid foundation for LangGraph migration while maintaining full backward compatibility with existing workflows.
