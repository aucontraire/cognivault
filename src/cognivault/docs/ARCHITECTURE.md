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

**RealLangGraphOrchestrator (Phase 2 Complete)**
- DAG-based execution with true parallel processing
- Concurrent state updates using Annotated types with operator.add reducers
- Advanced error recovery and circuit breaker patterns
- **[Phase 2]** Graph building logic extracted to dedicated `langgraph_backend/` module
- **[Phase 2]** GraphFactory with pattern support and intelligent caching
- **[Phase 2]** Multiple graph patterns (standard, parallel, conditional)
- Optional checkpointing and conversation persistence using LangGraph MemorySaver
- Thread-scoped memory management for multi-session workflows
- Parallel execution flow: Refiner → [Critic, Historian] → Synthesis

### 2. **Enhanced Agent System**
Four-agent pipeline with sophisticated LLM-powered capabilities:

- `RefinerAgent`: Query refinement and structured analysis using LLM
- `CriticAgent`: Critical evaluation and improvement suggestions
- `HistorianAgent`: **[Phase 2.1]** Intelligent historical context retrieval with multi-strategy search
- `SynthesisAgent`: Advanced thematic analysis and conflict resolution for wiki-ready output

**Key Phase 2 Enhancements:**
- **Graph Builder Extraction**: Dedicated `langgraph_backend/` module for clean separation of concerns
- **GraphFactory Architecture**: Centralized graph building with pattern support and caching
- **Performance Optimization**: LRU cache with TTL for compiled graphs (~90% improvement)
- **Pattern System**: Extensible graph patterns (standard, parallel, conditional)
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

**CogniVaultState** - TypedDict for LangGraph integration:
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
Complete LangGraph 0.5.1 integration with real StateGraph orchestration:

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

**Execution Flow:**
```
START → Refiner → [Critic, Historian] → Synthesis → END
```
- Refiner processes the initial query
- Critic and Historian execute in parallel with concurrent state updates
- Synthesis integrates all outputs for final analysis

### 8. **Phase 2: Graph Builder Architecture**
**[Phase 2]** Extracted graph building logic into dedicated backend module for improved architecture:

**Core Components:**
- `GraphFactory`: Centralized graph building and compilation with pattern support
- `GraphPattern`: Abstract base for extensible execution patterns (StandardPattern, ParallelPattern, ConditionalPattern)
- `GraphCache`: LRU cache with TTL for compiled graphs (~90% performance improvement)
- `PatternRegistry`: Centralized pattern management and registration

**Architecture Benefits:**
- **Separation of Concerns**: Graph building vs execution orchestration
- **Performance Optimization**: Intelligent caching reduces repeated compilation overhead
- **Enhanced Testability**: Graph building can be unit tested independently
- **Improved Maintainability**: Cleaner, more focused codebase
- **Extensibility**: Easy to add new graph patterns and execution modes

**Graph Factory Features:**
- Pattern-based graph construction with validation
- Thread-safe operations with comprehensive error handling
- Cache management with size estimation and TTL expiration
- Configuration-driven graph building (agents, checkpoints, patterns)
- Backward compatibility with existing orchestrator interfaces

### 9. **Memory Management & Checkpointing**
Advanced memory management and conversation persistence for long-running workflows:

**Core Components:**
- `CogniVaultMemoryManager`: LangGraph MemorySaver integration with thread ID scoping
- `ErrorPolicyManager`: Centralized retry logic, circuit breaker patterns, and fallback strategies
- `CheckpointConfig`: Configuration for optional checkpointing behavior with TTL management
- Enhanced CLI integration with `--enable-checkpoints`, `--thread-id`, and `--rollback-last-checkpoint` flags

**Memory Management Features:**
- **Thread-Scoped Persistence**: Multi-session conversation isolation using unique thread IDs
- **Optional Checkpointing**: Defaults to off for backward compatibility, can be enabled per execution
- **State Serialization**: Robust CogniVaultState persistence with comprehensive type handling
- **Rollback Mechanisms**: Failed execution recovery with checkpoint restoration
- **Cleanup Management**: TTL-based checkpoint expiration and memory optimization

**Error Policy Integration:**
- **Circuit Breaker Patterns**: Configurable failure thresholds per agent with recovery timeouts
- **Retry Strategies**: Exponential backoff, linear backoff, adaptive, and fixed interval delays
- **Error Classification**: Timeout, LLM, validation, resource exhaustion, network, and configuration errors
- **Centralized Configuration**: Agent-specific error policies with fallback chains

**Enhanced DAG Visualization:**
- Checkpoint-enabled nodes with 💾 indicators
- Error handling routes with 🔌 circuit breaker symbols
- Memory state visualization and thread ID tracking

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
│   ├── graph_builder.py     # DAG construction and validation
│   ├── routing.py           # Graph execution and routing
│   ├── memory_manager.py    # LangGraph MemorySaver integration with thread ID scoping
│   └── error_policies.py    # Centralized retry logic and circuit breaker patterns
├── langgraph_backend/ # **[Phase 2]** Dedicated graph building backend
│   ├── build_graph.py       # GraphFactory and core building logic
│   ├── graph_patterns.py    # Pattern definitions and registry
│   └── graph_cache.py       # LRU cache with TTL support
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
- ✅ **[Phase 2]** Graph builder extraction to dedicated `langgraph_backend/` module
- ✅ **[Phase 2]** GraphFactory with pattern support and intelligent caching
- ✅ **[Phase 2]** Performance optimization with 90% improvement for repeated builds
- ✅ **[Phase 2]** Multiple graph patterns (standard, parallel, conditional)
- ✅ Production LangGraph Integration with StateGraph orchestration
- ✅ Historian agent implementation with parallel execution
- ✅ Concurrent state updates with Annotated types and operator.add reducers
- ✅ Comprehensive test coverage (>90%) for all LangGraph components
- ✅ LangGraph MemorySaver integration with optional checkpointing
- ✅ Thread-scoped memory management for multi-session workflows
- ✅ Centralized error policies with circuit breakers and retry strategies
- ✅ Enhanced DAG visualization with checkpoint and error policy indicators
- 🔜 Web frontend and storage persistence layer

---

## ✨ Future Ideas

- ✅ **LangGraph Migration**: Seamless transition from current orchestrator to LangGraph DAG execution *(Completed)*
- ✅ **Graph Builder Extraction**: Clean separation of graph building from orchestration *(Completed in Phase 2)*
- ✅ **Performance Optimization**: Intelligent caching and pattern-based graph construction *(Completed in Phase 2)*
- ✅ **Graph Visualization**: DAG visualization and execution flow debugging *(Completed)*
- ✅ **Conditional Execution**: Smart routing based on agent outputs and context state *(Completed)*
- **Advanced Node Types**: DECISION and conditional routing nodes for complex workflows
- **Multi-LLM Blending**: Parallel execution with different LLM providers
- **Plugin Architecture**: Community agent ecosystem with dynamic loading
- **Persistent Storage Layer**: Database integration for long-term conversation history
- **Web Frontend**: Interactive UI for visual DAG editing and execution monitoring
- **GitHub Copilot-style inline annotation**: IDE integration for agent suggestions
- **Multi-Session DAG Workflows**: Complex workflows spanning multiple conversation sessions *(Foundation completed)*

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
