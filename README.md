# 🧠 CogniVault

![Python](https://img.shields.io/badge/python-3.12-blue)
![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Markdown Export](https://img.shields.io/badge/markdown-export-green)
![Wiki Ready](https://img.shields.io/badge/wiki-ready-blueviolet)

CogniVault is a modular, CLI-based multi-agent assistant designed to help you reflect, refine, and organize your thoughts through structured dialogue and cumulative insight. It simulates a memory-augmented thinking partner, enabling long-term knowledge building across multiple agent perspectives.


## ⚡ Quickstart

Clone the repo and run a basic question through the CLI:

```bash
git clone https://github.com/aucontraire/cognivault.git
cd cognivault
bash setup.sh
make run QUESTION="What are the long-term effects of AI in education?"
```

See [🖥️ Usage](#️usage) for running specific agents and debugging options.

---

## 🚀 Features

- ✅ **Fully working CLI** using [Typer](https://typer.tiangolo.com/)
- 🧠 **Multi-agent orchestration**: Refiner, Historian, Critic, Synthesis
- 🔁 **LangGraph-powered orchestration**: Production DAG-based execution (legacy mode deprecated and will be removed)
- 💾 **Checkpointing & Persistence**: LangGraph MemorySaver integration with conversation rollback
- 🔄 **Thread-Scoped Memory**: Multi-session conversation management with unique thread IDs
- 🛡️ **Centralized Error Policies**: Circuit breakers, retry logic, and fallback strategies per agent
- 📄 **Markdown-ready output** for integration with personal wikis
- 🧪 **Full test suite** with `pytest` for all core components (89% coverage with 1,600+ tests)
- 🔄 **Swappable LLM backend**: Plug-and-play support for OpenAI or stubs via configuration
- 📋 **Agent Registry**: Dynamic agent registration system for extensible architecture
- ⚙️ **Configuration Management**: Centralized configuration system with environment variables and JSON file support
- 🧠 **Enhanced Context Management**: Advanced memory management with compression, snapshots, and size monitoring
- 🛡️ **Enterprise Error Handling**: Comprehensive exception hierarchy with LangGraph-ready agent isolation
- 🔄 **Agent-Level Resilience**: Circuit breakers, retry policies, and timeout management per agent
- 📊 **Execution Tracing**: Structured metadata and trace logging for debugging and observability
- 🔀 **Failure Propagation**: LangGraph-compatible conditional execution with graceful degradation strategies
- 🏥 **Health Checks**: Agent validation system with dependency resolution and circular dependency detection
- 📊 **CLI Observability**: Rich diagnostics interface with `cognivault diagnostics` subcommands for health, metrics, and system status
- 🔍 **Execution Tracing**: Real-time pipeline execution traces with `--trace` flag, showing timing, dependencies, and routing decisions
- 🩺 **Health Checks**: Pre-execution validation with `--health-check` flag for agent readiness and configuration verification
- 🧪 **Dry Run Mode**: Pipeline validation without execution using `--dry-run` flag for configuration testing
- 📤 **Trace Export**: JSON export of detailed execution metadata with `--export-trace` for monitoring and analysis
- 🔍 **Structured Logging**: Enterprise-grade logging with automatic correlation ID tracking and context propagation
- 📈 **Performance Metrics**: Real-time collection of execution statistics, token usage, and success rates
- 🖥️ **Machine-Readable Output**: Multiple export formats (JSON, CSV, Prometheus, InfluxDB) for monitoring integration
- 🔗 **LangGraph Compatibility**: Complete DAG-ready architecture with node interfaces, graph builders, and execution engines
- 🧩 **Node Metadata System**: Comprehensive agent metadata with input/output schemas and dependency declarations
- 📊 **Graph Infrastructure**: DAG validation, cycle detection, edge routing, and execution ordering for LangGraph integration
- 🎯 **Production LangGraph Integration**: Real StateGraph orchestration with `--execution-mode=langgraph-real`
- 🌊 **TypedDict State Management**: Type-safe state schemas for all agent outputs with mypy compliance
- 🔧 **Circuit Breaker Node Patterns**: Robust node execution with configurable failure thresholds
- 📊 **DAG Visualization**: Mermaid diagram generation with `--visualize-dag` for pipeline analysis
- 🔀 **Performance Comparison**: Statistical benchmarking between execution modes with `--compare-modes`
- 🧾 **Historian Integration**: Parallel execution with intelligent historical context retrieval
- 🌊 **Concurrent State Updates**: LangGraph-compatible partial state updates with operator.add reducers

---

## 🧱 Project Structure

```
src/
├── cognivault/
│   ├── agents/
│   │   ├── critic/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── main.py
│   │   │   └── prompts.py
│   │   ├── historian/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   └── main.py
│   │   ├── refiner/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── main.py
│   │   │   └── prompts.py
│   │   ├── synthesis/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   └── main.py
│   │   ├── base_agent.py
│   │   └── registry.py
│   ├── cli.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── app_config.py
│   │   ├── logging_config.py
│   │   └── openai_config.py
│   ├── context.py
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── diagnostics.py
│   │   ├── formatters.py
│   │   ├── health.py
│   │   ├── metrics.py
│   │   └── visualize_dag.py
│   ├── docs/
│   │   ├── ARCHITECTURE.md
│   │   ├── FEEDBACK.md
│   │   ├── LANDSCAPE.md
│   │   ├── OBSERVABILITY.md
│   │   └── RESEARCH.md
│   ├── langraph/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── error_policies.py
│   │   ├── graph_builder.py
│   │   ├── langgraph_install.py
│   │   ├── memory_manager.py
│   │   ├── node_wrappers.py
│   │   ├── orchestrator.py
│   │   ├── prototype_dag.py
│   │   ├── real_orchestrator.py
│   │   ├── routing.py
│   │   ├── state_bridge.py
│   │   └── state_schemas.py
│   ├── langgraph_backend/
│   │   ├── __init__.py
│   │   ├── build_graph.py
│   │   ├── graph_cache.py
│   │   └── graph_patterns.py
│   ├── exceptions/
│   │   ├── __init__.py
│   │   ├── agent_errors.py
│   │   ├── config_errors.py
│   │   ├── io_errors.py
│   │   ├── llm_errors.py
│   │   └── orchestration_errors.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── llm_interface.py
│   │   ├── openai.py
│   │   ├── provider_enum.py
│   │   └── stub.py
│   ├── logs/
│   │   └── interaction_00001.json
│   ├── notes/
│   │   ├── 2025-06-26T06-45-24_what-is-cognition.md
│   │   ├── 2025-06-26T06-47-28_what-is-cognition.md
│   │   ├── 2025-06-26T10-04-47_what-is-cognition.md
│   │   └── sample_note.md
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── context.py
│   │   ├── formatters.py
│   │   └── logger.py
│   ├── orchestrator.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── vector_store.py
│   ├── store/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   └── wiki_adapter.py
│   └── utils/
│       ├── __init__.py
│       └── versioning.py
tests/
├── agents/
│   ├── critic/
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   ├── test_main.py
│   │   └── test_prompts.py
│   ├── historian/
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   └── test_main.py
│   ├── refiner/
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   └── test_main.py
│   ├── synthesis/
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   └── test_main.py
│   ├── test_base_agent.py
│   └── test_registry.py
├── config/
│   ├── __init__.py
│   ├── test_app_config.py
│   └── test_openai_config.py
├── diagnostics/
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_diagnostics.py
│   ├── test_formatters.py
│   ├── test_health.py
│   ├── test_metrics.py
│   └── test_visualize_dag.py
├── exceptions/
│   ├── __init__.py
│   ├── test_agent_errors.py
│   ├── test_base_exceptions.py
│   ├── test_config_errors.py
│   ├── test_io_errors.py
│   ├── test_llm_errors.py
│   ├── test_llm_errors_fixed.py
│   ├── test_orchestration_errors.py
│   └── test_orchestration_errors_fixed.py
├── langraph/
│   ├── __init__.py
│   ├── test_adapter.py
│   ├── test_graph_builder.py
│   ├── test_langgraph_install.py
│   ├── test_node_wrappers.py
│   ├── test_orchestrator.py
│   ├── test_prototype_dag.py
│   ├── test_real_orchestrator.py
│   ├── test_routing.py
│   ├── test_state_bridge.py
│   └── test_state_schemas.py
├── llm/
│   ├── __init__.py
│   ├── test_factory.py
│   ├── test_llm_interface.py
│   ├── test_openai.py
│   └── test_stub.py
├── observability/
│   ├── __init__.py
│   ├── test_context.py
│   ├── test_formatters.py
│   └── test_logger.py
├── store/
│   ├── __init__.py
│   ├── test_utils.py
│   └── test_wiki_adapter.py
├── utils/
│   ├── __init__.py
│   └── test_versioning.py
├── cli/
│   ├── __init__.py
│   ├── test_cli_langgraph_integration.py
│   └── test_cli_langgraph_real.py
├── test_cli.py
├── test_context.py
├── test_context_enhanced.py
└── test_orchestrator.py
```

---

## 🧠 Agent Roles

Each agent in CogniVault plays a distinct role in the cognitive reflection and synthesis pipeline:

- ### 🔍 Refiner
  The **RefinerAgent** takes the initial user input and clarifies intent, rephrases vague language, and ensures the prompt is structured for deeper analysis by the rest of the system. It uses a comprehensive system prompt with passive and active modes to guide its reasoning process. See [`prompts.py`](./src/cognivault/agents/refiner/prompts.py) for implementation details.

- ### 🧾 Historian
  The **HistorianAgent** provides relevant context from previous conversations or memory. It simulates long-term knowledge by surfacing pertinent background or earlier reflections.

- ### 🧠 Critic
  The **CriticAgent** evaluates the refined input or historical perspective. It identifies assumptions, weaknesses, or inconsistencies—acting as a thoughtful devil’s advocate.

- ### 🧵 Synthesis
  The **SynthesisAgent** gathers the outputs of the other agents and composes a final, unified response. This synthesis is designed to be insightful, coherent, and markdown-friendly for knowledge wikis or future reflection.

### 📋 Agent Registry

The **Agent Registry** provides a centralized system for managing agent types, dependencies, and creation logic. It enables dynamic agent loading while maintaining type safety and proper dependency injection. Key features include:

- **Dynamic Registration**: Register new agents programmatically with metadata
- **Dependency Tracking**: Define agent dependencies for proper execution order
- **LLM Interface Management**: Automatically handles LLM requirement validation
- **Pipeline Validation**: Validates agent pipelines before execution
- **Extensible Architecture**: Prepared for future LangGraph integration

The registry supports both the current architecture and future dynamic loading capabilities, featuring:

- **Dependency Resolution**: Automatic topological sorting of agent execution order using Kahn's algorithm
- **Failure Strategies**: Per-agent failure propagation policies (FAIL_FAST, WARN_CONTINUE, CONDITIONAL_FALLBACK, GRACEFUL_DEGRADATION)
- **Health Checks**: Agent validation system with configurable health check functions
- **Critical Agent Classification**: Distinguish between critical and optional agents for graceful degradation
- **Fallback Agent Support**: Alternative agent execution paths for failure scenarios

See [`registry.py`](./src/cognivault/agents/registry.py) for implementation details.

### 🧠 Enhanced Context Management

CogniVault features enterprise-grade context management designed to prevent memory bloat and provide robust state management for long-running agent conversations. Key features include:

- **Automatic Size Monitoring**: Real-time tracking of context size with configurable limits
- **Smart Compression**: Automatic gzip compression and content truncation when size limits are exceeded
- **Context Snapshots**: Create immutable snapshots of context state for rollback capabilities
- **Memory Optimization**: Intelligent cleanup of old data while preserving essential information
- **Parallel Processing Support**: Context cloning for safe concurrent agent execution

#### Context Management Features

The enhanced context system provides several key capabilities:

```python
from cognivault.context import AgentContext

# Create context with automatic size monitoring
context = AgentContext(query="What is AI safety?")

# Create a snapshot for later rollback
snapshot_id = context.create_snapshot(label="before_refinement")

# Add agent outputs (automatically monitored for size)
context.add_agent_output("refiner", "Refined query about AI safety...")

# Get memory usage statistics
usage = context.get_memory_usage()
print(f"Total size: {usage['total_size_bytes']} bytes")
print(f"Snapshots: {usage['snapshots_count']}")

# Optimize memory if needed
stats = context.optimize_memory()
print(f"Size reduced by {stats['size_reduction_bytes']} bytes")

# Restore from snapshot if needed
context.restore_snapshot(snapshot_id)

# Clone for parallel processing
cloned_context = context.clone()
```

#### Configurable Context Settings

Context management behavior can be configured via environment variables:

```env
# Context size limits (default: 1MB)
COGNIVAULT_MAX_CONTEXT_SIZE_BYTES=1048576

# Maximum snapshots to keep (default: 5)
COGNIVAULT_MAX_SNAPSHOTS=5

# Enable automatic compression (default: true)
COGNIVAULT_ENABLE_CONTEXT_COMPRESSION=true

# Compression threshold (default: 0.8 = 80% of max size)
COGNIVAULT_CONTEXT_COMPRESSION_THRESHOLD=0.8
```

The context management system automatically:
- Monitors context size during agent operations
- Applies compression when size limits are exceeded
- Truncates large outputs intelligently
- Maintains agent trace history with size limits
- Provides detailed memory usage statistics

This ensures CogniVault can handle long-running conversations and complex multi-agent workflows without memory issues, making it suitable for production deployments and extended research sessions.

### 🔗 LangGraph Compatibility Layer

CogniVault features a complete LangGraph compatibility layer that provides DAG-ready architecture while maintaining full backward compatibility with existing workflows. This foundation enables seamless future migration to LangGraph-based orchestration.

#### LangGraph Node Interface

All agents implement the standard LangGraph node interface:

```python
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.context import AgentContext

agent = RefinerAgent(llm)
context = AgentContext(query="Your question")

# LangGraph-compatible interface
result = await agent.invoke(context, config={"step_id": "custom_id"})

# Traditional interface still works
result = await agent.run_with_retry(context)
```

#### Node Metadata System

Each agent provides comprehensive metadata for graph construction:

```python
# Get complete node definition
node_def = agent.get_node_definition()

print(f"Node ID: {node_def.node_id}")
print(f"Type: {node_def.node_type}")  # PROCESSOR, AGGREGATOR, etc.
print(f"Dependencies: {node_def.dependencies}")
print(f"Inputs: {[inp.name for inp in node_def.inputs]}")
print(f"Outputs: {[out.name for out in node_def.outputs]}")

# Convert to dictionary for graph builders
graph_config = node_def.to_dict()
```

#### Graph Builder Infrastructure

Build and execute DAGs with automatic dependency resolution:

```python
from cognivault.langraph.graph_builder import GraphBuilder, GraphExecutor

# Build graph from agents
builder = GraphBuilder()
builder.add_agents([refiner, critic, historian, synthesis])

# Create validated DAG
graph_def = builder.build()

# Execute graph
executor = GraphExecutor(graph_def, agents_dict)
result = await executor.execute(initial_context)
```

#### Key LangGraph Features

- **Node Type Classification**: PROCESSOR, DECISION, TERMINATOR, AGGREGATOR
- **Dependency Resolution**: Automatic topological sorting and cycle detection  
- **Input/Output Schemas**: Type-safe node interfaces with validation
- **Graph Validation**: Comprehensive DAG structure validation
- **Edge Routing**: Sequential, conditional, parallel, and aggregation edges
- **Execution Ordering**: Proper dependency-aware execution flow
- **Configuration Support**: Per-node configuration and timeout overrides
- **Backward Compatibility**: Existing CLI and orchestrator workflows unchanged

#### Future LangGraph Migration

The compatibility layer provides:

- **Seamless Transition**: Drop-in replacement for current orchestration
- **Incremental Migration**: Can run hybrid legacy + LangGraph workflows  
- **Production Ready**: Full error handling, retry logic, and observability
- **Performance Optimized**: Efficient graph construction and execution

### 🔀 Failure Propagation & Conditional Execution

CogniVault features a sophisticated failure propagation system designed for LangGraph DAG compatibility. The system provides conditional execution semantics, graceful degradation strategies, and intelligent dependency resolution.

#### Failure Propagation Strategies

Each agent can be configured with specific failure handling strategies:

- **FAIL_FAST**: Stop immediately on any failure (default for critical agents)
- **WARN_CONTINUE**: Log warning but continue execution (for optional components)  
- **CONDITIONAL_FALLBACK**: Try alternative execution paths
- **GRACEFUL_DEGRADATION**: Skip non-critical agents and continue with reduced functionality

#### Execution Path Tracking

The system tracks execution paths for future LangGraph DAG migration:

- **NORMAL**: Standard execution flow
- **FALLBACK**: Alternative execution when primary agent fails
- **DEGRADED**: Reduced functionality mode
- **RECOVERY**: Recovery from previous failures

#### Agent Dependency Resolution

CogniVault automatically resolves agent dependencies using topological sorting:

- **Dependency Tracking**: Define which agents depend on others
- **Circular Dependency Detection**: Prevents infinite dependency loops
- **Optimal Execution Order**: Automatically determines best execution sequence
- **Health Checks**: Validates agent readiness before execution

#### Example: Conditional Execution Configuration

The orchestrator automatically handles different failure strategies:

- **Critical agent failures** (Refiner): FAIL_FAST - stops execution immediately
- **Optional agent failures** (Critic): GRACEFUL_DEGRADATION - skips and continues
- **Warning-only failures** (Historian): WARN_CONTINUE - logs warning but continues
- **Fallback strategies** (Synthesis): CONDITIONAL_FALLBACK - tries alternative paths

```python
orchestrator = AgentOrchestrator()
context = await orchestrator.run("Your question")

# Check execution path and degradation status
if context.execution_edges:
    print(f"Execution path: {context.path_metadata}")
if context.conditional_routing:
    print(f"Conditional decisions: {context.conditional_routing}")
```

#### LangGraph DAG Compatibility

The failure propagation system is designed for seamless LangGraph migration:

- **Node Isolation**: Each agent operates as an isolated node with error boundaries
- **Conditional Edges**: Failure strategies map directly to LangGraph conditional routing
- **State Management**: Execution context preserves state for DAG reentrance
- **Edge Metadata**: All execution decisions are tracked for DAG edge configuration

### 🛡️ Enterprise Error Handling & Agent Resilience

CogniVault features a comprehensive error handling system designed for production reliability and future LangGraph DAG compatibility. The system provides structured exception hierarchies, agent-isolated error boundaries, and sophisticated retry mechanisms.

#### Exception Hierarchy

The `src/cognivault/exceptions/` package provides organized, typed exceptions with LangGraph-compatible error routing:

- **Agent Errors** (`agent_errors.py`): AgentExecutionError, AgentTimeoutError, AgentDependencyMissingError, AgentResourceError
- **LLM Errors** (`llm_errors.py`): LLMQuotaError, LLMAuthError, LLMRateLimitError, LLMTimeoutError, LLMContextLimitError
- **Configuration Errors** (`config_errors.py`): ConfigurationError, ConfigValidationError, EnvironmentError, APIKeyMissingError
- **I/O Errors** (`io_errors.py`): FileOperationError, MarkdownExportError, DiskSpaceError, PermissionError
- **Orchestration Errors** (`orchestration_errors.py`): Pipeline and dependency management exceptions

#### Agent-Level Resilience Features

Each agent operates with isolated error boundaries and configurable resilience patterns:

- **Retry Configuration**: Configurable max retries, base delay, exponential backoff, and jitter
- **Circuit Breaker Protection**: Prevents cascade failures with configurable thresholds
- **Timeout Management**: Per-agent timeout configuration with graceful degradation
- **Error Isolation**: Agent boundaries prevent one failure from affecting others

#### Key Resilience Features

- **Circuit Breakers**: Prevent cascade failures with configurable failure thresholds
- **Exponential Backoff**: Intelligent retry delays with jitter to prevent thundering herd
- **Agent Isolation**: Error boundaries prevent one agent failure from affecting others
- **Trace Metadata**: All operations include step_id, agent_id, and timestamp for observability
- **LLM Error Mapping**: Comprehensive OpenAI error handling with structured exception conversion
- **Timeout Management**: Per-agent timeout configuration with graceful degradation

#### Structured Error Context

All exceptions include rich context for debugging and monitoring:

- **Agent Information**: Agent name, step ID, and execution context
- **Retry Policy**: Configured retry behavior and failure strategy
- **Trace Metadata**: Timestamp, execution path, and dependency information
- **User-Friendly Messages**: Human-readable error descriptions with troubleshooting tips

This error handling foundation prepares CogniVault for LangGraph migration by providing:
- Agent-isolated boundaries (future LangGraph nodes)
- Structured error routing (future conditional DAG edges)
- Trace-compatible metadata (future execution tracking)
- Reversible state management (future DAG reentrant execution)

---

## 🛠️ Installation & Setup

To get started quickly:

```bash
bash setup.sh
```

This script will:

- Create a Python 3.12.2 virtual environment using `pyenv`
- Install dependencies from `requirements.txt`
- Install Git hooks to enforce formatting, type-checking, and testing before commits and pushes

If you don't have `pyenv` installed, refer to: https://github.com/pyenv/pyenv#installation

### Git Hooks (Optional Manual Setup)

Hooks are installed automatically by `setup.sh`, but you can manually install or review them:

- `pre-commit`: Runs code formatter (`make format`) and type checks (`make typecheck`)
- `pre-push`: Runs test suite (`make test`)

---

## ⚙️ Configuration Management

CogniVault features a comprehensive configuration system that centralizes all application settings, replacing scattered magic numbers and constants throughout the codebase. The configuration system supports multiple environments and provides flexible configuration through environment variables and JSON files.

### Configuration Categories

- **ExecutionConfig**: Agent timeouts, retries, simulation delays, and pipeline settings
- **FileConfig**: Directory paths, file size limits, filename generation settings
- **ModelConfig**: LLM provider settings, token limits, temperature, and mock data
- **TestConfig**: Testing timeouts, simulation settings, and mock history data

### Environment Variables

Configure CogniVault using `COGNIVAULT_*` prefixed environment variables:

```env
# Environment and logging
COGNIVAULT_ENV=development  # development, testing, production
COGNIVAULT_LOG_LEVEL=INFO   # DEBUG, INFO, WARNING, ERROR
COGNIVAULT_DEBUG=false      # Enable debug mode

# Execution settings
COGNIVAULT_MAX_RETRIES=3
COGNIVAULT_TIMEOUT_SECONDS=10
COGNIVAULT_RETRY_DELAY=1.0
COGNIVAULT_SIMULATION_DELAY=false
COGNIVAULT_SIMULATION_DELAY_SECONDS=0.1
COGNIVAULT_CRITIC_ENABLED=true

# File handling
COGNIVAULT_NOTES_DIR=./src/cognivault/notes
COGNIVAULT_LOGS_DIR=./src/cognivault/logs
COGNIVAULT_QUESTION_TRUNCATE=40
COGNIVAULT_HASH_LENGTH=6
COGNIVAULT_MAX_FILE_SIZE=10485760  # 10MB

# Model settings
COGNIVAULT_LLM=openai  # LLM provider selection
COGNIVAULT_MAX_TOKENS=4096
COGNIVAULT_TEMPERATURE=0.7

# Testing
COGNIVAULT_TEST_TIMEOUT_MULTIPLIER=1.5
COGNIVAULT_TEST_SIMULATION=true

# Context Management
COGNIVAULT_MAX_CONTEXT_SIZE_BYTES=1048576  # 1MB
COGNIVAULT_MAX_SNAPSHOTS=5
COGNIVAULT_ENABLE_CONTEXT_COMPRESSION=true
COGNIVAULT_CONTEXT_COMPRESSION_THRESHOLD=0.8
```

### JSON Configuration Files

You can also use JSON configuration files for more complex setups:

```json
{
  "environment": "production",
  "log_level": "INFO",
  "debug_mode": false,
  "execution": {
    "max_retries": 5,
    "timeout_seconds": 30,
    "critic_enabled": true,
    "default_agents": ["refiner", "historian", "critic", "synthesis"]
  },
  "files": {
    "notes_directory": "/app/data/notes",
    "logs_directory": "/app/data/logs"
  },
  "models": {
    "default_provider": "openai",
    "temperature": 0.8
  }
}
```

### Programmatic Configuration

For advanced use cases, you can configure CogniVault programmatically:

```python
from cognivault.config.app_config import ApplicationConfig, Environment, set_config

# Create custom configuration
config = ApplicationConfig()
config.environment = Environment.PRODUCTION
config.execution.max_retries = 5
config.files.notes_directory = "/custom/path"

# Set as global configuration
set_config(config)

# Load from file
config = ApplicationConfig.from_file("/path/to/config.json")
set_config(config)
```

### Environment-Specific Behavior

The configuration system automatically adjusts behavior based on the environment:

- **Development**: Standard timeouts and full logging
- **Testing**: Extended timeouts (multiplied by `test_timeout_multiplier`), simulation enabled
- **Production**: Optimized settings, reduced logging

### Configuration Validation

All configuration values are automatically validated with clear error messages:

```python
config = ApplicationConfig()
config.execution.max_retries = -1  # Invalid
errors = config.validate()
# Returns: ["max_retries must be non-negative"]
```

---

## 🔐 LLM Configuration

CogniVault supports OpenAI out of the box via a `.env` file in the root of the project:

```env
# LLM Provider Configuration
COGNIVAULT_LLM=openai  # Change to "stub" to use a mock LLM for testing

# OpenAI-specific settings (only required when using OpenAI)
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4
OPENAI_API_BASE=https://api.openai.com/v1  # Optional

# Model behavior (part of configuration system)
COGNIVAULT_MAX_TOKENS=4096
COGNIVAULT_TEMPERATURE=0.7
```

You can define new LLM backends by extending the `LLMInterface` and registering them in the `LLMFactory`. The active backend is selected via the environment variable `COGNIVAULT_LLM`.

The LLM backend is now part of the centralized configuration system. Set `COGNIVAULT_LLM=openai` for OpenAI or `COGNIVAULT_LLM=stub` for testing with mock responses.

The `OPENAI_*` variables are only required when using the OpenAI backend:

---

## 🧩 Advanced: Adding a Custom LLM

To integrate your own model (e.g. hosted model or different provider like Anthropic, Mistral, or local inference):

1. **Implement the interface**:
   Create a new class that inherits from `LLMInterface` in `src/cognivault/llm/llm_interface.py`.  
   Note that the `generate` method now supports an optional `system_prompt` parameter to provide more flexible prompt control.

2. **Add to factory**:
   Register your new implementation in `LLMFactory` (`src/cognivault/llm/factory.py`) under a new provider name.

3. **Update the enum**:
   Add your provider to `LLMProvider` in `src/cognivault/llm/provider_enum.py`.

4. **Configure it**:
   In your `.env`, set:
   ```
   COGNIVAULT_LLM=yourprovider
   ```

This approach allows you to cleanly swap or combine LLMs in the future with minimal change to your orchestrator or agent code.

---

## 🖥️ Usage

### Run the assistant

Make sure your `.env` file is configured with your OpenAI credentials if using the OpenAI LLM backend.

To run the full pipeline with all agents:

```bash
make run QUESTION="Is democracy becoming more robust globally?"
```

This executes:

```bash
PYTHONPATH=src python -m cognivault.cli "$(QUESTION)" $(if $(AGENTS),--agents=$(AGENTS),) $(if $(LOG_LEVEL),--log-level=$(LOG_LEVEL),) $(if $(EXPORT_MD),--export-md,) $(if $(TRACE),--trace,) $(if $(HEALTH_CHECK),--health-check,) $(if $(DRY_RUN),--dry-run,) $(if $(EXPORT_TRACE),--export-trace=$(EXPORT_TRACE),)
```

⚠️ Note: `$(QUESTION)` is a Makefile variable — this syntax only works with `make run`. If you're calling the Python CLI directly, use standard shell quotes:

```bash
PYTHONPATH=src python -m cognivault.cli "What is cognition?" --agents=refiner,critic
```

You can also run a **single agent in isolation** using the `AGENTS` environment variable:

```bash
make run QUESTION="What are the benefits of a polycentric governance model?" AGENTS=refiner
make run QUESTION="How does historical context affect AI safety debates?" AGENTS=critic
make run QUESTION="What long-term trends influence democratic erosion?" AGENTS=historian
make run QUESTION="What’s the synthesized conclusion from all agents?" AGENTS=synthesis
```

This maps to the CLI flag `--agents=name1,name2`, allowing you to run any combination of agents by name. Leave unset to run the full pipeline.

### Control Log Level

You can control the logging verbosity using the `LOG_LEVEL` environment variable. Available levels include `DEBUG`, `INFO`, `WARNING`, and `ERROR`.

```bash
make run QUESTION="your query here" AGENTS=refiner,critic LOG_LEVEL=DEBUG
```

This helps in debugging and understanding agent behavior during development.

### Export Markdown Output

To save the output of agent responses as a markdown file (for integration into a personal wiki or digital garden), use the `EXPORT_MD=1` flag:

```bash
make run QUESTION="What is cognition?" AGENTS=refiner,critic EXPORT_MD=1
```

This will generate a `.md` file in `src/cognivault/notes/` with YAML frontmatter metadata including the title, date, agents, filename, source, and a UUID. The content is formatted for easy future retrieval and indexing.

📄 Output saved to: `src/cognivault/notes/2025-06-26T10-04-47_what-is-cognition.md`

With frontmatter like:

```markdown
---
agents:
  - Refiner
  - Critic
date: 2025-06-26T10:04:47
filename: 2025-06-26T10-04-47_what-is-cognition.md
source: cli
summary: Draft response from agents about the definition and scope of the question.
title: What is cognition?
uuid: 8fab709a-8fc4-464a-b16b-b7a55c84aedf
---
```

### 🔍 Enhanced CLI Observability Features

CogniVault includes enterprise-grade CLI observability features for debugging, monitoring, and production deployment:

#### Execution Tracing

Get detailed execution traces with timing, metadata, and pipeline flow:

```bash
# Enable detailed tracing
make run QUESTION="Your question" TRACE=1

# Trace with specific agents
make run QUESTION="Your question" AGENTS=refiner,critic TRACE=1
```

The trace output includes:
- **Pipeline Summary**: Execution time, context size, agent status
- **Agent Execution Status**: Detailed timing and success/failure information  
- **Execution Flow**: Visual representation of agent dependencies and routing
- **Conditional Routing**: Decision points and alternative paths taken
- **Rich Console Output**: Colored panels, tables, and structured information

#### 🩺 Health Checks

Validate system health without executing the pipeline:

```bash
# Run health checks for all agents
make run QUESTION="Any question" HEALTH_CHECK=1

# Check specific agents  
make run QUESTION="Any question" AGENTS=refiner,critic HEALTH_CHECK=1
```

Health checks validate:
- **Agent Readiness**: LLM connectivity, configuration validation
- **Dependency Resolution**: Agent dependency satisfaction
- **Resource Availability**: Memory, disk space, API quotas
- **Configuration Validation**: Environment variables, API keys

#### 🧪 Dry Run Validation

Validate pipeline configuration without execution:

```bash
# Validate full pipeline
make run QUESTION="Your question" DRY_RUN=1

# Validate specific agent subset
make run QUESTION="Your question" AGENTS=refiner,historian DRY_RUN=1
```

Dry run provides:
- **Pipeline Configuration**: Agent list, execution order, dependencies
- **Dependency Validation**: Visual dependency tree and execution flow
- **Health Check Integration**: Combined validation and health checking
- **Configuration Summary**: Complete pipeline setup overview

#### 📊 Trace Export

Export detailed execution traces for analysis and monitoring:

```bash
# Export trace to JSON file
make run QUESTION="Your question" EXPORT_TRACE=/tmp/trace.json

# Combined tracing and export
make run QUESTION="Your question" TRACE=1 EXPORT_TRACE=/tmp/detailed_trace.json
```

Exported traces include:
- **Complete Execution Metadata**: Pipeline ID, timing, context size
- **Agent Performance Data**: Execution times, success rates, outputs
- **Execution Flow**: Dependencies, routing decisions, edge metadata
- **Context State**: Full context snapshots and state transitions
- **Structured JSON Format**: Machine-readable for automation and analysis

#### Combined Usage Examples

The CLI flags can be combined for powerful debugging and monitoring workflows:

```bash
# Full observability pipeline
make run QUESTION="Complex research question" TRACE=1 EXPORT_TRACE=/tmp/research_trace.json EXPORT_MD=1

# Development debugging workflow  
make run QUESTION="Test query" AGENTS=refiner DRY_RUN=1 LOG_LEVEL=DEBUG

# Production health monitoring
make run QUESTION="Health check query" HEALTH_CHECK=1 LOG_LEVEL=INFO

# Performance analysis pipeline
make run QUESTION="Performance test" TRACE=1 EXPORT_TRACE=/tmp/perf_$(date +%s).json LOG_LEVEL=INFO
```

### 🚀 Execution Modes & Performance Comparison

CogniVault supports multiple execution modes with comprehensive performance comparison capabilities:

#### Execution Modes

**LangGraph Mode (default)**: Uses production `RealLangGraphOrchestrator` with real LangGraph 0.5.1 StateGraph integration
```bash
# Default execution - no flag needed
make run QUESTION="Your question"

# Explicit LangGraph mode (same as default)
make run QUESTION="Your question" EXECUTION_MODE=langgraph-real
```

**LangGraph DAG Mode (deprecated)**: Uses intermediate `LangGraphOrchestrator` with DAG-based execution
```bash
make run QUESTION="Your question" EXECUTION_MODE=langgraph
```

**Legacy Mode (DEPRECATED - WILL BE REMOVED)**: Uses the original `AgentOrchestrator` - **scheduled for removal in 2-3 weeks**
```bash
# NOT RECOMMENDED - Use default LangGraph mode instead
make run QUESTION="Your question" EXECUTION_MODE=legacy
```

#### Performance Comparison

Compare LangGraph execution modes side-by-side to validate performance and output consistency:
```bash
# Basic comparison - single run
make run QUESTION="Your question" COMPARE_MODES=1

# Statistical benchmarking - multiple runs with timing analysis
make run QUESTION="Your question" COMPARE_MODES=1 BENCHMARK_RUNS=5

# Comprehensive comparison with trace export
make run QUESTION="Your question" COMPARE_MODES=1 BENCHMARK_RUNS=3 EXPORT_TRACE=/tmp/comparison.json
```

The comparison provides:
- **Statistical Analysis**: Execution time averages, standard deviation, min/max
- **Memory Usage**: Memory consumption differences between modes  
- **Success Rate**: Reliability comparison across multiple runs
- **Context Size**: Data flow efficiency analysis
- **Output Comparison**: Side-by-side agent output validation
- **Performance Metrics**: Detailed timing and resource usage stats

Example output:
```
📊 Performance Benchmark Results

                         Performance Comparison                         
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ LangGraph Mode ┃ LangGraph-Real   ┃       Difference ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Avg Execution Time │         2.124s │           1.424s │ +0.700s (+33.0%) │
│ Success Rate       │         100.0% │           100.0% │            +0.0% │
│ Avg Memory Usage   │         5.2 MB │           4.1 MB │          +1.1 MB │
│ Avg Context Size   │      580 bytes │        460 bytes │       +120 bytes │
└────────────────────┴────────────────┴──────────────────┴──────────────────┘
```

### 📊 DAG Visualization

CogniVault provides advanced DAG visualization capabilities using Mermaid diagrams:

#### Basic DAG Visualization
```bash
# Output to stdout
make run QUESTION="Your question" VISUALIZE_DAG=stdout

# Output to file
make run QUESTION="Your question" VISUALIZE_DAG=dag.md

# Visualization only (no execution)
make run QUESTION="" VISUALIZE_DAG=stdout
```

#### Combined with Execution Modes
```bash
# Visualize DAG execution (default mode)
make run QUESTION="Your question" VISUALIZE_DAG=stdout

# Visualize specific agents
make run QUESTION="Your question" AGENTS=refiner,critic VISUALIZE_DAG=stdout
```

The visualization generates professional Mermaid diagrams showing:
- **Agent Dependencies**: Visual representation of agent execution order
- **State Flow**: How data flows between agents via TypedDict states
- **Node Metadata**: Agent types, confidence levels, and execution status
- **Execution Path**: Visual trace of actual execution flow
- **Phase Compatibility**: Automatic filtering for supported agents

### 🏗️ **Phase 2: Graph Builder Extraction & Architecture Enhancement**

CogniVault has completed **Phase 2 of the LangGraph migration** by extracting graph building logic from the orchestrator into a dedicated `langgraph_backend/` module, providing clean separation of concerns, performance optimization, and extensible graph patterns.

#### Graph Building Architecture

**GraphFactory**: Centralized graph building and compilation
- Pattern-based graph construction with caching
- Support for multiple graph patterns (standard, parallel, conditional)
- LRU cache with TTL for compiled graphs (~90% performance improvement)
- Thread-safe operations with comprehensive validation

**Graph Patterns System**: Extensible execution flow definitions
```bash
# Standard pattern (default)
make run QUESTION="Your question"  # Uses: refiner → [critic, historian] → synthesis

# Future patterns (extensible architecture)
# Parallel pattern: Maximum parallelization
# Conditional pattern: Dynamic routing based on agent outputs
```

**Performance Benefits**:
- **30-minute TTL** for compiled graphs with intelligent caching
- **Separation of concerns**: Graph building vs execution orchestration  
- **Enhanced testability**: Independent unit testing of graph building logic
- **Memory efficient**: Size estimation and automatic cache management

### 💾 **Checkpointing & Memory Management**

CogniVault supports **optional checkpointing and conversation persistence** using LangGraph's MemorySaver integration for long-running workflows and multi-session DAGs.

#### Memory & Checkpointing Features

**Enable Checkpointing**: Add conversation persistence and rollback capabilities
```bash
# Enable checkpointing with auto-generated thread ID
make run QUESTION="Your question" ENABLE_CHECKPOINTS=1

# Use custom thread ID for session scoping
make run QUESTION="Your question" ENABLE_CHECKPOINTS=1 THREAD_ID=my-session

# Default LangGraph mode with checkpointing
make run QUESTION="Your question" ENABLE_CHECKPOINTS=1
```

**Rollback Mechanisms**: Recover from failed executions using checkpoints
```bash
# Rollback to last checkpoint on failure
make run QUESTION="Your question" ENABLE_CHECKPOINTS=1 ROLLBACK_LAST_CHECKPOINT=1

# Resume specific thread session
make run QUESTION="Continue previous analysis" ENABLE_CHECKPOINTS=1 THREAD_ID=research-session
```

**Performance Testing**: Ensure no regression when checkpointing is disabled
```bash
# Benchmark with and without checkpointing
make run QUESTION="Your question" COMPARE_MODES=1 BENCHMARK_RUNS=5
make run QUESTION="Your question" COMPARE_MODES=1 BENCHMARK_RUNS=5 ENABLE_CHECKPOINTS=1
```

#### Checkpointing Architecture

The implementation provides:

- **Optional Checkpointing**: Defaults to off for backward compatibility
- **Thread ID Scoping**: Multi-session conversation isolation with auto-generation
- **Memory Management**: TTL-based cleanup, checkpoint limits, and thread management  
- **Error Resilience**: Circuit breakers, retry policies, and graceful degradation
- **State Serialization**: Robust CogniVaultState persistence with comprehensive type handling
- **Enhanced DAG Visualization**: Shows checkpoint nodes, error handling routes, and memory state

#### Memory Manager Features

- **LangGraph MemorySaver Integration**: Native checkpointing with LangGraph 0.5.1
- **Thread-Scoped Memory**: Conversation isolation with unique thread IDs
- **Checkpoint Lifecycle Management**: Creation, cleanup, and TTL-based expiration
- **Rollback Mechanisms**: Failed execution recovery with state restoration
- **Error Policy Integration**: Centralized retry logic and circuit breaker patterns

Example checkpoint workflow:
```bash
# Start a research session with checkpointing
make run QUESTION="What are the implications of AI governance?" ENABLE_CHECKPOINTS=1 THREAD_ID=ai-governance-research

# Continue the conversation in the same session
make run QUESTION="How do different countries approach AI regulation?" ENABLE_CHECKPOINTS=1 THREAD_ID=ai-governance-research  

# If execution fails, rollback to last checkpoint
make run QUESTION="What are the enforcement mechanisms?" ENABLE_CHECKPOINTS=1 THREAD_ID=ai-governance-research ROLLBACK_LAST_CHECKPOINT=1
```

The checkpointing system prepares CogniVault for **long-running workflows**, **multi-session DAGs**, and **conversation persistence** while maintaining full backward compatibility.

---

## 📊 CLI Diagnostics & Observability

CogniVault includes comprehensive diagnostics capabilities accessible via the `cognivault diagnostics` command suite:

### Health Checks

Check system health with detailed component analysis:

```bash
# Quick health overview
cognivault diagnostics health

# JSON output for automation
cognivault diagnostics health --format json

# Quiet mode (exit codes only)
cognivault diagnostics health --quiet
```

### Performance Metrics

Monitor system performance and statistics:

```bash
# Performance overview
cognivault diagnostics metrics

# Export to Prometheus format
cognivault diagnostics metrics --format prometheus

# Agent-specific metrics only
cognivault diagnostics metrics --agents

# Time-windowed metrics (last N minutes)
cognivault diagnostics metrics --window 30
```

### System Status

Get comprehensive system information:

```bash
# Detailed system status
cognivault diagnostics status

# JSON output with custom time window
cognivault diagnostics status --json --window 60
```

### Agent Diagnostics

Monitor individual agent performance:

```bash
# All agents status
cognivault diagnostics agents

# Specific agent details
cognivault diagnostics agents --agent refiner --json
```

### Configuration Validation

Validate system configuration:

```bash
# Configuration overview
cognivault diagnostics config

# Validation only
cognivault diagnostics config --validate

# JSON output
cognivault diagnostics config --json
```

### Complete Diagnostics

Run full system diagnostics with export options:

```bash
# Complete system report
cognivault diagnostics full

# Export to file in different formats
cognivault diagnostics full --format json --output system-report.json
cognivault diagnostics full --format csv --output metrics.csv
cognivault diagnostics full --format prometheus --output metrics.prom
```

### Monitoring Integration

The CLI supports multiple output formats for seamless monitoring integration:

- **JSON**: API consumption and dashboard integration
- **CSV**: Spreadsheet analysis and reporting
- **Prometheus**: Metrics collection and alerting
- **InfluxDB**: Time-series data storage

All commands include rich console output with colors, tables, and progress indicators for an excellent developer experience.

---

## 🧠 Example Output

```markdown
### 🔍 Refiner:
Clarifies that the user is asking about structural versus cultural forces in education systems.

### 🧾 Historian:
Recalls that prior conversations touched on ed-tech, teacher labor markets, and digital equity.

### 🧠 Critic:
Questions the assumption that AI improves access without reinforcing inequality.

### 🧵 Synthesis:
AI’s long-term effects on education depend on how we resolve tensions between scale and personalization.
```

---

## 🧪 Run Tests

```bash
make test
```

Covers:
- Agent context and orchestrator pipeline
- All 4 core agents with comprehensive system prompt testing
- Agent Registry with dynamic registration and dependency management
- Comprehensive exception hierarchy with error handling scenarios
- Agent-level resilience with circuit breakers and retry logic
- OpenAI LLM integration with extensive error mapping
- Comprehensive observability and diagnostics testing
- **LangGraph Phase 2.0 Integration**: Real StateGraph orchestration, TypedDict state management, and DAG visualization
- **356 new Phase 2.0 tests** covering state schemas, node wrappers, real orchestrator, and CLI integration
- 89% test coverage across all modules with critical paths at 100% (1,517 total tests)
- Both Refiner and Critic agents include comprehensive system prompt tests to ensure prompt correctness and robustness

Use the batch test tools for agent evaluation:  
```bash
make test-agent-refiner    # Refiner agent batch testing
make test-agent-critic     # Critic agent batch testing
```

### View Coverage

Run the full test suite with a coverage report:

```bash
make coverage-all
```

### Control Log Level During Coverage

You can set a log level when running test coverage to see debug output during test runs:

```bash
LOG_LEVEL=DEBUG make coverage-all
```

This can help trace detailed agent behavior while viewing test coverage results.

This executes:

```bash
PYTHONPATH=src pytest --cov=cognivault --cov-report=term-missing tests/
```

Run coverage on a specific module:

```bash
make coverage-one m=cli LOG_LEVEL=INFO
```

- `m` is required — it's the submodule path under `cognivault`.
- `LOG_LEVEL` is optional (defaults to `WARNING`). Set it to `INFO` or `DEBUG` to see logging output during test runs.

💡 Example:
```bash
make coverage-one m=orchestrator LOG_LEVEL=DEBUG
```

---

## 📈 Prompt Evaluation Tools

We provide specialized tools for evaluating prompt performance and behavior:

- `scripts/agents/refiner/test_batch.py` runs batch tests on the Refiner agent's prompts, enabling detailed analysis of output variations
- `scripts/agents/critic/test_batch.py` runs batch tests on the Critic agent's cognitive reflection pipeline, testing bias detection and confidence scoring
- Both tools include git version metadata in their output to help track prompt changes and reproducibility

These tools facilitate prompt tuning and validation during development and experimentation, ensuring consistent agent behavior across different scenarios.

---

## 💡 Use Cases

CogniVault can serve as a:

- 🧠 Personal knowledge management tool (Zettelkasten, digital garden)
- 💬 Reflection assistant for journaling or ideation
- 📚 Research co-pilot for synthesis and argument mapping
- 🧵 Semantic trace explorer for AI explainability
- 🧪 Experimentation harness for multi-agent reasoning

Future directions: wiki export, browser UI, plugin support (Obsidian, Notion).

---

## 🌍 How CogniVault Differs

Unlike typical LLM assistants or AutoGPT-style agents, CogniVault focuses on *structured introspection* rather than task completion. While tools like LangGraph or Reflexion optimize for task-solving or dynamic planning, CogniVault enables long-term insight formation across modular agent roles — including Refiner, Historian, Critic, and Synthesis.

It’s designed as a memory-enhanced thinking partner that integrates cleanly with personal wikis, supports test-driven CLI use, and remains light enough for future microservice deployment or API integration.

---

## 🔭 Roadmap

- [x] Agent toggles via CLI (`--agents=name1,name2`)
- [x] Asynchronous agent execution
- [x] **LangGraph Phase 2**: Graph builder extraction and architecture enhancement
- [x] **GraphFactory Implementation**: Centralized graph building with pattern support
- [x] **Performance Optimization**: Graph compilation caching with 90% improvement
- [x] **DAG Visualization**: Mermaid diagram generation for pipeline analysis
- [x] **Performance Comparison**: Statistical benchmarking between execution modes
- [x] **TypedDict State Management**: Type-safe state schemas with mypy compliance
- [x] **Circuit Breaker Patterns**: Robust error handling and resilience
- [x] Markdown exporter for wiki integration
- [ ] **LangGraph Phase 3**: Advanced conditional routing and custom patterns
- [ ] Optional file/vector store persistence
- [ ] API or microservice agent wrappers (e.g. FastAPI)
- [ ] Streamlit UI or Jupyter notebook support

---

## 🛠 Built With

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Typer](https://img.shields.io/badge/CLI-Typer-green)
![Pytest](https://img.shields.io/badge/Testing-Pytest-blueviolet)
![AGPL](https://img.shields.io/badge/License-AGPL_3.0-orange)

---

## 🤝 Contributing

Coming soon: contributor guide and code of conduct.

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
See the [LICENSE](./LICENSE) file for full terms.