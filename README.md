# 🧠 CogniVault

![Python](https://img.shields.io/badge/python-3.12-blue)
![Poetry](https://img.shields.io/badge/poetry-managed-blue)
![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen)
![API Status](https://img.shields.io/badge/API-operational-brightgreen)
![WebSocket](https://img.shields.io/badge/WebSocket-streaming-brightgreen)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Markdown Export](https://img.shields.io/badge/markdown-export-green)
![Wiki Ready](https://img.shields.io/badge/wiki-ready-blueviolet)

CogniVault is a **sophisticated multi-agent workflow orchestration system** that combines real LLM integration with LangGraph-based DAG orchestration, advanced node types, and comprehensive observability for intelligent agent coordination.

## 🧠 **Core Innovation**: Multi-Agent DAG Platform

CogniVault provides a sophisticated orchestration platform grounded in **cognitive science research** and validated through comprehensive testing:

- **4-Agent Pipeline**: Refiner, Critic, Historian, Synthesis agents with real LLM integration
- **LangGraph Orchestration**: StateGraph-based DAG execution with parallel processing
- **Advanced Node Types**: Decision, Aggregator, Validator, Terminator nodes for complex workflows
- **Event-Driven Architecture**: Comprehensive observability with correlation tracking
- **Multi-Axis Classification**: 6-axis metadata system for intelligent routing (see [AAD-002](src/cognivault/docs/architecture/AAD-002-Multi-Axis-Classification-And-Advanced-Node-Types.md))
- **Configurable Prompt Composition**: YAML-driven agent behavior customization without code changes
- **Enhanced Routing System**: OPERATIONAL algorithm-driven agent selection with performance optimization and constraint validation

**Status**: V1 fully functional system with complete multi-agent workflows, LangGraph orchestration, configurable prompt composition, and comprehensive observability. **Phase 1A Complete**: FastAPI service layer with WebSocket real-time streaming and Docker development environment.

> **📚 Research Foundations**: See [RESEARCH.md](src/cognivault/docs/RESEARCH.md) for comprehensive cognitive science foundations including distributed cognition theory, dual-process theory, and society of mind principles that inform the architecture.
> 
> **🌍 Strategic Position**: See [LANDSCAPE.md](src/cognivault/docs/LANDSCAPE.md) for competitive analysis and strategic positioning within the multi-agent ecosystem.
>
> **🎯 Pattern Documentation**: See [PATTERN_REGISTRY.md](src/cognivault/docs/PATTERN_REGISTRY.md) for comprehensive pattern documentation with validation framework and certification standards.
>
> **📊 Observability Guide**: See [OBSERVABILITY.md](src/cognivault/docs/OBSERVABILITY.md) for comprehensive observability architecture including event-driven monitoring and production deployment strategies.
>
> **🗄️ Database Exploration**: See [DATABASE_EXPLORATION.md](src/cognivault/docs/DATABASE_EXPLORATION.md) for complete guide to querying Pydantic AI integration data, JSONB analytics, and agent performance metrics.
>
> **🔍 Hybrid Search System**: The HistorianAgent features a production-ready hybrid search system combining PostgreSQL full-text search with file-based retrieval, configurable search ratios, intelligent deduplication, and comprehensive fallback mechanisms.

## 📋 Requirements

Before getting started, ensure you have the following installed:

- **Python 3.12+** (tested with Python 3.12.2)
- **Poetry** (for dependency management)
- **Git** (for cloning the repository)

### Installing Poetry

Poetry is the recommended way to manage dependencies for this project. Install it using one of these methods:

**Option 1: Official installer (recommended)**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Option 2: Via pip**
```bash
pip install poetry
```

**Option 3: Via package manager**
```bash
# macOS (Homebrew)
brew install poetry

# Ubuntu/Debian
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

After installation, verify Poetry is working:
```bash
poetry --version
```

For detailed installation instructions, see the [official Poetry documentation](https://python-poetry.org/docs/#installation).

## ⚡ Quickstart

Clone the repo and run a basic question through the CLI:

```bash
git clone https://github.com/aucontraire/cognivault.git
cd cognivault

# Install dependencies with Poetry
poetry install

# Run setup (creates environment and installs git hooks)
bash setup.sh

# Run your first question
make run QUESTION="What are the long-term effects of AI in education?"

# Or use the convenient cognivault command
cognivault main "What are the long-term effects of AI in education?"
cognivault diagnostics health  # Check system health

# Database & Pydantic AI Integration (optional)
make db-setup                  # Setup PostgreSQL 17 + pgvector
make test-pydantic-ai         # Test structured LLM responses with database
make db-explore               # Explore stored agent data and performance metrics
```

See [🖥️ Usage](#️usage) for running specific agents and debugging options, or check the [📖 Comprehensive CLI Usage Guide](src/cognivault/docs/CLI_USAGE.md) for detailed command reference including configurable prompt composition examples.

> **🔍 Semantic Validation**: See [SEMANTIC_VALIDATION_USAGE.md](src/cognivault/docs/SEMANTIC_VALIDATION_USAGE.md) for comprehensive workflow validation capabilities and usage examples.

---

## 🚀 Features

### 🎯 **Core Features** 
*What CogniVault does - the essential capabilities that make it a powerful thinking partner*

- 🧠 **Multi-agent orchestration**: Four specialized agents (Refiner, Historian, Critic, Synthesis) for structured thought reflection
- 🔁 **LangGraph-powered execution**: Advanced DAG-based orchestration with checkpointing and conversation rollback
- 🧠 **Enhanced intelligent routing**: OPERATIONAL algorithm-driven agent selection with performance optimization and constraint validation
- 🌊 **Event-driven architecture**: Comprehensive event system with workflow tracking, metrics collection, and observability
- 📄 **Markdown-ready output**: Integration-friendly output for personal wikis and knowledge management systems
- 🔄 **Swappable LLM backends**: Plug-and-play support for OpenAI or stubs via configuration
- 🌐 **API Service Layer**: FastAPI-based service layer with 9 functional endpoints and comprehensive testing
- ⚡ **Real-Time Streaming**: WebSocket-based live workflow progress updates with correlation ID filtering
- 🐳 **Docker Development Environment**: Containerized development setup with hot reload

### 🔧 **Developer Experience**
*How you work with CogniVault - tools and interfaces that make development productive*

- ✅ **Rich CLI interface**: Full-featured command-line tool using Typer with comprehensive help and options
- 📊 **Comprehensive diagnostics**: `cognivault diagnostics` suite for health checks, metrics, and system status
- 🔍 **Execution observability**: Real-time tracing (`--trace`), health checks (`--health-check`), dry run (`--dry-run`)
- 📊 **DAG visualization**: Mermaid diagram generation (`--visualize-dag`) for pipeline analysis and debugging
- 🔀 **Performance comparison**: Statistical benchmarking between execution modes (`--compare-modes`)
- 🧪 **Comprehensive testing**: Full test suite with 86% coverage and 3,454+ tests for reliability
- 🧩 **Pattern validation framework**: OPERATIONAL built-in validation, testing, and certification tools for custom graph patterns (see [Pattern Registry](src/cognivault/docs/PATTERN_REGISTRY.md) for comprehensive pattern documentation)
- 🎯 **Pydantic AI Testing**: Integrated testing tools (`scripts/validate_pydantic_ai_setup.py`, `scripts/test_pydantic_ai_integration.py`)
- 📤 **Trace export**: JSON export of detailed execution metadata (`--export-trace`) for monitoring
- 🌐 **API Manual Testing**: Comprehensive testing guide for API scenarios and WebSocket connections
- 🔧 **Docker Development**: Containerized development environment with hot reload

### 🏗️ **Architecture**
*How CogniVault is built - comprehensive foundation for reliability and extensibility*

- 🛡️ **Advanced error handling**: Comprehensive exception hierarchy with circuit breakers and agent isolation
- 🔄 **Agent-level resilience**: Individual retry policies, timeout management, and graceful degradation
- 🏗️ **API boundary implementation**: Clean external/internal separation with BaseAPI pattern for service extraction
- 🎭 **Mock-first design**: Comprehensive mock implementations with contract testing for reliable development
- ⚙️ **Pydantic configuration system**: Advanced data validation, type safety, and configurable agent behavior  
- 🎯 **Structured Data Pipeline**: Pydantic AI integration for type-safe agent outputs with JSONB analytics
- 💾 **Thread-scoped memory**: Multi-session conversation management with snapshots and compression
- 📋 **Agent registry**: Dynamic registration system with dependency resolution and health validation
- 🌊 **TypedDict state management**: Type-safe schemas for all agent outputs with mypy compliance

### 🚀 **Advanced Capabilities**
*Power-user features for optimization, monitoring, and development workflows*

- 🎯 **Resource optimization**: Multi-strategy optimization (PERFORMANCE, RELIABILITY, BALANCED, QUALITY, MINIMAL)
- 🔍 **Context complexity analysis**: Automated query analysis for optimal routing decisions and performance prediction
- 📊 **Routing decision framework**: Comprehensive decision tracking with confidence scoring and detailed reasoning
- 🖥️ **Interactive CLI tools**: DAG structure explorer, performance profiler, and pattern benchmark suite
- 🔍 **Structured logging**: OPERATIONAL comprehensive logging with automatic correlation ID tracking and context propagation (see [Observability Guide](src/cognivault/docs/OBSERVABILITY.md) for comprehensive observability architecture)
- 📈 **Performance metrics**: Real-time collection of execution statistics, token usage, and success rates
- 🖥️ **Machine-readable output**: Multiple export formats (JSON, CSV) for monitoring integration
- 🔗 **LangGraph compatibility**: Complete DAG-ready architecture with node interfaces and graph builders

### 🧠 **Multi-Agent Workflow Orchestration** 
*Advanced agent coordination and execution grounded in cognitive science*

- 🤖 **4-Agent Pipeline**: Complete Refiner → Critic → Historian → Synthesis workflow with real LLM integration
- 🏗️ **Advanced Node Types**: Decision, Aggregator, Validator, Terminator nodes for sophisticated routing
- 🔀 **LangGraph Orchestration**: StateGraph-based DAG execution with parallel processing capabilities
- ⚙️ **Event-Driven Architecture**: Comprehensive observability with correlation tracking and metadata
- 🔄 **Circuit Breaker Patterns**: Robust error handling and resilience throughout the system
- 📊 **Performance Monitoring**: Real-time metrics collection and execution statistics
- 🌐 **Fully Functional**: Complete CLI tooling and diagnostic capabilities
- 🎯 **CLI Integration**: Full command suite - `cognivault main`, `diagnostics`, comprehensive options
- 🎛️ **Configurable Prompt Composition**: OPERATIONAL YAML-driven agent behavior customization with 662-line PromptComposer
- 🔍 **Semantic Validation**: OPERATIONAL comprehensive workflow validation (see [Semantic Validation Usage](src/cognivault/docs/SEMANTIC_VALIDATION_USAGE.md))
- 🌐 **API Service Layer**: 9 FastAPI endpoints functional with comprehensive test coverage
- ⚡ **WebSocket Real-Time Streaming**: Live workflow progress with correlation ID filtering and event integration
- 🐳 **Docker Development**: Containerized environment for development

**Example Multi-Agent Execution**:
```bash
# Run complete 4-agent workflow
make run QUESTION="What are the implications of AI governance?"

# Execute specific agents with observability
cognivault main "Your question" --agents refiner,critic --trace --export-md

# Health check and system diagnostics
cognivault diagnostics health
cognivault diagnostics full --format json
```

**Advanced Node Configuration Example**:
```yaml
# Advanced workflow with decision and validation nodes
nodes:
  - node_id: "complexity_router"
    node_type: "decision"
    execution_pattern: "decision"
    config:
      criteria: ["query_complexity", "confidence_score"]
      
  - node_id: "quality_validator"
    node_type: "validator" 
    execution_pattern: "validator"
    config:
      validation_criteria:
        - name: "content_quality"
          threshold: 0.8
          weight: 1.0
```

CogniVault provides sophisticated multi-agent orchestration with comprehensive observability, advanced node types, and comprehensive reliability patterns.

### 🌐 **API Service Layer** (**Phase 1A Complete** ✅)

**FastAPI Service Infrastructure**
- **9 Functional Endpoints**: API coverage for workflow execution, topic management, and system monitoring
  - `POST /api/query` - Execute multi-agent workflows with progress tracking
  - `GET /api/query/status/{correlation_id}` - Workflow status tracking
  - `GET /api/query/history` - Execution history with in-memory storage
  - `GET /api/topics` - Dynamic topic discovery using keyword extraction
  - `GET /api/topics/{id}/wiki` - Knowledge synthesis from workflow outputs
  - `GET /api/workflows` - Workflow discovery from filesystem scanning
  - `GET /api/workflows/{id}` - Individual workflow details
  - `WS /ws/query/{correlation_id}` - Real-time workflow progress streaming
  - `WS /ws/health` - WebSocket health monitoring

**Real-Time WebSocket Streaming**
- **Live Progress Updates**: Real-time workflow execution progress (0-100%) with stage-based calculation
- **Correlation ID Filtering**: Targeted event streaming for specific workflow executions
- **Event Integration**: Integration with existing event system
- **Connection Management**: Thread-safe connection management with automatic cleanup
- **50 Comprehensive Tests**: Full WebSocket functionality coverage

**Docker Development Environment**
- **Development Setup**: Containerized development environment with DRY configuration
- **Hot Reload Support**: Development volumes for immediate code iteration
- **Service Dependencies**: Orchestration with health checks for database and cache services
- **Testing Support**: Profile-based testing with service dependency management

**Current Implementation**
- **167+ Passing Tests**: Comprehensive API coverage with real workflow integration
- **Real LLM Integration**: All endpoints work with actual workflow execution data
- **Manual Testing Guide**: [Complete API testing documentation](src/cognivault/docs/API_MANUAL_TESTING_GUIDE.md) for all endpoints and WebSocket scenarios
- **External Integration**: Platform ready for external consumer integration

**Technical Implementation**
- **Zero Breaking Changes**: All existing functionality preserved during API service implementation
- **Type-Safe Implementation**: Pydantic validation throughout API layer
- **Event-Driven Design**: Event emission and correlation tracking across all endpoints
- **Memory Management**: Thread-safe operations with asyncio locks and resource cleanup

**Current Limitations**
- **In-Memory Storage**: Topic discovery and workflow history use in-memory storage (database integration planned for Phase 1B)
- **File-Based Workflows**: Workflow discovery scans filesystem for YAML files
- **Keyword-Based Topics**: Topic clustering uses keyword extraction (semantic embeddings planned for Phase 1B)

### 🎯 **Advanced Pydantic Migration** (OPERATIONAL)

CogniVault features comprehensive **advanced Pydantic data models** throughout the entire codebase (see [ADR-006](src/cognivault/docs/architecture/ADR-006-Configurable-Prompt-Composition-Architecture.md)), providing enhanced validation, type safety, and modern Python architecture.

#### Pydantic Configuration System (OPERATIONAL)

**🏗️ Configurable Agent Behavior**
- **Agent-Specific Configurations**: RefinerConfig, CriticConfig, HistorianConfig, SynthesisConfig with specialized behavioral settings
- **Dynamic Runtime Modification**: Change agent behavior through YAML workflows and environment variables without code deployment
- **Multi-Environment Support**: Environment variables, JSON files, and programmatic configuration with automatic validation
- **Type-Safe Validation**: Comprehensive Pydantic validation with clear error messages and constraint enforcement

**📝 Advanced Prompt Composition**
- **Template System**: Dynamic prompt generation with configurable parameters and behavioral constraints
- **Custom System Prompts**: Override default prompts with custom templates while maintaining agent functionality
- **Variable Substitution**: Rich template variables with validation and domain-specific constraints

**Example Configuration Usage**:
```python
# Agent Configuration Example
from cognivault.config.agent_configs import CriticConfig

config = CriticConfig(
    analysis_depth="comprehensive",
    confidence_reporting=True,
    bias_detection=True,
    scoring_criteria=["accuracy", "completeness", "objectivity"]
)
```

**🎯 YAML Workflow Integration**:
```yaml
nodes:
  - node_id: "enhanced_critic"
    node_type: "critic"
    config:
      analysis_depth: "deep"
      confidence_reporting: true
      custom_constraints: ["avoid_speculation", "provide_evidence"]
```

#### Comprehensive Pydantic Data Models

**🔹 Advanced Data Validation**
- **15+ Core Models**: Complete migration of diagnostics, workflow, configuration, and context models
- **Field Constraints**: Comprehensive validation with min/max values, string lengths, and format validation
- **Runtime Type Safety**: Automatic type checking with clear error messages and coercion
- **Self-Documenting**: Rich field descriptions and constraint documentation for enhanced developer experience

**🔹 Comprehensive Architecture**
- **Zero MyPy Errors**: Complete type safety across all migrated models
- **100% Backward Compatibility**: All existing APIs work unchanged with `to_dict()` methods
- **Modern Python Standards**: Pydantic v2 compliance with performance optimizations
- **Configuration Validation**: Startup-time validation prevents configuration errors

**Key Benefits**:
- **Enhanced Developer Experience**: IDE auto-completion and rich field information
- **Robust Error Handling**: Structured validation errors with context and suggestions
- **Performance Optimized**: Modern Pydantic v2 performance improvements
- **High Standards**: Comprehensive validation and data integrity




---

## 🧱 Architecture

CogniVault is organized into focused modules following cognitive science principles (see [RESEARCH.md](src/cognivault/docs/RESEARCH.md)) and strategic positioning within the multi-agent ecosystem (see [LANDSCAPE.md](src/cognivault/docs/LANDSCAPE.md)):

- **`agents/`** - Complete 4-agent system (Refiner, Critic, Historian, Synthesis) with real LLM integration
- **`langgraph_backend/`** - LangGraph StateGraph orchestration with DAG execution and parallel processing
- **`events/`** - Comprehensive event-driven architecture with correlation tracking and multi-sink support
- **`config/`** - Centralized configuration management with environment variables and validation
- **`diagnostics/`** - Rich CLI diagnostic tools, health checks, and system observability
- **`llm/`** - LLM abstraction layer (OpenAI integration with fallback modes)
- **`context/`** - Advanced context management with snapshots and memory optimization
- **`cli/`** - Full-featured command-line interface with comprehensive options

> **🏛️ Architecture Documentation**: See [ARCHITECTURE.md](src/cognivault/docs/architecture/ARCHITECTURE.md) and the complete [ADR collection](src/cognivault/docs/architecture/) for detailed architectural decisions, patterns, and implementation strategies.

---

---

## 🌊 Event-Driven Architecture

CogniVault features a comprehensive OPERATIONAL event system for observability and future service extraction (see [ADR-005](src/cognivault/docs/architecture/ADR-005-Event-Driven-Architecture-Implementation.md)):

### Event Types

- **Workflow Events**: Lifecycle tracking (started, completed, failed, cancelled)
- **Agent Execution Events**: Individual agent performance and status monitoring
- **Routing Events**: Decision tracking with confidence scoring and reasoning
- **Performance Events**: Metrics collection and health check monitoring
- **API Events**: Service boundary tracking for future microservice extraction

### Event Features

- **Multi-axis Agent Classification**: Enhanced metadata with task classification and capability tracking
- **Correlation Context**: Full tracing with correlation IDs and parent span tracking
- **Comprehensive Sinks**: File, console, and in-memory event storage options
- **Event Filtering & Statistics**: Comprehensive querying and analytics capabilities
- **Serialization Support**: JSON-compatible event data for storage and transmission

### Usage

```python
from cognivault.events import (
    emit_workflow_started,
    emit_agent_execution_completed,
    get_global_event_emitter
)

# Events are automatically emitted during workflow execution
# Custom event sinks can be configured for monitoring integration
```

Events include comprehensive metadata from the 6-axis classification system and are essential for comprehensive observability. See [OBSERVABILITY.md](src/cognivault/docs/OBSERVABILITY.md) for comprehensive observability architecture and deployment strategies.

---

## 🧠 Agent Roles

Each agent in CogniVault plays a distinct role in the cognitive reflection and synthesis pipeline (grounded in cognitive science research detailed in [RESEARCH.md](src/cognivault/docs/RESEARCH.md)):

- ### 🔍 Refiner
  The **RefinerAgent** takes the initial user input and clarifies intent, rephrases vague language, and ensures the prompt is structured for deeper analysis by the rest of the system. It uses a comprehensive system prompt with passive and active modes to guide its reasoning process. See [`prompts.py`](./src/cognivault/agents/refiner/prompts.py) for implementation details.

- ### 🧾 Historian
  The **HistorianAgent** provides relevant context using a sophisticated **hybrid search system** combining file-based and database sources. Features configurable search ratios (60/40 file/database split), PostgreSQL full-text search, intelligent content deduplication, and automatic fallback mechanisms for maximum reliability and performance.

- ### 🧠 Critic
  The **CriticAgent** evaluates the refined input or historical perspective. It identifies assumptions, weaknesses, or inconsistencies—acting as a thoughtful devil's advocate. Features **structured output support** with Pydantic AI validation for type-safe critique analysis including assumptions, biases, and issue detection.

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

CogniVault features advanced context management designed to prevent memory bloat and provide robust state management for long-running agent conversations. Key features include:

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

This ensures CogniVault can handle long-running conversations and complex multi-agent workflows without memory issues, making it suitable for research applications and extended development sessions.

### 🔗 LangGraph Compatibility Layer (OPERATIONAL)

CogniVault features a complete OPERATIONAL LangGraph compatibility layer (see [ADR-001](src/cognivault/docs/architecture/ADR-001-Graph-Pattern-Architecture.md)) that provides DAG-ready architecture while maintaining full backward compatibility with existing workflows.

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
from cognivault.orchestration.graph_builder import GraphBuilder, GraphExecutor

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
- **Reliable**: Full error handling, retry logic, and observability
- **Performance Optimized**: Efficient graph construction and execution

### 🔀 Failure Propagation & Conditional Execution (OPERATIONAL)

CogniVault features a sophisticated OPERATIONAL failure propagation system (see [ADR-002](src/cognivault/docs/architecture/ADR-002-Conditional-Patterns-And-Developer-Tooling.md)) designed for LangGraph DAG compatibility.

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

### 🏗️ API Boundary Implementation (OPERATIONAL)

CogniVault features a sophisticated OPERATIONAL API boundary implementation following [ADR-004](src/cognivault/docs/architecture/ADR-004-API-Boundary-Implementation-Strategy.md) specifications that establishes clear external/internal API separation and enables future service extraction.

#### BaseAPI Interface Pattern

All APIs implement a standardized `BaseAPI` interface with lifecycle management:

```python
from cognivault.api.external import OrchestrationAPI
from cognivault.api.models import WorkflowRequest

# External API with stable interface
api = LangGraphOrchestrationAPI()  # Production implementation
await api.initialize()

# Execute workflow through API boundary
request = WorkflowRequest(
    query="What are the implications of AI governance?",
    agents=["refiner", "critic", "historian", "synthesis"]
)
response = await api.execute_workflow(request)
print(f"Workflow {response.workflow_id}: {response.status}")
```

#### Mock-First Design & Contract Testing

Comprehensive mock implementations enable immediate testing and parallel development:

```python
from tests.fakes.mock_orchestration import MockOrchestrationAPI

# Mock API for testing - identical interface
mock_api = MockOrchestrationAPI()
await mock_api.initialize()

# Configure failure scenarios for testing
mock_api.set_failure_mode("execution_failure")
mock_api.set_agent_outputs({"refiner": "Custom mock output"})

# All contract tests pass for both real and mock implementations
response = await mock_api.execute_workflow(request)
```

#### Key API Boundary Features

- **External API Contracts**: `OrchestrationAPI`, `LLMGatewayAPI` with backward compatibility guarantees
- **Internal API Contracts**: `InternalWorkflowExecutor`, `InternalPatternManager` (subject to refactor)
- **Runtime Validation**: `@ensure_initialized`, `@rate_limited`, `@circuit_breaker` decorators
- **Schema Management**: External schema protection with versioning and migration support
- **Contract Testing**: Comprehensive test suite ensuring implementation consistency
- **Service Extraction Ready**: Clear boundaries for future microservice deployment

#### Architecture Benefits

- **Clean Boundaries**: Explicit separation between stable external APIs and refactorable internals
- **Testability**: Mock-first design enables comprehensive testing from day one
- **Swappability**: Contract testing ensures implementations can be replaced seamlessly
- **Service Evolution**: Prepared for microservice extraction without breaking changes

### 🧠 Enhanced Intelligent Routing (OPERATIONAL)

CogniVault features an OPERATIONAL intelligent routing system that automatically selects optimal agents based on query complexity analysis, performance metrics, and resource constraints. The enhanced routing system provides algorithm-driven optimization for maximum efficiency and reliability.

#### Core Routing Features

**Context Complexity Analysis**: Automated analysis of query characteristics to determine optimal routing strategy
- **Complexity Scoring**: Multi-dimensional analysis including word count, technical terms, and domain indicators
- **Strategy Selection**: STREAMLINED, STANDARD, COMPREHENSIVE, or PERFORMANCE_OPTIMIZED routing
- **Dynamic Adaptation**: Real-time routing decisions based on query requirements

**Resource Optimization Strategies**: Multiple optimization approaches for different use cases
- **PERFORMANCE**: Optimize for execution speed with fastest agents
- **RELIABILITY**: Optimize for success rate with most reliable agents  
- **BALANCED**: Balance performance, reliability, and resource usage
- **QUALITY**: Optimize for output quality regardless of execution time
- **MINIMAL**: Use minimum viable agents for simple queries

**Constraint Validation & Management**: Comprehensive constraint handling for production deployment
- **Agent Constraints**: Required agents, forbidden agents, min/max agent counts
- **Performance Constraints**: Success rate thresholds, execution time limits
- **Resource Constraints**: Cost per request limits and resource allocation
- **Conflict Detection**: Automatic detection and resolution of constraint conflicts

#### Routing Decision Framework

**Comprehensive Decision Tracking**: Full visibility into routing decisions with structured reasoning
```python
from cognivault.routing.resource_optimizer import ResourceOptimizer
from cognivault.routing.routing_decision import RoutingDecision

optimizer = ResourceOptimizer()
decision = optimizer.select_optimal_agents(
    available_agents=["refiner", "critic", "historian", "synthesis"],
    complexity_score=0.7,
    performance_data=performance_metrics,
    strategy=OptimizationStrategy.BALANCED
)

print(f"Selected agents: {decision.selected_agents}")
print(f"Confidence: {decision.confidence_score:.2f}")
print(f"Strategy: {decision.routing_strategy}")
print(f"Reasoning: {decision.reasoning.strategy_rationale}")
```

**Performance Prediction & Analytics**: Sophisticated forecasting for execution planning
- **Execution Time Prediction**: Statistical models for accurate time estimation
- **Success Probability Calculation**: Historical data-driven success rate forecasting
- **Resource Utilization Estimates**: Memory, CPU, and cost projections
- **Parallel Execution Optimization**: Automatic detection of parallelization opportunities

**Risk Assessment & Mitigation**: Proactive risk identification and management
- **Risk Identification**: Automatic detection of potential failure scenarios
- **Mitigation Strategies**: Suggested fallback options and alternative approaches
- **Confidence Scoring**: Multi-factor confidence assessment for decision reliability
- **Fallback Planning**: Comprehensive backup execution paths

#### Usage Examples

**Basic Enhanced Routing**: Automatic optimal agent selection
```bash
# Enhanced routing with context analysis (default)
make run QUESTION="What are the implications of quantum computing for cryptography?"

# Force specific optimization strategy
make run QUESTION="Simple question" OPTIMIZATION_STRATEGY=MINIMAL
```

**Advanced Constraint Scenarios**: Complex routing with specific requirements
```bash
# Required agents with performance constraints
make run QUESTION="Complex analysis" REQUIRED_AGENTS=refiner,historian MIN_SUCCESS_RATE=0.9

# Forbidden agents with time limits
make run QUESTION="Quick analysis" FORBIDDEN_AGENTS=historian MAX_EXECUTION_TIME=5000
```

#### Architecture Benefits

The enhanced routing system provides:
- **30-40% Performance Improvement**: Optimal agent selection reduces unnecessary computation
- **Intelligent Resource Management**: Dynamic allocation based on query requirements
- **Predictable Execution**: Reliable time and success rate forecasting
- **Reliable**: Comprehensive error handling and fallback mechanisms
- **Event-Driven Integration**: Full event emission for monitoring and analytics

**Note**: The routing system uses sophisticated algorithmic approaches including regex pattern matching, weighted scoring, and constraint satisfaction algorithms - not machine learning or AI models.

### 🛡️ Enterprise Error Handling & Agent Resilience (OPERATIONAL)

CogniVault features a comprehensive OPERATIONAL error handling system designed for reliability and future LangGraph DAG compatibility. The system provides structured exception hierarchies, agent-isolated error boundaries, and sophisticated retry mechanisms.

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

**Prerequisites**: Make sure you have [Poetry installed](#-requirements) before proceeding.

### Quick Setup

To get started quickly:

```bash
# Clone the repository
git clone https://github.com/aucontraire/cognivault.git
cd cognivault

# Install dependencies with Poetry
poetry install

# Run setup script (creates environment and installs git hooks)
bash setup.sh
```

The `setup.sh` script will:

- Create a Python 3.12.2 virtual environment using `pyenv`
- Install dependencies from `pyproject.toml` using Poetry
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

> **📖 For comprehensive CLI documentation, see the [CLI Usage Guide](src/cognivault/docs/CLI_USAGE.md)**
> 
> The CLI Usage Guide provides detailed information on all commands, options, troubleshooting, and use cases.

### Run the assistant

Make sure your `.env` file is configured with your OpenAI credentials if using the OpenAI LLM backend.

#### Quick Execution

To run the full pipeline with all agents:

```bash
make run QUESTION="Is democracy becoming more robust globally?"
```

#### Safe Execution with Validation

For validated execution with type checking, formatting, and test validation:

```bash
make run-safe QUESTION="Is democracy becoming more robust globally?"
```

The `run-safe` target provides **"compilation-like" safety** for Python by running:
1. **Format checking** (`black` + `ruff`) 
2. **Type checking** (`mypy`)
3. **Test validation** (all tests must pass)
4. **Application execution** (your query)

This is recommended for CI/CD pipelines and when you want to ensure code quality before execution. All the same arguments work with `run-safe` as with `run`.

This executes:

```bash
cognivault main "$(QUESTION)" $(if $(AGENTS),--agents=$(AGENTS),) $(if $(LOG_LEVEL),--log-level=$(LOG_LEVEL),) $(if $(EXPORT_MD),--export-md,) $(if $(TRACE),--trace,) $(if $(HEALTH_CHECK),--health-check,) $(if $(DRY_RUN),--dry-run,) $(if $(EXPORT_TRACE),--export-trace=$(EXPORT_TRACE),)
```

⚠️ Note: `$(QUESTION)` is a Makefile variable — this syntax only works with `make run`. If you're calling the CLI directly, use standard shell quotes:

```bash
cognivault main "What is cognition?" --agents=refiner,critic
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

CogniVault includes comprehensive CLI observability features for debugging, monitoring, and development workflows:

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

**LangGraph Mode (default)**: Uses production `LangGraphOrchestrator` with real LangGraph 0.5.1 StateGraph integration
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

**Legacy Mode (DEPRECATED - REMOVED)**: The original `AgentOrchestrator` has been deprecated (see [ADR-003](src/cognivault/docs/architecture/ADR-003-Legacy-Cleanup-And-Future-Ready-Architecture.md))
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

### 🏗️ **Advanced Graph Architecture**

**GraphFactory & Pattern System**: Extensible graph execution patterns with intelligent caching (see [Pattern Registry](src/cognivault/docs/PATTERN_REGISTRY.md))
- **Pattern-Based Construction**: Standard, parallel, and conditional graph patterns
- **Performance Optimization**: LRU cache with TTL for compiled graphs (~90% performance improvement)
- **Thread-Safe Operations**: Comprehensive validation and memory management
- **Extensible Architecture**: Clean separation of graph building vs execution orchestration

**Developer Experience Tools**: Enterprise-grade debugging and validation capabilities
- **Execution Path Tracing**: Real-time debugging with performance analysis and breakpoint support
- **Interactive DAG Explorer**: 8 comprehensive CLI commands for structure analysis and profiling
- **Pattern Validation Framework**: 7 validation commands with multi-format output support
- **Advanced Test Coverage**: 86% coverage with systematic cluster analysis and comprehensive edge case testing

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

> **📖 For complete CLI command documentation, see the [CLI Usage Guide](src/cognivault/docs/CLI_USAGE.md)**

CogniVault includes comprehensive diagnostics capabilities accessible via the `cognivault diagnostics` command suite, enhanced with advanced developer experience tools:

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

### Pattern Validation & Development Tools

**Advanced Developer Tools**: Comprehensive pattern validation and development tools:

```bash
# Validate built-in patterns
cognivault diagnostics patterns validate standard
cognivault diagnostics patterns validate conditional --level comprehensive

# Test custom patterns
cognivault diagnostics patterns validate /path/to/custom_pattern.py --format json

# Pattern discovery and certification
cognivault diagnostics patterns discover --path ./patterns --validate
cognivault diagnostics patterns certify /path/to/pattern.py --cert-output pattern.cert

# Performance benchmarking
cognivault diagnostics patterns benchmark standard --baseline parallel --runs 10

# Interactive DAG exploration
cognivault diagnostics dag-explorer explore --pattern conditional --agents refiner,synthesis
cognivault diagnostics dag-explorer performance --runs 5
cognivault diagnostics dag-explorer interactive

# Execution tracing and debugging
cognivault diagnostics execution-tracer debug --query "Test execution" --breakpoints refiner,synthesis
cognivault diagnostics execution-tracer compare --baseline-file trace1.json --comparison-file trace2.json
```

### Developer Experience Features

**Pattern Validation Framework**:
- Structural validation (missing methods, signatures)
- Semantic validation (runtime behavior testing)
- Performance validation (timing and efficiency checks)
- Security validation (anti-pattern detection)

**Interactive DAG Explorer**:
- Real-time DAG structure visualization
- Performance profiling and bottleneck identification
- Pattern comparison and analysis tools
- Comprehensive benchmarking suite

**Execution Path Tracer**:
- Real-time execution debugging with breakpoints
- Performance analysis and timing measurements
- Trace comparison and statistical analysis
- Session management and replay capabilities

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
- **195 new Phase 1 tests** covering configurable prompt composition and comprehensive agent testing
- 86% test coverage across all modules with critical paths at 100% (3,454+ total tests)
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
poetry run pytest --cov=cognivault --cov-report=term-missing tests/
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

## 🎯 Pydantic AI Testing & Validation

CogniVault includes comprehensive testing tools for the structured data pipeline, validating the complete Pydantic AI integration from component setup through end-to-end database integration.

### Testing Tools

**Component Validation**:
```bash
# Quick validation of Pydantic AI setup and configuration
python scripts/validate_pydantic_ai_setup.py
```
- Validates Pydantic AI imports and model instantiation
- Tests structured LLM wrapper creation
- Verifies agent initialization with structured support
- Optional API call testing for live validation

**Integration Testing**:
```bash
# End-to-end pipeline test with real OpenAI API calls
python scripts/test_pydantic_ai_integration.py
```
- Tests structured LLM response generation
- Validates database storage of structured JSONB metadata
- Verifies JSONB query helper methods
- Confirms data consistency throughout pipeline

**Comprehensive Testing**:
```bash
# Full pytest integration test suite
pytest tests/integration/test_pydantic_ai_database_integration.py -v -s
```
- Real OpenAI API integration testing
- Performance benchmarking (structured vs unstructured)
- Fallback behavior validation
- Database analytics query testing

### Validation Features

**Pipeline Validation**:
- **Type Safety**: Ensures consistent Pydantic model validation
- **Database Integration**: JSONB storage and retrieval testing
- **Performance Analysis**: ~20-30% overhead measurement for structured calls
- **Fallback Testing**: Graceful degradation when validation fails

**Production Readiness**:
- **End-to-End Testing**: Complete workflow validation with real LLM calls
- **Error Handling**: Comprehensive validation failure scenario testing
- **Analytics Verification**: JSONB query methods for structured data insights
- **Backward Compatibility**: Ensures existing workflows continue operating

The testing framework validates that structured agent outputs are properly stored, efficiently queryable, and maintain consistency while preserving the flexibility needed for agent-swapping architectures.

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

Unlike typical LLM assistants or AutoGPT-style agents, CogniVault focuses on *structured introspection* rather than task completion (see [LANDSCAPE.md](src/cognivault/docs/LANDSCAPE.md) for competitive positioning). While tools like LangGraph or Reflexion optimize for task-solving or dynamic planning, CogniVault enables long-term insight formation across modular agent roles grounded in cognitive science research (see [RESEARCH.md](src/cognivault/docs/RESEARCH.md)).

It’s designed as a memory-enhanced thinking partner that integrates cleanly with personal wikis, supports test-driven CLI use, and remains light enough for future microservice deployment or API integration.

---

## 🔭 Roadmap

### ✅ **Phase 1A Complete** - API Service Layer Foundation
- [x] Agent toggles via CLI (`--agents=name1,name2`)
- [x] Asynchronous agent execution
- [x] **Enterprise Pydantic Migration**: Complete migration to modern Pydantic v2 BaseModels with enterprise-grade validation
- [x] **Configurable Agent System**: Dynamic behavior modification through YAML workflows and environment variables
- [x] **Advanced Graph Architecture**: GraphFactory with pattern-based construction and performance optimization (OPERATIONAL)
- [x] **Developer Experience Tools**: Execution tracing, DAG exploration, and pattern validation framework (OPERATIONAL)
- [x] **Performance Optimization**: Graph compilation caching with 90% improvement and statistical benchmarking (OPERATIONAL)
- [x] **Enhanced Routing System**: Algorithm-driven agent selection with resource optimization and constraint validation (OPERATIONAL)
- [x] **TypedDict State Management**: Type-safe state schemas with mypy compliance (OPERATIONAL)
- [x] **Circuit Breaker Patterns**: Robust error handling and resilience throughout the system (OPERATIONAL)
- [x] Markdown exporter for wiki integration
- [x] **FastAPI Service Layer**: 9 production-ready API endpoints with comprehensive testing
- [x] **WebSocket Real-Time Streaming**: Live workflow progress updates with correlation ID filtering
- [x] **Docker Development Environment**: Expert-validated production-grade development setup
- [x] **API Manual Testing Guide**: Complete testing documentation for real-world scenarios

### ✅ **Phase 1B Complete** - Database Integration & Structured Data Pipeline
*See [PHASE_1B_DATABASE_COMPLETION.md](src/cognivault/docs/github/PHASE_1B_DATABASE_COMPLETION.md) for comprehensive completion documentation*

- [x] **PostgreSQL + pgvector Integration**: Production-ready database layer with vector embeddings
- [x] **Pydantic AI Structured Data Pipeline**: Type-safe agent outputs with JSONB analytics
- [x] **Repository Pattern**: Complete CRUD operations with 78+ comprehensive tests
- [x] **Database Query Optimization**: Sub-500ms analytics queries with 8 specialized helper methods
- [x] **Integration Testing**: End-to-end validation tools and comprehensive test coverage
- [x] **Production Validation**: Performance benchmarking and fallback behavior testing

### 🎯 **Current Phase 1B+** - Authentication & Enhanced Features
- [ ] **API Authentication System**: API key management with rate limiting and usage tracking
- [ ] **Enhanced TopicAgent**: Semantic embeddings with text-embedding-3-large integration
- [ ] **Production Deployment Features**: Multi-stage Docker builds and comprehensive health checks

### 🔮 **Future Phases** - Platform Evolution
- [ ] **Advanced Conditional Routing**: Custom graph builders and sophisticated routing patterns
- [ ] **GraphRAG Knowledge System**: Advanced knowledge graph exploration and semantic relationships
- [ ] **Community Plugin Architecture**: Framework for community-contributed agents and configurations
- [ ] **Enterprise Features**: RBAC, multi-tenant workspaces, advanced analytics
- [ ] **Streamlit UI or Jupyter notebook support**: Interactive web interfaces for workflow management

---

## 🛠 Built With

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-0.0.14-brightgreen)
![Typer](https://img.shields.io/badge/CLI-Typer-green)
![Pytest](https://img.shields.io/badge/Testing-Pytest-blueviolet)
![AGPL](https://img.shields.io/badge/License-AGPL_3.0-orange)

---

## 🤝 Contributing

We welcome contributions to CogniVault! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information on:

- **Code Quality Standards**: Type safety, Pydantic models, automated formatting
- **Development Workflow**: Git hooks, testing requirements, review process  
- **Architecture Guidelines**: Design patterns, error handling, documentation standards
- **Getting Started**: Setup instructions, development commands, debugging tools

The project maintains high standards with 86% test coverage, 100% mypy compliance, and comprehensive automation to ensure code quality.

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
See the [LICENSE](./LICENSE) file for full terms.