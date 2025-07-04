# 🧠 CogniVault

![Python](https://img.shields.io/badge/python-3.12-blue)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Markdown Export](https://img.shields.io/badge/markdown-export-green)
![Wiki Ready](https://img.shields.io/badge/wiki-ready-blueviolet)

CogniVault is a modular, CLI-based multi-agent assistant designed to help you reflect, refine, and organize your thoughts through structured dialogue and cumulative insight. It simulates a memory-augmented thinking partner, enabling long-term knowledge building across multiple agent perspectives.

## 🎯 Recent Achievements

**LangGraph-Ready Foundation Complete** - CogniVault now features enterprise-grade error handling and state management designed for seamless future migration to LangGraph DAG-based orchestration:

✅ **Issue 1: Exception Package & Trace-Ready Hierarchy** - 100% Complete  
- Comprehensive `src/cognivault/exceptions/` package with organized modules
- Structured exception hierarchy with LangGraph-compatible error routing  
- Trace-compatible metadata (step_id, agent_id, timestamp) for all operations
- Agent boundary isolation and error severity classification

✅ **Issue 2: Agent-Level Error Handling & Node-Ready Design** - 100% Complete  
- Agent-isolated error boundaries with retry logic moved into individual agents
- Configurable retry policies and circuit breakers per agent type
- OpenAI-specific error mapping with comprehensive LLM exception handling
- Agent execution tracing with structured metadata for DAG compatibility

✅ **Issue 3: Context State Management & Reversible Transitions** - 98% Complete  
- Enhanced context system with reversible state transitions via snapshots
- Agent-isolated context mutations preventing shared global state issues
- Structured trace schema as sidecar context field
- Execution flow tracking with node_id and edge metadata

**Test Coverage Achievement**: Improved from 86% → 98% (+12 percentage points) with all critical modules at 98-100% coverage.

---

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
- 🔁 **Orchestrator pipeline** supports dynamic agent control
- 📄 **Markdown-ready output** for integration with personal wikis
- 🧪 **Full test suite** with `pytest` for all core components (98% coverage)
- 🔄 **Swappable LLM backend**: Plug-and-play support for OpenAI or stubs via configuration
- 📋 **Agent Registry**: Dynamic agent registration system for extensible architecture
- ⚙️ **Configuration Management**: Centralized configuration system with environment variables and JSON file support
- 🧠 **Enhanced Context Management**: Advanced memory management with compression, snapshots, and size monitoring
- 🛡️ **Enterprise Error Handling**: Comprehensive exception hierarchy with LangGraph-ready agent isolation
- 🔄 **Agent-Level Resilience**: Circuit breakers, retry policies, and timeout management per agent
- 📊 **Execution Tracing**: Structured metadata and trace logging for debugging and observability

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
│   │   └── synthesis/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       └── main.py
│   ├── base_agent.py
│   ├── registry.py
│   ├── cli.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── app_config.py
│   │   └── logging_config.py
│   ├── context.py
│   ├── exceptions/
│   │   ├── __init__.py
│   │   ├── agent_errors.py
│   │   ├── config_errors.py
│   │   ├── io_errors.py
│   │   ├── llm_errors.py
│   │   └── orchestration_errors.py
│   ├── docs/
│   │   ├── LANDSCAPE.md
│   │   └── RESEARCH.md
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_interface.py
│   │   ├── openai.py
│   │   └── stub.py
│   ├── logs/
│   │   └── interaction_00001.json
│   ├── notes/
│   │   ├── 2025-06-26T06-45-24_what-is-cognition.md
│   │   ├── 2025-06-26T06-47-28_what-is-cognition.md
│   │   ├── 2025-06-26T10-04-47_what-is-cognition.md
│   │   └── sample_note.md
│   ├── orchestrator.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── vector_store.py
│   └── store/
│       ├── __init__.py
│       ├── utils.py
│       └── wiki_adapter.py
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
│   └── test_registry.py
├── test_base_agent.py
├── config/
│   ├── __init__.py
│   ├── test_app_config.py
│   └── test_openai_config.py
├── exceptions/
│   ├── __init__.py
│   ├── test_agent_errors.py
│   ├── test_base_exceptions.py
│   ├── test_config_errors.py
│   ├── test_io_errors.py
│   ├── test_llm_errors.py
│   └── test_orchestration_errors.py
├── llm/
│   ├── __init__.py
│   ├── test_llm_interface.py
│   ├── test_openai.py
│   └── test_stub.py
├── store/
│   ├── __init__.py
│   ├── test_utils.py
│   └── test_wiki_adapter.py
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

The registry supports both the current architecture and future dynamic loading capabilities. See [`registry.py`](./src/cognivault/agents/registry.py) for implementation details.

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

```python
from cognivault.agents.base_agent import BaseAgent, RetryConfig, CircuitBreakerState

# Configure agent-specific retry behavior
retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    exponential_backoff=True,
    jitter=True
)

# Agent with circuit breaker protection
agent = ConcreteAgent(
    name="MyAgent",
    retry_config=retry_config,
    timeout_seconds=30.0,
    enable_circuit_breaker=True
)

# Execution with automatic retry and error isolation
result = await agent.run_with_retry(context)
```

#### Key Resilience Features

- **Circuit Breakers**: Prevent cascade failures with configurable failure thresholds
- **Exponential Backoff**: Intelligent retry delays with jitter to prevent thundering herd
- **Agent Isolation**: Error boundaries prevent one agent failure from affecting others
- **Trace Metadata**: All operations include step_id, agent_id, and timestamp for observability
- **LLM Error Mapping**: Comprehensive OpenAI error handling with structured exception conversion
- **Timeout Management**: Per-agent timeout configuration with graceful degradation

#### Structured Error Context

All exceptions include rich context for debugging and monitoring:

```python
try:
    result = await agent.run_with_retry(context)
except AgentExecutionError as e:
    print(f"Agent: {e.agent_name}")
    print(f"Step ID: {e.step_id}")
    print(f"Retry Policy: {e.retry_policy}")
    print(f"Context: {e.context}")
    print(f"User Message: {e.get_user_message()}")
```

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
PYTHONPATH=src python -m cognivault.cli "$(QUESTION)" $(if $(AGENTS),--agents=$(AGENTS),) $(if $(LOG_LEVEL),--log-level=$(LOG_LEVEL),) $(if $(EXPORT_MD),--export-md,)
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
- 98% test coverage across all modules with critical paths at 100%
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
- [ ] Markdown exporter for wiki integration
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