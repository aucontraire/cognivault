# CogniVault CLI Usage Guide

This comprehensive guide explains how to use the `cognivault` command-line interface for the production-ready CogniVault multi-agent workflow platform. The CLI provides complete access to the 4-agent pipeline, advanced orchestration features, and comprehensive diagnostic capabilities.

## Table of Contents

- [Global Command Setup](#global-command-setup)
- [Main Commands](#main-commands)
  - [Basic Agent Execution](#basic-agent-execution)
  - [Advanced Options](#advanced-options)
  - [Execution Modes](#execution-modes)
  - [Observability Features](#observability-features)
  - [Memory & Checkpointing](#memory--checkpointing)
  - [API Mode](#api-mode)
- [Diagnostic Commands](#diagnostic-commands)
  - [System Health](#system-health)
  - [Performance Metrics](#performance-metrics)
  - [Agent Diagnostics](#agent-diagnostics)
  - [Configuration Validation](#configuration-validation)
  - [Pattern Validation](#pattern-validation)
- [Migration Guide](#migration-guide)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)

---

## Global Command Setup

After cloning and setting up the repository with Poetry, you can use the `cognivault` script instead of complex `poetry run python -m` commands.

The script automatically:
- Sets up the correct `PYTHONPATH`
- Handles module execution
- Provides a clean CLI interface
- Works from any directory within the project

### Installation

```bash
# Clone and setup
git clone https://github.com/aucontraire/cognivault.git
cd cognivault
poetry install
bash setup.sh

# Verify installation
./cognivault --help
```

---

## Main Commands

### Command Structure

```bash
cognivault main [QUERY] [OPTIONS]
cognivault diagnostics [COMMAND] [OPTIONS]
```

### Basic Agent Execution

#### Run Full Pipeline

**Standard Agent Pipeline:**
```bash
# Run complete 4-agent workflow with real LLM integration
cognivault main "What are the implications of AI governance?"

# Equivalent using make (for complex queries with environment variables)
make run QUESTION="What are the implications of AI governance?"
```

**Declarative Workflow Execution:**
```bash
# Run using predefined YAML workflow definitions
cognivault workflow run examples/workflows/enhanced_prompts_example.yaml --query "What is the most complete protein?" --format json --export-md

```

**Workflow Management:**
```bash
# List available workflows
cognivault workflow list

# Validate workflow definition
cognivault workflow validate examples/workflows/my_custom_workflow.yaml

# Show workflow structure and node details
cognivault workflow show examples/workflows/enhanced_prompts_example.yaml
```

The full pipeline executes: **Refiner** → **Critic** + **Historian** (parallel) → **Synthesis** with ~25-30 second execution time using real OpenAI integration. Declarative workflows support advanced node types (Decision, Aggregator, Validator, Terminator) with custom routing and composition logic.

#### Run Specific Agents
```bash
# Run single agent
cognivault main "Your question" --agents refiner

# Run multiple agents
cognivault main "Your question" --agents refiner,critic,historian

# Run synthesis only (requires previous agent outputs)
cognivault main "Your question" --agents synthesis
```

### Advanced Options

#### Logging and Debug
```bash
# Set log level
cognivault main "Your question" --log-level DEBUG
cognivault main "Your question" --log-level WARNING

# Enable detailed execution trace
cognivault main "Your question" --trace

# Export trace to JSON file
cognivault main "Your question" --export-trace /tmp/trace.json
```

#### Output Export
```bash
# Export to markdown file
cognivault main "Your question" --export-md

# Combined trace and markdown export
cognivault main "Your question" --trace --export-md --export-trace /tmp/trace.json
```

### Execution Modes

CogniVault supports multiple execution modes for different use cases:

#### LangGraph Mode (Default - Production)
```bash
# Default execution mode (LangGraph StateGraph orchestration)
cognivault main "Your question"

# Explicit specification
cognivault main "Your question" --execution-mode langgraph-real
```

**Note**: CogniVault V1 uses LangGraph StateGraph orchestration as the primary execution mode. The system provides backward compatibility but defaults to the production-ready LangGraph implementation.

#### Performance Comparison
```bash
# Compare execution modes side-by-side
cognivault main "Your question" --compare-modes

# Statistical benchmarking with multiple runs
cognivault main "Your question" --compare-modes --benchmark-runs 5

# Comprehensive comparison with trace export
cognivault main "Your question" --compare-modes --benchmark-runs 3 --export-trace /tmp/comparison.json
```

### Observability Features

#### Health Checks
```bash
# Run health checks without executing pipeline
cognivault main "Any question" --health-check

# Check specific agents
cognivault main "Any question" --agents refiner,critic --health-check
```

#### Dry Run Validation
```bash
# Validate pipeline configuration without execution
cognivault main "Your question" --dry-run

# Validate specific agent subset
cognivault main "Your question" --agents refiner,historian --dry-run
```

#### DAG Visualization
```bash
# Output DAG to console
cognivault main "Your question" --visualize-dag stdout

# Export DAG to file
cognivault main "Your question" --visualize-dag dag.md

# Visualization only (no execution)
cognivault main "" --visualize-dag stdout
```

### Memory & Checkpointing

#### Enable Checkpointing
```bash
# Enable with auto-generated thread ID
cognivault main "Your question" --enable-checkpoints

# Use custom thread ID for session scoping
cognivault main "Your question" --enable-checkpoints --thread-id my-session
```

#### Rollback Mechanisms
```bash
# Rollback to last checkpoint on failure
cognivault main "Your question" --enable-checkpoints --rollback-last-checkpoint

# Resume specific thread session
cognivault main "Continue analysis" --enable-checkpoints --thread-id research-session
```

### API Mode

#### API Layer Testing
```bash
# Use API layer instead of direct orchestrator
cognivault main "Your question" --use-api

# Specify API mode
cognivault main "Your question" --use-api --api-mode real
cognivault main "Your question" --use-api --api-mode mock
```

---

## Diagnostic Commands

### System Health

#### Basic Health Check
```bash
# Quick health overview
cognivault diagnostics health

# JSON output for automation
cognivault diagnostics health --format json

# Quiet mode (exit codes only)
cognivault diagnostics health --quiet
```

### Performance Metrics

#### Metrics Overview
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

### Agent Diagnostics

#### Agent Status
```bash
# All agents status
cognivault diagnostics agents

# Specific agent details
cognivault diagnostics agents --agent refiner --json
```

### Configuration Validation

#### Configuration Check
```bash
# Configuration overview
cognivault diagnostics config

# Validation only
cognivault diagnostics config --validate

# JSON output
cognivault diagnostics config --json
```

### System Status

#### Detailed Status
```bash
# Comprehensive system information
cognivault diagnostics status

# JSON output with custom time window
cognivault diagnostics status --json --window 60
```

#### Complete Diagnostics
```bash
# Complete system report
cognivault diagnostics full

# Export to different formats
cognivault diagnostics full --format json --output system-report.json
cognivault diagnostics full --format csv --output metrics.csv
cognivault diagnostics full --format prometheus --output metrics.prom
```

### Pattern Validation

#### Pattern Validation Commands
```bash
# Validate built-in patterns
cognivault diagnostics patterns validate standard
cognivault diagnostics patterns validate conditional --level comprehensive

# Test custom patterns
cognivault diagnostics patterns validate /path/to/custom_pattern.py --format json

# Pattern discovery
cognivault diagnostics patterns discover --path ./patterns --validate

# Pattern certification
cognivault diagnostics patterns certify /path/to/pattern.py --cert-output pattern.cert

# Performance benchmarking
cognivault diagnostics patterns benchmark standard --baseline parallel --runs 10
```

---

## Migration Guide

### From Old Commands to New Commands

| Old Command | New Command |
|-------------|-------------|
| `poetry run python -m cognivault.cli main "question"` | `cognivault main "question"` |
| `poetry run python -m cognivault.diagnostics.cli health` | `cognivault diagnostics health` |
| `poetry run python -m cognivault.diagnostics.pattern_validator validate standard` | `cognivault diagnostics patterns validate standard` |
| `PYTHONPATH=src python -m cognivault.cli main "question"` | `cognivault main "question"` |
| `PYTHONPATH=src pytest tests/` | `poetry run pytest tests/` |

### Legacy vs Modern Workflow

**Legacy approach:**
```bash
# Old way - complex PYTHONPATH setup
export PYTHONPATH=src
python -m cognivault.cli main "question" --agents refiner,critic
```

**Modern approach:**
```bash
# New way - simple and consistent
cognivault main "question" --agents refiner,critic
```

---

## Common Use Cases

### Development and Testing

#### Quick Testing
```bash
# Test single agent
cognivault main "Test question" --agents refiner --log-level DEBUG

# Dry run validation
cognivault main "Test question" --dry-run --trace
```

#### Performance Analysis
```bash
# Benchmark different execution modes
cognivault main "Performance test" --compare-modes --benchmark-runs 5

# Export detailed performance traces
cognivault main "Performance test" --trace --export-trace /tmp/perf_$(date +%s).json
```

### Production Monitoring

#### Health Monitoring
```bash
# System health check
cognivault diagnostics health --format json

# Complete system diagnostics
cognivault diagnostics full --format prometheus --output /tmp/metrics.prom
```

#### Operational Workflows
```bash
# Production execution with full observability
cognivault main "Production query" --trace --export-trace /tmp/production_trace.json --export-md

# Health check before deployment
cognivault diagnostics health --quiet && echo "System healthy"
```

### Research and Analysis

#### Long-form Analysis
```bash
# Enable checkpointing for long sessions
cognivault main "Research question" --enable-checkpoints --thread-id research-session

# Continue previous analysis
cognivault main "Follow-up question" --enable-checkpoints --thread-id research-session
```

#### Pattern Development
```bash
# Validate custom patterns
cognivault diagnostics patterns validate /path/to/my_pattern.py

# Benchmark custom pattern
cognivault diagnostics patterns benchmark my_pattern --baseline standard --runs 10
```

### Integration and Automation

#### API Integration
```bash
# Test API boundaries
cognivault main "API test" --use-api --api-mode mock

# Production API usage
cognivault main "Production query" --use-api --api-mode real
```

#### CI/CD Integration
```bash
# Automated health checks
cognivault diagnostics health --quiet || exit 1

# Validate configuration in CI
cognivault diagnostics config --validate --format json
```

---

## Troubleshooting

### Common Issues

#### Command Not Found
```bash
# Ensure cognivault script is executable
chmod +x ./cognivault

# Verify setup
bash setup.sh
```

#### Import Errors
```bash
# Reinstall dependencies
poetry install

# Check Python path
poetry env info
```

#### Performance Issues
```bash
# Check system status
cognivault diagnostics status

# Run health checks
cognivault diagnostics health
```

### Debug Workflows

#### Detailed Debugging
```bash
# Maximum verbosity
cognivault main "Debug question" --log-level DEBUG --trace --dry-run

# Export debug information
cognivault main "Debug question" --trace --export-trace /tmp/debug_trace.json
```

#### Agent-Specific Issues
```bash
# Test individual agents
cognivault main "Test" --agents refiner --health-check
cognivault main "Test" --agents critic --log-level DEBUG
```

### Performance Optimization

#### Identify Bottlenecks
```bash
# Compare execution modes
cognivault main "Performance test" --compare-modes --benchmark-runs 10

# Profile specific patterns
cognivault diagnostics patterns benchmark standard --runs 5
```

---

## Environment Variables

Key environment variables for CLI configuration:

```bash
# Execution settings
export COGNIVAULT_LOG_LEVEL=INFO
export COGNIVAULT_MAX_RETRIES=3
export COGNIVAULT_TIMEOUT_SECONDS=30

# API settings
export COGNIVAULT_API_MODE=real
export COGNIVAULT_LLM=openai

# Checkpointing
export COGNIVAULT_ENABLE_CHECKPOINTS=false
export COGNIVAULT_MAX_SNAPSHOTS=5

# Development
export COGNIVAULT_DEBUG=false
export COGNIVAULT_TRACE=false
```

---

## Benefits of the New CLI

- **Simpler**: No need to remember `PYTHONPATH` setup or complex `poetry run` commands
- **Shorter**: Fewer characters to type for common operations
- **Consistent**: Same command structure across all operations
- **Portable**: Works from any directory within the project
- **Comprehensive**: Full feature coverage with extensive help text
- **Observable**: Rich tracing, health checks, and diagnostic capabilities
- **Production-Ready**: Monitoring integration and export capabilities

---

## Getting Help

For detailed help on any command:

```bash
# Main help
cognivault --help

# Command-specific help
cognivault main --help
cognivault diagnostics --help
cognivault diagnostics patterns --help

# Show available completions
cognivault --show-completion
```

For additional support:
- Check the [troubleshooting section](../../../README.md#troubleshooting) in the main README
- Review the [architecture documentation](../architecture/)
- File issues at the project repository