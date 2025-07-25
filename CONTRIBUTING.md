# Contributing to CogniVault

Thank you for your interest in contributing to CogniVault! This document outlines the development practices, standards, and workflows we use to maintain code quality and ensure a smooth collaboration experience.

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (tested with Python 3.12.2)
- **Poetry** for dependency management
- **Git** for version control

### Setup

```bash
# Clone the repository
git clone https://github.com/aucontraire/cognivault.git
cd cognivault

# Install dependencies and setup environment
poetry install
bash setup.sh  # Creates environment, installs git hooks

# Verify setup
make check     # Run format + typecheck
make test      # Run test suite
```

## 📋 Development Standards

### Code Quality Requirements

CogniVault maintains high code quality standards through automated tooling and strict practices:

#### Type Safety (Required)
- **100% mypy compliance** - All code must pass strict type checking
- **No `type: ignore` comments** - Find proper type solutions instead
- **Comprehensive type annotations** - All functions, methods, and variables must be typed
- **Pydantic models** for data validation and serialization

```python
# Good: Properly typed with Pydantic
from pydantic import BaseModel
from typing import List, Optional

class AgentConfig(BaseModel):
    name: str
    enabled: bool = True
    parameters: Optional[Dict[str, Any]] = None

async def process_agent(config: AgentConfig) -> AgentResult:
    # Implementation with full type safety
    pass

# Bad: Missing types, using Any
def process_agent(config):
    # Type checker cannot validate this
    pass
```

#### Code Formatting (Automated)
- **Black** for code formatting
- **Ruff** for linting and import sorting
- **Pre-commit hooks** enforce formatting automatically

#### Data Validation Standards
- **Pydantic v2** for all data models
- **Field validation** with constraints and descriptions
- **Runtime type safety** with clear error messages

```python
from pydantic import BaseModel, Field
from typing import Literal

class CriticConfig(BaseModel):
    """Configuration for Critic agent behavior."""
    analysis_depth: Literal["shallow", "medium", "deep"] = Field(
        default="medium",
        description="Depth of critical analysis to perform"
    )
    confidence_reporting: bool = Field(
        default=True,
        description="Whether to include confidence scores in output"
    )
```

### Testing Requirements

#### Comprehensive Test Coverage
- **Minimum 85% code coverage** - Current: 86% with 3,454+ tests
- **All new features must include tests**
- **Test both success and failure scenarios**
- **Mock external dependencies appropriately**

#### Test Categories
1. **Unit Tests** (`tests/unit/`) - Individual component testing
2. **Integration Tests** (`tests/integration/`) - Component interaction testing
3. **Contract Tests** (`tests/contracts/`) - API boundary validation
4. **Mock Implementations** (`tests/fakes/`) - Test doubles and stubs

#### Testing Best Practices
```python
import pytest
from unittest.mock import Mock, patch
from cognivault.agents.refiner.agent import RefinerAgent

class TestRefinerAgent:
    """Test RefinerAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_refiner_success_case(self):
        """Test successful query refinement."""
        # Arrange
        mock_llm = Mock()
        mock_llm.generate.return_value = "Refined query"
        agent = RefinerAgent(mock_llm)
        
        # Act
        result = await agent.run(context)
        
        # Assert
        assert result.success
        assert "Refined query" in result.output
        mock_llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_refiner_failure_handling(self):
        """Test error handling in query refinement."""
        # Test error scenarios
        pass
```

#### Running Tests
```bash
# Run all tests
make test

# Run with coverage
make coverage-all

# Run specific module
make coverage-one m=agents

# Run tests with debug output
LOG_LEVEL=DEBUG make test
```

## 🔄 Git Workflow

### Branch Strategy
- **`master` branch** - Stable development code
- **Feature branches** - `feature/description-of-feature`
- **Bug fixes** - `fix/description-of-fix`
- **Documentation** - `docs/description-of-change`

### Commit Standards
Follow conventional commit format:

```
type: brief description

Longer description if needed.

- List specific changes
- Include breaking changes
- Reference issues (#123)
```

**Commit Types:**
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code formatting (no logic changes)
- `refactor:` - Code restructuring (no feature changes)
- `test:` - Test additions or modifications
- `chore:` - Maintenance tasks

### Git Hooks (Automated)
The project includes automated git hooks:

- **Pre-commit**: Runs `make format` and `make typecheck`
- **Pre-push**: Runs `make test`

These hooks **will block commits/pushes** if quality checks fail.

### Pull Request Process

1. **Create feature branch** from `master`
2. **Make changes** following coding standards
3. **Ensure all checks pass**:
   ```bash
   make check     # Format + typecheck
   make test      # All tests pass
   ```
4. **Create pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Test coverage for new features
   - Documentation updates if needed

## 🏗️ Architecture Guidelines

### Design Principles

Follow the established architectural patterns:

#### Clean Separation of Concerns
- **Agents** - Single responsibility cognitive functions
- **Orchestration** - Workflow coordination and execution
- **Configuration** - Centralized settings management
- **Events** - Observability and monitoring
- **API Boundaries** - Clear external/internal separation

#### Dependency Management
- **Dependency injection** via constructor parameters
- **Interface-based design** for swappable components
- **Factory patterns** for object creation
- **Registry patterns** for dynamic component discovery

#### Error Handling
- **Structured exceptions** with context and metadata
- **Circuit breaker patterns** for resilience
- **Graceful degradation** when possible
- **Comprehensive logging** for debugging

### Code Organization

```
src/cognivault/
├── agents/          # Agent implementations
├── orchestration/   # Workflow orchestration
├── config/          # Configuration management
├── events/          # Event system
├── llm/             # LLM abstractions
├── diagnostics/     # Developer tools
└── cli/             # Command-line interface
```

### Adding New Components

#### New Agent Example
```python
from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
from pydantic import BaseModel

class MyAgentConfig(BaseModel):
    """Configuration for MyAgent."""
    setting: str = "default"

class MyAgent(BaseAgent):
    """Custom agent implementation."""
    
    def __init__(self, llm: LLMInterface, config: MyAgentConfig):
        super().__init__(llm)
        self.config = config
    
    async def run(self, context: AgentContext) -> AgentContext:
        """Execute agent logic."""
        # Implementation
        return context
```

#### Configuration Integration
```python
# Add to config/agent_configs.py
from .base_config import BaseAgentConfig

class MyAgentConfig(BaseAgentConfig):
    """MyAgent specific configuration."""
    custom_setting: str = Field(
        default="value",
        description="Description of custom setting"
    )
```

## 📚 Documentation Standards

### Code Documentation
- **Docstrings** for all public classes and methods
- **Type hints** for all parameters and return values
- **Inline comments** for complex logic
- **ADRs** for architectural decisions

### Documentation Format
Use Google/NumPy docstring style:

```python
def process_query(query: str, config: ProcessConfig) -> ProcessResult:
    """Process a user query through the agent pipeline.
    
    Args:
        query: The user's input query to process
        config: Configuration parameters for processing
        
    Returns:
        ProcessResult containing the processed output and metadata
        
    Raises:
        ProcessingError: If query processing fails
        ConfigurationError: If config is invalid
    """
```

### Architecture Documentation
- **Update ADRs** for significant architectural changes
- **Update README.md** for user-facing changes
- **Add examples** for new features
- **Link related documentation**

## 🧪 Development Workflow

### Development Commands

```bash
# Quality checks
make check          # Format + typecheck
make format         # Black + ruff formatting
make typecheck      # mypy type checking
make lint           # ruff linting only

# Testing
make test           # Run all tests
make coverage-all   # Full coverage report
make test-agent-refiner  # Agent-specific testing

# Development
make run QUESTION="test query"  # Run application
make run-safe QUESTION="test"   # Run with validation
cognivault diagnostics health   # System health check
```

### Debugging and Diagnostics

```bash
# Debug execution
make run QUESTION="test" LOG_LEVEL=DEBUG TRACE=1

# System diagnostics
cognivault diagnostics health
cognivault diagnostics metrics
cognivault diagnostics patterns validate standard

# Performance analysis
make run QUESTION="test" COMPARE_MODES=1 BENCHMARK_RUNS=5
```

### Environment Management

```bash
# Poetry environment
poetry shell                    # Activate environment
poetry install                  # Install dependencies
poetry add package_name         # Add new dependency
poetry run python script.py     # Run in environment

# Environment validation
python -c "import cognivault; print('Setup successful')"
make check                      # Verify all tools work
```

## 🔍 Review Guidelines

### Code Review Checklist

**Functionality:**
- [ ] Code solves the intended problem
- [ ] Edge cases are handled appropriately
- [ ] Error handling is comprehensive
- [ ] Performance implications are considered

**Code Quality:**
- [ ] Follows established patterns and conventions
- [ ] Type annotations are complete and accurate
- [ ] Pydantic models are used for data validation
- [ ] No mypy errors or warnings

**Testing:**
- [ ] Comprehensive test coverage (aim for >85%)
- [ ] Tests cover both success and failure scenarios
- [ ] Mock usage is appropriate and clean
- [ ] Integration tests for component interactions

**Documentation:**
- [ ] Public APIs are documented
- [ ] Complex logic has explanatory comments
- [ ] README updated for user-facing changes
- [ ] ADRs updated for architectural changes

### Review Process

1. **Automated Checks** - All CI checks must pass
2. **Code Review** - At least one approving review required
3. **Testing** - All tests must pass with adequate coverage
4. **Documentation** - Updates for user or developer-facing changes

## 🚨 Common Issues and Solutions

### MyPy Errors
```bash
# Fix import issues
poetry run mypy --show-error-codes src/

# Common fixes:
- Add type annotations
- Use proper Pydantic models
- Import types correctly
- Use Union[] for multiple types
```

### Test Failures
```bash
# Debug test failures
LOG_LEVEL=DEBUG make test

# Common issues:
- Mock setup incorrect
- Async/await patterns wrong
- Test isolation problems
- Environment dependencies
```

### Git Hook Failures
```bash
# Fix formatting issues
make format

# Fix type checking
make typecheck

# Skip hooks (emergency only)
git commit --no-verify
```

## 🤝 Community Guidelines

### Communication
- **Be respectful** and constructive in all interactions
- **Ask questions** when requirements are unclear
- **Share knowledge** and help other contributors
- **Follow up** on your pull requests and issues

### Getting Help
- **Check documentation** first (README, architecture docs, etc.)
- **Search existing issues** before creating new ones
- **Provide context** when asking questions
- **Include error messages** and relevant code

### Issue Reporting
When reporting bugs or requesting features:

1. **Search existing issues** first
2. **Use appropriate templates** (if available)
3. **Provide reproduction steps** for bugs
4. **Include environment information**
5. **Be specific** about expected vs actual behavior

## 📊 Project Status

**Current State:**
- 86% test coverage with 3,454+ tests
- Comprehensive type safety with mypy
- Sophisticated multi-agent orchestration system
- Research-grade foundation for advanced applications

**Focus Areas for Contributors:**
- Additional agent implementations
- Enhanced routing algorithms
- Performance optimizations
- Documentation improvements
- Integration testing
- Real-world use case examples

---

**Thank you for contributing to CogniVault!** Your efforts help advance the state of multi-agent cognitive systems and collaborative AI research.

For questions about contributing, please open an issue or start a discussion in the repository.