# Semantic Validation Usage Guide

## Overview

CogniVault's semantic validation layer provides domain-specific validation for agent workflows, ensuring that agent combinations and patterns create semantically meaningful executions.

## Basic Usage

### 1. Creating a Validator

```python
from cognivault.langgraph_backend import CogniVaultValidator, create_default_validator

# Create a permissive validator (warnings for issues)
validator = CogniVaultValidator(strict_mode=False)

# Create a strict validator (errors for issues)
strict_validator = CogniVaultValidator(strict_mode=True)

# Create default validator
default_validator = create_default_validator(strict_mode=False)
```

### 2. Using with GraphFactory

```python
from cognivault.langgraph_backend import GraphFactory, GraphConfig

# Create factory with default validator
factory = GraphFactory(default_validator=validator)

# Create graph with validation enabled
config = GraphConfig(
    agents_to_run=["refiner", "critic", "historian", "synthesis"],
    pattern_name="standard",
    enable_validation=True,  # Enable validation
    cache_enabled=True
)

try:
    graph = factory.create_graph(config)
    print("Graph created successfully!")
except GraphBuildError as e:
    print(f"Validation failed: {e}")
```

### 3. Standalone Validation

```python
# Validate workflow without building graph
result = factory.validate_workflow(
    agents=["synthesis"],  # Potentially problematic
    pattern="standard",
    strict_mode=True
)

if result.is_valid:
    print("Workflow is valid")
else:
    print("Validation errors:")
    for error in result.error_messages:
        print(f"  - {error}")
    
    print("Validation warnings:")
    for warning in result.warning_messages:
        print(f"  - {warning}")
```

## Validation Rules

### Standard Pattern Rules

1. **Synthesis without Analysis**: Synthesis agent should have analysis agents (critic/historian) providing input
2. **Complex workflows without Refiner**: Multi-agent workflows benefit from refiner preprocessing
3. **Unknown agents**: Unknown agents may not integrate properly with CogniVault

### Parallel Pattern Rules

1. **Single agent in parallel**: Single agents don't benefit from parallelization
2. **Good parallel candidates**: Multiple analysis agents can execute concurrently

### Conditional Pattern Rules

1. **Missing refiner**: Conditional patterns typically benefit from refiner as entry point
2. **Limited branching**: Few agents may not utilize dynamic routing effectively

## Configuration Options

### GraphConfig Validation Settings

```python
config = GraphConfig(
    agents_to_run=["refiner", "synthesis"],
    pattern_name="standard",
    enable_validation=True,           # Enable/disable validation
    validator=custom_validator,       # Override default validator
    validation_strict_mode=False,     # Strict mode for this validation
    cache_enabled=True
)
```

### Validation Modes

- **Permissive Mode** (`strict_mode=False`): Issues reported as warnings, graph building continues
- **Strict Mode** (`strict_mode=True`): Issues reported as errors, graph building fails

## Custom Validators

### Creating Custom Validators

```python
from cognivault.langgraph_backend import SemanticValidator, ValidationResult, ValidationSeverity

class CustomValidator(SemanticValidator):
    def validate_workflow(self, agents, pattern, **kwargs):
        result = ValidationResult(is_valid=True, issues=[])
        
        # Custom validation logic
        if "custom_agent" in agents and pattern != "custom_pattern":
            result.add_issue(
                ValidationSeverity.WARNING,
                "custom_agent works best with custom_pattern",
                agent="custom_agent",
                suggestion="Use custom_pattern for optimal results",
                code="CUSTOM_AGENT_PATTERN_MISMATCH"
            )
        
        return result
    
    def get_supported_patterns(self):
        return {"standard", "parallel", "conditional", "custom_pattern"}

# Use custom validator
factory = GraphFactory(default_validator=CustomValidator())
```

## Example Workflows

### Valid Workflows

```python
# Full 4-agent pipeline
valid_workflows = [
    {
        "agents": ["refiner", "critic", "historian", "synthesis"],
        "pattern": "standard",
        "description": "Complete analysis pipeline"
    },
    {
        "agents": ["refiner", "synthesis"],
        "pattern": "standard", 
        "description": "Simple preprocessing + synthesis"
    },
    {
        "agents": ["critic", "historian", "synthesis"],
        "pattern": "parallel",
        "description": "Parallel analysis with synthesis"
    }
]
```

### Problematic Workflows (Strict Mode)

```python
# These will fail in strict mode
problematic_workflows = [
    {
        "agents": ["synthesis"],
        "pattern": "standard",
        "issue": "Synthesis without analysis input"
    },
    {
        "agents": ["refiner", "unknown_agent"],
        "pattern": "standard",
        "issue": "Unknown agent not in registry"
    }
]
```

## Integration with Existing Code

### Backward Compatibility

Validation is **disabled by default** to maintain backward compatibility:

```python
# Existing code continues to work unchanged
factory = GraphFactory()
config = GraphConfig(
    agents_to_run=["refiner", "synthesis"],
    pattern_name="standard"
    # enable_validation defaults to False
)
graph = factory.create_graph(config)  # No validation performed
```

### Gradual Adoption

Enable validation gradually:

```python
# 1. Start with permissive validation for warnings
factory = GraphFactory(default_validator=CogniVaultValidator(strict_mode=False))

# 2. Enable validation on specific workflows
config = GraphConfig(
    agents_to_run=["refiner", "critic", "synthesis"],
    enable_validation=True,  # Enable for this workflow
    validation_strict_mode=False  # Warnings only
)

# 3. Move to strict validation for new workflows
strict_config = GraphConfig(
    agents_to_run=["refiner", "critic", "historian", "synthesis"],
    enable_validation=True,
    validation_strict_mode=True  # Fail on issues
)
```

## Best Practices

1. **Start Permissive**: Use non-strict mode initially to identify potential issues
2. **Review Warnings**: Address validation warnings to improve workflow quality  
3. **Strict for Production**: Use strict mode for production workflows
4. **Custom Validators**: Create domain-specific validators for specialized use cases
5. **Validation in Tests**: Include validation in automated tests to catch workflow issues early

## Performance Considerations

- Validation adds minimal overhead (~1-5ms per workflow)
- Validation is bypassed when using cached graphs
- Complex custom validators may impact performance
- Validation results can be cached for repeated workflows

## Troubleshooting

### Common Issues

1. **"Validation enabled but no validator available"**: Set a default validator or disable validation
2. **"Synthesis without analysis"**: Add critic and/or historian agents before synthesis
3. **"Unknown agent"**: Ensure agent is registered in GraphFactory.node_functions

### Debug Validation

```python
# Get detailed validation results
result = factory.validate_workflow(agents, pattern)

for issue in result.issues:
    print(f"{issue.severity.value}: {issue.message}")
    if issue.suggestion:
        print(f"  Suggestion: {issue.suggestion}")
    if issue.code:
        print(f"  Code: {issue.code}")
```