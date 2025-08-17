# Semantic Validation Usage Guide

**Production Documentation - v2.0.0**

## Overview

CogniVault's semantic validation layer provides production-ready domain-specific validation for agent workflows, ensuring that agent combinations and patterns create semantically meaningful executions. The validation system is fully integrated with LangGraph 0.6.4, enhanced routing capabilities, and the configurable prompt composition architecture.

**Current Status**: Semantic validation is operational with comprehensive pattern validation, agent compatibility checking, and integration with the enhanced routing system for intelligent workflow validation.

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

### 2. Using with Production GraphFactory (LangGraph 0.6.4)

```python
from cognivault.langgraph_backend import GraphFactory, GraphConfig, CacheConfig
from cognivault.langgraph_backend.exceptions import GraphBuildError

# Create factory with caching and validation
cache_config = CacheConfig(max_size=10, ttl_seconds=1800)
factory = GraphFactory(cache_config, default_validator=validator)

# Production graph with enhanced features
config = GraphConfig(
    agents_to_run=["refiner", "critic", "historian", "synthesis"],
    pattern_name="standard",
    enable_validation=True,  # Enable semantic validation
    cache_enabled=True,
    enable_checkpoints=True,  # Production checkpointing
    memory_manager=memory_manager
)

try:
    # Validation occurs during graph compilation
    compiled_graph = factory.create_graph(config)
    print("Graph created and validated successfully!")
    
    # Get validation statistics
    validation_stats = factory.get_validation_stats()
    print(f"Validation performed: {validation_stats}")
    
except GraphBuildError as e:
    print(f"Validation failed: {e}")
    print(f"Available patterns: {factory.get_available_patterns()}")
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
4. **Configuration Compatibility**: Agent configurations must be compatible with selected pattern
5. **Dependency Validation**: Agent dependencies must form valid DAG structure

### Parallel Pattern Rules

1. **Single agent in parallel**: Single agents don't benefit from parallelization
2. **Good parallel candidates**: Multiple analysis agents can execute concurrently
3. **Resource Constraints**: Parallel execution must respect memory and compute limits
4. **State Synchronization**: Parallel agents must have compatible state schemas

### Enhanced Conditional Pattern Rules

1. **Routing Prerequisites**: Enhanced routing requires context analysis capabilities
2. **Performance Data**: Conditional patterns benefit from historical performance metrics
3. **Multi-Axis Classification**: Agents should have proper metadata for intelligent routing
4. **Resource Optimization**: Routing decisions must respect execution constraints
5. **Fallback Scenarios**: Conditional patterns should define fallback execution paths

### Configurable Prompt Composition Rules

1. **Configuration Schema Validation**: Agent configurations must match Pydantic schema definitions
2. **Prompt Template Compatibility**: Custom prompts must be compatible with agent types
3. **Behavioral Constraint Validation**: Custom constraints must be semantically valid
4. **Template Variable Consistency**: Template variables must be properly defined across agents

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
from cognivault.langgraph_backend import WorkflowSemanticValidator, ValidationResult, ValidationSeverity

class CustomValidator(WorkflowSemanticValidator):
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

## Production Example Workflows

### Valid Production Workflows

```python
# Complete 4-agent pipeline with enhanced routing
production_workflows = [
    {
        "agents": ["refiner", "critic", "historian", "synthesis"],
        "pattern": "conditional",  # Enhanced routing
        "description": "Complete analysis with intelligent routing",
        "config": {
            "enable_validation": True,
            "use_enhanced_routing": True,
            "optimization_strategy": "balanced"
        }
    },
    {
        "agents": ["refiner", "synthesis"],
        "pattern": "standard", 
        "description": "Simple preprocessing + synthesis",
        "config": {
            "enable_validation": True,
            "validation_strict_mode": False
        }
    },
    {
        "agents": ["critic", "historian", "synthesis"],
        "pattern": "parallel",
        "description": "High-performance parallel analysis",
        "config": {
            "enable_validation": True,
            "resource_constraints": {
                "max_execution_time_ms": 60000,
                "max_agents": 3
            }
        }
    }
]
```

### Configurable Prompt Composition Workflows

```python
# Domain-specific workflows with validation
domain_workflows = [
    {
        "workflow_file": "src/cognivault/workflows/examples/academic_research.yaml",
        "description": "Academic research with scholarly agent behaviors",
        "validation_rules": [
            "academic_terminology_preservation",
            "research_methodology_validation",
            "citation_requirement_checks"
        ]
    },
    {
        "workflow_file": "src/cognivault/workflows/examples/executive_briefing.yaml", 
        "description": "Executive briefing with business-focused agents",
        "validation_rules": [
            "business_context_validation",
            "strategic_framing_checks",
            "roi_analysis_requirements"
        ]
    },
    {
        "workflow_file": "src/cognivault/workflows/examples/legal_analysis.yaml",
        "description": "Legal analysis with compliance-focused validation",
        "validation_rules": [
            "legal_terminology_precision",
            "jurisdictional_scope_validation", 
            "regulatory_compliance_checks"
        ]
    }
]
```

### Problematic Workflows (Strict Mode)

```python
# These will fail in strict mode with enhanced validation
problematic_workflows = [
    {
        "agents": ["synthesis"],
        "pattern": "standard",
        "issue": "Synthesis without analysis input",
        "validation_error": "SYNTHESIS_WITHOUT_ANALYSIS"
    },
    {
        "agents": ["refiner", "unknown_agent"],
        "pattern": "standard",
        "issue": "Unknown agent not in registry",
        "validation_error": "UNKNOWN_AGENT"
    },
    {
        "agents": ["refiner", "critic"],
        "pattern": "conditional",
        "issue": "Enhanced routing requires performance tracking data",
        "validation_error": "INSUFFICIENT_ROUTING_DATA"
    },
    {
        "agents": ["historian"],
        "pattern": "parallel",
        "issue": "Single agent doesn't benefit from parallelization",
        "validation_error": "INEFFECTIVE_PARALLELIZATION"
    },
    {
        "workflow_config": {
            "agents": ["refiner", "synthesis"],
            "agent_configs": {
                "refiner": {"invalid_field": "value"}
            }
        },
        "issue": "Invalid agent configuration schema",
        "validation_error": "INVALID_AGENT_CONFIG"
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

## Production Best Practices

### Development Workflow
1. **Start Permissive**: Use non-strict mode initially to identify potential issues
2. **Review Warnings**: Address validation warnings to improve workflow quality  
3. **Strict for Production**: Use strict mode for production workflows
4. **Custom Validators**: Create domain-specific validators for specialized use cases
5. **Validation in Tests**: Include validation in automated tests to catch workflow issues early

### Enhanced Routing Integration
6. **Performance Baseline**: Establish performance baselines before enabling conditional patterns
7. **Resource Profiling**: Profile resource usage for optimal constraint configuration
8. **Routing Analytics**: Monitor routing decisions and effectiveness over time
9. **Fallback Testing**: Test fallback scenarios for enhanced routing failures

### Configurable Prompt Composition
10. **Schema Validation**: Always validate agent configurations against Pydantic schemas
11. **Template Testing**: Test custom prompt templates with representative data
12. **Behavioral Validation**: Verify custom constraints produce expected agent behaviors
13. **Configuration Versioning**: Version control agent configuration changes

### Production Monitoring
14. **Validation Metrics**: Track validation success rates and common failure patterns
15. **Performance Impact**: Monitor validation overhead in production environments
16. **Error Correlation**: Correlate validation failures with execution performance
17. **Continuous Improvement**: Use validation data to refine workflows and patterns

## Performance Considerations

### Production Performance Characteristics
- **Validation Overhead**: Minimal impact (~1-5ms per workflow for standard validation)
- **Enhanced Routing Validation**: Additional ~2-10ms for complex routing decisions
- **Configuration Validation**: Schema validation adds ~1-3ms per agent configuration
- **Cache Optimization**: Validation bypassed when using cached compiled graphs
- **Complex Validators**: Custom domain validators may add 5-20ms depending on complexity

### Optimization Strategies
- **Graph Factory Caching**: Compiled graph caching eliminates repeated validation
- **Validation Result Caching**: Cache validation results for repeated workflow patterns
- **Async Validation**: Non-blocking validation for performance-critical paths
- **Selective Validation**: Enable validation only for critical workflows in high-throughput scenarios
- **Batch Validation**: Validate multiple workflows in single operations for bulk processing

## Troubleshooting

### Common Production Issues

1. **"Validation enabled but no validator available"**: Set a default validator or disable validation
2. **"Synthesis without analysis"**: Add critic and/or historian agents before synthesis
3. **"Unknown agent"**: Ensure agent is registered in GraphFactory.node_functions
4. **"Enhanced routing validation failed"**: Verify ContextAnalyzer and ResourceOptimizer are available
5. **"Invalid agent configuration schema"**: Check agent configs against Pydantic model definitions
6. **"Routing decision confidence too low"**: Adjust confidence thresholds or improve context analysis
7. **"Resource constraint violation"**: Review and adjust execution time or agent count limits
8. **"Pattern-agent compatibility error"**: Verify agent types are compatible with selected pattern

### Debug Validation

```python
# Get detailed validation results with enhanced information
result = factory.validate_workflow(agents, pattern, strict_mode=False)

print(f"Validation Summary: {result.is_valid}")
print(f"Total Issues: {len(result.issues)}")

for issue in result.issues:
    print(f"{issue.severity.value}: {issue.message}")
    if issue.suggestion:
        print(f"  Suggestion: {issue.suggestion}")
    if issue.code:
        print(f"  Code: {issue.code}")
    if hasattr(issue, 'agent') and issue.agent:
        print(f"  Agent: {issue.agent}")

# Enhanced routing validation debug
if pattern == "conditional":
    routing_validation = factory.validate_enhanced_routing(agents)
    print(f"Routing Validation: {routing_validation}")

# Configuration schema validation debug
config_validation = factory.validate_agent_configurations(agent_configs)
print(f"Configuration Validation: {config_validation}")
```

### CLI-Based Validation

```python
# Use CogniVault CLI for validation debugging
import subprocess

# Validate workflow patterns
subprocess.run([
    "cognivault", "diagnostics", "patterns", "validate", "conditional",
    "--level", "comprehensive", "--format", "json"
])

# Validate specific workflow files
subprocess.run([
    "cognivault", "workflow", "validate", 
    "src/cognivault/workflows/examples/academic_research.yaml",
    "--validate-config", "--verbose"
])
```

## Architecture Integration

This semantic validation system integrates comprehensively with CogniVault's production architecture:

### Core Documentation References
- **[ADR-001: Graph Pattern Architecture](./architecture/ADR-001-Graph-Pattern-Architecture.md)** - Pattern validation foundations (HISTORICAL REFERENCE - Principles Integrated)
- **[ADR-002: Conditional Patterns and Enhanced Developer Tooling](./architecture/ADR-002-Conditional-Patterns-And-Enhanced-Developer-Tooling.md)** - Enhanced routing validation (HISTORICAL REFERENCE - Implementation Complete)
- **[PATTERN_REGISTRY.md](./PATTERN_REGISTRY.md)** - Pattern system integration and validation rules
- **[CLI_USAGE.md](./CLI_USAGE.md)** - CLI validation commands and workflow examples

### Integration Points
- **LangGraph Backend**: Full validation integration with `cognivault.langgraph_backend`
- **Enhanced Routing**: Validation of routing decisions via `cognivault.routing` components
- **Event System**: Validation event tracking via `cognivault.events` with correlation context
- **Configuration System**: Pydantic schema validation for agent configurations
- **Observability**: Validation metrics and performance tracking via diagnostic commands

### Production Features
- **Status**: Semantic validation operational with comprehensive pattern and configuration validation
- **Performance**: Optimized validation with caching and minimal overhead impact
- **Observability**: Validation results tracking and correlation with execution performance
- **Tooling**: CLI validation commands, automated testing, and debugging capabilities

### Strategic Value
- **Quality Assurance**: Ensures workflow semantic correctness before execution
- **Performance Optimization**: Validates resource constraints and routing decisions
- **Configuration Management**: Prevents invalid agent configurations from reaching production
- **Developer Experience**: Provides clear feedback and suggestions for workflow improvement

---

*This documentation reflects the current production state of CogniVault's semantic validation system. Last updated: Production V1 with enhanced routing and configuration validation operational.*