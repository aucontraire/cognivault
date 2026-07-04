# ADR-001: Graph Pattern Architecture Design

## Status
**HISTORICAL REFERENCE - Principles Integrated** - Core implementation finished, architectural analysis integrated into strategic roadmap ✅

### Implementation Status Update (July 2025)
- ✅ **Phase 1**: StandardPattern logic gap fixed and operational
- 🔄 **Phase 2**: Semantic validation achieved through different architecture - 6-axis classification system (see AAD-002)
- 🔄 **Phase 3**: Pattern flexibility achieved through configurable prompt composition rather than pattern refactoring
- 📋 **Alternative Path**: Multi-axis classification and configurable agent behaviors provided the extensibility originally planned through semantic validation layer

## Context

During Phase 2 implementation of the LangGraph migration, we discovered a critical architectural decision point regarding graph pattern flexibility versus semantic enforcement. This decision emerged while fixing a logic gap in the `StandardPattern` implementation where certain agent combinations (specifically `["refiner", "synthesis"]`) were not creating expected execution flows.

### Current Implementation

CogniVault's `langgraph_backend` module implements graph patterns through the `GraphPattern` abstract base class with three concrete implementations:

1. **StandardPattern**: Enforces the canonical CogniVault workflow: `refiner → [critic, historian] → synthesis`
2. **ParallelPattern**: Maximizes parallelization with minimal dependencies
3. **ConditionalPattern**: Supports dynamic routing (currently delegates to StandardPattern)

### The Problem

The current `StandardPattern` implementation contains semantic assumptions about agent relationships that create inflexibility:

```python
# Problematic logic in StandardPattern.get_edges()
if "refiner" in agents_lower:
    if "critic" in agents_lower:
        edges.append({"from": "refiner", "to": "critic"})
    if "historian" in agents_lower:
        edges.append({"from": "refiner", "to": "historian"})
    # Bug: Missing direct refiner→synthesis connection when no intermediates
```

This created scenarios where valid agent combinations like `["refiner", "synthesis"]` would fall through to unintended behavior.

## Decision Drivers

1. **Domain Logic vs. Flexibility**: Should patterns enforce domain-specific workflows or provide flexible graph generation?
2. **Extensibility**: How should the system accommodate future agents with different functions?
3. **User Experience**: Should invalid combinations fail fast or degrade gracefully?
4. **Maintenance**: How complex should pattern logic become to handle edge cases?

## Considered Options

### Option A: Strict Domain-Driven Patterns
- **Approach**: Patterns enforce semantic meaning of agents and their relationships
- **Pros**: 
  - Clear domain boundaries
  - Prevents invalid workflows
  - Self-documenting through constraints
- **Cons**: 
  - Inflexible for new use cases
  - Complex edge case handling
  - Tight coupling between patterns and agent semantics

### Option B: Flexible Graph Generation
- **Approach**: Patterns focus on graph topology, minimal semantic assumptions
- **Pros**: 
  - Extensible to new agents
  - Simpler pattern logic
  - Loose coupling
- **Cons**: 
  - Allows semantically invalid workflows
  - Less domain guidance
  - Requires external validation

### Option C: Hybrid Approach (RECOMMENDED)
- **Approach**: Base patterns provide flexibility with optional semantic validation layers
- **Pros**: 
  - Balances flexibility and domain guidance
  - Extensible architecture
  - Optional strict mode for validation
- **Cons**: 
  - More complex initial implementation
  - Requires careful abstraction design

## Decision

We recommend **Option C: Hybrid Approach** with the following architecture:

### Core Pattern Layer
```python
class GraphPattern(ABC):
    """Base class focused on graph topology generation"""
    
    @abstractmethod
    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        """Generate edges based on topology rules, minimal semantic assumptions"""
    
    def validate_agents(self, agents: List[str]) -> bool:
        """Basic validation - can be overridden for semantic checks"""
        return True
```

### Semantic Validation Layer
```python
class WorkflowSemanticValidator(ABC):
    """Optional layer for domain-specific validation"""
    
    @abstractmethod
    def validate_workflow(self, agents: List[str], pattern: str) -> ValidationResult:
        """Validate agent combination makes semantic sense"""

class CogniVaultValidator(WorkflowSemanticValidator):
    """Domain-specific validator for CogniVault workflows"""
    
    def validate_workflow(self, agents: List[str], pattern: str) -> ValidationResult:
        # Implement CogniVault-specific rules
        # e.g., "synthesis should come after other agents"
        # e.g., "refiner typically comes first"
```

### Pattern Implementations
```python
class StandardPattern(GraphPattern):
    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        """Generate standard topology with graceful degradation"""
        # Flexible edge generation based on available agents
        # Handle all combinations gracefully
        
class FlexibleStandardPattern(StandardPattern):
    def __init__(self, validator: Optional[WorkflowSemanticValidator] = None):
        self.validator = validator
    
    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        if self.validator:
            validation = self.validator.validate_workflow(agents, "standard")
            if not validation.is_valid:
                raise ValidationError(validation.message)
        return super().get_edges(agents)
```

## Implementation Strategy

### Phase 1: Fix Current Issues (COMPLETED)
- ✅ Fixed StandardPattern logic gap for `["refiner", "synthesis"]` case
- ✅ Updated tests to match corrected behavior
- ✅ Minimal disruption approach

### Phase 2: Introduce Semantic Validation (SUPERSEDED)
- ❌ `WorkflowSemanticValidator` interface not implemented as proposed
- ✅ **Alternative Implementation**: Domain validation achieved through 6-axis classification system (AAD-002)
- ✅ **Alternative Implementation**: Agent metadata classification provides semantic validation capabilities
- ✅ **Backward Compatibility**: Maintained through ConfigMapper and PromptComposer architecture

### Phase 3: Enhanced Pattern Flexibility (SUPERSEDED)
- ❌ Pattern refactoring approach not taken
- ✅ **Alternative Implementation**: Flexibility achieved through configurable prompt composition (ADR-006)
- ✅ **Alternative Implementation**: Custom agent behaviors supported through RefinerConfig, CriticConfig, etc.
- ✅ **Alternative Implementation**: Advanced node types provide workflow composition capabilities

## Consequences (Updated July 2025)

### Positive
- ✅ **Immediate**: Fixed critical bug with minimal code changes
- ✅ **Short-term**: Architecture evolved successfully through alternative path (6-axis classification)
- ✅ **Long-term**: Achieved extensibility through configurable prompt composition and advanced node types

### Negative (Resolved/Mitigated)
- ✅ **Technical Debt**: Resolved through multi-axis classification system rather than semantic validation layer
- ✅ **Complexity**: Managed through clear separation in configuration system (ConfigMapper, PromptComposer)
- ❌ **Migration**: Pattern API updates not needed - flexibility achieved through configuration

### Risks (Historical)
- ✅ **Scope Creep**: Avoided by taking configuration-based approach rather than pattern refactoring
- ✅ **Over-Engineering**: Mitigated by implementing through existing agent configuration architecture
- ✅ **Breaking Changes**: Avoided through backward-compatible configuration system

## Alternative Considerations

### Pattern Registry Evolution
The current `PatternRegistry` could be extended to support:
- Pattern versioning
- Dynamic pattern loading
- Pattern composition and inheritance

### Agent Metadata (IMPLEMENTED DIFFERENTLY)
**Proposed Architecture** (from this ADR):
```python
@dataclass
class AgentMetadata:
    name: str
    role: str  # "preprocessor", "analyzer", "synthesizer"
    dependencies: List[str]
    parallel_compatible: bool
```

**Actual Implementation** (see `src/cognivault/agents/metadata.py`):
```python
@dataclass
class AgentMetadata:
    # 6-axis classification system (AAD-002)
    cognitive_speed: Literal["fast", "slow", "adaptive"]
    cognitive_depth: Literal["shallow", "deep", "variable"]
    processing_pattern: Literal["atomic", "composite", "chain"]
    execution_pattern: Literal["processor", "decision", "aggregator", "validator", "terminator"]
    pipeline_role: Literal["entry", "intermediate", "terminal", "standalone"]
    bounded_context: str
```

**Evolution**: The actual implementation provides richer semantic information through multi-axis classification, enabling more sophisticated workflow composition than originally proposed.

## Related Decisions
- **Infrastructure**: This decision impacts the `GraphFactory` and caching architecture
- **Testing**: Pattern flexibility affects test strategy and coverage requirements
- **CLI**: User interface may need to expose pattern validation options
- **AAD-002**: Multi-Axis Classification system superseded semantic validation concepts from this ADR
- **ADR-006**: Configurable Prompt Composition Architecture provided the flexibility originally planned through pattern refactoring
- **Current Architecture**: See ARCHITECTURE.md for how graph patterns evolved within the broader DAG orchestration platform

## Notes (Historical Context and Evolution)
This ADR was created following discovery of the StandardPattern logic gap during Phase 2 implementation. The immediate fix (Option A variant) was implemented to resolve the bug, while this document outlined an architectural path forward.

**Architectural Evolution**: While the hybrid approach recommended in this ADR was sound, the actual implementation achieved the same goals through a different architectural path:
- **Semantic Validation**: Achieved through 6-axis classification system (AAD-002) rather than separate validator layer
- **Pattern Flexibility**: Achieved through configurable prompt composition (ADR-006) rather than pattern refactoring
- **Extensibility**: Achieved through advanced node types and agent configuration rather than pattern composition

**Lessons Learned**: This demonstrates how architectural analysis can correctly identify problems and principles while the specific implementation path may evolve based on broader system architecture decisions. The hybrid approach principles influenced the eventual multi-axis classification and configuration architecture.

**Current Status**: The graph pattern system remains stable and functional. Future enhancements focus on higher-level workflow composition through the configuration system rather than low-level pattern modifications.

## Integration into Strategic Roadmap

The architectural principles and insights from this ADR have been integrated into the strategic roadmap for future platform development:

### ROADMAP.md Integration Points

**Phase 3: Microservice Evolution Strategy** (lines 218-241):
- **Pattern Evolution Principles**: Hybrid architecture approach concepts applied to microservice design
- **Domain Validation Strategy for Plugin Architecture**: Semantic validation layer concepts adapted for community plugin validation
- **Flexible-First Design**: Services focus on capability delivery with minimal semantic assumptions
- **Validation Composition**: Multiple validation layers can be composed for complex domain requirements

### Key Principles Preserved for Future Development

1. **Hybrid Architecture Approach**: Base service flexibility with optional semantic validation layers
2. **Graceful Degradation**: Service architecture supports fallback patterns when validation layers fail
3. **Capability-Based Routing**: Plugin selection based on declared capabilities rather than rigid semantic hierarchies
4. **Semantic Validation Strategy**: Optional domain-specific validation while maintaining core extensibility

These principles will guide the implementation of the community plugin system and microservice architecture in Phase 3 of the strategic roadmap, ensuring the lessons learned from graph pattern architecture evolution inform future platform design decisions.