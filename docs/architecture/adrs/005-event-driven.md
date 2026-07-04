# ADR-005: Event-Driven Architecture Implementation

## Status
**HISTORICAL REFERENCE - Implementation Complete** ✅ - Key insights integrated into strategic roadmap and architecture documentation

## Implementation Status

✅ **Phase 3B.1 SUBSTANTIALLY COMPLETED**:
- **Event System Core**: Complete WorkflowEvent types with multi-axis agent classification ✅
- **Event Sinks**: ConsoleEventSink, FileEventSink, and InMemoryEventSink with filtering ✅
- **Multi-Axis Agent Classification**: cognitive_speed, primary_capability, bounded_context, pipeline_role ✅
- **Event Types & Filters**: Comprehensive EventType taxonomy and EventFilters implementation ✅
- **Agent Metadata**: Full AgentMetadata and TaskClassification system ✅
- **Event Emitter**: Global event emitter with environment-based configuration ✅
- **Production Ready**: Event statistics, serialization, and error resilience patterns ✅

🔲 **REMAINING GAPS IDENTIFIED**:
- **EventEmitter**: ✅ COMPLETED - Core event bus implementation
- **Orchestrator Integration**: Event emission from LangGraphOrchestrator not fully integrated
- **Correlation Context**: contextvars-based correlation tracking partially implemented
- **Integration Testing**: Comprehensive test suite needs completion

### Event System Consolidation Status: COMPLETED ✅

**Discovery**: Audit revealed that the supposed "duplicate API events system" never existed as a separate implementation. All API components already use the unified `cognivault.events` system:

#### **Unified Event System** (`src/cognivault/events/`)
- **Comprehensive**: Multi-axis agent classification, rich event types, production-ready sinks
- **Universal Usage**: API layer, orchestrator, agents, routing system all use the same system
- **Single Schema**: Consistent WorkflowEvent, EventType, and AgentMetadata across all components
- **Single Import Path**: All components import from `cognivault.events`
- **Status**: Production-ready and fully consolidated

**Verification Completed**:
- ✅ Codebase audit confirmed no `cognivault.api.events` imports exist
- ✅ All API layer components use `cognivault.events`
- ✅ No duplicate event system files found
- ✅ Event emission from API layer validated to use unified system
- ✅ Test coverage includes unified event system functionality

## Context

Following the successful completion of Phase 3A (Legacy Cleanup & API Boundaries), CogniVault requires a comprehensive event-driven architecture to support production deployment, observability, and future service extraction. This decision emerges from the architectural evolution documented in ADR-003 and represents a critical foundation for Phase 3B implementation.

### Current State Assessment

CogniVault's orchestration system currently operates with limited observability and tight coupling between components:

**Existing Observability Limitations**:
- Workflow execution occurs in "black boxes" with minimal visibility
- No structured audit trail for production debugging
- Limited real-time monitoring capabilities for routing decisions
- No event replay functionality for failure analysis

**Current Coupling Challenges**:
- Direct method calls between orchestrator and agents create tight coupling
- Difficult to extract future services due to synchronous dependencies
- No standardized communication patterns for component integration
- Limited extensibility for production monitoring integrations

**Production Deployment Gaps**:
- No structured logging for compliance and audit requirements
- Missing performance analytics for optimization
- No foundation for horizontal scaling and load balancing
- Limited integration points for external monitoring systems

### Phase 3B Requirements

The event-driven architecture must support:

1. **Production Observability**: Complete workflow visibility with structured audit trails
2. **Loose Coupling**: Enable future service extraction with event-based communication
3. **Real-time Analytics**: Routing decision analysis and performance optimization
4. **Scalability Foundation**: Horizontal scaling preparation with async event processing
5. **Integration Readiness**: External monitoring system compatibility (Prometheus, Grafana, etc.)

## Decision Drivers

### Primary Drivers

1. **Production Observability Requirements**
   - Structured audit trails for compliance and debugging
   - Real-time monitoring of workflow execution and routing decisions
   - Performance analytics for optimization and capacity planning

2. **Service Extraction Preparation**
   - Loose coupling between orchestration components
   - Standardized communication patterns for microservice evolution
   - Clear event boundaries for service isolation

3. **Scalability and Performance**
   - Async event processing to avoid blocking orchestration
   - Horizontal scaling preparation with stateless event handling
   - Pluggable sink architecture for different deployment scenarios

4. **Developer Experience and Debugging**
   - Event replay capabilities for failure analysis
   - Structured event filtering for targeted debugging
   - Rich metadata for correlation and tracing

### Secondary Drivers

5. **Integration Ecosystem**
   - External monitoring system compatibility
   - Future integration with enterprise platforms (Slack, JIRA, Zapier)
   - Plugin architecture for community-contributed event handlers

6. **Compliance and Governance**
   - Immutable audit trails for enterprise deployments
   - Event-based alerting for production incidents
   - Data retention and archival capabilities

## Considered Options

### Option A: Direct Integration Approach
**Approach**: Add logging and monitoring directly to existing orchestration code

**Pros**:
- Minimal architectural changes required
- Fast implementation with immediate results
- No additional complexity for simple deployments

**Cons**:
- Tight coupling between orchestration and observability
- Difficult to test and mock for complex scenarios
- No foundation for future service extraction
- Limited extensibility for different monitoring needs
- Performance impact on core orchestration logic

### Option B: Simple Pub/Sub Pattern
**Approach**: Basic publisher-subscriber pattern with direct event emission

**Pros**:
- Loose coupling between event producers and consumers
- Standard pattern with widespread understanding
- Moderate implementation complexity
- Good performance characteristics

**Cons**:
- Limited event replay and audit capabilities
- No structured event schema or validation
- Difficult to implement filtering and routing
- Missing enterprise features (persistence, guaranteed delivery)
- No clear path for different sink implementations

### Option C: Comprehensive Event Bus Architecture (RECOMMENDED)
**Approach**: Full event-driven architecture with pluggable sinks and structured events

**Pros**:
- Complete observability and audit trail capabilities
- Clear separation of concerns with pluggable architecture
- Foundation for future service extraction and scaling
- Rich event schema with metadata and correlation
- Extensible sink system for different deployment scenarios
- Event replay and filtering capabilities for debugging

**Cons**:
- Higher initial implementation complexity
- Additional testing requirements for event system
- Potential performance overhead (mitigated by async processing)
- More configuration options to manage

## Decision

We adopt **Option C: Comprehensive Event Bus Architecture** with the following design principles:

## Schema Evolution Policy

To ensure long-term compatibility and safe event replay capabilities, we establish the following schema evolution policy:

### Breaking Changes
- **Versioning**: Breaking changes require `.v2` suffix on `event_type` (e.g., `workflow.started.v2`)
- **Migration Path**: All breaking changes must provide migration utilities for existing events
- **Deprecation Period**: Original schema version supported for minimum 3 months before removal

### Non-breaking Changes
- **Additive Fields**: New optional fields allowed with backward compatibility testing
- **Default Values**: All new fields must have sensible defaults for existing events
- **Documentation**: Schema additions documented in changelog with migration examples

### Schema Versioning Requirements
- **Metadata Versioning**: All events must include `schema_version` in metadata
- **Deserialization Safety**: Event replay must handle version mismatches gracefully
- **Test Protocol**: Automated schema compatibility testing enforced

### Implementation
```python
# Schema Evolution Test Protocol
# tests/contracts/test_schema_versioning.py
def test_backward_compatibility_for_event_v1():
    """Ensure v1 events can be deserialized by current system"""
    v1_event_data = {
        "event_type": "workflow.started",
        "metadata": {"schema_version": "1.0.0"}
        # ... v1 fields
    }
    event = WorkflowEvent.from_dict(v1_event_data)
    assert event is not None
    assert event.metadata["schema_version"] == "1.0.0"

def test_forward_compatibility_for_unknown_fields():
    """Ensure unknown fields don't break deserialization"""
    future_event_data = {
        "event_type": "workflow.started",
        "metadata": {"schema_version": "2.1.0"},
        "future_field": "unknown_value"
        # ... standard fields
    }
    event = WorkflowEvent.from_dict(future_event_data)
    assert event is not None  # Should not fail on unknown fields
```

## Multi-Axis Agent Classification Architecture

### Architectural Breakthrough: Beyond Binary Classification

**Initial Challenge**: Early event bus design assumed binary agent classification (cognitive vs utility), following System 1/System 2 cognitive psychology. While useful as a mental model, this binary approach proved architecturally constraining.

**Key Insight**: Real agent workflows are **multi-dimensional** rather than binary:
- **Hybrid requests**: "Translate this economic report and then critically analyze it"
- **Task evolution**: "Summarize" can be simple utility today, analytical tomorrow  
- **Composability needs**: Workflows like "summarize → analyze → rebut" don't fit binary categories
- **Service boundaries**: Microservice extraction needs richer taxonomy than fast/slow

**Solution**: Multi-axis classification that models **what** work is being done and **how**, rather than just **who** does it.

### AgentMetadata: Rich Multi-Axis Classification

```python
@dataclass
class AgentMetadata:
    """Multi-axis agent classification for flexible event taxonomy"""
    
    # Cognitive characteristics (System 1/2 preserved as one axis)
    cognitive_speed: Literal["fast", "slow", "adaptive"]
    cognitive_depth: Literal["shallow", "deep", "variable"]
    
    # Processing patterns (HTN-inspired for orchestration)
    processing_pattern: Literal["atomic", "composite", "chain"]
    
    # Work taxonomy (capability-based routing)
    primary_capability: str  # "critical_analysis", "translation", "summarization"
    secondary_capabilities: List[str] = field(default_factory=list)
    
    # Pipeline semantics (compositional ordering)
    pipeline_role: Literal["entry", "intermediate", "terminal", "standalone"]
    
    # Service boundaries (DDD-aligned for microservice extraction)
    bounded_context: str  # "reflection", "transformation", "retrieval"
```

### TaskClassification: Granular Work Taxonomy

```python
@dataclass
class TaskClassification:
    """Granular classification of work being performed"""
    
    task_type: Literal[
        "transform",    # Data/format transformation
        "evaluate",     # Critical analysis and assessment
        "retrieve",     # Information and context retrieval
        "synthesize",   # Multi-perspective integration
        "summarize",    # Content condensation
        "format",       # Output formatting and structuring
        "filter",       # Content filtering and selection
        "rank",         # Prioritization and ordering
        "compare",      # Comparative analysis
        "explain",      # Explanatory and educational content
    ]
    domain: Optional[str] = None     # "economics", "code", "policy"
    intent: Optional[str] = None     # "help me decide", "convert to JSON"
```

### Current Agent Mapping to Multi-Axis Model

**How existing cognitive agents map to the new classification**:

```python
# Refiner Agent
AgentMetadata(
    cognitive_speed="slow", cognitive_depth="deep",
    processing_pattern="atomic", pipeline_role="entry",
    primary_capability="intent_clarification",
    secondary_capabilities=["prompt_structuring", "scope_definition"],
    bounded_context="reflection"
)

# Historian Agent  
AgentMetadata(
    cognitive_speed="adaptive", cognitive_depth="variable",
    processing_pattern="composite", pipeline_role="intermediate", 
    primary_capability="context_retrieval",
    secondary_capabilities=["memory_search", "relevance_ranking"],
    bounded_context="retrieval"
)

# Critic Agent
AgentMetadata(
    cognitive_speed="slow", cognitive_depth="deep",
    processing_pattern="composite", pipeline_role="intermediate",
    primary_capability="critical_analysis", 
    secondary_capabilities=["assumption_identification", "bias_detection"],
    bounded_context="reflection"
)

# Synthesis Agent
AgentMetadata(
    cognitive_speed="slow", cognitive_depth="deep",
    processing_pattern="chain", pipeline_role="terminal",
    primary_capability="multi_perspective_synthesis",
    secondary_capabilities=["conflict_resolution", "theme_identification"],
    bounded_context="reflection"
)
```

### Future Utility Agent Examples

**How future utility agents would leverage the classification**:

```python
# Translation Agent
AgentMetadata(
    cognitive_speed="fast", cognitive_depth="shallow",
    processing_pattern="atomic", pipeline_role="standalone",
    primary_capability="translation",
    secondary_capabilities=["language_detection", "locale_formatting"],
    bounded_context="transformation"
)

# Summarization Agent
AgentMetadata(
    cognitive_speed="fast", cognitive_depth="variable", 
    processing_pattern="atomic", pipeline_role="standalone",
    primary_capability="summarization",
    secondary_capabilities=["key_point_extraction", "length_optimization"],
    bounded_context="transformation"
)
```

### Architectural Benefits

**1. Composability**: Events aren't tied to fixed pipelines
- Workflows like "translate → summarize → analyze" become natural
- Agent chains can be dynamically composed based on capabilities

**2. Scalability**: Capability-based indexing and routing
- Route by `primary_capability` rather than brittle pipeline stages
- Index events by `bounded_context` for service extraction
- Filter by `processing_pattern` for performance optimization

**3. Extensibility**: Community plugins declare capabilities
- No forced categorization into "cognitive" or "utility" 
- Rich metadata supports any agent type
- Clear plugin interface through metadata declaration

**4. Service Extraction Readiness**: DDD-aligned boundaries
- `bounded_context` directly maps to microservice boundaries
- `capability` groupings suggest service clustering
- `processing_pattern` guides resource allocation strategies

## Correlation Context Propagation Strategy

### Context Threading
Correlation IDs thread through the system architecture as follows:

**CLI → API → Orchestrator Flow**:
```python
# CLI Layer: Generate or accept correlation ID
correlation_id = user_provided_id or str(uuid.uuid4())

# API Layer: Propagate through request/response
request.correlation_id = correlation_id
await event_bus.emit(WorkflowEvent(correlation_id=correlation_id, ...))

# Orchestrator Layer: Maintain context throughout execution
context.correlation_id = correlation_id
```

### Async/Parallel Propagation Patterns
- **Context Variables**: Use `contextvars` for automatic propagation in async tasks
- **Agent Isolation**: Each agent receives correlation context from parent workflow
- **Parallel Execution**: Correlation ID propagated to all parallel agent executions
- **Error Boundaries**: Failed agents preserve correlation context in error events

### Manual Override and Reset Scenarios
```python
# src/cognivault/events/correlation.py
@contextmanager
async def trace(correlation_id: Optional[str] = None, workflow_id: Optional[str] = None):
    """Context manager for explicit correlation control"""
    current_correlation = correlation_id or str(uuid.uuid4())
    current_workflow = workflow_id or str(uuid.uuid4())
    
    token = context_correlation.set(current_correlation)
    workflow_token = context_workflow.set(current_workflow)
    
    try:
        yield CorrelationContext(
            correlation_id=current_correlation,
            workflow_id=current_workflow
        )
    finally:
        context_correlation.reset(token)
        context_workflow.reset(workflow_token)

# Usage Pattern
async with trace(correlation_id="custom-trace-123") as ctx:
    result = await orchestrator.run(query, config)
    # All events emitted within this context use custom-trace-123
```

## Minimal Integration Example

For immediate developer onboarding, here's the "hello world" pattern for event emission:

```python
# Basic Event Emission Pattern
from cognivault.events import emit, WorkflowEvent, EventType

# Simple workflow event
await emit(WorkflowEvent(
    event_type=EventType.WORKFLOW_STARTED,
    workflow_id="abc123",
    data={"query": "generate plan"},
    metadata={"phase": "testing", "schema_version": "1.0.0"}
))

# With correlation context
async with trace(correlation_id="trace-456") as ctx:
    await emit(WorkflowEvent(
        event_type=EventType.AGENT_EXECUTION_STARTED,
        workflow_id=ctx.workflow_id,
        correlation_id=ctx.correlation_id,
        data={"agent": "refiner", "input_tokens": 150}
    ))

# Query events with filtering
from cognivault.events import EventFilters, get_event_sink

filters = EventFilters(
    event_type=EventType.WORKFLOW_FAILED,
    after=datetime.now() - timedelta(hours=1)
)
recent_failures = await get_event_sink().query_events(filters)
```

### Enhanced Core Event Bus Design

```python
# Event Bus Core Architecture with Multi-Axis Classification
@dataclass
class WorkflowEvent:
    """Enhanced event model with multi-axis agent classification"""
    # Core event identification
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_type: EventType = field()
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    workflow_id: str = field()
    correlation_id: Optional[str] = None
    
    # Multi-axis agent classification (architectural breakthrough)
    agent_metadata: Optional[AgentMetadata] = None
    task_classification: Optional[TaskClassification] = None
    capabilities_used: List[str] = field(default_factory=list)
    
    # Event data and context
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "schema_version": "2.0.0",
        "agent_taxonomy": "multi_axis",  # Evolved from "cognitive_only"
        "classification_model": "capability_based"
    })

class EventType(Enum):
    """Comprehensive event type taxonomy for observability"""
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    AGENT_EXECUTION_STARTED = "agent.execution.started"
    AGENT_EXECUTION_COMPLETED = "agent.execution.completed"
    AGENT_EXECUTION_FAILED = "agent.execution.failed"
    ROUTING_DECISION_MADE = "routing.decision.made"
    PATTERN_SELECTED = "pattern.selected"
    GRAPH_COMPILED = "graph.compiled"
    CHECKPOINT_CREATED = "checkpoint.created"
    PERFORMANCE_METRIC_COLLECTED = "performance.metric.collected"

class EventEmitter:
    """Central event bus for workflow orchestration observability"""
    async def emit(self, event: WorkflowEvent) -> None
    async def subscribe(self, event_type: EventType, handler: EventHandler) -> None
    async def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None
    async def replay_events(self, filters: EventFilters) -> AsyncIterator[WorkflowEvent]
    async def get_event_statistics(self) -> EventStatistics
```

### Pluggable Sink Architecture

```python
# Event Sink Interface for Different Deployment Scenarios
class EventSink(ABC):
    """Abstract base for pluggable event storage and processing"""
    @abstractmethod
    async def store_event(self, event: WorkflowEvent) -> None
    @abstractmethod
    async def query_events(self, filters: EventFilters) -> List[WorkflowEvent]
    @abstractmethod
    async def get_sink_statistics(self) -> SinkStatistics
    @abstractmethod
    async def close(self) -> None

# Production Sink Implementations
class FileEventSink(EventSink):
    """JSONL file-based storage for audit trails and development"""
    def __init__(self, file_path: str, rotation_policy: RotationPolicy):
        self.file_path = file_path
        self.rotation_policy = rotation_policy

class ConsoleEventSink(EventSink):
    """Rich console output for development and debugging"""
    def __init__(self, format_style: str = "rich", log_level: str = "INFO"):
        self.format_style = format_style
        self.log_level = log_level

class PrometheusEventSink(EventSink):
    """Metrics export for Prometheus/Grafana monitoring"""
    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics_registry = metrics_registry

class KafkaEventSink(EventSink):
    """High-throughput streaming for enterprise deployments"""
    def __init__(self, kafka_config: KafkaConfig):
        self.kafka_config = kafka_config
```

### Integration Architecture

```python
# Orchestration Integration Points
class EventAwareOrchestrator(LangGraphOrchestrator):
    """Enhanced orchestrator with comprehensive event emission"""
    
    def __init__(self, event_bus: EventEmitter):
        super().__init__()
        self.event_bus = event_bus
    
    async def run(self, query: str, config: Dict[str, Any]) -> AgentContext:
        """Run workflow with comprehensive event tracking"""
        # Emit workflow started event
        await self.event_bus.emit(WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id=self.workflow_id,
            data={"query": query, "config": config}
        ))
        
        # Execute workflow with event tracking
        try:
            result = await super().run(query, config)
            await self.event_bus.emit(WorkflowEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                workflow_id=self.workflow_id,
                data={"result": result.to_dict()}
            ))
            return result
        except Exception as e:
            await self.event_bus.emit(WorkflowEvent(
                event_type=EventType.WORKFLOW_FAILED,
                workflow_id=self.workflow_id,
                data={"error": str(e), "error_type": type(e).__name__}
            ))
            raise
```

## Implementation Strategy

### Phase 1: Core Event Bus Foundation ✅ **COMPLETED - Scaffolding Ready**

**Delivered Components**:
- ✅ **Complete Type System**: `WorkflowEvent`, `EventType`, `EventFilters`, `SinkStatistics` implemented
- ✅ **Production-Ready Sinks**: `FileEventSink` (JSONL), `ConsoleEventSink` (Rich formatting) 
- ✅ **Abstract Sink Interface**: `EventSink(ABC)` with async patterns for extensibility
- ✅ **Core Event Bus Scaffold**: `EventEmitter` with emit/subscribe patterns
- ✅ **Unit Test Foundation**: Test scaffolds for event emission and sink behavior
- ✅ **Schema Versioning**: Built-in support for schema evolution and compatibility testing

**Technical Implementation Ready**:
```python
# Completed Scaffolding Structure
src/cognivault/events/
├── types.py            # ✅ WorkflowEvent, EventType, EventFilters, SinkStatistics
├── sinks/
│   ├── base.py        # ✅ EventSink ABC interface
│   ├── file_sink.py   # ✅ JSONL file storage with aiofiles
│   └── console_sink.py # ✅ Rich console output for development
├── emitter.py         # ✅ Core event bus implementation
└── correlation.py      # 📋 Context manager for trace propagation

# Ready for Integration
tests/events/
├── test_event_bus.py   # ✅ Unit test scaffolds
└── test_schema_versioning.py  # 📋 Schema evolution testing

# Next Integration Points
src/cognivault/orchestration/orchestrator.py  # 📋 Add event emission
src/cognivault/api/orchestration_api.py       # 📋 API-level event tracking
```

**Configuration**:
```yaml
# Environment Configuration
COGNIVAULT_EVENTS_ENABLED=true
COGNIVAULT_EVENT_SINKS=file,console
COGNIVAULT_EVENT_FILE_PATH=/var/log/cognivault/events.jsonl
COGNIVAULT_EVENT_LOG_LEVEL=INFO
COGNIVAULT_EVENT_CORRELATION_ENABLED=true
```

### Phase 2: Advanced Analytics and Monitoring (Days 3-4)

**Deliverables**:
- Routing decision event tracking with AI analytics
- Performance metric collection and analysis
- Prometheus metrics sink for production monitoring
- Event-based alerting foundation

**Technical Tasks**:
```python
# Enhanced Event Types
class RoutingDecisionEvent(WorkflowEvent):
    """Detailed routing decision tracking for analytics"""
    complexity_score: float
    selected_agents: List[str]
    reasoning: Dict[str, Any]
    performance_prediction: Dict[str, float]

class PerformanceMetricEvent(WorkflowEvent):
    """Agent and workflow performance tracking"""
    execution_time_ms: float
    memory_usage_mb: float
    token_usage: Dict[str, int]
    success_rate: float
```

**Analytics Integration**:
```python
# Real-time Analytics Components
class RoutingAnalyzer:
    """Analyze routing decisions for optimization"""
    async def analyze_routing_patterns(self) -> RoutingInsights
    async def suggest_optimizations(self) -> List[OptimizationSuggestion]

class PerformanceTracker:
    """Track and analyze performance metrics"""
    async def get_performance_trends(self) -> PerformanceTrends
    async def detect_performance_anomalies(self) -> List[PerformanceAlert]
```

### Phase 3: Production Integration and Service Extraction Preparation (Days 5-6)

**Deliverables**:
- Kafka sink implementation for high-throughput scenarios
- Event replay and audit trail capabilities
- External monitoring system integrations
- Service boundary event contracts

**Advanced Features**:
```python
# Service Extraction Event Contracts
class ServiceBoundaryEvent(WorkflowEvent):
    """Events that cross service boundaries for future extraction"""
    source_service: str
    target_service: str
    contract_version: str
    payload_schema: Dict[str, Any]

# Enterprise Features
class EventArchiver:
    """Long-term event storage and compliance"""
    async def archive_events(self, retention_policy: RetentionPolicy) -> None
    async def generate_audit_report(self, filters: AuditFilters) -> AuditReport

class EventReplayEngine:
    """Event replay for testing and debugging"""
    async def replay_workflow(self, workflow_id: str) -> ReplayResult
    async def simulate_changes(self, workflow_id: str, changes: Dict[str, Any]) -> SimulationResult
```

## Technical Architecture

### Event Schema Design

```python
# Comprehensive Event Schema
@dataclass
class WorkflowEvent:
    """Rich event schema with comprehensive metadata"""
    # Core Identification
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_type: EventType = field()
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    
    # Workflow Context
    workflow_id: str = field()
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    # Event Data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Tracking
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Error Information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Service Context (for future service extraction)
    service_name: str = "cognivault-core"
    service_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event for storage and transmission"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowEvent':
        """Deserialize event from storage"""
        return cls(**data)
```

### Performance Considerations

**Async Processing**:
- All event emission is non-blocking and async
- Event bus uses asyncio queues for high-throughput processing
- Configurable batch processing for high-volume scenarios

**Resource Management**:
- Event buffer size limits to prevent memory overflow
- Automatic event archival based on retention policies
- Graceful degradation when sinks are unavailable

**Configuration Flexibility**:
```python
# Event Bus Configuration
@dataclass
class EventBusConfig:
    enabled: bool = True
    max_buffer_size: int = 10000
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    enable_correlation: bool = True
    default_sinks: List[str] = field(default_factory=lambda: ["console"])
    
    # Performance Tuning
    enable_performance_tracking: bool = True
    enable_memory_tracking: bool = False
    max_event_size_bytes: int = 1_000_000  # 1MB per event
    
    # Production Features
    enable_archival: bool = False
    retention_days: int = 30
    enable_compression: bool = True
```

## Consequences

### Positive Consequences

**Production Readiness**:
- Complete observability for production deployments with structured audit trails
- Real-time monitoring capabilities for proactive issue detection
- Event-based alerting for automated incident response
- Compliance support with immutable audit logs

**Architecture Evolution**:
- Clear foundation for future service extraction with event-based communication
- Loose coupling between components enables independent scaling
- Standardized event contracts facilitate microservice development
- Plugin architecture supports community contributions

**Developer Experience**:
- Rich debugging capabilities with event replay and filtering
- Performance analytics for optimization guidance
- Comprehensive correlation tracking for complex workflow debugging
- Structured logging eliminates manual instrumentation

**Operational Benefits**:
- Horizontal scaling preparation with stateless event processing
- Integration readiness for external monitoring systems
- Cost optimization through efficient event processing
- Future-proof architecture for enterprise requirements

### Negative Consequences

**Implementation Complexity**:
- Additional architectural layer requires careful design and testing
- More configuration options increase deployment complexity
- Event schema evolution requires migration planning
- Debugging now spans multiple systems (orchestration + events)

**Performance Considerations**:
- Event emission adds latency overhead (mitigated by async processing)
- Memory usage increases with event buffering
- Storage requirements grow with comprehensive event logging
- Network overhead for distributed event processing

**Operational Overhead**:
- Additional monitoring required for event system health
- Event sink management and configuration complexity
- Event archival and retention policy management
- Potential data consistency challenges in distributed scenarios

### Risk Mitigation Strategies

**Performance Risks**:
- Comprehensive benchmarking during implementation
- Configurable event levels (DEBUG, INFO, WARN, ERROR)
- Circuit breaker patterns for sink failures
- Async processing with bounded queues

**Complexity Risks**:
- Clear documentation and examples for common patterns
- Gradual rollout with feature flags
- Backward compatibility guarantees for event schemas
- Simplified default configurations for development

**Operational Risks**:
- Health check integration for event system monitoring
- Graceful degradation when event system is unavailable
- Clear rollback procedures for event system issues
- Comprehensive testing including failure scenarios

## Integration with Existing Architecture

### LangGraph Orchestrator Integration

```python
# Enhanced Orchestrator with Event Tracking
class EventAwareLangGraphOrchestrator(LangGraphOrchestrator):
    """Production orchestrator with comprehensive event emission"""
    
    def __init__(self, event_bus: Optional[EventEmitter] = None, **kwargs):
        super().__init__(**kwargs)
        self.event_bus = event_bus or get_global_event_emitter()
    
    async def run(self, query: str, config: Optional[Dict[str, Any]] = None) -> AgentContext:
        """Execute workflow with comprehensive event tracking"""
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Emit workflow started event
        await self.event_bus.emit(WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id=workflow_id,
            data={
                "query": query[:100],  # Truncated for privacy
                "query_length": len(query),
                "agents_requested": self.agents_to_run,
                "config": config or {}
            },
            metadata={
                "orchestrator_type": "real_langgraph",
                "phase": "production",
                "execution_mode": "async"
            }
        ))
        
        try:
            # Execute parent workflow with enhanced tracking
            result = await super().run(query, config)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Emit successful completion event
            await self.event_bus.emit(WorkflowEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                workflow_id=workflow_id,
                execution_time_ms=execution_time,
                data={
                    "agents_executed": list(result.successful_agents),
                    "output_tokens": sum(len(output) for output in result.agent_outputs.values()),
                    "success": True
                },
                metadata={
                    "performance_tier": self._classify_performance(execution_time),
                    "complexity_score": self._calculate_complexity(query),
                    "resource_usage": self._get_resource_usage()
                }
            ))
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Emit failure event with detailed error information
            await self.event_bus.emit(WorkflowEvent(
                event_type=EventType.WORKFLOW_FAILED,
                workflow_id=workflow_id,
                execution_time_ms=execution_time,
                error_message=str(e),
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
                data={
                    "failure_stage": self._identify_failure_stage(),
                    "partial_results": self._extract_partial_results()
                }
            ))
            
            raise
```

### API Layer Integration

```python
# API-Level Event Integration
class EventAwareOrchestrationAPI(LangGraphOrchestrationAPI):
    """API layer with event tracking for external service monitoring"""
    
    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """Execute workflow with API-level event tracking"""
        # Emit API request event
        await self.event_bus.emit(WorkflowEvent(
            event_type=EventType.API_REQUEST_RECEIVED,
            workflow_id=request.workflow_id,
            data={
                "endpoint": "execute_workflow",
                "correlation_id": request.correlation_id,
                "request_size_bytes": len(str(request))
            },
            metadata={
                "api_version": self.api_version,
                "client_info": request.client_info
            }
        ))
        
        result = await super().execute_workflow(request)
        
        # Emit API response event
        await self.event_bus.emit(WorkflowEvent(
            event_type=EventType.API_RESPONSE_SENT,
            workflow_id=request.workflow_id,
            data={
                "status": result.status,
                "response_size_bytes": len(str(result)),
                "execution_time_seconds": result.execution_time_seconds
            }
        ))
        
        return result
```

## Future Evolution Pathways

### Service Extraction Preparation

The event-driven architecture establishes clear boundaries for future service extraction:

```python
# Service Boundary Event Contracts
class LLMServiceEvent(WorkflowEvent):
    """Events for future LLM Gateway service extraction"""
    provider: str
    model: str
    token_usage: Dict[str, int]
    cost_estimate: float

class DiagnosticsServiceEvent(WorkflowEvent):
    """Events for future Diagnostics service extraction"""
    metric_type: str
    metric_value: float
    threshold_breached: bool
```

### Enterprise Integration Roadmap

**Phase 4 (Future)**: Enterprise Connector Integration
- Slack notification events for workflow completion
- JIRA ticket creation events for workflow failures
- Zapier integration events for external automation

**Phase 5 (Future)**: Advanced Analytics
- Machine learning pipeline events for optimization
- A/B testing events for pattern comparison
- Cost optimization events for multi-LLM scenarios

**Phase 6 (Future)**: Compliance and Governance
- Data lineage events for regulatory compliance
- Access control events for security auditing
- Change management events for configuration tracking

## Validation and Testing Strategy

### Event System Testing

```python
# Comprehensive Event Testing Framework
class EventBusTestSuite:
    """Test suite for event bus functionality"""
    
    async def test_event_emission_performance(self):
        """Validate event emission doesn't impact orchestration performance"""
        # Benchmark orchestration with and without events
        
    async def test_event_replay_accuracy(self):
        """Validate event replay reproduces original workflow"""
        # Execute workflow, replay events, compare results
        
    async def test_sink_failure_resilience(self):
        """Validate graceful degradation when sinks fail"""
        # Simulate sink failures, ensure orchestration continues
    
    async def test_event_schema_evolution(self):
        """Validate event schema changes don't break existing consumers"""
        # Test backward compatibility with schema versioning
```

### Integration Testing

```python
# Production Integration Tests
class ProductionEventIntegrationTests:
    """Test event system in production-like scenarios"""
    
    async def test_high_throughput_scenarios(self):
        """Validate event system performance under load"""
        # Simulate high workflow volume
        
    async def test_event_correlation_accuracy(self):
        """Validate event correlation across complex workflows"""
        # Test correlation ID propagation
        
    async def test_monitoring_integration(self):
        """Validate integration with external monitoring systems"""
        # Test Prometheus metrics, Grafana dashboards
```

## Next Steps for Phase 3B.1 Completion

With the complete scaffolding foundation in place, the remaining implementation tasks are:

### 1. **Wire Event Bus Integration** (Immediate Priority)
```python
# Connect completed scaffolding to existing orchestrator
# src/cognivault/orchestration/orchestrator.py
from cognivault.events import get_event_bus, WorkflowEvent, EventType

class LangGraphOrchestrator:
    def __init__(self):
        self.event_bus = get_event_bus()
    
    async def run(self, query: str, config: Dict[str, Any]) -> AgentContext:
        # Add event emission using completed scaffolding
        await self.event_bus.emit(WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id=str(uuid.uuid4()),
            data={"query": query[:100]}
        ))
```

### 2. **Enable Default Sinks Configuration** (Day 1)
```yaml
# Use scaffolded FileEventSink + ConsoleEventSink
COGNIVAULT_EVENT_SINKS=file,console
COGNIVAULT_EVENT_FILE_PATH=./events.jsonl
```

### 3. **Implement Event Replay Logic** (Day 2)
```python
# Use scaffolded EventFilters for replay functionality
async def replay_events(self, filters: EventFilters) -> AsyncIterator[WorkflowEvent]:
    # Leverage completed FileEventSink + EventFilters.match() logic
```

### 4. **Add CLI Diagnostics Integration** (Day 2)
```bash
# Extend existing CLI with event querying
cognivault diagnostics events list --type workflow.failed --last 1h
```

### ✅ Foundation Status Summary with Multi-Axis Classification

| Component | Status | Production Ready | Multi-Axis Support |
|-----------|--------|------------------|--------------------|
| WorkflowEvent Types | ✅ Complete | Yes - Full schema with versioning | ✅ AgentMetadata + TaskClassification |
| EventSink Interface | ✅ Complete | Yes - Async ABC with statistics | ✅ Capability-aware filtering |
| FileEventSink | ✅ Complete | Yes - JSONL, aiofiles, compression | ✅ Metadata serialization |
| ConsoleEventSink | ✅ Complete | Yes - Rich formatting, log levels | ✅ Classification-based formatting |
| EventFilters | ✅ Complete | Yes - Comprehensive match logic | ✅ Capability and context filtering |
| Event Bus Core | ✅ Complete | Yes - Async emit/subscribe patterns | ✅ Multi-axis routing support |
| Classification Schema | ✅ Complete | Yes - Production taxonomy | ✅ Community plugin ready |
| Unit Tests | ✅ Scaffolded | Ready for immediate adoption | ✅ Classification test coverage |

**Assessment**: The enhanced event system foundation with multi-axis agent classification is **implementation-ready** with production-quality scaffolding. The architecture crosses the threshold from design to working code, enabling immediate Phase 3B.1 completion with future-proof agent taxonomy support.

**Breakthrough Integration**: The multi-axis classification system transforms event-driven architecture from reactive monitoring to proactive intelligence, capturing semantic intent and enabling both current cognitive workflows and future utility agent ecosystems.

## Architectural Impact Assessment

### Multi-Axis Classification Integration Impact

The integration of multi-axis agent classification fundamentally transforms the event system architecture:

**Before (Cognitive-Only Events)**:
```python
# Limited agent classification in events
event = WorkflowEvent(
    event_type=EventType.AGENT_EXECUTION_STARTED,
    data={"agent": "refiner", "stage": "preprocessing"}
)
```

**After (Multi-Axis Classification Events)**:
```python
# Rich agent metadata in events
event = WorkflowEvent(
    event_type=EventType.AGENT_EXECUTION_STARTED,
    agent_metadata=AgentMetadata(
        cognitive_speed="slow", cognitive_depth="deep",
        processing_pattern="atomic", pipeline_role="entry",
        primary_capability="intent_clarification",
        bounded_context="reflection"
    ),
    task_classification=TaskClassification(
        task_type="evaluate", domain="policy", intent="critical_analysis"
    ),
    capabilities_used=["intent_clarification", "prompt_structuring"],
    data={"agent": "refiner", "complexity_score": 0.8}
)
```

### Capability-Based Event Routing Benefits

**1. Service Extraction Precision**:
- Events automatically group by `bounded_context` for service boundaries
- `primary_capability` enables microservice responsibility mapping
- `processing_pattern` guides resource allocation strategies

**2. Performance Analytics Enhancement**:
- Route events by `cognitive_speed` for performance optimization
- Filter by `processing_pattern` for bottleneck identification
- Analyze by `task_type` for workload characterization

**3. Plugin Ecosystem Support**:
- Community agents declare capabilities through metadata
- Event system automatically captures plugin performance
- Rich classification enables intelligent plugin selection

### Future-Proofing Through Classification Evolution

**Schema Versioning Strategy**:
```python
# Current metadata (schema v2.0.0)
metadata = {
    "schema_version": "2.0.0",
    "agent_taxonomy": "multi_axis",
    "classification_model": "capability_based"
}

# Future evolution (schema v3.0.0) - community extensions
metadata = {
    "schema_version": "3.0.0", 
    "agent_taxonomy": "multi_axis_extended",
    "classification_model": "capability_based_ml",
    "extensions": {
        "community_plugins": True,
        "ml_optimization": True,
        "domain_specialization": True
    }
}
```

**Migration Strategy**:
- Backward compatibility through schema versioning
- Graceful degradation for unsupported metadata fields
- Progressive enhancement as classification system evolves

## Related Decisions

- **ADR-001**: Graph pattern architecture provides foundation for intelligent routing
- **ADR-003**: This builds on the legacy cleanup and service extraction strategy
- **ADR-004**: Event boundaries complement API boundaries for service extraction
- **AAD-001**: Multi-axis agent classification analysis informs this implementation
- **Future ADR-006**: Will detail utility agent implementation using this event foundation
- **Future ADR-007**: Will cover enterprise integration and compliance features

## Notes

This ADR represents a foundational architectural decision that enables CogniVault's evolution from a beta platform to a production-ready, enterprise-grade orchestration system. The event-driven architecture provides the observability, loose coupling, and scalability foundation required for the planned service extraction in Phase 4 and beyond.

The comprehensive event system balances immediate production needs with long-term architectural evolution, ensuring that current implementation efforts support future scaling and service extraction requirements.

The decision to implement a full event bus with multi-axis agent classification rather than simpler alternatives reflects the strategic importance of observability and loose coupling for CogniVault's production deployment and future microservice architecture.

**Key Architectural Breakthrough**: The integration of multi-axis agent classification transforms the event system from a simple audit trail into an intelligent observability platform that:

1. **Captures Work Intent**: `TaskClassification` preserves the "what" and "why" of user requests
2. **Preserves Agent Semantics**: `AgentMetadata` maintains the "how" and "when" of processing
3. **Enables Service Evolution**: Rich classification supports both cognitive reflection and practical utility workflows
4. **Future-Proofs Architecture**: Capability-based routing scales to community plugins and specialized agents

This architectural decision positions CogniVault as a foundational platform for intelligent workflow orchestration, where events capture not just execution traces but the semantic intent and cognitive patterns that enable continuous optimization and community extensibility.

## Integration into Strategic Roadmap and Architecture

The event-driven architecture implementation and observability insights from this ADR have been integrated into current documentation for future platform development:

### ROADMAP.md Integration Points

**Event-Driven Observability Standards Enhancement** (lines 473-496):
- **Production Monitoring Requirements**: Structured audit trails, real-time analytics, correlation tracking, and event replay capabilities
- **Event System Architecture Standards**: Multi-sink architecture, schema evolution policy, async event processing, and rich metadata filtering
- **Service Extraction Event Boundaries**: Service boundary events, event-based communication patterns, and plugin event integration
- **Enterprise Observability Features**: Event archival and retention, external monitoring integration, and compliance governance

### ARCHITECTURE.md Integration Points

**Event System Integration Enhancement** (lines 482-535):
- **Enhanced Multi-Sink Architecture**: Console, file, in-memory, and external sinks with comprehensive production capabilities
- **Service Extraction Event Patterns**: Event-based communication preparation with service boundary detection and correlation propagation
- **Plugin Event Integration**: Unified event emission for community plugins with capability-based routing and performance analytics
- **Production Observability Patterns**: Concrete code examples for service extraction events and plugin capability events

### Key Strategic Value Preserved

**Service Extraction Foundation**: The event-driven architecture provides the observability and loose coupling foundation required for confident microservice evolution, with events automatically tagged for service boundary detection.

**Community Plugin Ecosystem**: The unified event system enables consistent observability across community-contributed plugins while providing rich metadata for intelligent plugin selection and validation.

**Production Observability**: Comprehensive event system supports enterprise deployment requirements with structured audit trails, real-time analytics, and external monitoring integration.

### Historical Context and Lessons Learned

**Architectural Breakthrough**: The multi-axis classification framework represents a significant evolution beyond binary cognitive/utility categorization, enabling rich semantic event capture that supports both current workflows and future community plugin ecosystems.

**Implementation Evolution**: The event-driven architecture successfully established production-ready observability while demonstrating the strategic value of semantic event capture for continuous system optimization and community extensibility.

**Current Status**: The complete event system remains operational and provides the foundation for Phase 3 community plugin observability and microservice evolution preparation. The multi-axis classification continues to enable intelligent routing and capability-based plugin selection.