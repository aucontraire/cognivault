# AAD-001: Cognitive vs Utility Agent Taxonomy

## Document Type
**Architectural Analysis Document (AAD)**

## Status
**PARTIALLY IMPLEMENTED** - Core concepts integrated, specific implementations evolved differently

### Implementation Status Update (July 2025)
- ✅ **Dual-Process Foundation**: Implemented via 6-axis classification system with cognitive_speed/cognitive_depth axes
- ✅ **Configurable Agent Behaviors**: 662-line PromptComposer enables System 1/System 2 processing adaptation
- ✅ **Event-Driven Architecture**: Advanced event system operational (ADR-005 complete)
- ✅ **Service Boundaries**: API boundary separation implemented (ADR-004 complete)
- 🔄 **Plugin Architecture**: Foundation exists, community plugin system in roadmap (Phase 3)
- ❌ **Utility Agent Separation**: Not implemented as separate agent types - achieved through configuration instead

## Abstract

This document analyzes a fundamental architectural challenge in CogniVault: the tension between the current cognitive reflection agent design (Refiner, Historian, Critic, Synthesis) and the need to support practical utility requests that don't require deep analytical processing. This analysis explores the theoretical frameworks, design patterns, and architectural implications of extending the agent taxonomy to support both cognitive and utility operations.

## Problem Statement

### Current Agent Architecture (Updated July 2025)

**EVOLVED IMPLEMENTATION**: CogniVault now supports both cognitive reflection and utility processing through **configurable agent behaviors** rather than separate agent types.

```
🔍 Refiner → 🧾 Historian → 🧠 Critic → 🧵 Synthesis
    ↓            ↓            ↓            ↓
  Clarify    Contextualize  Analyze    Synthesize
    ↕            ↕            ↕            ↕
[Configurable Cognitive Depth via PromptComposer]
   System 1: Fast/Utility    System 2: Deep/Analytical
```

**Agent Characteristics (Now Configurable)**:
- **Refiner**: Configurable from simple clarification to deep intent analysis (RefinerConfig)
- **Historian**: Configurable search depth and contextual integration (HistorianConfig)  
- **Critic**: Configurable analysis depth and critique approaches (CriticConfig)
- **Synthesis**: Configurable synthesis strategies and integration modes (SynthesisConfig)

**Design Evolution**:
- ✅ **Achieved**: Agents can operate at different cognitive depths through configuration
- ✅ **Achieved**: Multi-axis classification enables intelligent routing (6 axes including cognitive_speed/cognitive_depth)
- ✅ **Achieved**: PromptComposer (662 lines) enables runtime behavior adaptation
- 🔄 **Evolved**: Instead of separate utility agents, achieved through configurable prompt composition

### The Utility Request Challenge

**Real-world user interactions include requests that don't align with cognitive reflection**:

#### Direct Utility Examples
```
"Can you list these things?"
"Translate this to French"  
"Summarize this article"
"Give me the key terms"
"Sort this by category"
"Check this for grammar"
"Write this as a tweet"
"Generate a title"
"Convert this to JSON"
"Count the words"
```

#### Mixed Request Examples
```
"Translate this economic report and then critically analyze it"
"Summarize these three articles and identify the contradictions"
"List the main points and tell me which ones are questionable"
```

### Architectural Mismatch

**The Problem**: Current agents are optimized for **System 2 thinking** (deliberate, analytical) but users need **System 1 operations** (fast, functional, direct).

**Symptoms**:
- Over-processing simple requests through unnecessary cognitive pipeline
- Poor user experience for straightforward utility tasks
- Architectural rigidity preventing efficient utility operations
- Resource waste on requests that don't benefit from reflection

**Core Question**: How do we design an agent taxonomy that elegantly supports both cognitive reflection and practical utility while maintaining architectural coherence?

## Theoretical Frameworks

### 1. Dual-Process Theory (Cognitive Science)

**Framework**: Human cognition operates through two distinct systems:

- **System 1**: Fast, automatic, intuitive, low-effort
  - *Examples*: Recognition, simple arithmetic, reading familiar words
  - *Agent Mapping*: Utility agents (translation, summarization, listing)

- **System 2**: Slow, deliberate, analytical, high-effort  
  - *Examples*: Complex reasoning, critical analysis, multi-step problems
  - *Agent Mapping*: Cognitive agents (refiner, critic, historian, synthesis)

**Architectural Implication**: The agent taxonomy should mirror this cognitive distinction, with different processing pathways for different types of thinking.

### 2. Hierarchical Task Networks (HTN)

**Framework**: Complex goals decompose into hierarchies of simpler tasks:

- **Atomic Tasks**: Indivisible operations (utility functions)
- **Composite Tasks**: Complex operations requiring planning and decomposition (cognitive operations)
- **Goal Reduction**: Higher-level goals reduce to lower-level tasks through planning

**Agent Mapping**:
```
Complex Goal: "Analyze the economic implications of this policy"
├── Atomic: "Translate the policy document" (UtilityAgent)
├── Composite: "Extract key economic assumptions" (CognitiveAgent)
├── Composite: "Identify potential contradictions" (CognitiveAgent)  
└── Composite: "Synthesize implications" (CognitiveAgent)
```

### 3. Actor Model Architecture

**Framework**: Independent actors communicate through message passing:

- **Actor Independence**: Each agent operates independently with its own state
- **Message Passing**: Asynchronous communication between agents
- **Supervision Trees**: Fault tolerance through hierarchical supervision
- **Location Transparency**: Actors can be distributed across services

**Agent Implications**:
- Utility and cognitive agents as independent actors
- Different supervision strategies for different agent types
- Clear message protocols between agent categories
- Foundation for microservice extraction

### 4. Domain-Driven Design (DDD)

**Framework**: Software structure reflects domain boundaries:

- **Bounded Contexts**: Clear boundaries between different domain areas
- **Ubiquitous Language**: Shared vocabulary within each context
- **Context Mapping**: Relationships between bounded contexts
- **Anti-Corruption Layers**: Protect domains from external influences

**Domain Mapping**:
```
Cognitive Domain:
- Language: reflection, analysis, critique, synthesis
- Entities: Insights, Perspectives, Conflicts, Themes
- Bounded Context: Deep analytical processing

Utility Domain:  
- Language: transform, extract, convert, format
- Entities: Documents, Data, Formats, Outputs
- Bounded Context: Functional transformations
```

### 5. Microservices Architecture Patterns

**Framework**: Service boundaries based on business capabilities:

- **Single Responsibility**: Each service has one reason to change
- **High Cohesion**: Related functionality grouped together
- **Loose Coupling**: Services communicate through well-defined interfaces
- **Independent Deployment**: Services can be deployed separately

**Service Implications**:
- Cognitive agents might form a "Reflection Service"
- Utility agents might form multiple specialized services (Translation, Summarization, etc.)
- Different scaling, monitoring, and deployment characteristics

## Current State Analysis

### Strengths of Current Architecture

**Cognitive Coherence**: 
- The four-agent pipeline creates a coherent analytical framework
- Each agent has a clear, non-overlapping role in the reflection process
- The sequential pipeline ensures comprehensive analysis

**Deep Insight Generation**:
- Multi-perspective analysis produces richer insights
- Critical evaluation prevents confirmation bias
- Historical context provides depth and continuity

**Architectural Clarity**:
- Clear agent boundaries and responsibilities
- Well-defined data flow through the pipeline
- Consistent interface across all agents

### Limitations for Utility Requests

**Processing Overhead**:
- Simple requests unnecessarily go through 4-agent pipeline
- Each agent adds latency even when not adding value
- Resource consumption disproportionate to request complexity

**Architectural Rigidity**:
- All requests follow the same processing pattern
- No mechanism for request classification or routing
- No way to bypass unnecessary processing steps

**User Experience Issues**:
- Slow response times for simple requests
- Over-complex responses for straightforward needs
- Potential confusion when cognitive analysis isn't desired

**Scalability Concerns**:
- Utility requests consume cognitive agent resources
- No way to scale different agent types independently
- Potential bottlenecks in high-utility-request scenarios

## Design Space Exploration

### Approach 1: Two-Tier Architecture

**Concept**: Separate cognitive and utility processing into distinct tiers with intelligent routing.

```
Request → Intent Classifier → [Cognitive Tier] → Response
                           ↘ [Utility Tier]   ↗
```

**Architecture**:
```python
class RequestRouter:
    def route(self, request: UserRequest) -> ProcessingTier:
        if self.is_cognitive_request(request):
            return CognitiveTier(agents=[Refiner, Historian, Critic, Synthesis])
        else:
            return UtilityTier(agent=self.select_utility_agent(request))

class CognitiveTier:
    """Existing agent pipeline for deep reflection"""
    def process(self, request: UserRequest) -> Response:
        # Current refiner → historian → critic → synthesis pipeline
        
class UtilityTier:
    """Fast, specialized processing for functional requests"""
    def process(self, request: UserRequest) -> Response:
        # Direct agent execution without pipeline overhead
```

**Advantages**:
- Clean separation of concerns
- Optimal performance for each request type
- Maintains existing cognitive pipeline integrity
- Simple to understand and implement

**Disadvantages**:
- Intent classification complexity
- Potential misrouting of edge cases
- Limited support for mixed requests
- Duplication of common functionality

### Approach 2: Capability-Based Agent Registry

**Concept**: Agents declare their capabilities, and requests are dynamically routed based on capability matching.

```python
class BaseAgent:
    capabilities: List[str] = []
    cognitive_depth: CognitiveDepth = CognitiveDepth.UTILITY
    
class TranslationAgent(BaseAgent):
    capabilities = ["translate", "language_detection"]
    cognitive_depth = CognitiveDepth.UTILITY
    
class CriticAgent(BaseAgent):
    capabilities = ["critical_analysis", "assumption_identification"] 
    cognitive_depth = CognitiveDepth.COGNITIVE

class AgentRegistry:
    def find_agents(self, required_capabilities: List[str]) -> List[BaseAgent]:
        return [agent for agent in self.agents 
                if any(cap in agent.capabilities for cap in required_capabilities)]
```

**Advantages**:
- Flexible and extensible
- Supports dynamic agent composition
- Clear capability declarations
- Natural evolution path for new agent types

**Disadvantages**:
- Complex orchestration logic
- Potential capability conflicts
- Difficulty in capability taxonomy management
- Performance overhead in agent selection

### Approach 3: Hierarchical Agent Composition

**Concept**: Requests are decomposed into sub-tasks, with different agent types handling different levels of the hierarchy.

```python
class TaskDecomposer:
    def decompose(self, request: UserRequest) -> TaskGraph:
        # Break complex requests into atomic and composite tasks
        
class TaskExecutor:
    def execute(self, task_graph: TaskGraph) -> Response:
        # Execute utility tasks directly
        # Execute cognitive tasks through reflection pipeline
        # Compose results hierarchically
```

**Example Flow**:
```
Request: "Translate this policy document and analyze its economic implications"
    ↓
Task Decomposition:
├── Atomic: Translate(document, target_language="en") → UtilityAgent
└── Composite: AnalyzeEconomicImplications(translated_doc) → CognitivePipeline
    ├── Refiner: Clarify economic analysis scope
    ├── Historian: Retrieve relevant economic context  
    ├── Critic: Identify economic assumptions and risks
    └── Synthesis: Compose comprehensive economic analysis
```

**Advantages**:
- Handles mixed requests elegantly
- Natural decomposition mirrors human thinking
- Maintains cognitive pipeline benefits where needed
- Supports complex multi-step workflows

**Disadvantages**:
- Task decomposition complexity
- Potential over-decomposition of simple requests
- Orchestration overhead
- Difficulty in error handling across task boundaries

### Approach 4: Event-Driven Agent Ecosystem

**Concept**: Agents communicate through events, allowing for flexible composition and loose coupling.

```python
class AgentOrchestrator:
    async def process_request(self, request: UserRequest):
        await self.event_bus.emit(UserRequestEvent(request))
        
class UtilityAgent(BaseAgent):
    async def on_user_request(self, event: UserRequestEvent):
        if self.can_handle(event.request):
            result = await self.process(event.request)
            await self.event_bus.emit(UtilityResponseEvent(result))
            
class CognitivePipeline:
    async def on_user_request(self, event: UserRequestEvent):
        if self.should_process_cognitively(event.request):
            await self.start_cognitive_processing(event.request)
```

**Advantages**:
- Loose coupling between agents
- Natural event-driven architecture alignment
- Supports parallel processing
- Flexible agent composition

**Disadvantages**:
- Event ordering complexity
- Potential race conditions
- Debugging difficulty
- Response aggregation challenges

## Architectural Implications

### Event Architecture Impact

**Event Taxonomy Expansion**:
Current events assume cognitive processing. New event types needed:

```python
# Current Cognitive Events
WORKFLOW_STARTED = "workflow.started"
AGENT_EXECUTION_STARTED = "agent.execution.started"  
ROUTING_DECISION_MADE = "routing.decision.made"

# New Utility Events  
UTILITY_REQUEST_RECEIVED = "utility.request.received"
TRANSLATION_COMPLETED = "utility.translation.completed"
SUMMARIZATION_COMPLETED = "utility.summarization.completed"
UTILITY_CHAIN_STARTED = "utility.chain.started"

# Mixed Processing Events
HYBRID_WORKFLOW_STARTED = "hybrid.workflow.started"
COGNITIVE_PHASE_STARTED = "hybrid.cognitive.started"
UTILITY_PHASE_STARTED = "hybrid.utility.started"
```

**Event Correlation Complexity**:
- Utility requests might have different correlation patterns
- Mixed requests need correlation across processing tiers
- Event replay must handle different agent types

### Service Extraction Considerations

**Service Boundary Implications**:
```
Current Monolith:
┌─────────────────────────────────┐
│        CogniVault Core          │
│  ┌─────────────────────────┐    │
│  │   Cognitive Agents      │    │
│  │  Refiner → Historian    │    │  
│  │  Critic → Synthesis     │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘

Future Microservices:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Reflection      │  │ Translation     │  │ Summarization   │
│ Service         │  │ Service         │  │ Service         │
│                 │  │                 │  │                 │
│ Refiner         │  │ Translation     │  │ Summary         │
│ Historian       │  │ Agent           │  │ Agent           │  
│ Critic          │  │                 │  │                 │
│ Synthesis       │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Service Extraction Priorities**:
1. **High Priority**: Utility services (translation, summarization) - stateless, cacheable
2. **Medium Priority**: Cognitive reflection service - stateful, complex
3. **Low Priority**: Request routing service - lightweight orchestration

### Resource Management Differences

**Performance Characteristics**:
```
Agent Type     | Latency | Memory | CPU | Cacheability | Scalability
---------------|---------|--------|-----|--------------|-------------
Utility        | Low     | Low    | Low | High         | Horizontal
Cognitive      | High    | High   | High| Low          | Vertical
Mixed          | Medium  | Medium | Med | Medium       | Complex
```

**Scaling Implications**:
- Utility agents can be stateless and horizontally scaled
- Cognitive agents may need session affinity for context
- Different monitoring and alerting requirements
- Different cost optimization strategies

### Request Routing Complexity

**Intent Classification Challenges**:
```python
# Clear Cases (Easy)
"Translate this" → UtilityAgent(Translation)
"Critically analyze this" → CognitivePipeline

# Ambiguous Cases (Hard)
"Explain this" → Could be utility (summarize) or cognitive (analyze)
"What does this mean?" → Could be utility (define) or cognitive (interpret)
"Help me understand this" → Almost certainly cognitive, but could be utility

# Mixed Cases (Complex)
"Translate and analyze" → Both utility and cognitive
"Summarize key criticism" → Utility operation on cognitive concept
```

**Classification Strategies**:
1. **Keyword-based**: Simple pattern matching on request text
2. **ML-based**: Intent classification model trained on examples
3. **Interactive**: Ask user to clarify when ambiguous
4. **Heuristic**: Default to cognitive with utility shortcuts

## Future Evolution Pathways

### Phase 1: Proof of Concept (Immediate)
**Goal**: Validate the utility agent concept with minimal disruption

**Implementation**:
- Add a single utility agent (e.g., TranslationAgent)
- Implement simple intent classification
- Route clear utility requests directly to utility agent
- Maintain existing cognitive pipeline for all other requests

**Success Criteria**:
- Translation requests bypass cognitive pipeline
- Response time improvement for translation requests
- No regression in cognitive processing quality
- Clean event emission for utility operations

### Phase 2: Utility Agent Ecosystem (3-6 months)
**Goal**: Build comprehensive utility agent support

**Implementation**:
- Add multiple utility agents (Summarization, Enumeration, Formatting)
- Implement capability-based agent registry
- Enhance intent classification with ML model
- Support simple utility chains (e.g., "Translate then summarize")

**Success Criteria**:
- 80% of utility requests correctly routed
- 5x response time improvement for utility requests
- Successful utility agent chains
- Event bus supports both cognitive and utility events

### Phase 3: Hybrid Processing (6-12 months)
**Goal**: Support complex mixed requests seamlessly

**Implementation**:
- Task decomposition engine for mixed requests
- Hierarchical agent composition
- Advanced orchestration with dependency resolution
- Cross-tier result aggregation

**Success Criteria**:
- Complex mixed requests successfully processed
- Intelligent task decomposition
- Optimal resource utilization across agent types
- Robust error handling in hybrid workflows

### Phase 4: Microservice Architecture (12-18 months)
**Goal**: Extract agent types into independent services

**Implementation**:
- Extract utility agents into specialized microservices
- Maintain cognitive agents in reflection service
- Implement service mesh for agent communication
- Advanced monitoring and observability

**Success Criteria**:
- Independent scaling of different agent types
- Service extraction without functionality loss
- Improved system reliability and maintainability
- Production-ready microservice architecture

### Extensibility Considerations

**Plugin Architecture**:
```python
class AgentPlugin:
    """Interface for external agent implementations"""
    capabilities: List[str]
    cognitive_depth: CognitiveDepth
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        ...
    
    def can_handle(self, request: AgentRequest) -> bool:
        ...

class PluginRegistry:
    """Manages agent plugins and capabilities"""
    def register(self, plugin: AgentPlugin) -> None:
        ...
    
    def find_capable_agents(self, requirements: List[str]) -> List[AgentPlugin]:
        ...
```

**Community Contributions**:
- Clear plugin interface for community-developed agents
- Agent marketplace for sharing specialized agents  
- Versioning and compatibility management
- Security sandbox for untrusted agent code

## Risk Analysis

### Technical Risks

**1. Intent Classification Accuracy**
- *Risk*: Misrouting requests leads to poor user experience
- *Mitigation*: Start with conservative classification, improve iteratively
- *Fallback*: Always allow manual override of routing decisions

**2. System Complexity Increase**
- *Risk*: Adding agent types increases overall system complexity
- *Mitigation*: Clear separation of concerns, comprehensive testing
- *Fallback*: Gradual rollout with feature flags

**3. Performance Regression**
- *Risk*: Routing overhead negates utility agent benefits
- *Mitigation*: Lightweight routing with caching
- *Fallback*: Direct routing bypass for performance-critical paths

### Architectural Risks

**1. Service Boundary Confusion**
- *Risk*: Unclear boundaries between cognitive and utility domains
- *Mitigation*: Clear domain modeling with DDD principles
- *Fallback*: Conservative boundaries with evolution over time

**2. Event Architecture Fragmentation**
- *Risk*: Different agent types create incompatible event patterns
- *Mitigation*: Unified event schema with agent-type extensions
- *Fallback*: Separate event buses with translation layers

**3. Orchestration Complexity**
- *Risk*: Mixed request handling becomes overly complex
- *Mitigation*: Start simple, add complexity gradually
- *Fallback*: Separate processing paths for mixed requests

### Business Risks

**1. User Experience Fragmentation**
- *Risk*: Different agent types create inconsistent user experience
- *Mitigation*: Unified response formatting and error handling
- *Fallback*: Configuration options for processing preferences

**2. Development Velocity Impact**
- *Risk*: Architectural changes slow down feature development
- *Mitigation*: Phased implementation with clear milestones
- *Fallback*: Parallel development tracks for different agent types

## Conclusion (Updated July 2025)

The cognitive vs utility agent taxonomy analysis successfully influenced CogniVault's architectural evolution, though the implementation took a different path than originally proposed. Rather than separate agent types, the system achieved dual-process capabilities through **configurable agent behaviors**.

**Key Insights - Implementation Results**:

1. ✅ **Dual-Process Architecture**: **ACHIEVED** through 6-axis classification system and PromptComposer enabling System 1/System 2 processing adaptation

2. ✅ **Evolutionary Approach**: **ACHIEVED** through gradual implementation of configurable prompt composition architecture

3. ✅ **Event Architecture Alignment**: **ACHIEVED** - ADR-005 Event-Driven Architecture implemented with comprehensive observability

4. ✅ **Service Extraction Preparation**: **ACHIEVED** - ADR-004 API Boundary Implementation Strategy provides microservice-ready architecture

5. 🔄 **Community Extensibility**: **IN PROGRESS** - Plugin architecture foundation exists, community system planned for Phase 3 (see ROADMAP.md)

**Implementation Achievements vs. Original Proposals**:
- **Alternative Solution**: Instead of separate utility/cognitive agent types, achieved flexibility through configurable behaviors
- **Better Integration**: Single agent types with configurable cognitive depth proved more maintainable
- **Foundation Complete**: All theoretical frameworks (Dual-Process, DDD, Actor Model) integrated into current architecture

**Current Status**:
- **Theory Implemented**: Core insights integrated into ARCHITECTURE.md design principles
- **Roadmap Alignment**: Service extraction and plugin concepts incorporated into strategic roadmap
- **Next Evolution**: Focus shifted to API service layer and community ecosystem (ROADMAP.md Phase 1-3)

**Document Status**: Analysis complete and integrated. Future agent taxonomy evolution will build on this foundation through community plugin system and microservice architecture phases.

**Related Documents**:
- ARCHITECTURE.md: Integrated theoretical foundations and design principles
- ROADMAP.md: Strategic evolution pathway including plugin architecture and service extraction
- ADR-006: Configurable Prompt Composition Architecture (implemented)

## References

- Kahneman, D. (2011). *Thinking, Fast and Slow*. Dual-process theory foundations.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Agent architectures and planning.
- Evans, E. (2003). *Domain-Driven Design*. Bounded contexts and domain modeling.
- Newman, S. (2021). *Building Microservices*. Service boundaries and extraction patterns.
- Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Architectural patterns and design guidance.

---

*This document will be updated as the agent taxonomy architecture evolves and new insights are gained through implementation and user feedback.*