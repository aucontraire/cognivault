"""Test factories for creating consistent test data objects.

Factory Organization:
- agent_context_factories: AgentContext liberation from bourgeois boilerplate oppression
- agent_output_factories: RefinerOutput, CriticOutput, SynthesisOutput, HistorianOutput
- agent_config_factories: Configuration liberation for RefinerConfig, CriticConfig, etc.
- api_model_factories: WorkflowRequest, WorkflowResponse via APIModelPatterns facade eliminating execution_config unfilled parameter warnings
- mock_llm_factories: Mock LLM infrastructure eliminating 118+ repetitive mock setups
- orchestration_factories: ExecutionMetadata and future orchestration objects
- advanced_orchestrator_factories: ResourceConstraint, OrchestratorConfig, ExecutionResults, ResourceAllocationResult, PipelineStage, DependencyNode eliminating 50+ orchestration parameter unfilled warnings
- state_container_factories: CogniVaultState and complete workflow state objects
- resource_scheduler_factories: ResourcePool, ResourceRequest, ResourceAllocation eliminating 14+ unfilled parameter warnings
- event_factory: WorkflowEvent, specialized event classes eliminating parameter unfilled warnings in event system tests
- routing_factories: RoutingDecision, RoutingReasoning, ResourceConstraints eliminating routing system parameter unfilled warnings
- diagnostic_health_factories: ComponentHealth, PerformanceMetrics, SystemDiagnostics eliminating diagnostic formatter parameter unfilled warnings

All factories include convenience methods to reduce verbose parameter passing:
- generate_valid_data(**overrides) - Standard valid object for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid object with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp for realistic tests

Revolutionary Liberation Statistics:
- AgentContextFactory: Liberates 105+ manual AgentContext instantiations across 44 test files
- AgentConfigFactories: Liberates 97+ manual configuration constructions across system
- APIModelPatterns: Eliminates execution_config unfilled parameter warnings and 8+ parameter specifications per WorkflowRequest instantiation
- MockLLMFactory: Eliminates 118+ mock LLM instantiations and 100+ lines of repetitive setup
- ResourceSchedulerFactory: Eliminates 14+ ResourcePool unfilled parameter warnings and 6+ parameter specifications per instantiation
- Expected boilerplate reduction: 8-12 lines → 1-2 lines per test method
- Developer morale improvement: From oppressive manual construction to joyful factory usage
"""

from .agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)

# Agent output factories
from .agent_output_factories import (
    RefinerOutputFactory,
    CriticOutputFactory,
    SynthesisOutputFactory,
    HistorianOutputFactory,
)

# Agent configuration factories
from .agent_config_factories import (
    PromptConfigFactory,
    BehavioralConfigFactory,
    OutputConfigFactory,
    AgentExecutionConfigFactory,
    RefinerConfigFactory,
    CriticConfigFactory,
    SynthesisConfigFactory,
    HistorianConfigFactory,
    AgentConfigFactorySelector,
)

# Orchestration factories
from .orchestration_factories import (
    ExecutionMetadataFactory,
)

# Advanced orchestrator factories
from .advanced_orchestrator_factories import (
    AdvancedResourceConstraintFactory,
    OrchestratorConfigFactory,
    ExecutionResultsFactory,
    ResourceAllocationResultFactory,
    PipelineStageFactory,
    DependencyNodeFactory,
    AdvancedOrchestratorTestPatterns,
)

# State container factories
from .state_container_factories import (
    CogniVaultStateFactory,
)

# Mock LLM factories
from .mock_llm_factories import (
    MockLLMFactory,
    MockLLMResponseFactory,
    ErrorScenarioFactory,
    AgentSpecificMockFactory,
    create_mock_llm,
    create_mock_response,
    create_agent_test_mocks,
)

# API model factories
from .api_model_factories import (
    APIModelFactory,
    APIModelPatterns,
)

# Resource scheduler factories
from .resource_scheduler_factories import (
    ResourcePoolFactory,
    ResourceRequestFactory,
    ResourceAllocationFactory,
    ResourceConstraintFactory,
)

# Event system factories
from .event_factory import (
    WorkflowEventFactory,
    WorkflowStartedEventFactory,
    WorkflowCompletedEventFactory,
    AgentExecutionEventFactory,
    RoutingDecisionEventFactory,
    EventFiltersFactory,
    EventStatisticsFactory,
    TaskClassificationFactory,
    AgentMetadataFactory,
)

# Routing system factories
from .routing_factories import (
    RoutingDecisionFactory,
    RoutingReasoningFactory,
    ResourceConstraintsFactory,
    RoutingTestPatterns,
)

# Diagnostic health factories
from .diagnostic_health_factories import (
    ComponentHealthFactory,
    PerformanceMetricsFactory,
    SystemDiagnosticsFactory,
    DiagnosticHealthTestPatterns,
)


__all__ = [
    # Agent context liberation
    "AgentContextFactory",
    "AgentContextPatterns",
    # Agent outputs
    "RefinerOutputFactory",
    "CriticOutputFactory",
    "SynthesisOutputFactory",
    "HistorianOutputFactory",
    # Agent configuration liberation
    "PromptConfigFactory",
    "BehavioralConfigFactory",
    "OutputConfigFactory",
    "AgentExecutionConfigFactory",
    "RefinerConfigFactory",
    "CriticConfigFactory",
    "SynthesisConfigFactory",
    "HistorianConfigFactory",
    "AgentConfigFactorySelector",
    # Orchestration
    "ExecutionMetadataFactory",
    # Advanced orchestrator liberation
    "AdvancedResourceConstraintFactory",
    "OrchestratorConfigFactory",
    "ExecutionResultsFactory",
    "ResourceAllocationResultFactory",
    "PipelineStageFactory",
    "DependencyNodeFactory",
    "AdvancedOrchestratorTestPatterns",
    # State containers
    "CogniVaultStateFactory",
    # Mock LLM infrastructure
    "MockLLMFactory",
    "MockLLMResponseFactory",
    "ErrorScenarioFactory",
    "AgentSpecificMockFactory",
    "create_mock_llm",
    "create_mock_response",
    "create_agent_test_mocks",
    # API model factories
    "APIModelFactory",
    "APIModelPatterns",
    # Resource scheduler liberation
    "ResourcePoolFactory",
    "ResourceRequestFactory",
    "ResourceAllocationFactory",
    "ResourceConstraintFactory",
    # Event system factories
    "WorkflowEventFactory",
    "WorkflowStartedEventFactory",
    "WorkflowCompletedEventFactory",
    "AgentExecutionEventFactory",
    "RoutingDecisionEventFactory",
    "EventFiltersFactory",
    "EventStatisticsFactory",
    "TaskClassificationFactory",
    "AgentMetadataFactory",
    # Routing system factories
    "RoutingDecisionFactory",
    "RoutingReasoningFactory",
    "ResourceConstraintsFactory",
    "RoutingTestPatterns",
    # Diagnostic health factories
    "ComponentHealthFactory",
    "PerformanceMetricsFactory",
    "SystemDiagnosticsFactory",
    "DiagnosticHealthTestPatterns",
]
