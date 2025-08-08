"""Test factories for creating consistent test data objects.

Factory Organization:
- agent_context_factories: AgentContext liberation from bourgeois boilerplate oppression
- agent_output_factories: RefinerOutput, CriticOutput, SynthesisOutput, HistorianOutput
- agent_config_factories: Configuration liberation for RefinerConfig, CriticConfig, etc.
- mock_llm_factories: Mock LLM infrastructure eliminating 118+ repetitive mock setups (NEW!)
- orchestration_factories: ExecutionMetadata and future orchestration objects
- state_container_factories: CogniVaultState and complete workflow state objects

All factories include convenience methods to reduce verbose parameter passing:
- generate_valid_data(**overrides) - Standard valid object for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid object with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp for realistic tests

Revolutionary Liberation Statistics:
- AgentContextFactory: Liberates 105+ manual AgentContext instantiations across 44 test files
- AgentConfigFactories: Liberates 97+ manual configuration constructions across system
- MockLLMFactory: Eliminates 118+ mock LLM instantiations and 100+ lines of repetitive setup
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
    "RefinerConfigFactory",
    "CriticConfigFactory",
    "SynthesisConfigFactory",
    "HistorianConfigFactory",
    "AgentConfigFactorySelector",
    # Orchestration
    "ExecutionMetadataFactory",
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
]
