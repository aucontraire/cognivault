"""
Tests for agent metadata and multi-axis classification system.

Tests the enhanced AgentMetadata class with multi-axis classification,
task compatibility checking, and performance tier calculations.
"""

from unittest.mock import Mock, patch
from pathlib import Path

from cognivault.agents.metadata import (
    AgentMetadata,
    TaskClassification,
    DiscoveryStrategy,
    classify_query_task,
)
from cognivault.exceptions import FailurePropagationStrategy
from tests.factories.event_factory import (
    AgentMetadataFactory,
    TaskClassificationFactory,
)


def create_mock_agent_class(
    name: str = "TestAgent", module: str = "test_module"
) -> Mock:
    """Create a properly configured mock agent class."""
    mock_class: Mock = Mock()
    mock_class.__name__ = name
    mock_class.__module__ = module
    return mock_class


class TestAgentMetadata:
    """Test AgentMetadata functionality."""

    def test_agent_metadata_creation(self) -> None:
        """Test basic AgentMetadata creation."""
        metadata = AgentMetadataFactory.basic_metadata(
            name="test_agent",
            description="Test agent for testing",
            cognitive_speed="fast",
            cognitive_depth="shallow",
            processing_pattern="atomic",
            execution_pattern="processor",
        )

        assert metadata.name == "test_agent"
        assert metadata.agent_class is not None
        assert metadata.cognitive_speed == "fast"
        assert metadata.cognitive_depth == "shallow"
        assert metadata.execution_pattern == "processor"
        assert metadata.agent_id == "test_agent"  # Auto-derived from name

    def test_agent_metadata_post_init_derivations(self) -> None:
        """Test automatic derivations in __post_init__."""
        metadata = AgentMetadataFactory.for_refiner_agent()

        # Test automatic derivations
        assert metadata.agent_id == "refiner"
        assert metadata.module_path == "cognivault.agents.refiner.RefinerAgent"
        assert metadata.primary_capability == "intent_clarification"
        assert "intent_clarification" in metadata.capabilities

    def test_derive_capability_from_name(self) -> None:
        """Test capability derivation from agent names."""
        test_cases = [
            ("refiner", "intent_clarification"),
            ("critic", "critical_analysis"),
            ("historian", "context_retrieval"),
            ("synthesis", "multi_perspective_synthesis"),
            ("custom_agent", "custom_agent"),  # Fallback to name
        ]

        for name, expected_capability in test_cases:
            metadata = AgentMetadataFactory.with_mock_agent_class(name=name)
            assert metadata.primary_capability == expected_capability

    def test_derive_capabilities_with_llm(self) -> None:
        """Test capability derivation for LLM-enabled agents."""
        metadata = AgentMetadataFactory.with_llm_capabilities()

        expected_capabilities = {
            "intent_clarification",
            "llm_integration",
            "multi_step_processing",
            "input_processing",
        }

        assert set(metadata.capabilities) == expected_capabilities

    def test_can_replace_same_agent(self) -> None:
        """Test agent replacement compatibility for same agent."""
        metadata1 = AgentMetadataFactory.for_replacement_testing(
            name="refiner",
            agent_id="refiner_v1",
            version="1.0.0",
            primary_capability="intent_clarification",
            capabilities=["intent_clarification", "llm_integration"],
        )

        metadata2 = AgentMetadataFactory.for_replacement_testing(
            name="refiner",
            agent_id="refiner_v1",
            version="1.1.0",
            primary_capability="intent_clarification",
            capabilities=[
                "intent_clarification",
                "llm_integration",
                "enhanced_processing",
            ],
        )

        # v1.1.0 can replace v1.0.0 (newer version with superset capabilities)
        assert metadata2.can_replace(metadata1)

    def test_can_replace_different_agent(self) -> None:
        """Test agent replacement compatibility for different agents."""
        refiner_metadata = AgentMetadataFactory.for_refiner_agent(agent_id="refiner")

        critic_metadata = AgentMetadataFactory.for_critic_agent(agent_id="critic")

        # Different agents cannot replace each other
        assert not refiner_metadata.can_replace(critic_metadata)
        assert not critic_metadata.can_replace(refiner_metadata)

    def test_can_replace_version_compatibility(self) -> None:
        """Test version compatibility in agent replacement."""
        old_metadata = AgentMetadataFactory.for_replacement_testing(
            name="agent",
            agent_id="test_agent",
            version="1.0.0",
            compatibility={"min_version": "1.1.0"},  # Requires at least v1.1.0
        )

        new_metadata = AgentMetadataFactory.for_replacement_testing(
            name="agent",
            agent_id="test_agent",
            version="1.0.5",  # Below minimum required
            primary_capability=old_metadata.primary_capability,
            capabilities=old_metadata.capabilities,
        )

        # v1.0.5 cannot replace agent requiring min v1.1.0
        assert not new_metadata.can_replace(old_metadata)

    def test_is_compatible_with_task_transform(self) -> None:
        """Test task compatibility for transform tasks."""
        translator_metadata = AgentMetadataFactory.for_capability_testing()

        assert translator_metadata.is_compatible_with_task("transform")
        assert not translator_metadata.is_compatible_with_task("evaluate")

    def test_is_compatible_with_task_evaluate(self) -> None:
        """Test task compatibility for evaluate tasks."""
        critic_metadata = AgentMetadataFactory.for_critic_agent()

        assert critic_metadata.is_compatible_with_task("evaluate")
        assert not critic_metadata.is_compatible_with_task("retrieve")

    def test_is_compatible_with_task_multiple_capabilities(self) -> None:
        """Test task compatibility with multiple capabilities."""
        multi_capability_metadata = AgentMetadataFactory.for_multi_capability_testing()

        # Should be compatible with multiple task types
        assert multi_capability_metadata.is_compatible_with_task("evaluate")
        assert multi_capability_metadata.is_compatible_with_task("transform")
        assert multi_capability_metadata.is_compatible_with_task("retrieve")

    def test_get_performance_tier(self) -> None:
        """Test performance tier calculation."""
        test_cases = [
            ("fast", "shallow", "fast"),
            ("slow", "deep", "thorough"),
            ("adaptive", "variable", "balanced"),
            ("fast", "deep", "balanced"),  # Mixed characteristics
            ("slow", "shallow", "balanced"),  # Mixed characteristics
        ]

        for speed, depth, expected_tier in test_cases:
            metadata = AgentMetadataFactory.for_performance_testing(
                cognitive_speed=speed,
                cognitive_depth=depth,
            )
            assert metadata.get_performance_tier() == expected_tier

    def test_to_dict_serialization(self) -> None:
        """Test dictionary serialization."""
        metadata = AgentMetadataFactory.for_serialization_testing()

        result_dict = metadata.to_dict()

        assert result_dict["name"] == "test_agent"
        assert result_dict["agent_class"] == "test.module.TestAgent"
        assert result_dict["cognitive_speed"] == "fast"
        assert result_dict["version"] == "1.0.0"
        assert result_dict["file_path"] == "/test/path"
        assert result_dict["discovery_strategy"] == "filesystem"

    def test_create_default_success(self) -> None:
        """Test successful creation of default metadata."""
        with patch("importlib.import_module") as mock_import:
            mock_module: Mock = Mock()
            mock_base_agent = create_mock_agent_class(
                "BaseAgent", "cognivault.agents.base_agent"
            )
            mock_module.BaseAgent = mock_base_agent
            mock_import.return_value = mock_module

            metadata = AgentMetadata.create_default(
                name="test_default", description="Default test agent"
            )

            assert metadata.name == "test_default"
            assert metadata.description == "Default test agent"
            assert metadata.agent_class == mock_base_agent
            assert metadata.cognitive_speed == "adaptive"
            assert metadata.primary_capability == "general_processing"

    def test_create_default_with_custom_agent_class(self) -> None:
        """Test create_default with provided agent class."""
        mock_agent_class: Mock = Mock()
        mock_agent_class.__name__ = "TestAgent"
        mock_agent_class.__module__ = "test_module"

        metadata = AgentMetadata.create_default(
            name="custom_agent",
            agent_class=mock_agent_class,
            description="Custom agent",
        )

        assert metadata.agent_class == mock_agent_class
        assert metadata.name == "custom_agent"

    def test_create_default_import_fallback(self) -> None:
        """Test create_default fallback when BaseAgent import fails."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            metadata = AgentMetadata.create_default()

            # Should create DummyAgent fallback
            assert metadata.name == "default_agent"
            assert hasattr(metadata.agent_class, "__name__")

    def test_from_dict_deserialization(self) -> None:
        """Test dictionary deserialization."""
        agent_dict = {
            "name": "test_agent",
            "agent_class": "test.module.TestAgent",
            "description": "Test agent",
            "cognitive_speed": "fast",
            "cognitive_depth": "shallow",
            "version": "1.0.0",
            "discovery_strategy": "filesystem",
            "failure_strategy": "fail_fast",
            "file_path": "/test/path",
        }

        with patch("importlib.import_module") as mock_import:
            mock_module: Mock = Mock()
            mock_base_agent = create_mock_agent_class(
                "BaseAgent", "cognivault.agents.base_agent"
            )
            mock_module.BaseAgent = mock_base_agent
            mock_import.return_value = mock_module

            metadata = AgentMetadata.from_dict(agent_dict)

            assert metadata.name == "test_agent"
            assert metadata.cognitive_speed == "fast"
            assert metadata.version == "1.0.0"
            assert metadata.discovery_strategy == DiscoveryStrategy.FILESYSTEM
            assert metadata.failure_strategy == FailurePropagationStrategy.FAIL_FAST

    def test_from_dict_import_fallback(self) -> None:
        """Test from_dict fallback when import fails."""
        agent_dict = {"name": "test_agent", "agent_class": "nonexistent.module.Agent"}

        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            metadata = AgentMetadata.from_dict(agent_dict)

            assert metadata.name == "test_agent"
            # Should use DummyAgent fallback
            assert hasattr(metadata.agent_class, "__name__")


class TestTaskClassification:
    """Test TaskClassification functionality."""

    def test_task_classification_creation(self) -> None:
        """Test basic TaskClassification creation."""
        classification = TaskClassificationFactory.transform_task()

        assert classification.task_type == "transform"
        assert classification.domain == "code"
        assert classification.intent == "convert to JSON"
        assert classification.complexity == "moderate"
        assert classification.urgency == "high"

    def test_task_classification_defaults(self) -> None:
        """Test TaskClassification with default values."""
        classification = TaskClassificationFactory.with_defaults()

        assert classification.task_type == "evaluate"
        assert classification.domain is None
        assert classification.intent is None
        assert classification.complexity == "moderate"
        assert classification.urgency == "normal"

    def test_to_dict_serialization(self) -> None:
        """Test TaskClassification dictionary serialization."""
        classification = TaskClassificationFactory.synthesize_task()

        result_dict = classification.to_dict()

        expected = {
            "task_type": "synthesize",
            "domain": "economics",
            "intent": "combine market data",
            "complexity": "complex",
            "urgency": "low",
        }

        assert result_dict == expected

    def test_from_dict_deserialization(self) -> None:
        """Test TaskClassification dictionary deserialization."""
        classification_dict = {
            "task_type": "retrieve",
            "domain": "medical",
            "intent": "find treatment options",
            "complexity": "simple",
            "urgency": "high",
        }

        classification = TaskClassification.from_dict(classification_dict)

        assert classification.task_type == "retrieve"
        assert classification.domain == "medical"
        assert classification.intent == "find treatment options"
        assert classification.complexity == "simple"
        assert classification.urgency == "high"


class TestClassifyQueryTask:
    """Test query task classification functionality."""

    def test_classify_transform_query(self) -> None:
        """Test classification of transform queries."""
        queries = [
            "translate this text to French",
            "convert the data to JSON format",
            "transform the output structure",
        ]

        for query in queries:
            classification = classify_query_task(query)
            assert classification.task_type == "transform"

    def test_classify_evaluate_query(self) -> None:
        """Test classification of evaluate queries."""
        queries = [
            "analyze the market trends",
            "evaluate this proposal",
            "critique the research methodology",
            "assess the risk factors",
        ]

        for query in queries:
            classification = classify_query_task(query)
            assert classification.task_type == "evaluate"

    def test_classify_retrieve_query(self) -> None:
        """Test classification of retrieve queries."""
        queries = [
            "find information about climate change",
            "search for relevant documents",
            "retrieve user preferences",
            "lookup the latest statistics",
        ]

        for query in queries:
            classification = classify_query_task(query)
            assert classification.task_type == "retrieve"

    def test_classify_synthesize_query(self) -> None:
        """Test classification of synthesize queries."""
        queries = [
            "combine the research findings",
            "synthesize the expert opinions",
            "merge the data sources",
            "integrate multiple perspectives",
        ]

        for query in queries:
            classification = classify_query_task(query)
            assert classification.task_type == "synthesize"

    def test_classify_explain_query(self) -> None:
        """Test classification of explain queries."""
        queries = [
            "explain quantum computing",
            "clarify the requirements",
            "help me understand the process",
        ]

        for query in queries:
            classification = classify_query_task(query)
            assert classification.task_type == "explain"

    def test_classify_complexity_by_length(self) -> None:
        """Test complexity classification based on query length."""
        short_query = "analyze data"
        medium_query = "analyze the market data trends and provide insights into potential growth opportunities"
        long_query = "analyze the comprehensive market data trends across multiple sectors and provide detailed insights into potential growth opportunities, considering economic factors, competitive landscape, regulatory environment, and technological disruptions that might impact future performance"

        short_classification = classify_query_task(short_query)
        medium_classification = classify_query_task(medium_query)
        long_classification = classify_query_task(long_query)

        assert short_classification.complexity == "simple"
        assert medium_classification.complexity == "moderate"
        assert long_classification.complexity == "complex"

    def test_classify_urgency_keywords(self) -> None:
        """Test urgency classification based on keywords."""
        urgent_query = "urgently analyze the data ASAP"
        normal_query = "analyze the data when convenient"
        low_query = "analyze the data when you have time, no rush"

        urgent_classification = classify_query_task(urgent_query)
        normal_classification = classify_query_task(normal_query)
        low_classification = classify_query_task(low_query)

        assert urgent_classification.urgency == "high"
        assert normal_classification.urgency == "low"  # "when convenient" triggers low
        assert low_classification.urgency == "low"

    def test_classify_domain_detection(self) -> None:
        """Test domain detection from query content."""
        test_cases = [
            ("analyze the economic indicators", "economics"),
            ("review the Python code functionality", "code"),
            ("evaluate the government policy impact", "policy"),
            ("assess the medical treatment options", "medical"),
            ("examine the research study methodology", "science"),
            ("analyze general data trends", None),  # No specific domain
        ]

        for query, expected_domain in test_cases:
            classification = classify_query_task(query)
            assert classification.domain == expected_domain

    def test_classify_default_fallback(self) -> None:
        """Test default classification for ambiguous queries."""
        ambiguous_query = "process this information"

        classification = classify_query_task(ambiguous_query)

        # Should default to "evaluate" for complex queries
        assert classification.task_type == "evaluate"
        assert classification.complexity == "simple"  # Short query
        assert classification.urgency == "normal"

    def test_classify_intent_truncation(self) -> None:
        """Test intent field truncation for long queries."""
        long_query = "this is a very long query that should be truncated when used as intent because it exceeds the fifty character limit"

        classification = classify_query_task(long_query)

        assert classification.intent is not None
        assert len(classification.intent) <= 53  # 50 chars + "..."
        assert classification.intent.endswith("...")


class TestAgentMetadataIntegration:
    """Integration tests for AgentMetadata with other systems."""

    def test_metadata_with_discovery_workflow(self) -> None:
        """Test metadata in agent discovery workflow."""
        metadata = AgentMetadataFactory.for_discovery_workflow()

        # Test discovery metadata
        assert metadata.discovery_strategy == DiscoveryStrategy.FILESYSTEM
        assert metadata.file_path == Path("/agents/discovered_agent.py")
        assert metadata.checksum == "abc123"
        assert metadata.load_count == 5
        assert metadata.is_loaded

    def test_metadata_performance_and_task_integration(self) -> None:
        """Test integration between performance tiers and task compatibility."""

        # Fast, shallow agent good for simple transforms
        fast_agent = AgentMetadataFactory.fast_agent()

        # Slow, deep agent good for complex evaluation
        thorough_agent = AgentMetadataFactory.thorough_agent()

        # Test performance tier mapping
        assert fast_agent.get_performance_tier() == "fast"
        assert thorough_agent.get_performance_tier() == "thorough"

        # Test task compatibility
        assert fast_agent.is_compatible_with_task("transform")
        assert not fast_agent.is_compatible_with_task("evaluate")

        assert thorough_agent.is_compatible_with_task("evaluate")
        assert not thorough_agent.is_compatible_with_task("transform")
