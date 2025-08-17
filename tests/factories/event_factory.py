"""
Event Factory Functions for Testing.

Provides factory functions for creating event objects with sensible defaults
to eliminate parameter unfilled warnings in tests and improve maintainability.

This implements the Event Object Factory Pattern for Testing as specified
in the Event System & Observability Specialist role requirements.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock

from cognivault.events.types import (
    EventType,
    EventCategory,
    WorkflowEvent,
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    RoutingDecisionEvent,
    EventFilters,
    EventStatistics,
)
from cognivault.agents.metadata import TaskClassification, AgentMetadata

from cognivault.dependencies import DiscoveryStrategy
from cognivault.exceptions import FailurePropagationStrategy


class WorkflowEventFactory:
    """Factory for creating WorkflowEvent instances with sensible defaults."""

    @staticmethod
    def basic_workflow_event(
        event_type: EventType = EventType.WORKFLOW_STARTED,
        event_category: EventCategory = EventCategory.ORCHESTRATION,
        workflow_id: str = "test-workflow-123",
        **overrides: Any,
    ) -> WorkflowEvent:
        """Create basic workflow event with minimal required parameters."""
        return WorkflowEvent(
            event_type=event_type,
            event_category=event_category,
            workflow_id=workflow_id,
            **overrides,
        )

    @staticmethod
    def workflow_event_with_metadata(
        event_type: EventType = EventType.AGENT_EXECUTION_COMPLETED,
        event_category: EventCategory = EventCategory.EXECUTION,
        workflow_id: str = "test-workflow-456",
        correlation_id: str = "test-correlation-123",
        agent_name: str = "critic",
        **overrides: Any,
    ) -> WorkflowEvent:
        """Create workflow event with agent metadata and task classification."""
        # Only create default task classification if not provided in overrides
        if "task_classification" not in overrides:
            overrides["task_classification"] = (
                TaskClassificationFactory.basic_task_classification()
            )

        event = WorkflowEvent(
            event_type=event_type,
            event_category=event_category,
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            capabilities_used=["critical_analysis", "bias_detection"],
            execution_time_ms=1250.5,
            memory_usage_mb=128.0,
            service_version="1.2.0",
            **overrides,
        )

        # Add agent name to data for consistency
        event.data["agent_name"] = agent_name

        return event

    @staticmethod
    def error_event(
        event_type: EventType = EventType.WORKFLOW_FAILED,
        workflow_id: str = "test-workflow-error",
        error_message: str = "Test error occurred",
        error_type: str = "RuntimeError",
        **overrides: Any,
    ) -> WorkflowEvent:
        """Create workflow event with error information."""
        return WorkflowEvent(
            event_type=event_type,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id=workflow_id,
            error_message=error_message,
            error_type=error_type,
            **overrides,
        )


class WorkflowStartedEventFactory:
    """Factory for creating WorkflowStartedEvent instances."""

    @staticmethod
    def basic_workflow_started(
        workflow_id: str = "test-workflow-started",
        query: str = "Analyze climate change impact",
        agents_requested: Optional[List[str]] = None,
        **overrides: Any,
    ) -> WorkflowStartedEvent:
        """Create basic workflow started event."""
        if agents_requested is None:
            agents_requested = ["refiner", "critic"]

        # Set defaults but allow overrides
        defaults = {
            "execution_config": {"timeout_seconds": 300},
            "orchestrator_type": "langgraph-real",
        }
        defaults.update(overrides)

        return WorkflowStartedEvent(
            workflow_id=workflow_id,
            query=query,
            agents_requested=agents_requested,
            **defaults,
        )

    @staticmethod
    def workflow_started_with_long_query(
        workflow_id: str = "test-workflow-long-query",
        query_length: int = 150,
        **overrides: Any,
    ) -> WorkflowStartedEvent:
        """Create workflow started event with long query for truncation testing."""
        long_query = "x" * query_length
        return WorkflowStartedEvent(
            workflow_id=workflow_id, query=long_query, **overrides
        )


class WorkflowCompletedEventFactory:
    """Factory for creating WorkflowCompletedEvent instances."""

    @staticmethod
    def successful_workflow_completed(
        workflow_id: str = "test-workflow-completed",
        status: str = "completed",
        execution_time_seconds: float = 42.5,
        **overrides: Any,
    ) -> WorkflowCompletedEvent:
        """Create successful workflow completed event."""
        # Set defaults but allow overrides
        defaults = {
            "agent_outputs": {
                "refiner": "Refined output",
                "critic": "Critical analysis",
            },
            "successful_agents": ["refiner", "critic"],
            "failed_agents": [],
        }
        defaults.update(overrides)

        return WorkflowCompletedEvent(
            workflow_id=workflow_id,
            status=status,
            execution_time_seconds=execution_time_seconds,
            **defaults,
        )

    @staticmethod
    def failed_workflow_completed(
        workflow_id: str = "test-workflow-failed",
        status: str = "failed",
        **overrides: Any,
    ) -> WorkflowCompletedEvent:
        """Create failed workflow completed event."""
        # Set defaults but allow overrides
        defaults = {
            "execution_time_seconds": 30.0,
            "agent_outputs": {"refiner": "Partial output"},
            "successful_agents": ["refiner"],
            "failed_agents": ["critic"],
        }
        defaults.update(overrides)

        return WorkflowCompletedEvent(
            workflow_id=workflow_id,
            status=status,
            **defaults,
        )


class AgentExecutionEventFactory:
    """Factory for creating agent execution events."""

    @staticmethod
    def agent_execution_started(
        workflow_id: str = "test-workflow-agent",
        agent_name: str = "critic",
        input_context: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> AgentExecutionStartedEvent:
        """Create agent execution started event."""
        if input_context is None:
            input_context = {"query": "test query", "input_tokens": 150}

        return AgentExecutionStartedEvent(
            workflow_id=workflow_id,
            agent_name=agent_name,
            input_context=input_context,
            **overrides,
        )

    @staticmethod
    def agent_execution_completed(
        workflow_id: str = "test-workflow-agent",
        agent_name: str = "critic",
        success: bool = True,
        output_context: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> AgentExecutionCompletedEvent:
        """Create agent execution completed event."""
        if output_context is None:
            output_context = {"result": "analysis complete", "output_tokens": 200}

        return AgentExecutionCompletedEvent(
            workflow_id=workflow_id,
            agent_name=agent_name,
            success=success,
            output_context=output_context,
            **overrides,
        )


class RoutingDecisionEventFactory:
    """Factory for creating routing decision events."""

    @staticmethod
    def basic_routing_decision(
        workflow_id: str = "test-workflow-routing",
        selected_agents: Optional[List[str]] = None,
        routing_strategy: str = "capability_based",
        confidence_score: float = 0.85,
        **overrides: Any,
    ) -> RoutingDecisionEvent:
        """Create basic routing decision event."""
        if selected_agents is None:
            selected_agents = ["refiner", "critic"]

        # Provide default reasoning if not specified in overrides
        default_reasoning = {
            "criteria": ["task_complexity"],
            "explanation": "Best fit for analysis",
        }

        # Use passed-in reasoning from overrides if provided, otherwise use default
        reasoning = overrides.pop("reasoning", default_reasoning)

        return RoutingDecisionEvent(
            workflow_id=workflow_id,
            selected_agents=selected_agents,
            routing_strategy=routing_strategy,
            confidence_score=confidence_score,
            reasoning=reasoning,
            **overrides,
        )


class EventFiltersFactory:
    """Factory for creating EventFilters instances."""

    @staticmethod
    def basic_event_filters(
        event_type: Optional[EventType] = None,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        **overrides: Any,
    ) -> EventFilters:
        """Create basic event filters."""
        return EventFilters(
            event_type=event_type,
            workflow_id=workflow_id,
            agent_name=agent_name,
            **overrides,
        )

    @staticmethod
    def comprehensive_filters(
        event_type: EventType = EventType.WORKFLOW_STARTED,
        workflow_id: str = "test-workflow",
        agent_name: str = "critic",
        bounded_context: str = "reflection",
        has_errors: bool = False,
        **overrides: Any,
    ) -> EventFilters:
        """Create comprehensive event filters with all common fields."""
        return EventFilters(
            event_type=event_type,
            workflow_id=workflow_id,
            agent_name=agent_name,
            bounded_context=bounded_context,
            has_errors=has_errors,
            **overrides,
        )

    @staticmethod
    def time_range_filters(
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **overrides: Any,
    ) -> EventFilters:
        """Create filters with time range."""
        if start_time is None:
            start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        if end_time is None:
            end_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        return EventFilters(start_time=start_time, end_time=end_time, **overrides)


class EventStatisticsFactory:
    """Factory for creating EventStatistics instances."""

    @staticmethod
    def empty_statistics(**overrides: Any) -> EventStatistics:
        """Create empty event statistics."""
        return EventStatistics(**overrides)

    @staticmethod
    def basic_statistics(total_events: int = 100, **overrides: Any) -> EventStatistics:
        """Create basic event statistics with sample data."""
        return EventStatistics(
            total_events=total_events,
            events_by_type={"workflow.started": 50, "workflow.completed": 45},
            events_by_agent={"refiner": 45, "critic": 45},
            average_execution_time_ms=1250.5,
            error_rate=0.05,
            **overrides,
        )


class TaskClassificationFactory:
    """Factory for creating TaskClassification instances."""

    @staticmethod
    def basic_task_classification(
        task_type: str = "evaluate",
        domain: Optional[str] = None,
        complexity: str = "moderate",
        urgency: str = "normal",
        **overrides: Any,
    ) -> TaskClassification:
        """Create basic task classification with standard defaults."""
        return TaskClassification(
            task_type=task_type,
            domain=domain,
            complexity=complexity,
            urgency=urgency,
            **overrides,
        )

    @staticmethod
    def detailed_task_classification(
        task_type: str = "evaluate",
        domain: str = "technology",
        intent: str = "help me understand the concept",
        complexity: str = "complex",
        urgency: str = "high",
        **overrides: Any,
    ) -> TaskClassification:
        """Create detailed task classification with all fields."""
        return TaskClassification(
            task_type=task_type,
            domain=domain,
            intent=intent,
            complexity=complexity,
            urgency=urgency,
            **overrides,
        )

    @staticmethod
    def transform_task(
        domain: Optional[str] = "code",
        intent: str = "convert to JSON",
        complexity: str = "moderate",
        urgency: str = "high",
        **overrides: Any,
    ) -> TaskClassification:
        """Create task classification for transform operations."""
        return TaskClassification(
            task_type="transform",
            domain=domain,
            intent=intent,
            complexity=complexity,
            urgency=urgency,
            **overrides,
        )

    @staticmethod
    def synthesize_task(
        domain: str = "economics",
        intent: str = "combine market data",
        complexity: str = "complex",
        urgency: str = "low",
        **overrides: Any,
    ) -> TaskClassification:
        """Create task classification for synthesize operations."""
        return TaskClassification(
            task_type="synthesize",
            domain=domain,
            intent=intent,
            complexity=complexity,
            urgency=urgency,
            **overrides,
        )

    @staticmethod
    def retrieve_task(
        domain: str = "medical",
        intent: str = "find treatment options",
        complexity: str = "simple",
        urgency: str = "high",
        **overrides: Any,
    ) -> TaskClassification:
        """Create task classification for retrieve operations."""
        return TaskClassification(
            task_type="retrieve",
            domain=domain,
            intent=intent,
            complexity=complexity,
            urgency=urgency,
            **overrides,
        )

    @staticmethod
    def with_defaults(**overrides: Any) -> TaskClassification:
        """Create task classification using all default values."""
        return TaskClassification(task_type="evaluate", **overrides)


def create_mock_agent_class(
    name: str = "TestAgent", module: str = "test_module"
) -> Mock:
    """Create a properly configured mock agent class for testing."""
    mock_class: Mock = Mock()
    mock_class.__name__ = name
    mock_class.__module__ = module
    return mock_class


class AgentMetadataFactory:
    """Factory for creating AgentMetadata instances for testing.

    This factory provides comprehensive methods to cover all agent metadata test scenarios,
    including basic metadata creation, capability testing, version compatibility,
    task compatibility, performance tiers, and discovery workflows.
    """

    @staticmethod
    def basic_metadata(
        name: str = "test_agent",
        description: str = "Test agent for testing",
        cognitive_speed: str = "adaptive",
        cognitive_depth: str = "variable",
        processing_pattern: str = "atomic",
        execution_pattern: str = "processor",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create basic agent metadata with standard defaults."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            description=description,
            cognitive_speed=cognitive_speed,
            cognitive_depth=cognitive_depth,
            processing_pattern=processing_pattern,
            execution_pattern=execution_pattern,
            **overrides,
        )

    @staticmethod
    def with_mock_agent_class(
        name: str = "test_agent",
        agent_class_name: str = "TestAgent",
        agent_module: str = "test.module",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata with specific mock agent class configuration."""
        mock_agent_class = create_mock_agent_class(agent_class_name, agent_module)

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            **overrides,
        )

    @staticmethod
    def for_refiner_agent(
        name: str = "refiner",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create metadata for refiner agent with appropriate capabilities."""
        mock_agent_class = create_mock_agent_class(
            "RefinerAgent", "cognivault.agents.refiner"
        )

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            primary_capability="intent_clarification",
            capabilities=["intent_clarification"],
            **overrides,
        )

    @staticmethod
    def for_critic_agent(
        name: str = "critic",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create metadata for critic agent with critical analysis capabilities."""
        mock_agent_class = create_mock_agent_class("CriticAgent", "test_module")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            primary_capability="critical_analysis",
            secondary_capabilities=["bias_detection"],
            **overrides,
        )

    @staticmethod
    def for_historian_agent(
        name: str = "historian",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create metadata for historian agent with context retrieval capabilities."""
        mock_agent_class = create_mock_agent_class("HistorianAgent", "test_module")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            primary_capability="context_retrieval",
            **overrides,
        )

    @staticmethod
    def for_synthesis_agent(
        name: str = "synthesis",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create metadata for synthesis agent with multi-perspective synthesis."""
        mock_agent_class = create_mock_agent_class("SynthesisAgent", "test_module")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            primary_capability="multi_perspective_synthesis",
            **overrides,
        )

    @staticmethod
    def for_capability_testing(
        primary_capability: str = "translation",
        secondary_capabilities: Optional[List[str]] = None,
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata for task compatibility testing."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        if secondary_capabilities is None:
            secondary_capabilities = ["output_formatting"]

        return AgentMetadata(
            name="translator",
            agent_class=mock_agent_class,
            primary_capability=primary_capability,
            secondary_capabilities=secondary_capabilities,
            **overrides,
        )

    @staticmethod
    def for_multi_capability_testing(
        primary_capability: str = "critical_analysis",
        secondary_capabilities: Optional[List[str]] = None,
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata with multiple capabilities for comprehensive testing."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        if secondary_capabilities is None:
            secondary_capabilities = ["translation", "context_retrieval"]

        return AgentMetadata(
            name="multi_agent",
            agent_class=mock_agent_class,
            primary_capability=primary_capability,
            secondary_capabilities=secondary_capabilities,
            **overrides,
        )

    @staticmethod
    def for_replacement_testing(
        name: str = "agent",
        agent_id: str = "test_agent",
        version: str = "1.0.0",
        primary_capability: str = "general_processing",
        capabilities: Optional[List[str]] = None,
        compatibility: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata for version and replacement compatibility testing."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        if capabilities is None:
            capabilities = ["general_processing"]

        if compatibility is None:
            compatibility = {}

        return AgentMetadata(
            name=name,
            agent_id=agent_id,
            agent_class=mock_agent_class,
            version=version,
            primary_capability=primary_capability,
            capabilities=capabilities,
            compatibility=compatibility,
            **overrides,
        )

    @staticmethod
    def for_performance_testing(
        cognitive_speed: str = "fast",
        cognitive_depth: str = "shallow",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata for performance tier testing."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        return AgentMetadata(
            name="test",
            agent_class=mock_agent_class,
            cognitive_speed=cognitive_speed,
            cognitive_depth=cognitive_depth,
            **overrides,
        )

    @staticmethod
    def for_serialization_testing(
        name: str = "test_agent",
        description: str = "Test agent",
        version: str = "1.0.0",
        file_path: Optional[Path] = None,
        discovery_strategy: DiscoveryStrategy = DiscoveryStrategy.FILESYSTEM,
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata for serialization/deserialization testing."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test.module")

        if file_path is None:
            file_path = Path("/test/path")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            description=description,
            cognitive_speed="fast",
            cognitive_depth="shallow",
            version=version,
            file_path=file_path,
            discovery_strategy=discovery_strategy,
            **overrides,
        )

    @staticmethod
    def for_discovery_workflow(
        name: str = "discovered_agent",
        discovery_strategy: DiscoveryStrategy = DiscoveryStrategy.FILESYSTEM,
        file_path: Optional[Path] = None,
        checksum: str = "abc123",
        load_count: int = 5,
        is_loaded: bool = True,
        discovered_at: Optional[float] = None,
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata for discovery workflow testing."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        if file_path is None:
            file_path = Path("/agents/discovered_agent.py")

        if discovered_at is None:
            discovered_at = time.time()

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            discovered_at=discovered_at,
            discovery_strategy=discovery_strategy,
            file_path=file_path,
            checksum=checksum,
            load_count=load_count,
            is_loaded=is_loaded,
            **overrides,
        )

    @staticmethod
    def with_llm_capabilities(
        name: str = "refiner",
        requires_llm: bool = True,
        processing_pattern: str = "composite",
        pipeline_role: str = "entry",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create agent metadata with LLM capabilities for enhanced testing."""
        mock_agent_class = create_mock_agent_class("CustomAgent", "test_module")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            requires_llm=requires_llm,
            processing_pattern=processing_pattern,
            pipeline_role=pipeline_role,
            **overrides,
        )

    @staticmethod
    def fast_agent(
        name: str = "fast_transformer",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create fast, shallow agent for transform tasks."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            cognitive_speed="fast",
            cognitive_depth="shallow",
            primary_capability="translation",
            **overrides,
        )

    @staticmethod
    def thorough_agent(
        name: str = "thorough_critic",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create slow, deep agent for thorough evaluation tasks."""
        mock_agent_class = create_mock_agent_class("TestAgent", "test_module")

        return AgentMetadata(
            name=name,
            agent_class=mock_agent_class,
            cognitive_speed="slow",
            cognitive_depth="deep",
            primary_capability="critical_analysis",
            **overrides,
        )

    @staticmethod
    def basic_agent_metadata(
        name: str = "test-agent",
        description: str = "Test agent for unit testing",
        **overrides: Any,
    ) -> AgentMetadata:
        """Create basic agent metadata with mock agent class (legacy method for compatibility)."""

        # Create a simple mock agent class for testing
        class MockAgentClass:
            __module__ = "test.module"
            __name__ = "MockAgent"

            def __init__(self) -> None:
                self.name = name

            async def invoke(self, state: Any, config: Any = None) -> Any:
                return state

        return AgentMetadata(
            name=name,
            agent_class=MockAgentClass,
            description=description,
            primary_capability="test_capability",
            **overrides,
        )
