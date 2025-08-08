"""
Comprehensive tests for Events System Pydantic migration.

Tests validation, serialization, and backward compatibility of event models.
"""

import pytest
from typing import Any
from datetime import datetime, timezone
from pydantic import ValidationError

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
from cognivault.exceptions import FailurePropagationStrategy


class TestWorkflowEventValidation:
    """Test WorkflowEvent Pydantic validation."""

    def test_valid_minimal_event(self) -> None:
        """Test valid minimal event creation."""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow-123",
        )
        assert event.event_type == EventType.WORKFLOW_STARTED
        assert event.event_category == EventCategory.ORCHESTRATION
        assert event.workflow_id == "test-workflow-123"
        assert len(event.event_id) == 32  # UUID hex string
        assert event.timestamp is not None
        assert event.correlation_id is None
        assert event.capabilities_used == []

    def test_valid_full_event(self) -> None:
        """Test valid event with all fields."""
        task_classification = TaskClassification(
            task_type="evaluate", domain="technology", complexity="complex"
        )

        event = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_COMPLETED,
            event_category=EventCategory.EXECUTION,
            workflow_id="workflow-456",
            correlation_id="trace-abc123",
            parent_span_id="span-def456",
            task_classification=task_classification,
            capabilities_used=["critical_analysis", "bias_detection"],
            data={"agent_name": "critic", "success": True},
            execution_time_ms=1250.5,
            memory_usage_mb=128.0,
            service_version="1.2.0",
        )

        assert event.correlation_id == "trace-abc123"
        assert event.task_classification.task_type == "evaluate"
        assert "critical_analysis" in event.capabilities_used
        assert event.execution_time_ms == 1250.5

    def test_workflow_id_validation(self) -> None:
        """Test workflow_id field validation."""
        # Empty workflow_id should fail
        with pytest.raises(ValidationError, match="at least 1 character"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="",
            )

        # Very long workflow_id should fail
        with pytest.raises(ValidationError, match="at most 200 characters"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="x" * 201,
            )

    def test_event_id_validation(self) -> None:
        """Test event_id validation."""
        # Invalid hex string (proper length but non-hex chars) should fail
        with pytest.raises(ValidationError, match="hexadecimal characters"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                event_id="gggggggggggggggggggggggggggggggg",  # 32 chars, but non-hex
            )

        # Wrong length should fail
        with pytest.raises(ValidationError, match="32-character hex string"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                event_id="abc123",  # Too short
            )

    def test_capabilities_validation(self) -> None:
        """Test capabilities_used validation."""
        # Non-list should fail
        with pytest.raises(ValidationError, match="valid list"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                capabilities_used="not_a_list",
            )

        # Empty string capability should fail
        with pytest.raises(ValidationError, match="non-empty strings"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                capabilities_used=["valid_capability", ""],
            )

    def test_performance_metrics_validation(self) -> None:
        """Test performance metrics validation."""
        # Negative execution time should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                execution_time_ms=-1.0,
            )

        # Negative memory usage should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                memory_usage_mb=-10.0,
            )

    def test_error_consistency_validation(self) -> None:
        """Test error field consistency validation."""
        # Error message without error type should fail
        with pytest.raises(ValidationError, match="error_type is required"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_FAILED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                error_message="Something went wrong",
            )

        # Error type without error message should fail
        with pytest.raises(ValidationError, match="error_message is required"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_FAILED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                error_type="RuntimeError",
            )

        # Both error fields provided should pass
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_FAILED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test",
            error_message="Something went wrong",
            error_type="RuntimeError",
        )
        assert event.error_message == "Something went wrong"
        assert event.error_type == "RuntimeError"

    def test_service_version_validation(self) -> None:
        """Test service_version pattern validation."""
        # Invalid version format should fail
        with pytest.raises(ValidationError, match="String should match pattern"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                service_version="invalid.version",
            )

        # Valid versions should pass
        valid_versions = ["1.0.0", "2.5.10", "1.0.0-beta.1", "3.2.1-rc.2"]
        for version in valid_versions:
            event = WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                service_version=version,
            )
            assert event.service_version == version

    def test_event_category_validation(self) -> None:
        """Test event_category field validation and dual emission architecture."""
        # Valid orchestration category should work
        orchestration_event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow-123",
        )
        assert orchestration_event.event_category == EventCategory.ORCHESTRATION

        # Valid execution category should work
        execution_event = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_STARTED,
            event_category=EventCategory.EXECUTION,
            workflow_id="test-workflow-456",
        )
        assert execution_event.event_category == EventCategory.EXECUTION

        # Both categories can be used for same event type (dual emission)
        agent_orchestration = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_STARTED,
            event_category=EventCategory.ORCHESTRATION,  # From node wrappers
            workflow_id="test-dual-emission",
        )
        agent_execution = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_STARTED,
            event_category=EventCategory.EXECUTION,  # From individual agents
            workflow_id="test-dual-emission",
        )

        # Verify both events are valid but have different categories
        assert agent_orchestration.event_category == EventCategory.ORCHESTRATION
        assert agent_execution.event_category == EventCategory.EXECUTION
        assert agent_orchestration.event_type == agent_execution.event_type

    def test_event_category_enum_values(self) -> None:
        """Test that event category enum has correct values for WebSocket consumption."""
        # Verify enum values match expected WebSocket output
        assert EventCategory.ORCHESTRATION.value == "orchestration"
        assert EventCategory.EXECUTION.value == "execution"

        # Test that enum can be serialized properly
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_COMPLETED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-enum-serialization",
        )

        # Should be able to get the string value for WebSocket transmission
        category_str = event.event_category.value
        assert category_str in ["orchestration", "execution"]
        assert isinstance(category_str, str)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                event_category=EventCategory.ORCHESTRATION,
                workflow_id="test",
                extra_field="not_allowed",
            )


class TestSpecializedEventValidation:
    """Test specialized event classes validation."""

    def test_workflow_started_event(self) -> None:
        """Test WorkflowStartedEvent validation."""
        event = WorkflowStartedEvent(
            workflow_id="test-workflow",
            query="Analyze climate change impact",
            agents_requested=["refiner", "critic"],
            execution_config={"timeout_seconds": 300},
            orchestrator_type="langgraph-real",
        )

        assert event.event_type == EventType.WORKFLOW_STARTED
        assert event.query == "Analyze climate change impact"
        assert event.agents_requested == ["refiner", "critic"]
        assert event.data["query_length"] == len("Analyze climate change impact")
        assert event.data["agents_requested"] == ["refiner", "critic"]

    def test_workflow_started_query_truncation(self) -> None:
        """Test query truncation in data."""
        long_query = "x" * 150
        event = WorkflowStartedEvent(workflow_id="test-workflow", query=long_query)

        # Original query should be preserved
        assert event.query == long_query
        # But data should have truncated version
        assert event.data["query"] == long_query[:100] + "..."
        assert event.data["query_length"] == 150

    def test_workflow_completed_event(self) -> None:
        """Test WorkflowCompletedEvent validation."""
        event = WorkflowCompletedEvent(
            workflow_id="test-workflow",
            status="completed",
            execution_time_seconds=42.5,
            agent_outputs={"refiner": "Refined output", "critic": "Critical analysis"},
            successful_agents=["refiner", "critic"],
            failed_agents=[],
        )

        assert event.event_type == EventType.WORKFLOW_COMPLETED
        assert event.status == "completed"
        assert event.execution_time_ms == 42500.0  # Converted to ms
        assert event.data["success_rate"] == 1.0  # All agents successful

    def test_workflow_completed_status_validation(self) -> None:
        """Test status pattern validation."""
        # Invalid status should fail
        with pytest.raises(ValidationError, match="String should match pattern"):
            WorkflowCompletedEvent(workflow_id="test", status="invalid_status")

        # Valid statuses should pass
        valid_statuses = ["completed", "failed", "cancelled", "partial_failure"]
        for status in valid_statuses:
            event = WorkflowCompletedEvent(workflow_id="test", status=status)
            assert event.status == status

    def test_agent_execution_events(self) -> None:
        """Test agent execution events."""
        # Started event
        started = AgentExecutionStartedEvent(
            workflow_id="test",
            agent_name="critic",
            input_context={"query": "test", "input_tokens": 150},
        )
        assert started.event_type == EventType.AGENT_EXECUTION_STARTED
        assert started.data["input_tokens"] == 150

        # Completed event
        completed = AgentExecutionCompletedEvent(
            workflow_id="test",
            agent_name="critic",
            success=True,
            output_context={"result": "analysis", "output_tokens": 200},
        )
        assert completed.event_type == EventType.AGENT_EXECUTION_COMPLETED
        assert completed.data["output_tokens"] == 200

    def test_routing_decision_event(self) -> None:
        """Test RoutingDecisionEvent validation."""
        event = RoutingDecisionEvent(
            workflow_id="test",
            selected_agents=["refiner", "critic"],
            routing_strategy="capability_based",
            confidence_score=0.85,
            reasoning={"criteria": ["task_complexity"], "explanation": "Best fit"},
        )

        assert event.event_type == EventType.ROUTING_DECISION_MADE
        assert event.confidence_score == 0.85
        assert event.data["agent_count"] == 2

    def test_routing_confidence_validation(self) -> None:
        """Test confidence score validation."""
        # Confidence below 0 should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            RoutingDecisionEvent(workflow_id="test", confidence_score=-0.1)

        # Confidence above 1 should fail
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            RoutingDecisionEvent(workflow_id="test", confidence_score=1.5)


class TestEventFiltersValidation:
    """Test EventFilters Pydantic validation."""

    def test_valid_filters(self) -> None:
        """Test valid filter creation."""
        filters = EventFilters(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id="test-workflow",
            agent_name="critic",
            bounded_context="reflection",
            has_errors=False,
        )

        assert filters.event_type == EventType.WORKFLOW_STARTED
        assert filters.bounded_context == "reflection"

    def test_time_range_validation(self) -> None:
        """Test time range validation."""
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)  # Before start

        # Start time after end time should fail
        with pytest.raises(ValidationError, match="start_time must be before end_time"):
            EventFilters(start_time=start_time, end_time=end_time)

        # Valid time range should pass
        valid_end_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        filters = EventFilters(start_time=start_time, end_time=valid_end_time)
        assert filters.start_time == start_time
        assert filters.end_time == valid_end_time

    def test_bounded_context_validation(self) -> None:
        """Test bounded_context pattern validation."""
        # Invalid bounded context should fail
        with pytest.raises(ValidationError, match="String should match pattern"):
            EventFilters(bounded_context="invalid_context")

        # Valid contexts should pass
        valid_contexts = ["reflection", "transformation", "retrieval"]
        for context in valid_contexts:
            filters = EventFilters(bounded_context=context)
            assert filters.bounded_context == context

    def test_filter_matching(self) -> None:
        """Test filter matching functionality."""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow",
            capabilities_used=["critical_analysis"],
        )
        event.data["agent_name"] = "critic"

        # Matching filters
        filters = EventFilters(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id="test-workflow",
            agent_name="critic",
            capability="critical_analysis",
        )
        assert filters.matches(event) is True

        # Non-matching filters
        non_matching_filters = EventFilters(
            event_type=EventType.WORKFLOW_COMPLETED  # Different type
        )
        assert non_matching_filters.matches(event) is False


class TestEventStatisticsValidation:
    """Test EventStatistics Pydantic validation."""

    def test_valid_statistics(self) -> None:
        """Test valid statistics creation."""
        stats = EventStatistics(
            total_events=100,
            events_by_type={"workflow.started": 50, "workflow.completed": 45},
            events_by_agent={"refiner": 45, "critic": 45},
            average_execution_time_ms=1250.5,
            error_rate=0.05,
        )

        assert stats.total_events == 100
        assert stats.error_rate == 0.05

    def test_count_validation(self) -> None:
        """Test event count validation."""
        # Negative total events should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            EventStatistics(total_events=-1)

        # Negative counts in dictionaries should fail
        with pytest.raises(ValidationError, match="non-negative integer"):
            EventStatistics(events_by_type={"workflow.started": -5})

    def test_error_rate_validation(self) -> None:
        """Test error rate validation."""
        # Error rate below 0 should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            EventStatistics(error_rate=-0.1)

        # Error rate above 1 should fail
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            EventStatistics(error_rate=1.5)

    def test_update_with_event(self) -> None:
        """Test statistics update functionality."""
        stats = EventStatistics()

        event = WorkflowEvent(
            event_type=EventType.AGENT_EXECUTION_COMPLETED,
            event_category=EventCategory.EXECUTION,
            workflow_id="test",
            execution_time_ms=1000.0,
            capabilities_used=["critical_analysis"],
        )
        event.data["agent_name"] = "critic"

        stats.update_with_event(event)

        assert stats.total_events == 1
        assert stats.events_by_type["agent.execution.completed"] == 1
        assert stats.events_by_agent["critic"] == 1
        assert stats.events_by_capability["critical_analysis"] == 1
        assert stats.average_execution_time_ms == 1000.0


class TestTaskClassificationValidation:
    """Test TaskClassification Pydantic validation."""

    def test_valid_classification(self) -> None:
        """Test valid task classification."""
        classification = TaskClassification(
            task_type="evaluate",
            domain="technology",
            intent="help me understand the concept",
            complexity="complex",
            urgency="high",
        )

        assert classification.task_type == "evaluate"
        assert classification.domain == "technology"
        assert classification.complexity == "complex"

    def test_required_task_type(self) -> None:
        """Test that task_type is required."""
        with pytest.raises(ValidationError, match="Field required"):
            TaskClassification()

    def test_string_length_validation(self) -> None:
        """Test string field length validation."""
        # Very long domain should fail
        with pytest.raises(ValidationError, match="at most 100 characters"):
            TaskClassification(task_type="evaluate", domain="x" * 101)

        # Very long intent should fail
        with pytest.raises(ValidationError, match="at most 500 characters"):
            TaskClassification(task_type="evaluate", intent="x" * 501)


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_to_dict_methods(self) -> None:
        """Test that all models have to_dict methods."""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test",
        )

        filters = EventFilters(event_type=EventType.WORKFLOW_STARTED)

        classification = TaskClassification(task_type="evaluate")

        # All should have to_dict methods
        assert hasattr(event, "to_dict")
        assert hasattr(classification, "to_dict")

        # to_dict should return dictionaries
        event_dict = event.to_dict()
        classification_dict = classification.to_dict()

        assert isinstance(event_dict, dict)
        assert isinstance(classification_dict, dict)

        # Should contain expected fields
        assert "event_type" in event_dict
        assert "workflow_id" in event_dict
        assert "task_type" in classification_dict

    def test_from_dict_methods(self) -> None:
        """Test from_dict class methods."""
        # Create original objects
        original_event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test-workflow",
        )

        original_classification = TaskClassification(
            task_type="evaluate", domain="technology"
        )

        # Convert to dict and back
        event_dict = original_event.to_dict()
        restored_event = WorkflowEvent.from_dict(event_dict)

        classification_dict = original_classification.to_dict()
        restored_classification = TaskClassification.from_dict(classification_dict)

        # Should match original
        assert restored_event.event_type == original_event.event_type
        assert restored_event.workflow_id == original_event.workflow_id

        assert restored_classification.task_type == original_classification.task_type
        assert restored_classification.domain == original_classification.domain

    def test_serialization_compatibility(self) -> None:
        """Test JSON serialization works correctly."""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            event_category=EventCategory.ORCHESTRATION,
            workflow_id="test",
            capabilities_used=["analysis", "synthesis"],
        )

        # Should be able to serialize/deserialize with Pydantic
        data = event.model_dump()
        restored = WorkflowEvent(**data)

        assert restored.event_type == event.event_type
        assert restored.workflow_id == event.workflow_id
        assert restored.capabilities_used == event.capabilities_used
