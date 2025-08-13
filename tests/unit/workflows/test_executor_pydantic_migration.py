"""
Test suite for workflows/executor.py Pydantic migration.

This test suite validates the Pydantic migration of ExecutionContext,
CompositionResult, and WorkflowResult models including validation,
serialization, and backward compatibility.
"""

import pytest
import time
from typing import Any, Dict, Union
from unittest.mock import MagicMock, Mock

from pydantic import ValidationError

from cognivault.workflows.executor import (
    ExecutionContext,
    CompositionResult,
    WorkflowResult,
)
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)


def create_test_agent_context(query: str = "test query") -> AgentContext:
    """Create a test AgentContext for testing."""
    return AgentContextPatterns.simple_query(query)


# Resolve forward references by importing and rebuilding
try:
    from cognivault.workflows.definition import WorkflowDefinition

    ExecutionContext.model_rebuild()
    CompositionResult.model_rebuild()
    WorkflowResult.model_rebuild()
except (ImportError, Exception):
    # If we can't resolve forward references, tests will need to handle it
    pass


def create_test_workflow() -> Union[object, Mock]:
    """Create a test WorkflowDefinition for testing."""
    try:
        from cognivault.workflows.definition import (
            WorkflowDefinition,
            NodeConfiguration,
            FlowDefinition,
        )

        node = NodeConfiguration(
            node_id="test_node", node_type="processor", category="BASE"
        )
        flow = FlowDefinition(entry_point="test_node", edges=[])
        return WorkflowDefinition(
            name="Test Workflow",
            version="1.0.0",
            workflow_id="test-workflow-123",
            nodes=[node],
            flow=flow,
        )
    except ImportError:
        # Fallback to using a mock with required attributes
        mock_workflow: Mock = Mock()
        mock_workflow.name = "Test Workflow"
        mock_workflow.version = "1.0.0"
        mock_workflow.workflow_id = "test-workflow-123"
        return mock_workflow


class TestExecutionContext:
    """Test ExecutionContext Pydantic model."""

    def test_default_values(self) -> None:
        """Test default field values."""
        # Create a real WorkflowDefinition instance for testing
        try:
            from cognivault.workflows.definition import (
                WorkflowDefinition,
                NodeConfiguration,
                FlowDefinition,
            )

            node = NodeConfiguration(
                node_id="test_node", node_type="processor", category="BASE"
            )
            flow = FlowDefinition(entry_point="test_node", edges=[])
            mock_workflow = WorkflowDefinition(
                name="Test Workflow",
                version="1.0.0",
                workflow_id="test-workflow-123",
                nodes=[node],
                flow=flow,
            )
        except ImportError:
            # Fallback to using model_validate with arbitrary_types_allowed
            mock_workflow_fallback: Mock = Mock()
            mock_workflow_fallback.name = "Test Workflow"
            mock_workflow_fallback.version = "1.0.0"
            mock_workflow = mock_workflow_fallback

        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
        )

        assert context.workflow_id == "test-123"
        assert context.workflow_definition == mock_workflow
        assert context.query == "test query"
        assert context.execution_config == {}
        assert context.status == "pending"
        assert context.metadata == {}
        assert isinstance(context.start_time, float)

    def test_field_descriptions(self) -> None:
        """Test that all fields have descriptions."""
        schema = ExecutionContext.model_json_schema()
        properties = schema["properties"]

        required_fields = [
            "workflow_id",
            "workflow_definition",
            "query",
            "execution_config",
            "start_time",
            "status",
            "metadata",
        ]

        for field in required_fields:
            assert field in properties
            assert "description" in properties[field]
            assert len(properties[field]["description"]) > 0

    def test_update_status_method(self) -> None:
        """Test update_status method."""
        mock_workflow = create_test_workflow()
        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
        )

        assert context.status == "pending"
        context.update_status("running")
        assert context.status == "running"

    def test_add_metadata_method(self) -> None:
        """Test add_metadata method."""
        mock_workflow = create_test_workflow()
        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
        )

        assert context.metadata == {}
        context.add_metadata("key1", "value1")
        assert context.metadata == {"key1": "value1"}

        context.add_metadata("key2", {"nested": "value"})
        assert context.metadata == {"key1": "value1", "key2": {"nested": "value"}}

    def test_serialization(self) -> None:
        """Test Pydantic serialization."""
        mock_workflow = create_test_workflow()

        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
            execution_config={"timeout": 30},
            status="running",
            metadata={"step": 1},
        )

        # Test dict serialization
        data = context.model_dump()
        assert data["workflow_id"] == "test-123"
        assert data["query"] == "test query"
        assert data["execution_config"] == {"timeout": 30}
        assert data["status"] == "running"
        assert data["metadata"] == {"step": 1}

    def test_model_validation(self) -> None:
        """Test model validation with required fields."""
        # Valid creation
        mock_workflow = create_test_workflow()
        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
        )
        assert context.workflow_id == "test-123"

        # Test that required fields are enforced
        with pytest.raises(ValidationError):
            ExecutionContext.model_validate(
                {
                    "workflow_id": "test-id",
                    "workflow_definition": "mock_definition",
                    # Missing query field should cause validation error
                }
            )


class TestCompositionResult:
    """Test CompositionResult Pydantic model."""

    def test_default_values(self) -> None:
        """Test default field values."""
        result = CompositionResult()

        assert result.node_mapping == {}
        assert result.edge_mapping == {}
        assert result.metadata == {}
        assert result.validation_errors == []

    def test_field_descriptions(self) -> None:
        """Test that all fields have descriptions."""
        schema = CompositionResult.model_json_schema()
        properties = schema["properties"]

        required_fields = [
            "node_mapping",
            "edge_mapping",
            "metadata",
            "validation_errors",
        ]

        for field in required_fields:
            assert field in properties
            assert "description" in properties[field]
            assert len(properties[field]["description"]) > 0

    def test_with_data(self) -> None:
        """Test CompositionResult with actual data."""
        node_mapping = {"node1": "instance1", "node2": "instance2"}
        edge_mapping = {"edge1": "connection1"}
        metadata = {"composition_time": 1.5}
        validation_errors = ["Error 1", "Error 2"]

        result = CompositionResult(
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
            metadata=metadata,
            validation_errors=validation_errors,
        )

        assert result.node_mapping == node_mapping
        assert result.edge_mapping == edge_mapping
        assert result.metadata == metadata
        assert result.validation_errors == validation_errors

    def test_serialization(self) -> None:
        """Test Pydantic serialization."""
        result = CompositionResult(
            node_mapping={"node1": "value1"},
            edge_mapping={"edge1": "value1"},
            metadata={"time": 1.0},
            validation_errors=["error1"],
        )

        data = result.model_dump()
        assert data["node_mapping"] == {"node1": "value1"}
        assert data["edge_mapping"] == {"edge1": "value1"}
        assert data["metadata"] == {"time": 1.0}
        assert data["validation_errors"] == ["error1"]


class TestWorkflowResult:
    """Test WorkflowResult Pydantic model."""

    def test_default_values(self) -> None:
        """Test default field values."""
        mock_context = create_test_agent_context()
        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
        )

        assert result.workflow_id == "workflow-123"
        assert result.execution_id == "exec-456"
        assert result.final_context == mock_context
        assert result.execution_metadata == {}
        assert result.node_execution_order == []
        assert result.execution_time_seconds == 0.0
        assert result.success is True
        assert result.error_message is None
        assert result.event_correlation_id == ""

    def test_field_descriptions(self) -> None:
        """Test that all fields have descriptions."""
        schema = WorkflowResult.model_json_schema()
        properties = schema["properties"]

        required_fields = [
            "workflow_id",
            "execution_id",
            "final_context",
            "execution_metadata",
            "node_execution_order",
            "execution_time_seconds",
            "success",
            "error_message",
            "event_correlation_id",
        ]

        for field in required_fields:
            assert field in properties
            assert "description" in properties[field]
            assert len(properties[field]["description"]) > 0

    def test_execution_time_validation(self) -> None:
        """Test execution_time_seconds validation."""
        mock_context = create_test_agent_context()

        # Valid positive value
        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
            execution_time_seconds=30.5,
        )
        assert result.execution_time_seconds == 30.5

        # Valid zero value
        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
            execution_time_seconds=0.0,
        )
        assert result.execution_time_seconds == 0.0

        # Invalid negative value
        with pytest.raises(ValidationError) as exc_info:
            WorkflowResult(
                workflow_id="workflow-123",
                execution_id="exec-456",
                final_context=mock_context,
                execution_time_seconds=-5.0,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_comprehensive_result(self) -> None:
        """Test WorkflowResult with all fields populated."""
        mock_context = create_test_agent_context()

        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
            execution_metadata={"nodes_executed": 4, "total_llm_calls": 8},
            node_execution_order=["refiner", "historian", "critic", "synthesis"],
            execution_time_seconds=25.7,
            success=True,
            error_message=None,
            event_correlation_id="corr-789",
        )

        assert result.workflow_id == "workflow-123"
        assert result.execution_id == "exec-456"
        assert result.final_context == mock_context
        assert result.execution_metadata == {"nodes_executed": 4, "total_llm_calls": 8}
        assert result.node_execution_order == [
            "refiner",
            "historian",
            "critic",
            "synthesis",
        ]
        assert result.execution_time_seconds == 25.7
        assert result.success is True
        assert result.error_message is None
        assert result.event_correlation_id == "corr-789"

    def test_failed_execution_result(self) -> None:
        """Test WorkflowResult for failed execution."""
        mock_context = create_test_agent_context()

        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
            execution_time_seconds=5.2,
            success=False,
            error_message="Agent execution timeout",
        )

        assert result.success is False
        assert result.error_message == "Agent execution timeout"
        assert result.execution_time_seconds == 5.2

    def test_serialization(self) -> None:
        """Test Pydantic serialization."""
        mock_context = create_test_agent_context()

        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
            execution_metadata={"test": "data"},
            node_execution_order=["node1", "node2"],
            execution_time_seconds=15.5,
            success=True,
            event_correlation_id="corr-123",
        )

        data = result.model_dump()
        assert data["workflow_id"] == "workflow-123"
        assert data["execution_id"] == "exec-456"
        assert data["execution_metadata"] == {"test": "data"}
        assert data["node_execution_order"] == ["node1", "node2"]
        assert data["execution_time_seconds"] == 15.5
        assert data["success"] is True
        assert data["event_correlation_id"] == "corr-123"

    def test_required_fields(self) -> None:
        """Test that required fields are enforced."""
        mock_context = create_test_agent_context()

        # Valid creation
        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
        )
        assert result.workflow_id == "workflow-123"

        # Missing required fields
        with pytest.raises(ValidationError):
            WorkflowResult.model_validate(
                {"workflow_id": "test", "execution_id": "test"}
            )  # Missing final_context

        with pytest.raises(ValidationError):
            WorkflowResult.model_validate(
                {"workflow_id": "test"}
            )  # Missing execution_id and final_context


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality."""

    def test_method_signatures_preserved(self) -> None:
        """Test that existing method signatures are preserved."""
        mock_workflow = create_test_workflow()
        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
        )

        # These methods should still exist and work
        assert hasattr(context, "update_status")
        assert hasattr(context, "add_metadata")

        # Test they work as expected
        context.update_status("running")
        assert context.status == "running"

        context.add_metadata("test_key", "test_value")
        assert context.metadata["test_key"] == "test_value"

    def test_attribute_access_preserved(self) -> None:
        """Test that attribute access patterns are preserved."""
        mock_workflow = create_test_workflow()
        mock_context = create_test_agent_context()

        # Test ExecutionContext
        exec_ctx = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
        )

        # These should work like before
        assert exec_ctx.workflow_id == "test-123"
        assert exec_ctx.status == "pending"

        # Test CompositionResult
        comp_result = CompositionResult()
        assert comp_result.node_mapping == {}
        assert comp_result.validation_errors == []

        # Test WorkflowResult
        wf_result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
        )
        assert wf_result.success is True
        assert wf_result.execution_time_seconds == 0.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_complex_nested_data(self) -> None:
        """Test with complex nested data structures."""
        mock_workflow = create_test_workflow()

        complex_config = {
            "nested": {"deep": {"data": [1, 2, 3], "mapping": {"key": "value"}}},
            "list_of_dicts": [{"item": 1}, {"item": 2}],
        }

        context = ExecutionContext(
            workflow_id="test-123",
            workflow_definition=mock_workflow,
            query="test query",
            execution_config=complex_config,
        )

        assert context.execution_config == complex_config

        # Test serialization roundtrip
        data = context.model_dump()
        assert data["execution_config"] == complex_config

    def test_large_data_structures(self) -> None:
        """Test with large data structures."""
        mock_context = create_test_agent_context()

        # Large node execution order
        large_order = [f"node_{i}" for i in range(1000)]

        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
            node_execution_order=large_order,
        )

        assert len(result.node_execution_order) == 1000
        assert result.node_execution_order[0] == "node_0"
        assert result.node_execution_order[999] == "node_999"

    def test_empty_and_none_values(self) -> None:
        """Test handling of empty and None values."""
        mock_context = create_test_agent_context()

        result = WorkflowResult(
            workflow_id="workflow-123",
            execution_id="exec-456",
            final_context=mock_context,
            error_message=None,  # Explicit None
            event_correlation_id="",  # Empty string
        )

        assert result.error_message is None
        assert result.event_correlation_id == ""

        # Test serialization handles None/empty correctly
        data = result.model_dump()
        assert data["error_message"] is None
        assert data["event_correlation_id"] == ""
