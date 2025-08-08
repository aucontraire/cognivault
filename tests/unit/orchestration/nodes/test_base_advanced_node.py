"""
Tests for Base Advanced Node Infrastructure.

This module tests the NodeExecutionContext dataclass and BaseAdvancedNode
abstract class that form the foundation for advanced node types.
"""

import pytest
from typing import Any, Dict
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

from cognivault.orchestration.nodes.base_advanced_node import (
    NodeExecutionContext,
    BaseAdvancedNode,
)
from cognivault.agents.metadata import AgentMetadata, TaskClassification


class TestNodeExecutionContext:
    """Test NodeExecutionContext dataclass."""

    @pytest.fixture
    def mock_task_classification(self) -> Any:
        """Create a mock TaskClassification object."""
        mock = Mock(spec=TaskClassification)
        mock.to_dict.return_value = {
            "complexity": "high",
            "domain": "analysis",
            "requires_context": True,
        }
        return mock

    @pytest.fixture
    def basic_context(self, mock_task_classification: Any) -> NodeExecutionContext:
        """Create a basic NodeExecutionContext for testing."""
        return NodeExecutionContext(
            correlation_id="test-correlation-123",
            workflow_id="workflow-456",
            cognitive_classification={
                "cognitive_speed": "slow",
                "cognitive_depth": "deep",
                "processing_pattern": "composite",
            },
            task_classification=mock_task_classification,
        )

    def test_node_execution_context_creation_basic(
        self, mock_task_classification: Any
    ) -> None:
        """Test basic NodeExecutionContext creation."""
        context = NodeExecutionContext(
            correlation_id="corr-123",
            workflow_id="wf-456",
            cognitive_classification={"speed": "fast"},
            task_classification=mock_task_classification,
        )

        assert context.correlation_id == "corr-123"
        assert context.workflow_id == "wf-456"
        assert context.cognitive_classification == {"speed": "fast"}
        assert context.task_classification == mock_task_classification
        assert context.execution_path == []
        assert context.confidence_score is None
        assert context.resource_usage is not None
        assert "start_time" in context.resource_usage
        assert isinstance(context.resource_usage["start_time"], datetime)

    def test_node_execution_context_with_all_fields(
        self, mock_task_classification: Any
    ) -> None:
        """Test NodeExecutionContext creation with all optional fields."""
        context = NodeExecutionContext(
            correlation_id="corr-789",
            workflow_id="wf-012",
            cognitive_classification={
                "cognitive_speed": "adaptive",
                "cognitive_depth": "variable",
                "processing_pattern": "chain",
            },
            task_classification=mock_task_classification,
            execution_path=["node1", "node2"],
            confidence_score=0.85,
            resource_usage={"memory_mb": 128},
            previous_nodes=["preprocessor"],
            available_inputs={"preprocessor": {"data": "processed"}},
            execution_metadata={"priority": "high"},
        )

        assert context.execution_path == ["node1", "node2"]
        assert context.confidence_score == 0.85
        assert context.resource_usage is not None
        assert context.resource_usage["memory_mb"] == 128
        assert "start_time" in context.resource_usage  # Should still add start_time
        assert context.previous_nodes == ["preprocessor"]
        assert context.available_inputs == {"preprocessor": {"data": "processed"}}
        assert context.execution_metadata == {"priority": "high"}

    def test_node_execution_context_confidence_score_validation(
        self, mock_task_classification: Any
    ) -> None:
        """Test confidence score validation."""
        # Valid confidence scores
        for score in [0.0, 0.5, 1.0]:
            context = NodeExecutionContext(
                correlation_id="corr",
                workflow_id="wf",
                cognitive_classification={},
                task_classification=mock_task_classification,
                confidence_score=score,
            )
            assert context.confidence_score == score

        # Invalid confidence scores now handled by Pydantic validation
        from pydantic import ValidationError

        for invalid_score in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValidationError):
                NodeExecutionContext(
                    correlation_id="corr",
                    workflow_id="wf",
                    cognitive_classification={},
                    task_classification=mock_task_classification,
                    confidence_score=invalid_score,
                )

    def test_add_to_execution_path(self, basic_context: NodeExecutionContext) -> None:
        """Test adding nodes to execution path."""
        assert basic_context.execution_path == []
        assert basic_context.previous_nodes == []

        basic_context.add_to_execution_path("decision_node_1")
        assert basic_context.execution_path == ["decision_node_1"]
        assert basic_context.previous_nodes == ["decision_node_1"]

        basic_context.add_to_execution_path("aggregator_node_1")
        assert basic_context.execution_path == ["decision_node_1", "aggregator_node_1"]
        assert basic_context.previous_nodes == ["decision_node_1", "aggregator_node_1"]

    def test_update_resource_usage(self, basic_context: NodeExecutionContext) -> None:
        """Test updating resource usage metrics."""
        assert basic_context.resource_usage is not None
        initial_start_time = basic_context.resource_usage["start_time"]

        basic_context.update_resource_usage(
            {"memory_mb": 256, "cpu_percent": 45.5, "tokens_used": 1500}
        )

        assert basic_context.resource_usage is not None
        assert basic_context.resource_usage["memory_mb"] == 256
        assert basic_context.resource_usage["cpu_percent"] == 45.5
        assert basic_context.resource_usage["tokens_used"] == 1500
        assert (
            basic_context.resource_usage["start_time"] == initial_start_time
        )  # Should preserve

    def test_get_execution_time_ms(self, basic_context: NodeExecutionContext) -> None:
        """Test execution time calculation."""
        # Initially no end_time, should return None
        assert basic_context.get_execution_time_ms() is None

        # Add end_time
        assert basic_context.resource_usage is not None
        start_time = basic_context.resource_usage["start_time"]
        end_time = start_time + timedelta(seconds=1.5)
        basic_context.update_resource_usage({"end_time": end_time})

        execution_time = basic_context.get_execution_time_ms()
        assert execution_time is not None
        assert abs(execution_time - 1500.0) < 0.1  # Should be ~1500ms

    def test_has_input_from(self, basic_context: NodeExecutionContext) -> None:
        """Test checking for input from specific nodes."""
        assert not basic_context.has_input_from("node1")

        basic_context.available_inputs = {
            "node1": {"output": "data"},
            "node2": {"result": "processed"},
        }

        assert basic_context.has_input_from("node1")
        assert basic_context.has_input_from("node2")
        assert not basic_context.has_input_from("node3")

    def test_get_input_from(self, basic_context: NodeExecutionContext) -> None:
        """Test retrieving input from specific nodes."""
        assert basic_context.get_input_from("node1") is None

        test_data = {"output": "processed_data", "confidence": 0.9}
        basic_context.available_inputs = {"node1": test_data}

        assert basic_context.get_input_from("node1") == test_data
        assert basic_context.get_input_from("node2") is None

    def test_to_dict(self, basic_context: NodeExecutionContext) -> None:
        """Test serialization to dictionary."""
        basic_context.add_to_execution_path("test_node")
        basic_context.confidence_score = 0.75
        basic_context.available_inputs = {"input_node": {"data": "test"}}

        result = basic_context.to_dict()

        assert result["correlation_id"] == "test-correlation-123"
        assert result["workflow_id"] == "workflow-456"
        assert result["cognitive_classification"]["cognitive_speed"] == "slow"
        assert result["task_classification"] == {
            "complexity": "high",
            "domain": "analysis",
            "requires_context": True,
        }
        assert result["execution_path"] == ["test_node"]
        assert result["confidence_score"] == 0.75
        assert "start_time" in result["resource_usage"]
        assert result["previous_nodes"] == ["test_node"]
        assert result["available_inputs"] == {"input_node": {"data": "test"}}
        assert result["execution_metadata"] == {}

    def test_resource_usage_none_handling(self, mock_task_classification: Any) -> None:
        """Test handling of None resource_usage in post_init."""
        context = NodeExecutionContext(
            correlation_id="corr",
            workflow_id="wf",
            cognitive_classification={},
            task_classification=mock_task_classification,
            resource_usage=None,  # Explicitly set to None
        )

        # Should be initialized to empty dict with start_time
        assert context.resource_usage is not None
        assert isinstance(context.resource_usage, dict)
        assert "start_time" in context.resource_usage

    @patch("cognivault.orchestration.nodes.base_advanced_node.datetime")
    def test_start_time_initialization(
        self, mock_datetime: Any, mock_task_classification: Any
    ) -> None:
        """Test that start_time is initialized with current UTC time."""
        mock_now = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = timezone

        context = NodeExecutionContext(
            correlation_id="corr",
            workflow_id="wf",
            cognitive_classification={},
            task_classification=mock_task_classification,
        )

        assert context.resource_usage is not None
        assert context.resource_usage["start_time"] == mock_now
        mock_datetime.now.assert_called_once_with(timezone.utc)


class TestBaseAdvancedNode:
    """Test BaseAdvancedNode abstract class."""

    @pytest.fixture
    def mock_agent_metadata(self) -> Any:
        """Create a mock AgentMetadata object."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"
        metadata.cognitive_speed = "fast"
        metadata.cognitive_depth = "shallow"
        metadata.processing_pattern = "atomic"
        metadata.pipeline_role = "intermediate"
        metadata.bounded_context = "transformation"
        metadata.capabilities = ["reasoning", "routing"]
        return metadata

    @pytest.fixture
    def basic_context(self) -> NodeExecutionContext:
        """Create a basic NodeExecutionContext."""
        mock_task = Mock(spec=TaskClassification)
        mock_task.to_dict.return_value = {"complexity": "medium"}

        return NodeExecutionContext(
            correlation_id="test-corr-123",
            workflow_id="test-wf-456",
            cognitive_classification={"speed": "fast"},
            task_classification=mock_task,
        )

    def test_concrete_node_implementation(
        self, mock_agent_metadata: Any, basic_context: NodeExecutionContext
    ) -> None:
        """Test creating a concrete implementation of BaseAdvancedNode."""

        class ConcreteNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {"result": "executed"}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        node = ConcreteNode(mock_agent_metadata, "test_node")

        assert node.node_name == "test_node"
        assert node.execution_pattern == "decision"
        assert node.metadata == mock_agent_metadata

    def test_base_advanced_node_invalid_execution_pattern(self) -> None:
        """Test that invalid execution patterns raise ValueError."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "invalid_pattern"

        class ConcreteNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        with pytest.raises(
            ValueError, match="Invalid execution pattern 'invalid_pattern'"
        ):
            ConcreteNode(metadata, "test_node")

    def test_get_fallback_patterns_all_types(self, mock_agent_metadata: Any) -> None:
        """Test fallback patterns for all execution types."""

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        # Test each execution pattern
        patterns_map = {
            "decision": ["processor", "terminator"],
            "aggregator": ["processor", "validator"],
            "validator": ["processor", "terminator"],
            "processor": ["terminator"],
            "terminator": [],
        }

        for pattern, expected_fallbacks in patterns_map.items():
            mock_agent_metadata.execution_pattern = pattern
            node = TestNode(mock_agent_metadata, f"{pattern}_node")
            assert node.get_fallback_patterns() == expected_fallbacks

    def test_get_node_info(self, mock_agent_metadata: Any) -> None:
        """Test getting node information."""

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        node = TestNode(mock_agent_metadata, "info_test_node")
        info = node.get_node_info()

        assert info["node_name"] == "info_test_node"
        assert info["execution_pattern"] == "decision"
        assert info["cognitive_speed"] == "fast"
        assert info["cognitive_depth"] == "shallow"
        assert info["processing_pattern"] == "atomic"
        assert info["pipeline_role"] == "intermediate"
        assert info["bounded_context"] == "transformation"
        assert info["capabilities"] == ["reasoning", "routing"]
        assert info["fallback_patterns"] == ["processor", "terminator"]

    def test_validate_context_valid(
        self, mock_agent_metadata: Any, basic_context: NodeExecutionContext
    ) -> None:
        """Test context validation with valid context."""

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        node = TestNode(mock_agent_metadata, "validation_node")
        errors = node.validate_context(basic_context)

        assert errors == []

    def test_validate_context_missing_fields(self, mock_agent_metadata: Any) -> None:
        """Test context validation with missing required fields."""

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        node = TestNode(mock_agent_metadata, "validation_node")

        # Now with Pydantic, invalid construction fails at creation time
        # Test that Pydantic prevents creation of invalid contexts
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionContext(
                correlation_id="",
                workflow_id="",
                cognitive_classification={},
                task_classification=None,
            )

        # Verify the expected validation errors
        validation_error = exc_info.value
        error_messages = [str(error) for error in validation_error.errors()]
        assert any(
            "String should have at least 1 character" in msg for msg in error_messages
        )
        assert any(
            "correlation_id" in str(error) for error in validation_error.errors()
        )
        assert any("workflow_id" in str(error) for error in validation_error.errors())
        assert any(
            "task_classification" in str(error) for error in validation_error.errors()
        )

        # Test validate_context method with a properly constructed context that has logically missing fields
        # Since Pydantic enforces validation on assignment too, we need to create a context using object.__setattr__
        # or test the validate_context method directly with mocked attributes

        # Create a valid context first
        mock_task = Mock(spec=TaskClassification)
        mock_task.to_dict.return_value = {"complexity": "medium"}
        valid_context = NodeExecutionContext(
            correlation_id="valid_id",
            workflow_id="valid_workflow",
            cognitive_classification={},
            task_classification=mock_task,
        )

        # Bypass Pydantic validation to set empty/None values for testing validate_context logic
        object.__setattr__(valid_context, "correlation_id", "")
        object.__setattr__(valid_context, "workflow_id", "")
        object.__setattr__(valid_context, "task_classification", None)

        errors = node.validate_context(valid_context)

        assert "Missing correlation_id in context" in errors
        assert "Missing workflow_id in context" in errors
        assert "Missing task_classification in context" in errors
        # cognitive_classification is {}, not None, so it passes

    @pytest.mark.asyncio
    async def test_pre_execute_hook(
        self, mock_agent_metadata: Any, basic_context: NodeExecutionContext
    ) -> None:
        """Test pre-execution hook adds node to execution path."""

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        node = TestNode(mock_agent_metadata, "pre_exec_node")

        assert basic_context.execution_path == []
        await node.pre_execute(basic_context)
        assert basic_context.execution_path == ["pre_exec_node"]
        assert basic_context.previous_nodes == ["pre_exec_node"]

    @pytest.mark.asyncio
    async def test_post_execute_hook(
        self, mock_agent_metadata: Any, basic_context: NodeExecutionContext
    ) -> None:
        """Test post-execution hook updates end time."""

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        node = TestNode(mock_agent_metadata, "post_exec_node")

        assert basic_context.resource_usage is not None
        assert "end_time" not in basic_context.resource_usage
        await node.post_execute(basic_context, {"result": "done"})
        assert basic_context.resource_usage is not None
        assert "end_time" in basic_context.resource_usage
        assert isinstance(basic_context.resource_usage["end_time"], datetime)

    def test_repr(self, mock_agent_metadata: Any) -> None:
        """Test string representation of node."""

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        node = TestNode(mock_agent_metadata, "repr_test_node")
        repr_str = repr(node)

        assert "TestNode" in repr_str
        assert "name='repr_test_node'" in repr_str
        assert "pattern='decision'" in repr_str

    @pytest.mark.asyncio
    async def test_abstract_methods_must_be_implemented(
        self, mock_agent_metadata: Any
    ) -> None:
        """Test that abstract methods must be implemented."""
        # This test verifies the abstract nature by attempting to instantiate
        # BaseAdvancedNode directly, which should fail

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAdvancedNode(mock_agent_metadata, "test")  # type: ignore[abstract]

    def test_execution_pattern_validation(self) -> None:
        """Test all valid execution patterns are accepted."""
        valid_patterns = [
            "processor",
            "decision",
            "aggregator",
            "validator",
            "terminator",
        ]

        class TestNode(BaseAdvancedNode):
            async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
                return {}

            def can_handle(self, context: NodeExecutionContext) -> bool:
                return True

        for pattern in valid_patterns:
            metadata = Mock(spec=AgentMetadata)
            metadata.execution_pattern = pattern

            # Should not raise an exception
            node = TestNode(metadata, f"{pattern}_node")
            assert node.execution_pattern == pattern
