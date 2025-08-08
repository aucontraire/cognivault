"""
Tests for AggregatorNode implementation.

This module tests the AggregatorNode class which handles parallel
output combination and synthesis in the advanced node execution system.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from cognivault.orchestration.nodes.aggregator_node import (
    AggregatorNode,
    AggregationStrategy,
    AggregationInput,
)
from cognivault.orchestration.nodes.base_advanced_node import NodeExecutionContext
from cognivault.agents.metadata import AgentMetadata, TaskClassification


class TestAggregatorNodeInitialization:
    """Test AggregatorNode initialization and validation."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata with aggregator execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "deep"
        return metadata

    def test_aggregator_node_creation_success(self, mock_metadata: Mock) -> None:
        """Test successful AggregatorNode creation."""
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test_aggregator",
            aggregation_strategy=AggregationStrategy.CONSENSUS,
            min_inputs=2,
            max_inputs=5,
            quality_threshold=0.7,
            confidence_threshold=0.6,
        )

        assert node.node_name == "test_aggregator"
        assert node.execution_pattern == "aggregator"
        assert node.aggregation_strategy == AggregationStrategy.CONSENSUS
        assert node.min_inputs == 2
        assert node.max_inputs == 5
        assert node.quality_threshold == 0.7
        assert node.confidence_threshold == 0.6

    def test_aggregator_node_wrong_execution_pattern(self) -> None:
        """Test that AggregatorNode requires aggregator execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"  # Wrong pattern

        with pytest.raises(
            ValueError, match="AggregatorNode requires execution_pattern='aggregator'"
        ):
            AggregatorNode(
                metadata=metadata,
                node_name="test",
                aggregation_strategy=AggregationStrategy.CONSENSUS,
            )

    def test_aggregator_node_invalid_min_inputs(self, mock_metadata: Mock) -> None:
        """Test that min_inputs must be at least 1."""
        with pytest.raises(ValueError, match="min_inputs must be at least 1"):
            AggregatorNode(
                metadata=mock_metadata,
                node_name="test",
                aggregation_strategy=AggregationStrategy.CONSENSUS,
                min_inputs=0,
            )

    def test_aggregator_node_invalid_max_inputs(self, mock_metadata: Mock) -> None:
        """Test that max_inputs must be >= min_inputs."""
        with pytest.raises(
            ValueError, match="max_inputs \\(1\\) must be >= min_inputs \\(3\\)"
        ):
            AggregatorNode(
                metadata=mock_metadata,
                node_name="test",
                aggregation_strategy=AggregationStrategy.CONSENSUS,
                min_inputs=3,
                max_inputs=1,
            )

    def test_aggregator_node_invalid_quality_threshold(
        self, mock_metadata: Mock
    ) -> None:
        """Test that quality_threshold must be between 0.0 and 1.0."""
        with pytest.raises(
            ValueError, match="quality_threshold must be between 0.0 and 1.0"
        ):
            AggregatorNode(
                metadata=mock_metadata,
                node_name="test",
                aggregation_strategy=AggregationStrategy.CONSENSUS,
                quality_threshold=1.5,
            )

    def test_aggregator_node_invalid_confidence_threshold(
        self, mock_metadata: Mock
    ) -> None:
        """Test that confidence_threshold must be between 0.0 and 1.0."""
        with pytest.raises(
            ValueError, match="confidence_threshold must be between 0.0 and 1.0"
        ):
            AggregatorNode(
                metadata=mock_metadata,
                node_name="test",
                aggregation_strategy=AggregationStrategy.CONSENSUS,
                confidence_threshold=-0.1,
            )

    def test_aggregator_node_default_values(self, mock_metadata: Mock) -> None:
        """Test AggregatorNode with default values."""
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test",
            aggregation_strategy=AggregationStrategy.WEIGHTED,
        )

        assert node.min_inputs == 2
        assert node.max_inputs is None
        assert node.quality_threshold == 0.0
        assert node.confidence_threshold == 0.0

    def test_aggregator_node_all_strategies(self, mock_metadata: Mock) -> None:
        """Test that all aggregation strategies can be used."""
        strategies = [
            AggregationStrategy.CONSENSUS,
            AggregationStrategy.WEIGHTED,
            AggregationStrategy.HIERARCHICAL,
            AggregationStrategy.FIRST_WINS,
            AggregationStrategy.MAJORITY_VOTE,
            AggregationStrategy.AVERAGE,
            AggregationStrategy.BEST_QUALITY,
        ]

        for strategy in strategies:
            node = AggregatorNode(
                metadata=mock_metadata,
                node_name=f"test_{strategy.value}",
                aggregation_strategy=strategy,
            )
            assert node.aggregation_strategy == strategy

    def test_aggregator_node_inherits_base_methods(self, mock_metadata: Mock) -> None:
        """Test that AggregatorNode inherits BaseAdvancedNode methods."""
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test",
            aggregation_strategy=AggregationStrategy.CONSENSUS,
        )

        # Should have base class methods
        assert hasattr(node, "get_fallback_patterns")
        assert hasattr(node, "get_node_info")
        assert hasattr(node, "validate_context")
        assert hasattr(node, "pre_execute")
        assert hasattr(node, "post_execute")

        # Test fallback patterns for aggregator node
        assert node.get_fallback_patterns() == ["processor", "validator"]


class TestAggregationInput:
    """Test AggregationInput dataclass."""

    def test_aggregation_input_creation_success(self) -> None:
        """Test successful AggregationInput creation."""
        input_data = AggregationInput(
            source="agent1",
            data={"result": "test"},
            confidence=0.85,
            weight=2.0,
            quality_score=0.92,
            timestamp=1234567890.0,
        )

        assert input_data.source == "agent1"
        assert input_data.data == {"result": "test"}
        assert input_data.confidence == 0.85
        assert input_data.weight == 2.0
        assert input_data.quality_score == 0.92
        assert input_data.timestamp == 1234567890.0

    def test_aggregation_input_default_values(self) -> None:
        """Test AggregationInput with default values."""
        input_data = AggregationInput(source="agent1", data={"result": "test"})

        assert input_data.confidence == 0.0
        assert input_data.weight == 1.0
        assert input_data.quality_score == 0.0
        assert input_data.timestamp is None

    def test_aggregation_input_invalid_confidence(self) -> None:
        """Test AggregationInput with invalid confidence."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AggregationInput(source="agent1", data={"result": "test"}, confidence=1.5)

    def test_aggregation_input_invalid_weight(self) -> None:
        """Test AggregationInput with invalid weight."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AggregationInput(source="agent1", data={"result": "test"}, weight=-1.0)

    def test_aggregation_input_invalid_quality_score(self) -> None:
        """Test AggregationInput with invalid quality score."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AggregationInput(
                source="agent1", data={"result": "test"}, quality_score=2.0
            )


class TestAggregatorNodeExecute:
    """Test AggregatorNode execute method."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        return metadata

    @pytest.fixture
    def mock_context(self) -> Any:
        """Create mock NodeExecutionContext with available inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.correlation_id = "test-correlation"
        context.workflow_id = "test-workflow"
        context.cognitive_classification = {"speed": "adaptive"}
        context.task_classification = Mock()
        context.execution_path = []
        context.resource_usage = {}
        context.add_to_execution_path = Mock()
        context.update_resource_usage = Mock()

        # Mock available inputs from parallel execution
        context.available_inputs = {
            "agent1": {
                "result": "Answer from agent 1",
                "confidence": 0.85,
                "quality_score": 0.9,
                "weight": 1.0,
                "timestamp": 1234567890.0,
            },
            "agent2": {
                "result": "Answer from agent 2",
                "confidence": 0.75,
                "quality_score": 0.8,
                "weight": 2.0,
                "timestamp": 1234567891.0,
            },
            "agent3": {
                "result": "Answer from agent 3",
                "confidence": 0.95,
                "quality_score": 0.85,
                "weight": 1.5,
                "timestamp": 1234567892.0,
            },
        }

        return context

    @pytest.fixture
    def aggregator_node(self, mock_metadata: Mock) -> Any:
        """Create an AggregatorNode for testing."""
        return AggregatorNode(
            metadata=mock_metadata,
            node_name="test_aggregator",
            aggregation_strategy=AggregationStrategy.CONSENSUS,
            min_inputs=2,
            max_inputs=5,
            quality_threshold=0.7,
            confidence_threshold=0.7,
        )

    @pytest.mark.asyncio
    async def test_execute_consensus_strategy(
        self, aggregator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute with consensus aggregation strategy."""
        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ) as mock_emit:
            result = await aggregator_node.execute(mock_context)

        # Check result structure
        assert "aggregated_data" in result
        assert "quality_score" in result
        assert "conflicts_resolved" in result
        assert "aggregation_strategy" in result
        assert "input_count" in result
        assert "input_sources" in result
        assert "aggregation_time_ms" in result

        # Check metadata
        assert result["aggregation_strategy"] == "consensus"
        assert result["input_count"] == 3
        assert set(result["input_sources"]) == {"agent1", "agent2", "agent3"}
        assert result["aggregation_time_ms"] > 0

        # Check event was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["workflow_id"] == "test-workflow"
        assert call_args["aggregation_strategy"] == "consensus"
        assert set(call_args["input_sources"]) == {"agent1", "agent2", "agent3"}

    @pytest.mark.asyncio
    async def test_execute_best_quality_strategy(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute with best quality strategy."""
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test_best_quality",
            aggregation_strategy=AggregationStrategy.BEST_QUALITY,
            min_inputs=2,
        )

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await node.execute(mock_context)

        # Should select agent1 (highest quality_score: 0.9)
        assert result["aggregated_data"]["result"] == "Answer from agent 1"
        assert result["quality_score"] == 0.9
        assert result["aggregation_strategy"] == "best_quality"

    @pytest.mark.asyncio
    async def test_execute_weighted_strategy(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute with weighted strategy."""
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test_weighted",
            aggregation_strategy=AggregationStrategy.WEIGHTED,
            min_inputs=2,
        )

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await node.execute(mock_context)

        # Should select agent2 (highest weight: 2.0)
        assert result["aggregated_data"]["result"] == "Answer from agent 2"
        assert result["aggregation_strategy"] == "weighted"

    @pytest.mark.asyncio
    async def test_execute_with_thresholds(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute with quality and confidence thresholds."""
        # Set high thresholds that only agent3 meets
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test_thresholds",
            aggregation_strategy=AggregationStrategy.BEST_QUALITY,
            min_inputs=1,
            quality_threshold=0.85,  # Only agent3 has 0.85+
            confidence_threshold=0.9,  # Only agent3 has 0.95
        )

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await node.execute(mock_context)

        # Should only use agent3
        assert result["input_count"] == 1
        assert result["input_sources"] == ["agent3"]
        assert result["aggregated_data"]["result"] == "Answer from agent 3"

    @pytest.mark.asyncio
    async def test_execute_insufficient_inputs(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute fails with insufficient inputs."""
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test_insufficient",
            aggregation_strategy=AggregationStrategy.CONSENSUS,
            min_inputs=2,
            quality_threshold=0.95,  # No agents meet this threshold
            confidence_threshold=0.95,
        )

        with pytest.raises(ValueError, match="Insufficient inputs for aggregation"):
            await node.execute(mock_context)

    @pytest.mark.asyncio
    async def test_execute_max_inputs_limit(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute respects max_inputs limit."""
        node = AggregatorNode(
            metadata=mock_metadata,
            node_name="test_max_inputs",
            aggregation_strategy=AggregationStrategy.CONSENSUS,
            min_inputs=1,
            max_inputs=2,  # Limit to 2 inputs
        )

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await node.execute(mock_context)

        # Should only use 2 inputs (highest quality ones)
        assert result["input_count"] == 2
        # Should select agent1 (0.9) and agent3 (0.85) based on quality
        assert set(result["input_sources"]) == {"agent1", "agent3"}

    @pytest.mark.asyncio
    async def test_execute_calls_pre_post_hooks(
        self, aggregator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute calls pre and post execution hooks."""
        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await aggregator_node.execute(mock_context)

        # Pre-execute should add to execution path
        mock_context.add_to_execution_path.assert_called_once_with("test_aggregator")

        # Post-execute should update resource usage
        mock_context.update_resource_usage.assert_called()

    @pytest.mark.asyncio
    async def test_execute_validation_failure(self, aggregator_node: Mock) -> None:
        """Test execute fails with invalid context."""
        invalid_context = Mock(spec=NodeExecutionContext)
        invalid_context.correlation_id = ""  # Invalid
        invalid_context.workflow_id = ""
        invalid_context.cognitive_classification = {}
        invalid_context.task_classification = None

        with pytest.raises(ValueError, match="Context validation failed"):
            await aggregator_node.execute(invalid_context)

    @pytest.mark.asyncio
    async def test_execute_all_strategies(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute works with all aggregation strategies."""
        strategies = [
            AggregationStrategy.CONSENSUS,
            AggregationStrategy.WEIGHTED,
            AggregationStrategy.HIERARCHICAL,
            AggregationStrategy.FIRST_WINS,
            AggregationStrategy.MAJORITY_VOTE,
            AggregationStrategy.AVERAGE,
            AggregationStrategy.BEST_QUALITY,
        ]

        for strategy in strategies:
            node = AggregatorNode(
                metadata=mock_metadata,
                node_name=f"test_{strategy.value}",
                aggregation_strategy=strategy,
                min_inputs=1,
            )

            with patch(
                "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
            ):
                result = await node.execute(mock_context)

            # Each strategy should produce a valid result
            assert "aggregated_data" in result
            assert "quality_score" in result
            assert result["aggregation_strategy"] == strategy.value


class TestAggregatorNodeCanHandle:
    """Test AggregatorNode can_handle method."""

    @pytest.fixture
    def aggregator_node(self) -> Any:
        """Create an AggregatorNode for testing."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"

        return AggregatorNode(
            metadata=metadata,
            node_name="test_aggregator",
            aggregation_strategy=AggregationStrategy.CONSENSUS,
            min_inputs=2,
            quality_threshold=0.7,
            confidence_threshold=0.6,
        )

    def test_can_handle_sufficient_inputs(self, aggregator_node: Mock) -> None:
        """Test can_handle returns True with sufficient inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "agent1": {"result": "Answer 1", "confidence": 0.8, "quality_score": 0.9},
            "agent2": {"result": "Answer 2", "confidence": 0.7, "quality_score": 0.8},
            "agent3": {"result": "Answer 3", "confidence": 0.6, "quality_score": 0.7},
        }

        assert aggregator_node.can_handle(context) is True

    def test_can_handle_insufficient_inputs(self, aggregator_node: Mock) -> None:
        """Test can_handle returns False with insufficient inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "agent1": {"result": "Answer 1", "confidence": 0.8, "quality_score": 0.9}
        }

        # Only 1 input, but min_inputs = 2
        assert aggregator_node.can_handle(context) is False

    def test_can_handle_no_inputs(self, aggregator_node: Mock) -> None:
        """Test can_handle returns False with no inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {}

        assert aggregator_node.can_handle(context) is False

    def test_can_handle_inputs_below_thresholds(self, aggregator_node: Mock) -> None:
        """Test can_handle returns False when inputs don't meet thresholds."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "agent1": {
                "result": "Answer 1",
                "confidence": 0.5,  # Below confidence_threshold (0.6)
                "quality_score": 0.6,  # Below quality_threshold (0.7)
            },
            "agent2": {
                "result": "Answer 2",
                "confidence": 0.4,  # Below confidence_threshold (0.6)
                "quality_score": 0.5,  # Below quality_threshold (0.7)
            },
        }

        # No inputs meet both thresholds
        assert aggregator_node.can_handle(context) is False

    def test_can_handle_partial_threshold_compliance(
        self, aggregator_node: Mock
    ) -> None:
        """Test can_handle with some inputs meeting thresholds."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "agent1": {
                "result": "Answer 1",
                "confidence": 0.8,  # Above thresholds
                "quality_score": 0.9,
            },
            "agent2": {
                "result": "Answer 2",
                "confidence": 0.7,  # Above thresholds
                "quality_score": 0.8,
            },
            "agent3": {
                "result": "Answer 3",
                "confidence": 0.5,  # Below confidence_threshold
                "quality_score": 0.6,  # Below quality_threshold
            },
        }

        # 2 inputs meet thresholds, which satisfies min_inputs = 2
        assert aggregator_node.can_handle(context) is True

    def test_can_handle_invalid_input_format(self, aggregator_node: Mock) -> None:
        """Test can_handle with invalid input format."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "agent1": "not_a_dict",  # Invalid format
            "agent2": {"result": "Answer 2", "confidence": 0.8, "quality_score": 0.9},
        }

        # Only 1 valid input (agent2), but min_inputs = 2
        assert aggregator_node.can_handle(context) is False

    def test_can_handle_missing_metadata(self, aggregator_node: Mock) -> None:
        """Test can_handle with missing confidence/quality metadata."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "agent1": {
                "result": "Answer 1"
                # Missing confidence and quality_score
            },
            "agent2": {"result": "Answer 2", "confidence": 0.8, "quality_score": 0.9},
        }

        # agent1 will have confidence=0.0, quality_score=0.0 (defaults)
        # Both below thresholds, so only agent2 qualifies
        # Only 1 valid input, but min_inputs = 2
        assert aggregator_node.can_handle(context) is False

    def test_can_handle_zero_thresholds(self) -> None:
        """Test can_handle with zero thresholds accepts all inputs."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"

        node = AggregatorNode(
            metadata=metadata,
            node_name="test_zero_thresholds",
            aggregation_strategy=AggregationStrategy.CONSENSUS,
            min_inputs=2,
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "agent1": {"result": "Answer 1", "confidence": 0.1, "quality_score": 0.1},
            "agent2": {"result": "Answer 2", "confidence": 0.2, "quality_score": 0.2},
        }

        # Both inputs meet zero thresholds
        assert node.can_handle(context) is True

    def test_can_handle_evaluation_error(self, aggregator_node: Mock) -> None:
        """Test can_handle returns False when evaluation raises exception."""
        context = Mock(spec=NodeExecutionContext)
        # Set up mock to raise exception when len() is called
        context.available_inputs = Mock()
        context.available_inputs.__len__ = Mock(side_effect=Exception("Access failed"))

        assert aggregator_node.can_handle(context) is False
