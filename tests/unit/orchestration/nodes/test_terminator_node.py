"""
Tests for TerminatorNode implementation.

This module tests the TerminatorNode class which handles early
termination based on confidence thresholds and completion criteria.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from cognivault.orchestration.nodes.terminator_node import (
    TerminatorNode,
    TerminationCriteria,
    TerminationReason,
    TerminationReport,
)
from cognivault.orchestration.nodes.base_advanced_node import NodeExecutionContext
from cognivault.agents.metadata import AgentMetadata, TaskClassification


class TestTerminatorNodeInitialization:
    """Test TerminatorNode initialization and validation."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata with terminator execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "terminator"
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "deep"
        return metadata

    @pytest.fixture
    def basic_criteria(self) -> Any:
        """Create basic termination criteria."""
        return [
            TerminationCriteria(
                name="confidence_check",
                evaluator=lambda data: data.get("confidence", 0.0) >= 0.9,
                threshold=0.9,
                weight=1.0,
                required=True,
                description="Check if confidence is high enough",
            ),
            TerminationCriteria(
                name="quality_check",
                evaluator=lambda data: data.get("quality_score", 0.0) >= 0.85,
                threshold=0.85,
                weight=1.0,
                required=False,
                description="Check if quality is sufficient",
            ),
        ]

    def test_terminator_node_creation_success(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test successful TerminatorNode creation."""
        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=basic_criteria,
            confidence_threshold=0.95,
            quality_threshold=0.9,
            resource_limit_threshold=0.8,
            time_limit_ms=30000.0,
            allow_partial_completion=True,
            strict_mode=False,
        )

        assert node.node_name == "test_terminator"
        assert node.execution_pattern == "terminator"
        assert len(node.termination_criteria) == 2
        assert node.confidence_threshold == 0.95
        assert node.quality_threshold == 0.9
        assert node.resource_limit_threshold == 0.8
        assert node.time_limit_ms == 30000.0
        assert node.allow_partial_completion is True
        assert node.strict_mode is False

    def test_terminator_node_wrong_execution_pattern(
        self, basic_criteria: Mock
    ) -> None:
        """Test that TerminatorNode requires terminator execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "processor"  # Wrong pattern

        with pytest.raises(
            ValueError, match="TerminatorNode requires execution_pattern='terminator'"
        ):
            TerminatorNode(
                metadata=metadata, node_name="test", termination_criteria=basic_criteria
            )

    def test_terminator_node_empty_criteria(self, mock_metadata: Mock) -> None:
        """Test that TerminatorNode requires at least one criterion."""
        with pytest.raises(
            ValueError,
            match="TerminatorNode requires at least one termination criterion",
        ):
            TerminatorNode(
                metadata=mock_metadata,
                node_name="test",
                termination_criteria=[],  # Empty criteria
            )

    def test_terminator_node_invalid_confidence_threshold(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that confidence_threshold must be between 0.0 and 1.0."""
        with pytest.raises(
            ValueError, match="confidence_threshold must be between 0.0 and 1.0"
        ):
            TerminatorNode(
                metadata=mock_metadata,
                node_name="test",
                termination_criteria=basic_criteria,
                confidence_threshold=1.5,
            )

    def test_terminator_node_invalid_quality_threshold(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that quality_threshold must be between 0.0 and 1.0."""
        with pytest.raises(
            ValueError, match="quality_threshold must be between 0.0 and 1.0"
        ):
            TerminatorNode(
                metadata=mock_metadata,
                node_name="test",
                termination_criteria=basic_criteria,
                quality_threshold=-0.1,
            )

    def test_terminator_node_invalid_resource_limit_threshold(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that resource_limit_threshold must be between 0.0 and 1.0."""
        with pytest.raises(
            ValueError, match="resource_limit_threshold must be between 0.0 and 1.0"
        ):
            TerminatorNode(
                metadata=mock_metadata,
                node_name="test",
                termination_criteria=basic_criteria,
                resource_limit_threshold=2.0,
            )

    def test_terminator_node_invalid_time_limit(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that time_limit_ms must be positive."""
        with pytest.raises(ValueError, match="time_limit_ms must be positive"):
            TerminatorNode(
                metadata=mock_metadata,
                node_name="test",
                termination_criteria=basic_criteria,
                time_limit_ms=-1000.0,
            )

    def test_terminator_node_default_values(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test TerminatorNode with default values."""
        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test",
            termination_criteria=basic_criteria,
        )

        assert node.confidence_threshold == 0.95
        assert node.quality_threshold == 0.9
        assert node.resource_limit_threshold == 0.8
        assert node.time_limit_ms is None
        assert node.allow_partial_completion is True
        assert node.strict_mode is False

    def test_terminator_node_inherits_base_methods(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that TerminatorNode inherits BaseAdvancedNode methods."""
        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test",
            termination_criteria=basic_criteria,
        )

        # Should have base class methods
        assert hasattr(node, "get_fallback_patterns")
        assert hasattr(node, "get_node_info")
        assert hasattr(node, "validate_context")
        assert hasattr(node, "pre_execute")
        assert hasattr(node, "post_execute")

        # Test fallback patterns for terminator node
        assert node.get_fallback_patterns() == []


class TestTerminationCriteria:
    """Test TerminationCriteria dataclass."""

    def test_termination_criteria_creation_success(self) -> None:
        """Test successful TerminationCriteria creation."""
        criterion = TerminationCriteria(
            name="test_criterion",
            evaluator=lambda data: data.get("score", 0.0) > 0.8,
            threshold=0.8,
            weight=2.0,
            required=True,
            description="Test criterion",
        )

        assert criterion.name == "test_criterion"
        assert criterion.threshold == 0.8
        assert criterion.weight == 2.0
        assert criterion.required is True
        assert criterion.description == "Test criterion"

    def test_termination_criteria_default_values(self) -> None:
        """Test TerminationCriteria with default values."""
        criterion = TerminationCriteria(name="test", evaluator=lambda data: True)

        assert criterion.threshold == 0.0
        assert criterion.weight == 1.0
        assert criterion.required is True
        assert criterion.description == ""

    def test_termination_criteria_evaluate_method(self) -> None:
        """Test TerminationCriteria evaluate method."""
        criterion = TerminationCriteria(
            name="has_key", evaluator=lambda data: "key" in data and data["key"] > 0.5
        )

        assert criterion.evaluate({"key": 0.8}) is True
        assert criterion.evaluate({"key": 0.3}) is False
        assert criterion.evaluate({"other": 0.8}) is False


class TestTerminationReport:
    """Test TerminationReport dataclass."""

    def test_termination_report_creation(self) -> None:
        """Test TerminationReport creation."""
        report = TerminationReport(
            should_terminate=True,
            termination_reason=TerminationReason.CONFIDENCE_THRESHOLD,
            confidence_score=0.95,
            criteria_results={"test": {"met": True}},
            resource_savings={"cpu_time_ms": 1000.0},
            completion_time_ms=50.0,
            met_criteria=["test"],
            unmet_criteria=[],
            termination_message="High confidence achieved",
        )

        assert report.should_terminate is True
        assert report.termination_reason == TerminationReason.CONFIDENCE_THRESHOLD
        assert report.confidence_score == 0.95
        assert report.criteria_results == {"test": {"met": True}}
        assert report.resource_savings == {"cpu_time_ms": 1000.0}
        assert report.completion_time_ms == 50.0
        assert report.met_criteria == ["test"]
        assert report.unmet_criteria == []
        assert report.termination_message == "High confidence achieved"


class TestTerminatorNodeExecute:
    """Test TerminatorNode execute method."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "terminator"
        return metadata

    @pytest.fixture
    def mock_context(self) -> Any:
        """Create mock NodeExecutionContext with execution history."""
        context = Mock(spec=NodeExecutionContext)
        context.correlation_id = "test-correlation"
        context.workflow_id = "test-workflow"
        context.cognitive_classification = {"speed": "adaptive"}
        context.task_classification = Mock()
        context.execution_path = ["refiner", "historian", "current"]
        context.resource_usage = {"cpu_usage": 0.3, "memory_usage": 0.4}
        context.add_to_execution_path = Mock()
        context.update_resource_usage = Mock()

        # Mock available inputs with confidence scores
        context.available_inputs = {
            "input1": {
                "content": "High quality result",
                "confidence": 0.96,
                "quality_score": 0.92,
                "metadata": {"source": "agent1"},
            },
            "input2": {
                "content": "Good result",
                "confidence": 0.85,
                "quality_score": 0.88,
                "metadata": {"source": "agent2"},
            },
        }

        return context

    @pytest.fixture
    def termination_criteria(self) -> Any:
        """Create termination criteria for testing."""
        return [
            TerminationCriteria(
                name="high_confidence",
                evaluator=lambda data: data.get("confidence", 0.0) >= 0.95,
                threshold=0.95,
                weight=1.0,
                required=True,
                description="Check if confidence is very high",
            ),
            TerminationCriteria(
                name="good_quality",
                evaluator=lambda data: data.get("quality_score", 0.0) >= 0.9,
                threshold=0.9,
                weight=0.8,
                required=False,
                description="Check if quality is good enough",
            ),
            TerminationCriteria(
                name="sufficient_progress",
                evaluator=lambda data: data.get("execution_progress", 0.0) >= 0.7,
                threshold=0.7,
                weight=0.5,
                required=False,
                description="Check if execution has progressed sufficiently",
            ),
        ]

    @pytest.fixture
    def terminator_node(self, mock_metadata: Mock, termination_criteria: Mock) -> Any:
        """Create a TerminatorNode for testing."""
        return TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=termination_criteria,
            confidence_threshold=0.95,
            quality_threshold=0.9,
            resource_limit_threshold=0.8,
            time_limit_ms=None,
            allow_partial_completion=True,
            strict_mode=False,
        )

    @pytest.mark.asyncio
    async def test_execute_should_terminate_high_confidence(
        self, terminator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute recommends termination due to high confidence."""
        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await terminator_node.execute(mock_context)

        # Should recommend termination
        assert result["should_terminate"] is True
        assert result["termination_reason"] == "confidence_threshold"
        assert result["confidence_score"] >= 0.95
        assert result["recommended_action"] == "terminate"
        assert result["allow_partial_completion"] is True

        # Check criteria results
        criteria_results = result["criteria_results"]
        assert len(criteria_results) == 3
        assert criteria_results["high_confidence"]["met"] is True
        assert criteria_results["good_quality"]["met"] is True
        # Progress might not be sufficient (depends on execution path calculation)
        assert "sufficient_progress" in criteria_results

        # Check met/unmet criteria lists
        assert "high_confidence" in result["met_criteria"]
        assert "good_quality" in result["met_criteria"]
        # Progress criterion may or may not be met
        assert len(result["met_criteria"]) >= 2

        # Check resource savings
        assert "cpu_time_ms" in result["resource_savings"]
        assert "memory_mb" in result["resource_savings"]
        assert "estimated_cost" in result["resource_savings"]

        # Check event was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["workflow_id"] == "test-workflow"
        assert call_args["termination_reason"] == "confidence_threshold"
        assert call_args["threshold"] == 0.95

    @pytest.mark.asyncio
    async def test_execute_should_not_terminate_low_confidence(
        self, mock_metadata: Mock, termination_criteria: Mock, mock_context: Mock
    ) -> None:
        """Test execute does not recommend termination due to low confidence."""
        # Set up context with low confidence
        mock_context.available_inputs = {
            "input1": {
                "content": "Low confidence result",
                "confidence": 0.7,
                "quality_score": 0.75,
                "metadata": {"source": "agent1"},
            }
        }

        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=termination_criteria,
            confidence_threshold=0.95,
            quality_threshold=0.9,
            resource_limit_threshold=0.8,
            allow_partial_completion=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should not recommend termination
        assert result["should_terminate"] is False
        assert result["termination_reason"] == "confidence_threshold"
        assert result["confidence_score"] < 0.95
        assert result["recommended_action"] == "continue"

        # Check criteria results
        criteria_results = result["criteria_results"]
        assert criteria_results["high_confidence"]["met"] is False
        assert criteria_results["good_quality"]["met"] is False

        # Check met/unmet criteria lists
        assert "high_confidence" in result["unmet_criteria"]
        assert "good_quality" in result["unmet_criteria"]

        # No termination event should be emitted
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_resource_limit_termination(
        self, mock_metadata: Mock, termination_criteria: Mock, mock_context: Mock
    ) -> None:
        """Test execute recommends termination due to resource limits."""
        # Set up context with high resource usage
        mock_context.resource_usage = {"cpu_usage": 0.85, "memory_usage": 0.9}
        mock_context.available_inputs = {
            "input1": {
                "content": "Medium confidence result",
                "confidence": 0.8,
                "quality_score": 0.85,
                "metadata": {"source": "agent1"},
            }
        }

        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=termination_criteria,
            confidence_threshold=0.95,
            quality_threshold=0.9,
            resource_limit_threshold=0.8,
            allow_partial_completion=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should recommend termination due to resource limits
        assert result["should_terminate"] is True
        assert result["termination_reason"] == "resource_limit"
        assert (
            result["confidence_score"] >= 0.8
        )  # Should be the higher of cpu/memory usage
        assert result["recommended_action"] == "terminate"

        # Check event was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["termination_reason"] == "resource_limit"

    @pytest.mark.asyncio
    async def test_execute_quality_threshold_termination(
        self, mock_metadata: Mock, termination_criteria: Mock, mock_context: Mock
    ) -> None:
        """Test execute recommends termination due to quality threshold."""
        # Set up context with high quality but lower confidence
        mock_context.available_inputs = {
            "input1": {
                "content": "High quality result",
                "confidence": 0.85,
                "quality_score": 0.92,
                "metadata": {"source": "agent1"},
            }
        }

        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=termination_criteria,
            confidence_threshold=0.95,
            quality_threshold=0.9,
            resource_limit_threshold=0.8,
            allow_partial_completion=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should recommend termination due to quality threshold
        assert result["should_terminate"] is True
        assert result["termination_reason"] == "quality_threshold"
        assert result["confidence_score"] == 0.92  # Should be the quality score
        assert result["recommended_action"] == "terminate"

        # Check event was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["termination_reason"] == "quality_threshold"

    @pytest.mark.asyncio
    async def test_execute_strict_mode_all_criteria_required(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute in strict mode requires all criteria to be met."""
        # Create criteria where some will fail
        strict_criteria = [
            TerminationCriteria(
                name="always_pass",
                evaluator=lambda data: True,
                threshold=0.0,
                weight=1.0,
                required=True,
                description="Always passes",
            ),
            TerminationCriteria(
                name="always_fail",
                evaluator=lambda data: False,
                threshold=1.0,
                weight=1.0,
                required=True,
                description="Always fails",
            ),
        ]

        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=strict_criteria,
            confidence_threshold=0.99,  # Higher than available confidence
            quality_threshold=0.95,  # Higher than available quality
            resource_limit_threshold=0.95,  # Higher than available resource usage
            allow_partial_completion=True,
            strict_mode=True,
        )

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should not recommend termination because not all criteria are met
        assert result["should_terminate"] is False
        assert "always_pass" in result["met_criteria"]
        assert "always_fail" in result["unmet_criteria"]

        # No termination event should be emitted
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_completion_criteria_sufficient(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute recommends termination when sufficient completion criteria are met."""
        # Create criteria where most will pass
        completion_criteria = [
            TerminationCriteria(
                name="criterion_1",
                evaluator=lambda data: True,
                threshold=0.0,
                weight=1.0,
                required=True,
                description="Criterion 1",
            ),
            TerminationCriteria(
                name="criterion_2",
                evaluator=lambda data: True,
                threshold=0.0,
                weight=1.0,
                required=True,
                description="Criterion 2",
            ),
            TerminationCriteria(
                name="criterion_3",
                evaluator=lambda data: False,
                threshold=1.0,
                weight=1.0,
                required=True,
                description="Criterion 3",
            ),
        ]

        # Lower thresholds so completion criteria can trigger
        mock_context.available_inputs = {
            "input1": {
                "content": "Medium result",
                "confidence": 0.7,
                "quality_score": 0.75,
                "metadata": {"source": "agent1"},
            }
        }

        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=completion_criteria,
            confidence_threshold=0.95,
            quality_threshold=0.95,
            resource_limit_threshold=0.95,
            allow_partial_completion=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should recommend termination due to completion criteria (2/3 = 66.7% >= 50%)
        assert result["should_terminate"] is True
        assert result["termination_reason"] == "completion_criteria"
        assert len(result["met_criteria"]) == 2
        assert len(result["unmet_criteria"]) == 1

        # Check event was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["termination_reason"] == "completion_criteria"

    @pytest.mark.asyncio
    async def test_execute_criterion_evaluation_error(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute handles criterion evaluation errors gracefully."""

        # Create a criterion that will raise an exception
        def failing_evaluator(data: Any) -> None:
            raise ValueError("Evaluation failed")

        error_criteria = [
            TerminationCriteria(
                name="failing_criterion",
                evaluator=failing_evaluator,
                threshold=0.5,
                weight=1.0,
                required=True,
                description="This will fail",
            )
        ]

        node = TerminatorNode(
            metadata=mock_metadata,
            node_name="test_terminator",
            termination_criteria=error_criteria,
            confidence_threshold=0.99,  # Higher than available confidence
            quality_threshold=0.95,  # Higher than available quality
            resource_limit_threshold=0.95,  # Higher than available resource usage
            allow_partial_completion=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should handle the error gracefully
        assert result["should_terminate"] is False
        assert "failing_criterion" in result["unmet_criteria"]

        # Check that the error was captured in criteria results
        criteria_results = result["criteria_results"]
        assert criteria_results["failing_criterion"]["met"] is False
        assert "error" in criteria_results["failing_criterion"]
        assert "Evaluation failed" in criteria_results["failing_criterion"]["error"]

    @pytest.mark.asyncio
    async def test_execute_no_available_inputs(self, terminator_node: Mock) -> None:
        """Test execute with no available inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.correlation_id = "test-correlation"
        context.workflow_id = "test-workflow"
        context.cognitive_classification = {"speed": "adaptive"}
        context.task_classification = Mock()
        context.execution_path = ["refiner", "historian", "current"]
        context.resource_usage = {"cpu_usage": 0.3, "memory_usage": 0.4}
        context.add_to_execution_path = Mock()
        context.update_resource_usage = Mock()
        context.available_inputs = {}  # No inputs

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ) as mock_emit:
            result = await terminator_node.execute(context)

        # Should still run but with default values
        assert "should_terminate" in result
        assert result["confidence_score"] == 0.0  # Default confidence
        assert len(result["criteria_results"]) == 3

    @pytest.mark.asyncio
    async def test_execute_calls_pre_post_hooks(
        self, terminator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute calls pre and post execution hooks."""
        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ):
            result = await terminator_node.execute(mock_context)

        # Pre-execute should add to execution path
        mock_context.add_to_execution_path.assert_called_once_with("test_terminator")

        # Post-execute should update resource usage
        mock_context.update_resource_usage.assert_called()

    @pytest.mark.asyncio
    async def test_execute_validation_failure_invalid_context(
        self, terminator_node: Mock
    ) -> None:
        """Test execute fails with invalid context."""
        invalid_context = Mock(spec=NodeExecutionContext)
        invalid_context.correlation_id = ""  # Invalid
        invalid_context.workflow_id = ""
        invalid_context.cognitive_classification = {}
        invalid_context.task_classification = None

        with pytest.raises(ValueError, match="Context validation failed"):
            await terminator_node.execute(invalid_context)

    @pytest.mark.asyncio
    async def test_execute_timing_measurement(
        self, terminator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute measures evaluation timing."""
        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ):
            result = await terminator_node.execute(mock_context)

        # Should include timing information
        assert "completion_time_ms" in result
        assert result["completion_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_resource_savings_calculation(
        self, terminator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute calculates resource savings correctly."""
        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ):
            result = await terminator_node.execute(mock_context)

        # Should include resource savings
        resource_savings = result["resource_savings"]
        assert "cpu_time_ms" in resource_savings
        assert "memory_mb" in resource_savings
        assert "estimated_cost" in resource_savings

        # Values should be reasonable
        assert resource_savings["cpu_time_ms"] >= 0
        assert resource_savings["memory_mb"] >= 0
        assert resource_savings["estimated_cost"] >= 0


class TestTerminatorNodeCanHandle:
    """Test TerminatorNode can_handle method."""

    @pytest.fixture
    def terminator_node(self) -> Any:
        """Create a TerminatorNode for testing."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "terminator"

        criteria = [
            TerminationCriteria(
                name="basic_check",
                evaluator=lambda data: True,
                threshold=0.5,
                weight=1.0,
                required=True,
            )
        ]

        return TerminatorNode(
            metadata=metadata,
            node_name="test_terminator",
            termination_criteria=criteria,
            confidence_threshold=0.95,
            quality_threshold=0.9,
            resource_limit_threshold=0.8,
        )

    def test_can_handle_with_sufficient_context(self, terminator_node: Mock) -> None:
        """Test can_handle returns True with sufficient context."""
        context = Mock(spec=NodeExecutionContext)
        context.execution_path = ["refiner", "historian", "current"]
        context.available_inputs = {"input1": {"content": "test"}}
        context.resource_usage = {"cpu_usage": 0.3, "memory_usage": 0.4}

        assert terminator_node.can_handle(context) is True

    def test_can_handle_insufficient_execution_history(
        self, terminator_node: Mock
    ) -> None:
        """Test can_handle returns False with insufficient execution history."""
        context = Mock(spec=NodeExecutionContext)
        context.execution_path = ["current"]  # Only 1 step
        context.available_inputs = {"input1": {"content": "test"}}
        context.resource_usage = {"cpu_usage": 0.3, "memory_usage": 0.4}

        assert terminator_node.can_handle(context) is False

    def test_can_handle_no_available_inputs_or_results(
        self, terminator_node: Mock
    ) -> None:
        """Test can_handle returns False with no available inputs or results."""
        context = Mock(spec=NodeExecutionContext)
        context.execution_path = ["refiner", "historian", "current"]
        context.available_inputs = {}  # No inputs
        context.resource_usage = {"cpu_usage": 0.3, "memory_usage": 0.4}

        assert terminator_node.can_handle(context) is False

    def test_can_handle_no_resource_usage(self, terminator_node: Mock) -> None:
        """Test can_handle returns False with no resource usage information."""
        context = Mock(spec=NodeExecutionContext)
        context.execution_path = ["refiner", "historian", "current"]
        context.available_inputs = {"input1": {"content": "test"}}
        context.resource_usage = {}  # No resource usage

        assert terminator_node.can_handle(context) is False

    def test_can_handle_with_intermediate_results(self, terminator_node: Mock) -> None:
        """Test can_handle returns True with intermediate results."""
        context = Mock(spec=NodeExecutionContext)
        context.execution_path = ["refiner", "historian", "current"]
        context.available_inputs = {}  # No inputs
        context.intermediate_results = {
            "result1": "test"
        }  # But has intermediate results
        context.resource_usage = {"cpu_usage": 0.3, "memory_usage": 0.4}

        assert terminator_node.can_handle(context) is True

    def test_can_handle_evaluation_error(self, terminator_node: Mock) -> None:
        """Test can_handle returns False when evaluation raises exception."""
        context = Mock(spec=NodeExecutionContext)
        # Set up mock to raise exception when len() is called
        context.execution_path = Mock()
        context.execution_path.__len__ = Mock(side_effect=Exception("Access failed"))

        assert terminator_node.can_handle(context) is False
