"""
Tests for DecisionNode implementation.

This module tests the DecisionNode class which handles conditional
routing and flow control in the advanced node execution system.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from cognivault.orchestration.nodes.decision_node import (
    DecisionNode,
    DecisionCriteria,
)
from cognivault.orchestration.nodes.base_advanced_node import NodeExecutionContext
from cognivault.agents.metadata import AgentMetadata, TaskClassification


class TestDecisionNodeInitialization:
    """Test DecisionNode initialization and validation."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata with decision execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "variable"
        return metadata

    @pytest.fixture
    def basic_criteria(self) -> Any:
        """Create basic decision criteria."""
        return [
            DecisionCriteria(
                name="complexity",
                evaluator=lambda ctx: ctx.confidence_score or 0.5,
                weight=1.0,
                threshold=0.7,
            ),
            DecisionCriteria(
                name="resources", evaluator=lambda ctx: 0.8, weight=0.5, threshold=0.5
            ),
        ]

    @pytest.fixture
    def basic_paths(self) -> Any:
        """Create basic path configuration."""
        return {
            "fast_path": ["refiner", "synthesis"],
            "deep_path": ["refiner", "historian", "critic", "synthesis"],
            "adaptive_path": ["refiner", "historian", "synthesis"],
        }

    def test_decision_node_creation_success(
        self, mock_metadata: Mock, basic_criteria: Mock, basic_paths: Mock
    ) -> None:
        """Test successful DecisionNode creation."""
        node = DecisionNode(
            metadata=mock_metadata,
            node_name="test_decision",
            decision_criteria=basic_criteria,
            paths=basic_paths,
        )

        assert node.node_name == "test_decision"
        assert node.execution_pattern == "decision"
        assert len(node.decision_criteria) == 2
        assert len(node.paths) == 3
        assert node.default_path == "fast_path"  # First path is default

    def test_decision_node_wrong_execution_pattern(
        self, basic_criteria: Mock, basic_paths: Mock
    ) -> None:
        """Test that DecisionNode requires decision execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "processor"  # Wrong pattern

        with pytest.raises(
            ValueError, match="DecisionNode requires execution_pattern='decision'"
        ):
            DecisionNode(
                metadata=metadata,
                node_name="test",
                decision_criteria=basic_criteria,
                paths=basic_paths,
            )

    def test_decision_node_empty_criteria(
        self, mock_metadata: Mock, basic_paths: Mock
    ) -> None:
        """Test that DecisionNode requires at least one criterion."""
        with pytest.raises(
            ValueError, match="DecisionNode requires at least one decision criterion"
        ):
            DecisionNode(
                metadata=mock_metadata,
                node_name="test",
                decision_criteria=[],  # Empty criteria
                paths=basic_paths,
            )

    def test_decision_node_empty_paths(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that DecisionNode requires at least one path."""
        with pytest.raises(ValueError, match="DecisionNode requires at least one path"):
            DecisionNode(
                metadata=mock_metadata,
                node_name="test",
                decision_criteria=basic_criteria,
                paths={},  # Empty paths
            )

    def test_decision_criteria_evaluate(self) -> None:
        """Test DecisionCriteria evaluation."""
        context = Mock(spec=NodeExecutionContext)
        context.confidence_score = 0.85

        criterion = DecisionCriteria(
            name="confidence_check",
            evaluator=lambda ctx: ctx.confidence_score,
            weight=2.0,
            threshold=0.8,
        )

        score = criterion.evaluate(context)
        assert score == 0.85

    def test_decision_node_inherits_base_methods(
        self, mock_metadata: Mock, basic_criteria: Mock, basic_paths: Mock
    ) -> None:
        """Test that DecisionNode inherits BaseAdvancedNode methods."""
        node = DecisionNode(
            metadata=mock_metadata,
            node_name="test",
            decision_criteria=basic_criteria,
            paths=basic_paths,
        )

        # Should have base class methods
        assert hasattr(node, "get_fallback_patterns")
        assert hasattr(node, "get_node_info")
        assert hasattr(node, "validate_context")
        assert hasattr(node, "pre_execute")
        assert hasattr(node, "post_execute")

        # Test fallback patterns for decision node
        assert node.get_fallback_patterns() == ["processor", "terminator"]


class TestDecisionNodeExecute:
    """Test DecisionNode execute method."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"
        return metadata

    @pytest.fixture
    def mock_context(self) -> Any:
        """Create mock NodeExecutionContext."""
        context = Mock(spec=NodeExecutionContext)
        context.correlation_id = "test-correlation"
        context.workflow_id = "test-workflow"
        context.cognitive_classification = {"speed": "fast"}
        context.task_classification = Mock()
        context.confidence_score = 0.85
        context.execution_path = []
        context.previous_nodes = []
        context.resource_usage = {}
        return context

    @pytest.fixture
    def decision_node(self, mock_metadata: Mock) -> Any:
        """Create a DecisionNode for testing."""
        criteria = [
            DecisionCriteria(
                name="complexity",
                evaluator=lambda ctx: ctx.confidence_score,
                weight=1.0,
                threshold=0.8,
            )
        ]
        paths = {
            "high_confidence": ["agent1", "agent2"],
            "low_confidence": ["agent3", "agent4", "agent5"],
        }
        return DecisionNode(mock_metadata, "test_decision", criteria, paths)

    @pytest.mark.asyncio
    async def test_execute_high_confidence_path(
        self, decision_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute selects high confidence path."""
        mock_context.confidence_score = 0.9  # Above threshold

        with patch(
            "cognivault.orchestration.nodes.decision_node.emit_decision_made"
        ) as mock_emit:
            result = await decision_node.execute(mock_context)

        # Check result
        assert result["selected_path"] == "high_confidence"
        assert result["selected_agents"] == ["agent1", "agent2"]
        assert result["confidence"] == 0.9
        assert "low_confidence" in result["alternatives"]

        # Check event was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["workflow_id"] == "test-workflow"
        assert call_args["decision_criteria"] == ["complexity"]
        assert call_args["selected_path"] == "high_confidence"
        assert call_args["confidence_score"] == 0.9

    @pytest.mark.asyncio
    async def test_execute_low_confidence_path(
        self, decision_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute selects low confidence path."""
        mock_context.confidence_score = 0.5  # Below threshold

        with patch(
            "cognivault.orchestration.nodes.decision_node.emit_decision_made"
        ) as mock_emit:
            result = await decision_node.execute(mock_context)

        # Check result - should select path with lower requirements
        assert result["selected_path"] in ["high_confidence", "low_confidence"]
        assert "confidence" in result
        assert result["alternatives"]

    @pytest.mark.asyncio
    async def test_execute_calls_pre_post_hooks(
        self, decision_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute calls pre and post execution hooks."""
        # Mock the methods that should be called
        mock_context.add_to_execution_path = Mock()
        mock_context.update_resource_usage = Mock()

        with patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"):
            result = await decision_node.execute(mock_context)

        # Pre-execute should add to execution path
        mock_context.add_to_execution_path.assert_called_once_with("test_decision")

        # Post-execute should update resource usage
        mock_context.update_resource_usage.assert_called()

    @pytest.mark.asyncio
    async def test_execute_validation_failure(self, decision_node: Mock) -> None:
        """Test execute fails with invalid context."""
        invalid_context = Mock(spec=NodeExecutionContext)
        invalid_context.correlation_id = ""  # Invalid
        invalid_context.workflow_id = ""
        invalid_context.cognitive_classification = {}
        invalid_context.task_classification = None

        with pytest.raises(ValueError, match="Context validation failed"):
            await decision_node.execute(invalid_context)

    @pytest.mark.asyncio
    async def test_execute_multiple_criteria(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute with multiple weighted criteria."""
        criteria = [
            DecisionCriteria(
                name="complexity", evaluator=lambda ctx: 0.9, weight=2.0, threshold=0.8
            ),
            DecisionCriteria(
                name="resources", evaluator=lambda ctx: 0.6, weight=1.0, threshold=0.5
            ),
        ]
        paths = {"optimal": ["agent1"], "fallback": ["agent2", "agent3"]}

        node = DecisionNode(mock_metadata, "multi_criteria", criteria, paths)

        with patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"):
            result = await node.execute(mock_context)

        # Both criteria pass, so weighted score should be high
        assert result["selected_path"] == "optimal"
        assert result["reasoning"]["criterion_scores"]["complexity"]["passed"] is True
        assert result["reasoning"]["criterion_scores"]["resources"]["passed"] is True


class TestDecisionNodeCanHandle:
    """Test DecisionNode can_handle method."""

    @pytest.fixture
    def decision_node(self) -> Any:
        """Create a DecisionNode for testing."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"

        criteria = [
            DecisionCriteria(
                name="has_confidence",
                evaluator=lambda ctx: ctx.confidence_score,
                weight=1.0,
                threshold=0.5,
            ),
            DecisionCriteria(
                name="has_classification",
                evaluator=lambda ctx: len(ctx.cognitive_classification) > 0,
                weight=1.0,
                threshold=0.5,
            ),
        ]
        paths = {"default": ["agent1"]}

        return DecisionNode(metadata, "can_handle_test", criteria, paths)

    def test_can_handle_valid_context(self, decision_node: Mock) -> None:
        """Test can_handle returns True for valid context."""
        context = Mock(spec=NodeExecutionContext)
        context.cognitive_classification = {"speed": "fast"}
        context.confidence_score = 0.75

        assert decision_node.can_handle(context) is True

    def test_can_handle_missing_classification(self, decision_node: Mock) -> None:
        """Test can_handle returns False when cognitive_classification is missing."""
        context = Mock(spec=NodeExecutionContext)
        context.cognitive_classification = None
        context.confidence_score = 0.75

        assert decision_node.can_handle(context) is False

    def test_can_handle_empty_classification(self, decision_node: Mock) -> None:
        """Test can_handle with empty cognitive_classification."""
        context = Mock(spec=NodeExecutionContext)
        context.cognitive_classification = {}
        context.confidence_score = 0.75

        # Should return False because len(ctx.cognitive_classification) = 0
        assert decision_node.can_handle(context) is False

    def test_can_handle_evaluation_error(self) -> None:
        """Test can_handle returns False when criteria evaluation fails."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"

        # Create a criterion that will fail
        def failing_evaluator(ctx: Any) -> None:
            raise ValueError("Evaluation failed")

        criteria = [
            DecisionCriteria(
                name="failing_criterion",
                evaluator=failing_evaluator,
                weight=1.0,
                threshold=0.5,
            )
        ]
        paths = {"default": ["agent1"]}

        node = DecisionNode(metadata, "fail_test", criteria, paths)

        context = Mock(spec=NodeExecutionContext)
        context.cognitive_classification = {"speed": "fast"}

        assert node.can_handle(context) is False

    def test_can_handle_with_complex_criteria(self) -> None:
        """Test can_handle with criteria that access nested properties."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"

        criteria = [
            DecisionCriteria(
                name="complex_check",
                evaluator=lambda ctx: ctx.task_classification.complexity_score,
                weight=1.0,
                threshold=0.5,
            )
        ]
        paths = {"default": ["agent1"]}

        node = DecisionNode(metadata, "complex_test", criteria, paths)

        # Valid context with nested property
        context = Mock(spec=NodeExecutionContext)
        context.cognitive_classification = {"speed": "fast"}
        context.task_classification = Mock()
        context.task_classification.complexity_score = 0.8

        assert node.can_handle(context) is True

        # Invalid context - missing nested property
        context2 = Mock(spec=NodeExecutionContext)
        context2.cognitive_classification = {"speed": "fast"}
        context2.task_classification = None

        assert node.can_handle(context2) is False


class TestDecisionNodeEvaluateCriteria:
    """Test DecisionNode _evaluate_criteria method."""

    @pytest.fixture
    def decision_node(self) -> Any:
        """Create a DecisionNode for testing."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"

        criteria = [
            DecisionCriteria(
                name="confidence",
                evaluator=lambda ctx: ctx.confidence_score,
                weight=2.0,
                threshold=0.8,
            ),
            DecisionCriteria(
                name="complexity",
                evaluator=lambda ctx: 0.7,  # Fixed value
                weight=1.0,
                threshold=0.6,
            ),
        ]
        paths = {
            "high_performance": ["agent1"],
            "balanced": ["agent2", "agent3"],
            "comprehensive": ["agent4", "agent5", "agent6"],
        }

        return DecisionNode(metadata, "eval_test", criteria, paths)

    @pytest.fixture
    def mock_context(self) -> Any:
        """Create mock context for testing."""
        context = Mock(spec=NodeExecutionContext)
        context.confidence_score = 0.9
        context.cognitive_classification = {"speed": "fast"}
        return context

    @pytest.mark.asyncio
    async def test_evaluate_criteria_high_scoring(
        self, decision_node: Mock, mock_context: Mock
    ) -> None:
        """Test _evaluate_criteria with high-scoring criteria."""
        result = await decision_node._evaluate_criteria(mock_context)

        # Check basic structure
        assert "selected_path" in result
        assert "selected_agents" in result
        assert "confidence" in result
        assert "alternatives" in result
        assert "reasoning" in result

        # Check reasoning structure
        reasoning = result["reasoning"]
        assert "criterion_scores" in reasoning
        assert "path_scores" in reasoning
        assert "decision_basis" in reasoning

        # Check criterion scores
        criterion_scores = reasoning["criterion_scores"]
        assert "confidence" in criterion_scores
        assert "complexity" in criterion_scores

        # Confidence criterion should pass (0.9 >= 0.8)
        confidence_score = criterion_scores["confidence"]
        assert confidence_score["score"] == 0.9
        assert confidence_score["weight"] == 2.0
        assert confidence_score["threshold"] == 0.8
        assert confidence_score["passed"] is True

        # Complexity criterion should pass (0.7 >= 0.6)
        complexity_score = criterion_scores["complexity"]
        assert complexity_score["score"] == 0.7
        assert complexity_score["weight"] == 1.0
        assert complexity_score["threshold"] == 0.6
        assert complexity_score["passed"] is True

    @pytest.mark.asyncio
    async def test_evaluate_criteria_path_selection(
        self, decision_node: Mock, mock_context: Mock
    ) -> None:
        """Test that _evaluate_criteria selects appropriate path."""
        result = await decision_node._evaluate_criteria(mock_context)

        # Should select one of the available paths
        assert result["selected_path"] in [
            "high_performance",
            "balanced",
            "comprehensive",
        ]

        # Selected agents should match the path
        selected_agents = result["selected_agents"]
        path_name = result["selected_path"]
        expected_agents = decision_node.paths[path_name]
        assert selected_agents == expected_agents

        # Alternatives should be the other paths
        alternatives = result["alternatives"]
        expected_alternatives = [p for p in decision_node.paths if p != path_name]
        assert set(alternatives) == set(expected_alternatives)

    @pytest.mark.asyncio
    async def test_evaluate_criteria_confidence_calculation(
        self, decision_node: Mock, mock_context: Mock
    ) -> None:
        """Test confidence score calculation."""
        result = await decision_node._evaluate_criteria(mock_context)

        # With our criteria:
        # - confidence: score=0.9, weight=2.0, passed=True -> contributes 1.8
        # - complexity: score=0.7, weight=1.0, passed=True -> contributes 0.7
        # Total: 2.5, max possible: 3.0, confidence = 2.5/3.0 = 0.833...

        confidence = result["confidence"]
        expected_confidence = (0.9 * 2.0 + 0.7 * 1.0) / (2.0 + 1.0)
        assert abs(confidence - expected_confidence) < 0.01

    @pytest.mark.asyncio
    async def test_evaluate_criteria_low_confidence(self, decision_node: Mock) -> None:
        """Test _evaluate_criteria with low confidence score."""
        context = Mock(spec=NodeExecutionContext)
        context.confidence_score = 0.3  # Below threshold
        context.cognitive_classification = {"speed": "fast"}

        result = await decision_node._evaluate_criteria(context)

        # Confidence criterion should fail (0.3 < 0.8)
        criterion_scores = result["reasoning"]["criterion_scores"]
        assert criterion_scores["confidence"]["passed"] is False
        assert criterion_scores["complexity"]["passed"] is True  # Still passes

        # Should still select a path (based on complexity alone)
        assert result["selected_path"] in decision_node.paths

    @pytest.mark.asyncio
    async def test_evaluate_criteria_all_fail(self, decision_node: Mock) -> None:
        """Test _evaluate_criteria when all criteria fail."""
        context = Mock(spec=NodeExecutionContext)
        context.confidence_score = 0.1  # Below threshold
        context.cognitive_classification = {"speed": "fast"}

        # Override complexity evaluator to fail
        decision_node.decision_criteria[1].evaluator = (
            lambda ctx: 0.1
        )  # Below 0.6 threshold

        result = await decision_node._evaluate_criteria(context)

        # Both criteria should fail
        criterion_scores = result["reasoning"]["criterion_scores"]
        assert criterion_scores["confidence"]["passed"] is False
        assert criterion_scores["complexity"]["passed"] is False

        # Should still select a path (likely first one as default)
        assert result["selected_path"] in decision_node.paths

        # Confidence should be 0 (no criteria passed)
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_criteria_confidence_capped(
        self, decision_node: Mock
    ) -> None:
        """Test that confidence is capped at 1.0."""
        context = Mock(spec=NodeExecutionContext)
        context.confidence_score = 2.0  # Artificially high
        context.cognitive_classification = {"speed": "fast"}

        result = await decision_node._evaluate_criteria(context)

        # Confidence should be capped at 1.0
        assert result["confidence"] <= 1.0
