"""
Integration Tests for Node Fallback Mechanisms.

This module tests error recovery and graceful degradation under failure
conditions to ensure advanced nodes handle failures appropriately.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from cognivault.orchestration.nodes.decision_node import (
    DecisionNode,
    DecisionCriteria,
)
from cognivault.orchestration.nodes.aggregator_node import (
    AggregatorNode,
    AggregationStrategy,
)
from cognivault.orchestration.nodes.validator_node import (
    ValidatorNode,
    ValidationCriteria,
)
from cognivault.orchestration.nodes.terminator_node import (
    TerminatorNode,
    TerminationCriteria,
)
from cognivault.orchestration.nodes.base_advanced_node import NodeExecutionContext
from cognivault.agents.metadata import AgentMetadata, TaskClassification


class TestNodeFailureRecovery:
    """Test node failure recovery mechanisms."""

    @pytest.fixture
    def mock_metadata(self):
        """Create generic mock metadata."""
        metadata = Mock(spec=AgentMetadata)
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "deep"
        metadata.bounded_context = "transformation"
        return metadata

    @pytest.fixture
    def base_context(self):
        """Create a base execution context."""
        context = NodeExecutionContext(
            workflow_id="fallback-test",
            correlation_id="fallback-correlation",
            cognitive_classification={"speed": "adaptive", "depth": "deep"},
            task_classification=TaskClassification(task_type="evaluate"),
        )
        return context

    @pytest.mark.asyncio
    async def test_decision_node_criteria_evaluation_failure(
        self, mock_metadata, base_context
    ):
        """Test decision node handling of criteria evaluation failures."""
        mock_metadata.execution_pattern = "decision"

        # Create criteria where one will fail with an exception
        def failing_evaluator(ctx):
            raise ValueError("Criteria evaluation failed")

        def working_evaluator(ctx):
            return 0.8

        criteria = [
            DecisionCriteria(
                name="failing_criterion",
                evaluator=failing_evaluator,
                weight=1.0,
                threshold=0.5,
            ),
            DecisionCriteria(
                name="working_criterion",
                evaluator=working_evaluator,
                weight=1.0,
                threshold=0.5,
            ),
        ]

        paths = {
            "success_path": ["agent1", "agent2"],
            "fallback_path": ["fallback_agent"],
        }

        decision_node = DecisionNode(
            mock_metadata,
            "resilient_decision",
            criteria,
            paths,
        )

        base_context.available_inputs = {
            "test_input": {
                "content": "Test decision input",
                "confidence": 0.8,
                "quality_score": 0.75,
            }
        }

        with patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"):
            # Should not raise exception, should handle failure gracefully
            result = await decision_node.execute(base_context)

            # Should still make a decision despite one failing criterion
            assert "selected_path" in result
            assert result["selected_path"] in ["success_path", "fallback_path"]

            # Check that reasoning includes error information
            reasoning = result["reasoning"]
            criterion_scores = reasoning["criterion_scores"]

            # Working criterion should have succeeded
            assert criterion_scores["working_criterion"]["score"] == 0.8
            assert criterion_scores["working_criterion"]["passed"] is True

            # Failing criterion should be marked as failed
            assert criterion_scores["failing_criterion"]["passed"] is False

    @pytest.mark.asyncio
    async def test_aggregator_partial_input_failure(self, mock_metadata, base_context):
        """Test aggregator handling when some inputs are corrupted."""
        mock_metadata.execution_pattern = "aggregator"

        aggregator_node = AggregatorNode(
            mock_metadata,
            "fault_tolerant_aggregator",
            AggregationStrategy.CONSENSUS,
            min_inputs=1,  # Allow operation with reduced inputs
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        # Mix of valid and invalid inputs
        base_context.available_inputs = {
            "valid_input1": {
                "content": "Valid content from reliable source",
                "confidence": 0.9,
                "quality_score": 0.85,
                "metadata": {"source": "reliable"},
            },
            "corrupted_input": {
                # Missing required fields - should be filtered out
                "confidence": "invalid_confidence",  # Wrong type
                "quality_score": None,  # Null value
            },
            "valid_input2": {
                "content": "Another valid input",
                "confidence": 0.8,
                "quality_score": 0.8,
                "metadata": {"source": "backup"},
            },
        }

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await aggregator_node.execute(base_context)

            # Should successfully aggregate valid inputs despite corrupted one
            assert result["aggregation_strategy"] == "consensus"
            assert result["input_count"] >= 1  # At least one valid input processed
            assert result["quality_score"] > 0

            # Should only include valid input sources
            valid_sources = [
                src for src in result["input_sources"] if src.startswith("valid_")
            ]
            assert len(valid_sources) >= 1

    @pytest.mark.asyncio
    async def test_validator_criteria_exception_handling(
        self, mock_metadata, base_context
    ):
        """Test validator handling of criteria that raise exceptions."""
        mock_metadata.execution_pattern = "validator"

        # Create criteria where some will raise exceptions
        def exception_validator(data):
            raise RuntimeError("Validation system error")

        def working_validator(data):
            return len(data.get("content", "")) > 5

        criteria = [
            ValidationCriteria(
                name="exception_criterion",
                validator=exception_validator,
                weight=1.0,
                required=False,  # Not required, so failure shouldn't block
                error_message="System error in validation",
            ),
            ValidationCriteria(
                name="working_criterion",
                validator=working_validator,
                weight=2.0,
                required=True,
                error_message="Content too short",
            ),
        ]

        validator_node = ValidatorNode(
            mock_metadata,
            "fault_tolerant_validator",
            criteria,
            quality_threshold=0.5,  # Lower threshold to accommodate failures
            allow_warnings=True,
        )

        base_context.available_inputs = {
            "test_data": {
                "content": "Sufficient content for validation testing",
                "confidence": 0.8,
                "quality_score": 0.75,
            }
        }

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ):
            result = await validator_node.execute(base_context)

            # Should pass despite exception in non-required criterion
            assert result["validation_result"] in ["pass", "warning"]
            assert result["passed"] is True

            # Should record the exception in results
            criteria_results = result["criteria_results"]
            assert criteria_results["exception_criterion"]["passed"] is False
            assert "error_message" in criteria_results["exception_criterion"]

            # Working criterion should succeed
            assert criteria_results["working_criterion"]["passed"] is True

            # Should have warnings about the failed criterion
            assert len(result["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_terminator_resource_exhaustion_handling(
        self, mock_metadata, base_context
    ):
        """Test terminator behavior under resource exhaustion conditions."""
        mock_metadata.execution_pattern = "terminator"

        # Create criteria that simulate resource checks
        def resource_check_evaluator(data):
            # Simulate checking available resources
            resource_usage = data.get("resource_usage", {})
            cpu_usage = resource_usage.get("cpu", 0.0)
            memory_usage = resource_usage.get("memory", 0.0)
            return max(cpu_usage, memory_usage) > 0.9  # Terminate if either > 90%

        criteria = [
            TerminationCriteria(
                name="resource_exhaustion",
                evaluator=resource_check_evaluator,
                threshold=0.9,
                weight=1.0,
                required=True,
                description="Check for resource exhaustion",
            ),
        ]

        terminator_node = TerminatorNode(
            mock_metadata,
            "resource_monitor",
            criteria,
            confidence_threshold=0.9,
            resource_limit_threshold=0.85,
        )

        # Set up context with high resource usage
        base_context.execution_path = ["decision", "aggregator", "validator"]
        base_context.resource_usage = {"cpu": 0.95, "memory": 0.88}  # High usage

        base_context.available_inputs = {
            "resource_data": {
                "content": "System under high load",
                "confidence": 0.7,
                "quality_score": 0.6,
                "resource_usage": {"cpu": 0.95, "memory": 0.88},
            }
        }

        with patch(
            "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
        ):
            result = await terminator_node.execute(base_context)

            # Should recommend termination due to resource exhaustion
            assert result["should_terminate"] is True
            assert result["termination_reason"] == "completion_criteria"

            # Should provide resource savings information
            assert "resource_savings" in result
            savings = result["resource_savings"]
            assert "cpu_time_ms" in savings
            assert "memory_mb" in savings


class TestGracefulDegradation:
    """Test graceful degradation patterns."""

    @pytest.fixture
    def mock_metadata(self):
        """Create generic mock metadata."""
        metadata = Mock(spec=AgentMetadata)
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "deep"
        metadata.bounded_context = "transformation"
        return metadata

    @pytest.mark.asyncio
    async def test_aggregator_degraded_quality_mode(self, mock_metadata):
        """Test aggregator operating in degraded quality mode."""
        mock_metadata.execution_pattern = "aggregator"

        aggregator_node = AggregatorNode(
            mock_metadata,
            "degraded_aggregator",
            AggregationStrategy.BEST_QUALITY,
            min_inputs=3,  # Prefer 3 inputs
            max_inputs=5,
            quality_threshold=0.8,  # High quality preferred
            confidence_threshold=0.8,
        )

        context = NodeExecutionContext(
            workflow_id="degradation-test",
            correlation_id="degradation-correlation",
            cognitive_classification={"speed": "fast"},
            task_classification=TaskClassification(task_type="transform"),
        )

        # Provide only low-quality inputs (below threshold)
        context.available_inputs = {
            "low_quality1": {
                "content": "Low quality input 1",
                "confidence": 0.6,  # Below threshold
                "quality_score": 0.65,  # Below threshold
            },
            "low_quality2": {
                "content": "Low quality input 2",
                "confidence": 0.55,  # Below threshold
                "quality_score": 0.6,  # Below threshold
            },
        }

        # Should fail with high thresholds
        with pytest.raises(ValueError, match="Insufficient inputs for aggregation"):
            await aggregator_node.execute(context)

        # Test with degraded mode (lower thresholds)
        degraded_aggregator = AggregatorNode(
            mock_metadata,
            "degraded_mode_aggregator",
            AggregationStrategy.BEST_QUALITY,
            min_inputs=2,
            quality_threshold=0.5,  # Lowered threshold
            confidence_threshold=0.5,  # Lowered threshold
        )

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await degraded_aggregator.execute(context)

            # Should succeed in degraded mode
            assert result["aggregation_strategy"] == "best_quality"
            assert result["input_count"] == 2
            assert result["quality_score"] > 0.5

    @pytest.mark.asyncio
    async def test_validator_relaxed_criteria_fallback(self, mock_metadata):
        """Test validator falling back to relaxed criteria."""
        mock_metadata.execution_pattern = "validator"

        # Create strict criteria that will fail
        strict_criteria = [
            ValidationCriteria(
                name="strict_content_length",
                validator=lambda data: len(data.get("content", "")) > 200,
                weight=2.0,
                required=True,
                error_message="Content must be very comprehensive",
            ),
            ValidationCriteria(
                name="strict_quality",
                validator=lambda data: data.get("quality_score", 0) > 0.95,
                weight=1.0,
                required=True,
                error_message="Quality must be exceptional",
            ),
        ]

        strict_validator = ValidatorNode(
            mock_metadata,
            "strict_validator",
            strict_criteria,
            quality_threshold=0.9,
            strict_mode=True,
        )

        context = NodeExecutionContext(
            workflow_id="relaxed-test",
            correlation_id="relaxed-correlation",
            cognitive_classification={"speed": "slow"},
            task_classification=TaskClassification(task_type="transform"),
        )

        context.available_inputs = {
            "moderate_quality": {
                "content": "Moderate length content that meets basic requirements",  # < 200 chars
                "confidence": 0.8,
                "quality_score": 0.85,  # < 0.95
            }
        }

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ):
            result = await strict_validator.execute(context)

            # Should fail with strict criteria
            assert result["validation_result"] == "fail"
            assert result["passed"] is False

        # Test with relaxed criteria
        relaxed_criteria = [
            ValidationCriteria(
                name="relaxed_content_length",
                validator=lambda data: len(data.get("content", ""))
                > 20,  # Much lower requirement
                weight=1.0,
                required=True,
                error_message="Content must meet minimum length",
            ),
            ValidationCriteria(
                name="relaxed_quality",
                validator=lambda data: data.get("quality_score", 0)
                > 0.7,  # Lower requirement
                weight=1.0,
                required=True,
                error_message="Quality must be acceptable",
            ),
        ]

        relaxed_validator = ValidatorNode(
            mock_metadata,
            "relaxed_validator",
            relaxed_criteria,
            quality_threshold=0.7,  # Lower threshold
            strict_mode=False,
            allow_warnings=True,
        )

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ):
            relaxed_result = await relaxed_validator.execute(context)

            # Should pass with relaxed criteria
            assert relaxed_result["validation_result"] == "pass"
            assert relaxed_result["passed"] is True


class TestFallbackPatternSelection:
    """Test fallback pattern selection mechanisms."""

    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata for pattern testing."""
        metadata = Mock(spec=AgentMetadata)
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "deep"
        metadata.bounded_context = "transformation"
        return metadata

    def test_decision_node_fallback_patterns(self, mock_metadata):
        """Test decision node fallback pattern retrieval."""
        mock_metadata.execution_pattern = "decision"

        decision_node = DecisionNode(
            mock_metadata,
            "pattern_decision",
            [
                DecisionCriteria(
                    name="basic", evaluator=lambda ctx: 0.5, weight=1.0, threshold=0.3
                )
            ],
            {"default": ["agent1"]},
        )

        fallback_patterns = decision_node.get_fallback_patterns()
        assert fallback_patterns == ["processor", "terminator"]

    def test_aggregator_node_fallback_patterns(self, mock_metadata):
        """Test aggregator node fallback pattern retrieval."""
        mock_metadata.execution_pattern = "aggregator"

        aggregator_node = AggregatorNode(
            mock_metadata,
            "pattern_aggregator",
            AggregationStrategy.CONSENSUS,
        )

        fallback_patterns = aggregator_node.get_fallback_patterns()
        assert fallback_patterns == ["processor", "validator"]

    def test_validator_node_fallback_patterns(self, mock_metadata):
        """Test validator node fallback pattern retrieval."""
        mock_metadata.execution_pattern = "validator"

        validator_node = ValidatorNode(
            mock_metadata,
            "pattern_validator",
            [
                ValidationCriteria(
                    name="basic", validator=lambda data: True, weight=1.0, required=True
                )
            ],
        )

        fallback_patterns = validator_node.get_fallback_patterns()
        assert fallback_patterns == ["processor", "terminator"]

    def test_terminator_node_fallback_patterns(self, mock_metadata):
        """Test terminator node fallback pattern retrieval."""
        mock_metadata.execution_pattern = "terminator"

        terminator_node = TerminatorNode(
            mock_metadata,
            "pattern_terminator",
            [
                TerminationCriteria(
                    name="basic",
                    evaluator=lambda data: True,
                    threshold=0.5,
                    weight=1.0,
                    required=True,
                )
            ],
        )

        fallback_patterns = terminator_node.get_fallback_patterns()
        assert fallback_patterns == []  # Terminator has no fallbacks


class TestCircuitBreakerPattern:
    """Test circuit breaker patterns for node resilience."""

    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata for circuit breaker testing."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        metadata.cognitive_speed = "fast"
        metadata.cognitive_depth = "shallow"
        metadata.bounded_context = "transformation"
        return metadata

    @pytest.mark.asyncio
    async def test_aggregator_circuit_breaker_simulation(self, mock_metadata):
        """Simulate circuit breaker behavior for aggregator under repeated failures."""
        aggregator_node = AggregatorNode(
            mock_metadata,
            "circuit_breaker_aggregator",
            AggregationStrategy.CONSENSUS,
            min_inputs=2,
            quality_threshold=0.8,  # High threshold to trigger failures
            confidence_threshold=0.8,
        )

        context = NodeExecutionContext(
            workflow_id="circuit-breaker-test",
            correlation_id="circuit-correlation",
            cognitive_classification={"speed": "fast"},
            task_classification=TaskClassification(task_type="transform"),
        )

        # Create inputs that will consistently fail threshold checks
        context.available_inputs = {
            "failing_input1": {
                "content": "Low quality content 1",
                "confidence": 0.5,  # Below threshold
                "quality_score": 0.6,  # Below threshold
            },
            "failing_input2": {
                "content": "Low quality content 2",
                "confidence": 0.45,  # Below threshold
                "quality_score": 0.55,  # Below threshold
            },
        }

        # Simulate multiple failure attempts
        failure_count = 0
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                await aggregator_node.execute(context)
                break  # Success, exit loop
            except ValueError as e:
                failure_count += 1
                if "Insufficient inputs" in str(e):
                    # Expected failure due to threshold filtering
                    continue
                else:
                    raise  # Unexpected error

        # Verify circuit breaker would trigger after repeated failures
        assert failure_count == max_attempts

        # Test recovery with better inputs
        context.available_inputs = {
            "recovery_input1": {
                "content": "High quality recovery content 1",
                "confidence": 0.9,  # Above threshold
                "quality_score": 0.85,  # Above threshold
            },
            "recovery_input2": {
                "content": "High quality recovery content 2",
                "confidence": 0.88,  # Above threshold
                "quality_score": 0.9,  # Above threshold
            },
        }

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            # Should recover successfully
            recovery_result = await aggregator_node.execute(context)
            assert recovery_result["aggregation_strategy"] == "consensus"
            assert recovery_result["input_count"] == 2
            assert recovery_result["quality_score"] > 0.8


class TestErrorPropagationControl:
    """Test controlled error propagation through node chains."""

    @pytest.mark.asyncio
    async def test_error_containment_in_node_chain(self):
        """Test that errors in one node don't cascade uncontrollably."""
        # Create metadata for different nodes
        decision_metadata = Mock(spec=AgentMetadata)
        decision_metadata.execution_pattern = "decision"

        aggregator_metadata = Mock(spec=AgentMetadata)
        aggregator_metadata.execution_pattern = "aggregator"

        # Create nodes with potential failure points
        decision_node = DecisionNode(
            decision_metadata,
            "error_prone_decision",
            [
                DecisionCriteria(
                    name="failing",
                    evaluator=lambda ctx: 1 / 0,
                    weight=1.0,
                    threshold=0.5,
                )
            ],  # Division by zero
            {"error_path": ["error_agent"], "safe_path": ["safe_agent"]},
        )

        aggregator_node = AggregatorNode(
            aggregator_metadata,
            "error_resistant_aggregator",
            AggregationStrategy.BEST_QUALITY,
            min_inputs=1,
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        context = NodeExecutionContext(
            workflow_id="error-containment-test",
            correlation_id="error-correlation",
            cognitive_classification={"speed": "adaptive"},
            task_classification=TaskClassification(task_type="transform"),
        )

        context.available_inputs = {
            "test_input": {
                "content": "Input for error containment testing",
                "confidence": 0.8,
                "quality_score": 0.75,
            }
        }

        # Decision node should handle the division by zero gracefully
        with patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"):
            decision_result = await decision_node.execute(context)

            # Should make a decision despite the failing criterion
            assert "selected_path" in decision_result
            assert decision_result["selected_path"] in ["error_path", "safe_path"]

        # Aggregator should work normally even if fed error-related data
        agg_context = NodeExecutionContext(
            workflow_id=context.workflow_id,
            correlation_id=context.correlation_id,
            cognitive_classification=context.cognitive_classification,
            task_classification=context.task_classification,
        )

        agg_context.available_inputs = {
            "error_recovery_input": {
                "content": "Recovered content after error handling",
                "confidence": 0.7,
                "quality_score": 0.8,
                "metadata": {"recovered_from_error": True},
            }
        }

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            agg_result = await aggregator_node.execute(agg_context)

            # Should aggregate successfully despite upstream errors
            assert agg_result["aggregation_strategy"] == "best_quality"
            assert agg_result["quality_score"] > 0.7
