"""Tests for advanced node event emission functions."""

import pytest
from typing import Any
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

from cognivault.events.emitter import (
    emit_decision_made,
    emit_aggregation_completed,
    emit_validation_completed,
    emit_termination_triggered,
)
from cognivault.events.types import EventType


class TestEmitDecisionMade:
    """Test emit_decision_made convenience function."""

    @pytest.mark.asyncio
    async def test_emit_decision_made_basic(self) -> None:
        """Test basic decision made event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_decision_made(
                workflow_id="workflow-123",
                decision_criteria=["confidence", "task_complexity"],
                selected_path="path_a",
                confidence_score=0.85,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.DECISION_MADE
            assert emitted_event.workflow_id == "workflow-123"
            assert emitted_event.data["decision_criteria"] == [
                "confidence",
                "task_complexity",
            ]
            assert emitted_event.data["selected_path"] == "path_a"
            assert emitted_event.data["confidence_score"] == 0.85
            assert emitted_event.data["alternative_paths"] == []
            assert emitted_event.data["reasoning"] == {}
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "decision"
            assert emitted_event.metadata["execution_pattern"] == "decision"

    @pytest.mark.asyncio
    async def test_emit_decision_made_with_all_params(self) -> None:
        """Test decision made event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            alternative_paths = ["path_b", "path_c", "path_d"]
            reasoning = {
                "primary_factor": "high_confidence",
                "secondary_factor": "resource_availability",
                "constraints": ["time_limit", "memory_usage"],
            }
            metadata = {"environment": "production", "version": "2.0"}

            await emit_decision_made(
                workflow_id="workflow-456",
                decision_criteria=["confidence", "resource_usage", "time_constraints"],
                selected_path="path_optimal",
                confidence_score=0.92,
                alternative_paths=alternative_paths,
                reasoning=reasoning,
                correlation_id="test-correlation-789",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.DECISION_MADE
            assert emitted_event.workflow_id == "workflow-456"
            assert emitted_event.data["decision_criteria"] == [
                "confidence",
                "resource_usage",
                "time_constraints",
            ]
            assert emitted_event.data["selected_path"] == "path_optimal"
            assert emitted_event.data["confidence_score"] == 0.92
            assert emitted_event.data["alternative_paths"] == alternative_paths
            assert emitted_event.data["reasoning"] == reasoning
            assert emitted_event.correlation_id == "test-correlation-789"
            assert emitted_event.metadata["environment"] == "production"
            assert emitted_event.metadata["version"] == "2.0"
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "decision"
            assert emitted_event.metadata["execution_pattern"] == "decision"

    @pytest.mark.asyncio
    async def test_emit_decision_made_with_correlation_fallback(self) -> None:
        """Test decision made event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-correlation-123"

            await emit_decision_made(
                workflow_id="workflow-789",
                decision_criteria=["simple"],
                selected_path="fast_path",
                confidence_score=0.95,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-correlation-123"

    @pytest.mark.asyncio
    async def test_emit_decision_made_none_handling(self) -> None:
        """Test decision made event emission with None values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_decision_made(
                workflow_id="workflow-abc",
                decision_criteria=["default"],
                selected_path="standard",
                confidence_score=0.75,
                alternative_paths=None,
                reasoning=None,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify None values default to empty lists/dicts
            assert emitted_event.data["alternative_paths"] == []
            assert emitted_event.data["reasoning"] == {}

    @pytest.mark.asyncio
    async def test_emit_decision_made_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "service": "decision-engine",
                "version": "3.1.0",
                "event_category": "custom_node",  # This should override default
            }

            await emit_decision_made(
                workflow_id="workflow-def",
                decision_criteria=["test"],
                selected_path="custom_path",
                confidence_score=0.80,
                metadata=custom_metadata,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["service"] == "decision-engine"
            assert emitted_event.metadata["version"] == "3.1.0"
            assert (
                emitted_event.metadata["event_category"] == "custom_node"
            )  # Overridden
            assert (
                emitted_event.metadata["node_type"] == "decision"
            )  # Default preserved
            assert (
                emitted_event.metadata["execution_pattern"] == "decision"
            )  # Default preserved

    @pytest.mark.asyncio
    async def test_emit_decision_made_timestamp_generation(self) -> None:
        """Test that decision made events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_decision_made(
                workflow_id="workflow-time",
                decision_criteria=["time_test"],
                selected_path="timed_path",
                confidence_score=0.90,
            )
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_emit_decision_made_confidence_score_values(self) -> None:
        """Test decision made event with various confidence score values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test with boundary values
            test_scores = [0.0, 0.5, 1.0, 0.999, 0.001]

            for score in test_scores:
                await emit_decision_made(
                    workflow_id=f"workflow-score-{score}",
                    decision_criteria=["test"],
                    selected_path="test_path",
                    confidence_score=score,
                )

            # Verify all scores were emitted correctly
            assert mock_emitter.emit.call_count == len(test_scores)

            for i, score in enumerate(test_scores):
                emitted_event = mock_emitter.emit.call_args_list[i][0][0]
                assert emitted_event.data["confidence_score"] == score


class TestEmitAggregationCompleted:
    """Test emit_aggregation_completed convenience function."""

    @pytest.mark.asyncio
    async def test_emit_aggregation_completed_basic(self) -> None:
        """Test basic aggregation completed event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_aggregation_completed(
                workflow_id="workflow-agg-123",
                aggregation_strategy="consensus",
                input_sources=["agent1", "agent2", "agent3"],
                output_quality_score=0.88,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.AGGREGATION_COMPLETED
            assert emitted_event.workflow_id == "workflow-agg-123"
            assert emitted_event.data["aggregation_strategy"] == "consensus"
            assert emitted_event.data["input_sources"] == ["agent1", "agent2", "agent3"]
            assert emitted_event.data["output_quality_score"] == 0.88
            assert emitted_event.data["conflicts_resolved"] == 0
            assert emitted_event.data["aggregation_time_ms"] is None
            assert emitted_event.execution_time_ms is None
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "aggregator"
            assert emitted_event.metadata["execution_pattern"] == "aggregator"

    @pytest.mark.asyncio
    async def test_emit_aggregation_completed_with_all_params(self) -> None:
        """Test aggregation completed event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            input_sources = ["refiner", "historian", "critic", "synthesis"]
            metadata = {"environment": "production", "optimization": "enabled"}

            await emit_aggregation_completed(
                workflow_id="workflow-agg-456",
                aggregation_strategy="weighted",
                input_sources=input_sources,
                output_quality_score=0.95,
                conflicts_resolved=3,
                aggregation_time_ms=150.75,
                correlation_id="agg-correlation-789",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.AGGREGATION_COMPLETED
            assert emitted_event.workflow_id == "workflow-agg-456"
            assert emitted_event.data["aggregation_strategy"] == "weighted"
            assert emitted_event.data["input_sources"] == input_sources
            assert emitted_event.data["output_quality_score"] == 0.95
            assert emitted_event.data["conflicts_resolved"] == 3
            assert emitted_event.data["aggregation_time_ms"] == 150.75
            assert emitted_event.execution_time_ms == 150.75
            assert emitted_event.correlation_id == "agg-correlation-789"
            assert emitted_event.metadata["environment"] == "production"
            assert emitted_event.metadata["optimization"] == "enabled"
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "aggregator"
            assert emitted_event.metadata["execution_pattern"] == "aggregator"

    @pytest.mark.asyncio
    async def test_emit_aggregation_completed_with_correlation_fallback(self) -> None:
        """Test aggregation completed event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-agg-correlation-123"

            await emit_aggregation_completed(
                workflow_id="workflow-agg-789",
                aggregation_strategy="hierarchical",
                input_sources=["source1", "source2"],
                output_quality_score=0.82,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-agg-correlation-123"

    @pytest.mark.asyncio
    async def test_emit_aggregation_completed_strategy_types(self) -> None:
        """Test aggregation completed event with various strategy types."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test different strategy types
            strategies = [
                "consensus",
                "weighted",
                "hierarchical",
                "majority_vote",
                "custom",
            ]

            for strategy in strategies:
                await emit_aggregation_completed(
                    workflow_id=f"workflow-strategy-{strategy}",
                    aggregation_strategy=strategy,
                    input_sources=["a", "b"],
                    output_quality_score=0.85,
                )

            # Verify all strategies were emitted correctly
            assert mock_emitter.emit.call_count == len(strategies)

            for i, strategy in enumerate(strategies):
                emitted_event = mock_emitter.emit.call_args_list[i][0][0]
                assert emitted_event.data["aggregation_strategy"] == strategy

    @pytest.mark.asyncio
    async def test_emit_aggregation_completed_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "service": "aggregator-engine",
                "version": "4.2.0",
                "event_category": "custom_aggregation",  # This should override default
            }

            await emit_aggregation_completed(
                workflow_id="workflow-meta",
                aggregation_strategy="consensus",
                input_sources=["x", "y", "z"],
                output_quality_score=0.91,
                metadata=custom_metadata,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["service"] == "aggregator-engine"
            assert emitted_event.metadata["version"] == "4.2.0"
            assert (
                emitted_event.metadata["event_category"] == "custom_aggregation"
            )  # Overridden
            assert (
                emitted_event.metadata["node_type"] == "aggregator"
            )  # Default preserved
            assert (
                emitted_event.metadata["execution_pattern"] == "aggregator"
            )  # Default preserved

    @pytest.mark.asyncio
    async def test_emit_aggregation_completed_timestamp_generation(self) -> None:
        """Test that aggregation completed events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_aggregation_completed(
                workflow_id="workflow-time",
                aggregation_strategy="weighted",
                input_sources=["t1", "t2"],
                output_quality_score=0.87,
            )
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_emit_aggregation_completed_conflict_resolution_tracking(
        self,
    ) -> None:
        """Test aggregation completed event tracks conflict resolution properly."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test with various conflict counts
            conflict_counts = [0, 1, 5, 10, 100]

            for count in conflict_counts:
                await emit_aggregation_completed(
                    workflow_id=f"workflow-conflicts-{count}",
                    aggregation_strategy="consensus",
                    input_sources=["s1", "s2", "s3"],
                    output_quality_score=0.80,
                    conflicts_resolved=count,
                )

            # Verify all conflict counts were tracked correctly
            assert mock_emitter.emit.call_count == len(conflict_counts)

            for i, count in enumerate(conflict_counts):
                emitted_event = mock_emitter.emit.call_args_list[i][0][0]
                assert emitted_event.data["conflicts_resolved"] == count


class TestEmitValidationCompleted:
    """Test emit_validation_completed convenience function."""

    @pytest.mark.asyncio
    async def test_emit_validation_completed_basic(self) -> None:
        """Test basic validation completed event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_validation_completed(
                workflow_id="workflow-val-123",
                validation_result="pass",
                quality_score=0.92,
                validation_criteria=["content_quality", "format_compliance"],
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.VALIDATION_COMPLETED
            assert emitted_event.workflow_id == "workflow-val-123"
            assert emitted_event.data["validation_result"] == "pass"
            assert emitted_event.data["quality_score"] == 0.92
            assert emitted_event.data["validation_criteria"] == [
                "content_quality",
                "format_compliance",
            ]
            assert emitted_event.data["recommendations"] == []
            assert emitted_event.data["validation_time_ms"] is None
            assert emitted_event.execution_time_ms is None
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "validator"
            assert emitted_event.metadata["execution_pattern"] == "validator"

    @pytest.mark.asyncio
    async def test_emit_validation_completed_with_all_params(self) -> None:
        """Test validation completed event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            validation_criteria = [
                "content_quality",
                "format_compliance",
                "safety_checks",
                "data_integrity",
            ]
            recommendations = [
                "Improve error handling in section 3",
                "Add more test coverage for edge cases",
                "Consider performance optimization in data processing",
            ]
            metadata = {"environment": "staging", "strict_mode": True}

            await emit_validation_completed(
                workflow_id="workflow-val-456",
                validation_result="warning",
                quality_score=0.78,
                validation_criteria=validation_criteria,
                recommendations=recommendations,
                validation_time_ms=85.25,
                correlation_id="val-correlation-789",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.VALIDATION_COMPLETED
            assert emitted_event.workflow_id == "workflow-val-456"
            assert emitted_event.data["validation_result"] == "warning"
            assert emitted_event.data["quality_score"] == 0.78
            assert emitted_event.data["validation_criteria"] == validation_criteria
            assert emitted_event.data["recommendations"] == recommendations
            assert emitted_event.data["validation_time_ms"] == 85.25
            assert emitted_event.execution_time_ms == 85.25
            assert emitted_event.correlation_id == "val-correlation-789"
            assert emitted_event.metadata["environment"] == "staging"
            assert emitted_event.metadata["strict_mode"] is True
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "validator"
            assert emitted_event.metadata["execution_pattern"] == "validator"

    @pytest.mark.asyncio
    async def test_emit_validation_completed_with_correlation_fallback(self) -> None:
        """Test validation completed event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-val-correlation-123"

            await emit_validation_completed(
                workflow_id="workflow-val-789",
                validation_result="pass",
                quality_score=0.95,
                validation_criteria=["basic_check"],
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-val-correlation-123"

    @pytest.mark.asyncio
    async def test_emit_validation_completed_result_types(self) -> None:
        """Test validation completed event with various result types."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test different validation results
            results = ["pass", "fail", "warning"]

            for result in results:
                await emit_validation_completed(
                    workflow_id=f"workflow-result-{result}",
                    validation_result=result,
                    quality_score=0.85,
                    validation_criteria=["test_criteria"],
                )

            # Verify all results were emitted correctly
            assert mock_emitter.emit.call_count == len(results)

            for i, result in enumerate(results):
                emitted_event = mock_emitter.emit.call_args_list[i][0][0]
                assert emitted_event.data["validation_result"] == result

    @pytest.mark.asyncio
    async def test_emit_validation_completed_none_handling(self) -> None:
        """Test validation completed event emission with None values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_validation_completed(
                workflow_id="workflow-val-none",
                validation_result="pass",
                quality_score=0.90,
                validation_criteria=["standard"],
                recommendations=None,
                validation_time_ms=None,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify None values default to empty lists
            assert emitted_event.data["recommendations"] == []
            assert emitted_event.data["validation_time_ms"] is None
            assert emitted_event.execution_time_ms is None

    @pytest.mark.asyncio
    async def test_emit_validation_completed_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "service": "validator-engine",
                "version": "5.0.1",
                "event_category": "custom_validation",  # This should override default
            }

            await emit_validation_completed(
                workflow_id="workflow-meta",
                validation_result="pass",
                quality_score=0.88,
                validation_criteria=["meta_test"],
                metadata=custom_metadata,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["service"] == "validator-engine"
            assert emitted_event.metadata["version"] == "5.0.1"
            assert (
                emitted_event.metadata["event_category"] == "custom_validation"
            )  # Overridden
            assert (
                emitted_event.metadata["node_type"] == "validator"
            )  # Default preserved
            assert (
                emitted_event.metadata["execution_pattern"] == "validator"
            )  # Default preserved

    @pytest.mark.asyncio
    async def test_emit_validation_completed_timestamp_generation(self) -> None:
        """Test that validation completed events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_validation_completed(
                workflow_id="workflow-time",
                validation_result="pass",
                quality_score=0.93,
                validation_criteria=["time_test"],
            )
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_emit_validation_completed_quality_score_range(self) -> None:
        """Test validation completed event with various quality score values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test with boundary values and typical values
            test_scores = [0.0, 0.25, 0.5, 0.75, 1.0]

            for score in test_scores:
                await emit_validation_completed(
                    workflow_id=f"workflow-score-{score}",
                    validation_result="pass" if score >= 0.5 else "fail",
                    quality_score=score,
                    validation_criteria=["score_test"],
                )

            # Verify all scores were emitted correctly
            assert mock_emitter.emit.call_count == len(test_scores)

            for i, score in enumerate(test_scores):
                emitted_event = mock_emitter.emit.call_args_list[i][0][0]
                assert emitted_event.data["quality_score"] == score


class TestEmitTerminationTriggered:
    """Test emit_termination_triggered convenience function."""

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_basic(self) -> None:
        """Test basic termination triggered event emission."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_termination_triggered(
                workflow_id="workflow-term-123",
                termination_reason="confidence_threshold_met",
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.TERMINATION_TRIGGERED
            assert emitted_event.workflow_id == "workflow-term-123"
            assert (
                emitted_event.data["termination_reason"] == "confidence_threshold_met"
            )
            assert emitted_event.data["confidence_score"] is None
            assert emitted_event.data["threshold"] is None
            assert emitted_event.data["resources_saved"] == {}
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "terminator"
            assert emitted_event.metadata["execution_pattern"] == "terminator"

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_with_all_params(self) -> None:
        """Test termination triggered event emission with all parameters."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            resources_saved = {
                "time_ms": 1250,
                "compute_units": 45,
                "tokens": 3200,
                "api_calls": 5,
            }
            metadata = {"environment": "production", "optimization_enabled": True}

            await emit_termination_triggered(
                workflow_id="workflow-term-456",
                termination_reason="quality_gate_passed",
                confidence_score=0.96,
                threshold=0.95,
                resources_saved=resources_saved,
                correlation_id="term-correlation-789",
                metadata=metadata,
            )

            # Verify emit was called once
            mock_emitter.emit.assert_called_once()

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify event properties
            assert emitted_event.event_type == EventType.TERMINATION_TRIGGERED
            assert emitted_event.workflow_id == "workflow-term-456"
            assert emitted_event.data["termination_reason"] == "quality_gate_passed"
            assert emitted_event.data["confidence_score"] == 0.96
            assert emitted_event.data["threshold"] == 0.95
            assert emitted_event.data["resources_saved"] == resources_saved
            assert emitted_event.correlation_id == "term-correlation-789"
            assert emitted_event.metadata["environment"] == "production"
            assert emitted_event.metadata["optimization_enabled"] is True
            assert emitted_event.metadata["event_category"] == "node_execution"
            assert emitted_event.metadata["node_type"] == "terminator"
            assert emitted_event.metadata["execution_pattern"] == "terminator"

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_with_correlation_fallback(self) -> None:
        """Test termination triggered event emission with correlation ID fallback."""
        with (
            patch(
                "cognivault.events.emitter.get_global_event_emitter"
            ) as mock_get_emitter,
            patch(
                "cognivault.events.emitter.get_correlation_id"
            ) as mock_get_correlation,
        ):
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter
            mock_get_correlation.return_value = "fallback-term-correlation-123"

            await emit_termination_triggered(
                workflow_id="workflow-term-789",
                termination_reason="resource_limit",
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify correlation ID fallback was used
            assert emitted_event.correlation_id == "fallback-term-correlation-123"

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_reason_types(self) -> None:
        """Test termination triggered event with various reason types."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test different termination reasons
            reasons = [
                "confidence_threshold_met",
                "quality_gate_passed",
                "resource_limit",
                "time_limit_exceeded",
                "user_requested",
                "error_threshold_exceeded",
            ]

            for reason in reasons:
                await emit_termination_triggered(
                    workflow_id=f"workflow-reason-{reason}",
                    termination_reason=reason,
                )

            # Verify all reasons were emitted correctly
            assert mock_emitter.emit.call_count == len(reasons)

            for i, reason in enumerate(reasons):
                emitted_event = mock_emitter.emit.call_args_list[i][0][0]
                assert emitted_event.data["termination_reason"] == reason

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_none_handling(self) -> None:
        """Test termination triggered event emission with None values."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            await emit_termination_triggered(
                workflow_id="workflow-term-none",
                termination_reason="early_exit",
                confidence_score=None,
                threshold=None,
                resources_saved=None,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify None values are handled correctly
            assert emitted_event.data["confidence_score"] is None
            assert emitted_event.data["threshold"] is None
            assert emitted_event.data["resources_saved"] == {}

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_metadata_merging(self) -> None:
        """Test that custom metadata merges with default metadata."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            custom_metadata = {
                "service": "terminator-engine",
                "version": "6.0.0",
                "event_category": "custom_termination",  # This should override default
            }

            await emit_termination_triggered(
                workflow_id="workflow-meta",
                termination_reason="test_termination",
                metadata=custom_metadata,
            )

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify metadata merging (custom should override defaults)
            assert emitted_event.metadata["service"] == "terminator-engine"
            assert emitted_event.metadata["version"] == "6.0.0"
            assert (
                emitted_event.metadata["event_category"] == "custom_termination"
            )  # Overridden
            assert (
                emitted_event.metadata["node_type"] == "terminator"
            )  # Default preserved
            assert (
                emitted_event.metadata["execution_pattern"] == "terminator"
            )  # Default preserved

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_timestamp_generation(self) -> None:
        """Test that termination triggered events have proper timestamp generation."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Record time before and after emission
            before_time = datetime.now(timezone.utc)
            await emit_termination_triggered(
                workflow_id="workflow-time",
                termination_reason="time_test",
            )
            after_time = datetime.now(timezone.utc)

            # Get the emitted event
            emitted_event = mock_emitter.emit.call_args[0][0]

            # Verify timestamp is within expected range
            assert before_time <= emitted_event.timestamp <= after_time
            assert emitted_event.timestamp.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_emit_termination_triggered_threshold_comparison(self) -> None:
        """Test termination triggered event with confidence vs threshold comparison."""
        with patch(
            "cognivault.events.emitter.get_global_event_emitter"
        ) as mock_get_emitter:
            mock_emitter = AsyncMock()
            mock_get_emitter.return_value = mock_emitter

            # Test cases where confidence meets/exceeds threshold
            test_cases = [
                (0.95, 0.90),  # Confidence exceeds threshold
                (0.85, 0.85),  # Confidence equals threshold
                (0.99, 0.95),  # High confidence, high threshold
                (0.60, 0.50),  # Lower values
            ]

            for confidence, threshold in test_cases:
                await emit_termination_triggered(
                    workflow_id=f"workflow-conf-{confidence}-thresh-{threshold}",
                    termination_reason="confidence_threshold_met",
                    confidence_score=confidence,
                    threshold=threshold,
                )

            # Verify all cases were emitted correctly
            assert mock_emitter.emit.call_count == len(test_cases)

            for i, (confidence, threshold) in enumerate(test_cases):
                emitted_event = mock_emitter.emit.call_args_list[i][0][0]
                assert emitted_event.data["confidence_score"] == confidence
                assert emitted_event.data["threshold"] == threshold
