"""
Integration Tests for Advanced Node Interoperability.

This module tests complex multi-node workflows to ensure advanced nodes
work together seamlessly in realistic scenarios.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List

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


class TestComplexWorkflows:
    """Test complex multi-node workflows."""

    @pytest.fixture
    def mock_metadata_decision(self) -> Any:
        """Create mock metadata for decision node."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "variable"
        metadata.bounded_context = "routing"
        return metadata

    @pytest.fixture
    def mock_metadata_aggregator(self) -> Any:
        """Create mock metadata for aggregator node."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        metadata.cognitive_speed = "fast"
        metadata.cognitive_depth = "shallow"
        metadata.bounded_context = "transformation"
        return metadata

    @pytest.fixture
    def mock_metadata_validator(self) -> Any:
        """Create mock metadata for validator node."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "validator"
        metadata.cognitive_speed = "slow"
        metadata.cognitive_depth = "deep"
        metadata.bounded_context = "reflection"
        return metadata

    @pytest.fixture
    def mock_metadata_terminator(self) -> Any:
        """Create mock metadata for terminator node."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "terminator"
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "variable"
        metadata.bounded_context = "reflection"
        return metadata

    @pytest.fixture
    def workflow_context(self) -> Any:
        """Create a workflow execution context."""
        context = NodeExecutionContext(
            workflow_id="workflow-001",
            correlation_id="correlation-001",
            cognitive_classification={"speed": "adaptive", "depth": "deep"},
            task_classification=TaskClassification(task_type="evaluate"),
        )

        # Add initial data for the workflow
        context.available_inputs = {
            "initial_query": {
                "content": "What are the implications of AI governance?",
                "confidence": 0.9,
                "quality_score": 0.85,
                "metadata": {"source": "user_input", "complexity": "high"},
            }
        }

        return context

    @pytest.mark.asyncio
    async def test_decision_to_aggregator_workflow(
        self,
        mock_metadata_decision: Any,
        mock_metadata_aggregator: Any,
        workflow_context: Any,
    ) -> None:
        """Test Decision → Aggregator workflow."""
        # Create decision node
        decision_criteria = [
            DecisionCriteria(
                name="complexity_level",
                evaluator=lambda ctx: 0.9,  # High complexity
                weight=1.0,
                threshold=0.8,
            )
        ]
        paths = {
            "complex_path": ["agent1", "agent2", "agent3"],
            "simple_path": ["agent1"],
        }
        decision_node = DecisionNode(
            mock_metadata_decision, "complexity_router", decision_criteria, paths
        )

        # Create aggregator node
        aggregator_node = AggregatorNode(
            mock_metadata_aggregator,
            "result_combiner",
            AggregationStrategy.CONSENSUS,
            min_inputs=2,
        )

        # Mock event emissions
        with (
            patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"),
            patch(
                "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
            ),
        ):
            # Execute decision node
            decision_result = await decision_node.execute(workflow_context)

            # Verify decision output
            assert decision_result["selected_path"] == "complex_path"
            assert decision_result["selected_agents"] == ["agent1", "agent2", "agent3"]

            # Prepare aggregator context with multiple inputs from decision
            aggregator_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
            )

            # Simulate outputs from the three selected agents
            aggregator_context.available_inputs = {
                "agent1_output": {
                    "content": "AI governance requires international cooperation",
                    "confidence": 0.88,
                    "quality_score": 0.85,
                    "metadata": {"source": "agent1", "analysis_depth": "surface"},
                },
                "agent2_output": {
                    "content": "Regulatory frameworks must balance innovation and safety",
                    "confidence": 0.92,
                    "quality_score": 0.90,
                    "metadata": {"source": "agent2", "analysis_depth": "detailed"},
                },
                "agent3_output": {
                    "content": "Technical standards and ethical guidelines are essential",
                    "confidence": 0.85,
                    "quality_score": 0.88,
                    "metadata": {"source": "agent3", "analysis_depth": "comprehensive"},
                },
            }

            # Execute aggregator node
            aggregation_result = await aggregator_node.execute(aggregator_context)

            # Verify aggregation output
            assert aggregation_result["aggregation_strategy"] == "consensus"
            assert aggregation_result["input_count"] == 3
            assert aggregation_result["input_sources"] == [
                "agent1_output",
                "agent2_output",
                "agent3_output",
            ]
            assert aggregation_result["quality_score"] > 0.8

    @pytest.mark.asyncio
    async def test_aggregator_to_validator_workflow(
        self,
        mock_metadata_aggregator: Any,
        mock_metadata_validator: Any,
        workflow_context: Any,
    ) -> None:
        """Test Aggregator → Validator workflow."""
        # Create aggregator node with low thresholds
        aggregator_node = AggregatorNode(
            mock_metadata_aggregator,
            "parallel_combiner",
            AggregationStrategy.BEST_QUALITY,
            min_inputs=2,
            quality_threshold=0.0,  # Accept all quality levels
            confidence_threshold=0.0,  # Accept all confidence levels
        )

        # Create validator node
        validation_criteria = [
            ValidationCriteria(
                name="content_completeness",
                validator=lambda data: len(data.get("content", "")) > 50,
                weight=2.0,
                required=True,
                error_message="Content must be substantial",
            ),
            ValidationCriteria(
                name="quality_threshold",
                validator=lambda data: data.get("quality_score", 0.0) >= 0.8,
                weight=1.0,
                required=True,
                error_message="Quality score must be high",
            ),
        ]
        validator_node = ValidatorNode(
            mock_metadata_validator,
            "quality_gate",
            validation_criteria,
            quality_threshold=0.85,
        )

        # Mock event emissions
        with (
            patch(
                "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
            ),
            patch(
                "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
            ),
        ):
            # Prepare aggregator context
            aggregator_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
            )

            aggregator_context.available_inputs = {
                "source1": {
                    "content": "Comprehensive analysis of AI governance challenges and opportunities",
                    "confidence": 0.95,
                    "quality_score": 0.92,
                    "metadata": {"analysis_type": "comprehensive"},
                },
                "source2": {
                    "content": "Detailed regulatory framework recommendations",
                    "confidence": 0.88,
                    "quality_score": 0.85,
                    "metadata": {"analysis_type": "regulatory"},
                },
            }

            # Execute aggregator
            aggregation_result = await aggregator_node.execute(aggregator_context)

            # Prepare validator context with aggregator output
            validator_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
            )

            # Use aggregator output as validator input
            validator_context.available_inputs = {
                "aggregated_result": aggregation_result["aggregated_data"]
            }
            validator_context.available_inputs["aggregated_result"]["quality_score"] = (
                aggregation_result["quality_score"]
            )

            # Execute validator
            validation_result = await validator_node.execute(validator_context)

            # Verify validation passed
            assert validation_result["validation_result"] == "pass"
            assert validation_result["passed"] is True
            assert validation_result["quality_score"] >= 0.85

    @pytest.mark.asyncio
    async def test_validator_to_terminator_workflow(
        self,
        mock_metadata_validator: Any,
        mock_metadata_terminator: Any,
        workflow_context: Any,
    ) -> None:
        """Test Validator → Terminator workflow."""
        # Create validator node
        validation_criteria = [
            ValidationCriteria(
                name="minimum_quality",
                validator=lambda data: data.get("quality_score", 0.0) >= 0.9,
                weight=1.0,
                required=True,
            ),
        ]
        validator_node = ValidatorNode(
            mock_metadata_validator,
            "quality_check",
            validation_criteria,
            quality_threshold=0.9,
        )

        # Create terminator node
        termination_criteria = [
            TerminationCriteria(
                name="quality_sufficient",
                evaluator=lambda data: data.get("quality_score", 0.0) >= 0.95,
                threshold=0.95,
                weight=1.0,
                required=True,
                description="Quality is sufficient for early termination",
            ),
        ]
        terminator_node = TerminatorNode(
            mock_metadata_terminator,
            "early_terminator",
            termination_criteria,
            confidence_threshold=0.95,
        )

        # Mock event emissions
        with (
            patch(
                "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
            ),
            patch(
                "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
            ),
        ):
            # Prepare validator context with high-quality data
            validator_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
            )

            validator_context.available_inputs = {
                "high_quality_input": {
                    "content": "Exceptionally detailed and accurate analysis",
                    "confidence": 0.98,
                    "quality_score": 0.96,
                    "metadata": {"validation_source": "expert_review"},
                }
            }

            # Execute validator
            validation_result = await validator_node.execute(validator_context)

            # Prepare terminator context
            terminator_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
                execution_path=["decision", "aggregator", "validator"],  # Some history
            )

            # Use validation result as terminator input
            terminator_context.available_inputs = {
                "validated_result": validation_result["validated_data"]
            }
            terminator_context.available_inputs["validated_result"]["quality_score"] = (
                validation_result["quality_score"]
            )

            # Execute terminator
            termination_result = await terminator_node.execute(terminator_context)

            # Verify termination was triggered
            assert termination_result["should_terminate"] is True
            assert termination_result["termination_reason"] == "confidence_threshold"
            assert termination_result["confidence_score"] >= 0.95

    @pytest.mark.asyncio
    async def test_full_pipeline_workflow(
        self,
        mock_metadata_decision: Any,
        mock_metadata_aggregator: Any,
        mock_metadata_validator: Any,
        mock_metadata_terminator: Any,
        workflow_context: Any,
    ) -> None:
        """Test complete Decision → Aggregator → Validator → Terminator pipeline."""
        # Create all nodes
        decision_node = DecisionNode(
            mock_metadata_decision,
            "router",
            [
                DecisionCriteria(
                    name="complexity",
                    evaluator=lambda ctx: 0.85,
                    weight=1.0,
                    threshold=0.8,
                )
            ],
            {"high_quality": ["expert_agent"], "standard": ["standard_agent"]},
        )

        aggregator_node = AggregatorNode(
            mock_metadata_aggregator,
            "combiner",
            AggregationStrategy.CONSENSUS,
            min_inputs=1,
        )

        validator_node = ValidatorNode(
            mock_metadata_validator,
            "gate",
            [
                ValidationCriteria(
                    name="basic_quality",
                    validator=lambda data: data.get("quality_score", 0) > 0.8,
                    weight=1.0,
                    required=True,
                )
            ],
        )

        terminator_node = TerminatorNode(
            mock_metadata_terminator,
            "terminator",
            [
                TerminationCriteria(
                    name="sufficient",
                    evaluator=lambda data: data.get("quality_score", 0) >= 0.9,
                    threshold=0.9,
                    weight=1.0,
                    required=True,
                )
            ],
        )

        # Mock all event emissions
        with (
            patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"),
            patch(
                "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
            ),
            patch(
                "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
            ),
            patch(
                "cognivault.orchestration.nodes.terminator_node.emit_termination_triggered"
            ),
        ):
            # Step 1: Decision
            decision_result = await decision_node.execute(workflow_context)
            assert decision_result["selected_path"] == "high_quality"

            # Step 2: Aggregator (simulate expert agent output)
            agg_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
            )
            agg_context.available_inputs = {
                "expert_output": {
                    "content": "Expert-level comprehensive analysis",
                    "confidence": 0.95,
                    "quality_score": 0.93,
                    "metadata": {"agent_type": "expert"},
                }
            }

            aggregation_result = await aggregator_node.execute(agg_context)
            assert aggregation_result["quality_score"] >= 0.9

            # Step 3: Validator
            val_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
            )
            val_context.available_inputs = {
                "aggregated": aggregation_result["aggregated_data"]
            }
            val_context.available_inputs["aggregated"]["quality_score"] = (
                aggregation_result["quality_score"]
            )

            validation_result = await validator_node.execute(val_context)
            assert validation_result["passed"] is True

            # Step 4: Terminator
            term_context = NodeExecutionContext(
                workflow_id=workflow_context.workflow_id,
                correlation_id=workflow_context.correlation_id,
                cognitive_classification=workflow_context.cognitive_classification,
                task_classification=workflow_context.task_classification,
                execution_path=["decision", "aggregator", "validator"],
            )
            term_context.available_inputs = {
                "final_result": validation_result["validated_data"]
            }
            term_context.available_inputs["final_result"]["quality_score"] = (
                validation_result["quality_score"]
            )

            termination_result = await terminator_node.execute(term_context)

            # Verify complete workflow success
            assert termination_result["should_terminate"] is True
            assert termination_result["confidence_score"] >= 0.9


class TestNodeFailureHandling:
    """Test how nodes handle failures in multi-node workflows."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create generic mock metadata."""
        metadata = Mock(spec=AgentMetadata)
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "deep"
        metadata.bounded_context = "transformation"
        return metadata

    @pytest.mark.asyncio
    async def test_aggregator_insufficient_inputs_fallback(
        self, mock_metadata: Any
    ) -> None:
        """Test aggregator fallback when insufficient inputs are available."""
        mock_metadata.execution_pattern = "aggregator"

        aggregator_node = AggregatorNode(
            mock_metadata,
            "fallback_aggregator",
            AggregationStrategy.CONSENSUS,
            min_inputs=3,  # Require 3 inputs
        )

        context = NodeExecutionContext(
            workflow_id="test-workflow",
            correlation_id="test-correlation",
            cognitive_classification={"speed": "fast"},
            task_classification=TaskClassification(task_type="transform"),
        )

        # Only provide 2 inputs (less than required 3)
        context.available_inputs = {
            "input1": {
                "content": "First input",
                "confidence": 0.8,
                "quality_score": 0.85,
            },
            "input2": {
                "content": "Second input",
                "confidence": 0.9,
                "quality_score": 0.90,
            },
        }

        # Should raise ValueError due to insufficient inputs
        with pytest.raises(ValueError, match="Insufficient inputs for aggregation"):
            await aggregator_node.execute(context)

    @pytest.mark.asyncio
    async def test_validator_with_failing_criteria(self, mock_metadata: Any) -> None:
        """Test validator handling of failing validation criteria."""
        mock_metadata.execution_pattern = "validator"

        # Create criteria that will fail
        failing_criteria = [
            ValidationCriteria(
                name="impossible_requirement",
                validator=lambda data: False,  # Always fails
                weight=1.0,
                required=True,
                error_message="This requirement cannot be met",
            ),
        ]

        validator_node = ValidatorNode(
            mock_metadata,
            "strict_validator",
            failing_criteria,
            quality_threshold=0.8,
            strict_mode=True,
        )

        context = NodeExecutionContext(
            workflow_id="test-workflow",
            correlation_id="test-correlation",
            cognitive_classification={"speed": "slow"},
            task_classification=TaskClassification(task_type="transform"),
        )

        context.available_inputs = {
            "test_input": {
                "content": "Valid content",
                "confidence": 0.9,
                "quality_score": 0.95,
            }
        }

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ):
            result = await validator_node.execute(context)

            # Should fail validation
            assert result["validation_result"] == "fail"
            assert result["passed"] is False
            # Check that the failing criterion is in the criteria results as failed
            assert (
                result["criteria_results"]["impossible_requirement"]["passed"] is False
            )

    @pytest.mark.asyncio
    async def test_terminator_context_validation_failure(
        self, mock_metadata: Any
    ) -> None:
        """Test terminator handling of invalid context."""
        from pydantic import ValidationError

        mock_metadata.execution_pattern = "terminator"

        terminator_node = TerminatorNode(
            mock_metadata,
            "context_validator",
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

        # Should raise ValidationError when creating invalid context
        # (Pydantic validates on construction now)
        with pytest.raises(ValidationError):
            invalid_context = NodeExecutionContext(
                workflow_id="",  # Invalid empty workflow_id
                correlation_id="",  # Invalid empty correlation_id
                cognitive_classification={},  # Empty classification
                task_classification=None,  # Null task classification
            )


class TestDataFlowIntegrity:
    """Test data integrity as it flows between nodes."""

    @pytest.fixture
    def mock_metadata_aggregator(self) -> Any:
        """Mock metadata for aggregator."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        return metadata

    @pytest.fixture
    def mock_metadata_validator(self) -> Any:
        """Mock metadata for validator."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "validator"
        return metadata

    @pytest.mark.asyncio
    async def test_metadata_preservation_across_nodes(
        self, mock_metadata_aggregator: Any, mock_metadata_validator: Any
    ) -> None:
        """Test that metadata is preserved as data flows between nodes."""
        # Create nodes with permissive thresholds
        aggregator = AggregatorNode(
            mock_metadata_aggregator,
            "metadata_aggregator",
            AggregationStrategy.BEST_QUALITY,
            min_inputs=1,  # Only need 1 input for this test
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        validator = ValidatorNode(
            mock_metadata_validator,
            "metadata_validator",
            [
                ValidationCriteria(
                    name="has_metadata",
                    validator=lambda data: "metadata" in data,
                    weight=1.0,
                    required=True,
                )
            ],
        )

        # Create context with rich metadata
        original_metadata = {
            "source": "expert_system",
            "timestamp": "2024-01-01T00:00:00Z",
            "analysis_depth": "comprehensive",
            "confidence_level": "high",
        }

        agg_context = NodeExecutionContext(
            workflow_id="metadata-test",
            correlation_id="metadata-correlation",
            cognitive_classification={"speed": "slow", "depth": "deep"},
            task_classification=TaskClassification(task_type="transform"),
        )

        agg_context.available_inputs = {
            "metadata_rich_input": {
                "content": "Content with rich metadata",
                "confidence": 0.95,
                "quality_score": 0.92,
                "metadata": original_metadata,
            }
        }

        with (
            patch(
                "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
            ),
            patch(
                "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
            ),
        ):
            # Execute aggregator
            agg_result = await aggregator.execute(agg_context)

            # Verify metadata is preserved in aggregation
            assert "metadata" in agg_result["aggregated_data"]
            aggregated_metadata = agg_result["aggregated_data"]["metadata"]
            assert aggregated_metadata["source"] == "expert_system"
            assert aggregated_metadata["analysis_depth"] == "comprehensive"

            # Pass to validator
            val_context = NodeExecutionContext(
                workflow_id="metadata-test",
                correlation_id="metadata-correlation",
                cognitive_classification={"speed": "slow", "depth": "deep"},
                task_classification=TaskClassification(task_type="transform"),
            )

            val_context.available_inputs = {"aggregated": agg_result["aggregated_data"]}

            # Execute validator
            val_result = await validator.execute(val_context)

            # Verify metadata is still preserved after validation
            assert val_result["passed"] is True
            assert "metadata" in val_result["validated_data"]
            validated_metadata = val_result["validated_data"]["metadata"]
            assert validated_metadata["source"] == "expert_system"

    @pytest.mark.asyncio
    async def test_correlation_id_propagation(
        self, mock_metadata_aggregator: Any
    ) -> None:
        """Test that correlation IDs are properly propagated."""
        aggregator = AggregatorNode(
            mock_metadata_aggregator,
            "correlation_aggregator",
            AggregationStrategy.CONSENSUS,
        )

        original_correlation_id = "correlation-123-test"

        context = NodeExecutionContext(
            workflow_id="correlation-test",
            correlation_id=original_correlation_id,
            cognitive_classification={"speed": "fast"},
            task_classification=TaskClassification(task_type="transform"),
        )

        context.available_inputs = {
            "input1": {
                "content": "Test content",
                "confidence": 0.8,
                "quality_score": 0.85,
            },
            "input2": {
                "content": "More content",
                "confidence": 0.9,
                "quality_score": 0.90,
            },
        }

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ) as mock_emit:
            await aggregator.execute(context)

            # Verify correlation ID was passed to event emission
            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args[1]
            assert call_kwargs["correlation_id"] == original_correlation_id
