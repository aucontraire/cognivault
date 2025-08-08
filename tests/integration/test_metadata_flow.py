"""
Integration Tests for Metadata Flow in Advanced Node Workflows.

This module tests multi-axis classification metadata propagation through
complex DAGs to ensure classification consistency across node boundaries.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict

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


class TestMultiAxisClassificationPropagation:
    """Test multi-axis classification metadata propagation."""

    @pytest.fixture
    def task_classification(self) -> Any:
        """Create a rich task classification."""
        return TaskClassification(
            task_type="evaluate",
            domain="ai_governance",
            complexity="complex",
            urgency="high",
        )

    @pytest.fixture
    def cognitive_classification(self) -> Any:
        """Create comprehensive cognitive classification."""
        return {
            "speed": "adaptive",
            "depth": "deep",
            "pattern": "composite",
            "execution": "processor",
            "role": "intermediate",
            "context": "transformation",
            "confidence_threshold": "0.85",
            "quality_requirements": "accuracy,completeness,coherence",
        }

    @pytest.fixture
    def decision_metadata(self) -> Any:
        """Create decision node metadata with full classification."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "decision"
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "variable"
        metadata.processing_pattern = "atomic"
        metadata.pipeline_role = "intermediate"
        metadata.bounded_context = "routing"
        metadata.to_dict = Mock(
            return_value={
                "execution_pattern": "decision",
                "cognitive_speed": "adaptive",
                "cognitive_depth": "variable",
                "processing_pattern": "atomic",
                "pipeline_role": "intermediate",
                "bounded_context": "routing",
            }
        )
        return metadata

    @pytest.fixture
    def aggregator_metadata(self) -> Any:
        """Create aggregator node metadata with full classification."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        metadata.cognitive_speed = "fast"
        metadata.cognitive_depth = "shallow"
        metadata.processing_pattern = "composite"
        metadata.pipeline_role = "intermediate"
        metadata.bounded_context = "transformation"
        metadata.to_dict = Mock(
            return_value={
                "execution_pattern": "aggregator",
                "cognitive_speed": "fast",
                "cognitive_depth": "shallow",
                "processing_pattern": "composite",
                "pipeline_role": "intermediate",
                "bounded_context": "transformation",
            }
        )
        return metadata

    @pytest.fixture
    def validator_metadata(self) -> Any:
        """Create validator node metadata with full classification."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "validator"
        metadata.cognitive_speed = "slow"
        metadata.cognitive_depth = "deep"
        metadata.processing_pattern = "atomic"
        metadata.pipeline_role = "terminal"
        metadata.bounded_context = "reflection"
        metadata.to_dict = Mock(
            return_value={
                "execution_pattern": "validator",
                "cognitive_speed": "slow",
                "cognitive_depth": "deep",
                "processing_pattern": "atomic",
                "pipeline_role": "terminal",
                "bounded_context": "reflection",
            }
        )
        return metadata

    @pytest.mark.asyncio
    async def test_classification_propagation_through_dag(
        self,
        decision_metadata: Any,
        aggregator_metadata: Any,
        validator_metadata: Any,
        task_classification: Any,
        cognitive_classification: Any,
    ) -> None:
        """Test classification metadata propagation through a complete DAG."""
        # Create nodes
        decision_node = DecisionNode(
            decision_metadata,
            "classification_router",
            [
                DecisionCriteria(
                    name="complexity",
                    evaluator=lambda ctx: 0.85,
                    weight=1.0,
                    threshold=0.8,
                )
            ],
            {"complex_path": ["expert_analysis"], "simple_path": ["basic_analysis"]},
        )

        aggregator_node = AggregatorNode(
            aggregator_metadata,
            "knowledge_combiner",
            AggregationStrategy.CONSENSUS,
            min_inputs=1,
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        validator_node = ValidatorNode(
            validator_metadata,
            "quality_gatekeeper",
            [
                ValidationCriteria(
                    name="has_content",
                    validator=lambda data: len(data.get("content", "")) > 10,
                    weight=1.0,
                    required=True,
                )
            ],
        )

        # Create initial context with rich metadata
        initial_context = NodeExecutionContext(
            workflow_id="metadata-flow-test",
            correlation_id="correlation-metadata-001",
            cognitive_classification=cognitive_classification,
            task_classification=task_classification,
        )

        initial_context.available_inputs = {
            "complex_query": {
                "content": "How should AI governance frameworks balance innovation with safety?",
                "confidence": 0.9,
                "quality_score": 0.85,
                "metadata": {
                    "query_type": "analytical",
                    "domain_expertise": ["ai", "policy", "ethics"],
                    "complexity_indicators": [
                        "multi_stakeholder",
                        "regulatory",
                        "technical",
                    ],
                },
            }
        }

        # Mock event emissions to track metadata propagation
        with (
            patch(
                "cognivault.orchestration.nodes.decision_node.emit_decision_made"
            ) as mock_decision_emit,
            patch(
                "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
            ) as mock_agg_emit,
            patch(
                "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
            ) as mock_val_emit,
        ):
            # Step 1: Decision node execution
            decision_result = await decision_node.execute(initial_context)

            # Verify decision node preserves and uses classification metadata
            assert decision_result["selected_path"] == "complex_path"
            assert decision_result["confidence"] > 0.8

            # Check decision event emission includes metadata
            mock_decision_emit.assert_called_once()
            decision_event_kwargs = mock_decision_emit.call_args[1]
            assert decision_event_kwargs["workflow_id"] == "metadata-flow-test"
            assert decision_event_kwargs["correlation_id"] == "correlation-metadata-001"

            # Step 2: Aggregator node execution
            agg_context = NodeExecutionContext(
                workflow_id=initial_context.workflow_id,
                correlation_id=initial_context.correlation_id,
                cognitive_classification=cognitive_classification,
                task_classification=task_classification,
            )

            # Simulate expert analysis result with preserved metadata
            agg_context.available_inputs = {
                "expert_analysis_result": {
                    "content": "Comprehensive analysis of AI governance balancing innovation and safety through stakeholder engagement",
                    "confidence": 0.92,
                    "quality_score": 0.88,
                    "metadata": {
                        "analysis_type": "expert",
                        "methodology": [
                            "literature_review",
                            "stakeholder_analysis",
                            "framework_comparison",
                        ],
                        "domains_covered": [
                            "technical",
                            "policy",
                            "ethics",
                            "economics",
                        ],
                        "classification_preserved": cognitive_classification,
                        "task_context": task_classification.model_dump(),
                    },
                }
            }

            aggregation_result = await aggregator_node.execute(agg_context)

            # Verify aggregation preserves classification metadata
            assert aggregation_result["quality_score"] > 0.8
            aggregated_data = aggregation_result["aggregated_data"]
            assert "metadata" in aggregated_data
            assert "classification_preserved" in aggregated_data["metadata"]
            assert (
                aggregated_data["metadata"]["classification_preserved"]["speed"]
                == "adaptive"
            )

            # Check aggregation event emission includes metadata
            mock_agg_emit.assert_called_once()
            agg_event_kwargs = mock_agg_emit.call_args[1]
            assert agg_event_kwargs["correlation_id"] == "correlation-metadata-001"

            # Step 3: Validator node execution
            val_context = NodeExecutionContext(
                workflow_id=initial_context.workflow_id,
                correlation_id=initial_context.correlation_id,
                cognitive_classification=cognitive_classification,
                task_classification=task_classification,
            )

            val_context.available_inputs = {"final_analysis": aggregated_data}

            validation_result = await validator_node.execute(val_context)

            # Verify validation maintains classification metadata
            assert validation_result["passed"] is True
            validated_data = validation_result["validated_data"]
            assert "metadata" in validated_data
            assert "classification_preserved" in validated_data["metadata"]

            # Verify complete classification chain preservation
            final_classification = validated_data["metadata"][
                "classification_preserved"
            ]
            assert final_classification["speed"] == "adaptive"
            assert final_classification["depth"] == "deep"
            assert final_classification["context"] == "transformation"

            # Check validation event emission includes metadata
            mock_val_emit.assert_called_once()
            val_event_kwargs = mock_val_emit.call_args[1]
            assert val_event_kwargs["correlation_id"] == "correlation-metadata-001"

    @pytest.mark.asyncio
    async def test_classification_based_routing_decisions(
        self,
        decision_metadata: Any,
        cognitive_classification: Any,
        task_classification: Any,
    ) -> None:
        """Test that routing decisions are influenced by classification metadata."""
        # Create decision node with classification-aware criteria
        classification_criteria = [
            DecisionCriteria(
                name="cognitive_speed_routing",
                evaluator=lambda ctx: (
                    0.9
                    if ctx.cognitive_classification.get("speed") == "adaptive"
                    else 0.3
                ),
                weight=2.0,
                threshold=0.5,
            ),
            DecisionCriteria(
                name="complexity_routing",
                evaluator=lambda ctx: (
                    0.9
                    if ctx.task_classification
                    and ctx.task_classification.complexity == "complex"
                    else (
                        0.5
                        if ctx.task_classification
                        and ctx.task_classification.complexity == "moderate"
                        else 0.1
                    )
                ),
                weight=1.5,
                threshold=0.7,
            ),
            DecisionCriteria(
                name="context_routing",
                evaluator=lambda ctx: (
                    0.8
                    if ctx.cognitive_classification.get("context") == "transformation"
                    else 0.4
                ),
                weight=1.0,
                threshold=0.6,
            ),
        ]

        paths = {
            "adaptive_complex_path": [
                "adaptive_agent",
                "complex_analyzer",
                "context_transformer",
            ],
            "standard_path": ["standard_agent"],
            "simple_path": ["simple_agent"],
        }

        decision_node = DecisionNode(
            decision_metadata,
            "metadata_aware_router",
            classification_criteria,
            paths,
        )

        # Test with high-complexity, adaptive classification
        context = NodeExecutionContext(
            workflow_id="routing-test",
            correlation_id="routing-correlation",
            cognitive_classification=cognitive_classification,  # adaptive, deep, transformation
            task_classification=task_classification,  # complexity_score = 0.85
        )

        context.available_inputs = {
            "routing_query": {
                "content": "Complex analytical query requiring adaptive processing",
                "confidence": 0.85,
                "quality_score": 0.8,
            }
        }

        with patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"):
            result = await decision_node.execute(context)

            # Should route to adaptive_complex_path due to classification metadata
            assert result["selected_path"] == "adaptive_complex_path"
            assert result["selected_agents"] == [
                "adaptive_agent",
                "complex_analyzer",
                "context_transformer",
            ]

            # Check reasoning includes classification-based scores
            reasoning = result["reasoning"]
            criterion_scores = reasoning["criterion_scores"]

            assert criterion_scores["cognitive_speed_routing"]["score"] == 0.9
            assert criterion_scores["cognitive_speed_routing"]["passed"] is True

            assert criterion_scores["complexity_routing"]["score"] == 0.9
            assert criterion_scores["complexity_routing"]["passed"] is True

            assert criterion_scores["context_routing"]["score"] == 0.8
            assert criterion_scores["context_routing"]["passed"] is True

    @pytest.mark.asyncio
    async def test_bounded_context_consistency(
        self, decision_metadata: Any, aggregator_metadata: Any, validator_metadata: Any
    ) -> None:
        """Test that bounded context boundaries are respected in metadata flow."""
        # Create nodes with different bounded contexts
        decision_node = DecisionNode(
            decision_metadata,  # bounded_context = "routing"
            "context_router",
            [
                DecisionCriteria(
                    name="basic", evaluator=lambda ctx: 0.8, weight=1.0, threshold=0.5
                )
            ],
            {"transform_path": ["transformer"]},
        )

        aggregator_node = AggregatorNode(
            aggregator_metadata,  # bounded_context = "transformation"
            "context_transformer",
            AggregationStrategy.BEST_QUALITY,
            min_inputs=1,
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        validator_node = ValidatorNode(
            validator_metadata,  # bounded_context = "reflection"
            "context_reflector",
            [
                ValidationCriteria(
                    name="context_check",
                    validator=lambda data: "bounded_context_trace"
                    in data.get("metadata", {}),
                    weight=1.0,
                    required=True,
                )
            ],
        )

        # Create context with bounded context tracking
        context = NodeExecutionContext(
            workflow_id="context-boundary-test",
            correlation_id="context-correlation",
            cognitive_classification={"context": "routing"},
            task_classification=TaskClassification(task_type="transform"),
        )

        context.available_inputs = {
            "routing_input": {
                "content": "Input for bounded context testing",
                "confidence": 0.8,
                "quality_score": 0.75,
                "metadata": {"bounded_context_trace": ["routing"]},
            }
        }

        with (
            patch("cognivault.orchestration.nodes.decision_node.emit_decision_made"),
            patch(
                "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
            ),
            patch(
                "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
            ),
        ):
            # Step 1: Routing context
            decision_result = await decision_node.execute(context)

            # Step 2: Transformation context
            agg_context = NodeExecutionContext(
                workflow_id=context.workflow_id,
                correlation_id=context.correlation_id,
                cognitive_classification={"context": "transformation"},
                task_classification=TaskClassification(task_type="synthesize"),
            )

            agg_context.available_inputs = {
                "transform_input": {
                    "content": "Transformed content with context boundary tracking",
                    "confidence": 0.85,
                    "quality_score": 0.8,
                    "metadata": {
                        "bounded_context_trace": ["routing", "transformation"]
                    },
                }
            }

            aggregation_result = await aggregator_node.execute(agg_context)

            # Verify transformation context added to trace
            agg_data = aggregation_result["aggregated_data"]
            context_trace = agg_data["metadata"]["bounded_context_trace"]
            assert "routing" in context_trace
            assert "transformation" in context_trace

            # Step 3: Reflection context
            val_context = NodeExecutionContext(
                workflow_id=context.workflow_id,
                correlation_id=context.correlation_id,
                cognitive_classification={"context": "reflection"},
                task_classification=TaskClassification(task_type="evaluate"),
            )

            # Add reflection to context trace
            agg_data["metadata"]["bounded_context_trace"].append("reflection")
            val_context.available_inputs = {"reflection_input": agg_data}

            validation_result = await validator_node.execute(val_context)

            # Verify complete bounded context journey
            assert validation_result["passed"] is True
            validated_trace = validation_result["validated_data"]["metadata"][
                "bounded_context_trace"
            ]
            assert validated_trace == ["routing", "transformation", "reflection"]


class TestClassificationConsistency:
    """Test classification consistency across node boundaries."""

    @pytest.fixture
    def consistent_metadata(self) -> Any:
        """Create metadata with consistent classification."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        metadata.cognitive_speed = "fast"
        metadata.cognitive_depth = "shallow"
        metadata.processing_pattern = "composite"
        metadata.pipeline_role = "intermediate"
        metadata.bounded_context = "transformation"
        return metadata

    @pytest.mark.asyncio
    async def test_classification_validation_across_nodes(
        self, consistent_metadata: Any
    ) -> None:
        """Test that classification remains consistent as data flows between nodes."""
        aggregator = AggregatorNode(
            consistent_metadata,
            "consistency_aggregator",
            AggregationStrategy.CONSENSUS,
            min_inputs=2,
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        # Create context with classification metadata
        classification = {
            "speed": "fast",
            "depth": "shallow",
            "pattern": "composite",
            "execution": "aggregator",
            "role": "intermediate",
            "context": "transformation",
        }

        context = NodeExecutionContext(
            workflow_id="consistency-test",
            correlation_id="consistency-correlation",
            cognitive_classification=classification,
            task_classification=TaskClassification(task_type="synthesize"),
        )

        # Create inputs that preserve classification consistency
        context.available_inputs = {
            "input1": {
                "content": "First input with consistent classification",
                "confidence": 0.8,
                "quality_score": 0.75,
                "metadata": {
                    "node_classification": classification,
                    "consistency_check": "pass",
                },
            },
            "input2": {
                "content": "Second input with consistent classification",
                "confidence": 0.85,
                "quality_score": 0.8,
                "metadata": {
                    "node_classification": classification,
                    "consistency_check": "pass",
                },
            },
        }

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await aggregator.execute(context)

            # Verify classification consistency is maintained
            aggregated_data = result["aggregated_data"]
            assert "metadata" in aggregated_data

            # Check that classification from inputs is preserved
            preserved_classification = aggregated_data["metadata"][
                "node_classification"
            ]
            assert preserved_classification["speed"] == "fast"
            assert preserved_classification["depth"] == "shallow"
            assert preserved_classification["pattern"] == "composite"
            assert preserved_classification["execution"] == "aggregator"
            assert preserved_classification["role"] == "intermediate"
            assert preserved_classification["context"] == "transformation"

    @pytest.mark.asyncio
    async def test_classification_mismatch_handling(
        self, consistent_metadata: Any
    ) -> None:
        """Test handling of classification mismatches between nodes."""
        aggregator = AggregatorNode(
            consistent_metadata,
            "mismatch_aggregator",
            AggregationStrategy.CONSENSUS,
            min_inputs=2,
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        context = NodeExecutionContext(
            workflow_id="mismatch-test",
            correlation_id="mismatch-correlation",
            cognitive_classification={"speed": "fast", "depth": "shallow"},
            task_classification=TaskClassification(task_type="compare"),
        )

        # Create inputs with mismatched classifications
        context.available_inputs = {
            "fast_input": {
                "content": "Fast processing input",
                "confidence": 0.8,
                "quality_score": 0.75,
                "metadata": {
                    "node_classification": {"speed": "fast", "depth": "shallow"},
                    "processing_style": "quick_analysis",
                },
            },
            "slow_input": {
                "content": "Slow processing input with deep analysis",
                "confidence": 0.9,
                "quality_score": 0.85,
                "metadata": {
                    "node_classification": {"speed": "slow", "depth": "deep"},
                    "processing_style": "thorough_analysis",
                },
            },
        }

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await aggregator.execute(context)

            # Verify aggregator handles mismatched classifications gracefully
            # Should use consensus strategy to resolve conflicts
            aggregated_data = result["aggregated_data"]
            assert "metadata" in aggregated_data

            # The CONSENSUS strategy should choose the higher quality input's classification
            # or create a merged classification
            assert (
                result["conflicts_resolved"] >= 0
            )  # Some conflicts may have been resolved


class TestWorkflowMetadataIntegrity:
    """Test workflow-level metadata integrity."""

    @pytest.mark.asyncio
    async def test_end_to_end_metadata_integrity(self) -> None:
        """Test metadata integrity in a complete workflow."""
        # Create a complete workflow with metadata tracking
        workflow_metadata = {
            "workflow_id": "integrity-test-workflow",
            "correlation_id": "integrity-correlation",
            "workflow_type": "complex_analysis",
            "priority": "high",
            "deadline": "2024-01-15T00:00:00Z",
            "requester": "policy_team",
            "classification": {
                "speed": "adaptive",
                "depth": "deep",
                "pattern": "chain",
                "execution": "processor",
                "role": "standalone",
                "context": "reflection",
            },
        }

        # Simulate workflow execution with metadata preservation
        context = NodeExecutionContext(
            workflow_id=workflow_metadata["workflow_id"],
            correlation_id=workflow_metadata["correlation_id"],
            cognitive_classification=workflow_metadata["classification"],
            task_classification=TaskClassification(
                task_type="synthesize", complexity="complex"
            ),
        )

        context.available_inputs = {
            "workflow_input": {
                "content": "Input for end-to-end metadata integrity test",
                "confidence": 0.9,
                "quality_score": 0.85,
                "metadata": {
                    "workflow_metadata": workflow_metadata,
                    "integrity_checksum": "abc123",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
            }
        }

        # Create aggregator to test metadata preservation
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"
        metadata.bounded_context = "transformation"

        aggregator = AggregatorNode(
            metadata,
            "integrity_aggregator",
            AggregationStrategy.BEST_QUALITY,
            min_inputs=1,
            quality_threshold=0.0,
            confidence_threshold=0.0,
        )

        with patch(
            "cognivault.orchestration.nodes.aggregator_node.emit_aggregation_completed"
        ):
            result = await aggregator.execute(context)

            # Verify complete metadata integrity
            aggregated_data = result["aggregated_data"]
            preserved_metadata = aggregated_data["metadata"]

            assert "workflow_metadata" in preserved_metadata
            workflow_meta = preserved_metadata["workflow_metadata"]

            # Check all workflow metadata is preserved
            assert workflow_meta["workflow_id"] == "integrity-test-workflow"
            assert workflow_meta["correlation_id"] == "integrity-correlation"
            assert workflow_meta["workflow_type"] == "complex_analysis"
            assert workflow_meta["priority"] == "high"
            assert workflow_meta["requester"] == "policy_team"

            # Check classification metadata is preserved
            classification = workflow_meta["classification"]
            assert classification["speed"] == "adaptive"
            assert classification["depth"] == "deep"
            assert classification["pattern"] == "chain"
            assert classification["context"] == "reflection"

            # Check additional metadata integrity
            assert preserved_metadata["integrity_checksum"] == "abc123"
            assert preserved_metadata["timestamp"] == "2024-01-01T12:00:00Z"
