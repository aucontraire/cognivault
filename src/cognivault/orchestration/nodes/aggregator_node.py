"""
Aggregator Node Implementation for CogniVault.

This module implements the AggregatorNode class which handles parallel
output combination and synthesis in the advanced node execution system.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass  # Keep for any remaining dataclasses
from enum import Enum
import statistics
import asyncio

from pydantic import BaseModel, Field, ConfigDict, field_validator
from cognivault.agents.metadata import AgentMetadata
from cognivault.events import emit_aggregation_completed
from .base_advanced_node import BaseAdvancedNode, NodeExecutionContext


class AggregationStrategy(Enum):
    """Available aggregation strategies."""

    CONSENSUS = "consensus"
    WEIGHTED = "weighted"
    HIERARCHICAL = "hierarchical"
    FIRST_WINS = "first_wins"
    MAJORITY_VOTE = "majority_vote"
    AVERAGE = "average"
    BEST_QUALITY = "best_quality"


class AggregationInput(BaseModel):
    """
    Represents input from a single source for aggregation.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    source: str = Field(
        ...,
        description="Source identifier for this aggregation input",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "agent_1"},
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Data payload from the source",
        json_schema_extra={"example": {"content": "Analysis result", "type": "text"}},
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence level of this input (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    weight: float = Field(
        default=1.0,
        description="Weight/importance of this input in aggregation",
        ge=0.0,
        json_schema_extra={"example": 1.5},
    )
    quality_score: float = Field(
        default=0.0,
        description="Quality score of this input (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92},
    )
    timestamp: Optional[float] = Field(
        default=None,
        description="Timestamp when this input was created",
        ge=0.0,
        json_schema_extra={"example": 1640995200.0},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class AggregatorNode(BaseAdvancedNode):
    """
    Parallel output combination and synthesis node.

    This node aggregates outputs from multiple parallel execution paths
    using configurable strategies to produce a unified result.
    """

    def __init__(
        self,
        metadata: AgentMetadata,
        node_name: str,
        aggregation_strategy: AggregationStrategy,
        min_inputs: int = 2,
        max_inputs: Optional[int] = None,
        quality_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the AggregatorNode.

        Parameters
        ----------
        metadata : AgentMetadata
            The agent metadata containing multi-axis classification
        node_name : str
            Unique name for this node instance
        aggregation_strategy : AggregationStrategy
            Strategy to use for aggregating inputs
        min_inputs : int
            Minimum number of inputs required for aggregation
        max_inputs : Optional[int]
            Maximum number of inputs to process (None for unlimited)
        quality_threshold : float
            Minimum quality threshold for inputs to be considered
        confidence_threshold : float
            Minimum confidence threshold for inputs to be considered
        """
        super().__init__(metadata, node_name)

        if self.execution_pattern != "aggregator":
            raise ValueError(
                f"AggregatorNode requires execution_pattern='aggregator', "
                f"got '{self.execution_pattern}'"
            )

        if min_inputs < 1:
            raise ValueError(f"min_inputs must be at least 1, got {min_inputs}")

        if max_inputs is not None and max_inputs < min_inputs:
            raise ValueError(
                f"max_inputs ({max_inputs}) must be >= min_inputs ({min_inputs})"
            )

        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError(
                f"quality_threshold must be between 0.0 and 1.0, got {quality_threshold}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )

        self.aggregation_strategy = aggregation_strategy
        self.min_inputs = min_inputs
        self.max_inputs = max_inputs
        self.quality_threshold = quality_threshold
        self.confidence_threshold = confidence_threshold

    async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
        """
        Execute the aggregation logic and combine parallel inputs.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, Any]
            Aggregation result with combined data and metadata
        """
        # Pre-execution setup
        await self.pre_execute(context)

        # Validate context
        validation_errors = self.validate_context(context)
        if validation_errors:
            raise ValueError(
                f"Context validation failed: {', '.join(validation_errors)}"
            )

        # Collect inputs from parallel execution paths
        inputs = await self._collect_parallel_inputs(context)

        # Apply aggregation strategy
        aggregated_result = await self._aggregate_inputs(inputs)

        # Emit aggregation event
        await emit_aggregation_completed(
            workflow_id=context.workflow_id,
            aggregation_strategy=self.aggregation_strategy.value,
            input_sources=list(inputs.keys()),
            output_quality_score=aggregated_result["quality_score"],
            conflicts_resolved=aggregated_result["conflicts_resolved"],
            aggregation_time_ms=aggregated_result.get("aggregation_time_ms"),
            correlation_id=context.correlation_id,
        )

        # Post-execution cleanup
        await self.post_execute(context, aggregated_result)

        return aggregated_result

    def can_handle(self, context: NodeExecutionContext) -> bool:
        """
        Check if this node can handle the given context.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context to evaluate

        Returns
        -------
        bool
            True if the node can handle the context
        """
        try:
            # Check if we have enough inputs available
            if len(context.available_inputs) < self.min_inputs:
                return False

            # Check if we have the necessary data structure
            if not context.available_inputs:
                return False

            # Check if inputs meet quality and confidence thresholds
            valid_inputs = 0
            for source, data in context.available_inputs.items():
                if isinstance(data, dict):
                    confidence = data.get("confidence", 0.0)
                    quality = data.get("quality_score", 0.0)

                    if (
                        confidence >= self.confidence_threshold
                        and quality >= self.quality_threshold
                    ):
                        valid_inputs += 1

            return valid_inputs >= self.min_inputs
        except Exception:
            return False

    async def _collect_parallel_inputs(
        self, context: NodeExecutionContext
    ) -> Dict[str, AggregationInput]:
        """
        Collect and prepare inputs from parallel execution paths.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, AggregationInput]
            Collected inputs ready for aggregation
        """
        inputs = {}

        for source, data in context.available_inputs.items():
            if isinstance(data, dict):
                try:
                    # Extract metadata from the input with type validation
                    confidence = data.get("confidence", 0.0)
                    quality_score = data.get("quality_score", 0.0)
                    weight = data.get("weight", 1.0)
                    timestamp = data.get("timestamp")

                    # Ensure numeric types (handle corrupted inputs)
                    if not isinstance(confidence, (int, float)):
                        confidence = 0.0
                    if not isinstance(quality_score, (int, float)):
                        quality_score = 0.0
                    if not isinstance(weight, (int, float)):
                        weight = 1.0

                    # Filter by thresholds
                    if (
                        confidence >= self.confidence_threshold
                        and quality_score >= self.quality_threshold
                    ):
                        inputs[source] = AggregationInput(
                            source=source,
                            data=data,
                            confidence=confidence,
                            weight=weight,
                            quality_score=quality_score,
                            timestamp=timestamp,
                        )
                except (TypeError, ValueError):
                    # Skip corrupted inputs gracefully
                    continue

        # Respect max_inputs limit
        if self.max_inputs and len(inputs) > self.max_inputs:
            # Sort by quality score descending and take top N
            sorted_inputs = sorted(
                inputs.items(), key=lambda x: x[1].quality_score, reverse=True
            )
            inputs = dict(sorted_inputs[: self.max_inputs])

        if len(inputs) < self.min_inputs:
            raise ValueError(
                f"Insufficient inputs for aggregation. "
                f"Required: {self.min_inputs}, Available: {len(inputs)}"
            )

        return inputs

    async def _aggregate_inputs(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """
        Apply aggregation strategy to combine inputs.

        Parameters
        ----------
        inputs : Dict[str, AggregationInput]
            Inputs to aggregate

        Returns
        -------
        Dict[str, Any]
            Aggregated result
        """
        start_time = asyncio.get_event_loop().time()

        # Apply the selected aggregation strategy
        if self.aggregation_strategy == AggregationStrategy.CONSENSUS:
            result = await self._consensus_aggregation(inputs)
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED:
            result = await self._weighted_aggregation(inputs)
        elif self.aggregation_strategy == AggregationStrategy.HIERARCHICAL:
            result = await self._hierarchical_aggregation(inputs)
        elif self.aggregation_strategy == AggregationStrategy.FIRST_WINS:
            result = await self._first_wins_aggregation(inputs)
        elif self.aggregation_strategy == AggregationStrategy.MAJORITY_VOTE:
            result = await self._majority_vote_aggregation(inputs)
        elif self.aggregation_strategy == AggregationStrategy.AVERAGE:
            result = await self._average_aggregation(inputs)
        elif self.aggregation_strategy == AggregationStrategy.BEST_QUALITY:
            result = await self._best_quality_aggregation(inputs)
        else:
            raise ValueError(
                f"Unknown aggregation strategy: {self.aggregation_strategy}"
            )

        end_time = asyncio.get_event_loop().time()
        aggregation_time_ms = (end_time - start_time) * 1000

        # Add metadata to result
        result.update(
            {
                "aggregation_strategy": self.aggregation_strategy.value,
                "input_count": len(inputs),
                "input_sources": list(inputs.keys()),
                "aggregation_time_ms": aggregation_time_ms,
                "conflicts_resolved": result.get("conflicts_resolved", 0),
                "quality_score": result.get("quality_score", 0.0),
            }
        )

        return result

    async def _consensus_aggregation(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """
        Aggregate inputs using consensus strategy.

        This strategy attempts to find common ground between inputs
        and resolves conflicts through weighted voting.
        """
        # Simple consensus implementation - in practice this would be more sophisticated
        all_data = [inp.data for inp in inputs.values()]

        # Find common keys across all inputs
        common_keys = set(all_data[0].keys())
        for data in all_data[1:]:
            common_keys.intersection_update(data.keys())

        consensus_data = {}
        conflicts_resolved = 0

        for key in common_keys:
            values = [data[key] for data in all_data]

            # If all values are the same, use consensus
            if len(set(str(v) for v in values)) == 1:
                consensus_data[key] = values[0]
            else:
                # Conflict resolution: use weighted average or most common
                conflicts_resolved += 1
                if all(isinstance(v, (int, float)) for v in values):
                    weights = [inp.weight for inp in inputs.values()]
                    consensus_data[key] = sum(
                        v * w for v, w in zip(values, weights)
                    ) / sum(weights)
                else:
                    # For non-numeric values, use the value from highest quality input
                    best_input = max(inputs.values(), key=lambda x: x.quality_score)
                    consensus_data[key] = best_input.data[key]

        # Calculate overall quality score
        quality_scores = [inp.quality_score for inp in inputs.values()]
        quality_score = statistics.mean(quality_scores)

        return {
            "aggregated_data": consensus_data,
            "quality_score": quality_score,
            "conflicts_resolved": conflicts_resolved,
        }

    async def _weighted_aggregation(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """Aggregate inputs using weighted strategy."""
        # Implementation placeholder - would combine inputs based on weights
        weights = [inp.weight for inp in inputs.values()]
        total_weight = sum(weights)

        # For now, return the highest weighted input
        best_input = max(inputs.values(), key=lambda x: x.weight)

        return {
            "aggregated_data": best_input.data,
            "quality_score": best_input.quality_score,
            "conflicts_resolved": 0,
        }

    async def _hierarchical_aggregation(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """Aggregate inputs using hierarchical strategy."""
        # Sort by quality score and apply hierarchical combination
        sorted_inputs = sorted(
            inputs.values(), key=lambda x: x.quality_score, reverse=True
        )

        return {
            "aggregated_data": sorted_inputs[0].data,
            "quality_score": sorted_inputs[0].quality_score,
            "conflicts_resolved": 0,
        }

    async def _first_wins_aggregation(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """Aggregate inputs using first-wins strategy."""
        # Use the first input (by timestamp or insertion order)
        first_input = next(iter(inputs.values()))

        return {
            "aggregated_data": first_input.data,
            "quality_score": first_input.quality_score,
            "conflicts_resolved": 0,
        }

    async def _majority_vote_aggregation(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """Aggregate inputs using majority vote strategy."""
        # Simple majority vote - would be more sophisticated in practice
        first_input = next(iter(inputs.values()))

        return {
            "aggregated_data": first_input.data,
            "quality_score": statistics.mean(
                [inp.quality_score for inp in inputs.values()]
            ),
            "conflicts_resolved": 0,
        }

    async def _average_aggregation(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """Aggregate inputs using average strategy."""
        # Average numeric values across inputs
        quality_scores = [inp.quality_score for inp in inputs.values()]

        # For now, return average of quality scores
        return {
            "aggregated_data": {"average_quality": statistics.mean(quality_scores)},
            "quality_score": statistics.mean(quality_scores),
            "conflicts_resolved": 0,
        }

    async def _best_quality_aggregation(
        self, inputs: Dict[str, AggregationInput]
    ) -> Dict[str, Any]:
        """Aggregate inputs using best quality strategy."""
        # Select the input with highest quality score
        best_input = max(inputs.values(), key=lambda x: x.quality_score)

        return {
            "aggregated_data": best_input.data,
            "quality_score": best_input.quality_score,
            "conflicts_resolved": 0,
        }
