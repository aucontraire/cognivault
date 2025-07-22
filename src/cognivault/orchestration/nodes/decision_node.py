"""
Decision Node Implementation for CogniVault.

This module implements the DecisionNode class which handles conditional
routing and flow control in the advanced node execution system.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass  # Keep for any remaining dataclasses

from pydantic import BaseModel, Field, ConfigDict
from cognivault.agents.metadata import AgentMetadata
from cognivault.events import emit_decision_made
from .base_advanced_node import BaseAdvancedNode, NodeExecutionContext


class DecisionCriteria(BaseModel):
    """
    Represents a single decision criterion.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    name: str = Field(
        ...,
        description="Name/identifier of the decision criterion",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "complexity_check"},
    )
    evaluator: Callable[[NodeExecutionContext], float] = Field(
        ...,
        description="Function that evaluates the criterion against context",
    )
    weight: float = Field(
        default=1.0,
        description="Weight/importance of this criterion in decision making",
        ge=0.0,
        le=10.0,
        json_schema_extra={"example": 2.0},
    )
    threshold: float = Field(
        default=0.5,
        description="Threshold value for criterion to be considered passed",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.8},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For Callable evaluator function
    )

    def evaluate(self, context: NodeExecutionContext) -> float:
        """Evaluate this criterion against the context."""
        return self.evaluator(context)


class DecisionNode(BaseAdvancedNode):
    """
    Conditional routing decision point node.

    This node evaluates multiple criteria to make routing decisions,
    selecting the appropriate execution path based on the context.
    """

    def __init__(
        self,
        metadata: AgentMetadata,
        node_name: str,
        decision_criteria: List[DecisionCriteria],
        paths: Dict[str, List[str]],
    ):
        """
        Initialize the DecisionNode.

        Parameters
        ----------
        metadata : AgentMetadata
            The agent metadata containing multi-axis classification
        node_name : str
            Unique name for this node instance
        decision_criteria : List[DecisionCriteria]
            List of criteria to evaluate for decision making
        paths : Dict[str, List[str]]
            Mapping of path names to lists of agent/node names
        """
        super().__init__(metadata, node_name)

        if self.execution_pattern != "decision":
            raise ValueError(
                f"DecisionNode requires execution_pattern='decision', "
                f"got '{self.execution_pattern}'"
            )

        if not decision_criteria:
            raise ValueError("DecisionNode requires at least one decision criterion")

        if not paths:
            raise ValueError("DecisionNode requires at least one path")

        self.decision_criteria = decision_criteria
        self.paths = paths
        self.default_path = list(paths.keys())[0]  # First path is default

    async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
        """
        Execute the decision logic and select an execution path.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, Any]
            Decision result with selected path and metadata
        """
        # Pre-execution setup
        await self.pre_execute(context)

        # Validate context
        validation_errors = self.validate_context(context)
        if validation_errors:
            raise ValueError(
                f"Context validation failed: {', '.join(validation_errors)}"
            )

        # Evaluate decision criteria
        decision_result = await self._evaluate_criteria(context)

        # Emit decision event
        await emit_decision_made(
            workflow_id=context.workflow_id,
            decision_criteria=[c.name for c in self.decision_criteria],
            selected_path=decision_result["selected_path"],
            confidence_score=decision_result["confidence"],
            alternative_paths=decision_result["alternatives"],
            reasoning=decision_result["reasoning"],
            correlation_id=context.correlation_id,
        )

        # Post-execution cleanup
        await self.post_execute(context, decision_result)

        return decision_result

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
        # Basic validation
        if not context.cognitive_classification:
            return False

        # Check if we have the necessary data to evaluate criteria
        try:
            for criterion in self.decision_criteria:
                # Try to evaluate - if it fails, we can't handle
                criterion.evaluate(context)
            return True
        except Exception:
            return False

    async def _evaluate_criteria(self, context: NodeExecutionContext) -> Dict[str, Any]:
        """
        Evaluate all decision criteria and select the best path.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, Any]
            Decision result with reasoning
        """
        # Evaluate each criterion
        criterion_scores: Dict[str, Any] = {}
        for criterion in self.decision_criteria:
            try:
                score = criterion.evaluate(context)
                criterion_scores[criterion.name] = {
                    "score": score,
                    "weight": criterion.weight,
                    "threshold": criterion.threshold,
                    "passed": score >= criterion.threshold,
                }
            except Exception as e:
                # Handle criteria evaluation failure gracefully
                criterion_scores[criterion.name] = {
                    "score": 0.0,  # Failed criteria get score 0
                    "weight": criterion.weight,
                    "threshold": criterion.threshold,
                    "passed": False,
                    "error": str(e),
                }

        # Calculate weighted scores for each path
        path_scores = {}
        for path_name in self.paths:
            # Simple scoring: sum of weighted scores for passed criteria
            # In real implementation, this could be more sophisticated
            total_score = sum(
                scores["score"] * scores["weight"]
                for scores in criterion_scores.values()
                if scores["passed"]
            )
            path_scores[path_name] = total_score

        # Select the path with highest score
        selected_path = max(path_scores, key=lambda k: path_scores[k])
        confidence_score = path_scores[selected_path] / sum(
            c.weight for c in self.decision_criteria
        )

        # Build result
        return {
            "selected_path": selected_path,
            "selected_agents": self.paths[selected_path],
            "confidence": min(confidence_score, 1.0),  # Cap at 1.0
            "alternatives": [p for p in self.paths if p != selected_path],
            "reasoning": {
                "criterion_scores": criterion_scores,
                "path_scores": path_scores,
                "decision_basis": f"Selected '{selected_path}' with score {path_scores[selected_path]:.2f}",
            },
        }
