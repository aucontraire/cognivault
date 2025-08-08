"""
Terminator Node Implementation for CogniVault.

This module implements the TerminatorNode class which handles early
termination based on confidence thresholds and completion criteria.
"""

from typing import Dict, List, Any, Optional, Callable, Union, cast
from dataclasses import dataclass
from enum import Enum
import asyncio

from pydantic import BaseModel, Field, ConfigDict

from cognivault.agents.metadata import AgentMetadata
from cognivault.events import emit_termination_triggered
from .base_advanced_node import BaseAdvancedNode, NodeExecutionContext


class TerminationReason(Enum):
    """Possible termination reasons."""

    CONFIDENCE_THRESHOLD = "confidence_threshold"
    QUALITY_THRESHOLD = "quality_threshold"
    RESOURCE_LIMIT = "resource_limit"
    TIME_LIMIT = "time_limit"
    COMPLETION_CRITERIA = "completion_criteria"
    MANUAL_TRIGGER = "manual_trigger"


class TerminationCriteria(BaseModel):
    """
    Represents a single termination criterion.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    name: str = Field(
        ...,
        description="Name/identifier of the termination criterion",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "confidence_check"},
    )
    evaluator: Callable[[Dict[str, Any]], bool] = Field(
        ...,
        description="Function that evaluates the criterion against input data",
    )
    threshold: float = Field(
        default=0.0,
        description="Threshold value for criterion evaluation",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.95},
    )
    weight: float = Field(
        default=1.0,
        description="Weight/importance of this criterion in termination decision",
        ge=0.0,
        le=10.0,
        json_schema_extra={"example": 2.0},
    )
    required: bool = Field(
        default=True,
        description="Whether this criterion is required for termination",
        json_schema_extra={"example": True},
    )
    description: str = Field(
        default="",
        description="Human-readable description of the criterion",
        max_length=500,
        json_schema_extra={"example": "Check if confidence threshold is met"},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For Callable evaluator function
    )

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate data against this criterion."""
        return self.evaluator(data)


class TerminationReport(BaseModel):
    """
    Detailed termination report.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    should_terminate: bool = Field(
        ...,
        description="Whether termination is recommended",
        json_schema_extra={"example": True},
    )
    termination_reason: TerminationReason = Field(
        ...,
        description="Primary reason for termination decision",
        json_schema_extra={"example": "confidence_threshold"},
    )
    confidence_score: float = Field(
        ...,
        description="Overall confidence score for termination decision",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.95},
    )
    criteria_results: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Detailed results for each termination criterion",
        json_schema_extra={"example": {"criterion1": {"met": True, "score": 0.9}}},
    )
    resource_savings: Dict[str, float] = Field(
        ...,
        description="Estimated resource savings from early termination",
        json_schema_extra={"example": {"cpu_time_ms": 1000.0, "memory_mb": 50.0}},
    )
    completion_time_ms: float = Field(
        ...,
        description="Time taken to complete termination evaluation in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 125.5},
    )
    met_criteria: List[str] = Field(
        ...,
        description="Names of criteria that were successfully met",
        json_schema_extra={"example": ["confidence_check", "quality_check"]},
    )
    unmet_criteria: List[str] = Field(
        ...,
        description="Names of criteria that were not met",
        json_schema_extra={"example": ["resource_limit"]},
    )
    termination_message: str = Field(
        ...,
        description="Human-readable message explaining the termination decision",
        min_length=1,
        max_length=1000,
        json_schema_extra={"example": "Confidence threshold met with score 0.95"},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class TerminatorNode(BaseAdvancedNode):
    """
    Early termination and completion node.

    This node evaluates termination criteria and can halt workflow
    execution when confidence thresholds are met or resource limits reached.
    """

    def __init__(
        self,
        metadata: AgentMetadata,
        node_name: str,
        termination_criteria: List[TerminationCriteria],
        confidence_threshold: float = 0.95,
        quality_threshold: float = 0.9,
        resource_limit_threshold: float = 0.8,
        time_limit_ms: Optional[float] = None,
        allow_partial_completion: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize the TerminatorNode.

        Parameters
        ----------
        metadata : AgentMetadata
            The agent metadata containing multi-axis classification
        node_name : str
            Unique name for this node instance
        termination_criteria : List[TerminationCriteria]
            List of criteria to evaluate for termination
        confidence_threshold : float
            Minimum confidence score to consider termination
        quality_threshold : float
            Minimum quality score to consider termination
        resource_limit_threshold : float
            Resource usage threshold to trigger termination
        time_limit_ms : Optional[float]
            Maximum execution time before termination
        allow_partial_completion : bool
            Whether to allow termination with partial results
        strict_mode : bool
            If True, all criteria must be met for termination
        """
        super().__init__(metadata, node_name)

        if self.execution_pattern != "terminator":
            raise ValueError(
                f"TerminatorNode requires execution_pattern='terminator', "
                f"got '{self.execution_pattern}'"
            )

        if not termination_criteria:
            raise ValueError(
                "TerminatorNode requires at least one termination criterion"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )

        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError(
                f"quality_threshold must be between 0.0 and 1.0, got {quality_threshold}"
            )

        if not 0.0 <= resource_limit_threshold <= 1.0:
            raise ValueError(
                f"resource_limit_threshold must be between 0.0 and 1.0, got {resource_limit_threshold}"
            )

        if time_limit_ms is not None and time_limit_ms <= 0:
            raise ValueError(f"time_limit_ms must be positive, got {time_limit_ms}")

        self.termination_criteria = termination_criteria
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold
        self.resource_limit_threshold = resource_limit_threshold
        self.time_limit_ms = time_limit_ms
        self.allow_partial_completion = allow_partial_completion
        self.strict_mode = strict_mode

    async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
        """
        Execute the termination logic and evaluate completion criteria.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, Any]
            Termination result with decision and resource savings
        """
        # Pre-execution setup
        await self.pre_execute(context)

        # Validate context
        validation_errors = self.validate_context(context)
        if validation_errors:
            raise ValueError(
                f"Context validation failed: {', '.join(validation_errors)}"
            )

        # Evaluate termination criteria
        termination_report = await self._evaluate_termination_criteria(context)

        # Calculate resource savings if termination occurs
        resource_savings = self._calculate_resource_savings(context, termination_report)

        # Emit termination event if termination is recommended
        if termination_report.should_terminate:
            await emit_termination_triggered(
                workflow_id=context.workflow_id,
                termination_reason=termination_report.termination_reason.value,
                confidence_score=termination_report.confidence_score,
                threshold=self.confidence_threshold,
                resources_saved=resource_savings,
                correlation_id=context.correlation_id,
            )

        # Create result
        result = {
            "should_terminate": termination_report.should_terminate,
            "termination_reason": termination_report.termination_reason.value,
            "confidence_score": termination_report.confidence_score,
            "criteria_results": termination_report.criteria_results,
            "resource_savings": resource_savings,
            "completion_time_ms": termination_report.completion_time_ms,
            "met_criteria": termination_report.met_criteria,
            "unmet_criteria": termination_report.unmet_criteria,
            "termination_message": termination_report.termination_message,
            "allow_partial_completion": self.allow_partial_completion,
            "recommended_action": (
                "terminate" if termination_report.should_terminate else "continue"
            ),
        }

        # Post-execution cleanup
        await self.post_execute(context, result)

        return result

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
            # Check if we have sufficient execution history
            if len(context.execution_path) < 2:
                return False

            # Check if we have available inputs or results to evaluate
            if not context.available_inputs and not hasattr(
                context, "intermediate_results"
            ):
                return False

            # Check if we have resource usage information
            if not context.resource_usage:
                return False

            return True
        except Exception:
            return False

    async def _evaluate_termination_criteria(
        self, context: NodeExecutionContext
    ) -> TerminationReport:
        """
        Evaluate all termination criteria.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        TerminationReport
            Detailed termination evaluation report
        """
        start_time = asyncio.get_event_loop().time()

        # Gather data for evaluation
        evaluation_data = await self._prepare_evaluation_data(context)

        criteria_results = {}
        met_criteria = []
        unmet_criteria = []

        # Evaluate each criterion
        for criterion in self.termination_criteria:
            try:
                met = criterion.evaluate(evaluation_data)
                criteria_results[criterion.name] = {
                    "met": met,
                    "threshold": criterion.threshold,
                    "weight": criterion.weight,
                    "required": criterion.required,
                    "description": criterion.description,
                }

                if met:
                    met_criteria.append(criterion.name)
                else:
                    unmet_criteria.append(criterion.name)

            except Exception as e:
                # Criterion evaluation failed
                criteria_results[criterion.name] = {
                    "met": False,
                    "threshold": criterion.threshold,
                    "weight": criterion.weight,
                    "required": criterion.required,
                    "description": criterion.description,
                    "error": str(e),
                }
                unmet_criteria.append(criterion.name)

        end_time = asyncio.get_event_loop().time()
        completion_time_ms = (end_time - start_time) * 1000

        # Determine termination decision
        should_terminate, reason, confidence_score, message = (
            self._make_termination_decision(
                evaluation_data, criteria_results, met_criteria, unmet_criteria
            )
        )

        return TerminationReport(
            should_terminate=should_terminate,
            termination_reason=reason,
            confidence_score=confidence_score,
            criteria_results=criteria_results,
            resource_savings={},  # Will be calculated separately
            completion_time_ms=completion_time_ms,
            met_criteria=met_criteria,
            unmet_criteria=unmet_criteria,
            termination_message=message,
        )

    async def _prepare_evaluation_data(
        self, context: NodeExecutionContext
    ) -> Dict[str, Any]:
        """
        Prepare data for termination criteria evaluation.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, Any]
            Data prepared for evaluation
        """
        # Get best available input for evaluation
        best_input = {}
        if context.available_inputs:
            best_quality = -1.0
            for source, data in context.available_inputs.items():
                if isinstance(data, dict):
                    quality = cast(
                        float, data.get("quality_score", data.get("confidence", 0.0))
                    )
                    if quality > best_quality:
                        best_quality = quality
                        best_input = data

        # Calculate execution progress
        execution_progress = len(context.execution_path) / max(
            len(context.execution_path) + 3, 1
        )

        # Resource usage metrics
        resource_usage = context.resource_usage or {}

        return {
            "confidence": best_input.get("confidence", 0.0),
            "quality_score": best_input.get("quality_score", 0.0),
            "execution_progress": execution_progress,
            "resource_usage": resource_usage,
            "execution_path_length": len(context.execution_path),
            "available_inputs_count": len(context.available_inputs),
            "best_input": best_input,
        }

    def _make_termination_decision(
        self,
        evaluation_data: Dict[str, Any],
        criteria_results: Dict[str, Dict[str, Any]],
        met_criteria: List[str],
        unmet_criteria: List[str],
    ) -> tuple[bool, TerminationReason, float, str]:
        """
        Make the final termination decision.

        Parameters
        ----------
        evaluation_data : Dict[str, Any]
            Prepared evaluation data
        criteria_results : Dict[str, Dict[str, Any]]
            Results from criteria evaluation
        met_criteria : List[str]
            Names of met criteria
        unmet_criteria : List[str]
            Names of unmet criteria

        Returns
        -------
        tuple[bool, TerminationReason, float, str]
            (should_terminate, reason, confidence_score, message)
        """
        confidence = evaluation_data.get("confidence", 0.0)
        quality_score = evaluation_data.get("quality_score", 0.0)
        resource_usage = evaluation_data.get("resource_usage", {})

        # Check confidence threshold
        if confidence >= self.confidence_threshold:
            return (
                True,
                TerminationReason.CONFIDENCE_THRESHOLD,
                confidence,
                f"Confidence threshold ({self.confidence_threshold}) met with score {confidence:.3f}",
            )

        # Check quality threshold
        if quality_score >= self.quality_threshold:
            return (
                True,
                TerminationReason.QUALITY_THRESHOLD,
                quality_score,
                f"Quality threshold ({self.quality_threshold}) met with score {quality_score:.3f}",
            )

        # Check resource limits
        cpu_usage = resource_usage.get("cpu_usage", 0.0)
        memory_usage = resource_usage.get("memory_usage", 0.0)
        if (
            cpu_usage >= self.resource_limit_threshold
            or memory_usage >= self.resource_limit_threshold
        ):
            return (
                True,
                TerminationReason.RESOURCE_LIMIT,
                max(cpu_usage, memory_usage),
                f"Resource limit threshold ({self.resource_limit_threshold}) exceeded",
            )

        # Check completion criteria
        if self.strict_mode:
            # All criteria must be met
            if not unmet_criteria:
                return (
                    True,
                    TerminationReason.COMPLETION_CRITERIA,
                    1.0,
                    "All termination criteria met in strict mode",
                )
        else:
            # Check if we have enough criteria met
            required_met = sum(
                1
                for name in met_criteria
                if any(c.name == name and c.required for c in self.termination_criteria)
            )
            total_required = sum(1 for c in self.termination_criteria if c.required)

            if total_required > 0 and required_met / total_required >= 0.5:
                completion_ratio = required_met / total_required
                return (
                    True,
                    TerminationReason.COMPLETION_CRITERIA,
                    completion_ratio,
                    f"Sufficient completion criteria met ({required_met}/{total_required})",
                )

        # No termination criteria met
        return (
            False,
            TerminationReason.CONFIDENCE_THRESHOLD,
            confidence,
            "No termination criteria met, continue execution",
        )

    def _calculate_resource_savings(
        self, context: NodeExecutionContext, report: TerminationReport
    ) -> Dict[str, float]:
        """
        Calculate estimated resource savings from early termination.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context
        report : TerminationReport
            Termination evaluation report

        Returns
        -------
        Dict[str, float]
            Estimated resource savings
        """
        if not report.should_terminate:
            return {"cpu_time_ms": 0.0, "memory_mb": 0.0, "estimated_cost": 0.0}

        # Estimate remaining work
        remaining_steps = self._estimate_remaining_steps(context)
        current_usage = context.resource_usage or {}

        # Calculate estimated savings
        avg_cpu_per_step = current_usage.get("cpu_usage", 0.0) / max(
            len(context.execution_path), 1
        )
        avg_memory_per_step = current_usage.get("memory_usage", 0.0) / max(
            len(context.execution_path), 1
        )

        estimated_cpu_savings = avg_cpu_per_step * remaining_steps
        estimated_memory_savings = avg_memory_per_step * remaining_steps
        estimated_cost_savings = estimated_cpu_savings * 0.001  # Simple cost model

        return {
            "cpu_time_ms": estimated_cpu_savings,
            "memory_mb": estimated_memory_savings,
            "estimated_cost": estimated_cost_savings,
        }

    def _estimate_remaining_steps(self, context: NodeExecutionContext) -> int:
        """
        Estimate remaining execution steps.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        int
            Estimated remaining steps
        """
        # Simple heuristic: assume 2-4 more steps for typical workflows
        current_steps = len(context.execution_path)
        if current_steps < 2:
            return 3
        elif current_steps < 4:
            return 2
        else:
            return 1
