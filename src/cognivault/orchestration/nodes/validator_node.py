"""
Validator Node Implementation for CogniVault.

This module implements the ValidatorNode class which handles quality
validation and gating in the advanced node execution system.
"""

from typing import Dict, List, Any, Optional, Callable, cast
from dataclasses import dataclass  # Still needed for ValidationReport
from enum import Enum
import asyncio

from pydantic import BaseModel, Field, ConfigDict
from cognivault.agents.metadata import AgentMetadata
from cognivault.events import emit_validation_completed
from .base_advanced_node import BaseAdvancedNode, NodeExecutionContext


class NodeValidationResult(Enum):
    """Possible validation results."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class ValidationCriteria(BaseModel):
    """
    Represents a single validation criterion.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    name: str = Field(
        ...,
        description="Name/identifier of the validation criterion",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "output_quality_check"},
    )
    validator: Callable[[Dict[str, Any]], bool] = Field(
        ..., description="Function that performs the validation logic"
    )
    weight: float = Field(
        default=1.0,
        description="Weight/importance of this criterion in overall validation",
        ge=0.0,
        le=10.0,
        json_schema_extra={"example": 1.0},
    )
    required: bool = Field(
        default=True,
        description="Whether this criterion must pass for overall validation to succeed",
    )
    error_message: str = Field(
        default="",
        description="Custom error message when validation fails",
        max_length=500,
        json_schema_extra={"example": "Output quality is below acceptable threshold"},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For Callable validator function
    )

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data against this criterion."""
        return self.validator(data)


class ValidationReport(BaseModel):
    """
    Detailed validation report.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    result: NodeValidationResult = Field(
        ...,
        description="Overall validation result",
        json_schema_extra={"example": "pass"},
    )
    quality_score: float = Field(
        ...,
        description="Calculated quality score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    criteria_results: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Results from each validation criterion",
        json_schema_extra={"example": {"criterion_1": {"passed": True, "weight": 1.0}}},
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of improvement recommendations",
        max_length=100,
        json_schema_extra={"example": ["Fix required criterion 'content_check'"]},
    )
    validation_time_ms: float = Field(
        ...,
        description="Time taken for validation in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 125.5},
    )
    total_criteria: int = Field(
        ...,
        description="Total number of validation criteria",
        ge=0,
        json_schema_extra={"example": 5},
    )
    passed_criteria: int = Field(
        ...,
        description="Number of criteria that passed",
        ge=0,
        json_schema_extra={"example": 4},
    )
    failed_criteria: int = Field(
        ...,
        description="Number of criteria that failed",
        ge=0,
        json_schema_extra={"example": 1},
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings",
        max_length=100,
        json_schema_extra={"example": ["Optional criterion failed"]},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate of validation criteria."""
        if self.total_criteria == 0:
            return 0.0
        return self.passed_criteria / self.total_criteria


class ValidatorNode(BaseAdvancedNode):
    """
    Quality validation and gating node.

    This node validates outputs against configurable criteria and
    provides quality gates for workflow execution.
    """

    def __init__(
        self,
        metadata: AgentMetadata,
        node_name: str,
        validation_criteria: List[ValidationCriteria],
        quality_threshold: float = 0.8,
        required_criteria_pass_rate: float = 1.0,
        allow_warnings: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize the ValidatorNode.

        Parameters
        ----------
        metadata : AgentMetadata
            The agent metadata containing multi-axis classification
        node_name : str
            Unique name for this node instance
        validation_criteria : List[ValidationCriteria]
            List of criteria to validate against
        quality_threshold : float
            Minimum quality score to pass validation
        required_criteria_pass_rate : float
            Minimum pass rate for required criteria (0.0 to 1.0)
        allow_warnings : bool
            Whether to allow warnings without failing
        strict_mode : bool
            If True, any failed criterion fails the entire validation
        """
        super().__init__(metadata, node_name)

        if self.execution_pattern != "validator":
            raise ValueError(
                f"ValidatorNode requires execution_pattern='validator', "
                f"got '{self.execution_pattern}'"
            )

        if not validation_criteria:
            raise ValueError("ValidatorNode requires at least one validation criterion")

        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError(
                f"quality_threshold must be between 0.0 and 1.0, got {quality_threshold}"
            )

        if not 0.0 <= required_criteria_pass_rate <= 1.0:
            raise ValueError(
                f"required_criteria_pass_rate must be between 0.0 and 1.0, got {required_criteria_pass_rate}"
            )

        self.validation_criteria = validation_criteria
        self.quality_threshold = quality_threshold
        self.required_criteria_pass_rate = required_criteria_pass_rate
        self.allow_warnings = allow_warnings
        self.strict_mode = strict_mode

    async def execute(self, context: NodeExecutionContext) -> Dict[str, Any]:
        """
        Execute the validation logic and assess quality.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, Any]
            Validation result with quality assessment and recommendations
        """
        # Pre-execution setup
        await self.pre_execute(context)

        # Validate context
        validation_errors = self.validate_context(context)
        if validation_errors:
            raise ValueError(
                f"Context validation failed: {', '.join(validation_errors)}"
            )

        # Get data to validate
        data_to_validate = await self._extract_validation_data(context)

        # Perform validation
        validation_report = await self._perform_validation(data_to_validate)

        # Emit validation event
        await emit_validation_completed(
            workflow_id=context.workflow_id,
            validation_result=validation_report.result.value,
            quality_score=validation_report.quality_score,
            validation_criteria=[c.name for c in self.validation_criteria],
            recommendations=validation_report.recommendations,
            validation_time_ms=validation_report.validation_time_ms,
            correlation_id=context.correlation_id,
        )

        # Create result
        result = {
            "validation_result": validation_report.result.value,
            "quality_score": validation_report.quality_score,
            "success_rate": validation_report.success_rate,
            "criteria_results": validation_report.criteria_results,
            "recommendations": validation_report.recommendations,
            "validation_time_ms": validation_report.validation_time_ms,
            "total_criteria": validation_report.total_criteria,
            "passed_criteria": validation_report.passed_criteria,
            "failed_criteria": validation_report.failed_criteria,
            "warnings": validation_report.warnings,
            "passed": validation_report.result
            in [NodeValidationResult.PASS, NodeValidationResult.WARNING],
            "validated_data": data_to_validate,
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
            # Check if we have data to validate
            if not context.available_inputs:
                return False

            # Check if we have at least one input with data
            has_data = False
            for source, data in context.available_inputs.items():
                if isinstance(data, dict) and data:
                    has_data = True
                    break

            return has_data
        except Exception:
            return False

    async def _extract_validation_data(
        self, context: NodeExecutionContext
    ) -> Dict[str, Any]:
        """
        Extract data to validate from the context.

        Parameters
        ----------
        context : NodeExecutionContext
            The execution context

        Returns
        -------
        Dict[str, Any]
            Data to validate
        """
        # For now, we'll validate the most recent or highest quality input
        # In a real implementation, this could be more sophisticated

        if not context.available_inputs:
            return {}

        # Find the best input to validate
        best_input: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for source, data in context.available_inputs.items():
            if isinstance(data, dict):
                # Use quality score or confidence as ranking
                score = cast(
                    float, data.get("quality_score", data.get("confidence", 0.0))
                )
                if score > best_score:
                    best_score = score
                    best_input = data

        return best_input or {}

    async def _perform_validation(self, data: Dict[str, Any]) -> ValidationReport:
        """
        Perform validation against all criteria.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to validate

        Returns
        -------
        ValidationReport
            Detailed validation report
        """
        start_time = asyncio.get_event_loop().time()

        criteria_results = {}
        passed_criteria = 0
        failed_criteria = 0
        warnings = []
        recommendations = []

        # Validate each criterion
        for criterion in self.validation_criteria:
            try:
                passed = criterion.validate_data(data)
                criteria_results[criterion.name] = {
                    "passed": passed,
                    "weight": criterion.weight,
                    "required": criterion.required,
                    "error_message": criterion.error_message if not passed else "",
                }

                if passed:
                    passed_criteria += 1
                else:
                    failed_criteria += 1
                    if criterion.required:
                        if criterion.error_message:
                            recommendations.append(
                                f"Fix required criterion '{criterion.name}': {criterion.error_message}"
                            )
                        else:
                            recommendations.append(
                                f"Fix required criterion '{criterion.name}'"
                            )
                    else:
                        warnings.append(f"Optional criterion '{criterion.name}' failed")

            except Exception as e:
                # Criterion validation failed due to error
                failed_criteria += 1
                error_msg = (
                    f"Validation error in criterion '{criterion.name}': {str(e)}"
                )
                warnings.append(error_msg)
                criteria_results[criterion.name] = {
                    "passed": False,
                    "weight": criterion.weight,
                    "required": criterion.required,
                    "error_message": error_msg,
                }

        end_time = asyncio.get_event_loop().time()
        validation_time_ms = (end_time - start_time) * 1000

        # Calculate quality score (weighted average)
        total_weight = sum(c.weight for c in self.validation_criteria)
        if total_weight > 0:
            passed_weight = sum(
                c.weight
                for c in self.validation_criteria
                if c.name in criteria_results and criteria_results[c.name]["passed"]
            )
            quality_score = passed_weight / total_weight
        else:
            quality_score = 0.0

        # Determine validation result
        result = self._determine_validation_result(
            quality_score, passed_criteria, failed_criteria, warnings, criteria_results
        )

        return ValidationReport(
            result=result,
            quality_score=quality_score,
            criteria_results=criteria_results,
            recommendations=recommendations,
            validation_time_ms=validation_time_ms,
            total_criteria=len(self.validation_criteria),
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            warnings=warnings,
        )

    def _determine_validation_result(
        self,
        quality_score: float,
        passed_criteria: int,
        failed_criteria: int,
        warnings: List[str],
        criteria_results: Dict[str, Dict[str, Any]],
    ) -> NodeValidationResult:
        """
        Determine the overall validation result.

        Parameters
        ----------
        quality_score : float
            Calculated quality score
        passed_criteria : int
            Number of passed criteria
        failed_criteria : int
            Number of failed criteria
        warnings : List[str]
            List of warnings
        criteria_results : Dict[str, Dict[str, Any]]
            Results from each validation criterion

        Returns
        -------
        NodeValidationResult
            Overall validation result
        """
        # Check required criteria pass rate
        required_criteria = [c for c in self.validation_criteria if c.required]
        if required_criteria:
            required_passed = sum(
                1
                for c in required_criteria
                if c.name in criteria_results and criteria_results[c.name]["passed"]
            )
            required_pass_rate = required_passed / len(required_criteria)

            if required_pass_rate < self.required_criteria_pass_rate:
                return NodeValidationResult.FAIL

        # Strict mode: any failure is a failure
        if self.strict_mode and failed_criteria > 0:
            return NodeValidationResult.FAIL

        # Check quality threshold
        if quality_score < self.quality_threshold:
            return NodeValidationResult.FAIL

        # Check if we have warnings
        if warnings and not self.allow_warnings:
            return NodeValidationResult.FAIL

        # Success with warnings
        if warnings and self.allow_warnings:
            return NodeValidationResult.WARNING

        # Full success
        return NodeValidationResult.PASS
