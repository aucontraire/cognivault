"""
Base validation models for CogniVault validation framework.

This module provides the foundation for unified validation across all CogniVault
domains (semantic, workflow, pattern) while maintaining backward compatibility.
"""

from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class ValidationSeverity(Enum):
    """Unified severity levels for validation issues across all domains.

    Consolidates:
    - Semantic: INFO, WARNING, ERROR
    - Workflow: INFO, WARNING, ERROR, STYLE (with int levels 1-10)
    - Pattern: PASS, WARN, FAIL, ERROR
    """

    # Core severity levels (shared across all domains)
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    # Workflow domain specific
    STYLE = "style"

    # Pattern domain specific
    FAIL = "fail"
    PASS = "pass"

    # Alias for compatibility
    WARN = "warning"  # Maps WARN to WARNING for pattern domain

    @classmethod
    def from_string(cls, value: str) -> "ValidationSeverity":
        """Convert string to ValidationSeverity with alias handling."""
        if value.lower() == "warn":
            return cls.WARNING
        return cls(value.lower())

    def to_pattern_result(self) -> str:
        """Convert to pattern validation result format."""
        if self == ValidationSeverity.WARNING:
            return "warn"
        return str(self.value)


class ValidationIssue(BaseModel):
    """Base validation issue for all CogniVault validation domains.

    This model provides a unified interface for validation issues while
    supporting domain-specific extensions through inheritance.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=False,  # Keep enum objects for type safety
    )

    severity: ValidationSeverity = Field(
        ..., description="Severity level of the validation issue"
    )
    message: str = Field(..., description="Human-readable description of the issue")
    location: Optional[str] = Field(
        default=None, description="Location of the issue (file, line, component, etc.)"
    )
    suggestion: Optional[str] = Field(
        default=None, description="Suggested fix or improvement for the issue"
    )
    code: Optional[str] = Field(
        default=None,
        description="Error/issue code identifier for programmatic handling",
    )
    category: Optional[str] = Field(
        default=None, description="Category or type of validation issue"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "code": self.code,
            "category": self.category,
        }

    def is_error(self) -> bool:
        """Check if this issue represents an error."""
        return self.severity in (ValidationSeverity.ERROR, ValidationSeverity.FAIL)

    def is_warning(self) -> bool:
        """Check if this issue represents a warning."""
        return self.severity == ValidationSeverity.WARNING

    def is_info(self) -> bool:
        """Check if this issue represents informational content."""
        return self.severity in (ValidationSeverity.INFO, ValidationSeverity.STYLE)


class SemanticValidationIssue(ValidationIssue):
    """Validation issue specific to semantic validation domain.

    Extends base ValidationIssue with agent-specific context.
    Used in langgraph_backend.semantic_validation module.
    """

    agent: Optional[str] = Field(
        default=None, description="Agent associated with the validation issue"
    )

    def to_legacy_dataclass(self) -> dict[str, Any]:
        """Convert to legacy dataclass format for backward compatibility."""
        return {
            "severity": self.severity,  # Keep as enum for dataclass compatibility
            "message": self.message,
            "agent": self.agent,
            "suggestion": self.suggestion,
            "code": self.code,
        }


class WorkflowValidationIssue(ValidationIssue):
    """Validation issue specific to workflow validation domain.

    Extends base ValidationIssue with severity levels and rule tracking.
    Used in workflows.validators module.
    """

    severity_level: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Numeric severity level (1-10, 10 being most severe)",
    )
    rule_id: Optional[str] = Field(
        default=None, description="Unique identifier for the validation rule"
    )
    issue_type: Optional[str] = Field(
        default=None,
        description="Type classification of the issue (maps to ValidationIssueType)",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with automatic issue_type mapping from severity."""
        if "issue_type" not in data and "severity" in data:
            # Auto-map severity to issue_type for backward compatibility
            severity = data["severity"]
            if isinstance(severity, ValidationSeverity):
                data["issue_type"] = severity.value
        super().__init__(**data)

    def to_legacy_format(self) -> dict[str, Any]:
        """Convert to legacy workflow validation format."""
        return {
            "issue_type": self.issue_type or self.severity.value,
            "severity": self.severity_level or self._default_severity_level(),
            "message": self.message,
            "location": self.location or "",
            "suggestion": self.suggestion,
            "rule_id": self.rule_id or "",
        }

    def _default_severity_level(self) -> int:
        """Map severity enum to default numeric level."""
        severity_map = {
            ValidationSeverity.ERROR: 9,
            ValidationSeverity.WARNING: 6,
            ValidationSeverity.INFO: 3,
            ValidationSeverity.STYLE: 2,
            ValidationSeverity.FAIL: 10,
            ValidationSeverity.PASS: 1,
        }
        return severity_map.get(self.severity, 5)

    def get_numeric_severity(self) -> int:
        """Get numeric severity level for this issue."""
        return self.severity_level or self._default_severity_level()


class PatternValidationIssue(ValidationIssue):
    """Validation issue specific to pattern validation domain.

    Maps directly to base ValidationIssue with pattern-specific semantics.
    Used in diagnostics.pattern_validator module.
    """

    def to_legacy_format(self) -> dict[str, Any]:
        """Convert to legacy pattern validation format."""
        # Map severity to PatternValidationResult values
        severity_map = {
            ValidationSeverity.PASS: "pass",
            ValidationSeverity.WARNING: "warn",
            ValidationSeverity.FAIL: "fail",
            ValidationSeverity.ERROR: "error",
            ValidationSeverity.INFO: "pass",  # INFO maps to PASS in pattern domain
            ValidationSeverity.STYLE: "pass",  # STYLE maps to PASS in pattern domain
        }

        return {
            "level": severity_map.get(self.severity, self.severity.value),
            "category": self.category or "",
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "code": self.code,
        }

    @property
    def level(self) -> Any:
        """Backward compatibility property for accessing severity as level.

        Returns the severity mapped to PatternValidationResult enum values for
        compatibility with existing test code.
        """
        # Import here to avoid circular imports
        from cognivault.diagnostics.pattern_validator import PatternValidationResult

        severity_map = {
            ValidationSeverity.PASS: PatternValidationResult.PASS,
            ValidationSeverity.WARNING: PatternValidationResult.WARN,
            ValidationSeverity.FAIL: PatternValidationResult.FAIL,
            ValidationSeverity.ERROR: PatternValidationResult.ERROR,
            ValidationSeverity.INFO: PatternValidationResult.PASS,  # INFO maps to PASS in pattern domain
            ValidationSeverity.STYLE: PatternValidationResult.PASS,  # STYLE maps to PASS in pattern domain
        }
        return severity_map.get(self.severity, PatternValidationResult.ERROR)
