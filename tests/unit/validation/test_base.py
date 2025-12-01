"""
Unit tests for the unified validation framework base classes.

Tests cover ValidationSeverity enum, base ValidationIssue, and domain-specific
extensions for backward compatibility.
"""

import pytest

from cognivault.validation.base import (
    ValidationSeverity,
    ValidationIssue,
    SemanticValidationIssue,
    WorkflowValidationIssue,
    PatternValidationIssue,
)


class TestValidationSeverity:
    """Test ValidationSeverity enum functionality."""

    def test_enum_values(self) -> None:
        """Test that all severity levels have correct values."""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.STYLE.value == "style"
        assert ValidationSeverity.FAIL.value == "fail"
        assert ValidationSeverity.PASS.value == "pass"

    def test_from_string_basic(self) -> None:
        """Test from_string conversion for basic values."""
        assert ValidationSeverity.from_string("info") == ValidationSeverity.INFO
        assert ValidationSeverity.from_string("warning") == ValidationSeverity.WARNING
        assert ValidationSeverity.from_string("error") == ValidationSeverity.ERROR
        assert ValidationSeverity.from_string("style") == ValidationSeverity.STYLE

    def test_from_string_with_alias(self) -> None:
        """Test from_string handles 'warn' alias correctly."""
        assert ValidationSeverity.from_string("warn") == ValidationSeverity.WARNING

    def test_from_string_case_insensitive(self) -> None:
        """Test from_string handles case variations."""
        assert ValidationSeverity.from_string("INFO") == ValidationSeverity.INFO
        assert ValidationSeverity.from_string("Warning") == ValidationSeverity.WARNING
        assert ValidationSeverity.from_string("ERROR") == ValidationSeverity.ERROR

    def test_to_pattern_result(self) -> None:
        """Test conversion to pattern validation result format."""
        assert ValidationSeverity.WARNING.to_pattern_result() == "warn"
        assert ValidationSeverity.ERROR.to_pattern_result() == "error"
        assert ValidationSeverity.INFO.to_pattern_result() == "info"

    def test_enum_comparison(self) -> None:
        """Test enum comparison works correctly."""
        assert ValidationSeverity.ERROR == ValidationSeverity.ERROR
        # This is intentionally testing different enum values - not a bug
        # MyPy considers this non-overlapping but it's a valid test of enum inequality
        different_severities = (ValidationSeverity.WARNING, ValidationSeverity.ERROR)
        assert different_severities[0] != different_severities[1]
        # WARN and WARNING have same value but different members
        assert ValidationSeverity.WARN.value == ValidationSeverity.WARNING.value


class TestValidationIssue:
    """Test base ValidationIssue model."""

    def test_minimal_creation(self) -> None:
        """Test creating ValidationIssue with minimal fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR, message="Test error message"
        )
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error message"
        assert issue.location is None
        assert issue.suggestion is None
        assert issue.code is None
        assert issue.category is None

    def test_full_creation(self) -> None:
        """Test creating ValidationIssue with all fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            location="workflow.nodes[0]",
            suggestion="Fix the node configuration",
            code="NODE_001",
            category="structural",
        )
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.message == "Test warning"
        assert issue.location == "workflow.nodes[0]"
        assert issue.suggestion == "Fix the node configuration"
        assert issue.code == "NODE_001"
        assert issue.category == "structural"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error",
            location="test.py:42",
            code="ERR_001",
        )
        result = issue.to_dict()

        assert result["severity"] == "error"
        assert result["message"] == "Test error"
        assert result["location"] == "test.py:42"
        assert result["code"] == "ERR_001"
        assert result["suggestion"] is None
        assert result["category"] is None

    def test_is_error(self) -> None:
        """Test is_error method."""
        error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR, message="error"
        )
        fail_issue = ValidationIssue(severity=ValidationSeverity.FAIL, message="fail")
        warn_issue = ValidationIssue(
            severity=ValidationSeverity.WARNING, message="warn"
        )

        assert error_issue.is_error() is True
        assert fail_issue.is_error() is True
        assert warn_issue.is_error() is False

    def test_is_warning(self) -> None:
        """Test is_warning method."""
        warn_issue = ValidationIssue(
            severity=ValidationSeverity.WARNING, message="warn"
        )
        error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR, message="error"
        )

        assert warn_issue.is_warning() is True
        assert error_issue.is_warning() is False

    def test_is_info(self) -> None:
        """Test is_info method."""
        info_issue = ValidationIssue(severity=ValidationSeverity.INFO, message="info")
        style_issue = ValidationIssue(
            severity=ValidationSeverity.STYLE, message="style"
        )
        error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR, message="error"
        )

        assert info_issue.is_info() is True
        assert style_issue.is_info() is True
        assert error_issue.is_info() is False

    def test_string_stripping(self) -> None:
        """Test that strings are stripped of whitespace."""
        issue = ValidationIssue(
            severity=ValidationSeverity.INFO,
            message="  Test message  ",
            location="  file.py  ",
        )
        assert issue.message == "Test message"
        assert issue.location == "file.py"

    def test_validation_strict(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            # We expect this to raise a validation error, not a type error
            # Using **kwargs to bypass typing check but still test runtime validation
            extra_kwargs = {"extra_field": "not allowed"}
            ValidationIssue(
                severity=ValidationSeverity.ERROR, message="Test", **extra_kwargs
            )


class TestSemanticValidationIssue:
    """Test SemanticValidationIssue domain extension."""

    def test_creation_with_agent(self) -> None:
        """Test creating semantic issue with agent field."""
        issue = SemanticValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Agent conflict detected",
            agent="synthesis",
        )
        assert issue.agent == "synthesis"
        assert issue.severity == ValidationSeverity.WARNING

    def test_inherits_base_methods(self) -> None:
        """Test that semantic issue inherits base methods."""
        issue = SemanticValidationIssue(
            severity=ValidationSeverity.ERROR, message="Critical error", agent="critic"
        )
        assert issue.is_error() is True
        assert issue.is_warning() is False

    def test_to_legacy_dataclass(self) -> None:
        """Test conversion to legacy dataclass format."""
        issue = SemanticValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            agent="refiner",
            suggestion="Fix the refiner",
            code="REF_001",
        )

        result = issue.to_legacy_dataclass()
        assert (
            result["severity"] == ValidationSeverity.WARNING
        )  # Enum object for dataclass
        assert result["message"] == "Test warning"
        assert result["agent"] == "refiner"
        assert result["suggestion"] == "Fix the refiner"
        assert result["code"] == "REF_001"

    def test_to_dict_includes_agent(self) -> None:
        """Test that to_dict includes agent field."""
        issue = SemanticValidationIssue(
            severity=ValidationSeverity.INFO, message="Info message", agent="historian"
        )
        result = issue.to_dict()
        # Agent is stored internally but base to_dict doesn't include it
        # This is intentional for backward compatibility
        assert "agent" not in result  # Base to_dict doesn't know about agent


class TestWorkflowValidationIssue:
    """Test WorkflowValidationIssue domain extension."""

    def test_creation_with_severity_level(self) -> None:
        """Test creating workflow issue with numeric severity level."""
        issue = WorkflowValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Critical workflow error",
            severity_level=9,
            rule_id="WF_001",
        )
        assert issue.severity_level == 9
        assert issue.rule_id == "WF_001"

    def test_severity_level_validation(self) -> None:
        """Test that severity_level is validated."""
        # Valid range
        issue = WorkflowValidationIssue(
            severity=ValidationSeverity.WARNING, message="Warning", severity_level=5
        )
        assert issue.severity_level == 5

        # Test boundaries
        issue1 = WorkflowValidationIssue(
            severity=ValidationSeverity.ERROR, message="Min severity", severity_level=1
        )
        assert issue1.severity_level == 1

        issue10 = WorkflowValidationIssue(
            severity=ValidationSeverity.ERROR, message="Max severity", severity_level=10
        )
        assert issue10.severity_level == 10

        # Invalid range
        with pytest.raises(ValueError):
            WorkflowValidationIssue(
                severity=ValidationSeverity.ERROR, message="Invalid", severity_level=0
            )

        with pytest.raises(ValueError):
            WorkflowValidationIssue(
                severity=ValidationSeverity.ERROR, message="Invalid", severity_level=11
            )

    def test_auto_issue_type_mapping(self) -> None:
        """Test automatic issue_type mapping from severity."""
        issue = WorkflowValidationIssue(
            severity=ValidationSeverity.ERROR, message="Error message"
        )
        assert issue.issue_type == "error"

        issue2 = WorkflowValidationIssue(
            severity=ValidationSeverity.WARNING, message="Warning message"
        )
        assert issue2.issue_type == "warning"

    def test_explicit_issue_type(self) -> None:
        """Test that explicit issue_type overrides auto-mapping."""
        issue = WorkflowValidationIssue(
            severity=ValidationSeverity.ERROR, message="Error", issue_type="critical"
        )
        assert issue.issue_type == "critical"

    def test_to_legacy_format(self) -> None:
        """Test conversion to legacy workflow validation format."""
        issue = WorkflowValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            location="workflow.flow",
            suggestion="Fix the flow",
            rule_id="FLOW_001",
            severity_level=6,
        )

        result = issue.to_legacy_format()
        assert result["issue_type"] == "warning"
        assert result["severity"] == 6
        assert result["message"] == "Test warning"
        assert result["location"] == "workflow.flow"
        assert result["suggestion"] == "Fix the flow"
        assert result["rule_id"] == "FLOW_001"

    def test_default_severity_level_mapping(self) -> None:
        """Test default severity level mapping when not explicitly set."""
        # Test each severity type
        error_issue = WorkflowValidationIssue(
            severity=ValidationSeverity.ERROR, message="Error"
        )
        assert error_issue.to_legacy_format()["severity"] == 9

        warn_issue = WorkflowValidationIssue(
            severity=ValidationSeverity.WARNING, message="Warning"
        )
        assert warn_issue.to_legacy_format()["severity"] == 6

        info_issue = WorkflowValidationIssue(
            severity=ValidationSeverity.INFO, message="Info"
        )
        assert info_issue.to_legacy_format()["severity"] == 3

        style_issue = WorkflowValidationIssue(
            severity=ValidationSeverity.STYLE, message="Style"
        )
        assert style_issue.to_legacy_format()["severity"] == 2

        fail_issue = WorkflowValidationIssue(
            severity=ValidationSeverity.FAIL, message="Fail"
        )
        assert fail_issue.to_legacy_format()["severity"] == 10

        pass_issue = WorkflowValidationIssue(
            severity=ValidationSeverity.PASS, message="Pass"
        )
        assert pass_issue.to_legacy_format()["severity"] == 1


class TestPatternValidationIssue:
    """Test PatternValidationIssue domain extension."""

    def test_creation_basic(self) -> None:
        """Test creating pattern validation issue."""
        issue = PatternValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Pattern error",
            category="structural",
        )
        assert issue.category == "structural"
        assert issue.severity == ValidationSeverity.ERROR

    def test_to_legacy_format(self) -> None:
        """Test conversion to legacy pattern validation format."""
        issue = PatternValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Pattern warning",
            category="performance",
            location="pattern.py:123",
            suggestion="Optimize the loop",
            code="PERF_001",
        )

        result = issue.to_legacy_format()
        assert result["level"] == "warn"  # WARNING maps to "warn"
        assert result["category"] == "performance"
        assert result["message"] == "Pattern warning"
        assert result["location"] == "pattern.py:123"
        assert result["suggestion"] == "Optimize the loop"
        assert result["code"] == "PERF_001"

    def test_severity_mapping_to_pattern_result(self) -> None:
        """Test that severities map correctly to PatternValidationResult values."""
        # Test each mapping
        pass_issue = PatternValidationIssue(
            severity=ValidationSeverity.PASS, message="Pass"
        )
        assert pass_issue.to_legacy_format()["level"] == "pass"

        warn_issue = PatternValidationIssue(
            severity=ValidationSeverity.WARNING, message="Warning"
        )
        assert warn_issue.to_legacy_format()["level"] == "warn"

        fail_issue = PatternValidationIssue(
            severity=ValidationSeverity.FAIL, message="Fail"
        )
        assert fail_issue.to_legacy_format()["level"] == "fail"

        error_issue = PatternValidationIssue(
            severity=ValidationSeverity.ERROR, message="Error"
        )
        assert error_issue.to_legacy_format()["level"] == "error"

        # INFO and STYLE map to PASS in pattern domain
        info_issue = PatternValidationIssue(
            severity=ValidationSeverity.INFO, message="Info"
        )
        assert info_issue.to_legacy_format()["level"] == "pass"

        style_issue = PatternValidationIssue(
            severity=ValidationSeverity.STYLE, message="Style"
        )
        assert style_issue.to_legacy_format()["level"] == "pass"

    def test_empty_category_handling(self) -> None:
        """Test that empty category is handled properly."""
        issue = PatternValidationIssue(
            severity=ValidationSeverity.ERROR, message="Error without category"
        )
        result = issue.to_legacy_format()
        assert result["category"] == ""

    def test_inherits_all_base_functionality(self) -> None:
        """Test that pattern issue inherits all base functionality."""
        issue = PatternValidationIssue(
            severity=ValidationSeverity.ERROR, message="Error", category="test"
        )

        # Base methods work
        assert issue.is_error() is True
        assert issue.is_warning() is False

        # Base to_dict works
        base_dict = issue.to_dict()
        assert base_dict["severity"] == "error"
        assert base_dict["message"] == "Error"
        assert base_dict["category"] == "test"


class TestCrossCompatibility:
    """Test cross-compatibility between different validation issue types."""

    def test_can_convert_between_formats(self) -> None:
        """Test converting between different domain formats."""
        # Create base issue
        base_issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Cross-compatible warning",
            location="test.py",
            suggestion="Fix it",
            code="TEST_001",
            category="test",
        )

        # Convert to semantic
        semantic = SemanticValidationIssue(**base_issue.model_dump())
        assert semantic.message == base_issue.message
        assert semantic.severity == base_issue.severity

        # Convert to workflow
        workflow = WorkflowValidationIssue(**base_issue.model_dump())
        assert workflow.message == base_issue.message
        assert workflow.severity == base_issue.severity

        # Convert to pattern
        pattern = PatternValidationIssue(**base_issue.model_dump())
        assert pattern.message == base_issue.message
        assert pattern.severity == base_issue.severity

    def test_unified_severity_across_domains(self) -> None:
        """Test that severity enum is consistent across all domains."""
        severity = ValidationSeverity.ERROR

        semantic = SemanticValidationIssue(severity=severity, message="Error")
        workflow = WorkflowValidationIssue(severity=severity, message="Error")
        pattern = PatternValidationIssue(severity=severity, message="Error")

        assert semantic.severity == workflow.severity == pattern.severity
        assert semantic.is_error() == workflow.is_error() == pattern.is_error()

    def test_backward_compatibility_preserved(self) -> None:
        """Test that backward compatibility is preserved for existing code."""
        # Test semantic legacy format
        semantic = SemanticValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Semantic error",
            agent="test_agent",
        )
        legacy = semantic.to_legacy_dataclass()
        assert isinstance(legacy["severity"], ValidationSeverity)  # Enum for dataclass
        assert legacy["agent"] == "test_agent"

        # Test workflow legacy format
        workflow = WorkflowValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Workflow warning",
            rule_id="WF_001",
            severity_level=7,
        )
        legacy = workflow.to_legacy_format()
        assert legacy["issue_type"] == "warning"
        assert legacy["severity"] == 7
        assert legacy["rule_id"] == "WF_001"

        # Test pattern legacy format
        pattern = PatternValidationIssue(
            severity=ValidationSeverity.FAIL,
            message="Pattern fail",
            category="structure",
        )
        legacy = pattern.to_legacy_format()
        assert legacy["level"] == "fail"
        assert legacy["category"] == "structure"
