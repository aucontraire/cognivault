"""Tests for semantic validation layer."""

import pytest
from typing import Any

from cognivault.langgraph_backend.semantic_validation import (
    WorkflowSemanticValidator,
    CogniVaultValidator,
    SemanticValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationError,
    create_default_validator,
)


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_validation_issue_creation(self) -> None:
        """Test creating ValidationIssue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error message",
            agent="refiner",
            suggestion="Test suggestion",
            code="TEST_ERROR",
        )

        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error message"
        assert issue.agent == "refiner"
        assert issue.suggestion == "Test suggestion"
        assert issue.code == "TEST_ERROR"

    def test_validation_issue_minimal(self) -> None:
        """Test creating ValidationIssue with minimal parameters."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING, message="Test warning"
        )

        assert issue.severity == ValidationSeverity.WARNING
        assert issue.message == "Test warning"
        assert issue.agent is None
        assert issue.suggestion is None
        assert issue.code is None


class TestSemanticValidationResult:
    """Test SemanticValidationResult functionality."""

    def test_validation_result_valid(self) -> None:
        """Test SemanticValidationResult for valid workflow."""
        result = SemanticValidationResult(is_valid=True, issues=[])

        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings
        assert result.error_messages == []
        assert result.warning_messages == []

    def test_validation_result_with_errors(self) -> None:
        """Test SemanticValidationResult with errors."""
        result = SemanticValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(ValidationSeverity.ERROR, "Error 1"),
                ValidationIssue(ValidationSeverity.ERROR, "Error 2"),
                ValidationIssue(ValidationSeverity.WARNING, "Warning 1"),
            ],
        )

        assert not result.is_valid
        assert result.has_errors
        assert result.has_warnings
        assert result.error_messages == ["Error 1", "Error 2"]
        assert result.warning_messages == ["Warning 1"]

    def test_add_issue_updates_validity(self) -> None:
        """Test that adding error issues updates validity."""
        result = SemanticValidationResult(is_valid=True, issues=[])

        # Adding warning doesn't change validity
        result.add_issue(ValidationSeverity.WARNING, "Warning message")
        assert result.is_valid

        # Adding error changes validity
        result.add_issue(ValidationSeverity.ERROR, "Error message")
        assert not result.is_valid

        assert len(result.issues) == 2


class TestCogniVaultValidator:
    """Test CogniVaultValidator implementation."""

    @pytest.fixture
    def validator(self) -> Any:
        """Fixture for CogniVaultValidator."""
        return CogniVaultValidator(strict_mode=False)

    @pytest.fixture
    def strict_validator(self) -> Any:
        """Fixture for strict CogniVaultValidator."""
        return CogniVaultValidator(strict_mode=True)

    def test_validator_initialization(
        self, validator: Any, strict_validator: Any
    ) -> None:
        """Test validator initialization."""
        assert not validator.strict_mode
        assert strict_validator.strict_mode

        assert "standard" in validator.get_supported_patterns()
        assert "parallel" in validator.get_supported_patterns()
        assert "conditional" in validator.get_supported_patterns()

    def test_validate_agents_basic(self, validator: Any) -> None:
        """Test basic agent validation."""
        # Valid agents
        result = validator.validate_agents(["refiner", "synthesis"])
        assert result.is_valid

        # Empty agents list
        result = validator.validate_agents([])
        assert not result.is_valid
        assert result.has_errors

    def test_validate_agents_duplicates(self, validator: Any) -> None:
        """Test agent validation with duplicates."""
        result = validator.validate_agents(["refiner", "refiner", "synthesis"])

        # Should be valid but with warning
        assert result.is_valid
        assert result.has_warnings
        assert "duplicate" in result.warning_messages[0].lower()

    def test_validate_standard_pattern_valid(self, validator: Any) -> None:
        """Test standard pattern validation with valid combinations."""
        # Full 4-agent workflow
        result = validator.validate_workflow(
            agents=["refiner", "critic", "historian", "synthesis"], pattern="standard"
        )
        assert result.is_valid

        # Refiner + synthesis
        result = validator.validate_workflow(
            agents=["refiner", "synthesis"], pattern="standard"
        )
        assert result.is_valid

    def test_validate_standard_pattern_synthesis_without_analysis(
        self, validator: Any, strict_validator: Any
    ) -> None:
        """Test synthesis without analysis agents."""
        agents = ["synthesis"]

        # Non-strict mode: warning
        result = validator.validate_workflow(agents=agents, pattern="standard")
        assert result.is_valid
        assert result.has_warnings

        # Strict mode: error
        result = strict_validator.validate_workflow(agents=agents, pattern="standard")
        assert not result.is_valid
        assert result.has_errors

    def test_validate_standard_pattern_analysis_without_synthesis(
        self, validator: Any
    ) -> None:
        """Test analysis agents without synthesis."""
        result = validator.validate_workflow(
            agents=["refiner", "critic", "historian"], pattern="standard"
        )

        # Should be valid with info message
        assert result.is_valid
        # Check for info about adding synthesis
        info_messages = [
            issue.message
            for issue in result.issues
            if issue.severity == ValidationSeverity.INFO
        ]
        assert any("synthesis" in msg.lower() for msg in info_messages)

    def test_validate_unknown_agents_strict_vs_permissive(
        self, validator: Any, strict_validator: Any
    ) -> None:
        """Test unknown agents in strict vs permissive mode."""
        agents = ["refiner", "unknown_agent", "synthesis"]

        # Non-strict mode: warning
        result = validator.validate_workflow(agents=agents, pattern="standard")
        assert result.is_valid
        assert result.has_warnings

        # Strict mode: error
        result = strict_validator.validate_workflow(agents=agents, pattern="standard")
        assert not result.is_valid
        assert result.has_errors

    def test_validate_parallel_pattern(self, validator: Any) -> None:
        """Test parallel pattern validation."""
        # Good parallel candidates
        result = validator.validate_workflow(
            agents=["refiner", "critic", "historian", "synthesis"], pattern="parallel"
        )
        assert result.is_valid

        # Single agent in parallel pattern
        result = validator.validate_workflow(agents=["synthesis"], pattern="parallel")
        assert result.is_valid
        assert result.has_warnings

    def test_validate_conditional_pattern(self, validator: Any) -> None:
        """Test conditional pattern validation."""
        # With refiner (good)
        result = validator.validate_workflow(
            agents=["refiner", "critic", "historian", "synthesis"],
            pattern="conditional",
        )
        assert result.is_valid

        # Without refiner (warning)
        result = validator.validate_workflow(
            agents=["critic", "synthesis"], pattern="conditional"
        )
        assert result.is_valid
        assert result.has_warnings

    def test_validate_unsupported_pattern(self, validator: Any) -> None:
        """Test validation with unsupported pattern."""
        result = validator.validate_workflow(
            agents=["refiner", "synthesis"], pattern="unsupported_pattern"
        )

        assert not result.is_valid
        assert result.has_errors
        assert "unsupported pattern" in result.error_messages[0].lower()

    def test_validation_issue_codes(
        self, validator: Any, strict_validator: Any
    ) -> None:
        """Test that validation issues have proper codes."""
        # Synthesis without analysis
        result = strict_validator.validate_workflow(
            agents=["synthesis"], pattern="standard"
        )

        synthesis_issues = [
            issue
            for issue in result.issues
            if issue.code == "SYNTHESIS_WITHOUT_ANALYSIS"
        ]
        assert len(synthesis_issues) > 0

    def test_validation_suggestions(self, validator: Any) -> None:
        """Test that validation issues include helpful suggestions."""
        result = validator.validate_workflow(agents=["synthesis"], pattern="standard")

        # Check that suggestions are provided
        suggestions = [
            issue.suggestion for issue in result.issues if issue.suggestion is not None
        ]
        assert len(suggestions) > 0
        assert any("critic" in suggestion.lower() for suggestion in suggestions)


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self) -> None:
        """Test creating ValidationError."""
        result = SemanticValidationResult(
            is_valid=False,
            issues=[ValidationIssue(ValidationSeverity.ERROR, "Test error")],
        )

        error = ValidationError("Validation failed", result)

        assert str(error) == "Validation failed"
        assert error.validation_result is result


class TestCreateDefaultValidator:
    """Test default validator creation."""

    def test_create_default_validator_normal(self) -> None:
        """Test creating default validator in normal mode."""
        validator = create_default_validator(strict_mode=False)

        assert isinstance(validator, CogniVaultValidator)
        assert not validator.strict_mode

    def test_create_default_validator_strict(self) -> None:
        """Test creating default validator in strict mode."""
        validator = create_default_validator(strict_mode=True)

        assert isinstance(validator, CogniVaultValidator)
        assert validator.strict_mode


class TestWorkflowSemanticValidatorIntegration:
    """Integration tests for semantic validation."""

    def test_validator_workflow_comprehensive(self) -> None:
        """Test comprehensive workflow validation scenarios."""
        validator = CogniVaultValidator(strict_mode=False)

        # Test scenarios that should be valid
        valid_scenarios = [
            (["refiner", "critic", "historian", "synthesis"], "standard"),
            (["refiner", "synthesis"], "standard"),
            (["critic", "historian", "synthesis"], "standard"),
            (["refiner", "critic", "historian"], "parallel"),
            (["refiner", "critic"], "conditional"),
        ]

        for agents, pattern in valid_scenarios:
            result = validator.validate_workflow(agents=agents, pattern=pattern)
            assert result.is_valid, f"Expected {agents} with {pattern} to be valid"

    def test_validator_edge_cases(self) -> None:
        """Test edge cases in validation."""
        validator = CogniVaultValidator(strict_mode=False)

        # Empty agents list
        result = validator.validate_workflow(agents=[], pattern="standard")
        assert not result.is_valid

        # Case insensitive agents
        result = validator.validate_workflow(
            agents=["REFINER", "Synthesis"], pattern="standard"
        )
        assert result.is_valid

    def test_validation_performance(self) -> None:
        """Test validation performance doesn't degrade with complex workflows."""
        validator = CogniVaultValidator(strict_mode=False)

        # Large agent list
        large_agent_list = ["refiner", "critic", "historian", "synthesis"] * 10 + [
            f"custom_agent_{i}" for i in range(20)
        ]

        result = validator.validate_workflow(
            agents=large_agent_list, pattern="standard"
        )

        # Should complete without issues (may have warnings about unknown agents)
        assert isinstance(result, SemanticValidationResult)


class MockValidator(WorkflowSemanticValidator):
    """Mock validator for testing abstract interface."""

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    def validate_workflow(
        self, agents: Any, pattern: Any, **kwargs: Any
    ) -> SemanticValidationResult:
        result = SemanticValidationResult(is_valid=True, issues=[])

        if self.should_fail:
            result.add_issue(
                ValidationSeverity.ERROR, "Mock validation failure", code="MOCK_ERROR"
            )

        return result

    def get_supported_patterns(self) -> set[str]:
        return {"mock_pattern"}


class TestWorkflowSemanticValidatorInterface:
    """Test the abstract WorkflowSemanticValidator interface."""

    def test_abstract_validator_cannot_be_instantiated(self) -> None:
        """Test that WorkflowSemanticValidator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            WorkflowSemanticValidator()  # type: ignore[abstract]

    def test_mock_validator_implementation(self) -> None:
        """Test mock validator implementation."""
        validator = MockValidator(should_fail=False)

        result = validator.validate_workflow(["test"], "mock_pattern")
        assert result.is_valid
        assert "mock_pattern" in validator.get_supported_patterns()

    def test_mock_validator_failure(self) -> None:
        """Test mock validator with failure."""
        validator = MockValidator(should_fail=True)

        result = validator.validate_workflow(["test"], "mock_pattern")
        assert not result.is_valid
        assert result.has_errors
