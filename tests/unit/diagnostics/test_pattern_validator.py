"""
Comprehensive tests for Pattern Validator module.

Tests the PatternValidationFramework and various validator classes with
validation levels, security checks, and certification capabilities.
"""

import pytest
import json
import tempfile
import time
from unittest.mock import Mock, patch
from pathlib import Path

from cognivault.diagnostics.pattern_validator import (
    PatternValidationFramework,
    StructuralValidator,
    SemanticValidator,
    PerformanceValidator,
    SecurityValidator,
    PatternValidationLevel,
    PatternValidationResult,
    ValidationIssue,
    PatternValidationReport,
)
from cognivault.langgraph_backend.graph_patterns.base import GraphPattern


class MockGraphPattern(GraphPattern):
    """Mock graph pattern for testing."""

    def __init__(
        self, name="test_pattern", has_build_graph=True, build_graph_params=None
    ):
        self._name = name
        self._has_build_graph = has_build_graph
        self._build_graph_params = build_graph_params or ["agents", "llm", "config"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Test pattern: {self._name}"

    def get_pattern_name(self) -> str:
        return self._name

    def build_graph(self, agents, llm, config):
        return "mock_graph"

    def get_edges(self, agents):
        return []

    def get_entry_point(self, agents):
        return "start"

    def get_exit_points(self, agents):
        return ["end"]


class TestValidationIssue:
    """Test suite for ValidationIssue dataclass."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            level=PatternValidationResult.WARN,
            category="structure",
            message="Test warning message",
            location="test_method",
            suggestion="Fix the issue",
            code="V001",
        )

        assert issue.level == PatternValidationResult.WARN
        assert issue.category == "structure"
        assert issue.message == "Test warning message"
        assert issue.location == "test_method"
        assert issue.suggestion == "Fix the issue"
        assert issue.code == "V001"

    def test_validation_issue_minimal(self):
        """Test ValidationIssue with minimal fields."""
        issue = ValidationIssue(
            level=PatternValidationResult.FAIL,
            category="semantic",
            message="Test error",
        )

        assert issue.level == PatternValidationResult.FAIL
        assert issue.category == "semantic"
        assert issue.message == "Test error"
        assert issue.location is None
        assert issue.suggestion is None
        assert issue.code is None


class TestPatternValidationReport:
    """Test suite for PatternValidationReport dataclass."""

    def test_validation_report_creation(self):
        """Test PatternValidationReport creation."""
        issues = [
            ValidationIssue(
                level=PatternValidationResult.WARN, category="test", message="warning"
            ),
            ValidationIssue(
                level=PatternValidationResult.FAIL, category="test", message="error"
            ),
        ]

        report = PatternValidationReport(
            pattern_name="test_pattern",
            pattern_class="TestPattern",
            validation_level=PatternValidationLevel.STRICT,
            overall_result=PatternValidationResult.FAIL,
            issues=issues,
            metrics={"complexity": 5},
            recommendations=["Improve structure"],
            certification_ready=False,
        )

        assert report.pattern_name == "test_pattern"
        assert report.pattern_class == "TestPattern"
        assert report.validation_level == PatternValidationLevel.STRICT
        assert report.overall_result == PatternValidationResult.FAIL
        assert len(report.issues) == 2
        assert report.metrics["complexity"] == 5
        assert len(report.recommendations) == 1
        assert report.certification_ready is False


class TestStructuralValidator:
    """Test suite for StructuralValidator class."""

    @pytest.fixture
    def validator(self):
        """Create StructuralValidator instance."""
        return StructuralValidator()

    def test_validator_name(self, validator):
        """Test validator name property."""
        assert validator.validator_name == "Structural Validator"

    def test_validate_valid_pattern(self, validator):
        """Test validation of valid pattern."""
        pattern = MockGraphPattern()
        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have no critical issues for basic pattern
        assert all(issue.level != PatternValidationResult.FAIL for issue in issues)

    def test_validate_missing_build_graph(self, validator):
        """Test validation with missing build_graph method."""
        pattern = Mock()
        delattr(pattern, "build_graph") if hasattr(pattern, "build_graph") else None

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have failure for missing required method
        assert any(
            issue.level == PatternValidationResult.FAIL
            and "build_graph" in issue.message
            for issue in issues
        )

    def test_validate_missing_get_pattern_name(self, validator):
        """Test validation with missing get_pattern_name method."""
        pattern = Mock()
        pattern.build_graph = Mock()
        (
            delattr(pattern, "get_pattern_name")
            if hasattr(pattern, "get_pattern_name")
            else None
        )

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have failure for missing required method
        assert any(
            issue.level == PatternValidationResult.FAIL
            and "get_pattern_name" in issue.message
            for issue in issues
        )

    def test_validate_strict_level_missing_params(self, validator):
        """Test strict validation with missing parameters."""
        pattern = Mock()
        pattern.build_graph = Mock()
        pattern.get_pattern_name = Mock(return_value="test")

        # Mock signature with missing parameters
        with patch("inspect.signature") as mock_sig:
            mock_sig.return_value.parameters.keys.return_value = {
                "agents"
            }  # Missing llm, config

            issues = validator.validate(pattern, PatternValidationLevel.STRICT)

            # Should have failure for missing parameters
            assert any(
                issue.level == PatternValidationResult.FAIL
                and "missing parameters" in issue.message
                for issue in issues
            )

    def test_validate_invalid_pattern_name(self, validator):
        """Test validation with invalid pattern name."""
        pattern = Mock()
        pattern.build_graph = Mock()
        pattern.get_pattern_name = Mock(return_value="invalid@pattern!")

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have warning for invalid pattern name
        assert any(
            issue.level == PatternValidationResult.WARN
            and "pattern name" in issue.message.lower()
            for issue in issues
        )

    def test_validate_empty_pattern_name(self, validator):
        """Test validation with empty pattern name."""
        pattern = Mock()
        pattern.build_graph = Mock()
        pattern.get_pattern_name = Mock(return_value="")

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have failure for empty pattern name
        assert any(
            issue.level == PatternValidationResult.FAIL
            and "non-empty string" in issue.message
            for issue in issues
        )

    def test_validate_get_pattern_name_exception(self, validator):
        """Test validation when get_pattern_name raises exception."""
        pattern = Mock()
        pattern.build_graph = Mock(return_value=Mock())  # Ensure non-async return
        pattern.get_pattern_name = Mock(side_effect=Exception("Test error"))

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have error for exception
        assert any(
            issue.level == PatternValidationResult.ERROR
            and "Error calling get_pattern_name" in issue.message
            for issue in issues
        )


class TestSemanticValidator:
    """Test suite for SemanticValidator class."""

    @pytest.fixture
    def validator(self):
        """Create SemanticValidator instance."""
        return SemanticValidator()

    def test_validator_name(self, validator):
        """Test validator name property."""
        assert validator.validator_name == "Semantic Validator"

    def test_validate_valid_pattern(self, validator):
        """Test validation of valid pattern."""
        pattern = MockGraphPattern()

        with patch("unittest.mock.Mock"):
            issues = validator.validate(pattern, PatternValidationLevel.BASIC)

            # Should not have critical issues for working pattern
            assert all(issue.level != PatternValidationResult.FAIL for issue in issues)

    def test_validate_build_graph_returns_none(self, validator):
        """Test validation when build_graph returns None."""
        pattern = Mock()
        pattern.build_graph = Mock(return_value=None)

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have failure for None return
        assert any(
            issue.level == PatternValidationResult.FAIL
            and "returned None" in issue.message
            for issue in issues
        )

    def test_validate_build_graph_exception(self, validator):
        """Test validation when build_graph raises exception."""
        pattern = Mock()
        pattern.build_graph = Mock(side_effect=Exception("Build error"))

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have error for exception
        assert any(
            issue.level == PatternValidationResult.ERROR
            and "Error building graph" in issue.message
            for issue in issues
        )

    def test_validate_no_build_graph_method(self, validator):
        """Test validation when pattern has no build_graph method."""
        pattern = Mock()
        delattr(pattern, "build_graph") if hasattr(pattern, "build_graph") else None

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should handle missing method gracefully
        assert isinstance(issues, list)

    def test_validate_semantic_exception(self, validator):
        """Test validation when semantic validation itself fails."""
        pattern = Mock()

        # Force an exception in the pattern itself to trigger exception handling
        pattern.build_graph.side_effect = Exception("Mock validation error")

        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Should have error for validation failure (build_graph level)
        assert any(
            issue.level == PatternValidationResult.ERROR
            and "Error building graph" in issue.message
            for issue in issues
        )


class TestPerformanceValidator:
    """Test suite for PerformanceValidator class."""

    @pytest.fixture
    def validator(self):
        """Create PerformanceValidator instance."""
        return PerformanceValidator()

    def test_validator_name(self, validator):
        """Test validator name property."""
        assert validator.validator_name == "Performance Validator"

    def test_validate_basic_level(self, validator):
        """Test validation at basic level."""
        pattern = MockGraphPattern()
        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Basic level should not run performance checks
        assert len(issues) == 0

    def test_validate_strict_level(self, validator):
        """Test validation at strict level."""
        pattern = MockGraphPattern()

        with patch.object(validator, "_check_performance_patterns", return_value=[]):
            with patch.object(validator, "_check_memory_patterns", return_value=[]):
                with patch.object(
                    validator, "_check_efficiency_patterns", return_value=[]
                ):
                    issues = validator.validate(pattern, PatternValidationLevel.STRICT)

                    # Should run all performance checks
                    assert isinstance(issues, list)

    def test_check_performance_patterns_time_sleep(self, validator):
        """Test detection of time.sleep anti-pattern."""
        pattern = Mock()

        with patch(
            "inspect.getsource", return_value="def method():\n    time.sleep(1)\n"
        ):
            issues = validator._check_performance_patterns(pattern)

            # Should detect synchronous sleep
            assert any(
                issue.level == PatternValidationResult.WARN
                and "synchronous sleep" in issue.message
                for issue in issues
            )

    def test_check_memory_patterns_global_usage(self, validator):
        """Test detection of global variable usage."""
        pattern = Mock()

        with patch("inspect.getsource", return_value="global my_var\nmy_var = 1\n"):
            issues = validator._check_memory_patterns(pattern)

            # Should detect global variable usage
            assert any(
                issue.level == PatternValidationResult.WARN
                and "global variable" in issue.message
                for issue in issues
            )

    def test_check_efficiency_patterns_many_loops(self, validator):
        """Test detection of many loops."""
        pattern = Mock()

        with patch(
            "inspect.getsource",
            return_value="for i in range(5):\n    for j in range(5):\n        for k in range(5):\n            while True:\n                break\n",
        ):
            issues = validator._check_efficiency_patterns(pattern)

            # Should detect high number of loops
            assert any(
                issue.level == PatternValidationResult.WARN
                and "loops detected" in issue.message
                for issue in issues
            )


class TestSecurityValidator:
    """Test suite for SecurityValidator class."""

    @pytest.fixture
    def validator(self):
        """Create SecurityValidator instance."""
        return SecurityValidator()

    def test_validator_name(self, validator):
        """Test validator name property."""
        assert validator.validator_name == "Security Validator"

    def test_validate_basic_level(self, validator):
        """Test validation at basic level."""
        pattern = MockGraphPattern()
        issues = validator.validate(pattern, PatternValidationLevel.BASIC)

        # Basic level should not run security checks
        assert len(issues) == 0

    def test_validate_strict_level(self, validator):
        """Test validation at strict level."""
        pattern = MockGraphPattern()

        with patch.object(validator, "_check_security_patterns", return_value=[]):
            with patch.object(validator, "_check_data_validation", return_value=[]):
                issues = validator.validate(pattern, PatternValidationLevel.STRICT)

                # Should run security checks
                assert isinstance(issues, list)

    def test_check_security_patterns_eval_usage(self, validator):
        """Test detection of eval() usage."""
        pattern = Mock()

        with patch("inspect.getsource", return_value="result = eval(user_input)\n"):
            issues = validator._check_security_patterns(pattern)

            # Should detect eval usage
            assert any(
                issue.level == PatternValidationResult.FAIL
                and "eval()" in issue.message
                for issue in issues
            )

    def test_check_security_patterns_exec_usage(self, validator):
        """Test detection of exec() usage."""
        pattern = Mock()

        with patch("inspect.getsource", return_value="exec(user_code)\n"):
            issues = validator._check_security_patterns(pattern)

            # Should detect exec usage
            assert any(
                issue.level == PatternValidationResult.FAIL
                and "exec()" in issue.message
                for issue in issues
            )

    def test_check_data_validation_no_validation(self, validator):
        """Test detection of missing input validation."""
        pattern = Mock()
        pattern.build_graph = Mock()

        with patch(
            "inspect.getsource",
            return_value="def build_graph(self, agents, llm, config):\n    return graph\n",
        ):
            issues = validator._check_data_validation(pattern)

            # Should detect missing validation
            assert any(
                issue.level == PatternValidationResult.WARN
                and "No input validation" in issue.message
                for issue in issues
            )

    def test_check_data_validation_has_validation(self, validator):
        """Test with proper input validation."""
        pattern = Mock()
        pattern.build_graph = Mock()

        with patch(
            "inspect.getsource",
            return_value="def build_graph(self, agents, llm, config):\n    if not agents:\n        raise ValueError('Invalid agents')\n    return graph\n",
        ):
            issues = validator._check_data_validation(pattern)

            # Should not detect missing validation
            assert not any("No input validation" in issue.message for issue in issues)


class TestPatternValidationFramework:
    """Test suite for PatternValidationFramework class."""

    @pytest.fixture
    def framework(self):
        """Create PatternValidationFramework instance."""
        return PatternValidationFramework()

    @pytest.fixture
    def mock_pattern(self):
        """Create mock pattern for testing."""
        return MockGraphPattern()

    def test_initialization(self, framework):
        """Test PatternValidationFramework initialization."""
        assert framework.console is not None
        assert len(framework.validators) == 4  # 4 built-in validators
        assert len(framework.custom_validators) == 0
        assert any(isinstance(v, StructuralValidator) for v in framework.validators)
        assert any(isinstance(v, SemanticValidator) for v in framework.validators)
        assert any(isinstance(v, PerformanceValidator) for v in framework.validators)
        assert any(isinstance(v, SecurityValidator) for v in framework.validators)

    def test_create_app(self, framework):
        """Test CLI app creation."""
        app = framework.create_app()
        assert app is not None
        assert app.info.name == "pattern-validator"

    def test_load_pattern(self, framework):
        """Test pattern loading."""
        # Test loading built-in pattern
        pattern = framework._load_pattern("standard")

        assert isinstance(pattern, GraphPattern)
        assert pattern.name == "standard"

        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            framework._load_pattern("test/path")

    def test_validate_pattern_comprehensive_pass(self, framework, mock_pattern):
        """Test comprehensive validation with passing pattern."""
        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.BASIC
        )

        assert isinstance(report, PatternValidationReport)
        assert report.pattern_name == "test_pattern"
        assert report.pattern_class == "MockGraphPattern"
        assert report.validation_level == PatternValidationLevel.BASIC

    def test_validate_pattern_comprehensive_with_issues(self, framework):
        """Test comprehensive validation with issues."""
        # Create pattern that will fail validation
        pattern = Mock()
        pattern.get_pattern_name.return_value = "test_pattern"
        # Remove required methods to trigger validation failures
        del pattern.build_graph

        report = framework._validate_pattern_comprehensive(
            pattern, PatternValidationLevel.BASIC
        )

        assert report.overall_result in [
            PatternValidationResult.FAIL,
            PatternValidationResult.ERROR,
        ]
        assert len(report.issues) > 0

    def test_validate_pattern_comprehensive_certification(
        self, framework, mock_pattern
    ):
        """Test certification level validation."""
        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.CERTIFICATION
        )

        assert report.validation_level == PatternValidationLevel.CERTIFICATION
        # Certification readiness depends on issues found

    def test_validate_pattern_comprehensive_validator_exception(
        self, framework, mock_pattern
    ):
        """Test handling of validator exceptions."""
        # Add a validator that will raise an exception
        bad_validator = Mock()
        bad_validator.validate.side_effect = Exception("Validator error")
        bad_validator.validator_name = "Bad Validator"
        framework.validators.append(bad_validator)

        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.BASIC
        )

        # Should handle validator exception gracefully
        assert any(
            issue.level == PatternValidationResult.ERROR
            and "Validator Bad Validator failed" in issue.message
            for issue in report.issues
        )

    def test_display_validation_report(self, framework, mock_pattern):
        """Test validation report display."""
        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.BASIC
        )

        with patch.object(framework.console, "print") as mock_print:
            framework._display_validation_report(report, True)

            # Should print report information
            mock_print.assert_called()

    def test_display_validation_report_with_issues(self, framework):
        """Test validation report display with issues."""
        issues = [
            ValidationIssue(
                level=PatternValidationResult.WARN,
                category="test",
                message="warning message",
                suggestion="fix it",
            ),
            ValidationIssue(
                level=PatternValidationResult.FAIL,
                category="test",
                message="error message",
            ),
        ]

        report = PatternValidationReport(
            pattern_name="test",
            pattern_class="Test",
            validation_level=PatternValidationLevel.BASIC,
            overall_result=PatternValidationResult.FAIL,
            issues=issues,
        )

        with patch.object(framework.console, "print") as mock_print:
            framework._display_validation_report(report, True)

            # Should print issues table
            mock_print.assert_called()

    def test_format_report_json(self, framework, mock_pattern):
        """Test JSON report formatting."""
        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.BASIC
        )

        json_output = framework._format_report_json(report)

        # Should be valid JSON
        parsed = json.loads(json_output)
        assert parsed["pattern_name"] == "test_pattern"
        assert parsed["validation_level"] == "basic"

    def test_format_report_markdown(self, framework, mock_pattern):
        """Test Markdown report formatting."""
        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.BASIC
        )

        markdown_output = framework._format_report_markdown(report)

        # Should contain Markdown formatting
        assert "# Pattern Validation Report" in markdown_output
        assert "**Pattern:** test_pattern" in markdown_output

    def test_load_test_queries_from_file(self, framework):
        """Test loading test queries from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Query 1\nQuery 2\nQuery 3\n")
            f.flush()

            queries = framework._load_test_queries(f.name)

            assert len(queries) == 3
            assert "Query 1" in queries

        Path(f.name).unlink()  # Clean up

    def test_load_test_queries_default(self, framework):
        """Test loading default test queries."""
        queries = framework._load_test_queries(None)

        assert isinstance(queries, list)
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)

    def test_load_test_queries_file_not_found(self, framework):
        """Test loading test queries when file not found."""
        queries = framework._load_test_queries("nonexistent.txt")

        # Should fall back to default queries
        assert isinstance(queries, list)
        assert len(queries) > 0

    def test_run_pattern_tests(self, framework, mock_pattern):
        """Test running pattern tests."""
        queries = ["Test query 1", "Test query 2"]
        agents = ["refiner", "critic"]

        results = framework._run_pattern_tests(mock_pattern, queries, agents, 3, 30)

        assert isinstance(results, dict)
        assert "total_tests" in results
        assert "passed" in results
        assert "failed" in results
        assert "success_rate" in results

    def test_display_test_results(self, framework):
        """Test test results display."""
        results = {"total_tests": 10, "passed": 8, "failed": 2, "success_rate": 0.8}

        with patch.object(framework.console, "print") as mock_print:
            framework._display_test_results(results)

            # Should print results table
            mock_print.assert_called()

    def test_run_certification_tests(self, framework, mock_pattern):
        """Test certification tests."""
        report = framework._run_certification_tests(mock_pattern, None)

        assert isinstance(report, PatternValidationReport)
        assert report.validation_level == PatternValidationLevel.CERTIFICATION

    def test_display_certification_results(self, framework, mock_pattern):
        """Test certification results display."""
        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.CERTIFICATION
        )

        with patch.object(framework.console, "print") as mock_print:
            framework._display_certification_results(report)

            # Should display validation report
            mock_print.assert_called()

    def test_generate_certificate(self, framework, mock_pattern):
        """Test certificate generation."""
        report = framework._validate_pattern_comprehensive(
            mock_pattern, PatternValidationLevel.CERTIFICATION
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            framework._generate_certificate(report, f.name)

            # Check certificate was created
            assert Path(f.name).exists()

            with open(f.name, "r") as cert_file:
                content = cert_file.read()
                assert "PATTERN CERTIFICATION CERTIFICATE" in content
                assert "test_pattern" in content

        Path(f.name).unlink()  # Clean up

    def test_discover_patterns_in_path(self, framework):
        """Test pattern discovery."""
        patterns = framework._discover_patterns_in_path(".", None)

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all("name" in p and "type" in p and "location" in p for p in patterns)

    def test_load_validator(self, framework):
        """Test custom validator loading."""
        validator = framework._load_validator("test/path")

        assert isinstance(validator, StructuralValidator)

    def test_generate_comprehensive_report(self, framework, mock_pattern):
        """Test comprehensive report generation."""
        report = framework._generate_comprehensive_report(
            mock_pattern, "detailed", True
        )

        assert isinstance(report, str)
        assert "# Comprehensive Pattern Report" in report
        assert "test_pattern" in report

    def test_benchmark_pattern_performance(self, framework, mock_pattern):
        """Test pattern performance benchmarking."""
        results = framework._benchmark_pattern_performance(
            mock_pattern, None, ["refiner"], 5
        )

        assert isinstance(results, dict)
        assert "avg_duration" in results
        assert "runs" in results

    def test_display_benchmark_results(self, framework):
        """Test benchmark results display."""
        results = {"avg_duration": 1.5, "baseline_comparison": "20% faster", "runs": 10}

        with patch.object(framework.console, "print") as mock_print:
            framework._display_benchmark_results(results)

            # Should print benchmark table
            mock_print.assert_called()


# Integration tests
class TestPatternValidationFrameworkIntegration:
    """Integration tests for PatternValidationFramework."""

    @pytest.fixture
    def framework(self):
        """Create framework for integration tests."""
        return PatternValidationFramework()

    def test_full_validation_workflow(self, framework):
        """Test complete validation workflow."""
        pattern = MockGraphPattern()

        # Should not raise exceptions
        report = framework._validate_pattern_comprehensive(
            pattern, PatternValidationLevel.STANDARD
        )
        assert isinstance(report, PatternValidationReport)

    def test_cli_app_integration(self, framework):
        """Test CLI app creation and commands."""
        app = framework.create_app()

        # Test that app was created successfully
        assert app is not None
        assert app.info.name == "pattern-validator"

    def test_custom_validator_registration(self, framework):
        """Test registering custom validators."""
        custom_validator = StructuralValidator()
        initial_count = len(framework.validators + framework.custom_validators)

        framework.custom_validators.append(custom_validator)

        assert (
            len(framework.validators + framework.custom_validators) == initial_count + 1
        )

    def test_validation_with_custom_validators(self, framework):
        """Test validation with custom validators."""
        # Add custom validator
        custom_validator = Mock()
        custom_validator.validate.return_value = [
            ValidationIssue(
                level=PatternValidationResult.WARN,
                category="custom",
                message="Custom validation warning",
            )
        ]
        custom_validator.validator_name = "Custom Validator"
        framework.custom_validators.append(custom_validator)

        pattern = MockGraphPattern()
        report = framework._validate_pattern_comprehensive(
            pattern, PatternValidationLevel.BASIC
        )

        # Should include custom validator results
        assert any(
            "Custom validation warning" in issue.message for issue in report.issues
        )


# Performance tests
class TestPatternValidationFrameworkPerformance:
    """Performance tests for PatternValidationFramework."""

    @pytest.fixture
    def framework(self):
        """Create framework for performance tests."""
        return PatternValidationFramework()

    def test_validation_performance(self, framework):
        """Test validation performance."""
        import time

        pattern = MockGraphPattern()

        start_time = time.time()
        report = framework._validate_pattern_comprehensive(
            pattern, PatternValidationLevel.BASIC
        )
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 2.0
        assert isinstance(report, PatternValidationReport)

    def test_multiple_patterns_validation(self, framework):
        """Test validating multiple patterns."""
        patterns = [MockGraphPattern(f"pattern_{i}") for i in range(10)]

        start_time = time.time()
        reports = []
        for pattern in patterns:
            report = framework._validate_pattern_comprehensive(
                pattern, PatternValidationLevel.BASIC
            )
            reports.append(report)
        end_time = time.time()

        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0
        assert len(reports) == 10
        assert all(isinstance(r, PatternValidationReport) for r in reports)


# Error handling tests
class TestPatternValidationFrameworkErrorHandling:
    """Error handling tests for PatternValidationFramework."""

    @pytest.fixture
    def framework(self):
        """Create framework for error handling tests."""
        return PatternValidationFramework()

    def test_validate_invalid_pattern_object(self, framework):
        """Test validation with invalid pattern object."""
        invalid_pattern = "not a pattern"

        # Should handle gracefully
        report = framework._validate_pattern_comprehensive(
            invalid_pattern, PatternValidationLevel.BASIC
        )
        assert isinstance(report, PatternValidationReport)

    def test_validate_pattern_missing_methods(self, framework):
        """Test validation of pattern missing required methods."""
        pattern = Mock()
        pattern.get_pattern_name.return_value = "test_pattern"
        # Remove all methods

        report = framework._validate_pattern_comprehensive(
            pattern, PatternValidationLevel.STRICT
        )

        # Should detect missing methods
        assert len(report.issues) > 0
        assert any(
            issue.level == PatternValidationResult.FAIL for issue in report.issues
        )

    def test_format_report_with_unicode(self, framework):
        """Test report formatting with unicode characters."""
        issues = [
            ValidationIssue(
                level=PatternValidationResult.WARN,
                category="test",
                message="Unicode message: 测试",
            )
        ]
        report = PatternValidationReport(
            pattern_name="test_pattern",
            pattern_class="TestPattern",
            validation_level=PatternValidationLevel.BASIC,
            overall_result=PatternValidationResult.WARN,
            issues=issues,
        )

        # Should handle unicode without errors
        json_output = framework._format_report_json(report)
        markdown_output = framework._format_report_markdown(report)

        assert isinstance(json_output, str)
        assert isinstance(markdown_output, str)
