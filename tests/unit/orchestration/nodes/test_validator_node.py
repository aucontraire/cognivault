"""
Tests for ValidatorNode implementation.

This module tests the ValidatorNode class which handles quality
validation and gating in the advanced node execution system.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from cognivault.orchestration.nodes.validator_node import (
    ValidatorNode,
    ValidationCriteria,
    NodeValidationResult,
    WorkflowValidationReport,
)
from cognivault.orchestration.nodes.base_advanced_node import NodeExecutionContext
from cognivault.agents.metadata import AgentMetadata, TaskClassification


class TestValidatorNodeInitialization:
    """Test ValidatorNode initialization and validation."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata with validator execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "validator"
        metadata.cognitive_speed = "adaptive"
        metadata.cognitive_depth = "deep"
        return metadata

    @pytest.fixture
    def basic_criteria(self) -> Any:
        """Create basic validation criteria."""
        return [
            ValidationCriteria(
                name="has_content",
                validator=lambda data: "content" in data and len(data["content"]) > 0,
                weight=2.0,
                required=True,
                error_message="Content is required",
            ),
            ValidationCriteria(
                name="has_confidence",
                validator=lambda data: data.get("confidence", 0.0) >= 0.7,
                weight=1.0,
                required=False,
                error_message="Confidence should be >= 0.7",
            ),
        ]

    def test_validator_node_creation_success(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test successful ValidatorNode creation."""
        node = ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=basic_criteria,
            quality_threshold=0.8,
            required_criteria_pass_rate=1.0,
            allow_warnings=True,
            strict_mode=False,
        )

        assert node.node_name == "test_validator"
        assert node.execution_pattern == "validator"
        assert len(node.validation_criteria) == 2
        assert node.quality_threshold == 0.8
        assert node.required_criteria_pass_rate == 1.0
        assert node.allow_warnings is True
        assert node.strict_mode is False

    def test_validator_node_wrong_execution_pattern(self, basic_criteria: Mock) -> None:
        """Test that ValidatorNode requires validator execution pattern."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "aggregator"  # Wrong pattern

        with pytest.raises(
            ValueError, match="ValidatorNode requires execution_pattern='validator'"
        ):
            ValidatorNode(
                metadata=metadata, node_name="test", validation_criteria=basic_criteria
            )

    def test_validator_node_empty_criteria(self, mock_metadata: Mock) -> None:
        """Test that ValidatorNode requires at least one criterion."""
        with pytest.raises(
            ValueError, match="ValidatorNode requires at least one validation criterion"
        ):
            ValidatorNode(
                metadata=mock_metadata,
                node_name="test",
                validation_criteria=[],  # Empty criteria
            )

    def test_validator_node_invalid_quality_threshold(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that quality_threshold must be between 0.0 and 1.0."""
        with pytest.raises(
            ValueError, match="quality_threshold must be between 0.0 and 1.0"
        ):
            ValidatorNode(
                metadata=mock_metadata,
                node_name="test",
                validation_criteria=basic_criteria,
                quality_threshold=1.5,
            )

    def test_validator_node_invalid_required_criteria_pass_rate(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that required_criteria_pass_rate must be between 0.0 and 1.0."""
        with pytest.raises(
            ValueError, match="required_criteria_pass_rate must be between 0.0 and 1.0"
        ):
            ValidatorNode(
                metadata=mock_metadata,
                node_name="test",
                validation_criteria=basic_criteria,
                required_criteria_pass_rate=1.5,
            )

    def test_validator_node_default_values(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test ValidatorNode with default values."""
        node = ValidatorNode(
            metadata=mock_metadata, node_name="test", validation_criteria=basic_criteria
        )

        assert node.quality_threshold == 0.8
        assert node.required_criteria_pass_rate == 1.0
        assert node.allow_warnings is True
        assert node.strict_mode is False

    def test_validator_node_inherits_base_methods(
        self, mock_metadata: Mock, basic_criteria: Mock
    ) -> None:
        """Test that ValidatorNode inherits BaseAdvancedNode methods."""
        node = ValidatorNode(
            metadata=mock_metadata, node_name="test", validation_criteria=basic_criteria
        )

        # Should have base class methods
        assert hasattr(node, "get_fallback_patterns")
        assert hasattr(node, "get_node_info")
        assert hasattr(node, "validate_context")
        assert hasattr(node, "pre_execute")
        assert hasattr(node, "post_execute")

        # Test fallback patterns for validator node
        assert node.get_fallback_patterns() == ["processor", "terminator"]


class TestValidationCriteria:
    """Test ValidationCriteria dataclass."""

    def test_validation_criteria_creation_success(self) -> None:
        """Test successful ValidationCriteria creation."""
        criterion = ValidationCriteria(
            name="test_criterion",
            validator=lambda data: True,
            weight=2.0,
            required=True,
            error_message="Test error",
        )

        assert criterion.name == "test_criterion"
        assert criterion.weight == 2.0
        assert criterion.required is True
        assert criterion.error_message == "Test error"

    def test_validation_criteria_default_values(self) -> None:
        """Test ValidationCriteria with default values."""
        criterion = ValidationCriteria(name="test", validator=lambda data: True)

        assert criterion.weight == 1.0
        assert criterion.required is True
        assert criterion.error_message == ""

    def test_validation_criteria_validate_method(self) -> None:
        """Test ValidationCriteria validate method."""
        criterion = ValidationCriteria(
            name="has_key", validator=lambda data: "key" in data
        )

        assert criterion.validate_data({"key": "value"}) is True
        assert criterion.validate_data({"other": "value"}) is False


class TestWorkflowValidationReport:
    """Test WorkflowValidationReport dataclass."""

    def test_validation_report_creation(self) -> None:
        """Test WorkflowValidationReport creation."""
        report = WorkflowValidationReport(
            result=NodeValidationResult.PASS,
            quality_score=0.85,
            criteria_results={"test": {"passed": True}},
            recommendations=["All good"],
            validation_time_ms=100.0,
            total_criteria=2,
            passed_criteria=2,
            failed_criteria=0,
            warnings=[],
        )

        assert report.result == NodeValidationResult.PASS
        assert report.quality_score == 0.85
        assert report.success_rate == 1.0  # 2/2
        assert report.total_criteria == 2
        assert report.passed_criteria == 2
        assert report.failed_criteria == 0

    def test_validation_report_success_rate(self) -> None:
        """Test WorkflowValidationReport success_rate calculation."""
        # Test normal case
        report = WorkflowValidationReport(
            result=NodeValidationResult.WARNING,
            quality_score=0.7,
            criteria_results={},
            recommendations=[],
            validation_time_ms=50.0,
            total_criteria=4,
            passed_criteria=3,
            failed_criteria=1,
            warnings=[],
        )

        assert report.success_rate == 0.75  # 3/4

        # Test edge case: no criteria
        report_empty = WorkflowValidationReport(
            result=NodeValidationResult.PASS,
            quality_score=0.0,
            criteria_results={},
            recommendations=[],
            validation_time_ms=0.0,
            total_criteria=0,
            passed_criteria=0,
            failed_criteria=0,
            warnings=[],
        )

        assert report_empty.success_rate == 0.0


class TestValidatorNodeExecute:
    """Test ValidatorNode execute method."""

    @pytest.fixture
    def mock_metadata(self) -> Any:
        """Create mock AgentMetadata."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "validator"
        return metadata

    @pytest.fixture
    def mock_context(self) -> Any:
        """Create mock NodeExecutionContext with available inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.correlation_id = "test-correlation"
        context.workflow_id = "test-workflow"
        context.cognitive_classification = {"speed": "adaptive"}
        context.task_classification = Mock()
        context.execution_path = []
        context.resource_usage = {}
        context.add_to_execution_path = Mock()
        context.update_resource_usage = Mock()

        # Mock available inputs for validation
        context.available_inputs = {
            "input1": {
                "content": "This is valid content",
                "confidence": 0.85,
                "quality_score": 0.9,
                "metadata": {"source": "agent1"},
            },
            "input2": {
                "content": "Another valid input",
                "confidence": 0.75,
                "quality_score": 0.8,
                "metadata": {"source": "agent2"},
            },
        }

        return context

    @pytest.fixture
    def validation_criteria(self) -> Any:
        """Create validation criteria for testing."""
        return [
            ValidationCriteria(
                name="has_content",
                validator=lambda data: "content" in data and len(data["content"]) > 0,
                weight=2.0,
                required=True,
                error_message="Content is required",
            ),
            ValidationCriteria(
                name="has_confidence",
                validator=lambda data: data.get("confidence", 0.0) >= 0.7,
                weight=1.0,
                required=False,
                error_message="Confidence should be >= 0.7",
            ),
            ValidationCriteria(
                name="has_metadata",
                validator=lambda data: "metadata" in data
                and isinstance(data["metadata"], dict),
                weight=1.0,
                required=True,
                error_message="Metadata is required",
            ),
        ]

    @pytest.fixture
    def validator_node(self, mock_metadata: Mock, validation_criteria: Mock) -> Any:
        """Create a ValidatorNode for testing."""
        return ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=validation_criteria,
            quality_threshold=0.8,
            required_criteria_pass_rate=1.0,
            allow_warnings=True,
            strict_mode=False,
        )

    @pytest.mark.asyncio
    async def test_execute_validation_success(
        self, validator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute with successful validation."""
        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await validator_node.execute(mock_context)

        # Check result structure
        assert result["validation_result"] == "pass"
        assert result["quality_score"] > 0.8
        assert result["success_rate"] == 1.0  # All criteria passed
        assert result["passed"] is True
        assert result["total_criteria"] == 3
        assert result["passed_criteria"] == 3
        assert result["failed_criteria"] == 0
        assert len(result["warnings"]) == 0

        # Check criteria results
        criteria_results = result["criteria_results"]
        assert len(criteria_results) == 3
        assert criteria_results["has_content"]["passed"] is True
        assert criteria_results["has_confidence"]["passed"] is True
        assert criteria_results["has_metadata"]["passed"] is True

        # Check event was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["workflow_id"] == "test-workflow"
        assert call_args["validation_result"] == "pass"
        assert call_args["quality_score"] > 0.8

    @pytest.mark.asyncio
    async def test_execute_validation_failure(
        self, mock_metadata: Mock, validation_criteria: Mock, mock_context: Mock
    ) -> None:
        """Test execute with validation failure."""
        # Set up context with invalid data
        mock_context.available_inputs = {
            "bad_input": {
                "confidence": 0.5,  # Below threshold
                "quality_score": 0.6,
                # Missing content and metadata
            }
        }

        node = ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=validation_criteria,
            quality_threshold=0.8,
            required_criteria_pass_rate=1.0,
            allow_warnings=False,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Check result structure
        assert result["validation_result"] == "fail"
        assert result["quality_score"] < 0.8
        assert result["success_rate"] < 1.0
        assert result["passed"] is False
        assert result["failed_criteria"] > 0

        # Check that required criteria failed
        criteria_results = result["criteria_results"]
        assert criteria_results["has_content"]["passed"] is False
        assert criteria_results["has_metadata"]["passed"] is False

        # Check event was emitted with failure
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["validation_result"] == "fail"

    @pytest.mark.asyncio
    async def test_execute_validation_with_warnings(
        self, mock_metadata: Mock, validation_criteria: Mock, mock_context: Mock
    ) -> None:
        """Test execute with warnings allowed."""
        # Set up context with data that passes required but fails optional criteria
        mock_context.available_inputs = {
            "warning_input": {
                "content": "Valid content",
                "confidence": 0.5,  # Below threshold (optional criterion)
                "quality_score": 0.85,
                "metadata": {"source": "test"},
            }
        }

        node = ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=validation_criteria,
            quality_threshold=0.5,  # Lower threshold so it can pass
            required_criteria_pass_rate=1.0,
            allow_warnings=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should pass with warnings
        assert result["validation_result"] == "warning"
        assert result["passed"] is True  # Warning still counts as passed
        assert result["passed_criteria"] == 2  # Required criteria passed
        assert result["failed_criteria"] == 1  # Optional criterion failed
        assert len(result["warnings"]) > 0

        # Check that required criteria passed
        criteria_results = result["criteria_results"]
        assert criteria_results["has_content"]["passed"] is True
        assert criteria_results["has_confidence"]["passed"] is False
        assert criteria_results["has_metadata"]["passed"] is True

    @pytest.mark.asyncio
    async def test_execute_strict_mode(
        self, mock_metadata: Mock, validation_criteria: Mock, mock_context: Mock
    ) -> None:
        """Test execute in strict mode."""
        # Set up context with data that would normally pass with warnings
        mock_context.available_inputs = {
            "strict_input": {
                "content": "Valid content",
                "confidence": 0.5,  # Below threshold
                "quality_score": 0.85,
                "metadata": {"source": "test"},
            }
        }

        node = ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=validation_criteria,
            quality_threshold=0.8,
            required_criteria_pass_rate=1.0,
            allow_warnings=True,
            strict_mode=True,  # Strict mode enabled
        )

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should fail in strict mode
        assert result["validation_result"] == "fail"
        assert result["passed"] is False
        assert result["failed_criteria"] > 0

    @pytest.mark.asyncio
    async def test_execute_quality_threshold(
        self, mock_metadata: Mock, validation_criteria: Mock, mock_context: Mock
    ) -> None:
        """Test execute with quality threshold enforcement."""
        # Create criteria with mixed weights - some pass, some fail
        mixed_criteria = [
            ValidationCriteria(
                name="always_pass",
                validator=lambda data: True,  # Always passes
                weight=0.1,  # Low weight
                required=True,
            ),
            ValidationCriteria(
                name="always_fail",
                validator=lambda data: False,  # Always fails
                weight=0.9,  # High weight
                required=True,
            ),
        ]

        node = ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=mixed_criteria,
            quality_threshold=0.8,  # High threshold
            required_criteria_pass_rate=0.5,  # Allow 50% of required to pass
            allow_warnings=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should fail due to low quality score (0.1 / 1.0 = 0.1 < 0.8)
        assert result["validation_result"] == "fail"
        assert result["quality_score"] < 0.8
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_execute_required_criteria_pass_rate(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute with required criteria pass rate enforcement."""
        # Create multiple required criteria, some will fail
        criteria = [
            ValidationCriteria(
                name="always_pass",
                validator=lambda data: True,
                weight=1.0,
                required=True,
            ),
            ValidationCriteria(
                name="always_fail",
                validator=lambda data: False,
                weight=1.0,
                required=True,
            ),
            ValidationCriteria(
                name="optional_pass",
                validator=lambda data: True,
                weight=1.0,
                required=False,
            ),
        ]

        node = ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=criteria,
            quality_threshold=0.0,  # Low threshold
            required_criteria_pass_rate=0.8,  # Need 80% of required to pass
            allow_warnings=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should fail because only 50% of required criteria pass (1 out of 2)
        assert result["validation_result"] == "fail"
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_execute_no_available_inputs(self, validator_node: Mock) -> None:
        """Test execute with no available inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.correlation_id = "test-correlation"
        context.workflow_id = "test-workflow"
        context.cognitive_classification = {"speed": "adaptive"}
        context.task_classification = Mock()
        context.execution_path = []
        context.resource_usage = {}
        context.add_to_execution_path = Mock()
        context.update_resource_usage = Mock()
        context.available_inputs = {}  # No inputs

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await validator_node.execute(context)

        # Should still run but validate empty data
        assert "validation_result" in result
        assert result["validated_data"] == {}

    @pytest.mark.asyncio
    async def test_execute_criterion_evaluation_error(
        self, mock_metadata: Mock, mock_context: Mock
    ) -> None:
        """Test execute handles criterion evaluation errors gracefully."""

        # Create a criterion that will raise an exception
        def failing_validator(data: Mock) -> None:
            raise ValueError("Validation failed")

        criteria = [
            ValidationCriteria(
                name="failing_criterion",
                validator=failing_validator,
                weight=1.0,
                required=True,
                error_message="This will fail",
            )
        ]

        node = ValidatorNode(
            metadata=mock_metadata,
            node_name="test_validator",
            validation_criteria=criteria,
            quality_threshold=0.8,
            required_criteria_pass_rate=1.0,
            allow_warnings=True,
            strict_mode=False,
        )

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await node.execute(mock_context)

        # Should handle the error gracefully
        assert result["validation_result"] == "fail"
        assert result["failed_criteria"] == 1
        assert len(result["warnings"]) > 0

        # Check that the error was captured in criteria results
        criteria_results = result["criteria_results"]
        assert criteria_results["failing_criterion"]["passed"] is False
        assert (
            "Validation error" in criteria_results["failing_criterion"]["error_message"]
        )

    @pytest.mark.asyncio
    async def test_execute_calls_pre_post_hooks(
        self, validator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute calls pre and post execution hooks."""
        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ):
            result = await validator_node.execute(mock_context)

        # Pre-execute should add to execution path
        mock_context.add_to_execution_path.assert_called_once_with("test_validator")

        # Post-execute should update resource usage
        mock_context.update_resource_usage.assert_called()

    @pytest.mark.asyncio
    async def test_execute_validation_failure_invalid_context(
        self, validator_node: Mock
    ) -> None:
        """Test execute fails with invalid context."""
        invalid_context = Mock(spec=NodeExecutionContext)
        invalid_context.correlation_id = ""  # Invalid
        invalid_context.workflow_id = ""
        invalid_context.cognitive_classification = {}
        invalid_context.task_classification = None

        with pytest.raises(ValueError, match="Context validation failed"):
            await validator_node.execute(invalid_context)

    @pytest.mark.asyncio
    async def test_execute_best_quality_input_selection(
        self, validator_node: Mock
    ) -> None:
        """Test execute selects highest quality input for validation."""
        context = Mock(spec=NodeExecutionContext)
        context.correlation_id = "test-correlation"
        context.workflow_id = "test-workflow"
        context.cognitive_classification = {"speed": "adaptive"}
        context.task_classification = Mock()
        context.execution_path = []
        context.resource_usage = {}
        context.add_to_execution_path = Mock()
        context.update_resource_usage = Mock()

        # Multiple inputs with different quality scores
        context.available_inputs = {
            "low_quality": {
                "content": "Low quality content",
                "confidence": 0.5,
                "quality_score": 0.6,
                "metadata": {"source": "low"},
            },
            "high_quality": {
                "content": "High quality content",
                "confidence": 0.9,
                "quality_score": 0.95,
                "metadata": {"source": "high"},
            },
            "medium_quality": {
                "content": "Medium quality content",
                "confidence": 0.7,
                "quality_score": 0.8,
                "metadata": {"source": "medium"},
            },
        }

        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await validator_node.execute(context)

        # Should validate the highest quality input
        validated_data = result["validated_data"]
        assert validated_data["content"] == "High quality content"
        assert validated_data["quality_score"] == 0.95
        assert validated_data["metadata"]["source"] == "high"

    @pytest.mark.asyncio
    async def test_execute_timing_measurement(
        self, validator_node: Mock, mock_context: Mock
    ) -> None:
        """Test execute measures validation timing."""
        with patch(
            "cognivault.orchestration.nodes.validator_node.emit_validation_completed"
        ) as mock_emit:
            result = await validator_node.execute(mock_context)

        # Should include timing information
        assert "validation_time_ms" in result
        assert result["validation_time_ms"] >= 0

        # Event should also include timing
        call_args = mock_emit.call_args[1]
        assert "validation_time_ms" in call_args
        assert call_args["validation_time_ms"] >= 0


class TestValidatorNodeCanHandle:
    """Test ValidatorNode can_handle method."""

    @pytest.fixture
    def validator_node(self) -> Any:
        """Create a ValidatorNode for testing."""
        metadata = Mock(spec=AgentMetadata)
        metadata.execution_pattern = "validator"

        criteria = [
            ValidationCriteria(
                name="has_data",
                validator=lambda data: len(data) > 0,
                weight=1.0,
                required=True,
            )
        ]

        return ValidatorNode(
            metadata=metadata,
            node_name="test_validator",
            validation_criteria=criteria,
            quality_threshold=0.8,
            required_criteria_pass_rate=1.0,
            allow_warnings=True,
            strict_mode=False,
        )

    def test_can_handle_with_valid_inputs(self, validator_node: Mock) -> None:
        """Test can_handle returns True with valid inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "input1": {"content": "Valid content", "confidence": 0.8},
            "input2": {"content": "Another valid input", "confidence": 0.7},
        }

        assert validator_node.can_handle(context) is True

    def test_can_handle_with_no_inputs(self, validator_node: Mock) -> None:
        """Test can_handle returns False with no inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {}

        assert validator_node.can_handle(context) is False

    def test_can_handle_with_empty_inputs(self, validator_node: Mock) -> None:
        """Test can_handle returns False with empty inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "input1": {},  # Empty dict
            "input2": None,  # None value
        }

        assert validator_node.can_handle(context) is False

    def test_can_handle_with_non_dict_inputs(self, validator_node: Mock) -> None:
        """Test can_handle returns False with non-dict inputs."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "input1": "string_input",  # Not a dict
            "input2": ["list", "input"],  # Not a dict
        }

        assert validator_node.can_handle(context) is False

    def test_can_handle_with_mixed_inputs(self, validator_node: Mock) -> None:
        """Test can_handle returns True when at least one input is valid."""
        context = Mock(spec=NodeExecutionContext)
        context.available_inputs = {
            "invalid1": "string_input",
            "invalid2": {},
            "valid": {"content": "Valid content", "confidence": 0.8},
        }

        assert validator_node.can_handle(context) is True

    def test_can_handle_evaluation_error(self, validator_node: Mock) -> None:
        """Test can_handle returns False when evaluation raises exception."""
        context = Mock(spec=NodeExecutionContext)
        # Set up mock to raise exception when accessed
        context.available_inputs = Mock()
        context.available_inputs.__bool__ = Mock(side_effect=Exception("Access failed"))

        assert validator_node.can_handle(context) is False
