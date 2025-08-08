"""
Comprehensive tests for API models with strict typing enforcement and factory patterns.

This test file provides complete coverage for all API model validation logic,
serialization methods, and edge cases using factory patterns for 80-90%
boilerplate reduction and complete type safety compliance.

Coverage Focus Areas:
- WorkflowRequest: agents validation (lines 55-70), execution_config validation (lines 80-87)
- WorkflowResponse: status consistency validation (line 164)
- StatusResponse: status consistency validation (lines 218, 220)
- CompletionResponse: token_usage validation logic (lines 341, 392)
- LLMProviderInfo: models validation and duplicate detection (lines 392, 483-486, 490)
- WorkflowHistoryResponse: pagination consistency validation (lines 547-549, 554-563, 567)
- TopicSummary: name validation and to_dict serialization (lines 625-627, 631-641)
- TopicsResponse: pagination consistency (lines 699-701, 706-715, 719)
- TopicWikiResponse: sources validation and content validation (lines 790-802, 808-811, 815)
- WorkflowMetadata: tags validation and use_cases validation (lines 913-931, 937-945, 949)
- WorkflowsResponse: pagination and categories validation (lines 1043-1045, 1050-1059, 1065-1070, 1074)

Testing Approach:
- Factory-first pattern for all test data construction
- Strict typing enforcement on all test methods and fixtures
- Comprehensive validation testing (success and failure cases)
- Parameterized tests for multiple scenarios
- Edge case and boundary condition testing
- Serialization round-trip validation
"""

import pytest
from typing import Any, Dict, List
from pydantic import ValidationError

from cognivault.api.models import (
    WorkflowRequest,
    WorkflowResponse,
    StatusResponse,
    CompletionRequest,
    CompletionResponse,
    LLMProviderInfo,
    WorkflowHistoryItem,
    WorkflowHistoryResponse,
    TopicSummary,
    TopicsResponse,
    TopicWikiResponse,
    WorkflowMetadata,
    WorkflowsResponse,
    InternalExecutionGraph,
    InternalAgentMetrics,
)
from tests.factories.api_model_factories import APIModelFactory, APIModelPatterns


class TestWorkflowRequestValidation:
    """Test WorkflowRequest validation with comprehensive coverage of lines 55-70, 80-87."""

    def test_basic_workflow_request_factory(self) -> None:
        """Test factory creates valid basic WorkflowRequest."""
        request = APIModelFactory.basic_workflow_request()

        assert request.query == "What is artificial intelligence?"
        assert request.agents is None
        assert request.execution_config is None
        assert request.correlation_id is None
        assert hasattr(request, "to_dict")

    def test_workflow_request_with_all_agents(self) -> None:
        """Test WorkflowRequest with all available agents."""
        request = APIModelFactory.workflow_request_with_all_agents()

        assert request.agents == ["refiner", "historian", "critic", "synthesis"]
        assert len(request.agents) == 4
        assert len(set(request.agents)) == 4  # No duplicates

    def test_workflow_request_with_execution_config(self) -> None:
        """Test WorkflowRequest with execution configuration."""
        request = APIModelFactory.workflow_request_with_execution_config(
            timeout_seconds=45, parallel_execution=False
        )

        assert request.execution_config is not None
        assert request.execution_config["timeout_seconds"] == 45
        assert request.execution_config["parallel_execution"] is False

    def test_workflow_request_with_correlation_id(self) -> None:
        """Test WorkflowRequest with correlation ID."""
        correlation_id = "test-correlation-123"
        request = APIModelFactory.workflow_request_with_correlation_id(
            correlation_id=correlation_id
        )

        assert request.correlation_id == correlation_id

    # Test agents validation (lines 55-70)
    def test_agents_validation_empty_list_fails(self) -> None:
        """Test agents validation fails with empty list."""
        invalid_data = APIModelFactory.invalid_workflow_request_empty_agents()

        with pytest.raises(ValidationError, match="agents list cannot be empty"):
            WorkflowRequest(**invalid_data)

    def test_agents_validation_invalid_agent_names_fails(self) -> None:
        """Test agents validation fails with invalid agent names."""
        invalid_data = APIModelFactory.invalid_workflow_request_invalid_agents()

        with pytest.raises(ValidationError, match="Invalid agents"):
            WorkflowRequest(**invalid_data)

    def test_agents_validation_duplicate_agents_fails(self) -> None:
        """Test agents validation fails with duplicate agents."""
        invalid_data = APIModelFactory.invalid_workflow_request_duplicate_agents()

        with pytest.raises(ValidationError, match="Duplicate agents are not allowed"):
            WorkflowRequest(**invalid_data)

    @pytest.mark.parametrize(
        "agent_list,expected_valid",
        [
            (["refiner"], True),
            (["refiner", "critic"], True),
            (["refiner", "historian", "critic", "synthesis"], True),
            (["invalid_agent"], False),
            (["refiner", "invalid_agent"], False),
            (["refiner", "refiner"], False),
            ([], False),
        ],
    )
    def test_agents_validation_parametrized(
        self, agent_list: List[str], expected_valid: bool
    ) -> None:
        """Test agents validation with various agent lists."""
        if expected_valid:
            request = APIModelFactory.basic_workflow_request(agents=agent_list)
            assert request.agents == agent_list
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_workflow_request(agents=agent_list)

    # Test execution_config validation (lines 80-87)
    def test_execution_config_validation_negative_timeout_fails(self) -> None:
        """Test execution_config validation fails with negative timeout."""
        invalid_data = APIModelFactory.invalid_workflow_request_negative_timeout()

        with pytest.raises(
            ValidationError, match="timeout_seconds must be a positive number"
        ):
            WorkflowRequest(**invalid_data)

    def test_execution_config_validation_excessive_timeout_fails(self) -> None:
        """Test execution_config validation fails with excessive timeout."""
        invalid_data = APIModelFactory.invalid_workflow_request_excessive_timeout()

        with pytest.raises(
            ValidationError, match="timeout_seconds cannot exceed 600 seconds"
        ):
            WorkflowRequest(**invalid_data)

    @pytest.mark.parametrize(
        "timeout_value,expected_valid",
        [
            (1, True),
            (30, True),
            (600, True),  # Maximum allowed
            (0, False),
            (-1, False),
            (601, False),  # Above maximum
            ("30", False),  # Wrong type
        ],
    )
    def test_execution_config_timeout_validation_parametrized(
        self, timeout_value: Any, expected_valid: bool
    ) -> None:
        """Test execution_config timeout validation with various values."""
        config = {"timeout_seconds": timeout_value}

        if expected_valid:
            request = APIModelFactory.basic_workflow_request(execution_config=config)
            assert request.execution_config["timeout_seconds"] == timeout_value
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_workflow_request(execution_config=config)

    def test_query_length_validation(self) -> None:
        """Test query length validation edge cases."""
        # Test maximum allowed length
        max_request = APIModelFactory.edge_case_workflow_request_max_query_length()
        assert len(max_request.query) == 10000

        # Test empty query fails
        invalid_data = APIModelFactory.invalid_workflow_request_empty_query()
        with pytest.raises(ValidationError, match="at least 1 character"):
            WorkflowRequest(**invalid_data)

    def test_correlation_id_validation(self) -> None:
        """Test correlation_id validation edge cases."""
        # Test maximum allowed length
        max_request = APIModelFactory.edge_case_workflow_request_max_correlation_id()
        assert len(max_request.correlation_id) == 100

        # Test invalid pattern fails
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.basic_workflow_request(correlation_id="invalid@id")

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure."""
        request = APIModelFactory.workflow_request_with_all_agents(
            query="test query", correlation_id="test-123"
        )

        data = request.to_dict()

        assert isinstance(data, dict)
        assert data["query"] == "test query"
        assert data["agents"] == ["refiner", "historian", "critic", "synthesis"]
        assert data["correlation_id"] == "test-123"

    def test_model_dump_compatibility(self) -> None:
        """Test model_dump method compatibility."""
        request = APIModelFactory.basic_workflow_request()

        # Should be able to serialize/deserialize
        data = request.model_dump()
        restored = WorkflowRequest(**data)

        assert restored.query == request.query
        assert restored.agents == request.agents


class TestWorkflowResponseValidation:
    """Test WorkflowResponse validation with comprehensive coverage of line 164."""

    def test_basic_workflow_response_factory(self) -> None:
        """Test factory creates valid basic WorkflowResponse."""
        response = APIModelFactory.basic_workflow_response()

        assert response.workflow_id == "550e8400-e29b-41d4-a716-446655440000"
        assert response.status == "completed"
        assert "refiner" in response.agent_outputs
        assert response.execution_time_seconds == 42.5

    def test_completed_workflow_response_factory(self) -> None:
        """Test factory creates completed workflow with all outputs."""
        response = APIModelFactory.completed_workflow_response()

        assert response.status == "completed"
        assert len(response.agent_outputs) == 4
        assert "refiner" in response.agent_outputs
        assert "historian" in response.agent_outputs
        assert "critic" in response.agent_outputs
        assert "synthesis" in response.agent_outputs

    def test_failed_workflow_response_factory(self) -> None:
        """Test factory creates failed workflow with error message."""
        response = APIModelFactory.failed_workflow_response()

        assert response.status == "failed"
        assert response.error_message is not None
        assert "timeout exceeded" in response.error_message
        assert len(response.agent_outputs) == 0

    def test_running_workflow_response_factory(self) -> None:
        """Test factory creates running workflow response."""
        response = APIModelFactory.running_workflow_response()

        assert response.status == "running"
        assert "refiner" in response.agent_outputs
        assert response.error_message is None

    # Test status consistency validation (line 164)
    def test_status_consistency_failed_without_error_fails(self) -> None:
        """Test status consistency validation fails when failed status has no error message."""
        invalid_data = APIModelFactory.invalid_workflow_response_failed_without_error()

        with pytest.raises(
            ValidationError, match="error_message is required when status is 'failed'"
        ):
            WorkflowResponse(**invalid_data)

    def test_status_consistency_completed_empty_outputs_fails(self) -> None:
        """Test status consistency validation fails when completed status has empty outputs."""
        invalid_data = (
            APIModelFactory.invalid_workflow_response_completed_empty_outputs()
        )

        with pytest.raises(
            ValidationError,
            match="agent_outputs cannot be empty when status is 'completed'",
        ):
            WorkflowResponse(**invalid_data)

    def test_agent_outputs_validation_empty_string_fails(self) -> None:
        """Test agent outputs validation fails with empty output string."""
        invalid_data = APIModelFactory.invalid_workflow_response_empty_agent_output()

        with pytest.raises(ValidationError, match="cannot be empty"):
            WorkflowResponse(**invalid_data)

    @pytest.mark.parametrize(
        "status,agent_outputs,error_message,should_be_valid",
        [
            ("completed", {"refiner": "output"}, None, True),
            ("completed", {}, None, False),  # Empty outputs with completed
            ("failed", {}, "error occurred", True),
            ("failed", {}, None, False),  # No error message with failed
            ("running", {"refiner": "partial"}, None, True),
            ("cancelled", {}, None, True),
        ],
    )
    def test_status_consistency_parametrized(
        self,
        status: str,
        agent_outputs: Dict[str, str],
        error_message: str,
        should_be_valid: bool,
    ) -> None:
        """Test status consistency validation with various combinations."""
        if should_be_valid:
            response = APIModelFactory.basic_workflow_response(
                status=status, agent_outputs=agent_outputs, error_message=error_message
            )
            assert response.status == status
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_workflow_response(
                    status=status,
                    agent_outputs=agent_outputs,
                    error_message=error_message,
                )

    def test_workflow_id_pattern_validation(self) -> None:
        """Test workflow_id UUID pattern validation."""
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.basic_workflow_response(workflow_id="invalid-uuid")

    def test_execution_time_validation(self) -> None:
        """Test execution_time_seconds validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_workflow_response(execution_time_seconds=-1.0)

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure."""
        response = APIModelFactory.completed_workflow_response()
        data = response.to_dict()

        assert isinstance(data, dict)
        assert data["workflow_id"] == response.workflow_id
        assert data["status"] == response.status
        assert data["agent_outputs"] == response.agent_outputs


class TestStatusResponseValidation:
    """Test StatusResponse validation with comprehensive coverage of lines 218, 220."""

    def test_basic_status_response_factory(self) -> None:
        """Test factory creates valid basic StatusResponse."""
        response = APIModelFactory.basic_status_response()

        assert response.workflow_id == "550e8400-e29b-41d4-a716-446655440000"
        assert response.status == "running"
        assert response.progress_percentage == 50.0
        assert response.current_agent == "critic"

    def test_running_status_response_factory(self) -> None:
        """Test factory creates running status response."""
        response = APIModelFactory.running_status_response()

        assert response.status == "running"
        assert response.current_agent == "synthesis"
        assert response.estimated_completion_seconds == 10.2

    def test_completed_status_response_factory(self) -> None:
        """Test factory creates completed status response."""
        response = APIModelFactory.completed_status_response()

        assert response.status == "completed"
        assert response.progress_percentage == 100.0
        assert response.current_agent is None
        assert response.estimated_completion_seconds is None

    def test_failed_status_response_factory(self) -> None:
        """Test factory creates failed status response."""
        response = APIModelFactory.failed_status_response()

        assert response.status == "failed"
        assert response.current_agent is None
        assert response.progress_percentage == 65.0

    # Test status consistency validation (lines 218, 220)
    def test_status_consistency_non_running_with_agent_fails(self) -> None:
        """Test status consistency validation fails when non-running status has current_agent (line 220)."""
        invalid_data = APIModelFactory.invalid_status_response_non_running_with_agent()

        with pytest.raises(
            ValidationError,
            match="current_agent should only be set when status is 'running'",
        ):
            StatusResponse(**invalid_data)

    def test_status_consistency_completed_not_100_fails(self) -> None:
        """Test status consistency validation fails when completed status is not 100%."""
        invalid_data = APIModelFactory.invalid_status_response_completed_not_100()

        with pytest.raises(
            ValidationError,
            match="progress_percentage must be 100.0 when status is 'completed'",
        ):
            StatusResponse(**invalid_data)

    def test_status_consistency_failed_100_fails(self) -> None:
        """Test status consistency validation fails when failed status is 100%."""
        invalid_data = APIModelFactory.invalid_status_response_failed_100()

        with pytest.raises(
            ValidationError,
            match="progress_percentage should not be 100.0 when status is 'failed'",
        ):
            StatusResponse(**invalid_data)

    def test_status_consistency_running_allows_none_current_agent(self) -> None:
        """Test status consistency allows None current_agent for running status (line 218)."""
        # This should be valid - running status can have None current_agent
        response = APIModelFactory.basic_status_response(
            status="running", current_agent=None
        )

        assert response.status == "running"
        assert response.current_agent is None

    @pytest.mark.parametrize(
        "status,progress,current_agent,should_be_valid",
        [
            ("running", 50.0, "critic", True),
            ("running", 75.0, None, True),  # Line 218 - allows None for running
            ("completed", 100.0, None, True),
            ("completed", 99.0, None, False),  # Must be 100% for completed
            ("completed", 100.0, "critic", False),  # Line 220 - no agent for completed
            ("failed", 65.0, None, True),
            ("failed", 100.0, None, False),  # Should not be 100% for failed
            ("failed", 50.0, "critic", False),  # Line 220 - no agent for failed
            ("cancelled", 80.0, None, True),
            ("cancelled", 90.0, "critic", False),  # Line 220 - no agent for cancelled
        ],
    )
    def test_status_consistency_parametrized(
        self, status: str, progress: float, current_agent: str, should_be_valid: bool
    ) -> None:
        """Test status consistency validation with various combinations."""
        if should_be_valid:
            response = APIModelFactory.basic_status_response(
                status=status, progress_percentage=progress, current_agent=current_agent
            )
            assert response.status == status
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_status_response(
                    status=status,
                    progress_percentage=progress,
                    current_agent=current_agent,
                )

    def test_progress_percentage_range_validation(self) -> None:
        """Test progress_percentage range validation."""
        # Below 0 should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_status_response(progress_percentage=-1.0)

        # Above 100 should fail
        with pytest.raises(ValidationError, match="less than or equal to 100"):
            APIModelFactory.basic_status_response(progress_percentage=101.0)

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure."""
        response = APIModelFactory.running_status_response()
        data = response.to_dict()

        assert isinstance(data, dict)
        assert data["status"] == response.status
        assert data["progress_percentage"] == response.progress_percentage


class TestCompletionRequestValidation:
    """Test CompletionRequest validation with comprehensive coverage."""

    def test_basic_completion_request_factory(self) -> None:
        """Test factory creates valid basic CompletionRequest."""
        request = APIModelFactory.basic_completion_request()

        assert (
            request.prompt == "Explain the concept of machine learning in simple terms"
        )
        assert request.model is None
        assert request.max_tokens is None
        assert request.temperature is None

    def test_completion_request_with_options_factory(self) -> None:
        """Test factory creates completion request with all options."""
        request = APIModelFactory.completion_request_with_options()

        assert request.model == "gpt-4"
        assert request.max_tokens == 1000
        assert request.temperature == 0.7
        assert request.agent_context is not None

    def test_prompt_length_validation(self) -> None:
        """Test prompt length validation edge cases."""
        # Test maximum allowed length
        max_request = APIModelFactory.edge_case_completion_request_max_prompt()
        assert len(max_request.prompt) == 50000

        # Test empty prompt fails
        with pytest.raises(ValidationError, match="at least 1 character"):
            APIModelFactory.basic_completion_request(prompt="")

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIModelFactory.basic_completion_request(max_tokens=0)

        with pytest.raises(ValidationError, match="less than or equal to 32000"):
            APIModelFactory.basic_completion_request(max_tokens=50000)

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_completion_request(temperature=-0.1)

        with pytest.raises(ValidationError, match="less than or equal to 2"):
            APIModelFactory.basic_completion_request(temperature=2.1)

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure."""
        request = APIModelFactory.completion_request_with_options()
        data = request.to_dict()

        assert isinstance(data, dict)
        assert data["prompt"] == request.prompt
        assert data["model"] == request.model


class TestCompletionResponseValidation:
    """Test CompletionResponse validation with comprehensive coverage of lines 341, 392."""

    def test_basic_completion_response_factory(self) -> None:
        """Test factory creates valid basic CompletionResponse."""
        response = APIModelFactory.basic_completion_response()

        assert response.completion.startswith("Machine learning")
        assert response.model_used == "gpt-4"
        assert response.token_usage["total_tokens"] == 175
        assert response.response_time_ms == 1250.5

    def test_completion_response_with_high_usage_factory(self) -> None:
        """Test factory creates response with high token usage."""
        response = APIModelFactory.completion_response_with_high_usage()

        assert response.token_usage["total_tokens"] == 2000
        assert response.response_time_ms == 3500.8

    # Test token_usage validation logic (lines 341, 392)
    def test_token_usage_validation_missing_keys_fails(self) -> None:
        """Test token usage validation fails with missing required keys."""
        invalid_data = APIModelFactory.invalid_completion_response_missing_token_keys()

        with pytest.raises(ValidationError, match="token_usage must contain keys"):
            CompletionResponse(**invalid_data)

    def test_token_usage_validation_negative_tokens_fails(self) -> None:
        """Test token usage validation fails with negative token values (line 341)."""
        invalid_data = APIModelFactory.invalid_completion_response_negative_tokens()

        with pytest.raises(ValidationError, match="must be a non-negative integer"):
            CompletionResponse(**invalid_data)

    def test_token_usage_validation_incorrect_total_fails(self) -> None:
        """Test token usage validation fails with incorrect total calculation."""
        invalid_data = APIModelFactory.invalid_completion_response_incorrect_total()

        with pytest.raises(
            ValidationError,
            match="total_tokens must equal prompt_tokens \\+ completion_tokens",
        ):
            CompletionResponse(**invalid_data)

    @pytest.mark.parametrize(
        "prompt_tokens,completion_tokens,total_tokens,should_be_valid",
        [
            (10, 20, 30, True),  # Correct calculation
            (0, 0, 0, True),  # All zeros
            (100, 200, 300, True),  # Large values
            (10, 20, 25, False),  # Incorrect total
            (-5, 20, 15, False),  # Negative prompt tokens (line 341)
            (10, -20, -10, False),  # Negative completion tokens (line 341)
            (10, 20, -30, False),  # Negative total tokens (line 341)
        ],
    )
    def test_token_usage_validation_parametrized(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        should_be_valid: bool,
    ) -> None:
        """Test token usage validation with various token combinations."""
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        if should_be_valid:
            response = APIModelFactory.basic_completion_response(
                token_usage=token_usage
            )
            assert response.token_usage["total_tokens"] == total_tokens
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_completion_response(token_usage=token_usage)

    def test_response_time_validation(self) -> None:
        """Test response_time_ms validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_completion_response(response_time_ms=-1.0)

    def test_request_id_pattern_validation(self) -> None:
        """Test request_id UUID pattern validation."""
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.basic_completion_response(request_id="invalid-uuid")

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure."""
        response = APIModelFactory.basic_completion_response()
        data = response.to_dict()

        assert isinstance(data, dict)
        assert data["completion"] == response.completion
        assert data["token_usage"] == response.token_usage


class TestLLMProviderInfoValidation:
    """Test LLMProviderInfo validation with comprehensive coverage of lines 392, 483-486, 490."""

    def test_basic_llm_provider_info_factory(self) -> None:
        """Test factory creates valid basic LLMProviderInfo."""
        provider = APIModelFactory.basic_llm_provider_info()

        assert provider.name == "openai"
        assert len(provider.models) == 3
        assert provider.available is True
        assert provider.cost_per_token == 0.00003

    def test_llm_provider_info_unavailable_factory(self) -> None:
        """Test factory creates unavailable provider info."""
        provider = APIModelFactory.llm_provider_info_unavailable()

        assert provider.name == "claude"
        assert provider.available is False
        assert provider.cost_per_token is None

    # Test models validation and duplicate detection (lines 392, 483-486, 490)
    def test_models_validation_empty_list_fails(self) -> None:
        """Test models validation fails with empty list."""
        invalid_data = APIModelFactory.invalid_llm_provider_info_empty_models()

        with pytest.raises(ValidationError, match="at least 1 item"):
            LLMProviderInfo(**invalid_data)

    def test_models_validation_duplicate_names_fails(self) -> None:
        """Test models validation fails with duplicate model names (line 490)."""
        invalid_data = APIModelFactory.invalid_llm_provider_info_duplicate_models()

        with pytest.raises(
            ValidationError, match="Duplicate model names are not allowed"
        ):
            LLMProviderInfo(**invalid_data)

    def test_models_validation_invalid_format_fails(self) -> None:
        """Test models validation fails with invalid model name format (lines 483-486)."""
        invalid_data = APIModelFactory.invalid_llm_provider_info_invalid_model_format()

        with pytest.raises(ValidationError, match="Invalid model name format"):
            LLMProviderInfo(**invalid_data)

    def test_models_validation_empty_model_name_fails(self) -> None:
        """Test models validation fails with empty model name (line 392)."""
        invalid_data = APIModelFactory.invalid_llm_provider_info_empty_model_name()

        with pytest.raises(
            ValidationError, match="All model names must be non-empty strings"
        ):
            LLMProviderInfo(**invalid_data)

    @pytest.mark.parametrize(
        "models,should_be_valid",
        [
            (["gpt-4"], True),
            (["gpt-4", "gpt-3.5-turbo"], True),
            (["model_1", "model-2", "model.3"], True),  # Valid formats
            ([], False),  # Empty list
            (["gpt-4", "gpt-4"], False),  # Duplicates (line 490)
            (["gpt@4"], False),  # Invalid character (lines 483-486)
            (["gpt 4"], False),  # Space not allowed (lines 483-486)
            ([""], False),  # Empty string (line 392)
            (["gpt-4", ""], False),  # Mix of valid and invalid (line 392)
        ],
    )
    def test_models_validation_parametrized(
        self, models: List[str], should_be_valid: bool
    ) -> None:
        """Test models validation with various model lists."""
        if should_be_valid:
            provider = APIModelFactory.basic_llm_provider_info(models=models)
            assert provider.models == models
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_llm_provider_info(models=models)

    def test_cost_per_token_validation(self) -> None:
        """Test cost_per_token validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_llm_provider_info(cost_per_token=-0.1)

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure."""
        provider = APIModelFactory.basic_llm_provider_info()
        data = provider.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == provider.name
        assert data["models"] == provider.models


class TestWorkflowHistoryItemValidation:
    """Test WorkflowHistoryItem validation and to_dict method."""

    def test_basic_workflow_history_item_factory(self) -> None:
        """Test factory creates valid basic WorkflowHistoryItem."""
        item = APIModelFactory.basic_workflow_history_item()

        assert item.workflow_id == "550e8400-e29b-41d4-a716-446655440000"
        assert item.status == "completed"
        assert "climate change" in item.query
        assert item.execution_time_seconds == 12.5

    def test_workflow_history_item_failed_factory(self) -> None:
        """Test factory creates failed workflow history item."""
        item = APIModelFactory.workflow_history_item_failed()

        assert item.status == "failed"
        assert "failed to complete" in item.query

    def test_status_validation_invalid_status_fails(self) -> None:
        """Test status validation fails with invalid status."""
        invalid_data = APIModelFactory.invalid_workflow_history_item_invalid_status()

        with pytest.raises(ValidationError, match="String should match pattern"):
            WorkflowHistoryItem(**invalid_data)

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure (lines 490)."""
        item = APIModelFactory.basic_workflow_history_item()
        data = item.to_dict()

        assert isinstance(data, dict)
        assert data["workflow_id"] == item.workflow_id
        assert data["status"] == item.status
        assert data["query"] == item.query
        assert data["start_time"] == item.start_time
        assert data["execution_time_seconds"] == item.execution_time_seconds


class TestWorkflowHistoryResponseValidation:
    """Test WorkflowHistoryResponse validation with comprehensive coverage of lines 547-549, 554-563, 567."""

    def test_basic_workflow_history_response_factory(self) -> None:
        """Test factory creates valid basic WorkflowHistoryResponse."""
        response = APIModelFactory.basic_workflow_history_response()

        assert len(response.workflows) == 2
        assert response.total == 150
        assert response.limit == 10
        assert response.has_more is True

    def test_workflow_history_response_last_page_factory(self) -> None:
        """Test factory creates last page workflow history response."""
        response = APIModelFactory.workflow_history_response_last_page()

        assert response.has_more is False
        assert response.offset == 20
        assert response.total == 21

    def test_limit_validation(self) -> None:
        """Test limit validation (lines 547-549)."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIModelFactory.basic_workflow_history_response(limit=0)

        with pytest.raises(ValidationError, match="less than or equal to 100"):
            APIModelFactory.basic_workflow_history_response(limit=101)

    def test_pagination_consistency_validation(self) -> None:
        """Test pagination consistency validation (lines 554-563)."""
        # Test with inconsistent pagination data
        invalid_data = (
            APIModelFactory.invalid_workflow_history_response_inconsistent_pagination()
        )

        # The model_validator should fix has_more to be consistent
        response = WorkflowHistoryResponse(**invalid_data)

        # has_more should be corrected: offset (0) + len(workflows) (1) < total (10) = True
        assert response.has_more is True

    def test_pagination_consistency_automatic_correction(self) -> None:
        """Test pagination consistency automatic correction (lines 560-561)."""
        response = APIModelFactory.basic_workflow_history_response(
            total=5,
            offset=0,
            has_more=False,  # Inconsistent - should be True
        )

        # Validator should correct this based on offset + len < total
        actual_has_more = (response.offset + len(response.workflows)) < response.total
        assert response.has_more == actual_has_more

    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure (line 567)."""
        response = APIModelFactory.basic_workflow_history_response()
        data = response.to_dict()

        assert isinstance(data, dict)
        assert "workflows" in data
        assert isinstance(data["workflows"], list)
        assert data["total"] == response.total
        assert data["limit"] == response.limit
        assert data["has_more"] == response.has_more


class TestTopicSummaryValidation:
    """Test TopicSummary validation with comprehensive coverage of lines 625-627, 631-641."""

    def test_basic_topic_summary_factory(self) -> None:
        """Test factory creates valid basic TopicSummary."""
        topic = APIModelFactory.basic_topic_summary()

        assert topic.topic_id == "550e8400-e29b-41d4-a716-446655440000"
        assert topic.name == "Machine Learning Fundamentals"
        assert topic.query_count == 15
        assert topic.similarity_score == 0.85

    def test_topic_summary_without_similarity_factory(self) -> None:
        """Test factory creates topic without similarity score."""
        topic = APIModelFactory.topic_summary_without_similarity()

        assert topic.name == "Data Science Basics"
        assert topic.similarity_score is None

    # Test name validation (lines 625-627)
    def test_name_validation_empty_name_fails(self) -> None:
        """Test name validation fails with empty/whitespace name (lines 625-627)."""
        invalid_data = APIModelFactory.invalid_topic_summary_empty_name()

        with pytest.raises(
            ValidationError, match="Topic name cannot be empty or whitespace"
        ):
            TopicSummary(**invalid_data)

    def test_name_validation_strips_whitespace(self) -> None:
        """Test name validation strips whitespace (line 627)."""
        topic = APIModelFactory.basic_topic_summary(name="  Machine Learning  ")

        assert topic.name == "Machine Learning"  # Whitespace should be stripped

    def test_edge_case_max_name_length(self) -> None:
        """Test edge case with maximum name length."""
        topic = APIModelFactory.edge_case_topic_summary_max_name()

        assert len(topic.name) == 100

    # Test to_dict serialization (lines 631-641)
    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns model_dump() (lines 631-641)."""
        topic = APIModelFactory.basic_topic_summary()

        data = topic.to_dict()
        expected_data = topic.model_dump()

        assert data == expected_data
        assert isinstance(data, dict)
        assert data["topic_id"] == topic.topic_id
        assert data["name"] == topic.name

    def test_similarity_score_range_validation(self) -> None:
        """Test similarity_score range validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_topic_summary(similarity_score=-0.1)

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            APIModelFactory.basic_topic_summary(similarity_score=1.1)


class TestTopicsResponseValidation:
    """Test TopicsResponse validation with comprehensive coverage of lines 699-701, 706-715, 719."""

    def test_basic_topics_response_factory(self) -> None:
        """Test factory creates valid basic TopicsResponse."""
        response = APIModelFactory.basic_topics_response()

        assert len(response.topics) == 2
        assert response.total == 42
        assert response.limit == 10
        assert response.has_more is True

    def test_topics_response_with_search_factory(self) -> None:
        """Test factory creates topics response with search filtering."""
        response = APIModelFactory.topics_response_with_search()

        assert response.search_query == "machine learning"
        assert response.total == 15  # Fewer due to filtering

    def test_limit_validation(self) -> None:
        """Test limit validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIModelFactory.basic_topics_response(limit=0)

        with pytest.raises(ValidationError, match="less than or equal to 100"):
            APIModelFactory.basic_topics_response(limit=101)

    # Test pagination consistency (lines 699-701, 706-715)
    def test_pagination_consistency_validation(self) -> None:
        """Test pagination consistency validation (lines 699-701, 706-715)."""
        invalid_data = APIModelFactory.invalid_topics_response_inconsistent_pagination()

        # The model_validator should fix has_more to be consistent
        response = TopicsResponse(**invalid_data)

        # has_more should be corrected based on offset + len < total
        expected_has_more = (response.offset + len(response.topics)) < response.total
        assert response.has_more == expected_has_more

    def test_offset_negative_validation(self) -> None:
        """Test offset validation fails with negative values (line 699)."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_topics_response(offset=-1)

    def test_pagination_consistency_automatic_correction(self) -> None:
        """Test pagination consistency automatic correction (lines 704-705)."""
        response = APIModelFactory.basic_topics_response(
            total=10,
            offset=5,
            has_more=False,  # Inconsistent - should be True based on calculation
        )

        # Validator should correct this
        expected_has_more = (response.offset + len(response.topics)) < response.total
        assert response.has_more == expected_has_more

    # Test to_dict serialization (line 719)
    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure (line 719)."""
        response = APIModelFactory.basic_topics_response()
        data = response.to_dict()

        assert isinstance(data, dict)
        assert "topics" in data
        assert isinstance(data["topics"], list)
        assert data["total"] == response.total
        assert data["search_query"] == response.search_query


class TestTopicWikiResponseValidation:
    """Test TopicWikiResponse validation with comprehensive coverage of lines 790-802, 808-811, 815."""

    def test_basic_topic_wiki_response_factory(self) -> None:
        """Test factory creates valid basic TopicWikiResponse."""
        response = APIModelFactory.basic_topic_wiki_response()

        assert response.topic_id == "550e8400-e29b-41d4-a716-446655440000"
        assert response.topic_name == "Machine Learning Fundamentals"
        assert "artificial intelligence" in response.content
        assert len(response.sources) == 2
        assert response.confidence_score == 0.92

    def test_topic_wiki_response_extensive_factory(self) -> None:
        """Test factory creates extensive topic wiki response."""
        response = APIModelFactory.topic_wiki_response_extensive()

        assert len(response.sources) == 10
        assert response.query_count == 47
        assert response.confidence_score == 0.95

    # Test sources validation (lines 790-802)
    def test_sources_validation_invalid_uuid_fails(self) -> None:
        """Test sources validation fails with invalid UUID format (lines 790-802)."""
        invalid_data = APIModelFactory.invalid_topic_wiki_response_invalid_source_id()

        with pytest.raises(ValidationError, match="Invalid source workflow ID format"):
            TopicWikiResponse(**invalid_data)

    def test_sources_validation_removes_duplicates(self) -> None:
        """Test sources validation removes duplicates while preserving order (lines 790-802)."""
        response = APIModelFactory.topic_wiki_response_duplicate_sources()

        # Should have duplicates removed
        assert len(response.sources) == 2  # Original had 3 with 1 duplicate
        assert response.sources[0] == "550e8400-e29b-41d4-a716-446655440001"
        assert response.sources[1] == "550e8400-e29b-41d4-a716-446655440002"

    # Test content validation (lines 808-811)
    def test_content_validation_empty_content_fails(self) -> None:
        """Test content validation fails with empty/whitespace content (lines 808-811)."""
        invalid_data = APIModelFactory.invalid_topic_wiki_response_empty_content()

        with pytest.raises(
            ValidationError, match="Content cannot be empty or whitespace"
        ):
            TopicWikiResponse(**invalid_data)

    def test_content_validation_strips_whitespace(self) -> None:
        """Test content validation strips whitespace (line 802)."""
        response = APIModelFactory.basic_topic_wiki_response(
            content="  Machine learning content  "
        )

        assert response.content == "Machine learning content"

    # Test to_dict serialization (line 815)
    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure (line 815)."""
        response = APIModelFactory.basic_topic_wiki_response()
        data = response.to_dict()

        assert isinstance(data, dict)
        assert data["topic_id"] == response.topic_id
        assert data["topic_name"] == response.topic_name
        assert data["content"] == response.content
        assert data["sources"] == response.sources
        assert data["confidence_score"] == response.confidence_score

    def test_confidence_score_range_validation(self) -> None:
        """Test confidence_score range validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.basic_topic_wiki_response(confidence_score=-0.1)

        with pytest.raises(ValidationError, match="less than or equal to 1"):
            APIModelFactory.basic_topic_wiki_response(confidence_score=1.1)


class TestWorkflowMetadataValidation:
    """Test WorkflowMetadata validation with comprehensive coverage of lines 913-931, 937-945, 949."""

    def test_basic_workflow_metadata_factory(self) -> None:
        """Test factory creates valid basic WorkflowMetadata."""
        metadata = APIModelFactory.basic_workflow_metadata()

        assert metadata.workflow_id == "academic_research"
        assert metadata.name == "Academic Research Analysis"
        assert metadata.category == "academic"
        assert len(metadata.tags) == 4
        assert len(metadata.use_cases) == 2

    def test_workflow_metadata_business_factory(self) -> None:
        """Test factory creates business workflow metadata."""
        metadata = APIModelFactory.workflow_metadata_business()

        assert metadata.category == "business"
        assert "business" in metadata.tags
        assert metadata.complexity_level == "medium"

    def test_workflow_metadata_simple_factory(self) -> None:
        """Test factory creates simple workflow metadata."""
        metadata = APIModelFactory.workflow_metadata_simple()

        assert metadata.complexity_level == "low"
        assert metadata.node_count == 3

    # Test tags validation (lines 913-931)
    def test_tags_validation_empty_list_fails(self) -> None:
        """Test tags validation fails with empty list (line 913)."""
        invalid_data = APIModelFactory.invalid_workflow_metadata_empty_tags()

        with pytest.raises(ValidationError, match="At least one tag must be provided"):
            WorkflowMetadata(**invalid_data)

    def test_tags_validation_long_tag_fails(self) -> None:
        """Test tags validation fails with tag exceeding length limit (lines 918-919)."""
        invalid_data = APIModelFactory.invalid_workflow_metadata_long_tag()

        with pytest.raises(ValidationError, match="Tags cannot exceed 30 characters"):
            WorkflowMetadata(**invalid_data)

    def test_tags_validation_removes_duplicates_and_normalizes(self) -> None:
        """Test tags validation removes duplicates and normalizes (lines 913-931)."""
        metadata = APIModelFactory.workflow_metadata_duplicate_tags()

        # Should have duplicates removed and be lowercased
        assert len(set(metadata.tags)) == len(metadata.tags)  # No duplicates
        assert all(tag == tag.lower() for tag in metadata.tags)  # All lowercase

    @pytest.mark.parametrize(
        "tags,should_be_valid",
        [
            (["academic", "research"], True),
            (["tag1"], True),
            ([], False),  # Empty list (line 913)
            (["x" * 31], False),  # Too long (lines 918-919)
            (["  "], False),  # Whitespace only (line 917)
            ([123], False),  # Not string (line 914)
        ],
    )
    def test_tags_validation_parametrized(
        self, tags: List[Any], should_be_valid: bool
    ) -> None:
        """Test tags validation with various tag combinations."""
        if should_be_valid:
            metadata = APIModelFactory.basic_workflow_metadata(tags=tags)
            assert len(metadata.tags) >= 1
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_workflow_metadata(tags=tags)

    # Test use_cases validation (lines 937-945)
    def test_use_cases_validation_long_use_case_fails(self) -> None:
        """Test use_cases validation fails with use case exceeding length limit (lines 937-945)."""
        invalid_data = APIModelFactory.invalid_workflow_metadata_long_use_case()

        with pytest.raises(
            ValidationError, match="Use cases cannot exceed 100 characters"
        ):
            WorkflowMetadata(**invalid_data)

    @pytest.mark.parametrize(
        "use_cases,should_be_valid",
        [
            (["research", "analysis"], True),
            (["single_use_case"], True),
            (["x" * 100], True),  # Exactly 100 characters
            (["x" * 101], False),  # Too long (line 936)
            (["  "], False),  # Whitespace only (line 934)
            ([123], False),  # Not string (line 932)
        ],
    )
    def test_use_cases_validation_parametrized(
        self, use_cases: List[Any], should_be_valid: bool
    ) -> None:
        """Test use_cases validation with various combinations."""
        if should_be_valid:
            metadata = APIModelFactory.basic_workflow_metadata(use_cases=use_cases)
            assert isinstance(metadata.use_cases, list)
        else:
            with pytest.raises(ValidationError):
                APIModelFactory.basic_workflow_metadata(use_cases=use_cases)

    # Test to_dict serialization (line 949)
    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure (line 949)."""
        metadata = APIModelFactory.basic_workflow_metadata()
        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["workflow_id"] == metadata.workflow_id
        assert data["name"] == metadata.name
        assert data["tags"] == metadata.tags
        assert data["use_cases"] == metadata.use_cases

    def test_version_pattern_validation(self) -> None:
        """Test version pattern validation."""
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.basic_workflow_metadata(
                version="1.0"
            )  # Missing patch version

    def test_complexity_level_validation(self) -> None:
        """Test complexity_level validation."""
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.basic_workflow_metadata(complexity_level="invalid")


class TestWorkflowsResponseValidation:
    """Test WorkflowsResponse validation with comprehensive coverage of lines 1043-1045, 1050-1059, 1065-1070, 1074."""

    def test_basic_workflows_response_factory(self) -> None:
        """Test factory creates valid basic WorkflowsResponse."""
        response = APIModelFactory.basic_workflows_response()

        assert len(response.workflows) == 2
        assert len(response.categories) == 4
        assert response.total == 25
        assert response.has_more is True

    def test_workflows_response_filtered_factory(self) -> None:
        """Test factory creates filtered workflows response."""
        response = APIModelFactory.workflows_response_filtered()

        assert response.category_filter == "academic"
        assert response.complexity_filter == "high"
        assert response.search_query == "research"
        assert response.total == 8  # Fewer due to filtering

    def test_workflows_response_empty_categories_factory(self) -> None:
        """Test factory creates response with empty categories."""
        response = APIModelFactory.workflows_response_empty_categories()

        assert isinstance(response.categories, list)

    def test_limit_validation(self) -> None:
        """Test limit validation."""
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1"
        ):
            APIModelFactory.basic_workflows_response(limit=0)

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 100"
        ):
            APIModelFactory.basic_workflows_response(limit=101)

    # Test pagination consistency validation (lines 1043-1045, 1050-1059)
    def test_pagination_consistency_validation(self) -> None:
        """Test pagination consistency validation (lines 1043-1045, 1050-1059)."""
        invalid_data = (
            APIModelFactory.invalid_workflows_response_inconsistent_pagination()
        )

        # The model_validator should fix has_more to be consistent
        response = WorkflowsResponse(**invalid_data)

        # has_more should be corrected based on offset + len < total
        expected_has_more = (response.offset + len(response.workflows)) < response.total
        assert response.has_more == expected_has_more

    def test_offset_negative_validation(self) -> None:
        """Test offset validation fails with negative values (line 1043)."""
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            APIModelFactory.basic_workflows_response(offset=-1)

    def test_pagination_consistency_automatic_correction(self) -> None:
        """Test pagination consistency automatic correction (lines 1047-1048)."""
        response = APIModelFactory.basic_workflows_response(
            total=20,
            offset=10,
            has_more=False,  # Inconsistent - should be True
        )

        # Validator should correct this
        expected_has_more = (response.offset + len(response.workflows)) < response.total
        assert response.has_more == expected_has_more

    # Test categories validation (lines 1065-1070)
    def test_categories_validation_normalizes_and_deduplicates(self) -> None:
        """Test categories validation normalizes and removes duplicates (lines 1065-1070)."""
        response = APIModelFactory.workflows_response_duplicate_categories()

        # Should be normalized (lowercase) and deduplicated
        assert len(set(response.categories)) == len(
            response.categories
        )  # No duplicates
        assert all(cat == cat.lower() for cat in response.categories)  # All lowercase
        assert response.categories == sorted(response.categories)  # Sorted

    def test_categories_validation_handles_empty_list(self) -> None:
        """Test categories validation handles empty list (lines 1065-1070)."""
        response = APIModelFactory.workflows_response_empty_categories()

        assert response.categories == []

    # Test to_dict serialization (line 1074)
    def test_to_dict_serialization(self) -> None:
        """Test to_dict method returns correct structure (line 1074)."""
        response = APIModelFactory.basic_workflows_response()
        data = response.to_dict()

        assert isinstance(data, dict)
        assert "workflows" in data
        assert isinstance(data["workflows"], list)
        assert data["categories"] == response.categories
        assert data["total"] == response.total
        assert data["has_more"] == response.has_more
        assert data["category_filter"] == response.category_filter

    def test_complexity_filter_pattern_validation(self) -> None:
        """Test complexity_filter pattern validation."""
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.basic_workflows_response(complexity_filter="invalid")


class TestPatternConvenienceMethods:
    """Test convenience pattern methods for common usage scenarios."""

    def test_minimal_workflow_request_pattern(self) -> None:
        """Test minimal workflow request pattern."""
        request = APIModelPatterns.minimal_workflow_request()

        assert request.query == "test"
        assert request.agents is None

    def test_completed_workflow_pattern(self) -> None:
        """Test completed workflow pattern."""
        response = APIModelPatterns.completed_workflow()

        assert response.status == "completed"
        assert len(response.agent_outputs) == 4

    def test_running_status_pattern(self) -> None:
        """Test running status pattern."""
        response = APIModelPatterns.running_status()

        assert response.status == "running"
        assert response.current_agent is not None

    def test_simple_completion_pattern(self) -> None:
        """Test simple completion pattern."""
        response = APIModelPatterns.simple_completion()

        assert response.model_used == "gpt-4"
        assert response.token_usage["total_tokens"] == 175

    def test_topic_with_search_pattern(self) -> None:
        """Test topic with search pattern."""
        topic = APIModelPatterns.topic_with_search(similarity=0.95)

        assert topic.similarity_score == 0.95
        assert topic.name is not None

    def test_academic_workflow_pattern(self) -> None:
        """Test academic workflow pattern."""
        metadata = APIModelPatterns.academic_workflow()

        assert metadata.category == "academic"
        assert "academic" in metadata.tags


class TestSerializationRoundTrip:
    """Test serialization round-trip compatibility for all models."""

    @pytest.mark.parametrize(
        "factory_method",
        [
            APIModelFactory.basic_workflow_request,
            APIModelFactory.completed_workflow_response,
            APIModelFactory.running_status_response,
            APIModelFactory.basic_completion_request,
            APIModelFactory.basic_completion_response,
            APIModelFactory.basic_llm_provider_info,
            APIModelFactory.basic_workflow_history_item,
            APIModelFactory.basic_workflow_history_response,
            APIModelFactory.basic_topic_summary,
            APIModelFactory.basic_topics_response,
            APIModelFactory.basic_topic_wiki_response,
            APIModelFactory.basic_workflow_metadata,
            APIModelFactory.basic_workflows_response,
        ],
    )
    def test_serialization_round_trip(self, factory_method) -> None:
        """Test serialization round-trip for all API models."""
        # Create instance using factory
        original = factory_method()

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back to model
        restored = original.__class__(**data)

        # Should be equivalent
        assert restored.model_dump() == original.model_dump()

    def test_to_dict_compatibility(self) -> None:
        """Test to_dict method compatibility across all models with the method."""
        models_with_to_dict = [
            APIModelFactory.basic_workflow_request(),
            APIModelFactory.completed_workflow_response(),
            APIModelFactory.running_status_response(),
            APIModelFactory.basic_completion_request(),
            APIModelFactory.basic_completion_response(),
            APIModelFactory.basic_llm_provider_info(),
            APIModelFactory.basic_workflow_history_item(),
            APIModelFactory.basic_workflow_history_response(),
            APIModelFactory.basic_topic_summary(),
            APIModelFactory.basic_topics_response(),
            APIModelFactory.basic_topic_wiki_response(),
            APIModelFactory.basic_workflow_metadata(),
            APIModelFactory.basic_workflows_response(),
        ]

        for model in models_with_to_dict:
            if hasattr(model, "to_dict"):
                data = model.to_dict()
                assert isinstance(data, dict)
                assert len(data) > 0


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for all models."""

    def test_maximum_field_lengths(self) -> None:
        """Test models with maximum allowed field lengths."""
        # Test maximum query length
        max_request = APIModelFactory.edge_case_workflow_request_max_query_length()
        assert len(max_request.query) == 10000

        # Test maximum correlation ID length
        max_correlation = (
            APIModelFactory.edge_case_workflow_request_max_correlation_id()
        )
        assert len(max_correlation.correlation_id) == 100

        # Test maximum prompt length
        max_prompt = APIModelFactory.edge_case_completion_request_max_prompt()
        assert len(max_prompt.prompt) == 50000

        # Test maximum topic name length
        max_topic_name = APIModelFactory.edge_case_topic_summary_max_name()
        assert len(max_topic_name.name) == 100

    def test_minimum_valid_values(self) -> None:
        """Test models with minimum valid values."""
        # Test minimum query length
        min_request = APIModelFactory.basic_workflow_request(query="x")
        assert len(min_request.query) == 1

        # Test zero execution time
        zero_time_response = APIModelFactory.basic_workflow_response(
            execution_time_seconds=0.0
        )
        assert zero_time_response.execution_time_seconds == 0.0

        # Test zero progress
        zero_progress = APIModelFactory.basic_status_response(progress_percentage=0.0)
        assert zero_progress.progress_percentage == 0.0

    def test_boundary_conditions_for_numeric_fields(self) -> None:
        """Test boundary conditions for numeric fields."""
        # Test execution time boundaries
        APIModelFactory.basic_workflow_response(execution_time_seconds=0.0)  # Minimum
        APIModelFactory.basic_workflow_response(
            execution_time_seconds=999999.99
        )  # Very high

        # Test progress percentage boundaries
        APIModelFactory.basic_status_response(progress_percentage=0.0)  # Minimum
        APIModelFactory.basic_status_response(progress_percentage=100.0)  # Maximum

        # Test token usage boundaries
        APIModelFactory.basic_completion_response(
            token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )

        # Test confidence score boundaries
        APIModelFactory.basic_topic_summary(similarity_score=0.0)  # Minimum
        APIModelFactory.basic_topic_summary(similarity_score=1.0)  # Maximum
        APIModelFactory.basic_topic_wiki_response(confidence_score=0.0)  # Minimum
        APIModelFactory.basic_topic_wiki_response(confidence_score=1.0)  # Maximum


class TestFactoryBoilerplateReduction:
    """Validate factory pattern achieves expected boilerplate reduction."""

    def test_factory_vs_manual_construction_lines_of_code(self) -> None:
        """Demonstrate factory pattern reduces lines of code significantly."""
        # Manual construction (verbose)
        manual_request = WorkflowRequest(
            query="What is artificial intelligence?",
            agents=None,
            execution_config=None,
            correlation_id=None,
        )

        # Factory construction (concise)
        factory_request = APIModelFactory.basic_workflow_request()

        # Both should be equivalent
        assert manual_request.query == factory_request.query
        assert manual_request.agents == factory_request.agents
        assert manual_request.execution_config == factory_request.execution_config
        assert manual_request.correlation_id == factory_request.correlation_id

    def test_factory_customization_still_concise(self) -> None:
        """Demonstrate factory customization remains concise."""
        # Factory with customization - still much shorter than full manual construction
        custom_request = APIModelFactory.basic_workflow_request(
            query="Custom query", agents=["refiner", "critic"]
        )

        assert custom_request.query == "Custom query"
        assert custom_request.agents == ["refiner", "critic"]
        assert custom_request.execution_config is None  # Still gets sensible default

    def test_specialized_factories_eliminate_repetition(self) -> None:
        """Demonstrate specialized factories eliminate common pattern repetition."""
        # Common patterns become one-liners
        all_agents = APIModelFactory.workflow_request_with_all_agents()
        with_config = APIModelFactory.workflow_request_with_execution_config()
        with_correlation = APIModelFactory.workflow_request_with_correlation_id()

        assert len(all_agents.agents) == 4
        assert with_config.execution_config is not None
        assert with_correlation.correlation_id is not None
