"""
Comprehensive tests for Pydantic API models migration.

Tests validation, serialization, and backward compatibility of external schemas.
"""

import pytest
from pydantic import ValidationError

from tests.factories.api_model_factories import APIModelFactory


class TestWorkflowRequestValidation:
    """Test WorkflowRequest Pydantic validation."""

    def test_valid_minimal_request(self) -> None:
        """Test valid minimal request."""
        request = APIModelFactory.create_valid_workflow_request(query="Test query")
        assert request.query == "Test query"
        assert request.agents is None
        assert request.execution_config is None
        assert request.correlation_id is None

    def test_valid_full_request(self) -> None:
        """Test valid full request."""
        request = APIModelFactory.create_valid_workflow_request(
            query="Test query",
            agents=["refiner", "historian"],
            execution_config={"timeout_seconds": 30},
            correlation_id="test-123",
        )
        assert request.query == "Test query"
        assert request.agents == ["refiner", "historian"]
        assert request.execution_config == {"timeout_seconds": 30}
        assert request.correlation_id == "test-123"

    def test_query_validation(self) -> None:
        """Test query field validation."""
        # Empty query should fail
        with pytest.raises(ValidationError, match="at least 1 character"):
            APIModelFactory.create_valid_workflow_request(query="")

        # Very long query should fail
        with pytest.raises(ValidationError, match="at most 10000 characters"):
            APIModelFactory.create_valid_workflow_request(query="x" * 10001)

    def test_agents_validation(self) -> None:
        """Test agents field validation."""
        # Empty agents list should fail
        with pytest.raises(ValidationError, match="agents list cannot be empty"):
            APIModelFactory.create_valid_workflow_request(query="test", agents=[])

        # Invalid agent names should fail
        with pytest.raises(ValidationError, match="Invalid agents"):
            APIModelFactory.create_valid_workflow_request(
                query="test", agents=["invalid_agent"]
            )

        # Duplicate agents should fail
        with pytest.raises(ValidationError, match="Duplicate agents"):
            APIModelFactory.create_valid_workflow_request(
                query="test", agents=["refiner", "refiner"]
            )

    def test_correlation_id_validation(self) -> None:
        """Test correlation_id pattern validation."""
        # Invalid pattern should fail
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.create_valid_workflow_request(
                query="test", correlation_id="invalid@id"
            )

        # Too long should fail
        with pytest.raises(ValidationError, match="at most 100 characters"):
            APIModelFactory.create_valid_workflow_request(
                query="test", correlation_id="x" * 101
            )

    def test_execution_config_validation(self) -> None:
        """Test execution_config validation."""
        # Invalid timeout should fail
        with pytest.raises(
            ValidationError, match="timeout_seconds must be a positive number"
        ):
            APIModelFactory.create_valid_workflow_request(
                query="test", execution_config={"timeout_seconds": -1}
            )

        # Timeout too high should fail
        with pytest.raises(ValidationError, match="timeout_seconds cannot exceed 600"):
            APIModelFactory.create_valid_workflow_request(
                query="test", execution_config={"timeout_seconds": 700}
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            APIModelFactory.create_valid_workflow_request(
                query="test", extra_field="not_allowed"
            )

    def test_to_dict_method(self) -> None:
        """Test backward compatibility to_dict method."""
        request = APIModelFactory.create_valid_workflow_request(
            query="test", agents=["refiner"]
        )
        data = request.to_dict()
        assert isinstance(data, dict)
        assert data["query"] == "test"
        assert data["agents"] == ["refiner"]


class TestWorkflowResponseValidation:
    """Test WorkflowResponse Pydantic validation."""

    def test_valid_completed_response(self) -> None:
        """Test valid completed response."""
        response = APIModelFactory.generate_valid_workflow_response(
            status="completed",
            agent_outputs={"refiner": "output"},
            execution_time_seconds=10.5,
        )
        assert response.status == "completed"
        assert response.agent_outputs == {"refiner": "output"}

    def test_valid_failed_response(self) -> None:
        """Test valid failed response."""
        response = APIModelFactory.generate_valid_workflow_response(
            status="failed",
            agent_outputs={},
            execution_time_seconds=5.0,
            error_message="Test error",
        )
        assert response.status == "failed"
        assert response.error_message == "Test error"

    def test_workflow_id_pattern_validation(self) -> None:
        """Test workflow_id UUID pattern validation."""
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.generate_valid_workflow_response(
                workflow_id="invalid-uuid",
                status="completed",
                agent_outputs={"refiner": "output"},
                execution_time_seconds=1.0,
            )

    def test_status_pattern_validation(self) -> None:
        """Test status field validation."""
        with pytest.raises(ValidationError, match="String should match pattern"):
            APIModelFactory.generate_valid_workflow_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="invalid_status",
                agent_outputs={"refiner": "output"},
                execution_time_seconds=1.0,
            )

    def test_status_consistency_validation(self) -> None:
        """Test cross-field status validation."""
        # Failed status without error message should fail
        with pytest.raises(
            ValidationError, match="error_message is required when status is 'failed'"
        ):
            APIModelFactory.generate_valid_workflow_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="failed",
                agent_outputs={},
                execution_time_seconds=1.0,
            )

        # Completed status with empty outputs should fail
        with pytest.raises(
            ValidationError,
            match="agent_outputs cannot be empty when status is 'completed'",
        ):
            APIModelFactory.generate_valid_workflow_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="completed",
                agent_outputs={},
                execution_time_seconds=1.0,
            )

    def test_agent_outputs_validation(self) -> None:
        """Test agent_outputs validation."""
        # Empty output string should fail
        with pytest.raises(ValidationError, match="cannot be empty"):
            APIModelFactory.generate_valid_workflow_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="completed",
                agent_outputs={"refiner": ""},
                execution_time_seconds=1.0,
            )

    def test_execution_time_validation(self) -> None:
        """Test execution_time_seconds validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.generate_valid_workflow_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="completed",
                agent_outputs={"refiner": "output"},
                execution_time_seconds=-1.0,
            )


class TestStatusResponseValidation:
    """Test StatusResponse Pydantic validation."""

    def test_valid_running_status(self) -> None:
        """Test valid running status."""
        response = APIModelFactory.generate_valid_status_response(
            status="running",
            progress_percentage=50.0,
            current_agent="critic",
            estimated_completion_seconds=30.0,
        )
        assert response.status == "running"
        assert response.current_agent == "critic"

    def test_valid_completed_status(self) -> None:
        """Test valid completed status."""
        response = APIModelFactory.generate_valid_status_response(
            status="completed",
            progress_percentage=100.0,
            current_agent=None,
            estimated_completion_seconds=None,
        )
        assert response.status == "completed"
        assert response.progress_percentage == 100.0

    def test_progress_percentage_range(self) -> None:
        """Test progress_percentage range validation."""
        # Below 0 should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.generate_valid_status_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="running",
                progress_percentage=-1.0,
            )

        # Above 100 should fail
        with pytest.raises(ValidationError, match="less than or equal to 100"):
            APIModelFactory.generate_valid_status_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="running",
                progress_percentage=101.0,
            )

    def test_status_consistency_validation(self) -> None:
        """Test status consistency with progress and current_agent."""
        # Completed status must have 100% progress
        with pytest.raises(
            ValidationError,
            match="progress_percentage must be 100.0 when status is 'completed'",
        ):
            APIModelFactory.generate_valid_status_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="completed",
                progress_percentage=99.0,
                current_agent=None,
                estimated_completion_seconds=None,
            )

        # Failed status should not have 100% progress
        with pytest.raises(
            ValidationError,
            match="progress_percentage should not be 100.0 when status is 'failed'",
        ):
            APIModelFactory.generate_valid_status_response(
                workflow_id="550e8400-e29b-41d4-a716-446655440000",
                status="failed",
                progress_percentage=100.0,
                current_agent=None,
                estimated_completion_seconds=None,
            )


class TestCompletionRequestValidation:
    """Test CompletionRequest Pydantic validation."""

    def test_valid_minimal_request(self) -> None:
        """Test valid minimal completion request."""
        request = APIModelFactory.generate_valid_completion_request(
            prompt="Test prompt"
        )
        assert request.prompt == "Test prompt"
        assert request.model is None

    def test_prompt_validation(self) -> None:
        """Test prompt field validation."""
        # Empty prompt should fail
        with pytest.raises(ValidationError, match="at least 1 character"):
            APIModelFactory.generate_valid_completion_request(prompt="")

        # Very long prompt should fail
        with pytest.raises(ValidationError, match="at most 50000 characters"):
            APIModelFactory.generate_valid_completion_request(prompt="x" * 50001)

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens validation."""
        # Zero tokens should fail
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIModelFactory.generate_valid_completion_request(
                prompt="test", max_tokens=0
            )

        # Too many tokens should fail
        with pytest.raises(ValidationError, match="less than or equal to 32000"):
            APIModelFactory.generate_valid_completion_request(
                prompt="test", max_tokens=50000
            )

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        # Below 0 should fail
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.generate_valid_completion_request(
                prompt="test", temperature=-0.1
            )

        # Above 2.0 should fail
        with pytest.raises(ValidationError, match="less than or equal to 2"):
            APIModelFactory.generate_valid_completion_request(
                prompt="test", temperature=2.1
            )


class TestCompletionResponseValidation:
    """Test CompletionResponse Pydantic validation."""

    def test_valid_response(self) -> None:
        """Test valid completion response."""
        response = APIModelFactory.generate_valid_completion_response(
            completion="Test completion",
            token_usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            response_time_ms=1500.0,
        )
        assert response.completion == "Test completion"
        assert response.token_usage["total_tokens"] == 30

    def test_token_usage_validation(self) -> None:
        """Test token_usage validation."""
        # Missing required keys should fail
        with pytest.raises(ValidationError, match="token_usage must contain keys"):
            APIModelFactory.generate_valid_completion_response(
                completion="test",
                token_usage={"prompt_tokens": 10},
                response_time_ms=1000.0,
            )

        # Incorrect total calculation should fail
        with pytest.raises(
            ValidationError, match="total_tokens must equal prompt_tokens"
        ):
            APIModelFactory.generate_valid_completion_response(
                completion="test",
                token_usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 25,
                },
                response_time_ms=1000.0,
            )

    def test_response_time_validation(self) -> None:
        """Test response_time_ms validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.generate_valid_completion_response(
                completion="test",
                token_usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                response_time_ms=-1.0,
            )


class TestLLMProviderValidation:
    """Test LLMProviderInfo Pydantic validation."""

    def test_valid_provider(self) -> None:
        """Test valid provider."""
        provider = APIModelFactory.generate_valid_llm_provider_info(
            models=["gpt-4", "gpt-3.5-turbo"],
            cost_per_token=0.00003,
        )
        assert provider.name == "openai"
        assert len(provider.models) == 2

    def test_models_validation(self) -> None:
        """Test models field validation."""
        # Empty models list should fail
        with pytest.raises(ValidationError, match="at least 1 item"):
            APIModelFactory.generate_valid_llm_provider_info(
                models=[],
            )

        # Duplicate models should fail
        with pytest.raises(ValidationError, match="Duplicate model names"):
            APIModelFactory.generate_valid_llm_provider_info(
                models=["gpt-4", "gpt-4"],
            )

        # Invalid model name format should fail
        with pytest.raises(ValidationError, match="Invalid model name format"):
            APIModelFactory.generate_valid_llm_provider_info(
                models=["gpt@4"],
            )

    def test_cost_per_token_validation(self) -> None:
        """Test cost_per_token validation."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            APIModelFactory.generate_valid_llm_provider_info(
                models=["gpt-4"],
                cost_per_token=-0.1,
            )


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_all_models_have_to_dict(self) -> None:
        """Test that all external models have to_dict method."""
        models = [
            APIModelFactory.create_valid_workflow_request(query="test"),
            APIModelFactory.generate_valid_workflow_response(
                status="completed",
                agent_outputs={"refiner": "output"},
                execution_time_seconds=1.0,
            ),
            APIModelFactory.generate_valid_status_response(
                status="completed",
                progress_percentage=100.0,
                current_agent=None,
                estimated_completion_seconds=None,
            ),
            APIModelFactory.generate_valid_completion_request(),
            APIModelFactory.generate_valid_completion_response(
                completion="test",
                token_usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                },
                response_time_ms=1000.0,
            ),
            APIModelFactory.generate_valid_llm_provider_info(models=["gpt-4"]),
        ]

        for model in models:
            assert hasattr(model, "to_dict")
            data = model.to_dict()
            assert isinstance(data, dict)

    def test_serialization_compatibility(self) -> None:
        """Test JSON serialization works correctly."""
        request = APIModelFactory.create_valid_workflow_request(
            query="test", agents=["refiner"]
        )

        # Should be able to serialize/deserialize
        data = request.model_dump()
        restored = APIModelFactory.create_valid_workflow_request(**data)
        assert restored.query == request.query
        assert restored.agents == request.agents
