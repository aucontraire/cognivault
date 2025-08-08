"""Factory functions for creating API model test data objects.

This module provides factory functions for comprehensive API model testing,
eliminating boilerplate construction and ensuring consistent test data
for all external API schemas with strict typing enforcement.

Revolutionary Factory Principles:
- Complete type safety with all parameters explicitly typed
- Factory methods with sensible defaults eliminate 80%+ of manual construction
- Specialized factory methods for validation testing (valid/invalid/edge cases)
- Easy override of specific fields for test customization
- Complete elimination of unfilled parameter warnings

Coverage Focus Areas:
- WorkflowRequest: agents validation, execution_config validation
- WorkflowResponse: status consistency validation
- StatusResponse: status consistency validation
- CompletionResponse: token_usage validation logic
- LLMProviderInfo: models validation and duplicate detection
- WorkflowHistoryResponse: pagination consistency validation
- TopicSummary: name validation and to_dict serialization
- TopicsResponse: pagination consistency
- TopicWikiResponse: sources validation and content validation
- WorkflowMetadata: tags validation and use_cases validation
- WorkflowsResponse: pagination and categories validation

Usage Examples:
    # Simple valid instance - zero boilerplate
    request = APIModelFactory.basic_workflow_request()

    # With customization - only specify what matters
    request = APIModelFactory.basic_workflow_request(
        query="What is machine learning?",
        agents=["refiner", "critic"]
    )

    # Invalid data for validation testing
    invalid_data = APIModelFactory.invalid_workflow_request_empty_agents()
    with pytest.raises(ValidationError):
        WorkflowRequest(**invalid_data)

    # Edge case testing
    edge_case = APIModelFactory.edge_case_workflow_request_max_query_length()
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
)


class APIModelFactory:
    """Factory for creating API model test data with comprehensive type safety."""

    # =============================================================================
    # WorkflowRequest Factories
    # =============================================================================

    @staticmethod
    def basic_workflow_request(
        query: str = "What is artificial intelligence?",
        agents: Optional[List[str]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create basic WorkflowRequest with sensible defaults.

        Args:
            query: User query string
            agents: List of agent names to execute
            execution_config: Execution configuration parameters
            correlation_id: Request correlation identifier
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest with valid defaults
        """
        defaults = {
            "query": query,
            "agents": agents,
            "execution_config": execution_config,
            "correlation_id": correlation_id,
        }
        defaults.update(overrides)
        return WorkflowRequest(**defaults)

    @staticmethod
    def workflow_request_with_all_agents(
        query: str = "Comprehensive analysis request",
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest with all available agents.

        Args:
            query: User query string
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest with all valid agents configured
        """
        return APIModelFactory.basic_workflow_request(
            query=query,
            agents=["refiner", "historian", "critic", "synthesis"],
            **overrides,
        )

    @staticmethod
    def workflow_request_with_execution_config(
        query: str = "Request with execution config",
        timeout_seconds: int = 30,
        parallel_execution: bool = True,
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest with execution configuration.

        Args:
            query: User query string
            timeout_seconds: Execution timeout in seconds
            parallel_execution: Enable parallel execution
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest with execution configuration
        """
        return APIModelFactory.basic_workflow_request(
            query=query,
            execution_config={
                "timeout_seconds": timeout_seconds,
                "parallel_execution": parallel_execution,
            },
            **overrides,
        )

    @staticmethod
    def workflow_request_with_correlation_id(
        query: str = "Request with correlation ID",
        correlation_id: str = "req-12345-abcdef",
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest with correlation ID.

        Args:
            query: User query string
            correlation_id: Request correlation identifier
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest with correlation ID
        """
        return APIModelFactory.basic_workflow_request(
            query=query,
            correlation_id=correlation_id,
            **overrides,
        )

    @staticmethod
    def invalid_workflow_request_empty_query(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowRequest data with empty query.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {"query": "", "agents": None, "execution_config": None}
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_request_empty_agents(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowRequest data with empty agents list.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "query": "test query",
            "agents": [],  # Empty list should fail validation
            "execution_config": None,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_request_duplicate_agents(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowRequest data with duplicate agents.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "query": "test query",
            "agents": ["refiner", "refiner"],  # Duplicates should fail
            "execution_config": None,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_request_invalid_agents(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowRequest data with invalid agent names.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "query": "test query",
            "agents": ["invalid_agent", "another_invalid"],
            "execution_config": None,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_request_negative_timeout(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowRequest data with negative timeout.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "query": "test query",
            "agents": None,
            "execution_config": {"timeout_seconds": -1},
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_request_excessive_timeout(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowRequest data with excessive timeout.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "query": "test query",
            "agents": None,
            "execution_config": {"timeout_seconds": 700},  # > 600 seconds
        }
        data.update(overrides)
        return data

    @staticmethod
    def edge_case_workflow_request_max_query_length(
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest with maximum allowed query length.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest with edge case data
        """
        max_query = "x" * 10000  # Maximum allowed length
        return APIModelFactory.basic_workflow_request(query=max_query, **overrides)

    @staticmethod
    def edge_case_workflow_request_max_correlation_id(
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest with maximum allowed correlation ID length.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest with edge case data
        """
        max_correlation_id = "x" * 100  # Maximum allowed length
        return APIModelFactory.basic_workflow_request(
            correlation_id=max_correlation_id, **overrides
        )

    # =============================================================================
    # WorkflowResponse Factories
    # =============================================================================

    @staticmethod
    def basic_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "completed",
        agent_outputs: Optional[Dict[str, str]] = None,
        execution_time_seconds: float = 42.5,
        correlation_id: Optional[str] = None,
        error_message: Optional[str] = None,
        **overrides: Any,
    ) -> WorkflowResponse:
        """Create basic WorkflowResponse with sensible defaults.

        Args:
            workflow_id: Unique workflow identifier
            status: Execution status
            agent_outputs: Dict of agent outputs
            execution_time_seconds: Total execution time
            correlation_id: Request correlation identifier
            error_message: Error message if failed
            **overrides: Override any field with custom values

        Returns:
            WorkflowResponse with valid defaults
        """
        if agent_outputs is None:
            agent_outputs = {"refiner": "Refined query output"}

        defaults = {
            "workflow_id": workflow_id,
            "status": status,
            "agent_outputs": agent_outputs,
            "execution_time_seconds": execution_time_seconds,
            "correlation_id": correlation_id,
            "error_message": error_message,
        }
        defaults.update(overrides)
        return WorkflowResponse(**defaults)

    @staticmethod
    def completed_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> WorkflowResponse:
        """Create completed WorkflowResponse with all agent outputs.

        Args:
            workflow_id: Unique workflow identifier
            **overrides: Override any field with custom values

        Returns:
            WorkflowResponse with completed status and all agent outputs
        """
        return APIModelFactory.basic_workflow_response(
            workflow_id=workflow_id,
            status="completed",
            agent_outputs={
                "refiner": "Refined and clarified query",
                "historian": "Relevant historical context",
                "critic": "Critical analysis and evaluation",
                "synthesis": "Comprehensive synthesis of insights",
            },
            execution_time_seconds=45.8,
            **overrides,
        )

    @staticmethod
    def failed_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        error_message: str = "Agent 'historian' failed: timeout exceeded",
        **overrides: Any,
    ) -> WorkflowResponse:
        """Create failed WorkflowResponse with error message.

        Args:
            workflow_id: Unique workflow identifier
            error_message: Error message describing failure
            **overrides: Override any field with custom values

        Returns:
            WorkflowResponse with failed status and error message
        """
        return APIModelFactory.basic_workflow_response(
            workflow_id=workflow_id,
            status="failed",
            agent_outputs={},
            error_message=error_message,
            execution_time_seconds=12.3,
            **overrides,
        )

    @staticmethod
    def running_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> WorkflowResponse:
        """Create running WorkflowResponse.

        Args:
            workflow_id: Unique workflow identifier
            **overrides: Override any field with custom values

        Returns:
            WorkflowResponse with running status
        """
        return APIModelFactory.basic_workflow_response(
            workflow_id=workflow_id,
            status="running",
            agent_outputs={"refiner": "Partial output"},
            execution_time_seconds=15.2,
            **overrides,
        )

    @staticmethod
    def invalid_workflow_response_failed_without_error(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid WorkflowResponse data with failed status but no error message.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "failed",
            "agent_outputs": {},
            "execution_time_seconds": 10.0,
            "error_message": None,  # Should be required for failed status
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_response_completed_empty_outputs(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid WorkflowResponse data with completed status but empty outputs.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "agent_outputs": {},  # Should not be empty for completed status
            "execution_time_seconds": 20.0,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_response_empty_agent_output(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid WorkflowResponse data with empty agent output string.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "agent_outputs": {"refiner": ""},  # Empty output should fail
            "execution_time_seconds": 10.0,
        }
        data.update(overrides)
        return data

    # =============================================================================
    # StatusResponse Factories
    # =============================================================================

    @staticmethod
    def basic_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "running",
        progress_percentage: float = 50.0,
        current_agent: Optional[str] = "critic",
        estimated_completion_seconds: Optional[float] = 15.5,
        **overrides: Any,
    ) -> StatusResponse:
        """Create basic StatusResponse with sensible defaults.

        Args:
            workflow_id: Unique workflow identifier
            status: Current execution status
            progress_percentage: Execution progress (0-100)
            current_agent: Currently executing agent
            estimated_completion_seconds: Estimated completion time
            **overrides: Override any field with custom values

        Returns:
            StatusResponse with valid defaults
        """
        defaults = {
            "workflow_id": workflow_id,
            "status": status,
            "progress_percentage": progress_percentage,
            "current_agent": current_agent,
            "estimated_completion_seconds": estimated_completion_seconds,
        }
        defaults.update(overrides)
        return StatusResponse(**defaults)

    @staticmethod
    def running_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> StatusResponse:
        """Create running StatusResponse.

        Args:
            workflow_id: Unique workflow identifier
            **overrides: Override any field with custom values

        Returns:
            StatusResponse with running status
        """
        return APIModelFactory.basic_status_response(
            workflow_id=workflow_id,
            status="running",
            progress_percentage=75.0,
            current_agent="synthesis",
            estimated_completion_seconds=10.2,
            **overrides,
        )

    @staticmethod
    def completed_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> StatusResponse:
        """Create completed StatusResponse.

        Args:
            workflow_id: Unique workflow identifier
            **overrides: Override any field with custom values

        Returns:
            StatusResponse with completed status
        """
        return APIModelFactory.basic_status_response(
            workflow_id=workflow_id,
            status="completed",
            progress_percentage=100.0,
            current_agent=None,  # No current agent when completed
            estimated_completion_seconds=None,
            **overrides,
        )

    @staticmethod
    def failed_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> StatusResponse:
        """Create failed StatusResponse.

        Args:
            workflow_id: Unique workflow identifier
            **overrides: Override any field with custom values

        Returns:
            StatusResponse with failed status
        """
        return APIModelFactory.basic_status_response(
            workflow_id=workflow_id,
            status="failed",
            progress_percentage=65.0,  # Failed partway through
            current_agent=None,
            estimated_completion_seconds=None,
            **overrides,
        )

    @staticmethod
    def invalid_status_response_non_running_with_agent(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid StatusResponse data with current_agent set for non-running status.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "progress_percentage": 100.0,
            "current_agent": "synthesis",  # Should be None for completed status
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_status_response_completed_not_100(**overrides: Any) -> Dict[str, Any]:
        """Create invalid StatusResponse data with completed status but not 100% progress.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "progress_percentage": 99.0,  # Should be 100.0 for completed
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_status_response_failed_100(**overrides: Any) -> Dict[str, Any]:
        """Create invalid StatusResponse data with failed status but 100% progress.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "failed",
            "progress_percentage": 100.0,  # Should not be 100.0 for failed
        }
        data.update(overrides)
        return data

    # =============================================================================
    # CompletionRequest Factories
    # =============================================================================

    @staticmethod
    def basic_completion_request(
        prompt: str = "Explain the concept of machine learning in simple terms",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        agent_context: Optional[str] = None,
        **overrides: Any,
    ) -> CompletionRequest:
        """Create basic CompletionRequest with sensible defaults.

        Args:
            prompt: The prompt to send to the LLM
            model: LLM model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            agent_context: Additional agent context
            **overrides: Override any field with custom values

        Returns:
            CompletionRequest with valid defaults
        """
        defaults = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "agent_context": agent_context,
        }
        defaults.update(overrides)
        return CompletionRequest(**defaults)

    @staticmethod
    def completion_request_with_options(
        prompt: str = "Analyze the following data",
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **overrides: Any,
    ) -> CompletionRequest:
        """Create CompletionRequest with all optional parameters.

        Args:
            prompt: The prompt to send to the LLM
            model: LLM model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **overrides: Override any field with custom values

        Returns:
            CompletionRequest with all options configured
        """
        return APIModelFactory.basic_completion_request(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            agent_context="Previous agent outputs and workflow context",
            **overrides,
        )

    @staticmethod
    def edge_case_completion_request_max_prompt(**overrides: Any) -> CompletionRequest:
        """Create CompletionRequest with maximum allowed prompt length.

        Args:
            **overrides: Override any field with custom values

        Returns:
            CompletionRequest with edge case data
        """
        max_prompt = "x" * 50000  # Maximum allowed length
        return APIModelFactory.basic_completion_request(prompt=max_prompt, **overrides)

    # =============================================================================
    # CompletionResponse Factories
    # =============================================================================

    @staticmethod
    def basic_completion_response(
        completion: str = "Machine learning is a subset of artificial intelligence...",
        model_used: str = "gpt-4",
        token_usage: Optional[Dict[str, int]] = None,
        response_time_ms: float = 1250.5,
        request_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> CompletionResponse:
        """Create basic CompletionResponse with sensible defaults.

        Args:
            completion: Generated completion text
            model_used: The model that generated completion
            token_usage: Token usage statistics
            response_time_ms: Response time in milliseconds
            request_id: Unique request identifier
            **overrides: Override any field with custom values

        Returns:
            CompletionResponse with valid defaults
        """
        if token_usage is None:
            token_usage = {
                "prompt_tokens": 25,
                "completion_tokens": 150,
                "total_tokens": 175,
            }

        defaults = {
            "completion": completion,
            "model_used": model_used,
            "token_usage": token_usage,
            "response_time_ms": response_time_ms,
            "request_id": request_id,
        }
        defaults.update(overrides)
        return CompletionResponse(**defaults)

    @staticmethod
    def completion_response_with_high_usage(
        completion: str = "Detailed analysis with comprehensive insights...",
        **overrides: Any,
    ) -> CompletionResponse:
        """Create CompletionResponse with high token usage.

        Args:
            completion: Generated completion text
            **overrides: Override any field with custom values

        Returns:
            CompletionResponse with high token usage
        """
        return APIModelFactory.basic_completion_response(
            completion=completion,
            token_usage={
                "prompt_tokens": 500,
                "completion_tokens": 1500,
                "total_tokens": 2000,
            },
            response_time_ms=3500.8,
            **overrides,
        )

    @staticmethod
    def invalid_completion_response_missing_token_keys(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid CompletionResponse data with missing token usage keys.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "completion": "test completion",
            "model_used": "gpt-4",
            "token_usage": {
                "prompt_tokens": 10
            },  # Missing completion_tokens and total_tokens
            "response_time_ms": 1000.0,
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_completion_response_negative_tokens(**overrides: Any) -> Dict[str, Any]:
        """Create invalid CompletionResponse data with negative token values.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "completion": "test completion",
            "model_used": "gpt-4",
            "token_usage": {
                "prompt_tokens": -5,  # Negative value should fail
                "completion_tokens": 20,
                "total_tokens": 15,
            },
            "response_time_ms": 1000.0,
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_completion_response_incorrect_total(**overrides: Any) -> Dict[str, Any]:
        """Create invalid CompletionResponse data with incorrect total tokens calculation.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "completion": "test completion",
            "model_used": "gpt-4",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 25,  # Should be 30 (10 + 20)
            },
            "response_time_ms": 1000.0,
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
        }
        data.update(overrides)
        return data

    # =============================================================================
    # LLMProviderInfo Factories
    # =============================================================================

    @staticmethod
    def basic_llm_provider_info(
        name: str = "openai",
        models: Optional[List[str]] = None,
        available: bool = True,
        cost_per_token: Optional[float] = 0.00003,
        **overrides: Any,
    ) -> LLMProviderInfo:
        """Create basic LLMProviderInfo with sensible defaults.

        Args:
            name: Provider name
            models: List of available models
            available: Whether provider is available
            cost_per_token: Cost per token in USD
            **overrides: Override any field with custom values

        Returns:
            LLMProviderInfo with valid defaults
        """
        if models is None:
            models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

        defaults = {
            "name": name,
            "models": models,
            "available": available,
            "cost_per_token": cost_per_token,
        }
        defaults.update(overrides)
        return LLMProviderInfo(**defaults)

    @staticmethod
    def llm_provider_info_unavailable(
        name: str = "claude",
        **overrides: Any,
    ) -> LLMProviderInfo:
        """Create LLMProviderInfo for unavailable provider.

        Args:
            name: Provider name
            **overrides: Override any field with custom values

        Returns:
            LLMProviderInfo for unavailable provider
        """
        return APIModelFactory.basic_llm_provider_info(
            name=name,
            models=["claude-3-sonnet", "claude-3-haiku"],
            available=False,
            cost_per_token=None,
            **overrides,
        )

    @staticmethod
    def invalid_llm_provider_info_empty_models(**overrides: Any) -> Dict[str, Any]:
        """Create invalid LLMProviderInfo data with empty models list.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "name": "test_provider",
            "models": [],  # Should have at least one model
            "available": True,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_llm_provider_info_duplicate_models(**overrides: Any) -> Dict[str, Any]:
        """Create invalid LLMProviderInfo data with duplicate model names.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "name": "test_provider",
            "models": ["gpt-4", "gpt-4"],  # Duplicates should fail
            "available": True,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_llm_provider_info_invalid_model_format(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid LLMProviderInfo data with invalid model name format.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "name": "test_provider",
            "models": ["gpt@4", "invalid model name"],  # Invalid characters
            "available": True,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_llm_provider_info_empty_model_name(**overrides: Any) -> Dict[str, Any]:
        """Create invalid LLMProviderInfo data with empty model name.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "name": "test_provider",
            "models": ["gpt-4", ""],  # Empty model name should fail
            "available": True,
        }
        data.update(overrides)
        return data

    # =============================================================================
    # WorkflowHistoryItem Factories
    # =============================================================================

    @staticmethod
    def basic_workflow_history_item(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "completed",
        query: str = "Analyze the impact of climate change on agriculture",
        start_time: float = 1703097600.0,
        execution_time_seconds: float = 12.5,
        **overrides: Any,
    ) -> WorkflowHistoryItem:
        """Create basic WorkflowHistoryItem with sensible defaults.

        Args:
            workflow_id: Unique workflow identifier
            status: Workflow execution status
            query: Original query (truncated)
            start_time: Start time as Unix timestamp
            execution_time_seconds: Total execution time
            **overrides: Override any field with custom values

        Returns:
            WorkflowHistoryItem with valid defaults
        """
        defaults = {
            "workflow_id": workflow_id,
            "status": status,
            "query": query,
            "start_time": start_time,
            "execution_time_seconds": execution_time_seconds,
        }
        defaults.update(overrides)
        return WorkflowHistoryItem(**defaults)

    @staticmethod
    def workflow_history_item_failed(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440001",
        **overrides: Any,
    ) -> WorkflowHistoryItem:
        """Create failed WorkflowHistoryItem.

        Args:
            workflow_id: Unique workflow identifier
            **overrides: Override any field with custom values

        Returns:
            WorkflowHistoryItem with failed status
        """
        return APIModelFactory.basic_workflow_history_item(
            workflow_id=workflow_id,
            status="failed",
            query="Query that failed to complete",
            execution_time_seconds=8.2,
            **overrides,
        )

    @staticmethod
    def invalid_workflow_history_item_invalid_status(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid WorkflowHistoryItem data with invalid status.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "invalid_status",  # Should be one of the valid statuses
            "query": "test query",
            "start_time": 1703097600.0,
            "execution_time_seconds": 10.0,
        }
        data.update(overrides)
        return data

    # =============================================================================
    # WorkflowHistoryResponse Factories
    # =============================================================================

    @staticmethod
    def basic_workflow_history_response(
        workflows: Optional[List[WorkflowHistoryItem]] = None,
        total: int = 150,
        limit: int = 10,
        offset: int = 0,
        has_more: bool = True,
        **overrides: Any,
    ) -> WorkflowHistoryResponse:
        """Create basic WorkflowHistoryResponse with sensible defaults.

        Args:
            workflows: List of workflow history items
            total: Total number of workflows available
            limit: Maximum number of results requested
            offset: Number of results skipped
            has_more: Whether there are more results
            **overrides: Override any field with custom values

        Returns:
            WorkflowHistoryResponse with valid defaults
        """
        if workflows is None:
            workflows = [
                APIModelFactory.basic_workflow_history_item(),
                APIModelFactory.workflow_history_item_failed(),
            ]

        defaults = {
            "workflows": workflows,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
        }
        defaults.update(overrides)
        return WorkflowHistoryResponse(**defaults)

    @staticmethod
    def workflow_history_response_last_page(
        **overrides: Any,
    ) -> WorkflowHistoryResponse:
        """Create WorkflowHistoryResponse for the last page of results.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowHistoryResponse with has_more=False
        """
        workflows = [APIModelFactory.basic_workflow_history_item()]
        return APIModelFactory.basic_workflow_history_response(
            workflows=workflows,
            total=21,
            limit=10,
            offset=20,
            has_more=False,  # Last page
            **overrides,
        )

    @staticmethod
    def invalid_workflow_history_response_inconsistent_pagination(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create WorkflowHistoryResponse data with inconsistent pagination.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with inconsistent pagination data
        """
        workflows = [APIModelFactory.basic_workflow_history_item()]
        data = {
            "workflows": [w.model_dump() for w in workflows],
            "total": 10,
            "limit": 5,
            "offset": 0,
            "has_more": False,  # Inconsistent: should be True based on offset + len < total
        }
        data.update(overrides)
        return data

    # =============================================================================
    # TopicSummary Factories
    # =============================================================================

    @staticmethod
    def basic_topic_summary(
        topic_id: str = "550e8400-e29b-41d4-a716-446655440000",
        name: str = "Machine Learning Fundamentals",
        description: str = "Core concepts and principles of machine learning algorithms",
        query_count: int = 15,
        last_updated: float = 1703097600.0,
        similarity_score: Optional[float] = 0.85,
        **overrides: Any,
    ) -> TopicSummary:
        """Create basic TopicSummary with sensible defaults.

        Args:
            topic_id: Unique topic identifier
            name: Human-readable topic name
            description: Brief topic description
            query_count: Number of related queries
            last_updated: Last update timestamp
            similarity_score: Similarity score for search
            **overrides: Override any field with custom values

        Returns:
            TopicSummary with valid defaults
        """
        defaults = {
            "topic_id": topic_id,
            "name": name,
            "description": description,
            "query_count": query_count,
            "last_updated": last_updated,
            "similarity_score": similarity_score,
        }
        defaults.update(overrides)
        return TopicSummary(**defaults)

    @staticmethod
    def topic_summary_without_similarity(
        name: str = "Data Science Basics",
        **overrides: Any,
    ) -> TopicSummary:
        """Create TopicSummary without similarity score.

        Args:
            name: Human-readable topic name
            **overrides: Override any field with custom values

        Returns:
            TopicSummary without similarity score
        """
        return APIModelFactory.basic_topic_summary(
            name=name,
            description="Introduction to data science concepts and methodologies",
            similarity_score=None,
            **overrides,
        )

    @staticmethod
    def invalid_topic_summary_empty_name(**overrides: Any) -> Dict[str, Any]:
        """Create invalid TopicSummary data with empty name.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "topic_id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "   ",  # Empty/whitespace name should fail
            "description": "test description",
            "query_count": 5,
            "last_updated": 1703097600.0,
        }
        data.update(overrides)
        return data

    @staticmethod
    def edge_case_topic_summary_max_name(**overrides: Any) -> TopicSummary:
        """Create TopicSummary with maximum allowed name length.

        Args:
            **overrides: Override any field with custom values

        Returns:
            TopicSummary with edge case data
        """
        max_name = "x" * 100  # Maximum allowed length
        return APIModelFactory.basic_topic_summary(name=max_name, **overrides)

    # =============================================================================
    # TopicsResponse Factories
    # =============================================================================

    @staticmethod
    def basic_topics_response(
        topics: Optional[List[TopicSummary]] = None,
        total: int = 42,
        limit: int = 10,
        offset: int = 0,
        has_more: bool = True,
        search_query: Optional[str] = None,
        **overrides: Any,
    ) -> TopicsResponse:
        """Create basic TopicsResponse with sensible defaults.

        Args:
            topics: List of topic summaries
            total: Total number of topics available
            limit: Maximum number of results requested
            offset: Number of results skipped
            has_more: Whether there are more results
            search_query: Search query used for filtering
            **overrides: Override any field with custom values

        Returns:
            TopicsResponse with valid defaults
        """
        if topics is None:
            topics = [
                APIModelFactory.basic_topic_summary(),
                APIModelFactory.topic_summary_without_similarity(),
            ]

        defaults = {
            "topics": topics,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "search_query": search_query,
        }
        defaults.update(overrides)
        return TopicsResponse(**defaults)

    @staticmethod
    def topics_response_with_search(
        search_query: str = "machine learning",
        **overrides: Any,
    ) -> TopicsResponse:
        """Create TopicsResponse with search query filtering.

        Args:
            search_query: Search query used for filtering
            **overrides: Override any field with custom values

        Returns:
            TopicsResponse with search filtering
        """
        return APIModelFactory.basic_topics_response(
            search_query=search_query,
            total=15,  # Fewer results due to filtering
            **overrides,
        )

    @staticmethod
    def invalid_topics_response_inconsistent_pagination(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create TopicsResponse data with inconsistent pagination.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with inconsistent pagination data
        """
        topics = [APIModelFactory.basic_topic_summary()]
        data = {
            "topics": [t.model_dump() for t in topics],
            "total": 20,
            "limit": 10,
            "offset": 5,
            "has_more": False,  # Inconsistent: should be True based on offset + len < total
            "search_query": None,
        }
        data.update(overrides)
        return data

    # =============================================================================
    # TopicWikiResponse Factories
    # =============================================================================

    @staticmethod
    def basic_topic_wiki_response(
        topic_id: str = "550e8400-e29b-41d4-a716-446655440000",
        topic_name: str = "Machine Learning Fundamentals",
        content: str = "Machine learning is a subset of artificial intelligence...",
        last_updated: float = 1703097600.0,
        sources: Optional[List[str]] = None,
        query_count: int = 15,
        confidence_score: float = 0.92,
        **overrides: Any,
    ) -> TopicWikiResponse:
        """Create basic TopicWikiResponse with sensible defaults.

        Args:
            topic_id: Unique topic identifier
            topic_name: Human-readable topic name
            content: Synthesized knowledge content
            last_updated: Last update timestamp
            sources: List of source workflow IDs
            query_count: Number of contributing queries
            confidence_score: Confidence in synthesized content
            **overrides: Override any field with custom values

        Returns:
            TopicWikiResponse with valid defaults
        """
        if sources is None:
            sources = [
                "550e8400-e29b-41d4-a716-446655440001",
                "550e8400-e29b-41d4-a716-446655440002",
            ]

        defaults = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "content": content,
            "last_updated": last_updated,
            "sources": sources,
            "query_count": query_count,
            "confidence_score": confidence_score,
        }
        defaults.update(overrides)
        return TopicWikiResponse(**defaults)

    @staticmethod
    def topic_wiki_response_extensive(
        **overrides: Any,
    ) -> TopicWikiResponse:
        """Create TopicWikiResponse with extensive content and many sources.

        Args:
            **overrides: Override any field with custom values

        Returns:
            TopicWikiResponse with extensive content
        """
        content = """
        Machine learning is a subset of artificial intelligence that enables systems to learn 
        and improve from experience without being explicitly programmed. The field encompasses 
        various algorithms and statistical models that computer systems use to perform tasks 
        without specific instructions, relying on patterns and inference instead.
        
        Key concepts include supervised learning, unsupervised learning, and reinforcement 
        learning. Applications span across industries from healthcare to finance, enabling 
        predictive analytics, automation, and intelligent decision-making systems.
        """.strip()

        sources = [str(uuid.uuid4()) for _ in range(10)]  # Many source workflows

        return APIModelFactory.basic_topic_wiki_response(
            content=content,
            sources=sources,
            query_count=47,
            confidence_score=0.95,
            **overrides,
        )

    @staticmethod
    def invalid_topic_wiki_response_invalid_source_id(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid TopicWikiResponse data with invalid source workflow ID.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "topic_id": "550e8400-e29b-41d4-a716-446655440000",
            "topic_name": "Test Topic",
            "content": "Test content",
            "last_updated": 1703097600.0,
            "sources": ["invalid-uuid-format"],  # Invalid UUID format
            "query_count": 5,
            "confidence_score": 0.8,
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_topic_wiki_response_empty_content(**overrides: Any) -> Dict[str, Any]:
        """Create invalid TopicWikiResponse data with empty content.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "topic_id": "550e8400-e29b-41d4-a716-446655440000",
            "topic_name": "Test Topic",
            "content": "   ",  # Empty/whitespace content should fail
            "last_updated": 1703097600.0,
            "sources": ["550e8400-e29b-41d4-a716-446655440001"],
            "query_count": 5,
            "confidence_score": 0.8,
        }
        data.update(overrides)
        return data

    @staticmethod
    def topic_wiki_response_duplicate_sources(
        **overrides: Any,
    ) -> TopicWikiResponse:
        """Create TopicWikiResponse with duplicate sources (should be removed).

        Args:
            **overrides: Override any field with custom values

        Returns:
            TopicWikiResponse that will have duplicates removed
        """
        duplicate_sources = [
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400-e29b-41d4-a716-446655440002",
            "550e8400-e29b-41d4-a716-446655440001",  # Duplicate
        ]

        return APIModelFactory.basic_topic_wiki_response(
            sources=duplicate_sources,
            **overrides,
        )

    # =============================================================================
    # WorkflowMetadata Factories
    # =============================================================================

    @staticmethod
    def basic_workflow_metadata(
        workflow_id: str = "academic_research",
        name: str = "Academic Research Analysis",
        description: str = "Comprehensive academic research workflow with peer-review standards",
        version: str = "1.0.0",
        category: str = "academic",
        tags: Optional[List[str]] = None,
        created_by: str = "CogniVault Team",
        created_at: float = 1703097600.0,
        estimated_execution_time: str = "45-60 seconds",
        complexity_level: str = "high",
        node_count: int = 7,
        use_cases: Optional[List[str]] = None,
        **overrides: Any,
    ) -> WorkflowMetadata:
        """Create basic WorkflowMetadata with sensible defaults.

        Args:
            workflow_id: Unique workflow identifier
            name: Human-readable workflow name
            description: Detailed workflow description
            version: Workflow version
            category: Primary workflow category
            tags: Workflow tags for filtering
            created_by: Workflow author
            created_at: Creation timestamp
            estimated_execution_time: Estimated execution time range
            complexity_level: Workflow complexity level
            node_count: Number of nodes in workflow
            use_cases: Common use cases
            **overrides: Override any field with custom values

        Returns:
            WorkflowMetadata with valid defaults
        """
        if tags is None:
            tags = ["academic", "research", "scholarly", "analysis"]

        if use_cases is None:
            use_cases = ["dissertation_research", "literature_review"]

        defaults = {
            "workflow_id": workflow_id,
            "name": name,
            "description": description,
            "version": version,
            "category": category,
            "tags": tags,
            "created_by": created_by,
            "created_at": created_at,
            "estimated_execution_time": estimated_execution_time,
            "complexity_level": complexity_level,
            "node_count": node_count,
            "use_cases": use_cases,
        }
        defaults.update(overrides)
        return WorkflowMetadata(**defaults)

    @staticmethod
    def workflow_metadata_business(
        **overrides: Any,
    ) -> WorkflowMetadata:
        """Create business-category WorkflowMetadata.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowMetadata for business workflows
        """
        return APIModelFactory.basic_workflow_metadata(
            workflow_id="business_analysis",
            name="Business Strategy Analysis",
            description="Strategic business analysis with market research and competitive intelligence",
            category="business",
            tags=["business", "strategy", "market", "competitive"],
            complexity_level="medium",
            node_count=5,
            use_cases=["strategic_planning", "market_analysis"],
            **overrides,
        )

    @staticmethod
    def workflow_metadata_simple(
        **overrides: Any,
    ) -> WorkflowMetadata:
        """Create simple low-complexity WorkflowMetadata.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowMetadata for simple workflows
        """
        return APIModelFactory.basic_workflow_metadata(
            workflow_id="simple_qa",
            name="Simple Question Answering",
            description="Basic question answering workflow for general inquiries",
            category="general",
            tags=["simple", "qa", "general"],
            complexity_level="low",
            node_count=3,
            estimated_execution_time="15-30 seconds",
            use_cases=["quick_answers", "basic_research"],
            **overrides,
        )

    @staticmethod
    def invalid_workflow_metadata_empty_tags(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowMetadata data with empty tags list.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "test_workflow",
            "name": "Test Workflow",
            "description": "Test description",
            "version": "1.0.0",
            "category": "test",
            "tags": [],  # Should have at least one tag
            "created_by": "Test User",
            "created_at": 1703097600.0,
            "estimated_execution_time": "30 seconds",
            "complexity_level": "low",
            "node_count": 3,
            "use_cases": ["testing"],
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_metadata_long_tag(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowMetadata data with tag exceeding length limit.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "test_workflow",
            "name": "Test Workflow",
            "description": "Test description",
            "version": "1.0.0",
            "category": "test",
            "tags": ["x" * 31],  # Exceeds 30 character limit
            "created_by": "Test User",
            "created_at": 1703097600.0,
            "estimated_execution_time": "30 seconds",
            "complexity_level": "low",
            "node_count": 3,
            "use_cases": ["testing"],
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_workflow_metadata_long_use_case(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowMetadata data with use case exceeding length limit.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data = {
            "workflow_id": "test_workflow",
            "name": "Test Workflow",
            "description": "Test description",
            "version": "1.0.0",
            "category": "test",
            "tags": ["test"],
            "created_by": "Test User",
            "created_at": 1703097600.0,
            "estimated_execution_time": "30 seconds",
            "complexity_level": "low",
            "node_count": 3,
            "use_cases": ["x" * 101],  # Exceeds 100 character limit
        }
        data.update(overrides)
        return data

    @staticmethod
    def workflow_metadata_duplicate_tags(
        **overrides: Any,
    ) -> WorkflowMetadata:
        """Create WorkflowMetadata with duplicate tags (should be removed).

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowMetadata that will have duplicate tags removed
        """
        duplicate_tags = [
            "research",
            "academic",
            "research",
            "analysis",
        ]  # Contains duplicate

        return APIModelFactory.basic_workflow_metadata(
            tags=duplicate_tags,
            **overrides,
        )

    # =============================================================================
    # WorkflowsResponse Factories
    # =============================================================================

    @staticmethod
    def basic_workflows_response(
        workflows: Optional[List[WorkflowMetadata]] = None,
        categories: Optional[List[str]] = None,
        total: int = 25,
        limit: int = 10,
        offset: int = 0,
        has_more: bool = True,
        search_query: Optional[str] = None,
        category_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
        **overrides: Any,
    ) -> WorkflowsResponse:
        """Create basic WorkflowsResponse with sensible defaults.

        Args:
            workflows: List of workflow metadata
            categories: Available workflow categories
            total: Total number of workflows available
            limit: Maximum number of results requested
            offset: Number of results skipped
            has_more: Whether there are more results
            search_query: Search query used for filtering
            category_filter: Category filter applied
            complexity_filter: Complexity filter applied
            **overrides: Override any field with custom values

        Returns:
            WorkflowsResponse with valid defaults
        """
        if workflows is None:
            workflows = [
                APIModelFactory.basic_workflow_metadata(),
                APIModelFactory.workflow_metadata_business(),
            ]

        if categories is None:
            categories = ["academic", "business", "general", "legal"]

        defaults = {
            "workflows": workflows,
            "categories": categories,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "search_query": search_query,
            "category_filter": category_filter,
            "complexity_filter": complexity_filter,
        }
        defaults.update(overrides)
        return WorkflowsResponse(**defaults)

    @staticmethod
    def workflows_response_filtered(
        category_filter: str = "academic",
        complexity_filter: str = "high",
        search_query: str = "research",
        **overrides: Any,
    ) -> WorkflowsResponse:
        """Create WorkflowsResponse with multiple filters applied.

        Args:
            category_filter: Category filter applied
            complexity_filter: Complexity filter applied
            search_query: Search query used for filtering
            **overrides: Override any field with custom values

        Returns:
            WorkflowsResponse with filtering applied
        """
        academic_workflows = [APIModelFactory.basic_workflow_metadata()]

        return APIModelFactory.basic_workflows_response(
            workflows=academic_workflows,
            total=8,  # Fewer results due to filtering
            category_filter=category_filter,
            complexity_filter=complexity_filter,
            search_query=search_query,
            **overrides,
        )

    @staticmethod
    def workflows_response_empty_categories(
        **overrides: Any,
    ) -> WorkflowsResponse:
        """Create WorkflowsResponse with empty categories list.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowsResponse with no categories
        """
        return APIModelFactory.basic_workflows_response(
            categories=[],  # Will be handled by validator
            **overrides,
        )

    @staticmethod
    def invalid_workflows_response_inconsistent_pagination(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create WorkflowsResponse data with inconsistent pagination.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with inconsistent pagination data
        """
        workflows = [APIModelFactory.basic_workflow_metadata()]
        data = {
            "workflows": [w.model_dump() for w in workflows],
            "categories": ["academic", "business"],
            "total": 15,
            "limit": 10,
            "offset": 5,
            "has_more": False,  # Inconsistent: should be True based on offset + len < total
            "search_query": None,
            "category_filter": None,
            "complexity_filter": None,
        }
        data.update(overrides)
        return data

    @staticmethod
    def workflows_response_duplicate_categories(
        **overrides: Any,
    ) -> WorkflowsResponse:
        """Create WorkflowsResponse with duplicate categories (should be normalized).

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowsResponse that will have duplicate categories removed
        """
        duplicate_categories = [
            "academic",
            "business",
            "Academic",
            "BUSINESS",
        ]  # Mixed case duplicates

        return APIModelFactory.basic_workflows_response(
            categories=duplicate_categories,
            **overrides,
        )


# Convenience aliases for common patterns
class APIModelPatterns:
    """Pre-configured factory patterns for the most common usage scenarios."""

    @staticmethod
    def minimal_workflow_request(query: str = "test") -> WorkflowRequest:
        """Most common pattern: minimal valid WorkflowRequest."""
        return APIModelFactory.basic_workflow_request(query=query)

    @staticmethod
    def completed_workflow() -> WorkflowResponse:
        """Common pattern: completed workflow with all outputs."""
        return APIModelFactory.completed_workflow_response()

    @staticmethod
    def running_status() -> StatusResponse:
        """Common pattern: workflow in progress."""
        return APIModelFactory.running_status_response()

    @staticmethod
    def simple_completion() -> CompletionResponse:
        """Common pattern: basic LLM completion response."""
        return APIModelFactory.basic_completion_response()

    @staticmethod
    def topic_with_search(similarity: float = 0.85) -> TopicSummary:
        """Common pattern: topic result with similarity score."""
        return APIModelFactory.basic_topic_summary(similarity_score=similarity)

    @staticmethod
    def academic_workflow() -> WorkflowMetadata:
        """Common pattern: academic research workflow."""
        return APIModelFactory.basic_workflow_metadata()
