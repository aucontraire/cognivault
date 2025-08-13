"""Hybrid Factory Architecture for API model test data objects.

This module implements a hybrid factory approach that provides clear separation of concerns:

**MODEL INSTANCE FACTORIES (for golden path testing):**
- Return actual Pydantic model instances for typed operations
- Methods: create_valid_* and create_*_with_* patterns
- Used when you need fully typed model objects for business logic testing
- Example: create_valid_workflow_request(), create_completed_workflow_response()

**PAYLOAD DICTIONARY FACTORIES (for validation testing):**
- Return Dict[str, Any] for edge case and validation testing
- Methods: invalid_* and create_*_payload patterns
- Used when testing Pydantic validation, malformed data, or API boundary testing
- Example: invalid_workflow_request_empty_agents(), create_workflow_request_payload()

Hybrid Factory Principles:
- **Type Safety**: Explicit return types distinguish model instances from payloads
- **Clear Separation**: Factory name indicates testing purpose and return type
- **Single Source of Truth**: Reduce redundancy while maintaining specialized variants
- **Test Clarity**: Method names reveal intended usage pattern

Usage Examples:
    # Model instance factories (typed objects)
    request = APIModelFactory.create_valid_workflow_request()
    response = APIModelFactory.create_completed_workflow_response()

    # Payload factories for customization
    payload = APIModelFactory.create_workflow_request_payload(
        query="Custom query",
        agents=["refiner", "critic"]
    )
    request = WorkflowRequest(**payload)

    # Invalid data for validation testing
    invalid_data = APIModelFactory.invalid_workflow_request_empty_agents()
    with pytest.raises(ValidationError):
        WorkflowRequest(**invalid_data)

Factory Categories:
- create_valid_*(): Standard model instances with sensible defaults
- create_*_with_*(): Model instances with specific configurations
- create_*_payload(): Customizable dictionary payloads
- invalid_*(): Invalid payloads for validation testing
- edge_case_*(): Edge case model instances for boundary testing
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
    InternalExecutionGraph,
    InternalAgentMetrics,
)


class APIModelFactory:
    """Hybrid factory for API model test data with clear separation of concerns.

    Provides two distinct factory patterns:
    1. Model Instance Factories: Return typed Pydantic model instances
    2. Payload Dictionary Factories: Return Dict[str, Any] for validation testing
    """

    # =============================================================================
    # WorkflowRequest: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_workflow_request(
        query: str = "What is artificial intelligence?",
        agents: Optional[List[str]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create valid WorkflowRequest instance with sensible defaults.

        Args:
            query: User query string
            agents: List of agent names to execute
            execution_config: Execution configuration parameters
            correlation_id: Request correlation identifier
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest instance with valid defaults
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
    def create_workflow_request_payload(
        query: str = "What is artificial intelligence?",
        agents: Optional[List[str]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable WorkflowRequest payload dictionary.

        Args:
            query: User query string
            agents: List of agent names to execute
            execution_config: Execution configuration parameters
            correlation_id: Request correlation identifier
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for WorkflowRequest construction
        """
        data: Dict[str, Any] = {
            "query": query,
            "agents": agents,
            "execution_config": execution_config,
            "correlation_id": correlation_id,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_workflow_request_with_all_agents(
        query: str = "Comprehensive analysis request",
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest instance with all available agents.

        Args:
            query: User query string
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest instance with all valid agents configured
        """
        return APIModelFactory.create_valid_workflow_request(
            query=query,
            agents=["refiner", "historian", "critic", "synthesis"],
            **overrides,
        )

    @staticmethod
    def create_workflow_request_with_execution_config(
        query: str = "Request with execution config",
        timeout_seconds: int = 30,
        parallel_execution: bool = True,
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest instance with execution configuration.

        Args:
            query: User query string
            timeout_seconds: Execution timeout in seconds
            parallel_execution: Enable parallel execution
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest instance with execution configuration
        """
        return APIModelFactory.create_valid_workflow_request(
            query=query,
            execution_config={
                "timeout_seconds": timeout_seconds,
                "parallel_execution": parallel_execution,
            },
            **overrides,
        )

    @staticmethod
    def create_workflow_request_with_correlation_id(
        query: str = "Request with correlation ID",
        correlation_id: str = "req-12345-abcdef",
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest instance with correlation ID.

        Args:
            query: User query string
            correlation_id: Request correlation identifier
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest instance with correlation ID
        """
        return APIModelFactory.create_valid_workflow_request(
            query=query,
            correlation_id=correlation_id,
            **overrides,
        )

    # =============================================================================
    # WorkflowRequest: Invalid Payload Factories (already properly named)
    # =============================================================================

    @staticmethod
    def invalid_workflow_request_empty_query(**overrides: Any) -> Dict[str, Any]:
        """Create invalid WorkflowRequest data with empty query.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {"query": "", "agents": None, "execution_config": None}
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        """Create WorkflowRequest instance with maximum allowed query length.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest instance with edge case data
        """
        max_query = "x" * 10000  # Maximum allowed length
        return APIModelFactory.create_valid_workflow_request(
            query=max_query, **overrides
        )

    @staticmethod
    def edge_case_workflow_request_max_correlation_id(
        **overrides: Any,
    ) -> WorkflowRequest:
        """Create WorkflowRequest instance with maximum allowed correlation ID length.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowRequest instance with edge case data
        """
        max_correlation_id = "x" * 100  # Maximum allowed length
        return APIModelFactory.create_valid_workflow_request(
            correlation_id=max_correlation_id, **overrides
        )

    # =============================================================================
    # WorkflowRequest: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def workflow_request_with_all_agents(
        query: str = "Comprehensive analysis request",
        **overrides: Any,
    ) -> WorkflowRequest:
        """Alias for create_workflow_request_with_all_agents for test compatibility."""
        return APIModelFactory.create_workflow_request_with_all_agents(
            query=query, **overrides
        )

    @staticmethod
    def workflow_request_with_execution_config(
        query: str = "Request with execution config",
        timeout_seconds: int = 30,
        parallel_execution: bool = True,
        **overrides: Any,
    ) -> WorkflowRequest:
        """Alias for create_workflow_request_with_execution_config for test compatibility."""
        return APIModelFactory.create_workflow_request_with_execution_config(
            query=query,
            timeout_seconds=timeout_seconds,
            parallel_execution=parallel_execution,
            **overrides,
        )

    @staticmethod
    def workflow_request_with_correlation_id(
        query: str = "Request with correlation ID",
        correlation_id: str = "req-12345-abcdef",
        **overrides: Any,
    ) -> WorkflowRequest:
        """Alias for create_workflow_request_with_correlation_id for test compatibility."""
        return APIModelFactory.create_workflow_request_with_correlation_id(
            query=query, correlation_id=correlation_id, **overrides
        )

    # =============================================================================
    # WorkflowResponse: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "completed",
        agent_outputs: Optional[Dict[str, str]] = None,
        execution_time_seconds: float = 42.5,
        correlation_id: Optional[str] = None,
        error_message: Optional[str] = None,
        **overrides: Any,
    ) -> WorkflowResponse:
        """Create valid WorkflowResponse instance with sensible defaults.

        Args:
            workflow_id: Unique workflow identifier
            status: Execution status
            agent_outputs: Dict of agent outputs
            execution_time_seconds: Total execution time
            correlation_id: Request correlation identifier
            error_message: Error message if failed
            **overrides: Override any field with custom values

        Returns:
            WorkflowResponse instance with valid defaults
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
    def create_workflow_response_payload(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "completed",
        agent_outputs: Optional[Dict[str, str]] = None,
        execution_time_seconds: float = 42.5,
        correlation_id: Optional[str] = None,
        error_message: Optional[str] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable WorkflowResponse payload dictionary.

        Args:
            workflow_id: Unique workflow identifier
            status: Execution status
            agent_outputs: Dict of agent outputs
            execution_time_seconds: Total execution time
            correlation_id: Request correlation identifier
            error_message: Error message if failed
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for WorkflowResponse construction
        """
        if agent_outputs is None:
            agent_outputs = {"refiner": "Refined query output"}

        data: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "status": status,
            "agent_outputs": agent_outputs,
            "execution_time_seconds": execution_time_seconds,
            "correlation_id": correlation_id,
            "error_message": error_message,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_completed_workflow_response(
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
        return APIModelFactory.create_valid_workflow_response(
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
    def create_failed_workflow_response(
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
        return APIModelFactory.create_valid_workflow_response(
            workflow_id=workflow_id,
            status="failed",
            agent_outputs={},
            error_message=error_message,
            execution_time_seconds=12.3,
            **overrides,
        )

    @staticmethod
    def create_running_workflow_response(
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
        return APIModelFactory.create_valid_workflow_response(
            workflow_id=workflow_id,
            status="running",
            agent_outputs={"refiner": "Partial output"},
            execution_time_seconds=15.2,
            **overrides,
        )

    # =============================================================================
    # WorkflowResponse: Invalid Payload Factories (already properly named)
    # =============================================================================

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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "agent_outputs": {"refiner": ""},  # Empty output should fail
            "execution_time_seconds": 10.0,
        }
        data.update(overrides)
        return data

    # =============================================================================
    # WorkflowResponse: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def completed_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> WorkflowResponse:
        """Alias for create_completed_workflow_response for test compatibility."""
        return APIModelFactory.create_completed_workflow_response(
            workflow_id=workflow_id, **overrides
        )

    @staticmethod
    def failed_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        error_message: str = "Agent 'historian' failed: timeout exceeded",
        **overrides: Any,
    ) -> WorkflowResponse:
        """Alias for create_failed_workflow_response for test compatibility."""
        return APIModelFactory.create_failed_workflow_response(
            workflow_id=workflow_id, error_message=error_message, **overrides
        )

    @staticmethod
    def running_workflow_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> WorkflowResponse:
        """Alias for create_running_workflow_response for test compatibility."""
        return APIModelFactory.create_running_workflow_response(
            workflow_id=workflow_id, **overrides
        )

    # =============================================================================
    # StatusResponse: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "running",
        progress_percentage: float = 50.0,
        current_agent: Optional[str] = "critic",
        estimated_completion_seconds: Optional[float] = 15.5,
        **overrides: Any,
    ) -> StatusResponse:
        """Create valid StatusResponse instance with sensible defaults.

        Args:
            workflow_id: Unique workflow identifier
            status: Current execution status
            progress_percentage: Execution progress (0-100)
            current_agent: Currently executing agent
            estimated_completion_seconds: Estimated completion time
            **overrides: Override any field with custom values

        Returns:
            StatusResponse instance with valid defaults
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
    def create_status_response_payload(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "running",
        progress_percentage: float = 50.0,
        current_agent: Optional[str] = "critic",
        estimated_completion_seconds: Optional[float] = 15.5,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable StatusResponse payload dictionary.

        Args:
            workflow_id: Unique workflow identifier
            status: Current execution status
            progress_percentage: Execution progress (0-100)
            current_agent: Currently executing agent
            estimated_completion_seconds: Estimated completion time
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for StatusResponse construction
        """
        data: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "status": status,
            "progress_percentage": progress_percentage,
            "current_agent": current_agent,
            "estimated_completion_seconds": estimated_completion_seconds,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_running_status_response(
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
        return APIModelFactory.create_valid_status_response(
            workflow_id=workflow_id,
            status="running",
            progress_percentage=75.0,
            current_agent="synthesis",
            estimated_completion_seconds=10.2,
            **overrides,
        )

    @staticmethod
    def create_completed_status_response(
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
        return APIModelFactory.create_valid_status_response(
            workflow_id=workflow_id,
            status="completed",
            progress_percentage=100.0,
            current_agent=None,  # No current agent when completed
            estimated_completion_seconds=None,
            **overrides,
        )

    @staticmethod
    def create_failed_status_response(
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
        return APIModelFactory.create_valid_status_response(
            workflow_id=workflow_id,
            status="failed",
            progress_percentage=65.0,  # Failed partway through
            current_agent=None,
            estimated_completion_seconds=None,
            **overrides,
        )

    # =============================================================================
    # StatusResponse: Invalid Payload Factories (already properly named)
    # =============================================================================

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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "failed",
            "progress_percentage": 100.0,  # Should not be 100.0 for failed
        }
        data.update(overrides)
        return data

    # =============================================================================
    # StatusResponse: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def running_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> StatusResponse:
        """Alias for create_running_status_response for test compatibility."""
        return APIModelFactory.create_running_status_response(
            workflow_id=workflow_id, **overrides
        )

    @staticmethod
    def completed_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> StatusResponse:
        """Alias for create_completed_status_response for test compatibility."""
        return APIModelFactory.create_completed_status_response(
            workflow_id=workflow_id, **overrides
        )

    @staticmethod
    def failed_status_response(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> StatusResponse:
        """Alias for create_failed_status_response for test compatibility."""
        return APIModelFactory.create_failed_status_response(
            workflow_id=workflow_id, **overrides
        )

    # =============================================================================
    # CompletionRequest: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_completion_request(
        prompt: str = "Explain the concept of machine learning in simple terms",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        agent_context: Optional[str] = None,
        **overrides: Any,
    ) -> CompletionRequest:
        """Create valid CompletionRequest instance with sensible defaults.

        Args:
            prompt: The prompt to send to the LLM
            model: LLM model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            agent_context: Additional agent context
            **overrides: Override any field with custom values

        Returns:
            CompletionRequest instance with valid defaults
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
    def create_completion_request_payload(
        prompt: str = "Explain the concept of machine learning in simple terms",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        agent_context: Optional[str] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable CompletionRequest payload dictionary.

        Args:
            prompt: The prompt to send to the LLM
            model: LLM model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            agent_context: Additional agent context
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for CompletionRequest construction
        """
        data: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "agent_context": agent_context,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_completion_request_with_options(
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
        return APIModelFactory.create_valid_completion_request(
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
        return APIModelFactory.create_valid_completion_request(
            prompt=max_prompt, **overrides
        )

    # =============================================================================
    # CompletionRequest: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def completion_request_with_options(
        prompt: str = "Analyze the following data",
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **overrides: Any,
    ) -> CompletionRequest:
        """Alias for create_completion_request_with_options for test compatibility."""
        return APIModelFactory.create_completion_request_with_options(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **overrides,
        )

    @staticmethod
    def generate_valid_completion_request(**overrides: Any) -> CompletionRequest:
        """Generate valid CompletionRequest with zero parameters required.

        Standard convenience method for most test scenarios.

        Args:
            **overrides: Optional field overrides for specific test needs

        Returns:
            CompletionRequest instance with valid defaults
        """
        return APIModelFactory.create_valid_completion_request(**overrides)

    # =============================================================================
    # CompletionResponse: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_completion_response(
        completion: str = "Machine learning is a subset of artificial intelligence...",
        model_used: str = "gpt-4",
        token_usage: Optional[Dict[str, int]] = None,
        response_time_ms: float = 1250.5,
        request_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> CompletionResponse:
        """Create valid CompletionResponse instance with sensible defaults.

        Args:
            completion: Generated completion text
            model_used: The model that generated completion
            token_usage: Token usage statistics
            response_time_ms: Response time in milliseconds
            request_id: Unique request identifier
            **overrides: Override any field with custom values

        Returns:
            CompletionResponse instance with valid defaults
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
    def create_completion_response_payload(
        completion: str = "Machine learning is a subset of artificial intelligence...",
        model_used: str = "gpt-4",
        token_usage: Optional[Dict[str, int]] = None,
        response_time_ms: float = 1250.5,
        request_id: str = "550e8400-e29b-41d4-a716-446655440000",
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable CompletionResponse payload dictionary.

        Args:
            completion: Generated completion text
            model_used: The model that generated completion
            token_usage: Token usage statistics
            response_time_ms: Response time in milliseconds
            request_id: Unique request identifier
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for CompletionResponse construction
        """
        if token_usage is None:
            token_usage = {
                "prompt_tokens": 25,
                "completion_tokens": 150,
                "total_tokens": 175,
            }

        data: Dict[str, Any] = {
            "completion": completion,
            "model_used": model_used,
            "token_usage": token_usage,
            "response_time_ms": response_time_ms,
            "request_id": request_id,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_completion_response_with_high_usage(
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
        return APIModelFactory.create_valid_completion_response(
            completion=completion,
            token_usage={
                "prompt_tokens": 500,
                "completion_tokens": 1500,
                "total_tokens": 2000,
            },
            response_time_ms=3500.8,
            **overrides,
        )

    # =============================================================================
    # CompletionResponse: Invalid Payload Factories (already properly named)
    # =============================================================================

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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
    # CompletionResponse: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def completion_response_with_high_usage(
        completion: str = "Detailed analysis with comprehensive insights...",
        **overrides: Any,
    ) -> CompletionResponse:
        """Alias for create_completion_response_with_high_usage for test compatibility."""
        return APIModelFactory.create_completion_response_with_high_usage(
            completion=completion, **overrides
        )

    # =============================================================================
    # LLMProviderInfo: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_llm_provider_info(
        name: str = "openai",
        models: Optional[List[str]] = None,
        available: bool = True,
        cost_per_token: Optional[float] = 0.00003,
        **overrides: Any,
    ) -> LLMProviderInfo:
        """Create valid LLMProviderInfo instance with sensible defaults.

        Args:
            name: Provider name
            models: List of available models
            available: Whether provider is available
            cost_per_token: Cost per token in USD
            **overrides: Override any field with custom values

        Returns:
            LLMProviderInfo instance with valid defaults
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
    def create_llm_provider_info_payload(
        name: str = "openai",
        models: Optional[List[str]] = None,
        available: bool = True,
        cost_per_token: Optional[float] = 0.00003,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable LLMProviderInfo payload dictionary.

        Args:
            name: Provider name
            models: List of available models
            available: Whether provider is available
            cost_per_token: Cost per token in USD
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for LLMProviderInfo construction
        """
        if models is None:
            models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

        data: Dict[str, Any] = {
            "name": name,
            "models": models,
            "available": available,
            "cost_per_token": cost_per_token,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_llm_provider_info_unavailable(
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
        return APIModelFactory.create_valid_llm_provider_info(
            name=name,
            models=["claude-3-sonnet", "claude-3-haiku"],
            available=False,
            cost_per_token=None,
            **overrides,
        )

    # =============================================================================
    # LLMProviderInfo: Invalid Payload Factories (already properly named)
    # =============================================================================

    @staticmethod
    def invalid_llm_provider_info_empty_models(**overrides: Any) -> Dict[str, Any]:
        """Create invalid LLMProviderInfo data with empty models list.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
            "name": "test_provider",
            "models": ["gpt-4", ""],  # Empty model name should fail
            "available": True,
        }
        data.update(overrides)
        return data

    # =============================================================================
    # LLMProviderInfo: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def llm_provider_info_unavailable(
        name: str = "claude",
        **overrides: Any,
    ) -> LLMProviderInfo:
        """Alias for create_llm_provider_info_unavailable for test compatibility."""
        return APIModelFactory.create_llm_provider_info_unavailable(
            name=name, **overrides
        )

    # =============================================================================
    # WorkflowHistoryItem: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_workflow_history_item(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "completed",
        query: str = "Analyze the impact of climate change on agriculture",
        start_time: float = 1703097600.0,
        execution_time_seconds: float = 12.5,
        **overrides: Any,
    ) -> WorkflowHistoryItem:
        """Create valid WorkflowHistoryItem instance with sensible defaults.

        Args:
            workflow_id: Unique workflow identifier
            status: Workflow execution status
            query: Original query (truncated)
            start_time: Start time as Unix timestamp
            execution_time_seconds: Total execution time
            **overrides: Override any field with custom values

        Returns:
            WorkflowHistoryItem instance with valid defaults
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
    def create_workflow_history_item_payload(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440000",
        status: str = "completed",
        query: str = "Analyze the impact of climate change on agriculture",
        start_time: float = 1703097600.0,
        execution_time_seconds: float = 12.5,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable WorkflowHistoryItem payload dictionary.

        Args:
            workflow_id: Unique workflow identifier
            status: Workflow execution status
            query: Original query (truncated)
            start_time: Start time as Unix timestamp
            execution_time_seconds: Total execution time
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for WorkflowHistoryItem construction
        """
        data: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "status": status,
            "query": query,
            "start_time": start_time,
            "execution_time_seconds": execution_time_seconds,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_workflow_history_item_failed(
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
        return APIModelFactory.create_valid_workflow_history_item(
            workflow_id=workflow_id,
            status="failed",
            query="Query that failed to complete",
            execution_time_seconds=8.2,
            **overrides,
        )

    # =============================================================================
    # WorkflowHistoryItem: Invalid Payload Factories (already properly named)
    # =============================================================================

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
        data: Dict[str, Any] = {
            "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "invalid_status",  # Should be one of the valid statuses
            "query": "test query",
            "start_time": 1703097600.0,
            "execution_time_seconds": 10.0,
        }
        data.update(overrides)
        return data

    # =============================================================================
    # WorkflowHistoryResponse: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_workflow_history_response(
        workflows: Optional[List[WorkflowHistoryItem]] = None,
        total: int = 150,
        limit: int = 10,
        offset: int = 0,
        has_more: bool = True,
        **overrides: Any,
    ) -> WorkflowHistoryResponse:
        """Create valid WorkflowHistoryResponse instance with sensible defaults.

        Args:
            workflows: List of workflow history items
            total: Total number of workflows available
            limit: Maximum number of results requested
            offset: Number of results skipped
            has_more: Whether there are more results
            **overrides: Override any field with custom values

        Returns:
            WorkflowHistoryResponse instance with valid defaults
        """
        if workflows is None:
            workflows = [
                APIModelFactory.create_valid_workflow_history_item(),
                APIModelFactory.create_workflow_history_item_failed(),
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
    def create_workflow_history_response_payload(
        workflows: Optional[List[WorkflowHistoryItem]] = None,
        total: int = 150,
        limit: int = 10,
        offset: int = 0,
        has_more: bool = True,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable WorkflowHistoryResponse payload dictionary.

        Args:
            workflows: List of workflow history items
            total: Total number of workflows available
            limit: Maximum number of results requested
            offset: Number of results skipped
            has_more: Whether there are more results
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for WorkflowHistoryResponse construction
        """
        if workflows is None:
            workflows = [
                APIModelFactory.create_valid_workflow_history_item(),
                APIModelFactory.create_workflow_history_item_failed(),
            ]

        data: Dict[str, Any] = {
            "workflows": [w.model_dump() for w in workflows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_workflow_history_response_last_page(
        **overrides: Any,
    ) -> WorkflowHistoryResponse:
        """Create WorkflowHistoryResponse for the last page of results.

        Args:
            **overrides: Override any field with custom values

        Returns:
            WorkflowHistoryResponse with has_more=False
        """
        workflows = [APIModelFactory.create_valid_workflow_history_item()]
        return APIModelFactory.create_valid_workflow_history_response(
            workflows=workflows,
            total=21,
            limit=10,
            offset=20,
            has_more=False,  # Last page
            **overrides,
        )

    # =============================================================================
    # WorkflowHistoryResponse: Invalid Payload Factories (already properly named)
    # =============================================================================

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
        workflows = [APIModelFactory.create_valid_workflow_history_item()]
        data: Dict[str, Any] = {
            "workflows": [w.model_dump() for w in workflows],
            "total": 10,
            "limit": 5,
            "offset": 0,
            "has_more": False,  # Inconsistent: should be True based on offset + len < total
        }
        data.update(overrides)
        return data

    # =============================================================================
    # WorkflowHistoryItem: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def workflow_history_item_failed(
        workflow_id: str = "550e8400-e29b-41d4-a716-446655440001",
        **overrides: Any,
    ) -> WorkflowHistoryItem:
        """Alias for create_workflow_history_item_failed for test compatibility."""
        return APIModelFactory.create_workflow_history_item_failed(
            workflow_id=workflow_id, **overrides
        )

    # =============================================================================
    # WorkflowHistoryResponse: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def workflow_history_response_last_page(
        **overrides: Any,
    ) -> WorkflowHistoryResponse:
        """Alias for create_workflow_history_response_last_page for test compatibility."""
        return APIModelFactory.create_workflow_history_response_last_page(**overrides)

    # =============================================================================
    # TopicSummary: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_topic_summary(
        topic_id: str = "550e8400-e29b-41d4-a716-446655440000",
        name: str = "Machine Learning Fundamentals",
        description: str = "Core concepts and principles of machine learning algorithms",
        query_count: int = 15,
        last_updated: float = 1703097600.0,
        similarity_score: Optional[float] = 0.85,
        **overrides: Any,
    ) -> TopicSummary:
        """Create valid TopicSummary instance with sensible defaults.

        Args:
            topic_id: Unique topic identifier
            name: Human-readable topic name
            description: Brief topic description
            query_count: Number of related queries
            last_updated: Last update timestamp
            similarity_score: Similarity score for search
            **overrides: Override any field with custom values

        Returns:
            TopicSummary instance with valid defaults
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
    def create_topic_summary_payload(
        topic_id: str = "550e8400-e29b-41d4-a716-446655440000",
        name: str = "Machine Learning Fundamentals",
        description: str = "Core concepts and principles of machine learning algorithms",
        query_count: int = 15,
        last_updated: float = 1703097600.0,
        similarity_score: Optional[float] = 0.85,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable TopicSummary payload dictionary.

        Args:
            topic_id: Unique topic identifier
            name: Human-readable topic name
            description: Brief topic description
            query_count: Number of related queries
            last_updated: Last update timestamp
            similarity_score: Similarity score for search
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for TopicSummary construction
        """
        data: Dict[str, Any] = {
            "topic_id": topic_id,
            "name": name,
            "description": description,
            "query_count": query_count,
            "last_updated": last_updated,
            "similarity_score": similarity_score,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_topic_summary_without_similarity(
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
        return APIModelFactory.create_valid_topic_summary(
            name=name,
            description="Introduction to data science concepts and methodologies",
            similarity_score=None,
            **overrides,
        )

    # =============================================================================
    # TopicSummary: Invalid Payload Factories (already properly named)
    # =============================================================================

    @staticmethod
    def invalid_topic_summary_empty_name(**overrides: Any) -> Dict[str, Any]:
        """Create invalid TopicSummary data with empty name.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {
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
        return APIModelFactory.create_valid_topic_summary(name=max_name, **overrides)

    # =============================================================================
    # TopicSummary: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def topic_summary_without_similarity(
        name: str = "Data Science Basics",
        **overrides: Any,
    ) -> TopicSummary:
        """Alias for create_topic_summary_without_similarity for test compatibility."""
        return APIModelFactory.create_topic_summary_without_similarity(
            name=name, **overrides
        )

    # =============================================================================
    # TopicsResponse: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_topics_response(
        topics: Optional[List[TopicSummary]] = None,
        total: int = 42,
        limit: int = 10,
        offset: int = 0,
        has_more: bool = True,
        search_query: Optional[str] = None,
        **overrides: Any,
    ) -> TopicsResponse:
        """Create valid TopicsResponse instance with sensible defaults.

        Args:
            topics: List of topic summaries
            total: Total number of topics available
            limit: Maximum number of results requested
            offset: Number of results skipped
            has_more: Whether there are more results
            search_query: Search query used for filtering
            **overrides: Override any field with custom values

        Returns:
            TopicsResponse instance with valid defaults
        """
        if topics is None:
            topics = [
                APIModelFactory.create_valid_topic_summary(),
                APIModelFactory.create_topic_summary_without_similarity(),
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
    def create_topics_response_payload(
        topics: Optional[List[TopicSummary]] = None,
        total: int = 42,
        limit: int = 10,
        offset: int = 0,
        has_more: bool = True,
        search_query: Optional[str] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable TopicsResponse payload dictionary.

        Args:
            topics: List of topic summaries
            total: Total number of topics available
            limit: Maximum number of results requested
            offset: Number of results skipped
            has_more: Whether there are more results
            search_query: Search query used for filtering
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for TopicsResponse construction
        """
        if topics is None:
            topics = [
                APIModelFactory.create_valid_topic_summary(),
                APIModelFactory.create_topic_summary_without_similarity(),
            ]

        data: Dict[str, Any] = {
            "topics": [t.model_dump() for t in topics],
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "search_query": search_query,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_topics_response_with_search(
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
        return APIModelFactory.create_valid_topics_response(
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
        topics = [APIModelFactory.create_valid_topic_summary()]
        data: Dict[str, Any] = {
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
    # TopicsResponse: Method Aliases for Test Compatibility
    # =============================================================================

    @staticmethod
    def topics_response_with_search(
        search_query: str = "machine learning",
        **overrides: Any,
    ) -> TopicsResponse:
        """Alias for create_topics_response_with_search for test compatibility."""
        return APIModelFactory.create_topics_response_with_search(
            search_query=search_query, **overrides
        )

    # =============================================================================
    # TopicWikiResponse: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_topic_wiki_response(
        topic_id: str = "550e8400-e29b-41d4-a716-446655440000",
        topic_name: str = "Machine Learning Fundamentals",
        content: str = "Machine learning is a subset of artificial intelligence...",
        last_updated: float = 1703097600.0,
        sources: Optional[List[str]] = None,
        query_count: int = 15,
        confidence_score: float = 0.92,
        **overrides: Any,
    ) -> TopicWikiResponse:
        """Create valid TopicWikiResponse instance with sensible defaults.

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
            TopicWikiResponse instance with valid defaults
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
    def create_topic_wiki_response_payload(
        topic_id: str = "550e8400-e29b-41d4-a716-446655440000",
        topic_name: str = "Machine Learning Fundamentals",
        content: str = "Machine learning is a subset of artificial intelligence...",
        last_updated: float = 1703097600.0,
        sources: Optional[List[str]] = None,
        query_count: int = 15,
        confidence_score: float = 0.92,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable TopicWikiResponse payload dictionary.

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
            Dictionary payload for TopicWikiResponse construction
        """
        if sources is None:
            sources = [
                "550e8400-e29b-41d4-a716-446655440001",
                "550e8400-e29b-41d4-a716-446655440002",
            ]

        data: Dict[str, Any] = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "content": content,
            "last_updated": last_updated,
            "sources": sources,
            "query_count": query_count,
            "confidence_score": confidence_score,
        }
        data.update(overrides)
        return data

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

        return APIModelFactory.create_valid_topic_wiki_response(
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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

        return APIModelFactory.create_valid_topic_wiki_response(
            sources=duplicate_sources,
            **overrides,
        )

    # =============================================================================
    # WorkflowMetadata: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_workflow_metadata(
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
        """Create valid WorkflowMetadata instance with sensible defaults.

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
            WorkflowMetadata instance with valid defaults
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
    def create_workflow_metadata_payload(
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
    ) -> Dict[str, Any]:
        """Create customizable WorkflowMetadata payload dictionary.

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
            Dictionary payload for WorkflowMetadata construction
        """
        if tags is None:
            tags = ["academic", "research", "scholarly", "analysis"]

        if use_cases is None:
            use_cases = ["dissertation_research", "literature_review"]

        data: Dict[str, Any] = {
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
        data.update(overrides)
        return data

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
        return APIModelFactory.create_valid_workflow_metadata(
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
        return APIModelFactory.create_valid_workflow_metadata(
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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
        data: Dict[str, Any] = {
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

        return APIModelFactory.create_valid_workflow_metadata(
            tags=duplicate_tags,
            **overrides,
        )

    # =============================================================================
    # WorkflowsResponse: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_workflows_response(
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
        """Create valid WorkflowsResponse instance with sensible defaults.

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
            WorkflowsResponse instance with valid defaults
        """
        if workflows is None:
            workflows = [
                APIModelFactory.create_valid_workflow_metadata(),
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
    def create_workflows_response_payload(
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
    ) -> Dict[str, Any]:
        """Create customizable WorkflowsResponse payload dictionary.

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
            Dictionary payload for WorkflowsResponse construction
        """
        if workflows is None:
            workflows = [
                APIModelFactory.create_valid_workflow_metadata(),
                APIModelFactory.workflow_metadata_business(),
            ]

        if categories is None:
            categories = ["academic", "business", "general", "legal"]

        data: Dict[str, Any] = {
            "workflows": [w.model_dump() for w in workflows],
            "categories": categories,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "search_query": search_query,
            "category_filter": category_filter,
            "complexity_filter": complexity_filter,
        }
        data.update(overrides)
        return data

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
        academic_workflows = [APIModelFactory.create_valid_workflow_metadata()]

        return APIModelFactory.create_valid_workflows_response(
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
        return APIModelFactory.create_valid_workflows_response(
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
        workflows = [APIModelFactory.create_valid_workflow_metadata()]
        data: Dict[str, Any] = {
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

        return APIModelFactory.create_valid_workflows_response(
            categories=duplicate_categories,
            **overrides,
        )

    # =============================================================================
    # InternalExecutionGraph: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_internal_execution_graph(
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> InternalExecutionGraph:
        """Create valid InternalExecutionGraph instance with sensible defaults.

        Args:
            nodes: Graph nodes representing execution units
            edges: Graph edges representing execution dependencies
            metadata: Additional graph metadata
            **overrides: Override any field with custom values

        Returns:
            InternalExecutionGraph instance with valid defaults
        """
        if nodes is None:
            nodes = [
                {"id": "refiner", "type": "processor", "config": {"timeout": 30}},
                {"id": "historian", "type": "processor", "config": {"max_results": 10}},
                {"id": "critic", "type": "processor", "config": {"threshold": 0.8}},
                {
                    "id": "synthesis",
                    "type": "processor",
                    "config": {"combine_mode": "weighted"},
                },
            ]

        if edges is None:
            edges = [
                {"from": "refiner", "to": "historian", "condition": "success"},
                {"from": "historian", "to": "critic", "condition": "success"},
                {"from": "critic", "to": "synthesis", "condition": "success"},
            ]

        if metadata is None:
            metadata = {
                "version": "1.0.0",
                "created_at": "2023-01-01T00:00:00Z",
                "execution_mode": "sequential",
                "estimated_duration_ms": 45000,
            }

        defaults = {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }
        defaults.update(overrides)
        return InternalExecutionGraph(**defaults)

    @staticmethod
    def create_internal_execution_graph_payload(
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable InternalExecutionGraph payload dictionary.

        Args:
            nodes: Graph nodes representing execution units
            edges: Graph edges representing execution dependencies
            metadata: Additional graph metadata
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for InternalExecutionGraph construction
        """
        if nodes is None:
            nodes = [
                {"id": "refiner", "type": "processor", "config": {"timeout": 30}},
                {"id": "historian", "type": "processor", "config": {"max_results": 10}},
                {"id": "critic", "type": "processor", "config": {"threshold": 0.8}},
                {
                    "id": "synthesis",
                    "type": "processor",
                    "config": {"combine_mode": "weighted"},
                },
            ]

        if edges is None:
            edges = [
                {"from": "refiner", "to": "historian", "condition": "success"},
                {"from": "historian", "to": "critic", "condition": "success"},
                {"from": "critic", "to": "synthesis", "condition": "success"},
            ]

        if metadata is None:
            metadata = {
                "version": "1.0.0",
                "created_at": "2023-01-01T00:00:00Z",
                "execution_mode": "sequential",
                "estimated_duration_ms": 45000,
            }

        data: Dict[str, Any] = {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_internal_execution_graph_simple(
        **overrides: Any,
    ) -> InternalExecutionGraph:
        """Create simple InternalExecutionGraph with minimal nodes.

        Args:
            **overrides: Override any field with custom values

        Returns:
            InternalExecutionGraph with simple configuration
        """
        simple_nodes = [
            {"id": "input", "type": "entry", "config": {}},
            {"id": "process", "type": "processor", "config": {"timeout": 15}},
            {"id": "output", "type": "terminator", "config": {}},
        ]

        simple_edges = [
            {"from": "input", "to": "process", "condition": "always"},
            {"from": "process", "to": "output", "condition": "success"},
        ]

        return APIModelFactory.create_valid_internal_execution_graph(
            nodes=simple_nodes,
            edges=simple_edges,
            metadata={"version": "0.1.0", "execution_mode": "linear"},
            **overrides,
        )

    @staticmethod
    def create_internal_execution_graph_complex(
        **overrides: Any,
    ) -> InternalExecutionGraph:
        """Create complex InternalExecutionGraph with parallel paths and decision nodes.

        Args:
            **overrides: Override any field with custom values

        Returns:
            InternalExecutionGraph with complex configuration
        """
        complex_nodes = [
            {"id": "input", "type": "entry", "config": {}},
            {"id": "classifier", "type": "decision", "config": {"threshold": 0.7}},
            {"id": "path_a", "type": "processor", "config": {"mode": "fast"}},
            {"id": "path_b", "type": "processor", "config": {"mode": "thorough"}},
            {"id": "aggregator", "type": "aggregator", "config": {"strategy": "merge"}},
            {"id": "validator", "type": "validator", "config": {"strict": True}},
            {"id": "output", "type": "terminator", "config": {}},
        ]

        complex_edges = [
            {"from": "input", "to": "classifier", "condition": "always"},
            {"from": "classifier", "to": "path_a", "condition": "confidence > 0.7"},
            {"from": "classifier", "to": "path_b", "condition": "confidence <= 0.7"},
            {"from": "path_a", "to": "aggregator", "condition": "success"},
            {"from": "path_b", "to": "aggregator", "condition": "success"},
            {"from": "aggregator", "to": "validator", "condition": "success"},
            {"from": "validator", "to": "output", "condition": "valid"},
        ]

        return APIModelFactory.create_valid_internal_execution_graph(
            nodes=complex_nodes,
            edges=complex_edges,
            metadata={
                "version": "2.0.0",
                "execution_mode": "parallel",
                "max_parallel": 2,
                "estimated_duration_ms": 60000,
            },
            **overrides,
        )

    @staticmethod
    def invalid_internal_execution_graph_empty_nodes(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid InternalExecutionGraph data with empty nodes list.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {
            "nodes": [],  # Empty nodes list
            "edges": [],
            "metadata": {},
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_internal_execution_graph_mismatched_edges(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid InternalExecutionGraph data with edges referencing non-existent nodes.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {
            "nodes": [{"id": "node1", "type": "processor"}],
            "edges": [
                {"from": "node1", "to": "non_existent_node", "condition": "success"}
            ],  # References non-existent node
            "metadata": {},
        }
        data.update(overrides)
        return data

    # =============================================================================
    # InternalAgentMetrics: Model Instance Factories
    # =============================================================================

    @staticmethod
    def create_valid_internal_agent_metrics(
        agent_name: str = "refiner",
        execution_time_ms: float = 1250.5,
        token_usage: Optional[Dict[str, int]] = None,
        success: bool = True,
        timestamp: Optional[datetime] = None,
        **overrides: Any,
    ) -> InternalAgentMetrics:
        """Create valid InternalAgentMetrics instance with sensible defaults.

        Args:
            agent_name: Name of the agent
            execution_time_ms: Execution time in milliseconds
            token_usage: Token usage statistics
            success: Whether agent execution was successful
            timestamp: Timestamp of the execution
            **overrides: Override any field with custom values

        Returns:
            InternalAgentMetrics instance with valid defaults
        """
        if token_usage is None:
            token_usage = {
                "prompt_tokens": 45,
                "completion_tokens": 120,
                "total_tokens": 165,
            }

        if timestamp is None:
            timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        defaults = {
            "agent_name": agent_name,
            "execution_time_ms": execution_time_ms,
            "token_usage": token_usage,
            "success": success,
            "timestamp": timestamp,
        }
        defaults.update(overrides)
        return InternalAgentMetrics(**defaults)

    @staticmethod
    def create_internal_agent_metrics_payload(
        agent_name: str = "refiner",
        execution_time_ms: float = 1250.5,
        token_usage: Optional[Dict[str, int]] = None,
        success: bool = True,
        timestamp: Optional[datetime] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create customizable InternalAgentMetrics payload dictionary.

        Args:
            agent_name: Name of the agent
            execution_time_ms: Execution time in milliseconds
            token_usage: Token usage statistics
            success: Whether agent execution was successful
            timestamp: Timestamp of the execution
            **overrides: Override any field with custom values

        Returns:
            Dictionary payload for InternalAgentMetrics construction
        """
        if token_usage is None:
            token_usage = {
                "prompt_tokens": 45,
                "completion_tokens": 120,
                "total_tokens": 165,
            }

        if timestamp is None:
            timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        data: Dict[str, Any] = {
            "agent_name": agent_name,
            "execution_time_ms": execution_time_ms,
            "token_usage": token_usage,
            "success": success,
            "timestamp": timestamp,
        }
        data.update(overrides)
        return data

    @staticmethod
    def create_internal_agent_metrics_failed(
        agent_name: str = "historian",
        **overrides: Any,
    ) -> InternalAgentMetrics:
        """Create InternalAgentMetrics for failed agent execution.

        Args:
            agent_name: Name of the failed agent
            **overrides: Override any field with custom values

        Returns:
            InternalAgentMetrics with failed execution
        """
        return APIModelFactory.create_valid_internal_agent_metrics(
            agent_name=agent_name,
            execution_time_ms=850.2,
            token_usage={
                "prompt_tokens": 30,
                "completion_tokens": 0,  # No completion due to failure
                "total_tokens": 30,
            },
            success=False,  # Failed execution
            **overrides,
        )

    @staticmethod
    def create_internal_agent_metrics_high_usage(
        agent_name: str = "synthesis",
        **overrides: Any,
    ) -> InternalAgentMetrics:
        """Create InternalAgentMetrics with high token usage.

        Args:
            agent_name: Name of the agent
            **overrides: Override any field with custom values

        Returns:
            InternalAgentMetrics with high token usage
        """
        return APIModelFactory.create_valid_internal_agent_metrics(
            agent_name=agent_name,
            execution_time_ms=4200.8,
            token_usage={
                "prompt_tokens": 800,
                "completion_tokens": 1200,
                "total_tokens": 2000,
            },
            success=True,
            **overrides,
        )

    @staticmethod
    def create_internal_agent_metrics_with_current_timestamp(
        agent_name: str = "critic",
        **overrides: Any,
    ) -> InternalAgentMetrics:
        """Create InternalAgentMetrics with current timestamp.

        Args:
            agent_name: Name of the agent
            **overrides: Override any field with custom values

        Returns:
            InternalAgentMetrics with current timestamp
        """
        return APIModelFactory.create_valid_internal_agent_metrics(
            agent_name=agent_name,
            timestamp=datetime.now(timezone.utc),
            **overrides,
        )

    @staticmethod
    def invalid_internal_agent_metrics_negative_execution_time(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid InternalAgentMetrics data with negative execution time.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {
            "agent_name": "test_agent",
            "execution_time_ms": -100.0,  # Negative execution time should fail
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "success": True,
            "timestamp": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_internal_agent_metrics_empty_agent_name(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid InternalAgentMetrics data with empty agent name.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {
            "agent_name": "",  # Empty agent name
            "execution_time_ms": 1000.0,
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "success": True,
            "timestamp": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        }
        data.update(overrides)
        return data

    @staticmethod
    def invalid_internal_agent_metrics_invalid_token_usage(
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Create invalid InternalAgentMetrics data with invalid token usage structure.

        Args:
            **overrides: Override any field with custom values

        Returns:
            Dict with invalid data for validation testing
        """
        data: Dict[str, Any] = {
            "agent_name": "test_agent",
            "execution_time_ms": 1000.0,
            "token_usage": {"invalid_key": 10},  # Missing required token usage keys
            "success": True,
            "timestamp": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        }
        data.update(overrides)
        return data

    # =============================================================================
    # Zero-Parameter Convenience Methods (Highest Priority for Test Warning Reduction)
    # =============================================================================

    @staticmethod
    def generate_valid_workflow_response(**overrides: Any) -> WorkflowResponse:
        """Standard valid WorkflowResponse for most test scenarios - ZERO required parameters."""
        return APIModelFactory.create_valid_workflow_response(**overrides)

    @staticmethod
    def generate_valid_status_response(**overrides: Any) -> StatusResponse:
        """Standard valid StatusResponse for most test scenarios - ZERO required parameters."""
        return APIModelFactory.create_valid_status_response(**overrides)

    @staticmethod
    def generate_valid_completion_response(**overrides: Any) -> CompletionResponse:
        """Standard valid CompletionResponse for most test scenarios - ZERO required parameters."""
        return APIModelFactory.create_valid_completion_response(**overrides)

    @staticmethod
    def generate_valid_llm_provider_info(**overrides: Any) -> LLMProviderInfo:
        """Standard valid LLMProviderInfo for most test scenarios - ZERO required parameters."""
        return APIModelFactory.create_valid_llm_provider_info(**overrides)


# Convenience aliases for common patterns
class APIModelPatterns:
    """Pre-configured factory patterns for the most common usage scenarios."""

    @staticmethod
    def minimal_workflow_request(query: str = "test") -> WorkflowRequest:
        """Most common pattern: minimal valid WorkflowRequest."""
        return APIModelFactory.create_valid_workflow_request(query=query)

    # =============================================================================
    # WorkflowRequest: Zero-Parameter Convenience Methods
    # =============================================================================

    @staticmethod
    def generate_valid_data(**overrides: Any) -> WorkflowRequest:
        """Standard valid WorkflowRequest for most test scenarios - ZERO required parameters.

        This is the most commonly used factory method that eliminates verbose parameter
        specifications. Provides sensible defaults for all parameters including execution_config.

        Args:
            **overrides: Optional field overrides for test-specific needs

        Returns:
            WorkflowRequest with complete valid defaults and no unfilled parameter warnings

        Example:
            # Zero parameters - most common usage (85% of test cases)
            request = APIModelFactory.generate_valid_data()

            # Override only specific values for test logic
            request = APIModelFactory.generate_valid_data(
                query="Custom test query",
                correlation_id="test-correlation-123"
            )
        """
        # Set defaults and apply overrides
        defaults: Dict[str, Any] = {
            "query": "What is artificial intelligence?",
            "agents": ["refiner", "critic"],
            "execution_config": {"timeout_seconds": 30, "parallel_execution": True},
            "correlation_id": None,
        }
        defaults.update(overrides)
        return APIModelFactory.create_valid_workflow_request(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> WorkflowRequest:
        """Minimal valid WorkflowRequest for lightweight test scenarios.

        Provides minimal but complete data for tests that don't require complex configuration.
        Still eliminates unfilled parameter warnings by providing explicit defaults.

        Args:
            **overrides: Optional field overrides

        Returns:
            WorkflowRequest with minimal valid configuration
        """
        # Set defaults and apply overrides
        defaults: Dict[str, Any] = {
            "query": "Test query",
            "agents": ["refiner"],
            "execution_config": {"timeout_seconds": 10},
            "correlation_id": None,
        }
        defaults.update(overrides)
        return APIModelFactory.create_valid_workflow_request(**defaults)

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> WorkflowRequest:
        """Valid WorkflowRequest with dynamic correlation ID for realistic tests.

        Uses timestamp-based correlation ID for tests that need unique identifiers.

        Args:
            **overrides: Optional field overrides

        Returns:
            WorkflowRequest with timestamp-based correlation ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        # Set defaults and apply overrides
        defaults: Dict[str, Any] = {
            "query": "Test query with timestamp",
            "agents": ["refiner", "critic"],
            "execution_config": {"timeout_seconds": 30, "parallel_execution": True},
            "correlation_id": f"test-{timestamp}-{uuid.uuid4().hex[:8]}",
        }
        defaults.update(overrides)
        return APIModelFactory.create_valid_workflow_request(**defaults)

    @staticmethod
    def completed_workflow() -> WorkflowResponse:
        """Common pattern: completed workflow with all outputs."""
        return APIModelFactory.create_completed_workflow_response()

    @staticmethod
    def running_status() -> StatusResponse:
        """Common pattern: workflow in progress."""
        return APIModelFactory.create_running_status_response()

    @staticmethod
    def simple_completion() -> CompletionResponse:
        """Common pattern: basic LLM completion response."""
        return APIModelFactory.create_valid_completion_response()

    @staticmethod
    def topic_with_search(similarity: float = 0.85) -> TopicSummary:
        """Common pattern: topic result with similarity score."""
        return APIModelFactory.create_valid_topic_summary(similarity_score=similarity)

    @staticmethod
    def academic_workflow() -> WorkflowMetadata:
        """Common pattern: academic research workflow."""
        return APIModelFactory.create_valid_workflow_metadata()

    @staticmethod
    def simple_execution_graph() -> InternalExecutionGraph:
        """Common pattern: simple execution graph."""
        return APIModelFactory.create_internal_execution_graph_simple()

    @staticmethod
    def basic_agent_metrics(agent_name: str = "refiner") -> InternalAgentMetrics:
        """Common pattern: basic agent execution metrics."""
        return APIModelFactory.create_valid_internal_agent_metrics(
            agent_name=agent_name
        )
