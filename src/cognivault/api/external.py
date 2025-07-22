"""
External API contracts for CogniVault.

These APIs form the stable interface that external consumers depend on.
Breaking changes require migration path and version bump.
"""

from typing import List, Dict
from .base import BaseAPI
from .decorators import ensure_initialized
from .models import (
    WorkflowRequest,
    WorkflowResponse,
    StatusResponse,
    CompletionRequest,
    CompletionResponse,
    LLMProviderInfo,
)


class OrchestrationAPI(BaseAPI):
    """
    Public orchestration API - STABLE INTERFACE.

    This API contract must remain backward compatible.
    Breaking changes require migration path and version bump.
    """

    @property
    def api_name(self) -> str:
        return "Orchestration API"

    @property
    def api_version(self) -> str:
        return "1.0.0"

    @ensure_initialized
    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """
        Execute a workflow with specified agents.

        Args:
            request: Workflow execution parameters

        Returns:
            WorkflowResponse with execution results

        Raises:
            ValueError: Invalid request parameters
            RuntimeError: Execution failure
        """
        raise NotImplementedError("Subclasses must implement execute_workflow")

    @ensure_initialized
    async def get_status(self, workflow_id: str) -> StatusResponse:
        """
        Get workflow execution status.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            StatusResponse with current status

        Raises:
            KeyError: Workflow ID not found
        """
        raise NotImplementedError("Subclasses must implement execute_workflow")

    @ensure_initialized
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel running workflow.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            True if successfully cancelled, False if already completed
        """
        raise NotImplementedError("Subclasses must implement cancel_workflow")


class LLMGatewayAPI(BaseAPI):
    """
    LLM Gateway API - Future service extraction boundary.

    Designed for eventual extraction as independent microservice.
    """

    @property
    def api_name(self) -> str:
        return "LLM Gateway API"

    @property
    def api_version(self) -> str:
        return "1.0.0"

    @ensure_initialized
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate LLM completion.

        Args:
            request: Completion parameters

        Returns:
            CompletionResponse with generated text
        """
        raise NotImplementedError("Subclasses must implement execute_workflow")

    @ensure_initialized
    async def get_providers(self) -> List[LLMProviderInfo]:
        """
        Get available LLM providers and models.

        Returns:
            List of available LLM providers
        """
        raise NotImplementedError("Subclasses must implement execute_workflow")

    @ensure_initialized
    async def estimate_cost(self, request: CompletionRequest) -> Dict[str, float]:
        """
        Estimate completion cost across providers.

        Args:
            request: Completion parameters

        Returns:
            Cost estimates by provider
        """
        raise NotImplementedError("Subclasses must implement estimate_cost")
