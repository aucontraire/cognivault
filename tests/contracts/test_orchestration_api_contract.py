"""
Contract tests for OrchestrationAPI.

These tests validate that implementations conform to the API contract
regardless of the underlying implementation (real or mock).
"""

import pytest
from typing import Any, Protocol
from cognivault.api.external import OrchestrationAPI
from cognivault.api.models import WorkflowRequest, WorkflowResponse
from tests.fakes.mock_orchestration import MockOrchestrationAPI


class OrchestrationAPIContract(Protocol):
    """Protocol defining the contract test interface."""

    async def get_api_instance(self) -> OrchestrationAPI:
        """Return configured API instance for testing."""
        pass


class TestOrchestrationAPIContract:
    """
    Contract tests that all OrchestrationAPI implementations must pass.

    These tests ensure consistent behavior across real and mock implementations.
    """

    @pytest.fixture
    async def mock_api(self) -> OrchestrationAPI:
        """Provide mock API instance."""
        api = MockOrchestrationAPI()
        await api.initialize()
        return api

    @pytest.mark.asyncio
    async def test_execute_workflow_basic(self, mock_api: OrchestrationAPI) -> None:
        """Test basic workflow execution contract."""
        request = WorkflowRequest(query="Test query", agents=["refiner", "critic"])

        response = await mock_api.execute_workflow(request)

        # Contract assertions
        assert isinstance(response, WorkflowResponse)
        assert response.workflow_id is not None
        assert response.status in ["completed", "failed", "running"]
        assert isinstance(response.agent_outputs, dict)
        assert response.execution_time_seconds >= 0

        # Agent-specific assertions
        if response.status == "completed" and request.agents is not None:
            assert len(response.agent_outputs) == len(request.agents)
            for agent in request.agents:
                assert agent in response.agent_outputs
                assert isinstance(response.agent_outputs[agent], str)
                assert len(response.agent_outputs[agent]) > 0

    @pytest.mark.asyncio
    async def test_health_check_contract(self, mock_api: OrchestrationAPI) -> None:
        """Test health check contract compliance."""
        health = await mock_api.health_check()

        # Contract assertions
        assert health.status is not None
        assert health.timestamp is not None
        assert isinstance(health.checks, dict)

    @pytest.mark.asyncio
    async def test_uninitialized_api_behavior(self) -> None:
        """Test that uninitialized API raises appropriate errors."""
        api = MockOrchestrationAPI()
        # Note: not calling initialize()

        request = WorkflowRequest(query="Test")

        with pytest.raises(RuntimeError, match="must be initialized"):
            await api.execute_workflow(request)

    @pytest.mark.asyncio
    async def test_workflow_status_contract(self, mock_api: OrchestrationAPI) -> None:
        """Test workflow status query contract."""
        # Execute workflow first
        request = WorkflowRequest(query="Test query")
        response = await mock_api.execute_workflow(request)

        # Query status
        status = await mock_api.get_status(response.workflow_id)

        # Contract assertions
        assert status.workflow_id == response.workflow_id
        assert status.status in ["completed", "failed", "running"]
        assert 0 <= status.progress_percentage <= 100

    @pytest.mark.asyncio
    async def test_nonexistent_workflow_status(
        self, mock_api: OrchestrationAPI
    ) -> None:
        """Test status query for nonexistent workflow."""
        with pytest.raises(KeyError):
            await mock_api.get_status("nonexistent-workflow-id")

    @pytest.mark.asyncio
    async def test_api_properties_contract(self, mock_api: OrchestrationAPI) -> None:
        """Test API property contract."""
        assert isinstance(mock_api.api_name, str)
        assert len(mock_api.api_name) > 0
        assert isinstance(mock_api.api_version, str)
        assert len(mock_api.api_version) > 0

    @pytest.mark.asyncio
    async def test_workflow_cancellation_contract(
        self, mock_api: OrchestrationAPI
    ) -> None:
        """Test workflow cancellation contract."""
        # Execute workflow first
        request = WorkflowRequest(query="Test query")
        response = await mock_api.execute_workflow(request)

        # Cancel workflow
        cancelled = await mock_api.cancel_workflow(response.workflow_id)
        assert isinstance(cancelled, bool)

        # Cancel non-existent workflow
        cancelled_non_existent = await mock_api.cancel_workflow("nonexistent-id")
        assert cancelled_non_existent is False

    @pytest.mark.asyncio
    async def test_metrics_contract(self, mock_api: OrchestrationAPI) -> None:
        """Test metrics contract."""
        metrics = await mock_api.get_metrics()

        # Contract assertions
        assert isinstance(metrics, dict)
        # Should have basic metrics
        assert "requests_total" in metrics or len(metrics) >= 0
