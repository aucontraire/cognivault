"""
Unit tests for LangGraphOrchestrationAPI.

Comprehensive test coverage for the production API implementation.
"""

import pytest
from typing import Any, Tuple
import uuid
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from datetime import datetime, timezone

from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.api.models import WorkflowRequest, WorkflowResponse, StatusResponse
from cognivault.api.base import APIHealthStatus
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)
from cognivault.exceptions import StateTransitionError


class TestLangGraphOrchestrationAPIInitialization:
    """Test LangGraphOrchestrationAPI initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialization_success(self) -> None:
        """Test successful API initialization."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            await api.initialize()

            assert api._initialized is True
            assert api._orchestrator is not None
            mock_orchestrator_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialization_idempotent(self) -> None:
        """Test that double initialization is idempotent."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            await api.initialize()
            first_orchestrator = api._orchestrator

            await api.initialize()  # Second call

            assert api._orchestrator is first_orchestrator
            mock_orchestrator_class.assert_called_once()  # Only called once

    @pytest.mark.asyncio
    async def test_initialization_failure(self) -> None:
        """Test API initialization failure handling."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator_class.side_effect = Exception(
                "Orchestrator initialization failed"
            )

            with pytest.raises(Exception, match="Orchestrator initialization failed"):
                await api.initialize()

            assert api._initialized is False
            assert api._orchestrator is None

    @pytest.mark.asyncio
    async def test_shutdown_success(self) -> None:
        """Test successful API shutdown."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator.clear_graph_cache = Mock()  # Not async
            mock_orchestrator_class.return_value = mock_orchestrator

            await api.initialize()
            await api.shutdown()

            assert api._initialized is False
            # Note: orchestrator is not set to None in real implementation, just cache cleared
            assert len(api._active_workflows) == 0

    @pytest.mark.asyncio
    async def test_shutdown_without_initialization(self) -> None:
        """Test shutdown when not initialized."""
        api = LangGraphOrchestrationAPI()

        # Should not raise an exception
        await api.shutdown()

        assert api._initialized is False
        assert api._orchestrator is None


class TestLangGraphOrchestrationAPIWorkflowExecution:
    """Test workflow execution functionality."""

    @pytest.fixture
    async def initialized_api(self) -> Any:
        """Provide an initialized API instance."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            # Set up clear_graph_cache as a regular (non-async) method to avoid warnings
            mock_orchestrator.clear_graph_cache = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful orchestrator execution
            mock_context = AgentContextFactory.basic(
                query="Test query",
                agent_outputs={
                    "refiner": "Refined output",
                    "critic": "Critical analysis",
                },
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()
            yield api, mock_orchestrator
            await api.shutdown()

    @pytest.mark.asyncio
    async def test_execute_workflow_success(
        self, initialized_api: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test successful workflow execution."""
        api, mock_orchestrator = initialized_api

        request = WorkflowRequest(
            query="Test workflow execution",
            agents=["refiner", "critic"],
            correlation_id="test-123",
        )

        # Mock event functions to prevent actual event emission during test
        with (
            patch("cognivault.events.emit_workflow_started") as mock_emit_started,
            patch("cognivault.events.emit_workflow_completed") as mock_emit_completed,
        ):
            response = await api.execute_workflow(request)

            # Validate response - this is the core API contract
            assert isinstance(response, WorkflowResponse)
            assert response.status == "completed"
            assert response.workflow_id is not None
            assert response.correlation_id == "test-123"
            assert "refiner" in response.agent_outputs
            assert "critic" in response.agent_outputs
            assert response.execution_time_seconds > 0
            assert response.error_message is None

            # Verify orchestrator was called correctly - this is the core functionality
            # Note: workflow_id is now passed to prevent duplicate ID generation (PATTERN 5 fix)
            call_args = mock_orchestrator.run.call_args[0]  # Get positional args
            call_kwargs = (
                mock_orchestrator.run.call_args[1]
                if mock_orchestrator.run.call_args[1]
                else {}
            )
            call_config = (
                call_args[1] if len(call_args) > 1 else call_kwargs.get("config", {})
            )

            assert call_args[0] == "Test workflow execution"  # Query should match
            assert (
                call_config["correlation_id"] == "test-123"
            )  # Correlation ID should match
            assert call_config["agents"] == ["refiner", "critic"]  # Agents should match
            assert (
                "workflow_id" in call_config
            )  # Workflow ID should be passed to prevent duplicates
            assert isinstance(
                call_config["workflow_id"], str
            )  # Should be a string UUID

            # Verify workflow is tracked - this is part of the API contract
            assert response.workflow_id in api._active_workflows

            # Note: Event emission testing is better done separately in dedicated event system tests
            # rather than asserting on internal implementation details here

    @pytest.mark.asyncio
    async def test_execute_workflow_not_initialized(self) -> None:
        """Test workflow execution when API is not initialized."""
        api = LangGraphOrchestrationAPI()

        request = WorkflowRequest(query="Test", agents=["refiner"])

        with pytest.raises(
            RuntimeError, match="LangGraphOrchestrationAPI must be initialized"
        ):
            await api.execute_workflow(request)

    @pytest.mark.asyncio
    async def test_execute_workflow_orchestrator_failure(
        self, initialized_api: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test workflow execution when orchestrator fails."""
        api, mock_orchestrator = initialized_api

        # Configure orchestrator to fail
        mock_orchestrator.run.side_effect = Exception("Orchestrator execution failed")

        request = WorkflowRequest(
            query="Test failure", agents=["refiner"], correlation_id="fail-123"
        )

        # Mock event functions to prevent actual event emission during test
        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Validate error response - this is the core API contract
            assert response.status == "failed"
            assert response.error_message is not None
            assert "Orchestrator execution failed" in response.error_message
            assert response.correlation_id == "fail-123"
            assert response.execution_time_seconds > 0

            # Focus on testing the API contract, not internal event implementation

    @pytest.mark.asyncio
    async def test_execute_workflow_state_transition_error(
        self, initialized_api: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test workflow execution with StateTransitionError."""
        api, mock_orchestrator = initialized_api

        # Configure orchestrator to raise StateTransitionError
        state_error = StateTransitionError(
            transition_type="agent_execution_failed",
            from_state="running",
            to_state="failed",
            state_details="Agent failed to execute",
            step_id="step-1",
            agent_id="refiner",
        )
        mock_orchestrator.run.side_effect = state_error

        request = WorkflowRequest(query="Test state error", agents=["refiner"])

        response = await api.execute_workflow(request)

        assert response.status == "failed"
        assert response.error_message is not None
        assert "StateTransitionError" in response.error_message
        assert "agent_execution_failed" in response.error_message

    @pytest.mark.asyncio
    async def test_execute_workflow_with_default_agents(
        self, initialized_api: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test workflow execution with default agents when none specified."""
        api, mock_orchestrator = initialized_api

        request = WorkflowRequest(query="Test default agents")  # No agents specified

        response = await api.execute_workflow(request)

        assert response.status == "completed"
        # Should use default agents from orchestrator


class TestLangGraphOrchestrationAPIStatus:
    """Test status query functionality."""

    @pytest.fixture
    async def api_with_workflow(self) -> Any:
        """Provide API with a completed workflow."""
        api = LangGraphOrchestrationAPI()

        with patch("cognivault.api.orchestration_api.LangGraphOrchestrator"):
            await api.initialize()

            # Simulate a completed workflow
            workflow_id = str(uuid.uuid4())
            workflow_data = {
                "status": "completed",
                "query": "Test query",
                "agents": ["refiner", "critic"],
                "created_at": datetime.now(timezone.utc),
                "execution_time": 1.5,
                "agent_outputs": {"refiner": "output1", "critic": "output2"},
            }
            api._active_workflows[workflow_id] = workflow_data

            yield api, workflow_id
            await api.shutdown()

    @pytest.mark.asyncio
    async def test_get_status_success(
        self, api_with_workflow: Tuple[LangGraphOrchestrationAPI, str]
    ) -> None:
        """Test successful status retrieval."""
        api, workflow_id = api_with_workflow

        status = await api.get_status(workflow_id)

        assert isinstance(status, StatusResponse)
        assert status.workflow_id == workflow_id
        assert status.status == "completed"
        assert status.progress_percentage == 100.0
        assert status.current_agent is None  # Completed workflows have no current agent

    @pytest.mark.asyncio
    async def test_get_status_not_found(
        self, api_with_workflow: Tuple[LangGraphOrchestrationAPI, str]
    ) -> None:
        """Test status query for non-existent workflow."""
        api, _ = api_with_workflow

        non_existent_id = str(uuid.uuid4())

        with pytest.raises(KeyError, match=f"Workflow {non_existent_id} not found"):
            await api.get_status(non_existent_id)

    @pytest.mark.asyncio
    async def test_get_status_not_initialized(self) -> None:
        """Test status query when API is not initialized."""
        api = LangGraphOrchestrationAPI()

        with pytest.raises(
            RuntimeError, match="LangGraphOrchestrationAPI must be initialized"
        ):
            await api.get_status("any-id")

    @pytest.mark.asyncio
    async def test_get_status_running_workflow(
        self, api_with_workflow: Tuple[LangGraphOrchestrationAPI, str]
    ) -> None:
        """Test status for a running workflow."""
        api, _ = api_with_workflow

        # Add a running workflow
        running_id = str(uuid.uuid4())
        api._active_workflows[running_id] = {
            "status": "running",
            "query": "Running query",
            "agents": ["refiner", "critic", "synthesis"],
            "start_time": time.time(),  # Use time.time() as expected by get_status
        }

        status = await api.get_status(running_id)

        assert status.status == "running"
        assert 0 < status.progress_percentage < 100
        assert status.current_agent is not None


class TestLangGraphOrchestrationAPICancellation:
    """Test workflow cancellation functionality."""

    @pytest.fixture
    async def api_with_workflows(self) -> Any:
        """Provide API with multiple workflows."""
        api = LangGraphOrchestrationAPI()

        with patch("cognivault.api.orchestration_api.LangGraphOrchestrator"):
            await api.initialize()

            # Add multiple workflows
            workflow_ids = []
            for i, status in enumerate(["running", "completed", "failed"]):
                workflow_id = str(uuid.uuid4())
                api._active_workflows[workflow_id] = {
                    "status": status,
                    "query": f"Test query {i}",
                    "created_at": datetime.now(timezone.utc),
                }
                workflow_ids.append(workflow_id)

            yield api, workflow_ids
            await api.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_workflow_running(
        self, api_with_workflows: Tuple[LangGraphOrchestrationAPI, list[str]]
    ) -> None:
        """Test cancelling a running workflow."""
        api, workflow_ids = api_with_workflows
        running_id = workflow_ids[0]  # First one is running

        result = await api.cancel_workflow(running_id)

        assert result is True
        assert running_id not in api._active_workflows

    @pytest.mark.asyncio
    async def test_cancel_workflow_completed(
        self, api_with_workflows: Tuple[LangGraphOrchestrationAPI, list[str]]
    ) -> None:
        """Test cancelling a completed workflow."""
        api, workflow_ids = api_with_workflows
        completed_id = workflow_ids[1]  # Second one is completed

        result = await api.cancel_workflow(completed_id)

        assert result is False  # Cannot cancel completed workflows
        assert completed_id in api._active_workflows  # Still tracked

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_found(
        self, api_with_workflows: Tuple[LangGraphOrchestrationAPI, list[str]]
    ) -> None:
        """Test cancelling a non-existent workflow."""
        api, _ = api_with_workflows

        non_existent_id = str(uuid.uuid4())
        result = await api.cancel_workflow(non_existent_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_initialized(self) -> None:
        """Test cancellation when API is not initialized."""
        api = LangGraphOrchestrationAPI()

        with pytest.raises(
            RuntimeError, match="LangGraphOrchestrationAPI must be initialized"
        ):
            await api.cancel_workflow("any-id")


class TestLangGraphOrchestrationAPIHealth:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_initialized(self) -> None:
        """Test health check when API is initialized."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator: Mock = Mock()
            mock_orchestrator.get_execution_statistics = Mock(
                return_value={"total_executions": 10, "failed_executions": 2}
            )
            mock_orchestrator_class.return_value = mock_orchestrator

            await api.initialize()

            health = await api.health_check()

            assert isinstance(health, APIHealthStatus)
            # With 2/10 failures (20%), status should be healthy
            assert health.status.value == "healthy"
            assert health.timestamp is not None
            assert health.checks["initialized"] is True
            assert health.checks["orchestrator_available"] is True

            await api.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self) -> None:
        """Test health check when API is not initialized."""
        api = LangGraphOrchestrationAPI()

        health = await api.health_check()

        assert health.status.value == "unhealthy"
        assert health.checks["initialized"] is False
        assert health.checks["orchestrator_available"] is False

    @pytest.mark.asyncio
    async def test_health_check_with_active_workflows(self) -> None:
        """Test health check includes active workflow count."""
        api = LangGraphOrchestrationAPI()

        with patch("cognivault.api.orchestration_api.LangGraphOrchestrator"):
            await api.initialize()

            # Add some active workflows
            for i in range(3):
                workflow_id = str(uuid.uuid4())
                api._active_workflows[workflow_id] = {"status": "running"}

            health = await api.health_check()

            assert health.checks["active_workflows"] == 3

            await api.shutdown()


class TestLangGraphOrchestrationAPIMetrics:
    """Test metrics functionality."""

    @pytest.mark.asyncio
    async def test_get_metrics_initialized(self) -> None:
        """Test metrics when API is initialized."""
        api = LangGraphOrchestrationAPI()

        with patch("cognivault.api.orchestration_api.LangGraphOrchestrator"):
            await api.initialize()

            # Add some workflows for metrics
            for i, status in enumerate(["running", "completed", "failed"]):
                workflow_id = str(uuid.uuid4())
                api._active_workflows[workflow_id] = {
                    "status": status,
                    "created_at": datetime.now(timezone.utc),
                }
            # Increment total workflows counter to match what execute_workflow does
            api._total_workflows = 3

            metrics = await api.get_metrics()

            assert isinstance(metrics, dict)
            assert metrics["api_initialized"] is True
            assert metrics["total_workflows_processed"] == 3
            assert metrics["active_workflows"] == 3  # All are in _active_workflows dict
            assert "timestamp" in metrics

            await api.shutdown()

    @pytest.mark.asyncio
    async def test_get_metrics_not_initialized(self) -> None:
        """Test metrics when API is not initialized."""
        api = LangGraphOrchestrationAPI()

        metrics = await api.get_metrics()

        assert metrics["api_initialized"] is False
        assert metrics["total_workflows_processed"] == 0
        assert metrics["active_workflows"] == 0

    @pytest.mark.asyncio
    async def test_get_metrics_empty_workflows(self) -> None:
        """Test metrics with no active workflows."""
        api = LangGraphOrchestrationAPI()

        with patch("cognivault.api.orchestration_api.LangGraphOrchestrator"):
            await api.initialize()

            metrics = await api.get_metrics()

            assert metrics["total_workflows_processed"] == 0
            assert metrics["active_workflows"] == 0

            await api.shutdown()


class TestLangGraphOrchestrationAPIErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_workflow_tracking_cleanup(self) -> None:
        """Test that workflows are properly tracked and cleaned up."""
        api = LangGraphOrchestrationAPI()

        with patch("cognivault.api.orchestration_api.LangGraphOrchestrator"):
            await api.initialize()

            # Add workflows
            workflow_ids = []
            for i in range(5):
                workflow_id = str(uuid.uuid4())
                api._active_workflows[workflow_id] = {"status": "completed"}
                workflow_ids.append(workflow_id)

            assert len(api._active_workflows) == 5

            # Cancel some workflows
            for workflow_id in workflow_ids[:3]:
                await api.cancel_workflow(workflow_id)

            # Only non-cancelled, non-completed workflows should remain
            # In this case, all were completed, so cancellation should fail
            assert len(api._active_workflows) == 5

            await api.shutdown()

            # After shutdown, completed workflows remain for history/metrics
            # Only running workflows would have been cancelled and removed
            assert (
                len(api._active_workflows) == 5
            )  # All were completed, so none removed

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self) -> None:
        """Test handling of concurrent workflow executions."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            # Set up clear_graph_cache as a regular (non-async) method to avoid warnings
            mock_orchestrator.clear_graph_cache = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock context for each execution
            def create_mock_context(query: str) -> Any:
                return AgentContextFactory.basic(
                    query=query, agent_outputs={"refiner": f"Output for {query}"}
                )

            mock_orchestrator.run.side_effect = lambda q, config: create_mock_context(q)

            await api.initialize()

            # Execute multiple workflows concurrently
            import asyncio

            requests = [
                WorkflowRequest(query=f"Query {i}", agents=["refiner"])
                for i in range(3)
            ]

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                responses = await asyncio.gather(
                    *[api.execute_workflow(req) for req in requests]
                )

            # Verify all workflows completed successfully
            assert len(responses) == 3
            workflow_ids = [resp.workflow_id for resp in responses]
            assert len(set(workflow_ids)) == 3  # All unique

            for response in responses:
                assert response.status == "completed"
                assert response.workflow_id in api._active_workflows

            await api.shutdown()

    @pytest.mark.asyncio
    async def test_event_emission_failure_handling(self) -> None:
        """Test that event emission failures don't break workflow execution."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            # Set up clear_graph_cache as a regular (non-async) method to avoid warnings
            mock_orchestrator.clear_graph_cache = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.run.return_value = AgentContextFactory.basic(
                query="Test", agent_outputs={"refiner": "output"}
            )

            await api.initialize()

            request = WorkflowRequest(query="Test", agents=["refiner"])

            # Make event emission fail
            with (
                patch(
                    "cognivault.events.emit_workflow_started",
                    side_effect=Exception("Event failed"),
                ),
                patch(
                    "cognivault.events.emit_workflow_completed",
                    side_effect=Exception("Event failed"),
                ),
            ):
                # Workflow should still complete successfully
                response = await api.execute_workflow(request)

                assert response.status == "completed"
                assert response.workflow_id is not None

            await api.shutdown()
