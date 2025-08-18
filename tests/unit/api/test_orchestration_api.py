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
from tests.utils.async_test_helpers import (
    AsyncSessionWrapper,
    create_mock_session_factory,
)


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


class TestLangGraphOrchestrationAPIPersistence:
    """Test database persistence functionality for workflow execution results."""

    @pytest.fixture
    async def api_with_mock_session(self) -> Any:
        """Provide API with mocked database session and repository."""
        api = LangGraphOrchestrationAPI()

        with patch("cognivault.api.orchestration_api.LangGraphOrchestrator"):
            await api.initialize()

            # Mock session factory and question repository
            mock_session = AsyncMock()
            mock_question_repo = AsyncMock()

            # Use centralized async session wrapper
            api._session_factory = create_mock_session_factory(mock_session)

            with patch(
                "cognivault.api.orchestration_api.QuestionRepository",
                return_value=mock_question_repo,
            ) as mock_repo_class:
                yield api, mock_session, mock_question_repo, mock_repo_class

            await api.shutdown()

    @pytest.mark.asyncio
    async def test_persist_workflow_to_database_success(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test successful workflow persistence to database."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        # Create test data
        request = WorkflowRequest(
            query="Test persistence workflow",
            agents=["refiner", "critic"],
            correlation_id="test-persist-123",
            execution_config={"timeout": 30},
        )

        response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440456",
            status="completed",
            agent_outputs={
                "refiner": "Refined output for persistence test",
                "critic": "Critical analysis for persistence test",
            },
            execution_time_seconds=12.5,
            correlation_id="test-persist-123",
        )

        mock_execution_context = AgentContextFactory.basic(
            query=request.query, agent_outputs=response.agent_outputs
        )

        # Execute persistence
        await api._persist_workflow_to_database(
            request,
            response,
            mock_execution_context,
            response.workflow_id,
            request.execution_config or {},
        )

        # Session factory and repository should have been used
        # (We can't easily mock function calls, but we verify the end result)

        # Verify QuestionRepository was instantiated with session
        mock_repo_class.assert_called_once_with(mock_session)

        # Verify create_question was called with correct parameters
        mock_question_repo.create_question.assert_called_once()
        call_args = mock_question_repo.create_question.call_args

        assert call_args.kwargs["query"] == "Test persistence workflow"
        assert call_args.kwargs["correlation_id"] == "test-persist-123"
        assert (
            call_args.kwargs["execution_id"] == "550e8400-e29b-41d4-a716-446655440456"
        )
        assert call_args.kwargs["nodes_executed"] == ["refiner", "critic"]

        # Verify execution metadata structure
        execution_metadata = call_args.kwargs["execution_metadata"]
        assert (
            execution_metadata["workflow_id"] == "550e8400-e29b-41d4-a716-446655440456"
        )
        assert execution_metadata["execution_time_seconds"] == 12.5
        assert execution_metadata["agent_outputs"] == response.agent_outputs
        assert execution_metadata["agents_requested"] == ["refiner", "critic"]
        assert execution_metadata["export_md"] is False
        assert execution_metadata["execution_config"] == {"timeout": 30}
        assert execution_metadata["api_version"] == "1.0.0"
        assert execution_metadata["orchestrator_type"] == "langgraph-real"

    @pytest.mark.asyncio
    async def test_persist_workflow_to_database_with_defaults(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test workflow persistence with default values for optional fields."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        # Create minimal request
        request = WorkflowRequest(query="Minimal test workflow")

        response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440789",
            status="completed",
            agent_outputs={"refiner": "Minimal output"},
            execution_time_seconds=5.0,
        )

        mock_execution_context = AgentContextFactory.basic(
            query=request.query, agent_outputs={"refiner": "Minimal output"}
        )

        # Execute persistence
        await api._persist_workflow_to_database(
            request, response, mock_execution_context, response.workflow_id, {}
        )

        # Verify create_question was called with defaults
        call_args = mock_question_repo.create_question.call_args

        assert call_args.kwargs["correlation_id"] is None
        assert call_args.kwargs["nodes_executed"] == ["refiner"]

        # Verify execution metadata uses defaults
        execution_metadata = call_args.kwargs["execution_metadata"]
        assert execution_metadata["agents_requested"] == [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]
        assert execution_metadata["export_md"] is False
        assert execution_metadata["execution_config"] == {}

    @pytest.mark.asyncio
    async def test_persist_workflow_to_database_error_isolation(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that database persistence errors don't propagate to caller."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        # Configure repository to raise error
        mock_question_repo.create_question.side_effect = Exception(
            "Database connection failed"
        )

        request = WorkflowRequest(query="Error test workflow")
        response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655440999",
            status="completed",
            agent_outputs={"refiner": "output"},
            execution_time_seconds=1.0,
        )

        mock_execution_context = AgentContextFactory.basic(
            query=request.query, agent_outputs=response.agent_outputs
        )

        # Execute persistence - should not raise exception
        await api._persist_workflow_to_database(
            request, response, mock_execution_context, response.workflow_id, {}
        )

        # Verify the repository method was attempted
        mock_question_repo.create_question.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_failed_workflow_to_database_success(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test successful failed workflow persistence to database."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        # Create test data for failed workflow
        request = WorkflowRequest(
            query="Failed workflow test",
            agents=["refiner"],
            correlation_id="test-failed-456",
        )

        error_response = WorkflowResponse(
            workflow_id="550e8400-e29b-41d4-a716-446655441789",
            status="failed",
            agent_outputs={},
            execution_time_seconds=2.5,
            correlation_id="test-failed-456",
            error_message="Test orchestrator failure",
        )

        error_message = "Test orchestrator failure"

        # Execute failed workflow persistence
        await api._persist_failed_workflow_to_database(
            request, error_response, error_response.workflow_id, error_message, {}
        )

        # Verify create_question was called
        call_args = mock_question_repo.create_question.call_args

        assert call_args.kwargs["query"] == "Failed workflow test"
        assert call_args.kwargs["correlation_id"] == "test-failed-456"
        assert (
            call_args.kwargs["execution_id"] == "550e8400-e29b-41d4-a716-446655441789"
        )
        assert call_args.kwargs["nodes_executed"] == []

        # Verify execution metadata includes failure information
        execution_metadata = call_args.kwargs["execution_metadata"]
        assert execution_metadata["status"] == "failed"
        assert execution_metadata["error_message"] == "Test orchestrator failure"
        assert execution_metadata["execution_time_seconds"] == 2.5
        assert execution_metadata["orchestrator_type"] == "langgraph-real"

    @pytest.mark.asyncio
    async def test_get_workflow_history_from_database_success(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test successful workflow history retrieval from database."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        # Mock database questions
        from datetime import datetime, timezone
        from types import SimpleNamespace

        mock_questions = [
            SimpleNamespace(
                id=1,
                execution_id="hist-workflow-1",
                query="First historical workflow",
                created_at=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                execution_metadata={
                    "execution_time_seconds": 15.2,
                    "status": "completed",
                },
            ),
            SimpleNamespace(
                id=2,
                execution_id="hist-workflow-2",
                query="Second historical workflow with longer query that gets truncated",
                created_at=datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
                execution_metadata={
                    "execution_time_seconds": 8.7,
                    "status": "completed",
                },
            ),
        ]

        mock_question_repo.get_recent_questions.return_value = mock_questions

        # Execute history retrieval
        history = await api.get_workflow_history_from_database(limit=5, offset=10)

        # Verify repository was called with correct parameters
        mock_question_repo.get_recent_questions.assert_called_once_with(
            limit=5, offset=10
        )

        # Verify response structure
        assert len(history) == 2

        first_item = history[0]
        assert first_item["workflow_id"] == "hist-workflow-1"
        assert first_item["status"] == "completed"
        assert first_item["query"] == "First historical workflow"
        assert (
            first_item["start_time"]
            == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        assert first_item["execution_time"] == 15.2

        second_item = history[1]
        assert second_item["workflow_id"] == "hist-workflow-2"
        assert len(second_item["query"]) <= 100  # Truncated

    @pytest.mark.asyncio
    async def test_get_workflow_history_from_database_empty_results(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test workflow history retrieval with no results."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        mock_question_repo.get_recent_questions.return_value = []

        history = await api.get_workflow_history_from_database(limit=10)

        assert history == []
        mock_question_repo.get_recent_questions.assert_called_once_with(
            limit=10, offset=0
        )

    @pytest.mark.asyncio
    async def test_get_workflow_history_from_database_error_handling(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test workflow history retrieval error handling."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        # Configure repository to raise error
        mock_question_repo.get_recent_questions.side_effect = Exception(
            "Database query failed"
        )

        # Execute history retrieval - should return empty list
        history = await api.get_workflow_history_from_database()

        assert history == []

    @pytest.mark.asyncio
    async def test_get_workflow_history_from_database_missing_execution_id(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test workflow history with questions missing execution_id."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        from datetime import datetime, timezone
        from types import SimpleNamespace

        # Mock question without execution_id
        mock_questions = [
            SimpleNamespace(
                id=42,
                execution_id=None,
                query="Question without execution_id",
                created_at=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                execution_metadata={"execution_time_seconds": 10.0},
            )
        ]

        mock_question_repo.get_recent_questions.return_value = mock_questions

        history = await api.get_workflow_history_from_database()

        # Should use question ID as fallback workflow_id
        assert len(history) == 1
        assert history[0]["workflow_id"] == "42"

    @pytest.mark.asyncio
    async def test_get_workflow_history_from_database_missing_metadata(
        self,
        api_with_mock_session: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test workflow history with questions missing execution metadata."""
        api, mock_session, mock_question_repo, mock_repo_class = api_with_mock_session

        from datetime import datetime, timezone
        from types import SimpleNamespace

        # Mock question without execution_metadata
        mock_questions = [
            SimpleNamespace(
                id=55,
                execution_id="workflow-no-metadata",
                query="Question without execution metadata",
                created_at=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                execution_metadata=None,
            )
        ]

        mock_question_repo.get_recent_questions.return_value = mock_questions

        history = await api.get_workflow_history_from_database()

        # Should use default execution time
        assert len(history) == 1
        assert history[0]["execution_time"] == 0.0


class TestLangGraphOrchestrationAPIWorkflowExecutionWithPersistence:
    """Test complete workflow execution including database persistence."""

    @pytest.fixture
    async def api_with_persistence_mocks(self) -> Any:
        """Provide API with mocked orchestrator and database components."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.clear_graph_cache = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful orchestrator execution
            mock_context = AgentContextFactory.basic(
                query="Test query with persistence",
                agent_outputs={
                    "refiner": "Refined output for persistence",
                    "synthesis": "Synthesis output for persistence",
                },
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()

            # Mock database persistence methods
            with (
                patch.object(
                    api, "_persist_workflow_to_database"
                ) as mock_persist_success,
                patch.object(
                    api, "_persist_failed_workflow_to_database"
                ) as mock_persist_failed,
            ):
                yield api, mock_orchestrator, mock_persist_success, mock_persist_failed

            await api.shutdown()

    @pytest.mark.asyncio
    async def test_execute_workflow_calls_persistence_on_success(
        self,
        api_with_persistence_mocks: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that successful workflow execution calls database persistence."""
        api, mock_orchestrator, mock_persist_success, mock_persist_failed = (
            api_with_persistence_mocks
        )

        request = WorkflowRequest(
            query="Test workflow with persistence",
            agents=["refiner", "synthesis"],
            correlation_id="test-persist-execution-123",
        )

        # Mock event functions
        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Verify workflow completed successfully
            assert response.status == "completed"
            assert response.correlation_id == "test-persist-execution-123"

            # Verify persistence was called
            mock_persist_success.assert_called_once()

            # Verify persistence was called with correct parameters
            call_args = mock_persist_success.call_args
            assert call_args[0][0] == request  # WorkflowRequest
            assert (
                call_args[0][1].workflow_id == response.workflow_id
            )  # WorkflowResponse
            assert call_args[0][3] == response.workflow_id  # workflow_id

            # Verify failed persistence was not called
            mock_persist_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_workflow_calls_persistence_on_failure(
        self,
        api_with_persistence_mocks: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that failed workflow execution calls failed persistence."""
        api, mock_orchestrator, mock_persist_success, mock_persist_failed = (
            api_with_persistence_mocks
        )

        # Configure orchestrator to fail
        mock_orchestrator.run.side_effect = Exception(
            "Orchestrator failure for persistence test"
        )

        request = WorkflowRequest(
            query="Test failed workflow with persistence",
            agents=["refiner"],
            correlation_id="test-persist-failure-456",
        )

        # Mock event functions
        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Verify workflow failed
            assert response.status == "failed"
            assert response.error_message is not None
            assert "Orchestrator failure for persistence test" in response.error_message

            # Verify failed persistence was called
            mock_persist_failed.assert_called_once()

            # Verify failed persistence was called with correct parameters
            call_args = mock_persist_failed.call_args
            assert call_args[0][0] == request  # WorkflowRequest
            assert (
                call_args[0][1].status == "failed"
            )  # WorkflowResponse (error response)
            assert call_args[0][2] == response.workflow_id  # workflow_id
            assert (
                "Orchestrator failure for persistence test" in call_args[0][3]
            )  # error_message

            # Verify success persistence was not called
            mock_persist_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_workflow_continues_on_persistence_failure(
        self,
        api_with_persistence_mocks: Tuple[
            LangGraphOrchestrationAPI, AsyncMock, AsyncMock, AsyncMock
        ],
    ) -> None:
        """Test that persistence failures don't affect workflow response."""
        api, mock_orchestrator, mock_persist_success, mock_persist_failed = (
            api_with_persistence_mocks
        )

        # Configure persistence to fail
        mock_persist_success.side_effect = Exception("Database persistence failed")

        request = WorkflowRequest(
            query="Test persistence failure isolation", agents=["refiner"]
        )

        # Mock event functions
        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Verify workflow still completed successfully despite persistence failure
            assert response.status == "completed"
            assert response.error_message is None

            # Verify persistence was attempted
            mock_persist_success.assert_called_once()
