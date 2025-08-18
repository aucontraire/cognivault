"""
Comprehensive error handling tests for API workflow persistence.

Tests various database failure scenarios to ensure proper error isolation
and graceful degradation when persistence operations fail.
"""

import pytest
from typing import Any, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.api.models import WorkflowRequest, WorkflowResponse
from tests.factories.agent_context_factories import AgentContextFactory
from tests.utils.async_test_helpers import (
    AsyncSessionWrapper,
    create_mock_session_factory,
)


class TestPersistenceErrorHandling:
    """Test error handling scenarios for database persistence operations."""

    @pytest.fixture
    async def api_with_orchestrator(self) -> Any:
        """Provide API with working orchestrator but controllable database."""
        api = LangGraphOrchestrationAPI()

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.clear_graph_cache = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful orchestrator execution
            mock_context = AgentContextFactory.basic(
                query="Error handling test",
                agent_outputs={"refiner": "Test output for error scenarios"},
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()
            yield api, mock_orchestrator
            await api.shutdown()

    @pytest.mark.asyncio
    async def test_session_factory_initialization_failure(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling of session factory initialization failure."""
        api, mock_orchestrator = api_with_orchestrator

        # Set session factory to None to simulate initialization failure
        api._session_factory = None  # type: ignore[assignment]

        request = WorkflowRequest(
            query="Session factory failure test", agents=["refiner"]
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Workflow should complete despite session factory failure
            assert response.status == "completed"
            assert response.error_message is None

    @pytest.mark.asyncio
    async def test_session_context_manager_failure(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling of session context manager entry failure."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session factory to fail when entering context
        mock_session_factory = AsyncMock()
        mock_session_factory.side_effect = Exception("Database connection timeout")
        api._session_factory = mock_session_factory

        request = WorkflowRequest(
            query="Session context failure test",
            agents=["refiner"],
            correlation_id="session-context-failure-123",
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Workflow should complete despite session failure
            assert response.status == "completed"
            assert response.correlation_id == "session-context-failure-123"
            assert response.error_message is None

    @pytest.mark.asyncio
    async def test_question_repository_instantiation_failure(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling of QuestionRepository instantiation failure."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session factory to work but repository to fail
        mock_session = AsyncMock()

        # Use centralized async session wrapper
        api._session_factory = create_mock_session_factory(mock_session)

        with patch(
            "cognivault.api.orchestration_api.QuestionRepository"
        ) as mock_repo_class:
            mock_repo_class.side_effect = Exception("Repository initialization failed")

            request = WorkflowRequest(
                query="Repository failure test", agents=["refiner"]
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                # Workflow should still complete
                assert response.status == "completed"
                assert response.error_message is None

                # Session factory should have been called (indirectly verified by error log)

    @pytest.mark.asyncio
    async def test_create_question_method_failure(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling of create_question method failure."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session and repository but make create_question fail
        mock_session = AsyncMock()
        mock_question_repo = AsyncMock()
        mock_question_repo.create_question.side_effect = Exception(
            "Database constraint violation"
        )

        # Use centralized async session wrapper
        api._session_factory = create_mock_session_factory(mock_session)

        with patch(
            "cognivault.api.orchestration_api.QuestionRepository",
            return_value=mock_question_repo,
        ):
            request = WorkflowRequest(
                query="Create question failure test",
                agents=["refiner"],
                correlation_id="create-question-failure-456",
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                # Workflow should complete successfully
                assert response.status == "completed"
                assert response.correlation_id == "create-question-failure-456"
                assert response.error_message is None

                # create_question should have been attempted
                mock_question_repo.create_question.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_workflow_persistence_database_error(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test database error during failed workflow persistence."""
        api, mock_orchestrator = api_with_orchestrator

        # Configure orchestrator to fail
        mock_orchestrator.run.side_effect = Exception(
            "Orchestrator failure for persistence test"
        )

        # Configure database to also fail
        mock_session_factory = AsyncMock()
        mock_session_factory.side_effect = Exception(
            "Database also fails during error handling"
        )
        api._session_factory = mock_session_factory

        request = WorkflowRequest(
            query="Double failure test",
            agents=["refiner"],
            correlation_id="double-failure-789",
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Should still return failed workflow response
            assert response.status == "failed"
            assert response.error_message is not None
            assert "Orchestrator failure for persistence test" in response.error_message
            assert response.correlation_id == "double-failure-789"

            # Database failure shouldn't affect the error response

    @pytest.mark.asyncio
    async def test_session_exit_failure_handling(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling when session context manager exit fails."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session that fails on exit
        mock_session = AsyncMock()
        mock_question_repo = AsyncMock()

        # Create a context manager that fails on exit
        class FailingSessionContext:
            async def __aenter__(self) -> AsyncMock:
                return mock_session

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                raise Exception("Session cleanup failed")

        def mock_session_factory() -> FailingSessionContext:
            return FailingSessionContext()

        api._session_factory = mock_session_factory  # type: ignore[assignment]

        with patch(
            "cognivault.api.orchestration_api.QuestionRepository",
            return_value=mock_question_repo,
        ):
            request = WorkflowRequest(
                query="Session exit failure test", agents=["refiner"]
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                # Workflow should still complete
                assert response.status == "completed"
                assert response.error_message is None

                # create_question should have been called
                mock_question_repo.create_question.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_history_database_connection_failure(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test workflow history retrieval when database connection fails."""
        api, mock_orchestrator = api_with_orchestrator

        # Configure session factory to fail
        mock_session_factory = AsyncMock()
        mock_session_factory.side_effect = Exception("Database connection lost")
        api._session_factory = mock_session_factory

        # Should return empty list instead of raising exception
        history = await api.get_workflow_history_from_database(limit=10, offset=0)

        assert history == []

    @pytest.mark.asyncio
    async def test_workflow_history_repository_query_failure(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test workflow history when repository query fails."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session but make repository query fail
        mock_session = AsyncMock()
        mock_question_repo = AsyncMock()
        mock_question_repo.get_recent_questions.side_effect = Exception(
            "Query execution failed"
        )

        # Use centralized async session wrapper
        api._session_factory = create_mock_session_factory(mock_session)

        with patch(
            "cognivault.api.orchestration_api.QuestionRepository",
            return_value=mock_question_repo,
        ):
            # Should return empty list instead of raising exception
            history = await api.get_workflow_history_from_database(limit=5, offset=10)

            assert history == []

            # Repository method should have been attempted
            mock_question_repo.get_recent_questions.assert_called_once_with(
                limit=5, offset=10
            )

    @pytest.mark.asyncio
    async def test_concurrent_persistence_failures(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling of multiple concurrent persistence failures."""
        api, mock_orchestrator = api_with_orchestrator

        # Configure session factory to fail randomly
        call_count = 0

        def failing_session_factory() -> Any:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception(f"Database failure {call_count}")
            else:
                # Return working session
                mock_session = AsyncMock()
                return AsyncSessionWrapper(mock_session)

        api._session_factory = failing_session_factory  # type: ignore[assignment]

        # Execute multiple workflows concurrently
        import asyncio

        # Patch QuestionRepository at module level for concurrent test
        mock_question_repo = AsyncMock()

        requests = [
            WorkflowRequest(
                query=f"Concurrent test {i}",
                agents=["refiner"],
                correlation_id=f"concurrent-{i}",
            )
            for i in range(4)
        ]

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
            patch(
                "cognivault.api.orchestration_api.QuestionRepository",
                return_value=mock_question_repo,
            ),
        ):
            responses = await asyncio.gather(
                *[api.execute_workflow(req) for req in requests]
            )

            # All workflows should complete successfully
            for response in responses:
                assert response.status == "completed"
                assert response.error_message is None

    @pytest.mark.asyncio
    async def test_persistence_timeout_handling(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling of database operation timeouts."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session that times out
        async def timeout_session() -> None:
            import asyncio

            await asyncio.sleep(1)  # Simulate timeout
            raise asyncio.TimeoutError("Database operation timed out")

        mock_session_factory = AsyncMock()
        mock_session_factory.side_effect = timeout_session
        api._session_factory = mock_session_factory

        request = WorkflowRequest(query="Timeout test", agents=["refiner"])

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
        ):
            response = await api.execute_workflow(request)

            # Workflow should complete despite timeout
            assert response.status == "completed"
            assert response.error_message is None

    @pytest.mark.asyncio
    async def test_metadata_serialization_failure(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling when execution metadata cannot be serialized."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session and repository
        mock_session = AsyncMock()
        mock_question_repo = AsyncMock()

        # Configure create_question to fail with serialization error
        mock_question_repo.create_question.side_effect = Exception(
            "JSON serialization failed"
        )

        # Use centralized async session wrapper
        api._session_factory = create_mock_session_factory(mock_session)

        with patch(
            "cognivault.api.orchestration_api.QuestionRepository",
            return_value=mock_question_repo,
        ):
            # Create request with complex execution config that might cause serialization issues
            complex_config = {
                "nested": {
                    "deep": {
                        "structure": {"with": ["lists", "and", {"mixed": "types"}]}
                    }
                },
                "special_chars": "Special characters: 测试数据 αβγ @#$%^&*()",
            }

            request = WorkflowRequest(
                query="Serialization test with complex metadata",
                agents=["refiner"],
                execution_config=complex_config,
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                # Workflow should complete despite serialization failure
                assert response.status == "completed"
                assert response.error_message is None

    @pytest.mark.asyncio
    async def test_database_constraint_violation_handling(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test handling of database constraint violations."""
        api, mock_orchestrator = api_with_orchestrator

        # Mock session and repository
        mock_session = AsyncMock()
        mock_question_repo = AsyncMock()

        # Simulate various database constraint violations
        constraint_errors = [
            Exception("UNIQUE constraint failed: questions.correlation_id"),
            Exception("CHECK constraint failed: execution_time >= 0"),
            Exception("FOREIGN KEY constraint failed"),
            Exception("NOT NULL constraint failed: questions.query"),
        ]

        for error in constraint_errors:
            mock_question_repo.create_question.side_effect = error

            # Use centralized async session wrapper
            api._session_factory = create_mock_session_factory(mock_session)

            with patch(
                "cognivault.api.orchestration_api.QuestionRepository",
                return_value=mock_question_repo,
            ):
                request = WorkflowRequest(
                    query="Constraint violation test",
                    agents=["refiner"],
                    correlation_id=f"constraint-test-{constraint_errors.index(error)}",
                )

                with (
                    patch("cognivault.events.emit_workflow_started"),
                    patch("cognivault.events.emit_workflow_completed"),
                ):
                    response = await api.execute_workflow(request)

                    # Workflow should complete despite constraint violation
                    assert response.status == "completed"
                    assert response.error_message is None

    @pytest.mark.asyncio
    async def test_persistence_error_logging(
        self, api_with_orchestrator: Tuple[LangGraphOrchestrationAPI, AsyncMock]
    ) -> None:
        """Test that persistence errors are properly logged."""
        api, mock_orchestrator = api_with_orchestrator

        # Configure session factory to fail
        mock_session_factory = AsyncMock()
        mock_session_factory.side_effect = Exception("Database logging test failure")
        api._session_factory = mock_session_factory

        request = WorkflowRequest(
            query="Logging test workflow",
            agents=["refiner"],
            correlation_id="logging-test-123",
        )

        with (
            patch("cognivault.events.emit_workflow_started"),
            patch("cognivault.events.emit_workflow_completed"),
            patch("cognivault.api.orchestration_api.logger") as mock_logger,
        ):
            response = await api.execute_workflow(request)

            # Workflow should complete
            assert response.status == "completed"

            # Error should be logged
            mock_logger.error.assert_called()

            # Verify log message contains relevant information
            error_log_call = mock_logger.error.call_args[0][0]
            assert "Failed to persist workflow" in error_log_call
            assert response.workflow_id in error_log_call
