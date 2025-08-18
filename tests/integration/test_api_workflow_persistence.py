"""
Integration tests for API workflow persistence implementation.

Tests end-to-end workflow execution with actual database persistence,
ensuring proper integration between API, orchestrator, and database layers.
"""

import pytest
import uuid
from typing import Any
from datetime import datetime, timezone

from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.database.repositories.factory import RepositoryFactory

from tests.utils.async_test_helpers import create_mock_session_factory
from tests.factories import APIModelPatterns


@pytest.mark.asyncio
async def test_workflow_persistence_end_to_end_success(
    integration_db_session: Any,
) -> None:
    """Test complete workflow execution with database persistence."""
    # Use provided integration database session
    session = integration_db_session
    repos = RepositoryFactory(session)

    # Create API instance with real database session
    api = LangGraphOrchestrationAPI()

    # Replace session factory to use our test session
    original_session_factory = api._session_factory

    # Use centralized async session wrapper
    api._session_factory = create_mock_session_factory(session)

    try:
        # Initialize API (mock orchestrator since we're testing persistence only)
        from unittest.mock import AsyncMock, patch

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock successful orchestrator execution
            from tests.factories.agent_context_factories import AgentContextFactory

            mock_context = AgentContextFactory.basic(
                query="Integration test workflow",
                agent_outputs={
                    "refiner": "Refined query for integration test",
                    "critic": "Critical analysis for integration test",
                    "synthesis": "Integration test demonstrates successful persistence",
                },
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()

            # Create test workflow request
            request = APIModelPatterns.generate_valid_data(
                query="Integration test workflow persistence",
                agents=["refiner", "critic", "synthesis"],
                correlation_id=f"integration-test-{uuid.uuid4().hex[:8]}",
                execution_config={"timeout": 45, "test_mode": True},
            )

            # Mock event functions to prevent actual event emission
            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                # Execute workflow with real persistence
                response = await api.execute_workflow(request)

                # Verify workflow completed successfully
                assert response.status == "completed"
                assert response.correlation_id == request.correlation_id
                assert response.workflow_id is not None
                assert "refiner" in response.agent_outputs
                assert "critic" in response.agent_outputs
                assert "synthesis" in response.agent_outputs

                # Verify database persistence occurred
                questions = await repos.questions.get_recent_questions(limit=1)
                assert len(questions) >= 1

                # Find our question by correlation_id
                persisted_question = None
                for q in questions:
                    if q.correlation_id == request.correlation_id:
                        persisted_question = q
                        break

                assert persisted_question is not None, (
                    "Workflow was not persisted to database"
                )

                # Verify database record structure
                assert (
                    persisted_question.query == "Integration test workflow persistence"
                )
                assert persisted_question.correlation_id == request.correlation_id
                assert persisted_question.execution_id == response.workflow_id
                assert persisted_question.nodes_executed == [
                    "refiner",
                    "critic",
                    "synthesis",
                ]

                # Verify execution metadata structure
                metadata = persisted_question.execution_metadata
                assert metadata is not None
                assert metadata["workflow_id"] == response.workflow_id
                assert (
                    metadata["execution_time_seconds"]
                    == response.execution_time_seconds
                )
                assert metadata["agent_outputs"] == response.agent_outputs
                assert metadata["agents_requested"] == [
                    "refiner",
                    "critic",
                    "synthesis",
                ]
                assert metadata["export_md"] is False
                assert metadata["execution_config"] == {
                    "timeout": 45,
                    "test_mode": True,
                }
                assert metadata["api_version"] == "1.0.0"
                assert metadata["orchestrator_type"] == "langgraph-real"

                # Verify timestamp is recent
                time_diff = datetime.now(timezone.utc) - persisted_question.created_at
                assert time_diff.total_seconds() < 60  # Created within last minute

    finally:
        # Restore original session factory
        api._session_factory = original_session_factory
        await api.shutdown()


@pytest.mark.asyncio
async def test_failed_workflow_persistence_end_to_end(
    integration_db_session: Any,
) -> None:
    """Test failed workflow execution with database persistence."""
    session = integration_db_session
    repos = RepositoryFactory(session)

    api = LangGraphOrchestrationAPI()

    # Use centralized async session wrapper
    original_session_factory = api._session_factory
    api._session_factory = create_mock_session_factory(session)

    try:
        from unittest.mock import AsyncMock, patch

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Configure orchestrator to fail
            mock_orchestrator.run.side_effect = Exception(
                "Integration test orchestrator failure"
            )

            await api.initialize()

            request = APIModelPatterns.generate_valid_data(
                query="Failed integration test workflow",
                agents=["refiner"],
                correlation_id=f"integration-failed-{uuid.uuid4().hex[:8]}",
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                # Execute failed workflow
                response = await api.execute_workflow(request)

                # Verify workflow failed
                assert response.status == "failed"
                assert response.error_message is not None
                assert "Integration test orchestrator failure" in response.error_message
                assert response.correlation_id == request.correlation_id

                # Verify failed workflow was persisted
                questions = await repos.questions.get_recent_questions(limit=5)

                persisted_question = None
                for q in questions:
                    if q.correlation_id == request.correlation_id:
                        persisted_question = q
                        break

                assert persisted_question is not None, (
                    "Failed workflow was not persisted to database"
                )

                # Verify failed workflow metadata
                metadata = persisted_question.execution_metadata
                assert metadata is not None
                assert metadata["status"] == "failed"
                assert (
                    metadata["error_message"] == "Integration test orchestrator failure"
                )
                assert metadata["execution_time_seconds"] > 0
                assert metadata["agent_outputs"] == {}  # No outputs for failed workflow

    finally:
        api._session_factory = original_session_factory
        await api.shutdown()


@pytest.mark.asyncio
async def test_workflow_history_database_integration(
    integration_db_session: Any,
) -> None:
    """Test workflow history retrieval from database after persistence."""
    session = integration_db_session
    repos = RepositoryFactory(session)

    api = LangGraphOrchestrationAPI()

    # Use centralized async session wrapper
    original_session_factory = api._session_factory
    api._session_factory = create_mock_session_factory(session)

    try:
        from unittest.mock import AsyncMock, patch

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from tests.factories.agent_context_factories import AgentContextFactory

            await api.initialize()

            # Execute multiple workflows to build history
            workflow_requests = [
                APIModelPatterns.generate_valid_data(
                    query=f"History test workflow {i}",
                    agents=["refiner", "synthesis"],
                    correlation_id=f"history-test-{i}-{uuid.uuid4().hex[:4]}",
                )
                for i in range(3)
            ]

            executed_workflows = []

            for i, request in enumerate(workflow_requests):
                # Mock different outputs for each workflow
                mock_context = AgentContextFactory.basic(
                    query=request.query,
                    agent_outputs={
                        "refiner": f"Refined output {i}",
                        "synthesis": f"Synthesis output {i}",
                    },
                )
                mock_orchestrator.run.return_value = mock_context

                with (
                    patch("cognivault.events.emit_workflow_started"),
                    patch("cognivault.events.emit_workflow_completed"),
                ):
                    response = await api.execute_workflow(request)
                    executed_workflows.append((request, response))

            # Test history retrieval
            history = await api.get_workflow_history_from_database(limit=5, offset=0)

            # Verify we got history records
            assert len(history) >= 3, "Should have at least 3 workflow history records"

            # Verify our executed workflows are in the history
            history_workflow_ids = {item["workflow_id"] for item in history}
            for request, response in executed_workflows:
                assert response.workflow_id in history_workflow_ids, (
                    f"Workflow {response.workflow_id} not found in history"
                )

            # Verify history item structure
            for history_item in history[:3]:  # Check first 3 items
                assert "workflow_id" in history_item
                assert "status" in history_item
                assert "query" in history_item
                assert "start_time" in history_item
                assert "execution_time" in history_item

                # Verify data types
                assert isinstance(history_item["workflow_id"], str)
                assert isinstance(history_item["status"], str)
                assert isinstance(history_item["query"], str)
                assert isinstance(history_item["start_time"], float)
                assert isinstance(history_item["execution_time"], float)

                # Verify reasonable values
                assert len(history_item["workflow_id"]) > 0
                assert history_item["status"] in ["completed", "failed"]
                assert len(history_item["query"]) > 0
                assert history_item["start_time"] > 0
                assert history_item["execution_time"] >= 0

            # Test pagination
            history_page1 = await api.get_workflow_history_from_database(
                limit=2, offset=0
            )
            history_page2 = await api.get_workflow_history_from_database(
                limit=2, offset=2
            )

            assert len(history_page1) <= 2
            assert len(history_page2) <= 2

            # Verify no overlap between pages (assuming enough records)
            if len(history_page1) == 2 and len(history_page2) > 0:
                page1_ids = {item["workflow_id"] for item in history_page1}
                page2_ids = {item["workflow_id"] for item in history_page2}
                assert len(page1_ids.intersection(page2_ids)) == 0, (
                    "Pages should not overlap"
                )

    finally:
        api._session_factory = original_session_factory
        await api.shutdown()


@pytest.mark.asyncio
async def test_database_error_isolation_integration(
    integration_db_session: Any,
) -> None:
    """Test that database persistence errors don't break workflow execution."""
    api = LangGraphOrchestrationAPI()

    # Mock session factory to fail
    async def failing_session_factory() -> None:
        raise Exception("Database connection failed")

    original_session_factory = api._session_factory
    api._session_factory = failing_session_factory  # type: ignore[assignment]

    try:
        from unittest.mock import AsyncMock, patch

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from tests.factories.agent_context_factories import AgentContextFactory

            mock_context = AgentContextFactory.basic(
                query="Database error test workflow",
                agent_outputs={"refiner": "Test output despite database error"},
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()

            request = APIModelPatterns.generate_valid_data(
                query="Database error test workflow",
                agents=["refiner"],
                correlation_id=f"db-error-test-{uuid.uuid4().hex[:8]}",
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                # Execute workflow - should succeed despite database error
                response = await api.execute_workflow(request)

                # Verify workflow still completed successfully
                assert response.status == "completed"
                assert response.error_message is None
                assert response.correlation_id == request.correlation_id
                assert "refiner" in response.agent_outputs

                # The workflow should succeed even though persistence failed

    finally:
        api._session_factory = original_session_factory
        await api.shutdown()


@pytest.mark.asyncio
async def test_workflow_persistence_with_export_md_flag(
    integration_db_session: Any,
) -> None:
    """Test workflow persistence includes export_md flag in metadata."""
    session = integration_db_session
    repos = RepositoryFactory(session)

    api = LangGraphOrchestrationAPI()

    # Use centralized async session wrapper
    original_session_factory = api._session_factory
    api._session_factory = create_mock_session_factory(session)

    try:
        from unittest.mock import AsyncMock, patch

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from tests.factories.agent_context_factories import AgentContextFactory

            mock_context = AgentContextFactory.basic(
                query="Export MD test workflow",
                agent_outputs={"synthesis": "Test output for MD export"},
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()

            # Test with export_md enabled
            request = APIModelPatterns.generate_valid_data(
                query="Export MD test workflow",
                agents=["synthesis"],
                correlation_id=f"export-md-test-{uuid.uuid4().hex[:8]}",
                export_md=True,
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
                # Mock markdown export components to prevent actual file operations
                patch("cognivault.store.wiki_adapter.MarkdownExporter"),
                patch("cognivault.store.topic_manager.TopicManager"),
                patch("cognivault.llm.openai.OpenAIChatLLM"),
                patch("cognivault.config.openai_config.OpenAIConfig.load"),
            ):
                response = await api.execute_workflow(request)

                # Verify workflow completed
                assert response.status == "completed"

                # Verify persistence includes export_md flag
                questions = await repos.questions.get_recent_questions(limit=5)

                persisted_question = None
                for q in questions:
                    if q.correlation_id == request.correlation_id:
                        persisted_question = q
                        break

                assert persisted_question is not None

                metadata = persisted_question.execution_metadata
                assert metadata is not None
                assert metadata["export_md"] is True

    finally:
        api._session_factory = original_session_factory
        await api.shutdown()


@pytest.mark.asyncio
async def test_complex_execution_metadata_structure(
    integration_db_session: Any,
) -> None:
    """Test that complex execution metadata is properly structured and stored."""
    session = integration_db_session
    repos = RepositoryFactory(session)

    api = LangGraphOrchestrationAPI()

    # Use centralized async session wrapper
    original_session_factory = api._session_factory
    api._session_factory = create_mock_session_factory(session)

    try:
        from unittest.mock import AsyncMock, patch

        with patch(
            "cognivault.api.orchestration_api.LangGraphOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            from tests.factories.agent_context_factories import AgentContextFactory

            # Create complex agent outputs
            complex_agent_outputs = {
                "refiner": "Refined query with detailed analysis including special characters: @#$%^&*()",
                "critic": 'Critical evaluation with JSON-like structure: {"key": "value", "nested": {"data": [1, 2, 3]}}',
                "historian": "Historical context with unicode: 测试数据 αβγ δεζ",
                "synthesis": "Comprehensive synthesis spanning multiple lines\nwith newlines and\ttabs\nand various formatting",
            }

            mock_context = AgentContextFactory.basic(
                query="Complex metadata test", agent_outputs=complex_agent_outputs
            )
            mock_orchestrator.run.return_value = mock_context

            await api.initialize()

            # Create complex execution config
            complex_config = {
                "timeout": 60,
                "retries": 3,
                "advanced_settings": {
                    "model_parameters": {"temperature": 0.7, "max_tokens": 2048},
                    "feature_flags": ["experimental_mode", "verbose_logging"],
                },
                "metadata": {
                    "user_id": "test-user-123",
                    "session_id": "session-456",
                    "environment": "integration-test",
                },
            }

            request = APIModelPatterns.generate_valid_data(
                query="Complex metadata structure test workflow with unicode 测试 and special chars @#$%",
                agents=["refiner", "critic", "historian", "synthesis"],
                correlation_id=f"complex-metadata-{uuid.uuid4().hex[:8]}",
                execution_config=complex_config,
            )

            with (
                patch("cognivault.events.emit_workflow_started"),
                patch("cognivault.events.emit_workflow_completed"),
            ):
                response = await api.execute_workflow(request)

                assert response.status == "completed"

                # Verify complex metadata persistence
                questions = await repos.questions.get_recent_questions(limit=5)

                persisted_question = None
                for q in questions:
                    if q.correlation_id == request.correlation_id:
                        persisted_question = q
                        break

                assert persisted_question is not None

                # Verify query with special characters persisted correctly
                assert persisted_question.query == request.query

                # Verify nodes_executed includes all agents
                assert persisted_question.nodes_executed is not None
                assert set(persisted_question.nodes_executed) == {
                    "refiner",
                    "critic",
                    "historian",
                    "synthesis",
                }

                # Verify complex execution metadata structure
                metadata = persisted_question.execution_metadata
                assert metadata is not None

                # Verify all required fields exist
                required_fields = [
                    "workflow_id",
                    "execution_time_seconds",
                    "agent_outputs",
                    "agents_requested",
                    "export_md",
                    "execution_config",
                    "api_version",
                    "orchestrator_type",
                ]
                for field in required_fields:
                    assert field in metadata, (
                        f"Required field {field} missing from metadata"
                    )

                # Verify complex agent outputs preserved
                stored_outputs = metadata["agent_outputs"]
                assert stored_outputs == complex_agent_outputs

                # Verify complex execution config preserved
                stored_config = metadata["execution_config"]
                assert stored_config == complex_config
                assert (
                    stored_config["advanced_settings"]["model_parameters"][
                        "temperature"
                    ]
                    == 0.7
                )
                assert (
                    "experimental_mode"
                    in stored_config["advanced_settings"]["feature_flags"]
                )

                # Verify data types
                assert isinstance(metadata["execution_time_seconds"], (int, float))
                assert isinstance(metadata["agent_outputs"], dict)
                assert isinstance(metadata["agents_requested"], list)
                assert isinstance(metadata["export_md"], bool)
                assert isinstance(metadata["execution_config"], dict)
                assert isinstance(metadata["api_version"], str)
                assert isinstance(metadata["orchestrator_type"], str)

    finally:
        api._session_factory = original_session_factory
        await api.shutdown()
