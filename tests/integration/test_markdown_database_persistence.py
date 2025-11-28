"""
Integration tests for markdown database persistence feature.

These tests verify that workflow results are persisted to the historian_documents
table after successful markdown export, with proper error handling and graceful
degradation when the database is unavailable.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from uuid import UUID

from cognivault.api.orchestration_api import LangGraphOrchestrationAPI
from cognivault.api.models import WorkflowRequest, WorkflowResponse
from cognivault.database.models import HistorianDocument
from cognivault.database import RepositoryFactory
from cognivault.database.session_factory import DatabaseSessionFactory
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
async def test_orchestration_api() -> AsyncGenerator[LangGraphOrchestrationAPI, None]:
    """Create and initialize orchestration API for testing."""
    api = LangGraphOrchestrationAPI()
    await api.initialize()

    yield api

    # Cleanup
    await api.shutdown()


@pytest.fixture
def sample_workflow_request() -> WorkflowRequest:
    """Create a sample workflow request for testing."""
    import uuid
    # Generate unique correlation ID for each test to avoid database conflicts
    correlation_id = f"test-correlation-{uuid.uuid4().hex[:8]}"
    return WorkflowRequest(
        query="What is machine learning and how does it work?",
        agents=["refiner", "critic", "historian", "synthesis"],
        correlation_id=correlation_id,
        export_md=True,
        execution_config={}
    )


@pytest.fixture
def mock_agent_outputs() -> Dict[str, str]:
    """Mock agent outputs for testing - strings as required by WorkflowResponse."""
    return {
        "refiner": "Refined question: Explain machine learning fundamentals and mechanisms. Confidence: 0.9",
        "critic": "Good question structure. Suggestions: Add specific ML algorithm examples. Severity: low",
        "historian": "Search results: ML is supervised learning, Neural networks. Found 2 relevant documents",
        "synthesis": "Machine learning is a subset of AI with key insights: Supervised learning, Deep learning. Confidence: 0.85"
    }


@pytest.mark.asyncio
class TestMarkdownDatabasePersistence:
    """Tests for database persistence of markdown exports."""

    async def test_successful_markdown_persistence_to_database(
        self,
        integration_db_session: AsyncSession,
        integration_repository_factory: RepositoryFactory,
        sample_workflow_request: WorkflowRequest,
        mock_agent_outputs: Dict[str, str]
    ) -> None:
        """Verify markdown is persisted to database after successful export."""
        # Arrange
        api = LangGraphOrchestrationAPI()
        await api.initialize()

        # Mock the orchestrator execution to avoid real LLM calls
        mock_context = Mock()
        mock_context.agent_outputs = mock_agent_outputs

        # Mock topic analysis
        mock_topic_analysis = Mock()
        mock_topic_analysis.suggested_topics = [
            Mock(topic="machine learning"),
            Mock(topic="artificial intelligence"),
            Mock(topic="neural networks")
        ]
        mock_topic_analysis.suggested_domain = "computer-science"

        try:
            with patch.object(api._orchestrator, 'run', new_callable=AsyncMock) as mock_run, \
                 patch('cognivault.store.topic_manager.TopicManager') as mock_topic_manager_class:

                mock_run.return_value = mock_context

                # Mock the TopicManager instance and its method
                mock_topic_manager_instance = AsyncMock()
                mock_topic_manager_instance.analyze_and_suggest_topics = AsyncMock(return_value=mock_topic_analysis)
                mock_topic_manager_class.return_value = mock_topic_manager_instance

                # Act
                result = await api.execute_workflow(sample_workflow_request)

                # Assert - Workflow completed successfully
                assert result is not None
                assert result.status == "completed"
                assert result.markdown_export is not None
                assert "file_path" in result.markdown_export
                assert "suggested_topics" in result.markdown_export
                assert "suggested_domain" in result.markdown_export

                # Verify markdown file was created
                md_path = Path(result.markdown_export["file_path"])
                assert md_path.exists(), "Markdown file should be created"

                # Verify database record exists
                doc_repo = integration_repository_factory.historian_documents

                # Search by workflow_id in metadata
                stmt = select(HistorianDocument).where(
                    HistorianDocument.document_metadata['workflow_id'].astext == result.workflow_id
                )
                db_result = await integration_db_session.execute(stmt)
                doc = db_result.scalar_one_or_none()

                assert doc is not None, "Document should be persisted to database"
                assert doc.title.startswith(sample_workflow_request.query[:50])
                assert doc.content is not None and len(doc.content) > 0, "Document should have content"
                assert doc.source_path is not None

                # Verify metadata structure
                metadata = doc.document_metadata
                assert metadata['workflow_id'] == result.workflow_id
                assert metadata['correlation_id'] == sample_workflow_request.correlation_id
                assert 'topics' in metadata
                assert isinstance(metadata['topics'], list)
                assert len(metadata['topics']) <= 5  # Max 5 topics
                assert metadata['domain'] == "computer-science"
                assert 'export_timestamp' in metadata
                assert 'agents_executed' in metadata
                assert isinstance(metadata['agents_executed'], list)
                assert set(metadata['agents_executed']) == set(mock_agent_outputs.keys())

                # Verify timestamp format (ISO 8601)
                try:
                    datetime.fromisoformat(metadata['export_timestamp'])
                except ValueError:
                    pytest.fail("export_timestamp should be in ISO 8601 format")

                # Cleanup markdown file
                if md_path.exists():
                    md_path.unlink()

        finally:
            await api.shutdown()

    async def test_database_failure_doesnt_break_workflow(
        self,
        sample_workflow_request: WorkflowRequest,
        mock_agent_outputs: Dict[str, str],
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify workflow completes even if database persistence fails."""
        # Arrange
        api = LangGraphOrchestrationAPI()
        await api.initialize()

        mock_context = Mock()
        mock_context.agent_outputs = mock_agent_outputs

        mock_topic_analysis = Mock()
        mock_topic_analysis.suggested_topics = [Mock(topic="test")]
        mock_topic_analysis.suggested_domain = "test-domain"

        md_file_created = None

        try:
            with patch.object(api._orchestrator, 'run', new_callable=AsyncMock) as mock_run, \
                 patch('cognivault.store.topic_manager.TopicManager') as mock_topic_manager_class, \
                 patch.object(api, '_get_or_create_db_session_factory',
                              new_callable=AsyncMock) as mock_db_factory:

                mock_run.return_value = mock_context

                # Mock the TopicManager instance
                mock_topic_manager_instance = AsyncMock()
                mock_topic_manager_instance.analyze_and_suggest_topics = AsyncMock(return_value=mock_topic_analysis)
                mock_topic_manager_class.return_value = mock_topic_manager_instance

                # Make database session factory raise an exception
                mock_db_factory.side_effect = Exception("Database connection failed")

                # Act
                result = await api.execute_workflow(sample_workflow_request)

                # Assert
                assert result is not None, "Workflow should complete despite DB failure"
                assert result.status == "completed"
                assert result.markdown_export is not None
                assert "file_path" in result.markdown_export

                # Verify markdown file was still created
                md_path = Path(result.markdown_export["file_path"])
                assert md_path.exists(), "Markdown file should still be created"
                md_file_created = md_path

                # Verify error was logged
                assert "Failed to persist markdown to database" in caplog.text or \
                       "Failed to initialize database session factory" in caplog.text

        finally:
            # Cleanup markdown file
            if md_file_created and md_file_created.exists():
                md_file_created.unlink()
            await api.shutdown()

    async def test_database_unavailable_graceful_degradation(
        self,
        sample_workflow_request: WorkflowRequest,
        mock_agent_outputs: Dict[str, str],
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify workflow completes when database is unavailable."""
        # Arrange
        api = LangGraphOrchestrationAPI()
        await api.initialize()

        mock_context = Mock()
        mock_context.agent_outputs = mock_agent_outputs

        mock_topic_analysis = Mock()
        mock_topic_analysis.suggested_topics = [Mock(topic="test")]
        mock_topic_analysis.suggested_domain = "test-domain"

        md_file_created = None

        try:
            with patch.object(api._orchestrator, 'run', new_callable=AsyncMock) as mock_run, \
                 patch('cognivault.store.topic_manager.TopicManager') as mock_topic_manager_class, \
                 patch.object(api, '_get_or_create_db_session_factory',
                              new_callable=AsyncMock) as mock_db_factory:

                mock_run.return_value = mock_context

                # Mock the TopicManager instance
                mock_topic_manager_instance = AsyncMock()
                mock_topic_manager_instance.analyze_and_suggest_topics = AsyncMock(return_value=mock_topic_analysis)
                mock_topic_manager_class.return_value = mock_topic_manager_instance

                # Database session factory returns None (unavailable)
                mock_db_factory.return_value = None

                # Act
                result = await api.execute_workflow(sample_workflow_request)

                # Assert
                assert result is not None, "Workflow should complete when DB unavailable"
                assert result.status == "completed"
                assert result.markdown_export is not None

                # Verify markdown file was created
                md_path = Path(result.markdown_export["file_path"])
                assert md_path.exists(), "Markdown file should be created"
                md_file_created = md_path

                # Verify warning was logged about database unavailability
                assert "Database not available" in caplog.text or \
                       "skipping markdown persistence" in caplog.text

        finally:
            # Cleanup markdown file
            if md_file_created and md_file_created.exists():
                md_file_created.unlink()
            await api.shutdown()

    async def test_deduplication_works_get_or_create(
        self,
        integration_db_session: AsyncSession,
        integration_repository_factory: RepositoryFactory,
        sample_workflow_request: WorkflowRequest,
        mock_agent_outputs: Dict[str, str]
    ) -> None:
        """Verify deduplication works when running same workflow twice."""
        # Arrange
        api = LangGraphOrchestrationAPI()
        await api.initialize()

        mock_context = Mock()
        mock_context.agent_outputs = mock_agent_outputs

        mock_topic_analysis = Mock()
        mock_topic_analysis.suggested_topics = [Mock(topic="test")]
        mock_topic_analysis.suggested_domain = "test-domain"

        md_files_created = []

        try:
            with patch.object(api._orchestrator, 'run', new_callable=AsyncMock) as mock_run, \
                 patch('cognivault.store.topic_manager.TopicManager') as mock_topic_manager_class:

                mock_run.return_value = mock_context

                # Mock the TopicManager instance
                mock_topic_manager_instance = AsyncMock()
                mock_topic_manager_instance.analyze_and_suggest_topics = AsyncMock(return_value=mock_topic_analysis)
                mock_topic_manager_class.return_value = mock_topic_manager_instance

                # Act - Run workflow twice with same content
                result1 = await api.execute_workflow(sample_workflow_request)
                if result1.markdown_export:
                    md_files_created.append(Path(result1.markdown_export["file_path"]))

                result2 = await api.execute_workflow(sample_workflow_request)
                if result2.markdown_export:
                    md_files_created.append(Path(result2.markdown_export["file_path"]))

                # Assert - Both workflows completed successfully
                assert result1.status == "completed"
                assert result2.status == "completed"

                # Verify only ONE database record exists (deduplication by content hash)
                # Both workflows should reference the same document due to content hash matching
                doc_repo = integration_repository_factory.historian_documents

                # Get all documents with matching content
                # Since content is identical, content_hash should be the same
                # and get_or_create should return existing document

                # Read the actual markdown content to verify deduplication
                if len(md_files_created) >= 2:
                    content1 = md_files_created[0].read_text(encoding="utf-8")
                    content2 = md_files_created[1].read_text(encoding="utf-8")

                    # If content is exactly the same, only 1 DB record should exist
                    if content1 == content2:
                        # Count documents with this content hash
                        import hashlib
                        content_hash = hashlib.sha256(content1.encode("utf-8")).hexdigest()

                        stmt = select(HistorianDocument).where(
                            HistorianDocument.content_hash == content_hash
                        )
                        db_result = await integration_db_session.execute(stmt)
                        docs = db_result.scalars().all()

                        assert len(docs) == 1, "Should only have ONE database record for identical content (deduplication)"

        finally:
            # Cleanup markdown files
            for md_file in md_files_created:
                if md_file.exists():
                    md_file.unlink()
            await api.shutdown()

    async def test_metadata_structure_complete(
        self,
        integration_db_session: AsyncSession,
        integration_repository_factory: RepositoryFactory,
        sample_workflow_request: WorkflowRequest,
        mock_agent_outputs: Dict[str, str]
    ) -> None:
        """Verify document_metadata contains all expected keys and correct types."""
        # Arrange
        api = LangGraphOrchestrationAPI()
        await api.initialize()

        mock_context = Mock()
        mock_context.agent_outputs = mock_agent_outputs

        mock_topic_analysis = Mock()
        mock_topic_analysis.suggested_topics = [
            Mock(topic="topic1"),
            Mock(topic="topic2"),
            Mock(topic="topic3"),
            Mock(topic="topic4"),
            Mock(topic="topic5"),
            Mock(topic="topic6")  # 6 topics to test max 5 limit
        ]
        mock_topic_analysis.suggested_domain = "test-domain"

        md_file_created = None

        try:
            with patch.object(api._orchestrator, 'run', new_callable=AsyncMock) as mock_run, \
                 patch('cognivault.store.topic_manager.TopicManager') as mock_topic_manager_class:

                mock_run.return_value = mock_context

                # Mock the TopicManager instance
                mock_topic_manager_instance = AsyncMock()
                mock_topic_manager_instance.analyze_and_suggest_topics = AsyncMock(return_value=mock_topic_analysis)
                mock_topic_manager_class.return_value = mock_topic_manager_instance

                # Act
                result = await api.execute_workflow(sample_workflow_request)

                # Assert
                assert result is not None
                assert result.status == "completed"

                if result.markdown_export:
                    md_file_created = Path(result.markdown_export["file_path"])

                # Retrieve document from database
                stmt = select(HistorianDocument).where(
                    HistorianDocument.document_metadata['workflow_id'].astext == result.workflow_id
                )
                db_result = await integration_db_session.execute(stmt)
                doc = db_result.scalar_one_or_none()

                assert doc is not None, "Document should exist in database"

                # Verify metadata structure
                metadata = doc.document_metadata

                # Required keys
                required_keys = [
                    'workflow_id',
                    'correlation_id',
                    'topics',
                    'domain',
                    'export_timestamp',
                    'agents_executed'
                ]

                for key in required_keys:
                    assert key in metadata, f"Metadata should contain '{key}'"

                # Verify types
                assert isinstance(metadata['workflow_id'], str), "workflow_id should be string"
                assert isinstance(metadata['correlation_id'], str), "correlation_id should be string"
                assert isinstance(metadata['topics'], list), "topics should be list"
                assert isinstance(metadata['domain'], str), "domain should be string"
                assert isinstance(metadata['export_timestamp'], str), "export_timestamp should be string"
                assert isinstance(metadata['agents_executed'], list), "agents_executed should be list"

                # Verify values
                assert metadata['workflow_id'] == result.workflow_id
                assert metadata['correlation_id'] == sample_workflow_request.correlation_id
                assert len(metadata['topics']) <= 5, "Should have max 5 topics"
                assert metadata['domain'] == "test-domain"

                # Verify ISO timestamp format
                try:
                    parsed_timestamp = datetime.fromisoformat(metadata['export_timestamp'])
                    # Verify it's recent (within last minute)
                    now = datetime.now(timezone.utc)
                    time_diff = (now - parsed_timestamp.replace(tzinfo=timezone.utc)).total_seconds()
                    assert abs(time_diff) < 60, "Timestamp should be recent"
                except ValueError:
                    pytest.fail("export_timestamp should be valid ISO 8601 format")

                # Verify agents_executed contains all agents from mock_agent_outputs
                assert set(metadata['agents_executed']) == set(mock_agent_outputs.keys())

        finally:
            # Cleanup markdown file
            if md_file_created and md_file_created.exists():
                md_file_created.unlink()
            await api.shutdown()

    async def test_no_markdown_export_when_export_md_false(
        self,
        integration_db_session: AsyncSession,
        sample_workflow_request: WorkflowRequest,
        mock_agent_outputs: Dict[str, str]
    ) -> None:
        """Verify no markdown or database persistence when export_md=False."""
        # Arrange
        api = LangGraphOrchestrationAPI()
        await api.initialize()

        # Disable markdown export
        sample_workflow_request.export_md = False

        mock_context = Mock()
        mock_context.agent_outputs = mock_agent_outputs

        try:
            with patch.object(api._orchestrator, 'run', new_callable=AsyncMock) as mock_run:
                mock_run.return_value = mock_context

                # Act
                result = await api.execute_workflow(sample_workflow_request)

                # Assert
                assert result is not None
                assert result.status == "completed"
                assert result.markdown_export is None, "Should not have markdown_export when export_md=False"

                # Verify no database record was created for markdown
                # (Note: Workflow execution record is still created, but not markdown document)
                stmt = select(HistorianDocument).where(
                    HistorianDocument.document_metadata['workflow_id'].astext == result.workflow_id
                )
                db_result = await integration_db_session.execute(stmt)
                doc = db_result.scalar_one_or_none()

                assert doc is None, "Should not create HistorianDocument when export_md=False"

        finally:
            await api.shutdown()

    async def test_markdown_export_failure_doesnt_break_workflow(
        self,
        sample_workflow_request: WorkflowRequest,
        mock_agent_outputs: Dict[str, str],
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify workflow completes even if markdown export itself fails."""
        # Arrange
        api = LangGraphOrchestrationAPI()
        await api.initialize()

        mock_context = Mock()
        mock_context.agent_outputs = mock_agent_outputs

        # Mock topic analysis to avoid LLM calls
        mock_topic_analysis = Mock()
        mock_topic_analysis.suggested_topics = [Mock(topic="test-topic")]
        mock_topic_analysis.suggested_domain = "test-domain"

        try:
            with patch.object(api._orchestrator, 'run', new_callable=AsyncMock) as mock_run, \
                 patch('cognivault.store.wiki_adapter.MarkdownExporter') as mock_exporter_class, \
                 patch('cognivault.store.topic_manager.TopicManager') as mock_topic_manager_class:

                mock_run.return_value = mock_context

                # Mock the TopicManager instance to avoid LLM calls
                mock_topic_manager_instance = AsyncMock()
                mock_topic_manager_instance.analyze_and_suggest_topics = AsyncMock(return_value=mock_topic_analysis)
                mock_topic_manager_class.return_value = mock_topic_manager_instance

                # Mock the MarkdownExporter instance
                mock_exporter_instance = Mock()
                mock_exporter_instance.export = Mock(side_effect=Exception("Markdown export failed"))
                mock_exporter_class.return_value = mock_exporter_instance

                # Act
                result = await api.execute_workflow(sample_workflow_request)

                # Assert
                assert result is not None, "Workflow should complete despite markdown export failure"
                assert result.status == "completed"

                # Verify error was captured in markdown_export
                assert result.markdown_export is not None
                assert "error" in result.markdown_export
                assert result.markdown_export["error"] == "Export failed"
                assert "Markdown export failed" in result.markdown_export["message"]

                # Verify warning was logged
                assert "Markdown export failed" in caplog.text

        finally:
            await api.shutdown()
