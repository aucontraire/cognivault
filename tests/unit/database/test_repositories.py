"""
Test cases for database repository pattern implementation.
"""

import pytest
import uuid

from typing import Any
from unittest.mock import AsyncMock

from cognivault.database import RepositoryFactory
from cognivault.database.models import Topic, Question, WikiEntry


# Test database configuration now handled by centralized config
from tests.utils.test_database_config import get_test_database_url


# Repository factory fixture is now provided by conftest.py


class TestTopicRepository:
    """Test cases for TopicRepository."""

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_create_topic(self, repository_factory: RepositoryFactory) -> None:
        """Test creating a topic."""
        unique_name = f"Test Topic {uuid.uuid4().hex[:8]}"

        topic = await repository_factory.topics.create_topic(
            name=unique_name, description="Test topic description"
        )

        assert topic.id is not None
        assert topic.name == unique_name
        assert topic.description == "Test topic description"
        assert topic.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_topic_by_id(self, repository_factory: RepositoryFactory) -> None:
        """Test retrieving topic by ID."""
        # Create topic
        unique_name = f"Test Topic ID {uuid.uuid4().hex[:8]}"
        created_topic = await repository_factory.topics.create_topic(
            name=unique_name, description="Test retrieval"
        )

        # Retrieve topic
        retrieved_topic = await repository_factory.topics.get_by_id(created_topic.id)

        assert retrieved_topic is not None
        assert retrieved_topic.id == created_topic.id
        assert retrieved_topic.name == unique_name

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_topic_by_name(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test retrieving topic by name."""
        unique_name = f"Test Topic Name {uuid.uuid4().hex[:8]}"

        # Create topic
        await repository_factory.topics.create_topic(
            name=unique_name, description="Test name retrieval"
        )

        # Retrieve by name
        found_topic = await repository_factory.topics.get_by_name(unique_name)

        assert found_topic is not None
        assert found_topic.name == unique_name

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_search_topics_by_name(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test searching topics by name pattern."""
        search_term = f"SearchTest{uuid.uuid4().hex[:6]}"

        # Create topics with searchable names
        await repository_factory.topics.create_topic(
            name=f"{search_term} Topic 1", description="First search test topic"
        )
        await repository_factory.topics.create_topic(
            name=f"{search_term} Topic 2", description="Second search test topic"
        )

        # Search topics
        results = await repository_factory.topics.search_by_name(search_term)

        assert len(results) >= 2
        assert all(search_term in topic.name for topic in results)

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_hierarchical_topics(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test hierarchical topic relationships."""
        # Create parent topic
        parent = await repository_factory.topics.create_topic(
            name=f"Parent Topic {uuid.uuid4().hex[:8]}",
            description="Parent topic for hierarchy test",
        )

        # Create child topic
        child = await repository_factory.topics.create_topic(
            name=f"Child Topic {uuid.uuid4().hex[:8]}",
            description="Child topic for hierarchy test",
            parent_topic_id=parent.id,
        )

        # Test parent-child relationship
        assert child.parent_topic_id == parent.id

        # Get children of parent
        children = await repository_factory.topics.get_children(parent.id)
        assert len(children) >= 1
        assert any(c.id == child.id for c in children)

        # Get topic with hierarchy loaded
        topic_with_hierarchy = await repository_factory.topics.get_with_hierarchy(
            child.id
        )
        assert topic_with_hierarchy is not None
        assert topic_with_hierarchy.parent_topic_id == parent.id


class TestQuestionRepository:
    """Test cases for QuestionRepository."""

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_create_question(self, repository_factory: RepositoryFactory) -> None:
        """Test creating a question."""
        # Create topic first
        topic = await repository_factory.topics.create_topic(
            name=f"Question Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for question testing",
        )

        correlation_id = f"test-correlation-{uuid.uuid4().hex[:8]}"

        question = await repository_factory.questions.create_question(
            query="What is the repository pattern?",
            topic_id=topic.id,
            correlation_id=correlation_id,
            nodes_executed=["refiner", "critic"],
            execution_metadata={"test": True, "execution_time": 1.5},
        )

        assert question.id is not None
        assert question.query == "What is the repository pattern?"
        assert question.topic_id == topic.id
        assert question.correlation_id == correlation_id
        assert question.nodes_executed == ["refiner", "critic"]
        assert question.execution_metadata is not None
        assert question.execution_metadata["test"] is True

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_question_by_correlation_id(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test retrieving question by correlation ID."""
        # Create topic and question
        topic = await repository_factory.topics.create_topic(
            name=f"Correlation Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for correlation testing",
        )

        correlation_id = f"test-correlation-{uuid.uuid4().hex[:8]}"

        created_question = await repository_factory.questions.create_question(
            query="Test correlation query",
            topic_id=topic.id,
            correlation_id=correlation_id,
        )

        # Retrieve by correlation ID
        found_question = await repository_factory.questions.get_by_correlation_id(
            correlation_id
        )

        assert found_question is not None
        assert found_question.id == created_question.id
        assert found_question.correlation_id == correlation_id

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_questions_by_topic(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test retrieving questions for a specific topic."""
        # Create topic
        topic = await repository_factory.topics.create_topic(
            name=f"Topic Questions Test {uuid.uuid4().hex[:8]}",
            description="Topic for testing question retrieval",
        )

        # Create multiple questions for the topic
        for i in range(3):
            await repository_factory.questions.create_question(
                query=f"Test question {i + 1}",
                topic_id=topic.id,
                correlation_id=f"test-{i}-{uuid.uuid4().hex[:8]}",
            )

        # Retrieve questions for topic
        topic_questions = await repository_factory.questions.get_by_topic(topic.id)

        assert len(topic_questions) >= 3
        assert all(q.topic_id == topic.id for q in topic_questions)

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_search_questions_by_text(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test searching questions by query text."""
        search_term = f"SearchQuery{uuid.uuid4().hex[:6]}"

        # Create topic
        topic = await repository_factory.topics.create_topic(
            name=f"Search Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for search testing",
        )

        # Create question with searchable text
        await repository_factory.questions.create_question(
            query=f"This is a {search_term} for testing search functionality",
            topic_id=topic.id,
            correlation_id=f"search-test-{uuid.uuid4().hex[:8]}",
        )

        # Search questions
        results = await repository_factory.questions.search_by_query_text(search_term)

        assert len(results) >= 1
        assert any(search_term in (q.query or "") for q in results)

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_execution_statistics(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test getting execution statistics."""
        # Create topic and questions with execution metadata
        topic = await repository_factory.topics.create_topic(
            name=f"Stats Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for statistics testing",
        )

        for i in range(2):
            await repository_factory.questions.create_question(
                query=f"Stats test question {i + 1}",
                topic_id=topic.id,
                correlation_id=f"stats-test-{i}-{uuid.uuid4().hex[:8]}",
                nodes_executed=["refiner", "critic"],
                execution_metadata={"execution_time": float(i + 1), "test_run": True},
            )

        # Get statistics
        stats = await repository_factory.questions.get_execution_statistics()

        assert stats["total_questions"] >= 2
        assert stats["questions_with_metadata"] >= 2
        assert "refiner" in stats["node_usage_counts"]
        assert "critic" in stats["node_usage_counts"]

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_questions_with_nodes(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test getting questions by node execution patterns."""
        # Create topic
        topic = await repository_factory.topics.create_topic(
            name=f"Node Query Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for node query testing",
        )

        # Create questions with different node patterns
        question1 = await repository_factory.questions.create_question(
            query="Question with refiner and critic",
            topic_id=topic.id,
            correlation_id=f"node-test-1-{uuid.uuid4().hex[:8]}",
            nodes_executed=["refiner", "critic"],
        )

        question2 = await repository_factory.questions.create_question(
            query="Question with all four nodes",
            topic_id=topic.id,
            correlation_id=f"node-test-2-{uuid.uuid4().hex[:8]}",
            nodes_executed=["refiner", "critic", "historian", "synthesis"],
        )

        question3 = await repository_factory.questions.create_question(
            query="Question with historian only",
            topic_id=topic.id,
            correlation_id=f"node-test-3-{uuid.uuid4().hex[:8]}",
            nodes_executed=["historian"],
        )

        # Test ANY match (overlap - questions that have at least one of the specified nodes)
        any_results = await repository_factory.questions.get_questions_with_nodes(
            node_names=["refiner", "synthesis"], match_all=False
        )

        # Should include question1 (has refiner) and question2 (has both)
        any_ids = [q.id for q in any_results]
        assert question1.id in any_ids
        assert question2.id in any_ids
        assert question3.id not in any_ids  # historian only, no refiner or synthesis

        # Test ALL match (contains - questions that have all specified nodes)
        all_results = await repository_factory.questions.get_questions_with_nodes(
            node_names=["refiner", "critic"], match_all=True
        )

        # Should include question1 and question2 (both have refiner AND critic)
        all_ids = [q.id for q in all_results]
        assert question1.id in all_ids
        assert question2.id in all_ids
        assert question3.id not in all_ids  # historian only, missing both


class TestWikiRepository:
    """Test cases for WikiRepository."""

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_create_wiki_entry(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test creating a wiki entry."""
        # Create topic and question
        topic = await repository_factory.topics.create_topic(
            name=f"Wiki Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for wiki testing",
        )

        question = await repository_factory.questions.create_question(
            query="What is a wiki entry?",
            topic_id=topic.id,
            correlation_id=f"wiki-test-{uuid.uuid4().hex[:8]}",
        )

        # Create wiki entry
        wiki_entry = await repository_factory.wiki.create_wiki_entry(
            topic_id=topic.id,
            question_id=question.id,
            content="A wiki entry is a structured knowledge article that synthesizes information from multiple sources.",
        )

        assert wiki_entry.id is not None
        assert wiki_entry.topic_id == topic.id
        assert wiki_entry.question_id == question.id
        assert wiki_entry.version == 1
        assert wiki_entry.content is not None
        assert "wiki entry" in wiki_entry.content

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_latest_wiki_for_topic(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test retrieving latest wiki entry for a topic."""
        # Create topic
        topic = await repository_factory.topics.create_topic(
            name=f"Latest Wiki Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for latest wiki testing",
        )

        # Create multiple versions
        wiki_v1 = await repository_factory.wiki.create_wiki_entry(
            topic_id=topic.id, content="Version 1 content", version=1
        )

        wiki_v2 = await repository_factory.wiki.create_wiki_entry(
            topic_id=topic.id, content="Version 2 content", version=2
        )

        # Get latest version
        latest_wiki = await repository_factory.wiki.get_latest_for_topic(topic.id)

        assert latest_wiki is not None
        assert latest_wiki.version == 2
        assert latest_wiki.content == "Version 2 content"

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_all_versions_for_topic(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test retrieving all versions of wiki entries for a topic."""
        # Create topic
        topic = await repository_factory.topics.create_topic(
            name=f"All Versions Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for version testing",
        )

        # Create multiple versions
        for version in range(1, 4):
            await repository_factory.wiki.create_wiki_entry(
                topic_id=topic.id, content=f"Version {version} content", version=version
            )

        # Get all versions
        all_versions = await repository_factory.wiki.get_all_versions_for_topic(
            topic.id
        )

        assert len(all_versions) >= 3
        # Should be ordered by version descending
        assert all_versions[0].version >= all_versions[1].version

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_wiki_with_relationships(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test loading wiki entry with topic and question relationships."""
        # Create topic and question
        topic = await repository_factory.topics.create_topic(
            name=f"Relationship Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for relationship testing",
        )

        question = await repository_factory.questions.create_question(
            query="Test relationship question",
            topic_id=topic.id,
            correlation_id=f"relationship-test-{uuid.uuid4().hex[:8]}",
        )

        # Create wiki entry
        wiki_entry = await repository_factory.wiki.create_wiki_entry(
            topic_id=topic.id,
            question_id=question.id,
            content="Wiki entry with relationships",
        )

        # Get with relationships loaded
        wiki_with_relationships = await repository_factory.wiki.get_with_relationships(
            wiki_entry.id
        )

        assert wiki_with_relationships is not None
        assert wiki_with_relationships.topic is not None
        assert wiki_with_relationships.topic.id == topic.id
        assert wiki_with_relationships.source_question is not None
        assert wiki_with_relationships.source_question.id == question.id


class TestRepositoryFactory:
    """Test cases for RepositoryFactory."""

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_repository_factory_initialization(self, db_session: Any) -> None:
        """Test that repository factory initializes correctly."""
        factory = RepositoryFactory(db_session)

        assert factory.topics is not None
        assert factory.questions is not None
        assert factory.wiki is not None
        assert factory.api_keys is not None

        # Test that repositories share the same session
        assert factory.topics.session is db_session
        assert factory.questions.session is db_session
        assert factory.wiki.session is db_session
        assert factory.api_keys.session is db_session

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_transaction_consistency(self, db_session: Any) -> None:
        """Test that all repositories share transaction consistency."""
        factory = RepositoryFactory(db_session)

        # Create topic using topics repository
        topic = await factory.topics.create_topic(
            name=f"Transaction Test Topic {uuid.uuid4().hex[:8]}",
            description="Testing transaction consistency",
        )

        # Create question using questions repository in same transaction
        question = await factory.questions.create_question(
            query="Transaction test query",
            topic_id=topic.id,
            correlation_id=f"transaction-test-{uuid.uuid4().hex[:8]}",
        )

        # Both should be committed and retrievable
        retrieved_topic = await factory.topics.get_by_id(topic.id)
        retrieved_question = await factory.questions.get_by_id(question.id)

        assert retrieved_topic is not None
        assert retrieved_question is not None
        assert retrieved_question.topic_id == retrieved_topic.id


class TestRepositoryErrorHandling:
    """Test error handling in repositories."""

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_nonexistent_topic(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test retrieving non-existent topic."""
        fake_id = uuid.uuid4()
        topic = await repository_factory.topics.get_by_id(fake_id)
        assert topic is None

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_get_nonexistent_question(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test retrieving non-existent question."""
        fake_id = uuid.uuid4()
        question = await repository_factory.questions.get_by_id(fake_id)
        assert question is None

    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_duplicate_correlation_id_handling(
        self, repository_factory: RepositoryFactory
    ) -> None:
        """Test handling duplicate correlation IDs."""
        # Create topic
        topic = await repository_factory.topics.create_topic(
            name=f"Duplicate Test Topic {uuid.uuid4().hex[:8]}",
            description="Topic for duplicate testing",
        )

        correlation_id = f"duplicate-test-{uuid.uuid4().hex[:8]}"

        # Create first question
        await repository_factory.questions.create_question(
            query="First question", topic_id=topic.id, correlation_id=correlation_id
        )

        # Attempt to create second question with same correlation_id should raise error
        with pytest.raises(
            Exception
        ):  # Should be IntegrityError due to unique constraint
            await repository_factory.questions.create_question(
                query="Second question",
                topic_id=topic.id,
                correlation_id=correlation_id,
            )
