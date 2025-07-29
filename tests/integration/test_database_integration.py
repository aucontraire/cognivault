"""
Integration tests for database repository pattern with full workflow.
"""

import pytest
import uuid
import os
from typing import List

from cognivault.database import (
    get_database_session,
    init_database,
    RepositoryFactory,
    health_check,
)


# Test database configuration
TEST_DATABASE_URL = (
    "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5435/cognivault"
)


@pytest.fixture(scope="module", autouse=True)
def setup_integration_database():
    """Setup test database configuration."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL


@pytest.mark.asyncio
async def test_database_health_check():
    """Test database health check functionality."""
    await init_database()

    health_status = await health_check()

    assert health_status["status"] == "healthy"
    assert health_status["pgvector_available"] is True
    assert "response_time_ms" in health_status
    assert health_status["response_time_ms"] > 0


@pytest.mark.asyncio
async def test_complete_workflow_integration():
    """Test complete workflow: Topic -> Question -> Wiki with repository pattern."""
    await init_database()

    async with get_database_session() as session:
        repos = RepositoryFactory(session)

        # Step 1: Create hierarchical topics
        parent_topic = await repos.topics.create_topic(
            name=f"Integration Test Parent {uuid.uuid4().hex[:8]}",
            description="Parent topic for integration testing",
        )

        child_topic = await repos.topics.create_topic(
            name=f"Integration Test Child {uuid.uuid4().hex[:8]}",
            description="Child topic for integration testing",
            parent_topic_id=parent_topic.id,
        )

        # Step 2: Create questions for both topics
        parent_question = await repos.questions.create_question(
            query="What is the parent concept in this integration test?",
            topic_id=parent_topic.id,
            correlation_id=f"integration-parent-{uuid.uuid4().hex[:8]}",
            nodes_executed=["refiner", "critic", "historian", "synthesis"],
            execution_metadata={
                "execution_time": 25.3,
                "model_used": "gpt-4",
                "integration_test": True,
                "complexity": "high",
            },
        )

        child_question = await repos.questions.create_question(
            query="How does the child concept relate to the parent in this integration test?",
            topic_id=child_topic.id,
            correlation_id=f"integration-child-{uuid.uuid4().hex[:8]}",
            nodes_executed=["refiner", "historian", "synthesis"],
            execution_metadata={
                "execution_time": 18.7,
                "model_used": "gpt-4",
                "integration_test": True,
                "complexity": "medium",
            },
        )

        # Step 3: Create wiki entries for both topics
        parent_wiki = await repos.wiki.create_wiki_entry(
            topic_id=parent_topic.id,
            question_id=parent_question.id,
            content="The parent concept represents the foundational knowledge base that encompasses broader principles and serves as the root for more specific implementations. In this integration test, it demonstrates the hierarchical organization of knowledge and the ability to maintain relationships between abstract and concrete concepts.",
            sources=[parent_question.id],
            related_topics=[child_topic.id],
        )

        child_wiki = await repos.wiki.create_wiki_entry(
            topic_id=child_topic.id,
            question_id=child_question.id,
            content="The child concept builds upon the parent concept by providing specific implementations and detailed examples. It inherits the foundational principles while adding specialized knowledge that makes the abstract concepts practical and actionable.",
            sources=[child_question.id, parent_question.id],
            related_topics=[parent_topic.id],
        )

        # Step 4: Test complex queries and relationships

        # Test hierarchical topic queries
        children = await repos.topics.get_children(parent_topic.id)
        assert len(children) >= 1
        assert any(c.id == child_topic.id for c in children)

        # Test topic hierarchy loading
        child_with_hierarchy = await repos.topics.get_with_hierarchy(child_topic.id)
        assert child_with_hierarchy.parent_topic_id == parent_topic.id

        # Test question retrieval by topic
        parent_questions = await repos.questions.get_by_topic(parent_topic.id)
        child_questions = await repos.questions.get_by_topic(child_topic.id)

        assert len(parent_questions) >= 1
        assert len(child_questions) >= 1
        assert any(q.id == parent_question.id for q in parent_questions)
        assert any(q.id == child_question.id for q in child_questions)

        # Test execution statistics
        stats = await repos.questions.get_execution_statistics()
        assert stats["total_questions"] >= 2
        assert stats["questions_with_metadata"] >= 2
        assert "refiner" in stats["node_usage_counts"]
        assert "synthesis" in stats["node_usage_counts"]

        # Test wiki relationships
        parent_wiki_with_relations = await repos.wiki.get_with_relationships(
            parent_wiki.id
        )
        child_wiki_with_relations = await repos.wiki.get_with_relationships(
            child_wiki.id
        )

        assert parent_wiki_with_relations.topic.id == parent_topic.id
        assert parent_wiki_with_relations.source_question.id == parent_question.id
        assert child_wiki_with_relations.topic.id == child_topic.id
        assert child_wiki_with_relations.source_question.id == child_question.id

        # Test search functionality
        search_results = await repos.questions.search_by_query_text("integration test")
        assert len(search_results) >= 2

        topic_search_results = await repos.topics.search_by_name("Integration Test")
        assert len(topic_search_results) >= 2

        wiki_search_results = await repos.wiki.search_content("concept")
        assert len(wiki_search_results) >= 2

        # Step 5: Test wiki versioning
        updated_parent_wiki = await repos.wiki.create_new_version(
            current_entry_id=parent_wiki.id,
            new_content="Updated parent concept: The foundational knowledge base has evolved to include additional frameworks and methodologies that enhance the understanding of hierarchical knowledge organization. This version incorporates lessons learned from practical implementations.",
            additional_sources=[child_question.id],
        )

        assert updated_parent_wiki is not None
        assert updated_parent_wiki.version == 2
        assert updated_parent_wiki.supersedes == parent_wiki.id

        # Test version retrieval
        latest_parent_wiki = await repos.wiki.get_latest_for_topic(parent_topic.id)
        assert latest_parent_wiki.version == 2
        assert latest_parent_wiki.id == updated_parent_wiki.id

        all_versions = await repos.wiki.get_all_versions_for_topic(parent_topic.id)
        assert len(all_versions) >= 2
        assert all_versions[0].version == 2  # Latest first
        assert all_versions[1].version == 1

        # Step 6: Test multi-topic relationships
        multi_topic_entries = await repos.wiki.get_multi_topic_entries()
        # Both wikis have related_topics, so should appear in results
        multi_topic_ids = [entry.id for entry in multi_topic_entries]
        assert parent_wiki.id in multi_topic_ids or child_wiki.id in multi_topic_ids


@pytest.mark.asyncio
async def test_repository_factory_session_consistency():
    """Test that repository factory maintains session consistency across operations."""
    await init_database()

    async with get_database_session() as session:
        repos = RepositoryFactory(session)

        # Create data using different repositories in same transaction
        topic = await repos.topics.create_topic(
            name=f"Session Test Topic {uuid.uuid4().hex[:8]}",
            description="Testing session consistency",
        )

        question = await repos.questions.create_question(
            query="Test session consistency question",
            topic_id=topic.id,
            correlation_id=f"session-test-{uuid.uuid4().hex[:8]}",
        )

        wiki = await repos.wiki.create_wiki_entry(
            topic_id=topic.id,
            question_id=question.id,
            content="Testing session consistency across repositories",
        )

        # All repositories should have access to the same transaction data
        assert repos.topics.session is session
        assert repos.questions.session is session
        assert repos.wiki.session is session

        # Verify all data is accessible within the same transaction
        retrieved_topic = await repos.topics.get_by_id(topic.id)
        retrieved_question = await repos.questions.get_by_id(question.id)
        retrieved_wiki = await repos.wiki.get_by_id(wiki.id)

        assert retrieved_topic is not None
        assert retrieved_question is not None
        assert retrieved_wiki is not None

        # Test relationship consistency
        assert retrieved_question.topic_id == retrieved_topic.id
        assert retrieved_wiki.topic_id == retrieved_topic.id
        assert retrieved_wiki.question_id == retrieved_question.id


@pytest.mark.asyncio
async def test_repository_error_handling_integration():
    """Test error handling across repository operations."""
    await init_database()

    async with get_database_session() as session:
        repos = RepositoryFactory(session)

        # Test cascade of operations with error handling
        try:
            # Create topic successfully
            topic = await repos.topics.create_topic(
                name=f"Error Test Topic {uuid.uuid4().hex[:8]}",
                description="Testing error handling",
            )

            # Test 1: Create question with valid topic should succeed
            valid_question = await repos.questions.create_question(
                query="Test error handling question with valid topic",
                topic_id=topic.id,
                correlation_id=f"error-test-valid-{uuid.uuid4().hex[:8]}",
            )

            assert valid_question is not None
            assert valid_question.topic_id == topic.id

            # Test 2: Try to create question with invalid topic_id (should fail due to FK constraint)
            fake_topic_id = uuid.uuid4()

            # This should raise a foreign key violation error
            try:
                await repos.questions.create_question(
                    query="Test error handling question with invalid topic",
                    topic_id=fake_topic_id,  # Non-existent topic
                    correlation_id=f"error-test-invalid-{uuid.uuid4().hex[:8]}",
                )
                # If we get here, the test should fail because FK constraint should prevent this
                pytest.fail("Expected foreign key violation did not occur")
            except Exception as fk_error:
                # This is expected - foreign key constraint should prevent the creation
                assert "ForeignKeyViolationError" in str(
                    type(fk_error)
                ) or "IntegrityError" in str(type(fk_error))

            # Test 3: Create question with null topic_id (should succeed)
            null_topic_question = await repos.questions.create_question(
                query="Test error handling question with null topic",
                topic_id=None,  # Null is allowed
                correlation_id=f"error-test-null-{uuid.uuid4().hex[:8]}",
            )

            assert null_topic_question is not None
            assert null_topic_question.topic_id is None

            # Try to get question with topic loaded (should handle null relationship)
            question_with_topic = await repos.questions.get_with_topic(
                null_topic_question.id
            )
            assert question_with_topic is not None
            assert question_with_topic.topic is None  # Topic is null

        except Exception as e:
            # If any unexpected error occurs, test should provide context
            pytest.fail(f"Unexpected error in integration test: {e}")


if __name__ == "__main__":
    # Can run this file directly for manual integration testing
    import asyncio

    async def main():
        print("🔧 Running Database Integration Tests...")

        try:
            await test_database_health_check()
            print("✅ Health check passed")

            await test_complete_workflow_integration()
            print("✅ Complete workflow integration passed")

            await test_repository_factory_session_consistency()
            print("✅ Session consistency test passed")

            await test_repository_error_handling_integration()
            print("✅ Error handling integration passed")

            print("\n🎉 All integration tests passed!")

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            raise

    asyncio.run(main())
