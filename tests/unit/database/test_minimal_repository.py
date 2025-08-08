"""
Minimal test to debug repository creation issues.
"""

import pytest
import uuid

from cognivault.database import RepositoryFactory


@pytest.mark.asyncio
@pytest.mark.database
async def test_minimal_repository(repository_factory: RepositoryFactory) -> None:
    """Minimal test to create a topic using proper fixtures."""
    print("✅ Repository factory fixture loaded")

    # Create topic (repository handles its own transaction)
    unique_name = f"Test Topic {uuid.uuid4().hex[:8]}"

    topic = await repository_factory.topics.create_topic(
        name=unique_name, description="Test topic description"
    )

    print(f"✅ Topic created: {topic.name}")

    assert topic.id is not None
    assert topic.name == unique_name
    assert topic.description == "Test topic description"

    print("✅ All assertions passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
