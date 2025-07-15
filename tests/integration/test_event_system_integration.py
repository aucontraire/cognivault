"""
Integration tests for the enhanced event system.

Tests the complete event pipeline from agent execution through
event emission to sink collection, verifying multi-axis classification
and correlation context propagation work end-to-end.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.events import (
    get_global_event_emitter,
)
from cognivault.events.sinks import InMemoryEventSink
from cognivault.correlation import trace


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str = "MockAgent", should_fail: bool = False):
        super().__init__(name)
        self.should_fail = should_fail

    async def run(self, context: AgentContext) -> AgentContext:
        if self.should_fail:
            raise ValueError(f"Simulated failure in {self.name}")

        # Simulate some processing
        await asyncio.sleep(0.01)
        context.add_agent_output(self.name, f"Output from {self.name}")
        return context


@pytest.fixture
def mock_registry():
    """Setup mock agent registry with proper cleanup."""
    registry = get_agent_registry()

    # Store original registered agents for cleanup
    original_agents = (
        set(registry._agents.keys()) if hasattr(registry, "_agents") else set()
    )

    # Clean up any existing test agents first
    test_agent_names = [
        "test_refiner",
        "test_critic",
        "test_agent",
        "failing_agent",
        "resilient_agent",
        "simple_test_agent",
    ]
    for agent_name in test_agent_names:
        if hasattr(registry, "_agents") and agent_name in registry._agents:
            del registry._agents[agent_name]

    # Register test agents with multi-axis metadata
    registry.register(
        name="test_refiner",
        agent_class=MockAgent,
        description="Test refiner agent",
        cognitive_speed="slow",
        cognitive_depth="deep",
        processing_pattern="atomic",
        primary_capability="intent_clarification",
        pipeline_role="entry",
        bounded_context="reflection",
    )

    registry.register(
        name="test_critic",
        agent_class=MockAgent,
        description="Test critic agent",
        cognitive_speed="slow",
        cognitive_depth="deep",
        processing_pattern="composite",
        primary_capability="critical_analysis",
        pipeline_role="intermediate",
        bounded_context="reflection",
    )

    yield registry

    # Cleanup: Remove test agents after test completion
    for agent_name in test_agent_names:
        if hasattr(registry, "_agents") and agent_name in registry._agents:
            del registry._agents[agent_name]


@pytest.fixture
def event_sink():
    """Setup in-memory event sink for testing."""
    sink = InMemoryEventSink(max_events=100)

    # Add to global emitter
    emitter = get_global_event_emitter()
    emitter.enable()  # Ensure events are enabled
    emitter.add_sink(sink)

    yield sink

    # Cleanup
    emitter.remove_sink(sink)
    sink.clear_events()


@pytest.mark.asyncio
async def test_agent_event_emission_with_correlation(mock_registry, event_sink):
    """Test that agent execution emits events with correct correlation context."""

    # Create agent context
    context = AgentContext(query="Test query for event emission")

    # Execute with explicit correlation context
    async with trace(
        correlation_id="test-correlation-123", workflow_id="test-workflow-456"
    ) as ctx:
        # Create and run mock agent
        agent = MockAgent("test_agent")
        result = await agent.run_with_retry(context)

        # Verify context has output
        assert "test_agent" in result.agent_outputs
        assert result.agent_outputs["test_agent"] == "Output from test_agent"

    # Wait for async event emission
    await asyncio.sleep(0.1)

    # Verify events were emitted
    events = event_sink.get_events()
    assert len(events) >= 2, f"Expected at least 2 events, got {len(events)}"

    # Find agent execution events
    started_events = event_sink.get_events(event_type="agent.execution.started")
    completed_events = event_sink.get_events(event_type="agent.execution.completed")

    assert len(started_events) == 1, (
        f"Expected 1 started event, got {len(started_events)}"
    )
    assert len(completed_events) == 1, (
        f"Expected 1 completed event, got {len(completed_events)}"
    )

    # Verify correlation context in events
    started_event = started_events[0]
    completed_event = completed_events[0]

    assert started_event.correlation_id == "test-correlation-123"
    assert started_event.workflow_id == "test-workflow-456"
    assert completed_event.correlation_id == "test-correlation-123"
    assert completed_event.workflow_id == "test-workflow-456"

    # Verify agent metadata
    assert hasattr(started_event, "agent_name")
    assert started_event.agent_name == "test_agent"  # From data field
    assert completed_event.success is True


@pytest.mark.asyncio
async def test_agent_failure_event_emission(mock_registry, event_sink):
    """Test that agent failures emit appropriate events."""

    context = AgentContext(query="Test query for failure")

    async with trace(correlation_id="test-failure-123") as ctx:
        # Create failing agent
        agent = MockAgent("failing_agent", should_fail=True)

        # Expect agent to raise exception
        with pytest.raises(Exception):
            await agent.run_with_retry(context)

    # Wait for async event emission
    await asyncio.sleep(0.1)

    # Verify failure events
    completed_events = event_sink.get_events(event_type="agent.execution.completed")
    assert len(completed_events) == 1

    failure_event = completed_events[0]
    assert failure_event.success is False
    assert failure_event.error_message is not None
    assert "Simulated failure" in failure_event.error_message


@pytest.mark.asyncio
async def test_multi_axis_agent_metadata_in_events(mock_registry, event_sink):
    """Test that events contain proper multi-axis agent metadata."""

    context = AgentContext(query="Test query for metadata")

    async with trace(correlation_id="test-metadata-123") as ctx:
        # Use registry to create agent with metadata
        agent = mock_registry.create_agent("test_refiner")
        await agent.run_with_retry(context)

    await asyncio.sleep(0.1)

    # Get started event
    started_events = event_sink.get_events(event_type="agent.execution.started")
    assert len(started_events) == 1

    event = started_events[0]

    # Verify agent metadata is present and correct
    assert event.agent_metadata is not None
    metadata = event.agent_metadata

    assert metadata.cognitive_speed == "slow"
    assert metadata.cognitive_depth == "deep"
    assert metadata.processing_pattern == "atomic"
    assert metadata.primary_capability == "intent_clarification"
    assert metadata.pipeline_role == "entry"
    assert metadata.bounded_context == "reflection"


@pytest.mark.asyncio
async def test_event_filtering_and_statistics(mock_registry, event_sink):
    """Test event filtering and statistics collection."""

    context = AgentContext(query="Test query for statistics")

    async with trace(correlation_id="test-stats-123") as ctx:
        # Run multiple agents
        for agent_name in ["test_refiner", "test_critic"]:
            agent = mock_registry.create_agent(agent_name)
            await agent.run_with_retry(context)

    await asyncio.sleep(0.1)

    # Test filtering by agent name
    refiner_events = event_sink.get_events(agent_name="test_refiner")
    critic_events = event_sink.get_events(agent_name="test_critic")

    assert len(refiner_events) >= 2  # Started + completed
    assert len(critic_events) >= 2  # Started + completed

    # Test filtering by event type
    started_events = event_sink.get_events(event_type="agent.execution.started")
    completed_events = event_sink.get_events(event_type="agent.execution.completed")

    assert len(started_events) == 2  # Two agents
    assert len(completed_events) == 2  # Two agents

    # Test statistics
    stats = event_sink.get_statistics()
    assert stats.total_events >= 4  # At least 4 events
    assert "agent.execution.started" in stats.events_by_type
    assert "agent.execution.completed" in stats.events_by_type
    assert "test_refiner" in stats.events_by_agent
    assert "test_critic" in stats.events_by_agent


@pytest.mark.asyncio
async def test_event_serialization(mock_registry, event_sink):
    """Test that events can be serialized and deserialized correctly."""

    context = AgentContext(query="Test query for serialization")

    async with trace(correlation_id="test-serialize-123") as ctx:
        agent = mock_registry.create_agent("test_refiner")
        await agent.run_with_retry(context)

    await asyncio.sleep(0.1)

    # Get an event
    events = event_sink.get_events()
    assert len(events) >= 1

    event = events[0]

    # Test to_dict serialization
    event_dict = event.to_dict()
    assert isinstance(event_dict, dict)
    assert "event_id" in event_dict
    assert "event_type" in event_dict
    assert "timestamp" in event_dict
    assert "workflow_id" in event_dict
    assert "correlation_id" in event_dict

    # Verify agent metadata serialization if present
    if event.agent_metadata:
        assert "agent_metadata" in event_dict
        metadata_dict = event_dict["agent_metadata"]
        assert isinstance(metadata_dict, dict)
        assert "cognitive_speed" in metadata_dict
        assert "primary_capability" in metadata_dict


@pytest.mark.asyncio
async def test_event_emission_resilience(mock_registry, event_sink):
    """Test that event emission failures don't break agent execution."""

    # Create a failing sink
    failing_sink = Mock()
    failing_sink.emit = AsyncMock(side_effect=Exception("Sink failure"))
    failing_sink.close = AsyncMock()

    emitter = get_global_event_emitter()
    emitter.add_sink(failing_sink)

    try:
        context = AgentContext(query="Test resilience")

        async with trace(correlation_id="test-resilience-123") as ctx:
            agent = MockAgent("resilient_agent")
            result = await agent.run_with_retry(context)

            # Agent execution should succeed despite event sink failure
            assert "resilient_agent" in result.agent_outputs

    finally:
        # Cleanup failing sink
        emitter.remove_sink(failing_sink)


if __name__ == "__main__":
    """Run integration tests manually."""
    import os

    # Enable events for testing
    os.environ["COGNIVAULT_EVENTS_ENABLED"] = "true"
    os.environ["COGNIVAULT_EVENTS_IN_MEMORY"] = "true"

    # Run a simple test
    async def simple_test():
        # Setup
        sink = InMemoryEventSink()
        emitter = get_global_event_emitter()
        emitter.enable()
        emitter.add_sink(sink)

        try:
            # Test basic agent execution
            context = AgentContext(query="Simple test query")
            agent = MockAgent("simple_test_agent")

            async with trace(correlation_id="simple-test-123") as ctx:
                result = await agent.run_with_retry(context)
                print(f"Agent executed successfully: {result.agent_outputs}")

            await asyncio.sleep(0.1)

            # Check events
            events = sink.get_events()
            print(f"Events emitted: {len(events)}")
            for event in events:
                print(f"  - {event.event_type.value}: {event.correlation_id}")

            return len(events) > 0

        finally:
            emitter.remove_sink(sink)

    # Run the test
    success = asyncio.run(simple_test())
    print(f"Integration test {'PASSED' if success else 'FAILED'}")
