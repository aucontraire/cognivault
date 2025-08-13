"""
Tests for observability context management.

This module tests correlation ID tracking, observability context management,
and thread-local storage functionality.
"""

import pytest
from typing import Any
import threading
import time
import uuid
from cognivault.observability.context import (
    ObservabilityContext,
    get_observability_context,
    set_observability_context,
    clear_observability_context,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    observability_context,
    ObservabilityContextManager,
)


class TestObservabilityContext:
    """Test ObservabilityContext dataclass functionality."""

    def test_default_context_creation(self) -> None:
        """Test creating context with default values."""
        context = ObservabilityContext()

        assert context.correlation_id is not None
        assert len(context.correlation_id) == 36  # UUID format
        assert context.agent_name is None
        assert context.step_id is None
        assert context.pipeline_id is None
        assert context.execution_phase is None
        assert context.metadata == {}

    def test_context_creation_with_values(self) -> None:
        """Test creating context with explicit values."""
        correlation_id = str(uuid.uuid4())
        metadata = {"key": "value", "count": 42}

        context = ObservabilityContext(
            correlation_id=correlation_id,
            agent_name="TestAgent",
            step_id="step_123",
            pipeline_id="pipeline_456",
            execution_phase="testing",
            metadata=metadata,
        )

        assert context.correlation_id == correlation_id
        assert context.agent_name == "TestAgent"
        assert context.step_id == "step_123"
        assert context.pipeline_id == "pipeline_456"
        assert context.execution_phase == "testing"
        assert context.metadata == metadata

    def test_with_agent_method(self) -> None:
        """Test creating new context with agent information."""
        original = ObservabilityContext(
            agent_name="OriginalAgent",
            step_id="original_step",
            metadata={"original": True},
        )

        new_context = original.with_agent("NewAgent", "new_step")

        # Should preserve correlation_id and other fields
        assert new_context.correlation_id == original.correlation_id
        assert new_context.pipeline_id == original.pipeline_id
        assert new_context.execution_phase == original.execution_phase

        # Should update agent info
        assert new_context.agent_name == "NewAgent"
        assert new_context.step_id == "new_step"

        # Should copy metadata
        assert new_context.metadata == original.metadata
        assert new_context.metadata is not original.metadata  # Different object

    def test_with_step_method(self) -> None:
        """Test creating new context with step information."""
        original = ObservabilityContext(agent_name="TestAgent", step_id="old_step")

        new_context = original.with_step("new_step")

        assert new_context.correlation_id == original.correlation_id
        assert new_context.agent_name == original.agent_name
        assert new_context.step_id == "new_step"

    def test_with_phase_method(self) -> None:
        """Test creating new context with execution phase."""
        original = ObservabilityContext(execution_phase="old_phase")

        new_context = original.with_phase("new_phase")

        assert new_context.correlation_id == original.correlation_id
        assert new_context.execution_phase == "new_phase"

    def test_with_metadata_method(self) -> None:
        """Test creating new context with additional metadata."""
        original = ObservabilityContext(metadata={"existing": "value"})

        new_context = original.with_metadata(new_key="new_value", count=42)

        assert new_context.correlation_id == original.correlation_id
        assert new_context.metadata == {
            "existing": "value",
            "new_key": "new_value",
            "count": 42,
        }
        assert new_context.metadata is not original.metadata


class TestContextStorage:
    """Test thread-local context storage functionality."""

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_observability_context()

    def test_get_context_when_none_set(self) -> None:
        """Test getting context when none is set."""
        context = get_observability_context()
        assert context is None

    def test_set_and_get_context(self) -> None:
        """Test setting and getting context."""
        original_context = ObservabilityContext(agent_name="TestAgent")

        set_observability_context(original_context)
        retrieved_context = get_observability_context()

        assert retrieved_context is original_context
        assert retrieved_context.agent_name == "TestAgent"

    def test_clear_context(self) -> None:
        """Test clearing context."""
        context = ObservabilityContext(agent_name="TestAgent")
        set_observability_context(context)

        assert get_observability_context() is not None

        clear_observability_context()

        assert get_observability_context() is None

    def test_thread_isolation(self) -> None:
        """Test that context is isolated between threads."""
        results = {}

        def thread_worker(thread_id: Any) -> None:
            # Set different context in each thread
            context = ObservabilityContext(agent_name=f"Agent{thread_id}")
            set_observability_context(context)

            # Wait a bit to let other threads set their contexts
            time.sleep(0.1)

            # Retrieve context (should be thread-specific)
            retrieved = get_observability_context()
            results[thread_id] = retrieved.agent_name if retrieved else None

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Each thread should have its own context
        assert results[0] == "Agent0"
        assert results[1] == "Agent1"
        assert results[2] == "Agent2"


class TestCorrelationIdHelpers:
    """Test correlation ID helper functions."""

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_correlation_id()

    def test_get_correlation_id_when_no_context(self) -> None:
        """Test getting correlation ID when no context is set."""
        correlation_id = get_correlation_id()
        assert correlation_id is None

    def test_get_correlation_id_with_context(self) -> None:
        """Test getting correlation ID when context exists."""
        context = ObservabilityContext(correlation_id="test-correlation-id")
        set_observability_context(context)

        correlation_id = get_correlation_id()
        assert correlation_id == "test-correlation-id"

    def test_set_correlation_id_with_existing_context(self) -> None:
        """Test setting correlation ID when context exists."""
        context = ObservabilityContext(correlation_id="old-id", agent_name="TestAgent")
        set_observability_context(context)

        set_correlation_id("new-correlation-id")

        updated_context = get_observability_context()
        assert updated_context is not None
        assert updated_context.correlation_id == "new-correlation-id"
        assert updated_context.agent_name == "TestAgent"  # Other fields preserved

    def test_set_correlation_id_without_existing_context(self) -> None:
        """Test setting correlation ID when no context exists."""
        set_correlation_id("new-correlation-id")

        context = get_observability_context()
        assert context is not None
        assert context.correlation_id == "new-correlation-id"
        assert context.agent_name is None

    def test_clear_correlation_id(self) -> None:
        """Test clearing correlation ID."""
        context = ObservabilityContext(correlation_id="test-id")
        set_observability_context(context)

        assert get_correlation_id() == "test-id"

        clear_correlation_id()

        assert get_observability_context() is None
        assert get_correlation_id() is None


class TestObservabilityContextManager:
    """Test ObservabilityContextManager functionality."""

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_observability_context()

    def test_context_manager_basic_usage(self) -> None:
        """Test basic context manager usage."""
        test_context = ObservabilityContext(agent_name="TestAgent")

        assert get_observability_context() is None

        with ObservabilityContextManager(test_context) as ctx:
            assert ctx is test_context
            assert get_observability_context() is test_context

        assert get_observability_context() is None

    def test_context_manager_with_existing_context(self) -> None:
        """Test context manager when context already exists."""
        original_context = ObservabilityContext(agent_name="OriginalAgent")
        new_context = ObservabilityContext(agent_name="NewAgent")

        set_observability_context(original_context)

        with ObservabilityContextManager(new_context):
            assert get_observability_context() is new_context

        # Should restore original context
        assert get_observability_context() is original_context

    def test_context_manager_with_exception(self) -> None:
        """Test context manager behavior when exception occurs."""
        original_context = ObservabilityContext(agent_name="OriginalAgent")
        new_context = ObservabilityContext(agent_name="NewAgent")

        set_observability_context(original_context)

        with pytest.raises(ValueError):
            with ObservabilityContextManager(new_context):
                assert get_observability_context() is new_context
                raise ValueError("Test exception")

        # Should still restore original context
        assert get_observability_context() is original_context


class TestObservabilityContextFunction:
    """Test observability_context convenience function."""

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_observability_context()

    def test_observability_context_with_all_parameters(self) -> None:
        """Test observability_context function with all parameters."""
        with observability_context(
            correlation_id="test-correlation",
            agent_name="TestAgent",
            step_id="test-step",
            pipeline_id="test-pipeline",
            execution_phase="testing",
            custom_field="custom_value",
        ) as ctx:
            assert ctx.correlation_id == "test-correlation"
            assert ctx.agent_name == "TestAgent"
            assert ctx.step_id == "test-step"
            assert ctx.pipeline_id == "test-pipeline"
            assert ctx.execution_phase == "testing"
            assert ctx.metadata["custom_field"] == "custom_value"

            # Should be set in thread-local storage
            assert get_observability_context() is ctx

    def test_observability_context_generates_correlation_id(self) -> None:
        """Test that observability_context generates correlation ID if not provided."""
        with observability_context(agent_name="TestAgent") as ctx:
            assert ctx.correlation_id is not None
            assert len(ctx.correlation_id) == 36  # UUID format
            assert ctx.agent_name == "TestAgent"

    def test_observability_context_nested(self) -> None:
        """Test nested observability contexts."""
        with observability_context(agent_name="OuterAgent") as outer_ctx:
            outer_observability_context = get_observability_context()
            assert outer_observability_context is not None
            assert outer_observability_context.agent_name == "OuterAgent"

            with observability_context(agent_name="InnerAgent") as inner_ctx:
                inner_observability_context = get_observability_context()
                assert inner_observability_context is not None
                assert inner_observability_context.agent_name == "InnerAgent"
                assert inner_ctx.agent_name == "InnerAgent"

            # Should restore outer context
            outer_observability_ctx = get_observability_context()
            assert outer_observability_ctx is not None
            assert outer_observability_ctx.agent_name == "OuterAgent"
