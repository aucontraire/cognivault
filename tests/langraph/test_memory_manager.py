"""
Tests for LangGraph Memory Manager and checkpointing functionality.

This module provides comprehensive tests for the CogniVaultMemoryManager,
including checkpoint creation, serialization, rollback mechanisms, and
integration with LangGraph MemorySaver.
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from cognivault.langraph.memory_manager import (
    CogniVaultMemoryManager,
    CheckpointConfig,
    CheckpointInfo,
    create_memory_manager,
)
from cognivault.langraph.state_schemas import create_initial_state


@pytest.fixture
def sample_state():
    """Fixture providing a sample CogniVaultState for testing."""
    return create_initial_state("test query", "test_execution_123")


@pytest.fixture
def memory_manager():
    """Fixture providing a memory manager with checkpointing enabled."""
    config = CheckpointConfig(enabled=True, thread_id="test_thread")
    return CogniVaultMemoryManager(config)


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CheckpointConfig()

        assert config.enabled is False
        assert config.thread_id is None
        assert config.auto_generate_thread_id is True
        assert config.checkpoint_dir is None
        assert config.max_checkpoints_per_thread == 10
        assert config.checkpoint_ttl_hours == 24
        assert config.enable_rollback is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CheckpointConfig(
            enabled=True,
            thread_id="test_thread",
            auto_generate_thread_id=False,
            max_checkpoints_per_thread=5,
            checkpoint_ttl_hours=12,
            enable_rollback=False,
        )

        assert config.enabled is True
        assert config.thread_id == "test_thread"
        assert config.auto_generate_thread_id is False
        assert config.max_checkpoints_per_thread == 5
        assert config.checkpoint_ttl_hours == 12
        assert config.enable_rollback is False


class TestCheckpointInfo:
    """Test CheckpointInfo dataclass."""

    def test_checkpoint_info_creation(self):
        """Test checkpoint info creation."""
        timestamp = datetime.now(timezone.utc)
        metadata = {"test": "value", "execution_id": "test_123"}

        info = CheckpointInfo(
            checkpoint_id="checkpoint_123",
            thread_id="thread_456",
            timestamp=timestamp,
            agent_step="refiner",
            state_size_bytes=1024,
            success=True,
            metadata=metadata,
        )

        assert info.checkpoint_id == "checkpoint_123"
        assert info.thread_id == "thread_456"
        assert info.timestamp == timestamp
        assert info.agent_step == "refiner"
        assert info.state_size_bytes == 1024
        assert info.success is True
        assert info.metadata == metadata


class TestCogniVaultMemoryManager:
    """Test CogniVaultMemoryManager functionality."""

    @pytest.fixture
    def disabled_config(self):
        """Fixture for disabled checkpoint configuration."""
        return CheckpointConfig(enabled=False)

    @pytest.fixture
    def enabled_config(self):
        """Fixture for enabled checkpoint configuration."""
        return CheckpointConfig(
            enabled=True,
            thread_id="test_thread",
            max_checkpoints_per_thread=3,
            checkpoint_ttl_hours=1,
        )

    @pytest.fixture
    def memory_manager(self, enabled_config):
        """Fixture for memory manager with checkpointing enabled."""
        with patch(
            "cognivault.langraph.memory_manager.MemorySaver"
        ) as mock_memory_saver:
            mock_memory_saver.return_value = Mock()
            manager = CogniVaultMemoryManager(enabled_config)
            yield manager

    @pytest.fixture
    def disabled_memory_manager(self, disabled_config):
        """Fixture for memory manager with checkpointing disabled."""
        return CogniVaultMemoryManager(disabled_config)

    @pytest.fixture
    def sample_state(self):
        """Fixture for sample CogniVaultState."""
        return create_initial_state("Test query", "test_execution_123")

    def test_initialization_disabled(self, disabled_memory_manager):
        """Test memory manager initialization with checkpointing disabled."""
        manager = disabled_memory_manager

        assert not manager.is_enabled()
        assert manager.memory_saver is None
        assert manager.checkpoints == {}

    def test_initialization_enabled(self, memory_manager):
        """Test memory manager initialization with checkpointing enabled."""
        assert memory_manager.is_enabled()
        assert memory_manager.memory_saver is not None
        assert memory_manager.checkpoints == {}

    def test_generate_thread_id(self, memory_manager):
        """Test thread ID generation."""
        thread_id = memory_manager.generate_thread_id()

        assert isinstance(thread_id, str)
        assert thread_id.startswith("cognivault_")
        assert len(thread_id.split("_")) == 4  # cognivault_YYYYMMDD_HHMMSS_uuid

        # Ensure uniqueness
        thread_id2 = memory_manager.generate_thread_id()
        assert thread_id != thread_id2

    def test_get_thread_id_provided(self, memory_manager):
        """Test getting thread ID when provided explicitly."""
        provided_id = "explicit_thread_123"
        result = memory_manager.get_thread_id(provided_id)
        assert result == provided_id

    def test_get_thread_id_from_config(self, memory_manager):
        """Test getting thread ID from configuration."""
        result = memory_manager.get_thread_id()
        assert result == "test_thread"  # From enabled_config fixture

    def test_get_thread_id_auto_generate(self):
        """Test auto-generation when no thread ID provided."""
        config = CheckpointConfig(
            enabled=True, thread_id=None, auto_generate_thread_id=True
        )
        with patch("cognivault.langraph.memory_manager.MemorySaver"):
            manager = CogniVaultMemoryManager(config)

            result = manager.get_thread_id()
            assert result.startswith("cognivault_")

    def test_get_thread_id_fallback(self):
        """Test fallback to default thread."""
        config = CheckpointConfig(
            enabled=True, thread_id=None, auto_generate_thread_id=False
        )
        with patch("cognivault.langraph.memory_manager.MemorySaver"):
            manager = CogniVaultMemoryManager(config)

            result = manager.get_thread_id()
            assert result == "default_thread"

    def test_create_checkpoint_disabled(self, disabled_memory_manager, sample_state):
        """Test checkpoint creation when disabled."""
        result = disabled_memory_manager.create_checkpoint(
            thread_id="test_thread", state=sample_state, agent_step="refiner"
        )

        assert result == ""
        assert disabled_memory_manager.checkpoints == {}

    def test_create_checkpoint_enabled(self, memory_manager, sample_state):
        """Test checkpoint creation when enabled."""
        thread_id = "test_thread_123"
        agent_step = "refiner"
        metadata = {"test": "value"}

        checkpoint_id = memory_manager.create_checkpoint(
            thread_id=thread_id,
            state=sample_state,
            agent_step=agent_step,
            metadata=metadata,
        )

        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) > 0

        # Check checkpoint was stored
        assert thread_id in memory_manager.checkpoints
        checkpoints = memory_manager.checkpoints[thread_id]
        assert len(checkpoints) == 1

        checkpoint = checkpoints[0]
        assert checkpoint.checkpoint_id == checkpoint_id
        assert checkpoint.thread_id == thread_id
        assert checkpoint.agent_step == agent_step
        assert checkpoint.success is True
        assert checkpoint.metadata == metadata
        assert checkpoint.state_size_bytes > 0

    def test_checkpoint_limit_enforcement(self, memory_manager, sample_state):
        """Test that checkpoint limit per thread is enforced."""
        thread_id = "test_thread_limit"

        # Create more checkpoints than the limit (3 from enabled_config)
        checkpoint_ids = []
        for i in range(5):
            checkpoint_id = memory_manager.create_checkpoint(
                thread_id=thread_id, state=sample_state, agent_step=f"step_{i}"
            )
            checkpoint_ids.append(checkpoint_id)

        # Should only have 3 checkpoints (the limit)
        checkpoints = memory_manager.checkpoints[thread_id]
        assert len(checkpoints) == 3

        # Should have the 3 most recent checkpoints
        stored_ids = [c.checkpoint_id for c in checkpoints]
        assert checkpoint_ids[2] in stored_ids  # 3rd checkpoint
        assert checkpoint_ids[3] in stored_ids  # 4th checkpoint
        assert checkpoint_ids[4] in stored_ids  # 5th checkpoint
        assert checkpoint_ids[0] not in stored_ids  # 1st should be removed
        assert checkpoint_ids[1] not in stored_ids  # 2nd should be removed

    def test_get_checkpoint_history(self, memory_manager, sample_state):
        """Test getting checkpoint history."""
        thread_id = "test_history"

        # Create multiple checkpoints
        for i in range(3):
            memory_manager.create_checkpoint(
                thread_id=thread_id, state=sample_state, agent_step=f"step_{i}"
            )

        history = memory_manager.get_checkpoint_history(thread_id)

        assert len(history) == 3
        # Should be sorted by timestamp, newest first
        for i in range(len(history) - 1):
            assert history[i].timestamp >= history[i + 1].timestamp

    def test_get_checkpoint_history_empty(self, memory_manager):
        """Test getting history for thread with no checkpoints."""
        history = memory_manager.get_checkpoint_history("nonexistent_thread")
        assert history == []

    def test_get_latest_checkpoint(self, memory_manager, sample_state):
        """Test getting latest checkpoint."""
        thread_id = "test_latest"

        # Create checkpoints
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = memory_manager.create_checkpoint(
                thread_id=thread_id, state=sample_state, agent_step=f"step_{i}"
            )
            checkpoint_ids.append(checkpoint_id)

        latest = memory_manager.get_latest_checkpoint(thread_id)

        assert latest is not None
        assert latest.checkpoint_id == checkpoint_ids[-1]  # Most recent

    def test_get_latest_checkpoint_none(self, memory_manager):
        """Test getting latest checkpoint when none exist."""
        latest = memory_manager.get_latest_checkpoint("nonexistent_thread")
        assert latest is None

    def test_cleanup_expired_checkpoints(self, memory_manager, sample_state):
        """Test cleanup of expired checkpoints."""
        thread_id = "test_cleanup"

        # Create checkpoints with manipulated timestamps
        now = datetime.now(timezone.utc)

        # Create recent checkpoint
        recent_id = memory_manager.create_checkpoint(
            thread_id=thread_id, state=sample_state, agent_step="recent"
        )

        # Create expired checkpoint by manipulating timestamp
        expired_id = memory_manager.create_checkpoint(
            thread_id=thread_id, state=sample_state, agent_step="expired"
        )

        # Manually set expired timestamp
        expired_checkpoint = memory_manager.checkpoints[thread_id][-1]
        expired_checkpoint.timestamp = now - timedelta(hours=2)  # TTL is 1 hour

        # Run cleanup
        removed_count = memory_manager.cleanup_expired_checkpoints()

        assert removed_count == 1
        remaining_checkpoints = memory_manager.checkpoints[thread_id]
        assert len(remaining_checkpoints) == 1
        assert remaining_checkpoints[0].checkpoint_id == recent_id

    def test_cleanup_no_ttl(self, sample_state):
        """Test cleanup when TTL is not set."""
        config = CheckpointConfig(enabled=True, checkpoint_ttl_hours=None)
        with patch("cognivault.langraph.memory_manager.MemorySaver"):
            manager = CogniVaultMemoryManager(config)

            removed_count = manager.cleanup_expired_checkpoints()
            assert removed_count == 0

    def test_cleanup_removes_empty_threads(self, memory_manager, sample_state):
        """Test that threads with no valid checkpoints are removed."""
        thread_id = "test_remove_thread"

        # Create checkpoint
        memory_manager.create_checkpoint(
            thread_id=thread_id, state=sample_state, agent_step="test"
        )

        # Manually set expired timestamp
        checkpoint = memory_manager.checkpoints[thread_id][0]
        checkpoint.timestamp = datetime.now(timezone.utc) - timedelta(hours=2)

        # Run cleanup
        memory_manager.cleanup_expired_checkpoints()

        # Thread should be removed entirely
        assert thread_id not in memory_manager.checkpoints

    def test_rollback_disabled_checkpointing(self, disabled_memory_manager):
        """Test rollback when checkpointing is disabled."""
        result = disabled_memory_manager.rollback_to_checkpoint("test_thread")
        assert result is None

    def test_rollback_disabled_rollback_flag(self, sample_state):
        """Test rollback when rollback flag is disabled."""
        config = CheckpointConfig(enabled=True, enable_rollback=False)
        with patch("cognivault.langraph.memory_manager.MemorySaver"):
            manager = CogniVaultMemoryManager(config)

            result = manager.rollback_to_checkpoint("test_thread")
            assert result is None

    def test_rollback_no_checkpoints(self, memory_manager):
        """Test rollback when no checkpoints exist."""
        result = memory_manager.rollback_to_checkpoint("nonexistent_thread")
        assert result is None

    def test_rollback_specific_checkpoint_not_found(self, memory_manager, sample_state):
        """Test rollback to specific checkpoint that doesn't exist."""
        thread_id = "test_rollback"

        # Create a checkpoint
        memory_manager.create_checkpoint(
            thread_id=thread_id, state=sample_state, agent_step="test"
        )

        # Try to rollback to non-existent checkpoint
        result = memory_manager.rollback_to_checkpoint(
            thread_id=thread_id, checkpoint_id="nonexistent_checkpoint"
        )
        assert result is None

    @patch("cognivault.langraph.memory_manager.create_initial_state")
    def test_rollback_fallback_success(
        self, mock_create_state, memory_manager, sample_state
    ):
        """Test successful rollback fallback when MemorySaver data unavailable."""
        thread_id = "test_rollback_fallback"

        # Create checkpoint
        checkpoint_id = memory_manager.create_checkpoint(
            thread_id=thread_id, state=sample_state, agent_step="test"
        )

        # Mock MemorySaver to return None
        memory_manager.memory_saver.get_tuple.return_value = None

        # Mock create_initial_state return
        mock_fallback_state = {
            "query": "fallback",
            "execution_id": "fallback_123",
            "execution_metadata": {},
        }
        mock_create_state.return_value = mock_fallback_state

        result = memory_manager.rollback_to_checkpoint(thread_id)

        assert result is not None
        assert result == mock_fallback_state
        assert result["execution_metadata"]["rollback_performed"] is True
        mock_create_state.assert_called_once()

    def test_rollback_with_memory_saver_data(self, memory_manager, sample_state):
        """Test rollback with actual MemorySaver data."""
        thread_id = "test_rollback_with_data"

        # Create checkpoint
        checkpoint_id = memory_manager.create_checkpoint(
            thread_id=thread_id, state=sample_state, agent_step="test"
        )

        # Mock MemorySaver to return checkpoint data
        mock_checkpoint_tuple = Mock()
        mock_checkpoint_tuple.checkpoint = {"channel_values": sample_state}
        memory_manager.memory_saver.get_tuple.return_value = mock_checkpoint_tuple

        result = memory_manager.rollback_to_checkpoint(thread_id)

        assert result is not None
        assert result == sample_state

    def test_state_serialization(self, memory_manager, sample_state):
        """Test state serialization functionality."""
        serialized = memory_manager._serialize_state(sample_state)

        assert isinstance(serialized, str)

        # Should be valid JSON
        data = json.loads(serialized)
        assert "_cognivault_version" in data
        assert "_serialization_timestamp" in data
        assert "_state_type" in data
        assert "data" in data

    def test_state_deserialization(self, memory_manager, sample_state):
        """Test state deserialization functionality."""
        # Serialize then deserialize
        serialized = memory_manager._serialize_state(sample_state)
        deserialized = memory_manager.deserialize_state(serialized)

        assert deserialized is not None
        assert isinstance(deserialized, dict)

    def test_state_serialization_error_fallback(self, memory_manager):
        """Test serialization error fallback."""
        # Create a state with non-serializable content
        problematic_state = {
            "query": "test",
            "execution_id": "test_123",
            "non_serializable": lambda x: x,  # Function can't be serialized
        }

        serialized = memory_manager._serialize_state(problematic_state)

        # Should still return valid JSON (enhanced serialization handles functions)
        data = json.loads(serialized)
        assert "_cognivault_version" in data
        assert data["_state_type"] == "CogniVaultState"

        # Check that function was serialized as object
        function_data = data["data"]["non_serializable"]
        assert function_data["_type"] == "object"
        assert function_data["_class"] == "function"

    def test_deserialization_error_handling(self, memory_manager):
        """Test deserialization error handling."""
        invalid_json = "invalid json content"

        result = memory_manager.deserialize_state(invalid_json)
        assert result is None

    def test_complex_type_serialization(self, memory_manager):
        """Test serialization of complex types."""
        now = datetime.now(timezone.utc)
        complex_state = {
            "query": "test",
            "execution_id": "test_123",
            "datetime_field": now,
            "tuple_field": (1, 2, 3),
            "set_field": {1, 2, 3},
            "nested_dict": {
                "inner_datetime": now,
                "inner_list": [1, 2, {"nested": "value"}],
            },
        }

        serialized = memory_manager._serialize_state(complex_state)
        deserialized = memory_manager.deserialize_state(serialized)

        assert deserialized is not None
        assert isinstance(deserialized["datetime_field"], datetime)
        assert isinstance(deserialized["tuple_field"], tuple)
        assert isinstance(deserialized["set_field"], set)

    def test_get_memory_stats(self, memory_manager, sample_state):
        """Test memory statistics collection."""
        thread_id = "test_stats"

        # Create some checkpoints
        for i in range(2):
            memory_manager.create_checkpoint(
                thread_id=thread_id, state=sample_state, agent_step=f"step_{i}"
            )

        stats = memory_manager.get_memory_stats()

        assert stats["enabled"] is True
        assert stats["total_threads"] == 1
        assert stats["total_checkpoints"] == 2
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] >= 0  # Small data may round to 0.0
        assert "config" in stats
        assert "threads" in stats
        assert thread_id in stats["threads"]


class TestFactoryFunction:
    """Test create_memory_manager factory function."""

    def test_factory_disabled(self):
        """Test factory with checkpointing disabled."""
        manager = create_memory_manager(enable_checkpoints=False)

        assert not manager.is_enabled()
        assert manager.config.enabled is False

    @patch("cognivault.langraph.memory_manager.MemorySaver")
    def test_factory_enabled(self, mock_memory_saver):
        """Test factory with checkpointing enabled."""
        mock_memory_saver.return_value = Mock()

        manager = create_memory_manager(
            enable_checkpoints=True,
            thread_id="test_thread",
            max_checkpoints_per_thread=5,
        )

        assert manager.is_enabled()
        assert manager.config.enabled is True
        assert manager.config.thread_id == "test_thread"
        assert manager.config.max_checkpoints_per_thread == 5

    def test_factory_with_kwargs(self):
        """Test factory with additional keyword arguments."""
        manager = create_memory_manager(
            enable_checkpoints=False, checkpoint_ttl_hours=48, enable_rollback=False
        )

        assert manager.config.checkpoint_ttl_hours == 48
        assert manager.config.enable_rollback is False


class TestErrorHandling:
    """Test error handling in memory manager."""

    @pytest.fixture
    def manager_with_mock_memory_saver(self):
        """Fixture with mocked MemorySaver that can raise exceptions."""
        config = CheckpointConfig(enabled=True)
        with patch("cognivault.langraph.memory_manager.MemorySaver") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            manager = CogniVaultMemoryManager(config)
            yield manager, mock_instance

    def test_rollback_exception_handling(
        self, manager_with_mock_memory_saver, sample_state
    ):
        """Test rollback exception handling."""
        manager, mock_memory_saver = manager_with_mock_memory_saver
        thread_id = "test_exception"

        # Create checkpoint
        manager.create_checkpoint(
            thread_id=thread_id, state=sample_state, agent_step="test"
        )

        # Make MemorySaver raise an exception
        mock_memory_saver.get_tuple.side_effect = Exception("MemorySaver error")

        result = manager.rollback_to_checkpoint(thread_id)
        assert result is None

    def test_serialization_with_circular_reference(self, memory_manager):
        """Test serialization with circular references."""
        # Create circular reference
        state_dict = {"query": "test", "execution_id": "test_123"}
        state_dict["circular"] = state_dict

        # Should handle gracefully with fallback
        serialized = memory_manager._serialize_state(state_dict)
        data = json.loads(serialized)

        # Should have error information
        assert "_serialization_error" in data or "_cognivault_version" in data


if __name__ == "__main__":
    pytest.main([__file__])
