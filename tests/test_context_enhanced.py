import pytest
import json
from unittest.mock import patch, MagicMock
from cognivault.context import (
    AgentContext,
    ContextSnapshot,
    ContextCompressionManager,
)


class TestContextCompressionManager:
    """Test the context compression manager functionality."""

    def test_calculate_size(self):
        """Test size calculation for various data types."""
        manager = ContextCompressionManager()

        # Test string data
        data = {"key": "value"}
        size = manager.calculate_size(data)
        assert size > 0
        assert isinstance(size, int)

        # Test larger data
        large_data = {"key": "x" * 1000}
        large_size = manager.calculate_size(large_data)
        assert large_size > size

    def test_compress_and_decompress_data(self):
        """Test data compression and decompression."""
        manager = ContextCompressionManager()

        original_data = {
            "query": "What is democracy?",
            "outputs": {"agent1": "response1", "agent2": "response2"},
            "config": {"setting1": "value1"},
        }

        # Compress data
        compressed = manager.compress_data(original_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        # Decompress data
        decompressed = manager.decompress_data(compressed)
        assert decompressed == original_data

    def test_truncate_large_outputs(self):
        """Test truncation of large outputs."""
        manager = ContextCompressionManager()

        outputs = {
            "short_output": "small",
            "long_output": "x" * 1000,
            "medium_output": "y" * 50,
        }

        max_size = 100
        truncated = manager.truncate_large_outputs(outputs, max_size)

        assert truncated["short_output"] == "small"
        assert (
            len(truncated["long_output"]) <= max_size + 50
        )  # Account for truncation message
        assert "truncated" in truncated["long_output"]
        assert truncated["medium_output"] == "y" * 50

    def test_calculate_size_with_unserializable_data(self):
        """Test size calculation with data that can't be JSON serialized."""
        manager = ContextCompressionManager()

        # Create an object that can't be JSON serialized
        class UnserializableObject:
            def __init__(self):
                self.circular_ref = self

        unserializable_data = UnserializableObject()

        # Should fall back to str() conversion
        size = manager.calculate_size(unserializable_data)
        assert size > 0
        assert isinstance(size, int)

    def test_calculate_size_json_exception_handling(self):
        """Test that calculate_size properly handles JSON encoding errors."""
        manager = ContextCompressionManager()

        # Create an object that raises a specific JSON encoding error
        class ProblematicObject:
            def __str__(self):
                return "ProblematicObject instance"

        # Use a set with the problematic object to trigger TypeError in json.dumps
        problematic_data = {ProblematicObject()}

        # Should catch the TypeError and fall back to str() conversion
        size = manager.calculate_size(problematic_data)
        assert size > 0
        assert isinstance(size, int)

    @patch("cognivault.context.json.dumps")
    def test_calculate_size_handles_json_exceptions(self, mock_json_dumps):
        """Test that calculate_size handles both TypeError and ValueError from json.dumps."""
        manager = ContextCompressionManager()

        # Test TypeError exception handling
        mock_json_dumps.side_effect = TypeError("Object not JSON serializable")
        size = manager.calculate_size({"key": "value"})
        assert size > 0
        assert isinstance(size, int)

        # Test ValueError exception handling
        mock_json_dumps.side_effect = ValueError("Out of range float values")
        size = manager.calculate_size({"key": "value"})
        assert size > 0
        assert isinstance(size, int)


class TestContextSnapshot:
    """Test context snapshot functionality."""

    def test_snapshot_creation(self):
        """Test creating a context snapshot."""
        snapshot = ContextSnapshot(
            context_id="test_id",
            timestamp="2024-01-01T00:00:00Z",
            query="test query",
            agent_outputs={"agent1": "output1"},
            retrieved_notes=["note1", "note2"],
            user_config={"setting": "value"},
            final_synthesis="synthesis",
            agent_trace={"agent1": [{"input": "in", "output": "out"}]},
            size_bytes=100,
        )

        assert snapshot.context_id == "test_id"
        assert snapshot.query == "test query"
        assert snapshot.agent_outputs == {"agent1": "output1"}
        assert not snapshot.compressed  # Default value

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = ContextSnapshot(
            context_id="test_id",
            timestamp="2024-01-01T00:00:00Z",
            query="test query",
            agent_outputs={"agent1": "output1"},
            retrieved_notes=["note1"],
            user_config={"setting": "value"},
            final_synthesis="synthesis",
            agent_trace={},
            size_bytes=100,
        )

        snapshot_dict = snapshot.to_dict()
        assert isinstance(snapshot_dict, dict)
        assert snapshot_dict["context_id"] == "test_id"
        assert snapshot_dict["query"] == "test query"

    def test_snapshot_from_dict(self):
        """Test creating snapshot from dictionary."""
        data = {
            "context_id": "test_id",
            "timestamp": "2024-01-01T00:00:00Z",
            "query": "test query",
            "agent_outputs": {"agent1": "output1"},
            "retrieved_notes": ["note1"],
            "user_config": {"setting": "value"},
            "final_synthesis": "synthesis",
            "agent_trace": {},
            "size_bytes": 100,
            "compressed": False,
        }

        snapshot = ContextSnapshot.from_dict(data)
        assert isinstance(snapshot, ContextSnapshot)
        assert snapshot.context_id == "test_id"


class TestEnhancedAgentContext:
    """Test enhanced agent context functionality."""

    def test_context_initialization(self):
        """Test context initialization with new features."""
        context = AgentContext(query="What is AI?")

        assert context.query == "What is AI?"
        assert hasattr(context, "context_id")
        assert hasattr(context, "snapshots")
        assert hasattr(context, "current_size")
        assert hasattr(context, "compression_manager")
        assert context.current_size >= 0

    def test_context_id_generation(self):
        """Test context ID generation."""
        context1 = AgentContext(query="Query 1")
        context2 = AgentContext(query="Query 2")

        # Context IDs should be different
        assert context1.get_context_id() != context2.get_context_id()
        assert len(context1.get_context_id()) == 8  # MD5 hash truncated to 8 chars

    def test_size_monitoring(self):
        """Test context size monitoring."""
        context = AgentContext(query="Small query")
        initial_size = context.get_current_size_bytes()

        # Add some data and check size increases
        context.add_agent_output(
            "agent1", "This is a longer output that should increase size"
        )
        new_size = context.get_current_size_bytes()
        assert new_size > initial_size

    @patch("cognivault.context.get_config")
    def test_size_limit_enforcement(self, mock_get_config):
        """Test that size limits are enforced."""
        # Mock config to return small size limit
        mock_config = MagicMock()
        mock_config.testing.max_context_size_bytes = 100
        mock_get_config.return_value = mock_config

        context = AgentContext(query="Test query")

        # Add large output that exceeds limit
        large_output = "x" * 1000
        context.add_agent_output("agent1", large_output)

        # Context should have been compressed
        assert context.get_current_size_bytes() <= 1000  # Should be compressed

    def test_create_snapshot(self):
        """Test creating context snapshots."""
        context = AgentContext(query="Test query")
        context.add_agent_output("agent1", "output1")
        context.update_user_config({"setting": "value"})

        snapshot_id = context.create_snapshot(label="test_snapshot")

        assert isinstance(snapshot_id, str)
        assert len(context.snapshots) == 1
        assert context.snapshots[0].query == "Test query"
        assert context.snapshots[0].agent_outputs == {"agent1": "output1"}

    def test_restore_snapshot(self):
        """Test restoring from snapshots."""
        context = AgentContext(query="Original query")
        context.add_agent_output("agent1", "original_output")

        # Create snapshot
        snapshot_id = context.create_snapshot()

        # Modify context
        context.query = "Modified query"
        context.add_agent_output("agent2", "new_output")

        # Restore from snapshot
        success = context.restore_snapshot(snapshot_id)

        assert success
        assert context.query == "Original query"
        assert context.agent_outputs == {"agent1": "original_output"}
        assert "agent2" not in context.agent_outputs

    def test_list_snapshots(self):
        """Test listing snapshots."""
        context = AgentContext(query="Test query")

        # Create multiple snapshots
        context.add_agent_output("agent1", "output1")
        snapshot1 = context.create_snapshot()

        context.add_agent_output("agent2", "output2")
        snapshot2 = context.create_snapshot()

        snapshots = context.list_snapshots()

        assert len(snapshots) == 2
        assert all("timestamp" in s for s in snapshots)
        assert all("size_bytes" in s for s in snapshots)
        assert all("agents_present" in s for s in snapshots)
        assert snapshots[1]["agents_present"] == ["agent1", "agent2"]

    def test_clear_snapshots(self):
        """Test clearing snapshots."""
        context = AgentContext(query="Test query")

        # Create snapshots
        context.create_snapshot()
        context.create_snapshot()

        assert len(context.snapshots) == 2

        context.clear_snapshots()

        assert len(context.snapshots) == 0

    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        context = AgentContext(query="Test query")
        context.add_agent_output("agent1", "output1")
        context.log_trace("agent1", "input", "output")
        context.create_snapshot()

        usage = context.get_memory_usage()

        assert isinstance(usage, dict)
        assert "total_size_bytes" in usage
        assert "agent_outputs_size" in usage
        assert "agent_trace_size" in usage
        assert "snapshots_count" in usage
        assert "snapshots_total_size" in usage
        assert "context_id" in usage

        assert usage["snapshots_count"] == 1
        assert usage["total_size_bytes"] > 0

    def test_optimize_memory(self):
        """Test memory optimization."""
        context = AgentContext(query="Test query")

        # Create many snapshots
        for i in range(10):
            context.add_agent_output(f"agent{i}", f"output{i}")
            context.create_snapshot()

        initial_snapshots = len(context.snapshots)
        assert initial_snapshots == 10

        # Optimize memory
        stats = context.optimize_memory()

        assert isinstance(stats, dict)
        assert "size_before" in stats
        assert "size_after" in stats
        assert "snapshots_before" in stats
        assert "snapshots_after" in stats
        assert "snapshots_removed" in stats

        # Should keep only 5 most recent snapshots
        assert len(context.snapshots) == 5
        assert stats["snapshots_removed"] == 5

    def test_clone_context(self):
        """Test cloning context for parallel processing."""
        context = AgentContext(query="Original query")
        context.add_agent_output("agent1", "output1")
        context.update_user_config({"setting": "value"})
        context.set_final_synthesis("synthesis")

        cloned = context.clone()

        # Should be a different instance
        assert cloned is not context
        assert cloned.get_context_id() != context.get_context_id()

        # Should have same data
        assert cloned.query == context.query
        assert cloned.agent_outputs == context.agent_outputs
        assert cloned.user_config == context.user_config
        assert cloned.final_synthesis == context.final_synthesis

        # Should be independent (deep copy)
        cloned.add_agent_output("agent2", "output2")
        assert "agent2" not in context.agent_outputs

    def test_size_update_on_operations(self):
        """Test that size is updated on various operations."""
        context = AgentContext(query="Test query")
        initial_size = context.get_current_size_bytes()

        # Add agent output
        context.add_agent_output("agent1", "output")
        size_after_output = context.get_current_size_bytes()
        assert size_after_output > initial_size

        # Update user config
        context.update_user_config({"key": "value"})
        size_after_config = context.get_current_size_bytes()
        assert size_after_config > size_after_output

        # Set final synthesis
        context.set_final_synthesis("Final synthesis text")
        size_after_synthesis = context.get_current_size_bytes()
        assert size_after_synthesis > size_after_config

        # Log trace
        context.log_trace("agent1", "input", "output")
        size_after_trace = context.get_current_size_bytes()
        assert size_after_trace > size_after_synthesis

    def test_snapshot_restore_invalid_id(self):
        """Test restoring with invalid snapshot ID."""
        context = AgentContext(query="Test query")

        # Try to restore non-existent snapshot
        success = context.restore_snapshot("invalid_id")

        assert not success

    @patch("cognivault.context.get_config")
    def test_compression_on_large_context(self, mock_get_config):
        """Test compression when context becomes too large."""
        # Mock config with very small size limit
        mock_config = MagicMock()
        mock_config.testing.max_context_size_bytes = 50
        mock_get_config.return_value = mock_config

        context = AgentContext(query="Test")

        # Add large agent trace
        for i in range(10):
            context.log_trace(f"agent{i}", f"input{i}" * 100, f"output{i}" * 100)

        # Context should have been compressed
        # Agent trace should be limited to last 3 entries per agent
        for agent_name in context.agent_trace:
            assert len(context.agent_trace[agent_name]) <= 3

    @patch("cognivault.context.get_config")
    def test_agent_trace_compression_in_context(self, mock_get_config):
        """Test agent trace compression when context exceeds size limit."""
        # Mock config with small size limit to trigger compression
        mock_config = MagicMock()
        mock_config.testing.max_context_size_bytes = 100
        mock_get_config.return_value = mock_config

        context = AgentContext(query="Test query")

        # Add multiple traces for a single agent to trigger compression
        agent_name = "test_agent"
        for i in range(10):
            context.log_trace(
                agent_name, f"large_input_{i}" * 20, f"large_output_{i}" * 20
            )

        # Verify the agent trace was compressed to 3 entries
        assert len(context.agent_trace[agent_name]) <= 3

    @patch("cognivault.context.get_config")
    def test_optimize_memory_with_compression(self, mock_get_config):
        """Test memory optimization that triggers context compression."""
        # Mock config with small size limit
        mock_config = MagicMock()
        mock_config.testing.max_context_size_bytes = 200
        mock_get_config.return_value = mock_config

        context = AgentContext(query="Test query")

        # Add large amount of data to exceed size limit
        for i in range(3):
            context.add_agent_output(f"agent{i}", "x" * 200)

        # Manually set current_size to a large value to trigger compression
        context.current_size = 1000

        # Optimize memory - this should trigger compression
        stats = context.optimize_memory()

        # Verify compression was triggered
        assert stats["size_after"] <= stats["size_before"]

    def test_getter_methods_coverage(self):
        """Test getter methods to achieve 100% coverage."""
        context = AgentContext(query="Test query")

        # Test get_output with existing and non-existing agents
        context.add_agent_output("agent1", "output1")
        assert context.get_output("agent1") == "output1"
        assert context.get_output("nonexistent") is None

        # Test get_user_config with default value
        context.update_user_config({"key1": "value1"})
        assert context.get_user_config("key1") == "value1"
        assert context.get_user_config("nonexistent", "default") == "default"

        # Test get_final_synthesis
        context.set_final_synthesis("Test synthesis")
        assert context.get_final_synthesis() == "Test synthesis"
