"""
Comprehensive tests for AgentContext Pydantic migration (Phase 2.1).

This test suite covers all aspects of the enhanced Pydantic-based AgentContext:
- Field validation and constraints
- Model validators for data consistency
- Agent execution state management
- Snapshot and rollback functionality
- Memory management and compression
- Agent isolation and field locking
- LangGraph compatibility features
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, Mock
from typing import Dict, Any

from pydantic import ValidationError

from cognivault.context import AgentContext, ContextSnapshot, ContextCompressionManager
from cognivault.exceptions import StateTransitionError


class TestContextSnapshotPydanticValidation:
    """Test ContextSnapshot Pydantic validation."""

    def test_context_snapshot_valid_creation(self):
        """Test creating valid ContextSnapshot."""
        snapshot = ContextSnapshot(
            context_id="test_123",
            timestamp="2025-01-20T10:00:00+00:00",
            query="test query",
            agent_outputs={"refiner": "output"},
            retrieved_notes=["note1", "note2"],
            user_config={"key": "value"},
            final_synthesis="final result",
            agent_trace={
                "refiner": [{"timestamp": "now", "input": "in", "output": "out"}]
            },
            size_bytes=1024,
            compressed=False,
        )

        assert snapshot.context_id == "test_123"
        assert snapshot.query == "test query"
        assert snapshot.size_bytes == 1024
        assert not snapshot.compressed

    def test_context_snapshot_invalid_timestamp(self):
        """Test ContextSnapshot with invalid timestamp format."""
        with pytest.raises(ValidationError, match="Invalid timestamp format"):
            ContextSnapshot(
                context_id="test_123",
                timestamp="invalid-timestamp",
                query="test query",
                agent_outputs={},
                user_config={},
                agent_trace={},
                size_bytes=100,
            )

    def test_context_snapshot_negative_size_bytes(self):
        """Test ContextSnapshot with negative size_bytes."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ContextSnapshot(
                context_id="test_123",
                timestamp="2025-01-20T10:00:00+00:00",
                query="test query",
                agent_outputs={},
                user_config={},
                agent_trace={},
                size_bytes=-100,
            )

    def test_context_snapshot_empty_context_id(self):
        """Test ContextSnapshot with empty context_id."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            ContextSnapshot(
                context_id="",
                timestamp="2025-01-20T10:00:00+00:00",
                query="test query",
                agent_outputs={},
                user_config={},
                agent_trace={},
                size_bytes=100,
            )

    def test_context_snapshot_to_dict(self):
        """Test ContextSnapshot to_dict method."""
        snapshot = ContextSnapshot(
            context_id="test_123",
            timestamp="2025-01-20T10:00:00+00:00",
            query="test query",
            agent_outputs={"refiner": "output"},
            retrieved_notes=None,
            user_config={"key": "value"},
            final_synthesis=None,
            agent_trace={},
            size_bytes=100,
        )

        data = snapshot.to_dict()

        assert data["context_id"] == "test_123"
        assert data["query"] == "test query"
        assert data["agent_outputs"] == {"refiner": "output"}
        assert data["retrieved_notes"] is None
        assert data["final_synthesis"] is None

    def test_context_snapshot_from_dict(self):
        """Test ContextSnapshot from_dict class method."""
        data = {
            "context_id": "test_123",
            "timestamp": "2025-01-20T10:00:00+00:00",
            "query": "test query",
            "agent_outputs": {"refiner": "output"},
            "retrieved_notes": ["note1"],
            "user_config": {"key": "value"},
            "final_synthesis": "result",
            "agent_trace": {},
            "size_bytes": 100,
            "compressed": False,
        }

        snapshot = ContextSnapshot.from_dict(data)

        assert snapshot.context_id == "test_123"
        assert snapshot.query == "test query"
        assert snapshot.retrieved_notes == ["note1"]


class TestAgentContextPydanticValidation:
    """Test AgentContext Pydantic validation."""

    def test_agent_context_valid_creation(self):
        """Test creating valid AgentContext."""
        context = AgentContext(query="What is AI?")

        assert context.query == "What is AI?"
        assert context.agent_outputs == {}
        assert context.user_config == {}
        assert context.final_synthesis is None
        assert context.agent_trace == {}
        assert len(context.context_id) == 8
        assert context.current_size >= 0
        assert context.success is True

    def test_agent_context_empty_query_validation(self):
        """Test AgentContext with empty query."""
        with pytest.raises(
            ValidationError, match="Query cannot be empty or just whitespace"
        ):
            AgentContext(query="")

    def test_agent_context_whitespace_query_validation(self):
        """Test AgentContext with whitespace-only query."""
        with pytest.raises(
            ValidationError, match="Query cannot be empty or just whitespace"
        ):
            AgentContext(query="   \n\t  ")

    def test_agent_context_query_trimming(self):
        """Test AgentContext trims whitespace from query."""
        context = AgentContext(query="  What is AI?  ")
        assert context.query == "What is AI?"

    def test_agent_context_negative_current_size(self):
        """Test AgentContext with negative current_size."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            AgentContext(query="test", current_size=-100)

    def test_agent_context_invalid_agent_status(self):
        """Test AgentContext with invalid agent execution status."""
        with pytest.raises(ValidationError, match="Invalid agent status"):
            AgentContext(
                query="test", agent_execution_status={"refiner": "invalid_status"}
            )

    def test_agent_context_valid_agent_statuses(self):
        """Test AgentContext with valid agent execution statuses."""
        context = AgentContext(
            query="test",
            agent_execution_status={
                "refiner": "pending",
                "critic": "running",
                "historian": "completed",
                "synthesis": "failed",
            },
        )

        assert context.agent_execution_status["refiner"] == "pending"
        assert context.agent_execution_status["critic"] == "running"
        assert context.agent_execution_status["historian"] == "completed"
        assert context.agent_execution_status["synthesis"] == "failed"

    def test_agent_context_overlapping_success_failure_sets(self):
        """Test AgentContext model validator for overlapping success/failure sets."""
        with pytest.raises(
            ValidationError, match="Agents cannot be both successful and failed"
        ):
            AgentContext(
                query="test",
                successful_agents={"refiner", "critic"},
                failed_agents={"critic", "historian"},
            )

    def test_agent_context_inconsistent_status_and_success_set(self):
        """Test AgentContext model validator for inconsistent status and success set."""
        with pytest.raises(
            ValidationError, match="is in successful_agents but has status"
        ):
            AgentContext(
                query="test",
                agent_execution_status={"refiner": "running"},
                successful_agents={"refiner"},
            )

    def test_agent_context_inconsistent_status_and_failure_set(self):
        """Test AgentContext model validator for inconsistent status and failure set."""
        with pytest.raises(ValidationError, match="is in failed_agents but has status"):
            AgentContext(
                query="test",
                agent_execution_status={"refiner": "completed"},
                failed_agents={"refiner"},
            )

    def test_agent_context_consistent_execution_state(self):
        """Test AgentContext with consistent execution state."""
        context = AgentContext(
            query="test",
            agent_execution_status={
                "refiner": "completed",
                "critic": "failed",
                "historian": "running",
            },
            successful_agents={"refiner"},
            failed_agents={"critic"},
        )

        assert "refiner" in context.successful_agents
        assert "critic" in context.failed_agents
        assert "historian" not in context.successful_agents
        assert "historian" not in context.failed_agents

    def test_agent_context_long_context_id_validation(self):
        """Test AgentContext context_id length constraint."""
        with pytest.raises(ValidationError, match="at most 50 characters"):
            AgentContext(
                query="test",
                context_id="a" * 51,  # 51 characters, exceeds max_length=50
            )

    def test_agent_context_field_descriptions(self):
        """Test that all fields have proper descriptions."""
        context = AgentContext(query="test")

        # Check that field descriptions exist (this tests that our Field() definitions are correct)
        schema = AgentContext.model_json_schema()
        properties = schema["properties"]

        assert "description" in properties["query"]
        assert "description" in properties["retrieved_notes"]
        assert "description" in properties["agent_outputs"]
        assert "description" in properties["user_config"]
        assert "description" in properties["context_id"]
        assert "description" in properties["successful_agents"]
        assert "description" in properties["failed_agents"]


class TestAgentContextBasicFunctionality:
    """Test basic AgentContext functionality with Pydantic."""

    def test_add_agent_output(self):
        """Test adding agent output."""
        context = AgentContext(query="test")

        context.add_agent_output("refiner", "refined output")

        assert context.agent_outputs["refiner"] == "refined output"
        assert context.current_size > 0

    def test_get_agent_output(self):
        """Test getting agent output."""
        context = AgentContext(query="test")
        context.add_agent_output("refiner", "refined output")

        output = context.get_output("refiner")

        assert output == "refined output"

    def test_get_nonexistent_agent_output(self):
        """Test getting output for non-existent agent."""
        context = AgentContext(query="test")

        output = context.get_output("nonexistent")

        assert output is None

    def test_update_user_config(self):
        """Test updating user configuration."""
        context = AgentContext(query="test")

        context.update_user_config({"key1": "value1", "key2": "value2"})

        assert context.user_config["key1"] == "value1"
        assert context.user_config["key2"] == "value2"

    def test_get_user_config_with_default(self):
        """Test getting user config with default value."""
        context = AgentContext(query="test")
        context.update_user_config({"existing_key": "existing_value"})

        existing_value = context.get_user_config("existing_key")
        default_value = context.get_user_config("nonexistent_key", "default")

        assert existing_value == "existing_value"
        assert default_value == "default"

    def test_set_and_get_final_synthesis(self):
        """Test setting and getting final synthesis."""
        context = AgentContext(query="test")

        context.set_final_synthesis("This is the final synthesis")

        assert context.get_final_synthesis() == "This is the final synthesis"
        assert context.final_synthesis == "This is the final synthesis"

    def test_log_trace(self):
        """Test logging agent trace."""
        context = AgentContext(query="test")

        context.log_trace("refiner", "input data", "output data")

        assert "refiner" in context.agent_trace
        assert len(context.agent_trace["refiner"]) == 1
        trace_entry = context.agent_trace["refiner"][0]
        assert trace_entry["input"] == "input data"
        assert trace_entry["output"] == "output data"
        assert "timestamp" in trace_entry

    def test_context_id_generation(self):
        """Test that context_id is automatically generated."""
        context1 = AgentContext(query="test1")
        context2 = AgentContext(query="test2")

        assert len(context1.context_id) == 8
        assert len(context2.context_id) == 8
        assert context1.context_id != context2.context_id

    def test_clone_context(self):
        """Test cloning context."""
        original = AgentContext(query="original query")
        original.add_agent_output("refiner", "original output")
        original.update_user_config({"key": "value"})

        cloned = original.clone()

        assert cloned.query == "original query"
        assert cloned.agent_outputs["refiner"] == "original output"
        assert cloned.user_config["key"] == "value"
        assert cloned.context_id != original.context_id
        assert "_clone_" in cloned.context_id


class TestAgentContextExecutionStateManagement:
    """Test AgentContext execution state management features."""

    def test_start_agent_execution(self):
        """Test starting agent execution."""
        context = AgentContext(query="test")

        context.start_agent_execution("refiner", "step_123")

        assert context.agent_execution_status["refiner"] == "running"
        assert context.execution_state["refiner_step_id"] == "step_123"
        assert "refiner_start_time" in context.execution_state

    def test_complete_agent_execution_success(self):
        """Test completing agent execution successfully."""
        context = AgentContext(query="test")
        context.start_agent_execution("refiner")

        context.complete_agent_execution("refiner", success=True)

        assert context.agent_execution_status["refiner"] == "completed"
        assert "refiner" in context.successful_agents
        assert "refiner" not in context.failed_agents
        assert context.success is True
        assert "refiner_end_time" in context.execution_state

    def test_complete_agent_execution_failure(self):
        """Test completing agent execution with failure."""
        context = AgentContext(query="test")
        context.start_agent_execution("refiner")

        context.complete_agent_execution("refiner", success=False)

        assert context.agent_execution_status["refiner"] == "failed"
        assert "refiner" in context.failed_agents
        assert "refiner" not in context.successful_agents
        assert context.success is False

    def test_set_and_check_agent_dependencies(self):
        """Test setting and checking agent dependencies."""
        context = AgentContext(query="test")

        context.set_agent_dependencies("critic", ["refiner", "historian"])

        dependencies = context.agent_dependencies["critic"]
        assert dependencies == ["refiner", "historian"]

    def test_can_agent_execute_with_satisfied_dependencies(self):
        """Test checking if agent can execute with satisfied dependencies."""
        context = AgentContext(query="test")
        context.set_agent_dependencies("critic", ["refiner"])
        context.complete_agent_execution("refiner", success=True)

        can_execute = context.can_agent_execute("critic")

        assert can_execute is True

    def test_can_agent_execute_with_unsatisfied_dependencies(self):
        """Test checking if agent can execute with unsatisfied dependencies."""
        context = AgentContext(query="test")
        context.set_agent_dependencies("critic", ["refiner", "historian"])
        context.complete_agent_execution("refiner", success=True)
        # historian hasn't completed successfully

        can_execute = context.can_agent_execute("critic")

        assert can_execute is False

    def test_get_execution_summary(self):
        """Test getting execution summary."""
        context = AgentContext(query="test")
        context.start_agent_execution("refiner")
        context.complete_agent_execution("refiner", success=True)
        context.start_agent_execution("critic")
        context.complete_agent_execution("critic", success=False)
        context.agent_execution_status["historian"] = "pending"

        summary = context.get_execution_summary()

        assert summary["total_agents"] == 3
        assert summary["successful_agents"] == ["refiner"]
        assert summary["failed_agents"] == ["critic"]
        assert summary["pending_agents"] == ["historian"]
        assert summary["overall_success"] is False
        assert summary["context_id"] == context.context_id


class TestAgentContextSnapshotFunctionality:
    """Test AgentContext snapshot and rollback functionality."""

    def test_create_snapshot(self):
        """Test creating context snapshot."""
        context = AgentContext(query="test query")
        context.add_agent_output("refiner", "output")

        snapshot_id = context.create_snapshot("test_label")

        assert len(context.snapshots) == 1
        snapshot = context.snapshots[0]
        assert snapshot.query == "test query"
        assert snapshot.agent_outputs["refiner"] == "output"
        assert snapshot.context_id == context.context_id

    def test_restore_snapshot(self):
        """Test restoring from snapshot."""
        context = AgentContext(query="original query")
        context.add_agent_output("refiner", "original output")

        # Create snapshot
        snapshot_id = context.create_snapshot()

        # Modify context
        context.query = "modified query"
        context.add_agent_output("refiner", "modified output")

        # Restore snapshot
        success = context.restore_snapshot(snapshot_id)

        assert success is True
        assert context.query == "original query"
        assert context.agent_outputs["refiner"] == "original output"

    def test_restore_nonexistent_snapshot(self):
        """Test restoring from non-existent snapshot."""
        context = AgentContext(query="test")

        success = context.restore_snapshot("nonexistent_snapshot_id")

        assert success is False

    def test_list_snapshots(self):
        """Test listing snapshots."""
        context = AgentContext(query="test")
        context.add_agent_output("refiner", "output")

        context.create_snapshot()
        context.create_snapshot()

        snapshots = context.list_snapshots()

        assert len(snapshots) == 2
        for snapshot_info in snapshots:
            assert "timestamp" in snapshot_info
            assert "size_bytes" in snapshot_info
            assert "compressed" in snapshot_info
            assert "agents_present" in snapshot_info

    def test_clear_snapshots(self):
        """Test clearing all snapshots."""
        context = AgentContext(query="test")
        context.create_snapshot()
        context.create_snapshot()

        assert len(context.snapshots) == 2

        context.clear_snapshots()

        assert len(context.snapshots) == 0

    def test_create_execution_snapshot(self):
        """Test creating execution snapshot with extended state."""
        context = AgentContext(query="test")
        context.start_agent_execution("refiner")
        context.complete_agent_execution("refiner", success=True)

        snapshot_id = context.create_execution_snapshot("execution_test")

        assert len(context.snapshots) == 1
        # Check that execution state data was stored
        execution_data_key = f"snapshot_{snapshot_id}_execution_data"
        assert execution_data_key in context.execution_state

    def test_restore_execution_snapshot(self):
        """Test restoring execution snapshot."""
        context = AgentContext(query="test")
        context.start_agent_execution("refiner")
        context.complete_agent_execution("refiner", success=True)

        # Create execution snapshot
        snapshot_id = context.create_execution_snapshot()

        # Modify execution state
        context.complete_agent_execution("critic", success=False)
        assert context.success is False

        # Restore execution snapshot
        success = context.restore_execution_snapshot(snapshot_id)

        assert success is True
        assert "refiner" in context.successful_agents
        assert "critic" not in context.failed_agents
        assert context.success is True


class TestAgentContextAgentIsolation:
    """Test AgentContext agent isolation features."""

    def test_lock_and_unlock_field(self):
        """Test locking and unlocking fields."""
        context = AgentContext(query="test")

        context.lock_field("test_field")
        assert "test_field" in context.locked_fields

        context.unlock_field("test_field")
        assert "test_field" not in context.locked_fields

    def test_add_agent_output_isolated_success(self):
        """Test adding agent output with isolation - success case."""
        context = AgentContext(query="test")

        success = context.add_agent_output_isolated("refiner", "output")

        assert success is True
        assert context.agent_outputs["refiner"] == "output"
        assert "agent_outputs.refiner" in context.agent_mutations["refiner"]

    def test_add_agent_output_isolated_locked_field(self):
        """Test adding agent output with isolation - locked field."""
        context = AgentContext(query="test")
        context.lock_field("agent_outputs.refiner")

        success = context.add_agent_output_isolated("refiner", "output")

        assert success is False
        assert "refiner" not in context.agent_outputs

    def test_add_agent_output_isolated_already_modified(self):
        """Test adding agent output with isolation - field already modified by another agent."""
        context = AgentContext(query="test")

        # First agent modifies the field
        success1 = context.add_agent_output_isolated("refiner", "output1")
        assert success1 is True

        # Same agent tries to modify the same field again (should fail)
        success2 = context.add_agent_output_isolated("refiner", "output2")

        assert success2 is False
        assert context.agent_outputs["refiner"] == "output1"  # Should remain unchanged

    def test_get_agent_mutation_history(self):
        """Test getting agent mutation history."""
        context = AgentContext(query="test")
        context.add_agent_output_isolated("refiner", "output1")
        context.add_agent_output_isolated("critic", "output2")

        history = context.get_agent_mutation_history()

        assert "refiner" in history
        assert "critic" in history
        assert "agent_outputs.refiner" in history["refiner"]
        assert "agent_outputs.critic" in history["critic"]


class TestAgentContextMemoryManagement:
    """Test AgentContext memory management and compression."""

    def test_get_memory_usage(self):
        """Test getting memory usage information."""
        context = AgentContext(query="test query")
        context.add_agent_output("refiner", "some output")
        context.log_trace("refiner", "input", "output")

        memory_usage = context.get_memory_usage()

        assert "total_size_bytes" in memory_usage
        assert "agent_outputs_size" in memory_usage
        assert "agent_trace_size" in memory_usage
        assert "snapshots_count" in memory_usage
        assert "snapshots_total_size" in memory_usage
        assert "retrieved_notes_size" in memory_usage
        assert "context_id" in memory_usage
        assert memory_usage["context_id"] == context.context_id

    def test_optimize_memory(self):
        """Test memory optimization."""
        context = AgentContext(query="test")

        # Create many snapshots
        for i in range(10):
            context.create_snapshot(f"snapshot_{i}")

        assert len(context.snapshots) == 10

        stats = context.optimize_memory()

        # Should keep only 5 most recent snapshots
        assert len(context.snapshots) == 5
        assert stats["snapshots_removed"] == 5
        assert "size_before" in stats
        assert "size_after" in stats

    @patch("cognivault.context.get_config")
    def test_size_limit_enforcement(self, mock_get_config):
        """Test that size limits are enforced."""
        # Mock config to return small size limit
        mock_config = Mock()
        mock_config.testing.max_context_size_bytes = 100  # Very small limit
        mock_get_config.return_value = mock_config

        context = AgentContext(query="test")

        # Add large output that should trigger compression
        large_output = "x" * 1000  # 1000 characters
        context.add_agent_output("refiner", large_output)

        # Should have been compressed/truncated
        assert len(context.agent_outputs["refiner"]) < 1000


class TestAgentContextLangGraphCompatibility:
    """Test AgentContext LangGraph compatibility features."""

    def test_add_execution_edge(self):
        """Test adding execution edge."""
        context = AgentContext(query="test")

        context.add_execution_edge(
            "refiner", "critic", "conditional", "confidence > 0.8", {"meta": "data"}
        )

        assert len(context.execution_edges) == 1
        edge = context.execution_edges[0]
        assert edge["from_agent"] == "refiner"
        assert edge["to_agent"] == "critic"
        assert edge["edge_type"] == "conditional"
        assert edge["condition"] == "confidence > 0.8"
        assert edge["metadata"]["meta"] == "data"

    def test_record_conditional_routing(self):
        """Test recording conditional routing decision."""
        context = AgentContext(query="test")

        context.record_conditional_routing(
            "decision_point_1",
            "confidence > 0.8",
            "critic",
            ["historian", "synthesis"],
            {"decision_meta": "value"},
        )

        assert "decision_point_1" in context.conditional_routing
        routing = context.conditional_routing["decision_point_1"][0]
        assert routing["condition"] == "confidence > 0.8"
        assert routing["chosen_path"] == "critic"
        assert routing["alternative_paths"] == ["historian", "synthesis"]
        assert routing["metadata"]["decision_meta"] == "value"

    def test_set_path_metadata(self):
        """Test setting execution path metadata."""
        context = AgentContext(query="test")

        context.set_path_metadata("execution_mode", "parallel")
        context.set_path_metadata("branch_factor", 3)

        assert context.path_metadata["execution_mode"] == "parallel"
        assert context.path_metadata["branch_factor"] == 3

    def test_get_execution_graph(self):
        """Test getting execution graph representation."""
        context = AgentContext(query="test")
        context.add_agent_output("refiner", "output1")
        context.add_agent_output("critic", "output2")
        context.complete_agent_execution("refiner", success=True)
        context.complete_agent_execution("critic", success=False)
        context.add_execution_edge("refiner", "critic")
        context.record_conditional_routing(
            "test_decision", "test_condition", "chosen", ["alt"]
        )
        context.set_path_metadata("test_key", "test_value")

        graph = context.get_execution_graph()

        assert len(graph["nodes"]) == 2
        assert graph["nodes"][0]["id"] in ["refiner", "critic"]
        assert graph["nodes"][1]["id"] in ["refiner", "critic"]

        # Check execution summary
        summary = graph["execution_summary"]
        assert summary["total_agents"] == 2
        assert summary["successful_agents"] == 1
        assert summary["failed_agents"] == 1
        assert summary["success_rate"] == 0.5

        # Check other graph components
        assert len(graph["edges"]) == 1
        assert "test_decision" in graph["conditional_routing"]
        assert graph["path_metadata"]["test_key"] == "test_value"


class TestAgentContextErrorHandling:
    """Test AgentContext error handling."""

    def test_state_transition_error_on_snapshot_failure(self):
        """Test StateTransitionError on snapshot creation failure."""
        context = AgentContext(query="test")

        # Mock the ContextSnapshot to raise an exception
        with patch(
            "cognivault.context.ContextSnapshot",
            side_effect=Exception("Snapshot failed"),
        ):
            with pytest.raises(StateTransitionError, match="snapshot_creation_failed"):
                context.create_execution_snapshot()

    def test_state_transition_error_on_restore_failure(self):
        """Test StateTransitionError on snapshot restore failure."""
        context = AgentContext(query="test")
        snapshot_id = context.create_execution_snapshot()

        # Corrupt the execution state to cause restore failure
        context.execution_state.clear()

        # Use patch on the module level instead of object level for Pydantic compatibility
        with patch(
            "cognivault.context.AgentContext.restore_snapshot",
            side_effect=Exception("Restore failed"),
        ):
            with pytest.raises(StateTransitionError, match="snapshot_restore_failed"):
                context.restore_execution_snapshot(snapshot_id)


class TestContextCompressionManager:
    """Test ContextCompressionManager functionality."""

    def test_calculate_size(self):
        """Test calculating data size."""
        data = {"key": "value", "number": 123}

        size = ContextCompressionManager.calculate_size(data)

        assert size > 0
        assert isinstance(size, int)

    def test_compress_and_decompress_data(self):
        """Test data compression and decompression."""
        original_data = {"key": "value", "list": [1, 2, 3], "nested": {"inner": "data"}}

        compressed = ContextCompressionManager.compress_data(original_data)
        decompressed = ContextCompressionManager.decompress_data(compressed)

        assert isinstance(compressed, bytes)
        assert decompressed == original_data

    def test_truncate_large_outputs(self):
        """Test truncating large outputs."""
        outputs = {"short": "short text", "long": "x" * 1000, "number": 123}
        max_size = 50

        truncated = ContextCompressionManager.truncate_large_outputs(outputs, max_size)

        assert truncated["short"] == "short text"  # Unchanged
        assert len(truncated["long"]) <= max_size  # Should be within max_size now
        assert "truncated" in truncated["long"]
        assert truncated["number"] == 123  # Non-string unchanged


class TestAgentContextPydanticIntegration:
    """Test AgentContext integration with Pydantic features."""

    def test_model_json_schema(self):
        """Test that AgentContext generates proper JSON schema."""
        schema = AgentContext.model_json_schema()

        assert "properties" in schema
        assert "required" in schema
        assert "query" in schema["required"]

        # Check that field descriptions are present
        properties = schema["properties"]
        assert (
            properties["query"]["description"]
            == "The user's query or question to be processed"
        )
        assert (
            properties["context_id"]["description"]
            == "Unique identifier for this context instance"
        )

    def test_model_dump(self):
        """Test Pydantic model_dump functionality."""
        context = AgentContext(query="test query")
        context.add_agent_output("refiner", "output")

        data = context.model_dump()

        assert data["query"] == "test query"
        assert data["agent_outputs"]["refiner"] == "output"
        assert "context_id" in data
        assert "current_size" in data

    def test_model_dump_exclude(self):
        """Test Pydantic model_dump with field exclusion."""
        context = AgentContext(query="test query")

        data = context.model_dump(exclude={"snapshots", "execution_state"})

        assert "query" in data
        assert "snapshots" not in data
        assert "execution_state" not in data

    def test_model_validate(self):
        """Test Pydantic model validation from dict."""
        data = {
            "query": "test query",
            "agent_outputs": {"refiner": "output"},
            "user_config": {"key": "value"},
            "success": True,
        }

        context = AgentContext.model_validate(data)

        assert context.query == "test query"
        assert context.agent_outputs["refiner"] == "output"
        assert context.current_size > 0  # Will be auto-calculated
        assert context.success is True

    def test_model_copy(self):
        """Test Pydantic model_copy functionality."""
        original = AgentContext(query="original")
        original.add_agent_output("refiner", "output")

        copy = original.model_copy(update={"query": "updated query"})

        assert copy.query == "updated query"
        assert copy.agent_outputs["refiner"] == "output"
        assert copy.context_id != original.context_id  # Should be different instances

    def test_field_info_access(self):
        """Test accessing Pydantic field information."""
        field_info = AgentContext.model_fields

        assert "query" in field_info
        assert "agent_outputs" in field_info
        assert "current_size" in field_info

        # Check that constraints are properly set
        current_size_field = field_info["current_size"]
        assert current_size_field.annotation is int

        context_id_field = field_info["context_id"]
        assert context_id_field.annotation is str
