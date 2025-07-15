"""
Tests for AgentContextStateBridge.

This module tests the bidirectional conversion between AgentContext and LangGraph state,
ensuring all features are preserved and round-trip integrity is maintained.
"""

import pytest
import time
from unittest.mock import patch

from cognivault.context import AgentContext
from cognivault.orchestration.state_bridge import (
    AgentContextStateBridge,
    StateConversionError,
)


class TestAgentContextStateBridge:
    """Test suite for AgentContextStateBridge."""

    def test_to_langgraph_state_basic(self):
        """Test basic conversion from AgentContext to LangGraph state."""
        # Arrange
        context = AgentContext(query="Test query", context_id="test-123")
        context.add_agent_output("refiner", "Refined output")
        context.successful_agents.add("refiner")

        # Act
        state = AgentContextStateBridge.to_langgraph_state(context)

        # Assert
        assert isinstance(state, dict)
        assert state["_context_id"] == "test-123"
        assert state["_query"] == "Test query"
        assert state["_agent_outputs"]["refiner"] == "Refined output"
        assert "refiner" in state["_successful_agents"]
        assert "_cognivault_metadata" in state
        assert state["_cognivault_metadata"]["bridge_version"] == "1.0.0"

    def test_to_langgraph_state_comprehensive(self):
        """Test comprehensive conversion with all AgentContext features."""
        # Arrange
        context = AgentContext(query="Complex query", context_id="complex-456")

        # Add agent outputs
        context.add_agent_output("refiner", "Refined output")
        context.add_agent_output("critic", "Critical analysis")

        # Set agent status
        context.successful_agents.add("refiner")
        context.failed_agents.add("historian")

        # Add execution metadata
        context.execution_state["mode"] = "test"
        context.execution_state["timestamp"] = time.time()

        # Add trace information
        context.agent_trace["refiner"] = [{"event": "start", "timestamp": time.time()}]

        # Add execution status
        context.agent_execution_status["refiner"] = "completed"
        context.agent_execution_status["critic"] = "running"

        # Add execution edges
        context.execution_edges.append({"from": "refiner", "to": "critic"})

        # Add conditional routing
        context.conditional_routing["refiner_decision"] = "success"

        # Add path metadata
        context.path_metadata["execution_path"] = "refiner->critic"

        # Act
        state = AgentContextStateBridge.to_langgraph_state(context)

        # Assert all preserved features
        assert state["_query"] == "Complex query"
        assert state["_context_id"] == "complex-456"
        assert len(state["_agent_outputs"]) == 2
        assert state["_agent_outputs"]["refiner"] == "Refined output"
        assert state["_agent_outputs"]["critic"] == "Critical analysis"
        assert "refiner" in state["_successful_agents"]
        assert "historian" in state["_failed_agents"]
        assert state["_execution_state"]["mode"] == "test"
        assert len(state["_agent_trace"]) == 1
        assert state["_agent_execution_status"]["refiner"] == "completed"
        assert len(state["_execution_edges"]) == 1
        assert state["_conditional_routing"]["refiner_decision"] == "success"
        assert state["_path_metadata"]["execution_path"] == "refiner->critic"

    def test_from_langgraph_state_basic(self):
        """Test basic conversion from LangGraph state to AgentContext."""
        # Arrange
        state = {
            "_context_id": "test-789",
            "_query": "Test query",
            "_current_size": 1024,
            "_successful_agents": ["refiner"],
            "_failed_agents": [],
            "_agent_outputs": {"refiner": "Refined output"},
            "_execution_state": {"mode": "test"},
            "_agent_trace": {},
            "_agent_execution_status": {"refiner": "completed"},
            "_execution_edges": [],
            "_conditional_routing": {},
            "_path_metadata": {},
            "_snapshots": {},
            "_cognivault_metadata": {
                "bridge_version": "1.0.0",
                "conversion_timestamp": time.time(),
            },
        }

        # Act
        context = AgentContextStateBridge.from_langgraph_state(state)

        # Assert
        assert isinstance(context, AgentContext)
        assert context.context_id == "test-789"
        assert context.query == "Test query"
        assert context.current_size == 1024
        assert "refiner" in context.successful_agents
        assert len(context.failed_agents) == 0
        assert context.get_output("refiner") == "Refined output"
        assert context.execution_state["mode"] == "test"

    def test_from_langgraph_state_comprehensive(self):
        """Test comprehensive conversion from LangGraph state to AgentContext."""
        # Arrange
        state = {
            "_context_id": "comprehensive-123",
            "_query": "Complex query",
            "_current_size": 2048,
            "_successful_agents": ["refiner", "critic"],
            "_failed_agents": ["historian"],
            "_agent_outputs": {
                "refiner": "Refined output",
                "critic": "Critical analysis",
            },
            "_execution_state": {
                "mode": "production",
                "timestamp": time.time(),
                "config": {"timeout": 30},
            },
            "_agent_trace": {
                "refiner": [{"event": "start", "timestamp": time.time()}],
                "critic": [{"event": "start", "timestamp": time.time()}],
            },
            "_agent_execution_status": {
                "refiner": "completed",
                "critic": "completed",
                "historian": "failed",
            },
            "_execution_edges": [
                {"from": "refiner", "to": "critic"},
                {"from": "critic", "to": "synthesis"},
            ],
            "_conditional_routing": {
                "refiner_decision": "success",
                "critic_decision": "continue",
            },
            "_path_metadata": {
                "execution_path": "refiner->critic->synthesis",
                "total_time": 5.5,
            },
            "_snapshots": {
                "snapshot-1": {
                    "timestamp": time.time(),
                    "label": "after_refiner",
                    "size": 1024,
                    "metadata": {"stage": "refiner_complete"},
                }
            },
            "_cognivault_metadata": {
                "bridge_version": "1.0.0",
                "conversion_timestamp": time.time(),
            },
        }

        # Act
        context = AgentContextStateBridge.from_langgraph_state(state)

        # Assert all features restored
        assert context.context_id == "comprehensive-123"
        assert context.query == "Complex query"
        assert context.current_size == 2048
        assert context.successful_agents == {"refiner", "critic"}
        assert context.failed_agents == {"historian"}
        assert len(context.agent_outputs) == 2
        assert context.get_output("refiner") == "Refined output"
        assert context.get_output("critic") == "Critical analysis"
        assert context.execution_state["mode"] == "production"
        assert context.execution_state["config"]["timeout"] == 30
        assert len(context.agent_trace) == 2
        assert context.agent_execution_status["refiner"] == "completed"
        assert context.agent_execution_status["historian"] == "failed"
        assert len(context.execution_edges) == 2
        assert context.conditional_routing["refiner_decision"] == "success"
        assert context.path_metadata["execution_path"] == "refiner->critic->synthesis"
        assert context.path_metadata["total_time"] == 5.5

    def test_validate_round_trip_success(self):
        """Test successful round trip validation."""
        # Arrange
        context = AgentContext(query="Round trip test", context_id="roundtrip-123")
        context.add_agent_output("refiner", "Refined output")
        context.add_agent_output("critic", "Critical analysis")
        context.successful_agents.add("refiner")
        context.successful_agents.add("critic")
        context.execution_state["mode"] = "test"
        context.agent_trace["refiner"] = [{"event": "start"}]
        context.agent_execution_status["refiner"] = "completed"
        context.execution_edges.append({"from": "refiner", "to": "critic"})
        context.conditional_routing["decision"] = "success"
        context.path_metadata["path"] = "refiner->critic"

        # Act
        result = AgentContextStateBridge.validate_round_trip(context)

        # Assert
        assert result is True

    def test_validate_round_trip_failure(self):
        """Test round trip validation with data corruption."""
        # Arrange
        context = AgentContext(query="Test query", context_id="fail-123")
        context.add_agent_output("refiner", "Original output")

        # Act with mocked conversion that corrupts data
        with patch.object(AgentContextStateBridge, "from_langgraph_state") as mock_from:
            # Mock returns a different context
            corrupted_context = AgentContext(
                query="Different query", context_id="different-123"
            )
            mock_from.return_value = corrupted_context

            result = AgentContextStateBridge.validate_round_trip(context)

        # Assert
        assert result is False

    def test_state_conversion_error_handling(self):
        """Test error handling during state conversion."""
        # Arrange - create a context that will cause serialization issues
        context = AgentContext(query="Test query", context_id="error-123")

        # Act with mocked serialization failure
        with patch("cognivault.orchestration.state_bridge.dict") as mock_dict:
            mock_dict.side_effect = Exception("Serialization failed")

            with pytest.raises(StateConversionError) as exc_info:
                AgentContextStateBridge.to_langgraph_state(context)

        # Assert
        assert "AgentContext to LangGraph state conversion failed" in str(
            exc_info.value
        )

    def test_invalid_state_dict_validation(self):
        """Test validation of invalid state dictionaries."""
        # Arrange - missing required keys
        invalid_state = {
            "_context_id": "test-123",
            "_query": "Test query",
            # Missing other required keys
        }

        # Act & Assert
        with pytest.raises(StateConversionError) as exc_info:
            AgentContextStateBridge.from_langgraph_state(invalid_state)

        assert "Missing required keys in state dict" in str(exc_info.value)

    def test_invalid_metadata_validation(self):
        """Test validation of invalid metadata in state dict."""
        # Arrange - invalid metadata format
        state = {
            "_context_id": "test-123",
            "_query": "Test query",
            "_current_size": 1024,
            "_successful_agents": [],
            "_failed_agents": [],
            "_agent_outputs": {},
            "_execution_state": {},
            "_agent_trace": {},
            "_agent_execution_status": {},
            "_execution_edges": [],
            "_conditional_routing": {},
            "_path_metadata": {},
            "_snapshots": {},
            "_cognivault_metadata": "invalid",  # Should be dict
        }

        # Act & Assert
        with pytest.raises(StateConversionError) as exc_info:
            AgentContextStateBridge.from_langgraph_state(state)

        assert "Invalid metadata format in state dict" in str(exc_info.value)

    def test_missing_bridge_version_validation(self):
        """Test validation when bridge version is missing."""
        # Arrange
        state = {
            "_context_id": "test-123",
            "_query": "Test query",
            "_current_size": 1024,
            "_successful_agents": [],
            "_failed_agents": [],
            "_agent_outputs": {},
            "_execution_state": {},
            "_agent_trace": {},
            "_agent_execution_status": {},
            "_execution_edges": [],
            "_conditional_routing": {},
            "_path_metadata": {},
            "_snapshots": {},
            "_cognivault_metadata": {},  # Missing bridge_version
        }

        # Act & Assert
        with pytest.raises(StateConversionError) as exc_info:
            AgentContextStateBridge.from_langgraph_state(state)

        assert "Missing bridge version in metadata" in str(exc_info.value)

    def test_serialize_snapshots_with_data(self):
        """Test snapshot serialization with actual data."""
        # Arrange
        context = AgentContext(query="Test query", context_id="snap-123")

        # Mock snapshots if the feature exists
        if hasattr(context, "_snapshots"):
            context._snapshots = {
                "snap-1": {
                    "timestamp": 1234567890,
                    "label": "test_snapshot",
                    "size": 1024,
                    "metadata": {"stage": "after_refiner"},
                }
            }

        # Act
        snapshots = AgentContextStateBridge._serialize_snapshots(context)

        # Assert
        if hasattr(context, "_snapshots") and context._snapshots:
            assert "snap-1" in snapshots
            assert snapshots["snap-1"]["timestamp"] == 1234567890
            assert snapshots["snap-1"]["label"] == "test_snapshot"
            assert snapshots["snap-1"]["size"] == 1024
            assert snapshots["snap-1"]["metadata"]["stage"] == "after_refiner"
        else:
            # If snapshots feature doesn't exist, should return empty dict
            assert snapshots == {}

    def test_serialize_snapshots_error_handling(self):
        """Test snapshot serialization error handling."""
        # Arrange
        context = AgentContext(query="Test query", context_id="snap-error-123")

        # Act with mocked error
        with patch("cognivault.orchestration.state_bridge.hasattr") as mock_hasattr:
            mock_hasattr.side_effect = Exception("Serialization error")

            snapshots = AgentContextStateBridge._serialize_snapshots(context)

        # Assert - should return empty dict on error
        assert snapshots == {}

    def test_get_state_summary(self):
        """Test state summary generation."""
        # Arrange
        state = {
            "_context_id": "summary-123",
            "_query": "Test query for summary",
            "_current_size": 2048,
            "_successful_agents": ["refiner", "critic"],
            "_failed_agents": ["historian"],
            "_agent_outputs": {"refiner": "output1", "critic": "output2"},
            "_execution_edges": [{"from": "refiner", "to": "critic"}],
            "_snapshots": {"snap-1": {}, "snap-2": {}},
            "_cognivault_metadata": {
                "bridge_version": "1.0.0",
                "conversion_timestamp": time.time(),
            },
        }

        # Act
        summary = AgentContextStateBridge.get_state_summary(state)

        # Assert
        assert summary["context_id"] == "summary-123"
        assert summary["query_length"] == len("Test query for summary")
        assert summary["current_size"] == 2048
        assert summary["agent_count"] == 2
        assert summary["successful_agents"] == 2
        assert summary["failed_agents"] == 1
        assert summary["execution_edges"] == 1
        assert summary["snapshots"] == 2
        assert summary["metadata"]["bridge_version"] == "1.0.0"

    def test_get_state_summary_error_handling(self):
        """Test state summary error handling."""
        # Arrange - invalid state
        invalid_state = None

        # Act
        summary = AgentContextStateBridge.get_state_summary(invalid_state)

        # Assert
        assert "error" in summary
        assert isinstance(summary["error"], str)

    def test_reserved_keys_constant(self):
        """Test that reserved keys are properly defined."""
        # Assert
        assert isinstance(AgentContextStateBridge.RESERVED_KEYS, set)
        assert "_cognivault_metadata" in AgentContextStateBridge.RESERVED_KEYS
        assert "_agent_outputs" in AgentContextStateBridge.RESERVED_KEYS
        assert "_context_id" in AgentContextStateBridge.RESERVED_KEYS
        assert (
            len(AgentContextStateBridge.RESERVED_KEYS) > 10
        )  # Should have many reserved keys

    def test_edge_case_empty_context(self):
        """Test conversion with minimal/empty context."""
        # Arrange
        context = AgentContext(query="", context_id="empty-123")

        # Act
        state = AgentContextStateBridge.to_langgraph_state(context)
        restored_context = AgentContextStateBridge.from_langgraph_state(state)

        # Assert
        assert restored_context.context_id == "empty-123"
        assert restored_context.query == ""
        assert len(restored_context.agent_outputs) == 0
        assert len(restored_context.successful_agents) == 0
        assert len(restored_context.failed_agents) == 0

    def test_edge_case_large_context(self):
        """Test conversion with large context data."""
        # Arrange
        context = AgentContext(query="Large test query", context_id="large-123")

        # Add many agent outputs
        for i in range(100):
            context.add_agent_output(f"agent_{i}", f"Large output {i}" * 100)
            context.successful_agents.add(f"agent_{i}")

        # Act
        state = AgentContextStateBridge.to_langgraph_state(context)
        restored_context = AgentContextStateBridge.from_langgraph_state(state)

        # Assert
        assert len(restored_context.agent_outputs) == 100
        assert len(restored_context.successful_agents) == 100
        assert restored_context.get_output("agent_0") == "Large output 0" * 100
        assert restored_context.get_output("agent_99") == "Large output 99" * 100

    def test_state_bridge_thread_safety(self):
        """Test that state bridge operations are thread-safe."""
        import threading

        # Arrange
        contexts = []
        for i in range(10):
            context = AgentContext(query=f"Query {i}", context_id=f"thread-{i}")
            context.add_agent_output("refiner", f"Output {i}")
            contexts.append(context)

        results = []
        errors = []

        def convert_context(ctx):
            try:
                state = AgentContextStateBridge.to_langgraph_state(ctx)
                restored = AgentContextStateBridge.from_langgraph_state(state)
                results.append(restored)
            except Exception as e:
                errors.append(e)

        # Act
        threads = []
        for context in contexts:
            thread = threading.Thread(target=convert_context, args=(context,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Assert
        assert len(errors) == 0
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.query == f"Query {i}"
            assert result.context_id == f"thread-{i}"
