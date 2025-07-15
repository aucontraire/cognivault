"""
Integration tests for LangGraph checkpointing and memory management.

This module provides comprehensive integration tests that verify the
interaction between memory management, error policies, and real
LangGraph orchestration in various scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from cognivault.langraph.memory_manager import (
    CogniVaultMemoryManager,
    CheckpointConfig,
)
from cognivault.langraph.error_policies import (
    get_error_policy_manager,
)
from cognivault.langraph.orchestrator import LangGraphOrchestrator
from cognivault.langraph.state_schemas import create_initial_state
from cognivault.context import AgentContext


class TestMemoryManagerIntegration:
    """Test memory manager integration with real orchestrator."""

    @pytest.fixture
    def enabled_memory_manager(self):
        """Fixture for enabled memory manager."""
        config = CheckpointConfig(
            enabled=True,
            thread_id="test_integration",
            max_checkpoints_per_thread=5,
            checkpoint_ttl_hours=1,
        )
        with patch(
            "cognivault.langraph.memory_manager.MemorySaver"
        ) as mock_memory_saver:
            mock_memory_saver.return_value = Mock()
            yield CogniVaultMemoryManager(config)

    @pytest.fixture
    def disabled_memory_manager(self):
        """Fixture for disabled memory manager."""
        config = CheckpointConfig(enabled=False)
        return CogniVaultMemoryManager(config)

    @pytest.fixture
    def mock_orchestrator_with_checkpoints(self, enabled_memory_manager):
        """Fixture for orchestrator with checkpointing enabled."""
        with patch.multiple(
            "cognivault.langraph.orchestrator",
            get_agent_registry=Mock(),
            get_logger=Mock(return_value=Mock()),
        ):
            orchestrator = LangGraphOrchestrator(
                agents_to_run=["refiner", "synthesis"],
                enable_checkpoints=True,
                thread_id="test_orchestrator",
                memory_manager=enabled_memory_manager,
            )

            # Mock the compiled graph
            mock_compiled_graph = AsyncMock()
            orchestrator._compiled_graph = mock_compiled_graph

            yield orchestrator, mock_compiled_graph

    @pytest.fixture
    def mock_orchestrator_without_checkpoints(self, disabled_memory_manager):
        """Fixture for orchestrator with checkpointing disabled."""
        with patch.multiple(
            "cognivault.langraph.orchestrator",
            get_agent_registry=Mock(),
            get_logger=Mock(return_value=Mock()),
        ):
            orchestrator = LangGraphOrchestrator(
                agents_to_run=["refiner", "synthesis"],
                enable_checkpoints=False,
                memory_manager=disabled_memory_manager,
            )

            # Mock the compiled graph
            mock_compiled_graph = AsyncMock()
            orchestrator._compiled_graph = mock_compiled_graph

            yield orchestrator, mock_compiled_graph

    @pytest.mark.asyncio
    async def test_orchestrator_checkpoints_enabled_flow(
        self, mock_orchestrator_with_checkpoints
    ):
        """Test orchestrator execution flow with checkpoints enabled."""
        orchestrator, mock_compiled_graph = mock_orchestrator_with_checkpoints

        # Mock successful execution
        final_state = {
            "query": "test query",
            "execution_id": "test_123",
            "successful_agents": ["refiner", "synthesis"],
            "failed_agents": [],
            "errors": [],
            "refiner": {
                "refined_question": "refined test query",
                "topics": ["test"],
                "confidence": 0.9,
            },
            "synthesis": {
                "final_analysis": "test analysis",
                "key_insights": ["insight1"],
                "themes_identified": ["theme1"],
            },
            "execution_metadata": {},
        }
        mock_compiled_graph.ainvoke.return_value = final_state

        # Execute
        result = await orchestrator.run("test query")

        # Verify execution
        assert isinstance(result, AgentContext)
        assert result.query == "test query"
        assert "refiner" in result.agent_outputs
        assert "synthesis" in result.agent_outputs

        # Verify checkpointing was enabled
        assert result.execution_state["checkpoints_enabled"] is True
        assert "thread_id" in result.execution_state

        # Verify memory manager was used
        assert orchestrator.memory_manager.is_enabled()

        # Mock compiled graph should have been called with thread config
        mock_compiled_graph.ainvoke.assert_called_once()
        call_args = mock_compiled_graph.ainvoke.call_args
        assert "config" in call_args[1]
        assert "configurable" in call_args[1]["config"]
        assert "thread_id" in call_args[1]["config"]["configurable"]

    @pytest.mark.asyncio
    async def test_orchestrator_checkpoints_disabled_flow(
        self, mock_orchestrator_without_checkpoints
    ):
        """Test orchestrator execution flow with checkpoints disabled."""
        orchestrator, mock_compiled_graph = mock_orchestrator_without_checkpoints

        # Mock successful execution
        final_state = {
            "query": "test query",
            "execution_id": "test_123",
            "successful_agents": ["refiner", "synthesis"],
            "failed_agents": [],
            "errors": [],
            "refiner": {
                "refined_question": "refined test query",
                "topics": ["test"],
                "confidence": 0.9,
            },
            "synthesis": {
                "final_analysis": "test analysis",
                "key_insights": ["insight1"],
                "themes_identified": ["theme1"],
            },
            "execution_metadata": {},
        }
        mock_compiled_graph.ainvoke.return_value = final_state

        # Execute
        result = await orchestrator.run("test query")

        # Verify execution
        assert isinstance(result, AgentContext)
        assert result.query == "test query"

        # Verify checkpointing was disabled
        assert result.execution_state["checkpoints_enabled"] is False

        # Verify memory manager is disabled
        assert not orchestrator.memory_manager.is_enabled()

    @pytest.mark.asyncio
    async def test_checkpoint_creation_during_execution(
        self, mock_orchestrator_with_checkpoints
    ):
        """Test that checkpoints are created during execution."""
        orchestrator, mock_compiled_graph = mock_orchestrator_with_checkpoints

        # Mock successful execution
        final_state = {
            "query": "test query",
            "execution_id": "test_123",
            "successful_agents": ["refiner", "synthesis"],
            "failed_agents": [],
            "errors": [],
            "refiner": {
                "refined_question": "refined test query",
                "topics": ["test"],
                "confidence": 0.9,
            },
            "synthesis": {
                "final_analysis": "test analysis",
                "key_insights": ["insight1"],
                "themes_identified": ["theme1"],
            },
            "execution_metadata": {},
        }
        mock_compiled_graph.ainvoke.return_value = final_state

        # Track checkpoint creation
        initial_checkpoint_count = len(
            orchestrator.memory_manager.checkpoints.get("test_integration", [])
        )

        # Execute
        await orchestrator.run("test query")

        # Verify checkpoints were created
        thread_checkpoints = orchestrator.memory_manager.checkpoints.get(
            "test_integration", []
        )
        assert len(thread_checkpoints) > initial_checkpoint_count

        # Should have at least initialization and completion checkpoints
        checkpoint_steps = [cp.agent_step for cp in thread_checkpoints]
        assert "initialization" in checkpoint_steps
        assert "completion" in checkpoint_steps

    @pytest.mark.asyncio
    async def test_rollback_functionality(self, mock_orchestrator_with_checkpoints):
        """Test rollback functionality in orchestrator."""
        orchestrator, mock_compiled_graph = mock_orchestrator_with_checkpoints

        # Create some checkpoints first
        test_state = create_initial_state("test query", "test_exec")
        checkpoint_id = orchestrator.memory_manager.create_checkpoint(
            thread_id="test_orchestrator", state=test_state, agent_step="test_step"
        )

        # Mock MemorySaver to return the state
        mock_checkpoint_tuple = Mock()
        mock_checkpoint_tuple.checkpoint = {"channel_values": test_state}
        orchestrator.memory_manager.memory_saver.get_tuple.return_value = (
            mock_checkpoint_tuple
        )

        # Test rollback
        result = await orchestrator.rollback_to_checkpoint()

        assert result is not None
        assert isinstance(result, AgentContext)
        assert result.execution_state["rollback_performed"] is True

    @pytest.mark.asyncio
    async def test_rollback_with_no_checkpoints(
        self, mock_orchestrator_with_checkpoints
    ):
        """Test rollback when no checkpoints exist."""
        orchestrator, _ = mock_orchestrator_with_checkpoints

        # Try rollback with no checkpoints
        result = await orchestrator.rollback_to_checkpoint(
            thread_id="nonexistent_thread"
        )

        assert result is None

    def test_checkpoint_history_retrieval(self, mock_orchestrator_with_checkpoints):
        """Test checkpoint history retrieval."""
        orchestrator, _ = mock_orchestrator_with_checkpoints

        # Create some checkpoints
        test_state = create_initial_state("test query", "test_exec")
        for i in range(3):
            orchestrator.memory_manager.create_checkpoint(
                thread_id="test_orchestrator", state=test_state, agent_step=f"step_{i}"
            )

        # Get history
        history = orchestrator.get_checkpoint_history()

        assert len(history) == 3
        assert all("checkpoint_id" in item for item in history)
        assert all("timestamp" in item for item in history)
        assert all("agent_step" in item for item in history)

    def test_memory_statistics_collection(self, mock_orchestrator_with_checkpoints):
        """Test memory statistics collection."""
        orchestrator, _ = mock_orchestrator_with_checkpoints

        # Create some checkpoints
        test_state = create_initial_state("test query", "test_exec")
        orchestrator.memory_manager.create_checkpoint(
            thread_id="test_orchestrator", state=test_state, agent_step="test_step"
        )

        # Get statistics
        stats = orchestrator.get_memory_statistics()

        assert "enabled" in stats
        assert "total_threads" in stats
        assert "total_checkpoints" in stats
        assert "orchestrator_type" in stats
        assert "checkpointing_enabled" in stats

        assert stats["enabled"] is True
        assert stats["checkpointing_enabled"] is True
        assert stats["total_checkpoints"] >= 1

    def test_cleanup_expired_checkpoints(self, mock_orchestrator_with_checkpoints):
        """Test cleanup of expired checkpoints."""
        orchestrator, _ = mock_orchestrator_with_checkpoints

        # Create checkpoint
        test_state = create_initial_state("test query", "test_exec")
        orchestrator.memory_manager.create_checkpoint(
            thread_id="test_orchestrator", state=test_state, agent_step="test_step"
        )

        # Run cleanup
        removed_count = orchestrator.cleanup_expired_checkpoints()

        # Should be non-negative (actual removal depends on TTL and timing)
        assert removed_count >= 0


class TestErrorPolicyIntegration:
    """Test error policy integration with memory management."""

    @pytest.fixture
    def policy_manager(self):
        """Fixture for error policy manager."""
        return get_error_policy_manager()

    def test_error_policy_manager_singleton(self, policy_manager):
        """Test that error policy manager is singleton."""
        manager2 = get_error_policy_manager()
        assert policy_manager is manager2

    def test_circuit_breaker_state_persistence(self, policy_manager):
        """Test circuit breaker state persistence across calls."""
        # Get circuit breaker for historian
        circuit_breaker = policy_manager.get_circuit_breaker("historian")

        if circuit_breaker:
            initial_failure_count = circuit_breaker.failure_count

            # Record failure
            circuit_breaker.record_failure()

            # Get circuit breaker again
            circuit_breaker2 = policy_manager.get_circuit_breaker("historian")

            # Should be same instance with updated state
            assert circuit_breaker is circuit_breaker2
            assert circuit_breaker2.failure_count == initial_failure_count + 1

    def test_error_statistics_integration(self, policy_manager):
        """Test error statistics integration."""
        from cognivault.langraph.error_policies import get_error_statistics

        stats = get_error_statistics()

        # Should include default policies
        assert stats["policies_configured"] >= 4

        # Check for historian circuit breaker
        if "historian" in stats["circuit_breaker_states"]:
            historian_stats = stats["circuit_breaker_states"]["historian"]
            assert "state" in historian_stats
            assert "failure_count" in historian_stats

    @patch("cognivault.langraph.error_policies.get_error_policy_manager")
    def test_custom_error_policy_in_orchestrator(self, mock_get_manager):
        """Test custom error policy usage in orchestrator context."""
        # Create mock policy manager with custom policies
        mock_manager = Mock()
        mock_policy = Mock()
        mock_policy.timeout_seconds = 5.0
        mock_manager.get_policy.return_value = mock_policy
        mock_get_manager.return_value = mock_manager

        # Import function that uses the policy manager
        from cognivault.langraph.error_policies import timeout_policy

        # The decorator should use our mocked manager
        @timeout_policy("test_node")
        async def test_func():
            return "success"

        # Execute the decorated function to trigger policy manager call

        asyncio.run(test_func())

        # Verify the manager was called
        mock_manager.get_policy.assert_called_with("test_node")


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def full_orchestrator_setup(self):
        """Fixture for full orchestrator setup with mocked dependencies."""
        with (
            patch.multiple(
                "cognivault.langraph.orchestrator",
                get_agent_registry=Mock(),
                get_logger=Mock(return_value=Mock()),
            ),
            patch(
                "cognivault.langraph.memory_manager.MemorySaver"
            ) as mock_memory_saver,
        ):
            mock_memory_saver.return_value = Mock()

            # Create orchestrator with checkpointing
            orchestrator = LangGraphOrchestrator(
                agents_to_run=["refiner", "critic", "historian", "synthesis"],
                enable_checkpoints=True,
                thread_id="e2e_test",
            )

            # Mock compiled graph
            mock_compiled_graph = AsyncMock()
            orchestrator._compiled_graph = mock_compiled_graph

            yield orchestrator, mock_compiled_graph

    @pytest.mark.asyncio
    async def test_full_execution_with_checkpointing(self, full_orchestrator_setup):
        """Test full execution flow with checkpointing and error handling."""
        orchestrator, mock_compiled_graph = full_orchestrator_setup

        # Mock successful execution with all agents
        final_state = {
            "query": "comprehensive test query",
            "execution_id": "e2e_test_123",
            "successful_agents": ["refiner", "critic", "historian", "synthesis"],
            "failed_agents": [],
            "errors": [],
            "refiner": {
                "refined_question": "refined comprehensive test query",
                "topics": ["test", "comprehensive"],
                "confidence": 0.95,
            },
            "critic": {
                "critique": "Good question structure",
                "suggestions": ["Consider adding examples"],
                "severity": "low",
            },
            "historian": {
                "historical_summary": "Previous similar queries found",
                "retrieved_notes": ["note1", "note2"],
                "search_strategy": "semantic_search",
                "topics_found": ["test"],
                "confidence": 0.8,
            },
            "synthesis": {
                "final_analysis": "Comprehensive analysis of test query",
                "key_insights": ["insight1", "insight2"],
                "themes_identified": ["testing", "analysis"],
            },
            "execution_metadata": {
                "total_time_ms": 1500.0,
                "agent_execution_order": [
                    "refiner",
                    "critic",
                    "historian",
                    "synthesis",
                ],
            },
        }
        mock_compiled_graph.ainvoke.return_value = final_state

        # Execute the full pipeline
        result = await orchestrator.run("comprehensive test query")

        # Verify comprehensive result
        assert isinstance(result, AgentContext)
        assert result.query == "comprehensive test query"

        # Verify all agents executed
        expected_agents = ["refiner", "critic", "historian", "synthesis"]
        for agent in expected_agents:
            assert agent in result.agent_outputs
            assert agent in result.successful_agents

        # Verify execution metadata
        assert result.execution_state["orchestrator_type"] == "langgraph-real"
        assert result.execution_state["phase"] == "phase2_1"
        assert result.execution_state["checkpoints_enabled"] is True
        assert result.execution_state["successful_agents_count"] == 4
        assert result.execution_state["failed_agents_count"] == 0

        # Verify checkpoints were created
        checkpoints = orchestrator.memory_manager.get_checkpoint_history("e2e_test")
        assert len(checkpoints) >= 2  # At least init and completion

        checkpoint_steps = [cp.agent_step for cp in checkpoints]
        assert "initialization" in checkpoint_steps
        assert "completion" in checkpoint_steps

    @pytest.mark.asyncio
    async def test_execution_with_partial_failure(self, full_orchestrator_setup):
        """Test execution with partial agent failure."""
        orchestrator, mock_compiled_graph = full_orchestrator_setup

        # Mock execution with some failures
        final_state = {
            "query": "test query with failures",
            "execution_id": "partial_fail_123",
            "successful_agents": ["refiner", "synthesis"],
            "failed_agents": ["critic", "historian"],
            "errors": [
                {"agent": "critic", "error": "Timeout error", "type": "TimeoutError"},
                {
                    "agent": "historian",
                    "error": "Connection failed",
                    "type": "ConnectionError",
                },
            ],
            "refiner": {
                "refined_question": "refined test query",
                "topics": ["test"],
                "confidence": 0.9,
            },
            "synthesis": {
                "final_analysis": "Analysis based on available data",
                "key_insights": ["limited insights due to failures"],
                "themes_identified": ["resilience"],
            },
            "execution_metadata": {"total_time_ms": 2000.0, "partial_execution": True},
        }
        mock_compiled_graph.ainvoke.return_value = final_state

        # Execute
        result = await orchestrator.run("test query with failures")

        # Verify partial success
        assert isinstance(result, AgentContext)
        assert result.query == "test query with failures"

        # Verify successful agents
        assert "refiner" in result.agent_outputs
        assert "synthesis" in result.agent_outputs

        # Verify execution state reflects failures
        assert result.execution_state["successful_agents_count"] == 2
        assert result.execution_state["failed_agents_count"] == 2
        assert result.execution_state["errors_count"] == 2

        # Verify checkpoints still created despite failures
        checkpoints = orchestrator.memory_manager.get_checkpoint_history("e2e_test")
        assert len(checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_execution_statistics_tracking(self, full_orchestrator_setup):
        """Test that execution statistics are properly tracked."""
        orchestrator, mock_compiled_graph = full_orchestrator_setup

        # Mock successful execution
        final_state = {
            "query": "stats test query",
            "execution_id": "stats_test_123",
            "successful_agents": ["refiner", "synthesis"],
            "failed_agents": [],
            "errors": [],
            "refiner": {"refined_question": "refined", "topics": [], "confidence": 0.8},
            "synthesis": {
                "final_analysis": "analysis",
                "key_insights": [],
                "themes_identified": [],
            },
            "execution_metadata": {},
        }
        mock_compiled_graph.ainvoke.return_value = final_state

        # Get initial statistics
        initial_stats = orchestrator.get_execution_statistics()
        initial_total = initial_stats["total_executions"]
        initial_successful = initial_stats["successful_executions"]

        # Execute
        await orchestrator.run("stats test query")

        # Get updated statistics
        updated_stats = orchestrator.get_execution_statistics()

        # Verify statistics were updated
        assert updated_stats["total_executions"] == initial_total + 1
        assert updated_stats["successful_executions"] == initial_successful + 1
        assert updated_stats["orchestrator_type"] == "langgraph-real"
        assert (
            updated_stats["implementation_status"]
            == "phase2_production_with_graph_factory"
        )

    def test_memory_and_error_policy_interaction(self, full_orchestrator_setup):
        """Test interaction between memory management and error policies."""
        orchestrator, _ = full_orchestrator_setup

        # Get both systems
        memory_manager = orchestrator.memory_manager
        error_policy_manager = get_error_policy_manager()

        # Verify both are available
        assert memory_manager.is_enabled()
        assert error_policy_manager is not None

        # Verify error policies exist for agents
        for agent in orchestrator.agents_to_run:
            policy = error_policy_manager.get_policy(agent)
            assert policy is not None

        # Verify memory statistics include error policy information
        memory_stats = orchestrator.get_memory_statistics()
        assert "orchestrator_type" in memory_stats
        assert "checkpointing_enabled" in memory_stats


if __name__ == "__main__":
    pytest.main([__file__])
