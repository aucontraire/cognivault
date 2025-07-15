"""
Detailed tests for historian_node() function behavior and edge cases.

This test suite provides comprehensive coverage of the historian_node()
function with focus on:
- Circuit breaker behavior
- Node metrics collection
- Error handling scenarios
- Performance characteristics
- State management edge cases
"""

import pytest
import asyncio
import time
from typing import List
from unittest.mock import patch

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.orchestration.state_schemas import (
    CogniVaultState,
    RefinerOutput,
    create_initial_state,
    set_agent_output,
)
from cognivault.orchestration.node_wrappers import (
    historian_node,
    NodeExecutionError,
    circuit_breaker,
    node_metrics,
)


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset the circuit breaker state before each test."""
    # Reset the global historian_node circuit breaker state
    if hasattr(historian_node, "_failure_count"):
        historian_node._failure_count = 0
    if hasattr(historian_node, "_last_failure_time"):
        historian_node._last_failure_time = None
    if hasattr(historian_node, "_circuit_open"):
        historian_node._circuit_open = False
    yield
    # Reset again after test
    if hasattr(historian_node, "_failure_count"):
        historian_node._failure_count = 0
    if hasattr(historian_node, "_last_failure_time"):
        historian_node._last_failure_time = None
    if hasattr(historian_node, "_circuit_open"):
        historian_node._circuit_open = False


class ControlledHistorianAgent(BaseAgent):
    """Historian agent with controlled behavior for testing."""

    def __init__(self, name: str = "Historian", behavior: str = "success"):
        # Disable retries for controlled failure testing
        from cognivault.agents.base_agent import RetryConfig

        retry_config = (
            RetryConfig(max_retries=0)
            if behavior in ["failure", "timeout_error", "memory_error"]
            else None
        )
        super().__init__(name, retry_config=retry_config)
        self.behavior = behavior
        self.execution_count = 0
        self.execution_times: List[float] = []

    async def run(self, context: AgentContext) -> AgentContext:
        """Controlled execution based on behavior setting."""
        start_time = time.time()
        self.execution_count += 1

        if self.behavior == "timeout":
            await asyncio.sleep(10)  # Simulate timeout
        elif self.behavior == "slow":
            await asyncio.sleep(0.1)  # Simulate slow execution
        elif self.behavior == "failure":
            raise Exception("Controlled failure for testing")
        elif self.behavior == "memory_error":
            raise MemoryError("Out of memory during search")
        elif self.behavior == "timeout_error":
            raise TimeoutError("Search timed out")
        elif self.behavior == "partial_failure":
            # Simulate partial failure - some data but with issues, but should succeed
            context.add_agent_output(self.name, "Partial historical context")
            context.execution_state["search_results_count"] = 5
            context.execution_state["filtered_results_count"] = (
                0  # No results after filtering
            )
            # Don't fail - just return partial data (the test should handle this scenario)
        elif self.behavior == "no_llm":
            # Simulate execution without LLM
            context.add_agent_output(self.name, "Basic historical context without LLM")
            context.execution_state.update(
                {
                    "search_results_count": 8,
                    "filtered_results_count": 8,
                    "search_strategy": "keyword",
                    "topics_found": ["basic_topic"],
                    "confidence": 0.6,
                    "llm_analysis_used": False,
                    "historian_metadata": {"fallback_used": True},
                }
            )
        elif self.behavior == "success":
            # Normal successful execution
            context.add_agent_output(
                self.name, f"Historical context for: {context.query}"
            )
            context.retrieved_notes = ["/notes/test1.md", "/notes/test2.md"]
            context.execution_state.update(
                {
                    "search_results_count": 12,
                    "filtered_results_count": 6,
                    "search_strategy": "hybrid",
                    "topics_found": ["topic1", "topic2"],
                    "confidence": 0.85,
                    "llm_analysis_used": True,
                    "historian_metadata": {"search_time_ms": 200},
                }
            )

        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)

        return context


class TestHistorianNodeCircuitBreaker:
    """Test circuit breaker behavior for historian node."""

    @pytest.fixture
    def initial_state(self) -> CogniVaultState:
        """Create initial state with refiner output."""
        state = create_initial_state("Test query", "exec-123")
        refiner_output: RefinerOutput = {
            "refined_question": "Test refined query",
            "topics": ["test"],
            "confidence": 0.8,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        return set_agent_output(state, "refiner", refiner_output)

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, initial_state):
        """Test that circuit breaker opens after multiple failures."""
        failing_agent = ControlledHistorianAgent(behavior="failure")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=failing_agent,
        ):
            # First failure
            with pytest.raises(NodeExecutionError):
                await historian_node(initial_state)

            # Second failure
            with pytest.raises(NodeExecutionError):
                await historian_node(initial_state)

            # Third failure - should open circuit breaker
            with pytest.raises(NodeExecutionError):
                await historian_node(initial_state)

            # Fourth attempt - should fail due to circuit breaker
            with pytest.raises(NodeExecutionError, match="Circuit breaker open"):
                await historian_node(initial_state)

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_after_timeout(self, initial_state):
        """Test circuit breaker reset after timeout."""
        failing_agent = ControlledHistorianAgent(behavior="failure")

        # Override circuit breaker with shorter timeout for testing
        original_historian_node = historian_node

        # Create a version with shorter timeout
        @circuit_breaker(max_failures=2, reset_timeout=0.1)
        @node_metrics
        async def test_historian_node(state):
            return await original_historian_node.__wrapped__.__wrapped__(state)

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=failing_agent,
        ):
            # Trigger circuit breaker
            with pytest.raises(NodeExecutionError):
                await test_historian_node(initial_state)

            with pytest.raises(NodeExecutionError):
                await test_historian_node(initial_state)

            # Should be open now
            with pytest.raises(NodeExecutionError, match="Circuit breaker open"):
                await test_historian_node(initial_state)

            # Wait for reset
            await asyncio.sleep(0.2)

            # Should work again (will still fail, but not due to circuit breaker)
            with pytest.raises(NodeExecutionError):
                await test_historian_node(initial_state)

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_resets_count(self, initial_state):
        """Test that successful execution resets failure count."""

        # Create agent that fails first, then succeeds (disable retries for circuit breaker testing)
        class FailThenSucceedAgent(BaseAgent):
            def __init__(self, name: str = "Historian"):
                from cognivault.agents.base_agent import RetryConfig

                super().__init__(
                    name, retry_config=RetryConfig(max_retries=0)
                )  # No retries
                self.call_count = 0

            async def run(self, context: AgentContext) -> AgentContext:
                self.call_count += 1
                # Fail first call, succeed on second
                if self.call_count == 1:
                    raise Exception("First call fails")

                # Second call succeeds
                context.add_agent_output(self.name, "Success after failure")
                context.execution_state.update(
                    {
                        "search_results_count": 5,
                        "filtered_results_count": 3,
                        "search_strategy": "keyword",
                        "topics_found": ["recovery"],
                        "confidence": 0.7,
                        "llm_analysis_used": True,
                        "historian_metadata": {},
                    }
                )
                return context

        agent = FailThenSucceedAgent()

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=agent,
        ):
            # First call should fail after exhausting retries
            with pytest.raises(NodeExecutionError):
                await historian_node(initial_state)

            # Second call should succeed
            result_state = await historian_node(initial_state)
            assert result_state["historian"] is not None
            assert (
                result_state["historian"]["historical_summary"]
                == "Success after failure"
            )

            # Third call should also succeed (failure count was reset)
            result_state = await historian_node(initial_state)
            assert result_state["historian"] is not None


class TestHistorianNodeMetrics:
    """Test metrics collection for historian node."""

    @pytest.fixture
    def initial_state(self) -> CogniVaultState:
        """Create initial state with refiner output."""
        state = create_initial_state("Test query", "exec-123")
        refiner_output: RefinerOutput = {
            "refined_question": "Test refined query",
            "topics": ["test"],
            "confidence": 0.8,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        return set_agent_output(state, "refiner", refiner_output)

    @pytest.mark.asyncio
    async def test_metrics_collection_success(self, initial_state):
        """Test metrics collection for successful execution."""
        success_agent = ControlledHistorianAgent(behavior="success")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=success_agent,
        ):
            # Capture log output to verify metrics
            with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
                result_state = await historian_node(initial_state)

                # Verify success metrics were logged
                mock_logger.info.assert_any_call("Starting execution of historian node")
                mock_logger.info.assert_any_call(
                    "Historian node completed successfully"
                )

                # Check for completion log with timing
                completion_calls = [
                    call
                    for call in mock_logger.info.call_args_list
                    if "Completed historian node execution" in str(call)
                ]
                assert len(completion_calls) > 0

    @pytest.mark.asyncio
    async def test_metrics_collection_failure(self, initial_state):
        """Test metrics collection for failed execution."""
        failing_agent = ControlledHistorianAgent(behavior="failure")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=failing_agent,
        ):
            with patch("cognivault.orchestration.node_wrappers.logger") as mock_logger:
                with pytest.raises(NodeExecutionError):
                    await historian_node(initial_state)

                # Verify failure metrics were logged
                mock_logger.info.assert_any_call("Starting execution of historian node")
                # Error message now includes AgentExecutionError wrapping
                error_calls = [
                    call
                    for call in mock_logger.error.call_args_list
                    if "Controlled failure for testing" in str(call)
                ]
                assert len(error_calls) > 0, (
                    f"Expected error call with 'Controlled failure for testing' not found in {mock_logger.error.call_args_list}"
                )

                # Check for failure log with timing
                failure_calls = [
                    call
                    for call in mock_logger.error.call_args_list
                    if "Failed historian node execution" in str(call)
                ]
                assert len(failure_calls) > 0

    @pytest.mark.asyncio
    async def test_metrics_timing_accuracy(self, initial_state):
        """Test that metrics timing is accurate."""
        slow_agent = ControlledHistorianAgent(behavior="slow")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=slow_agent,
        ):
            start_time = time.time()
            result_state = await historian_node(initial_state)
            end_time = time.time()

            # Verify execution took expected time
            actual_duration = end_time - start_time
            assert actual_duration >= 0.1  # Should take at least 100ms due to sleep

            # Verify agent recorded execution time
            assert len(slow_agent.execution_times) == 1
            assert slow_agent.execution_times[0] >= 0.1


class TestHistorianNodeErrorHandling:
    """Test error handling scenarios for historian node."""

    @pytest.fixture
    def initial_state(self) -> CogniVaultState:
        """Create initial state with refiner output."""
        state = create_initial_state("Test query", "exec-123")
        refiner_output: RefinerOutput = {
            "refined_question": "Test refined query",
            "topics": ["test"],
            "confidence": 0.8,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        return set_agent_output(state, "refiner", refiner_output)

    @pytest.mark.asyncio
    async def test_memory_error_handling(self, initial_state):
        """Test handling of memory errors."""
        memory_error_agent = ControlledHistorianAgent(behavior="memory_error")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=memory_error_agent,
        ):
            with pytest.raises(NodeExecutionError, match="Historian execution failed"):
                await historian_node(initial_state)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, initial_state):
        """Test handling of timeout errors."""
        timeout_agent = ControlledHistorianAgent(behavior="timeout_error")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=timeout_agent,
        ):
            with pytest.raises(NodeExecutionError, match="Historian execution failed"):
                await historian_node(initial_state)

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, initial_state):
        """Test handling of partial failures."""
        partial_failure_agent = ControlledHistorianAgent(behavior="partial_failure")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=partial_failure_agent,
        ):
            # First execution should succeed with partial data
            result_state = await historian_node(initial_state)
            assert result_state["historian"] is not None
            assert (
                result_state["historian"]["historical_summary"]
                == "Partial historical context"
            )
            assert result_state["historian"]["search_results_count"] == 5
            assert result_state["historian"]["filtered_results_count"] == 0

            # Second execution should also succeed with same partial data
            result_state2 = await historian_node(initial_state)
            assert result_state2["historian"] is not None
            assert (
                result_state2["historian"]["historical_summary"]
                == "Partial historical context"
            )

    @pytest.mark.asyncio
    async def test_error_state_recording(self, initial_state):
        """Test that errors are properly recorded in state."""
        failing_agent = ControlledHistorianAgent(behavior="failure")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=failing_agent,
        ):
            with patch(
                "cognivault.orchestration.node_wrappers.record_agent_error"
            ) as mock_record:
                with pytest.raises(NodeExecutionError):
                    await historian_node(initial_state)

                # Verify error was recorded
                mock_record.assert_called_once()
                call_args = mock_record.call_args
                assert call_args[0][1] == "historian"  # Agent name
                assert isinstance(call_args[0][2], Exception)  # Error object


class TestHistorianNodeEdgeCases:
    """Test edge cases and boundary conditions for historian node."""

    @pytest.fixture
    def initial_state(self) -> CogniVaultState:
        """Create initial state with refiner output."""
        state = create_initial_state("Test query", "exec-123")
        refiner_output: RefinerOutput = {
            "refined_question": "Test refined query",
            "topics": ["test"],
            "confidence": 0.8,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        return set_agent_output(state, "refiner", refiner_output)

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, initial_state):
        """Test handling of empty queries."""
        # Modify state to have empty query
        initial_state["query"] = ""

        success_agent = ControlledHistorianAgent(behavior="success")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=success_agent,
        ):
            result_state = await historian_node(initial_state)

            # Should still execute successfully
            assert result_state["historian"] is not None
            assert (
                result_state["historian"]["historical_summary"]
                == "Historical context for: "
            )

    @pytest.mark.asyncio
    async def test_very_long_query_handling(self, initial_state):
        """Test handling of very long queries."""
        # Create a very long query
        long_query = "What is artificial intelligence" + " and machine learning" * 100
        initial_state["query"] = long_query

        success_agent = ControlledHistorianAgent(behavior="success")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=success_agent,
        ):
            result_state = await historian_node(initial_state)

            # Should handle long query gracefully
            assert result_state["historian"] is not None
            assert long_query in result_state["historian"]["historical_summary"]

    @pytest.mark.asyncio
    async def test_no_llm_fallback_behavior(self, initial_state):
        """Test behavior when LLM is not available (fallback mode)."""
        no_llm_agent = ControlledHistorianAgent(behavior="no_llm")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=no_llm_agent,
        ):
            result_state = await historian_node(initial_state)

            # Should succeed with fallback behavior
            assert result_state["historian"] is not None
            assert (
                result_state["historian"]["historical_summary"]
                == "Basic historical context without LLM"
            )
            assert result_state["historian"]["llm_analysis_used"] is False
            assert result_state["historian"]["search_strategy"] == "keyword"
            assert result_state["historian"]["metadata"]["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_zero_confidence_handling(self, initial_state):
        """Test handling of zero confidence results."""

        class ZeroConfidenceAgent(BaseAgent):
            def __init__(self, name: str = "Historian"):
                super().__init__(name)

            async def run(self, context: AgentContext) -> AgentContext:
                context.add_agent_output(self.name, "Low confidence historical context")
                context.execution_state.update(
                    {
                        "search_results_count": 3,
                        "filtered_results_count": 0,
                        "search_strategy": "keyword",
                        "topics_found": [],
                        "confidence": 0.0,
                        "llm_analysis_used": False,
                        "historian_metadata": {"low_confidence": True},
                    }
                )
                return context

        zero_conf_agent = ZeroConfidenceAgent()

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=zero_conf_agent,
        ):
            result_state = await historian_node(initial_state)

            # Should handle zero confidence gracefully
            assert result_state["historian"] is not None
            assert result_state["historian"]["confidence"] == 0.0
            assert result_state["historian"]["filtered_results_count"] == 0
            assert result_state["historian"]["metadata"]["low_confidence"] is True

    @pytest.mark.asyncio
    async def test_maximum_results_handling(self, initial_state):
        """Test handling of maximum search results."""

        class MaxResultsAgent(BaseAgent):
            def __init__(self, name: str = "Historian"):
                super().__init__(name)

            async def run(self, context: AgentContext) -> AgentContext:
                context.add_agent_output(
                    self.name, "Maximum results historical context"
                )
                context.retrieved_notes = [f"/notes/note_{i}.md" for i in range(100)]
                context.execution_state.update(
                    {
                        "search_results_count": 1000,
                        "filtered_results_count": 100,
                        "search_strategy": "hybrid",
                        "topics_found": [f"topic_{i}" for i in range(50)],
                        "confidence": 0.95,
                        "llm_analysis_used": True,
                        "historian_metadata": {"max_results": True},
                    }
                )
                return context

        max_results_agent = MaxResultsAgent()

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=max_results_agent,
        ):
            result_state = await historian_node(initial_state)

            # Should handle maximum results gracefully
            assert result_state["historian"] is not None
            assert result_state["historian"]["search_results_count"] == 1000
            assert result_state["historian"]["filtered_results_count"] == 100
            assert len(result_state["historian"]["retrieved_notes"]) == 100
            assert len(result_state["historian"]["topics_found"]) == 50

    @pytest.mark.asyncio
    async def test_state_mutation_prevention(self, initial_state):
        """Test that historian node doesn't mutate original state."""
        original_state = initial_state.copy()
        success_agent = ControlledHistorianAgent(behavior="success")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=success_agent,
        ):
            result_state = await historian_node(initial_state)

            # Original state should be unchanged
            assert initial_state["historian"] is None
            assert original_state["historian"] is None

            # Result state should have historian output (partial state)
            assert result_state["historian"] is not None
            assert result_state["successful_agents"] == ["historian"]

            # LangGraph nodes return partial state updates only
            # The original state is not mutated and full state is preserved by LangGraph

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self, initial_state):
        """Test that concurrent historian node executions are safe."""
        success_agent = ControlledHistorianAgent(behavior="success")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=success_agent,
        ):
            # Execute multiple concurrent calls
            tasks = [
                historian_node(initial_state),
                historian_node(initial_state),
                historian_node(initial_state),
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 3
            for result in results:
                assert result["historian"] is not None
                assert (
                    result["historian"]["historical_summary"]
                    == "Historical context for: Test query"
                )

            # Agent should have been called for each concurrent execution
            # Note: With run_with_retry, execution count may include retry attempts
            assert success_agent.execution_count >= 3


class TestHistorianNodePerformance:
    """Test performance characteristics of historian node."""

    @pytest.fixture
    def initial_state(self) -> CogniVaultState:
        """Create initial state with refiner output."""
        state = create_initial_state("Performance test query", "exec-123")
        refiner_output: RefinerOutput = {
            "refined_question": "Performance test refined query",
            "topics": ["performance", "test"],
            "confidence": 0.8,
            "processing_notes": None,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        return set_agent_output(state, "refiner", refiner_output)

    @pytest.mark.asyncio
    async def test_performance_under_load(self, initial_state):
        """Test historian node performance under load."""
        fast_agent = ControlledHistorianAgent(behavior="success")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=fast_agent,
        ):
            # Measure performance for multiple executions
            start_time = time.time()

            tasks = [historian_node(initial_state) for _ in range(10)]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # All should succeed
            assert len(results) == 10
            for result in results:
                assert result["historian"] is not None

            # Performance should be reasonable (less than 1 second for 10 concurrent executions)
            assert total_time < 1.0

            # Agent should have been called at least 10 times (accounting for retry logic)
            assert fast_agent.execution_count >= 10

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, initial_state):
        """Test that historian node doesn't have memory leaks."""
        success_agent = ControlledHistorianAgent(behavior="success")

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=success_agent,
        ):
            # Execute many times to detect memory leaks
            for i in range(100):
                result_state = await historian_node(initial_state)
                assert result_state["historian"] is not None

                # Clear result to help with garbage collection
                del result_state

            # Should complete without memory issues (accounting for retry logic)
            assert success_agent.execution_count >= 100

    @pytest.mark.asyncio
    async def test_timeout_handling_performance(self, initial_state):
        """Test that timeout handling doesn't impact performance."""

        # Create agent with very short execution time
        class FastAgent(BaseAgent):
            def __init__(self, name: str = "Historian"):
                super().__init__(name)

            async def run(self, context: AgentContext) -> AgentContext:
                # Very fast execution
                context.add_agent_output(self.name, "Fast historical context")
                context.execution_state.update(
                    {
                        "search_results_count": 1,
                        "filtered_results_count": 1,
                        "search_strategy": "fast",
                        "topics_found": ["fast"],
                        "confidence": 0.9,
                        "llm_analysis_used": False,
                        "historian_metadata": {"fast": True},
                    }
                )
                return context

        fast_agent = FastAgent()

        with patch(
            "cognivault.orchestration.node_wrappers.create_agent_with_llm",
            return_value=fast_agent,
        ):
            start_time = time.time()
            result_state = await historian_node(initial_state)
            end_time = time.time()

            # Should complete very quickly
            execution_time = end_time - start_time
            assert execution_time < 0.1  # Less than 100ms

            # Should still produce correct output
            assert result_state["historian"] is not None
            assert (
                result_state["historian"]["historical_summary"]
                == "Fast historical context"
            )
