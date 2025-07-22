"""
Tests for the failure manager.

Covers failure handling, cascade prevention, retry strategies,
circuit breakers, and recovery mechanisms.
"""

import pytest
import time

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.dependencies.graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
    DependencyType,
    ExecutionPriority,
)
from cognivault.dependencies.failure_manager import (
    FailureManager,
    FailureType,
    CascadePreventionStrategy,
    RetryStrategy,
    RetryConfiguration,
    FailureRecord,
    FailureImpactAnalysis,
    DependencyCircuitBreaker,
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def run(self, context: AgentContext) -> AgentContext:
        context.agent_outputs[self.name] = f"Output from {self.name}"
        return context


@pytest.fixture
def graph_engine():
    """Create a graph engine with sample agents."""
    engine = DependencyGraphEngine()

    agents = {
        "agent_a": MockAgent("agent_a"),
        "agent_b": MockAgent("agent_b"),
        "agent_c": MockAgent("agent_c"),
        "agent_d": MockAgent("agent_d"),
    }

    for agent_id, agent in agents.items():
        node = DependencyNode(
            agent_id=agent_id,
            agent=agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=30000,
        )
        engine.add_node(node)

    return engine


@pytest.fixture
def failure_manager(graph_engine):
    """Create a failure manager for testing."""
    return FailureManager(graph_engine)


@pytest.fixture
def context():
    """Create a basic agent context."""
    return AgentContext(query="test query")


class TestRetryConfiguration:
    """Test RetryConfiguration functionality."""

    def test_retry_config_creation(self):
        """Test creating retry configuration."""
        config = RetryConfiguration(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5,
            base_delay_ms=2000.0,
            backoff_multiplier=1.5,
        )

        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.max_attempts == 5
        assert config.base_delay_ms == 2000.0
        assert config.backoff_multiplier == 1.5

    def test_fixed_interval_delay(self):
        """Test fixed interval delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.FIXED_INTERVAL,
            base_delay_ms=1000.0,
            jitter=False,  # Disable jitter for predictable tests
        )

        assert config.calculate_delay(1) == 1000.0
        assert config.calculate_delay(3) == 1000.0

    def test_linear_backoff_delay(self):
        """Test linear backoff delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay_ms=1000.0,
            jitter=False,  # Disable jitter for predictable tests
        )

        assert config.calculate_delay(1) == 1000.0
        assert config.calculate_delay(2) == 2000.0
        assert config.calculate_delay(3) == 3000.0

    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay_ms=1000.0,
            backoff_multiplier=2.0,
            jitter=False,  # Disable jitter for predictable tests
        )

        assert config.calculate_delay(1) == 1000.0
        assert config.calculate_delay(2) == 2000.0
        assert config.calculate_delay(3) == 4000.0

    def test_adaptive_delay(self):
        """Test adaptive delay calculation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.ADAPTIVE,
            base_delay_ms=1000.0,
            backoff_multiplier=2.0,
            success_rate_threshold=0.7,
            failure_window_size=10,
            jitter=False,
        )

        # High failure rate - should use exponential backoff
        delay = config.calculate_delay(2, recent_failures=8)
        assert (
            delay == 4000.0
        )  # base * multiplier^attempt (not attempt-1 in adaptive mode)

        # Low failure rate - should use fixed interval
        delay = config.calculate_delay(2, recent_failures=2)
        assert delay == 1000.0  # base delay

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay_ms."""
        config = RetryConfiguration(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay_ms=1000.0,
            max_delay_ms=5000.0,
            backoff_multiplier=2.0,
            jitter=False,
        )

        # Should be capped at max_delay
        delay = config.calculate_delay(10)  # Would be 512000 without cap
        assert delay == 5000.0

    def test_no_retry_strategy(self):
        """Test no retry strategy."""
        config = RetryConfiguration(strategy=RetryStrategy.NO_RETRY)

        assert config.calculate_delay(1) == 0
        assert config.calculate_delay(5) == 0

    def test_jitter_variation(self):
        """Test that jitter adds variation."""
        config = RetryConfiguration(
            strategy=RetryStrategy.FIXED_INTERVAL,
            base_delay_ms=1000.0,
            jitter=True,
        )

        delays = [config.calculate_delay(1) for _ in range(10)]

        # Should have variation due to jitter
        assert len(set(delays)) > 1
        # All delays should be within jitter range (0.8 to 1.2 * base)
        assert all(800 <= delay <= 1200 for delay in delays)


class TestFailureRecord:
    """Test FailureRecord functionality."""

    def test_failure_record_creation(self):
        """Test creating a failure record."""
        record = FailureRecord(
            agent_id="test_agent",
            failure_type=FailureType.TIMEOUT,
            error_message="Agent timed out",
            timestamp=time.time(),
            attempt_number=2,
            context_snapshot={"key": "value"},
            stack_trace="stack trace here",
            impact_score=0.5,
        )

        assert record.agent_id == "test_agent"
        assert record.failure_type == FailureType.TIMEOUT
        assert record.error_message == "Agent timed out"
        assert record.attempt_number == 2
        assert record.impact_score == 0.5

    def test_failure_record_to_dict(self):
        """Test converting failure record to dictionary."""
        record = FailureRecord(
            agent_id="test_agent",
            failure_type=FailureType.LLM_ERROR,
            error_message="API error",
            timestamp=12345.0,
            attempt_number=1,
            context_snapshot={"agents": 2},
        )

        result = record.to_dict()

        expected = {
            "agent_id": "test_agent",
            "failure_type": "llm_error",
            "error_message": "API error",
            "timestamp": 12345.0,
            "attempt_number": 1,
            "context_snapshot": {"agents": 2},
            "stack_trace": None,
            "recovery_action": None,
            "impact_score": 0.0,
        }

        assert result == expected


class TestFailureImpactAnalysis:
    """Test FailureImpactAnalysis functionality."""

    def test_impact_analysis_creation(self):
        """Test creating failure impact analysis."""
        analysis = FailureImpactAnalysis(
            failed_agent="agent_a",
            directly_affected=["agent_b", "agent_c"],
            transitively_affected=["agent_d"],
            critical_path_affected=True,
            estimated_delay_ms=5000.0,
            alternative_paths=[["fallback_agent"]],
            recovery_options=["retry", "fallback"],
            severity_score=0.8,
        )

        assert analysis.failed_agent == "agent_a"
        assert analysis.directly_affected == ["agent_b", "agent_c"]
        assert analysis.transitively_affected == ["agent_d"]
        assert analysis.critical_path_affected is True
        assert analysis.estimated_delay_ms == 5000.0
        assert analysis.severity_score == 0.8

    def test_total_affected_count(self):
        """Test getting total affected count."""
        analysis = FailureImpactAnalysis(
            failed_agent="agent_a",
            directly_affected=["agent_b", "agent_c"],
            transitively_affected=["agent_c", "agent_d"],  # agent_c appears in both
            critical_path_affected=False,
            estimated_delay_ms=1000.0,
            alternative_paths=[],
            recovery_options=[],
            severity_score=0.5,
        )

        # Should deduplicate agent_c
        assert analysis.get_total_affected_count() == 3

    def test_has_recovery_options(self):
        """Test checking if recovery options are available."""
        # With recovery options
        analysis1 = FailureImpactAnalysis(
            failed_agent="agent_a",
            directly_affected=[],
            transitively_affected=[],
            critical_path_affected=False,
            estimated_delay_ms=0.0,
            alternative_paths=[],
            recovery_options=["retry"],
            severity_score=0.0,
        )
        assert analysis1.has_recovery_options() is True

        # With alternative paths
        analysis2 = FailureImpactAnalysis(
            failed_agent="agent_a",
            directly_affected=[],
            transitively_affected=[],
            critical_path_affected=False,
            estimated_delay_ms=0.0,
            alternative_paths=[["fallback"]],
            recovery_options=[],
            severity_score=0.0,
        )
        assert analysis2.has_recovery_options() is True

        # Without any recovery options
        analysis3 = FailureImpactAnalysis(
            failed_agent="agent_a",
            directly_affected=[],
            transitively_affected=[],
            critical_path_affected=False,
            estimated_delay_ms=0.0,
            alternative_paths=[],
            recovery_options=[],
            severity_score=0.0,
        )
        assert analysis3.has_recovery_options() is False


class TestCircuitBreaker:
    """Test DependencyCircuitBreaker functionality."""

    def test_circuit_breaker_creation(self):
        """Test creating a circuit breaker."""
        cb = DependencyCircuitBreaker(
            failure_threshold=3,
            recovery_timeout_ms=60000.0,
            half_open_max_calls=2,
        )

        assert cb.failure_threshold == 3
        assert cb.recovery_timeout_ms == 60000.0
        assert cb.half_open_max_calls == 2
        assert cb.state == "CLOSED"

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        cb = DependencyCircuitBreaker(failure_threshold=2)

        # Initially closed - should allow execution
        assert cb.can_execute() is True
        assert cb.get_state() == "CLOSED"

        # Record success - should remain closed
        cb.record_success()
        assert cb.get_state() == "CLOSED"

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failure threshold."""
        cb = DependencyCircuitBreaker(failure_threshold=2)

        # First failure
        cb.record_failure()
        assert cb.get_state() == "CLOSED"
        assert cb.can_execute() is True

        # Second failure - should open
        cb.record_failure()
        assert cb.get_state() == "OPEN"
        assert cb.can_execute() is False

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = DependencyCircuitBreaker(
            failure_threshold=1,
            recovery_timeout_ms=100.0,  # Short timeout for testing
        )

        # Trigger opening
        cb.record_failure()
        assert cb.get_state() == "OPEN"
        assert cb.can_execute() is False

        # Wait for recovery timeout
        time.sleep(0.2)  # 200ms > 100ms timeout

        # Should transition to half-open
        assert cb.can_execute() is True
        assert cb.get_state() == "HALF_OPEN"

    def test_circuit_breaker_half_open_success(self):
        """Test circuit breaker success in half-open state."""
        cb = DependencyCircuitBreaker(failure_threshold=1, recovery_timeout_ms=1.0)

        # Open the circuit
        cb.record_failure()
        time.sleep(0.002)  # Wait for recovery

        # Should be half-open
        cb.can_execute()  # This transitions to half-open
        assert cb.get_state() == "HALF_OPEN"

        # Success should close the circuit
        cb.record_success()
        assert cb.get_state() == "CLOSED"

    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state."""
        cb = DependencyCircuitBreaker(failure_threshold=1, recovery_timeout_ms=1.0)

        # Open the circuit
        cb.record_failure()
        time.sleep(0.002)  # Wait for recovery

        # Should be half-open
        cb.can_execute()  # This transitions to half-open
        assert cb.get_state() == "HALF_OPEN"

        # Failure should reopen the circuit
        cb.record_failure()
        assert cb.get_state() == "OPEN"

    def test_circuit_breaker_half_open_call_limit(self):
        """Test circuit breaker call limit in half-open state."""
        cb = DependencyCircuitBreaker(
            failure_threshold=1,
            recovery_timeout_ms=1.0,
            half_open_max_calls=2,
        )

        # Open the circuit
        cb.record_failure()
        time.sleep(0.002)  # Wait for recovery

        # First call - should be allowed
        assert cb.can_execute() is True

        # Second call - should be allowed
        assert cb.can_execute() is True

        # Third call - should be denied
        assert cb.can_execute() is False


class TestFailureManager:
    """Test FailureManager functionality."""

    def test_failure_manager_creation(self, graph_engine):
        """Test creating a failure manager."""
        fm = FailureManager(graph_engine)

        assert fm.graph_engine == graph_engine
        assert len(fm.failure_history) == 0
        assert len(fm.circuit_breakers) == 0
        assert fm.cascade_prevention == CascadePreventionStrategy.CIRCUIT_BREAKER

    def test_configure_retry(self, failure_manager):
        """Test configuring retry behavior."""
        config = RetryConfiguration(max_attempts=5)
        failure_manager.configure_retry("agent_a", config)

        assert "agent_a" in failure_manager.retry_configs
        assert failure_manager.retry_configs["agent_a"].max_attempts == 5

    def test_configure_circuit_breaker(self, failure_manager):
        """Test configuring circuit breaker."""
        failure_manager.configure_circuit_breaker("agent_a", failure_threshold=3)

        assert "agent_a" in failure_manager.circuit_breakers
        assert failure_manager.circuit_breakers["agent_a"].failure_threshold == 3

    def test_set_cascade_prevention(self, failure_manager):
        """Test setting cascade prevention strategy."""
        failure_manager.set_cascade_prevention(CascadePreventionStrategy.ISOLATION)
        assert failure_manager.cascade_prevention == CascadePreventionStrategy.ISOLATION

    def test_add_fallback_chain(self, failure_manager):
        """Test adding fallback chain."""
        fallback_agents = ["agent_b", "agent_c"]
        failure_manager.add_fallback_chain("agent_a", fallback_agents)

        assert "agent_a" in failure_manager.fallback_chains
        assert failure_manager.fallback_chains["agent_a"] == fallback_agents

    def test_can_execute_agent_basic(self, failure_manager, context):
        """Test basic agent execution check."""
        can_execute, reason = failure_manager.can_execute_agent("agent_a", context)

        assert can_execute is True
        assert reason == "OK"

    def test_can_execute_agent_blocked(self, failure_manager, context):
        """Test execution check for blocked agent."""
        failure_manager.blocked_agents.add("agent_a")

        can_execute, reason = failure_manager.can_execute_agent("agent_a", context)

        assert can_execute is False
        assert "blocked" in reason

    def test_can_execute_agent_circuit_breaker(self, failure_manager, context):
        """Test execution check with circuit breaker."""
        # Configure and open circuit breaker
        failure_manager.configure_circuit_breaker("agent_a", failure_threshold=1)
        failure_manager.circuit_breakers["agent_a"].record_failure()

        can_execute, reason = failure_manager.can_execute_agent("agent_a", context)

        assert can_execute is False
        assert "Circuit breaker" in reason

    @pytest.mark.asyncio
    async def test_handle_agent_failure_basic(self, failure_manager, context):
        """Test basic agent failure handling."""
        error = Exception("Test error")

        should_retry, recovery_action = await failure_manager.handle_agent_failure(
            "agent_a", error, context, attempt_number=1
        )

        # Should have recorded the failure
        assert len(failure_manager.failure_history) == 1
        assert failure_manager.failure_history[0].agent_id == "agent_a"
        assert failure_manager.failure_history[0].error_message == "Test error"

        # Should suggest retry by default
        assert should_retry is True

    @pytest.mark.asyncio
    async def test_handle_agent_failure_max_retries(self, failure_manager, context):
        """Test failure handling when max retries exceeded."""
        error = Exception("Test error")
        config = RetryConfiguration(max_attempts=2)
        failure_manager.configure_retry("agent_a", config)

        should_retry, recovery_action = await failure_manager.handle_agent_failure(
            "agent_a",
            error,
            context,
            attempt_number=3,  # Exceeds max
        )

        assert should_retry is False

    def test_classify_failure_timeout(self, failure_manager):
        """Test classifying timeout failures."""
        error = TimeoutError("Operation timed out")
        failure_type = failure_manager._classify_failure(error)
        assert failure_type == FailureType.TIMEOUT

    def test_classify_failure_llm_error(self, failure_manager):
        """Test classifying LLM errors."""
        error = Exception("OpenAI API error")
        failure_type = failure_manager._classify_failure(error)
        assert failure_type == FailureType.LLM_ERROR

    def test_classify_failure_validation(self, failure_manager):
        """Test classifying validation errors."""
        error = ValueError("Invalid input")
        failure_type = failure_manager._classify_failure(error)
        assert failure_type == FailureType.VALIDATION_ERROR

    def test_classify_failure_unknown(self, failure_manager):
        """Test classifying unknown errors."""
        error = Exception("Some unknown error")
        failure_type = failure_manager._classify_failure(error)
        assert failure_type == FailureType.UNKNOWN

    def test_analyze_failure_impact(self, failure_manager, context):
        """Test analyzing failure impact."""
        # Add dependencies
        failure_manager.graph_engine.add_dependency(
            "agent_a", "agent_b", DependencyType.HARD
        )
        failure_manager.graph_engine.add_dependency(
            "agent_b", "agent_c", DependencyType.HARD
        )

        analysis = failure_manager.analyze_failure_impact("agent_a", context)

        assert analysis.failed_agent == "agent_a"
        assert "agent_b" in analysis.directly_affected
        assert analysis.severity_score > 0

    def test_analyze_dependency_failures(self, failure_manager, context):
        """Test analyzing dependency failures."""
        # Add dependencies
        failure_manager.graph_engine.add_dependency(
            "agent_a", "agent_b", DependencyType.HARD
        )

        # Record failure for dependency
        failure_record = FailureRecord(
            agent_id="agent_a",
            failure_type=FailureType.TIMEOUT,
            error_message="timeout",
            timestamp=time.time(),
            attempt_number=1,
            context_snapshot={},
        )
        failure_manager.failure_history.append(failure_record)

        analysis = failure_manager.analyze_dependency_failures("agent_b", context)

        assert analysis is not None
        assert analysis.failed_agent == "agent_a"

    @pytest.mark.asyncio
    async def test_attempt_recovery_fallback_chain(self, failure_manager, context):
        """Test recovery using fallback chain."""
        # Set up fallback chain
        failure_manager.add_fallback_chain("agent_a", ["agent_b"])

        result = await failure_manager.attempt_recovery(
            "agent_a", "fallback_chain", context
        )

        assert result is True
        # Should have executed fallback agent
        assert "agent_b" in context.agent_outputs

    @pytest.mark.asyncio
    async def test_attempt_recovery_checkpoint_rollback(self, failure_manager, context):
        """Test recovery using checkpoint rollback."""
        # Create checkpoint
        context.agent_outputs["test"] = "value"
        failure_manager.create_checkpoint("checkpoint1", context)

        # Modify context
        context.agent_outputs["test"] = "modified"

        result = await failure_manager.attempt_recovery(
            "agent_a", "checkpoint_rollback", context
        )

        assert result is True
        assert context.agent_outputs["test"] == "value"  # Should be restored

    @pytest.mark.asyncio
    async def test_attempt_recovery_graceful_degradation(
        self, failure_manager, context
    ):
        """Test recovery using graceful degradation."""
        result = await failure_manager.attempt_recovery(
            "agent_a", "graceful_degradation", context
        )

        assert result is True
        assert "agent_a" in context.execution_state.get("degraded_agents", [])
        assert "[DEGRADED]" in context.agent_outputs.get("agent_a", "")

    @pytest.mark.asyncio
    async def test_attempt_recovery_isolation(self, failure_manager, context):
        """Test recovery using isolation."""
        result = await failure_manager.attempt_recovery("agent_a", "isolation", context)

        assert result is True
        assert "agent_a" in failure_manager.blocked_agents
        assert "agent_a" in context.execution_state.get("isolated_agents", [])

    def test_create_checkpoint(self, failure_manager, context):
        """Test creating recovery checkpoint."""
        context.agent_outputs["test"] = "value"

        failure_manager.create_checkpoint("test_checkpoint", context)

        assert "test_checkpoint" in failure_manager.recovery_checkpoints
        checkpoint = failure_manager.recovery_checkpoints["test_checkpoint"]
        assert checkpoint.agent_outputs["test"] == "value"

    def test_get_failure_statistics(self, failure_manager):
        """Test getting failure statistics."""
        # Add some failure history
        for i in range(3):
            record = FailureRecord(
                agent_id=f"agent_{i}",
                failure_type=FailureType.TIMEOUT,
                error_message="test",
                timestamp=time.time(),
                attempt_number=1,
                context_snapshot={},
            )
            failure_manager.failure_history.append(record)
            failure_manager.failure_counts[f"agent_{i}"] += 1

        stats = failure_manager.get_failure_statistics()

        assert stats["total_failures"] == 3
        assert stats["unique_failed_agents"] == 3
        assert "agent_failure_rates" in stats
        assert "failure_type_distribution" in stats


class TestIntegration:
    """Integration tests for failure manager."""

    @pytest.mark.asyncio
    async def test_complete_failure_handling_workflow(self, graph_engine):
        """Test complete failure handling workflow."""
        # Set up complex dependency graph
        failure_manager = FailureManager(graph_engine)

        # Add dependencies
        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)
        graph_engine.add_dependency("agent_a", "agent_c", DependencyType.HARD)
        graph_engine.add_dependency("agent_b", "agent_d", DependencyType.HARD)

        # Configure failure handling
        failure_manager.configure_retry("agent_a", RetryConfiguration(max_attempts=2))
        failure_manager.configure_circuit_breaker("agent_a", failure_threshold=2)
        failure_manager.add_fallback_chain("agent_a", ["agent_b"])
        failure_manager.set_cascade_prevention(
            CascadePreventionStrategy.GRACEFUL_DEGRADATION
        )

        context = AgentContext(query="test")

        # Simulate failure
        error = Exception("Simulated failure")
        should_retry, recovery_action = await failure_manager.handle_agent_failure(
            "agent_a", error, context, attempt_number=1
        )

        # Verify failure was recorded and handled
        assert len(failure_manager.failure_history) == 1
        assert should_retry is True

        # Analyze impact
        impact = failure_manager.analyze_failure_impact("agent_a", context)
        assert impact.failed_agent == "agent_a"
        assert len(impact.directly_affected) == 2  # agent_b, agent_c

        # Test recovery
        if recovery_action and recovery_action != "no_action":
            recovery_result = await failure_manager.attempt_recovery(
                "agent_a", recovery_action, context
            )
            assert recovery_result in [True, False]  # Should complete successfully

    @pytest.mark.asyncio
    async def test_cascade_prevention_scenarios(self, graph_engine):
        """Test different cascade prevention scenarios."""
        failure_manager = FailureManager(graph_engine)
        context = AgentContext(query="test")

        # Test circuit breaker strategy
        failure_manager.set_cascade_prevention(
            CascadePreventionStrategy.CIRCUIT_BREAKER
        )

        # Create high-impact failure
        high_impact_analysis = FailureImpactAnalysis(
            failed_agent="agent_a",
            directly_affected=["agent_b", "agent_c", "agent_d"],
            transitively_affected=["agent_e", "agent_f"],
            critical_path_affected=True,
            estimated_delay_ms=10000.0,
            alternative_paths=[],
            recovery_options=[],
            severity_score=0.9,  # High severity
        )

        # Should trigger circuit breaker
        recovery_action = await failure_manager._apply_cascade_prevention(
            "agent_a", high_impact_analysis, context
        )

        assert "circuit_breaker" in recovery_action
        assert "agent_a" in failure_manager.blocked_agents

    def test_failure_type_classification_comprehensive(self, failure_manager):
        """Test comprehensive failure type classification."""
        test_cases = [
            (TimeoutError("timeout"), FailureType.TIMEOUT),
            (Exception("timeout occurred"), FailureType.TIMEOUT),
            (Exception("OpenAI error"), FailureType.LLM_ERROR),
            (Exception("LLM failed"), FailureType.LLM_ERROR),
            (ValueError("validation failed"), FailureType.VALIDATION_ERROR),
            (Exception("invalid data"), FailureType.VALIDATION_ERROR),
            (MemoryError("out of memory"), FailureType.RESOURCE_EXHAUSTION),
            (Exception("resource unavailable"), FailureType.RESOURCE_EXHAUSTION),
            (ConnectionError("network error"), FailureType.NETWORK_ERROR),
            (Exception("connection failed"), FailureType.NETWORK_ERROR),
            (Exception("configuration error"), FailureType.CONFIGURATION_ERROR),
            (Exception("config invalid"), FailureType.CONFIGURATION_ERROR),
            (Exception("mysterious error"), FailureType.UNKNOWN),
        ]

        for error, expected_type in test_cases:
            result = failure_manager._classify_failure(error)
            assert result == expected_type, (
                f"Failed for {error}: expected {expected_type}, got {result}"
            )
