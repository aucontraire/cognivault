"""
Tests for LangGraph error policies and circuit breaker functionality.

This module provides comprehensive tests for error handling policies,
circuit breakers, retry mechanisms, and fallback strategies in the
LangGraph execution environment.
"""

import pytest
from typing import Any
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from cognivault.orchestration.error_policies import (
    ErrorPolicyType,
    FallbackStrategy,
    PolicyRetryConfig,
    CircuitBreakerConfig,
    ErrorPolicy,
    LangGraphExecutionError,
    CircuitBreakerState,
    CircuitBreaker,
    ErrorPolicyManager,
    get_error_policy_manager,
    retry_with_policy,
    circuit_breaker_policy,
    timeout_policy,
    comprehensive_error_policy,
    handle_node_fallback,
    get_error_statistics,
)


class TestEnums:
    """Test enum classes."""

    def test_error_policy_type_values(self) -> None:
        """Test ErrorPolicyType enum values."""
        assert ErrorPolicyType.FAIL_FAST.value == "fail_fast"
        assert ErrorPolicyType.RETRY_WITH_BACKOFF.value == "retry_with_backoff"
        assert ErrorPolicyType.CIRCUIT_BREAKER.value == "circuit_breaker"
        assert ErrorPolicyType.GRACEFUL_DEGRADATION.value == "graceful_degradation"

    def test_fallback_strategy_values(self) -> None:
        """Test FallbackStrategy enum values."""
        assert FallbackStrategy.SKIP_NODE.value == "skip_node"
        assert FallbackStrategy.USE_CACHED_RESULT.value == "use_cached_result"
        assert FallbackStrategy.SUBSTITUTE_AGENT.value == "substitute_agent"
        assert FallbackStrategy.PARTIAL_RESULT.value == "partial_result"

    def test_circuit_breaker_state_values(self) -> None:
        """Test CircuitBreakerState enum values."""
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"


class TestPolicyRetryConfig:
    """Test PolicyRetryConfig dataclass."""

    def test_default_retry_config(self) -> None:
        """Test default retry configuration."""
        config = PolicyRetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retry_on_types == [Exception]

    def test_custom_retry_config(self) -> None:
        """Test custom retry configuration."""
        config = PolicyRetryConfig(
            max_attempts=5,
            base_delay_seconds=0.5,
            max_delay_seconds=30.0,
            exponential_base=1.5,
            jitter=False,
            retry_on_types=[ValueError, TypeError],
        )

        assert config.max_attempts == 5
        assert config.base_delay_seconds == 0.5
        assert config.max_delay_seconds == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.retry_on_types == [ValueError, TypeError]


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_default_circuit_breaker_config(self) -> None:
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60.0
        assert config.half_open_max_calls == 3

    def test_custom_circuit_breaker_config(self) -> None:
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_calls=5,
        )

        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30.0
        assert config.half_open_max_calls == 5


class TestErrorPolicy:
    """Test ErrorPolicy dataclass."""

    def test_basic_error_policy(self) -> None:
        """Test basic error policy creation."""
        policy = ErrorPolicy(policy_type=ErrorPolicyType.FAIL_FAST)

        assert policy.policy_type == ErrorPolicyType.FAIL_FAST
        assert policy.retry_config is None
        assert policy.circuit_breaker_config is None
        assert policy.fallback_strategy is None
        assert policy.timeout_seconds is None
        assert policy.critical_errors == []
        assert policy.recoverable_errors == [Exception]

    def test_complex_error_policy(self) -> None:
        """Test complex error policy with all options."""
        retry_config = PolicyRetryConfig(max_attempts=2)
        cb_config = CircuitBreakerConfig(failure_threshold=3)

        policy = ErrorPolicy(
            policy_type=ErrorPolicyType.CIRCUIT_BREAKER,
            retry_config=retry_config,
            circuit_breaker_config=cb_config,
            fallback_strategy=FallbackStrategy.PARTIAL_RESULT,
            timeout_seconds=30.0,
            critical_errors=[KeyboardInterrupt],
            recoverable_errors=[ValueError, TypeError],
        )

        assert policy.policy_type == ErrorPolicyType.CIRCUIT_BREAKER
        assert policy.retry_config == retry_config
        assert policy.circuit_breaker_config == cb_config
        assert policy.fallback_strategy == FallbackStrategy.PARTIAL_RESULT
        assert policy.timeout_seconds == 30.0
        assert policy.critical_errors == [KeyboardInterrupt]
        assert policy.recoverable_errors == [ValueError, TypeError]


class TestLangGraphExecutionError:
    """Test LangGraphExecutionError exception class."""

    def test_basic_execution_error(self) -> None:
        """Test basic execution error creation."""
        error = LangGraphExecutionError(message="Test error", node_name="test_node")

        assert str(error) == "Test error"
        assert error.node_name == "test_node"
        assert error.original_error is None
        assert error.retry_count == 0
        assert error.execution_context == {}
        assert isinstance(error.timestamp, float)

    def test_detailed_execution_error(self) -> None:
        """Test detailed execution error with all fields."""
        original_error = ValueError("Original error")
        context = {"agent": "refiner", "query": "test"}

        error = LangGraphExecutionError(
            message="Wrapped error",
            node_name="refiner",
            original_error=original_error,
            retry_count=2,
            execution_context=context,
        )

        assert str(error) == "Wrapped error"
        assert error.node_name == "refiner"
        assert error.original_error == original_error
        assert error.retry_count == 2
        assert error.execution_context == context


class TestCircuitBreaker:
    """Test CircuitBreaker implementation."""

    @pytest.fixture
    def circuit_breaker(self) -> Any:
        """Fixture for circuit breaker with test configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0,
            half_open_max_calls=2,
        )
        return CircuitBreaker(config)

    def test_initial_state(self, circuit_breaker: CircuitBreaker) -> None:
        """Test circuit breaker initial state."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert circuit_breaker.can_execute() is True

    def test_failure_threshold_reached(self, circuit_breaker: CircuitBreaker) -> None:
        """Test circuit breaker opens when failure threshold reached."""
        # Record failures up to threshold
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == 3
        assert circuit_breaker.can_execute() is False

    def test_timeout_recovery(self, circuit_breaker: CircuitBreaker) -> None:
        """Test circuit breaker recovery after timeout."""
        # Trigger circuit breaker to open
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for timeout (using small timeout for test)
        time.sleep(1.1)

        # Should transition to half-open when can_execute is called
        # The can_execute() method changes state internally
        can_execute_result = circuit_breaker.can_execute()
        assert can_execute_result is True
        # After calling can_execute(), the state should be HALF_OPEN
        # Note: This is correct behavior - can_execute() transitions OPEN -> HALF_OPEN
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN  # type: ignore[comparison-overlap]

    def test_half_open_success_recovery(self, circuit_breaker: CircuitBreaker) -> None:
        """Test recovery from half-open state with successes."""
        # Open the circuit breaker
        for _ in range(3):
            circuit_breaker.record_failure()

        # Wait for timeout
        time.sleep(1.1)
        circuit_breaker.can_execute()  # Transition to half-open

        # Record enough successes to close
        for _ in range(2):
            circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_half_open_failure_reopens(self, circuit_breaker: CircuitBreaker) -> None:
        """Test circuit breaker reopens on failure in half-open state."""
        # Open the circuit breaker
        for _ in range(3):
            circuit_breaker.record_failure()

        # Wait for timeout
        time.sleep(1.1)
        circuit_breaker.can_execute()  # Transition to half-open

        # Record failure - should reopen
        circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.can_execute() is False

    def test_half_open_call_limit(self, circuit_breaker: CircuitBreaker) -> None:
        """Test half-open state call limit enforcement."""
        # Open the circuit breaker
        for _ in range(3):
            circuit_breaker.record_failure()

        # Wait for timeout
        time.sleep(1.1)

        # Should allow up to half_open_max_calls (2)
        assert circuit_breaker.can_execute() is True  # Call 1
        assert circuit_breaker.can_execute() is True  # Call 2
        assert circuit_breaker.can_execute() is False  # Call 3 - should be blocked

    def test_success_resets_failure_count(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test success resets failure count in closed state."""
        # Record some failures (but not enough to open)
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker.failure_count == 2

        # Record success - should reset
        circuit_breaker.record_success()
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == CircuitBreakerState.CLOSED


class TestErrorPolicyManager:
    """Test ErrorPolicyManager class."""

    @pytest.fixture
    def policy_manager(self) -> Any:
        """Fixture for fresh policy manager."""
        return ErrorPolicyManager()

    def test_default_policies_setup(self, policy_manager: ErrorPolicyManager) -> None:
        """Test default policies are set up correctly."""
        # Check that default policies exist for all agents
        assert "refiner" in policy_manager.policies
        assert "critic" in policy_manager.policies
        assert "historian" in policy_manager.policies
        assert "synthesis" in policy_manager.policies

        # Check policy types
        assert (
            policy_manager.policies["refiner"].policy_type
            == ErrorPolicyType.RETRY_WITH_BACKOFF
        )
        assert (
            policy_manager.policies["critic"].policy_type
            == ErrorPolicyType.GRACEFUL_DEGRADATION
        )
        assert (
            policy_manager.policies["historian"].policy_type
            == ErrorPolicyType.CIRCUIT_BREAKER
        )
        assert (
            policy_manager.policies["synthesis"].policy_type
            == ErrorPolicyType.RETRY_WITH_BACKOFF
        )

    def test_get_existing_policy(self, policy_manager: ErrorPolicyManager) -> None:
        """Test getting existing policy."""
        policy = policy_manager.get_policy("refiner")
        assert policy.policy_type == ErrorPolicyType.RETRY_WITH_BACKOFF

    def test_get_default_policy(self, policy_manager: ErrorPolicyManager) -> None:
        """Test getting default policy for unknown node."""
        policy = policy_manager.get_policy("unknown_node")
        assert policy.policy_type == ErrorPolicyType.RETRY_WITH_BACKOFF
        assert policy.retry_config is not None  # Type safety assertion
        assert policy.retry_config.max_attempts == 3

    def test_set_custom_policy(self, policy_manager: ErrorPolicyManager) -> None:
        """Test setting custom policy."""
        custom_policy = ErrorPolicy(
            policy_type=ErrorPolicyType.FAIL_FAST, timeout_seconds=10.0
        )

        policy_manager.set_policy("custom_node", custom_policy)

        retrieved_policy = policy_manager.get_policy("custom_node")
        assert retrieved_policy == custom_policy

    def test_circuit_breaker_initialization(
        self, policy_manager: ErrorPolicyManager
    ) -> None:
        """Test circuit breaker is initialized for circuit breaker policies."""
        # Historian has circuit breaker policy by default
        circuit_breaker = policy_manager.get_circuit_breaker("historian")
        assert circuit_breaker is not None
        assert isinstance(circuit_breaker, CircuitBreaker)

    def test_no_circuit_breaker_for_non_cb_policy(
        self, policy_manager: ErrorPolicyManager
    ) -> None:
        """Test no circuit breaker for non-circuit breaker policies."""
        # Refiner doesn't have circuit breaker policy
        circuit_breaker = policy_manager.get_circuit_breaker("refiner")
        assert circuit_breaker is None

    def test_set_policy_with_circuit_breaker(
        self, policy_manager: ErrorPolicyManager
    ) -> None:
        """Test setting policy with circuit breaker creates circuit breaker."""
        cb_config = CircuitBreakerConfig(failure_threshold=2)
        policy = ErrorPolicy(
            policy_type=ErrorPolicyType.CIRCUIT_BREAKER,
            circuit_breaker_config=cb_config,
        )

        policy_manager.set_policy("test_node", policy)

        circuit_breaker = policy_manager.get_circuit_breaker("test_node")
        assert circuit_breaker is not None
        assert circuit_breaker.config.failure_threshold == 2


class TestGlobalPolicyManager:
    """Test global policy manager functions."""

    def test_get_global_manager(self) -> None:
        """Test getting global error policy manager."""
        manager1 = get_error_policy_manager()
        manager2 = get_error_policy_manager()

        # Should return the same instance
        assert manager1 is manager2
        assert isinstance(manager1, ErrorPolicyManager)


class TestRetryDecorator:
    """Test retry_with_policy decorator."""

    @pytest.fixture
    def mock_function(self) -> Any:
        """Fixture for mock async function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self, mock_function: AsyncMock) -> None:
        """Test successful execution on first attempt."""
        mock_function.return_value = "success"

        @retry_with_policy("test_node")
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        assert result == "success"
        assert mock_function.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, mock_function: AsyncMock) -> None:
        """Test successful execution after initial failures."""
        # Fail twice, then succeed
        mock_function.side_effect = [
            ValueError("error1"),
            ValueError("error2"),
            "success",
        ]

        @retry_with_policy("test_node")
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        assert result == "success"
        assert mock_function.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_all_attempts_fail(self, mock_function: AsyncMock) -> None:
        """Test all retry attempts fail."""
        mock_function.side_effect = ValueError("persistent error")

        @retry_with_policy("test_node")
        async def test_func() -> Any:
            return await mock_function()

        with pytest.raises(LangGraphExecutionError) as exc_info:
            await test_func()

        assert "All retries exhausted" in str(exc_info.value)
        assert exc_info.value.node_name == "test_node"
        # Should try default max_attempts (from default policy)
        assert mock_function.call_count >= 2

    @pytest.mark.asyncio
    async def test_retry_non_retryable_error(self, mock_function: AsyncMock) -> None:
        """Test non-retryable error fails immediately."""
        # Set up policy manager to only retry on ValueError
        with patch(
            "cognivault.orchestration.error_policies.get_error_policy_manager"
        ) as mock_manager:
            mock_policy = ErrorPolicy(
                policy_type=ErrorPolicyType.RETRY_WITH_BACKOFF,
                retry_config=PolicyRetryConfig(retry_on_types=[ValueError]),
            )
            mock_manager.return_value.get_policy.return_value = mock_policy

            mock_function.side_effect = TypeError("non-retryable")

            @retry_with_policy("test_node")
            async def test_func() -> Any:
                return await mock_function()

            with pytest.raises(LangGraphExecutionError) as exc_info:
                await test_func()

            assert "Non-retryable error" in str(exc_info.value)
            assert mock_function.call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_retry_delay_calculation(self, mock_function: AsyncMock) -> None:
        """Test retry delay calculation with exponential backoff."""
        mock_function.side_effect = [
            ValueError("error1"),
            ValueError("error2"),
            "success",
        ]

        start_time = time.time()

        @retry_with_policy("test_node")
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        end_time = time.time()
        execution_time = end_time - start_time

        assert result == "success"
        # Should have taken some time due to delays (at least 1 second for base delay)
        assert execution_time >= 1.0


class TestCircuitBreakerDecorator:
    """Test circuit_breaker_policy decorator."""

    @pytest.fixture
    def mock_function(self) -> Any:
        """Fixture for mock async function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, mock_function: AsyncMock) -> None:
        """Test successful execution with circuit breaker."""
        mock_function.return_value = "success"

        @circuit_breaker_policy("historian")  # Has circuit breaker by default
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        assert result == "success"
        assert mock_function.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_no_policy(self, mock_function: AsyncMock) -> None:
        """Test execution with no circuit breaker policy."""
        mock_function.return_value = "success"

        @circuit_breaker_policy("refiner")  # No circuit breaker by default
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        assert result == "success"
        assert mock_function.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_execution(
        self, mock_function: AsyncMock
    ) -> None:
        """Test circuit breaker blocks execution when open."""
        # Get the circuit breaker and force it open
        manager = get_error_policy_manager()
        circuit_breaker = manager.get_circuit_breaker("historian")
        assert circuit_breaker is not None  # Type safety assertion

        # Force circuit breaker open
        for _ in range(3):  # Default failure threshold
            circuit_breaker.record_failure()

        @circuit_breaker_policy("historian")
        async def test_func() -> Any:
            return await mock_function()

        with pytest.raises(LangGraphExecutionError) as exc_info:
            await test_func()

        assert "Circuit breaker open" in str(exc_info.value)
        assert mock_function.call_count == 0  # Function not called

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_recording(
        self, mock_function: AsyncMock
    ) -> None:
        """Test circuit breaker records failures."""
        mock_function.side_effect = ValueError("test error")

        @circuit_breaker_policy("historian")
        async def test_func() -> Any:
            return await mock_function()

        with pytest.raises(LangGraphExecutionError):
            await test_func()

        # Circuit breaker should have recorded the failure
        manager = get_error_policy_manager()
        circuit_breaker = manager.get_circuit_breaker("historian")
        assert circuit_breaker is not None  # Type safety assertion
        assert circuit_breaker.failure_count > 0


class TestTimeoutDecorator:
    """Test timeout_policy decorator."""

    @pytest.fixture
    def mock_function(self) -> Any:
        """Fixture for mock async function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_timeout_success_within_limit(self, mock_function: AsyncMock) -> None:
        """Test successful execution within timeout."""
        mock_function.return_value = "success"

        @timeout_policy("refiner")  # Has 30s timeout by default
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        assert result == "success"
        assert mock_function.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_no_limit(self, mock_function: AsyncMock) -> None:
        """Test execution with no timeout limit."""
        mock_function.return_value = "success"

        # Create policy with no timeout
        with patch(
            "cognivault.orchestration.error_policies.get_error_policy_manager"
        ) as mock_manager:
            mock_policy = ErrorPolicy(
                policy_type=ErrorPolicyType.RETRY_WITH_BACKOFF, timeout_seconds=None
            )
            mock_manager.return_value.get_policy.return_value = mock_policy

            @timeout_policy("test_node")
            async def test_func() -> Any:
                return await mock_function()

            result = await test_func()

            assert result == "success"
            assert mock_function.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self) -> None:
        """Test timeout exception when limit exceeded."""

        async def slow_function() -> str:
            await asyncio.sleep(2)  # Sleep longer than timeout
            return "success"

        # Create policy with very short timeout
        with patch(
            "cognivault.orchestration.error_policies.get_error_policy_manager"
        ) as mock_manager:
            mock_policy = ErrorPolicy(
                policy_type=ErrorPolicyType.RETRY_WITH_BACKOFF,
                timeout_seconds=0.1,  # Very short timeout
            )
            mock_manager.return_value.get_policy.return_value = mock_policy

            @timeout_policy("test_node")
            async def test_func() -> Any:
                return await slow_function()

            with pytest.raises(LangGraphExecutionError) as exc_info:
                await test_func()

            assert "Timeout after" in str(exc_info.value)
            assert exc_info.value.node_name == "test_node"


class TestComprehensiveDecorator:
    """Test comprehensive_error_policy decorator."""

    @pytest.fixture
    def mock_function(self) -> Any:
        """Fixture for mock async function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_comprehensive_policy_success(self, mock_function: AsyncMock) -> None:
        """Test comprehensive policy with successful execution."""
        mock_function.return_value = "success"

        @comprehensive_error_policy("refiner")
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        assert result == "success"
        assert mock_function.call_count == 1

    @pytest.mark.asyncio
    async def test_comprehensive_policy_with_retries(
        self, mock_function: AsyncMock
    ) -> None:
        """Test comprehensive policy with retries."""
        # Fail once, then succeed
        mock_function.side_effect = [ValueError("error"), "success"]

        @comprehensive_error_policy("refiner")
        async def test_func() -> Any:
            return await mock_function()

        result = await test_func()

        assert result == "success"
        assert mock_function.call_count == 2


class TestFallbackHandling:
    """Test fallback handling functions."""

    def test_skip_node_fallback(self) -> None:
        """Test skip node fallback strategy."""
        error = ValueError("test error")
        result = handle_node_fallback("test_node", error, FallbackStrategy.SKIP_NODE)

        assert result is None

    def test_cached_result_fallback(self) -> None:
        """Test cached result fallback strategy."""
        error = ValueError("test error")
        result = handle_node_fallback(
            "test_node", error, FallbackStrategy.USE_CACHED_RESULT
        )

        assert isinstance(result, dict)
        assert result["fallback"] == "cached_result"
        assert result["node"] == "test_node"

    def test_partial_result_fallback(self) -> None:
        """Test partial result fallback strategy."""
        error = ValueError("test error")
        result = handle_node_fallback(
            "test_node", error, FallbackStrategy.PARTIAL_RESULT
        )

        assert isinstance(result, dict)
        assert result["fallback"] == "partial_result"
        assert result["node"] == "test_node"
        assert result["error"] == "test error"

    def test_substitute_agent_fallback(self) -> None:
        """Test substitute agent fallback strategy."""
        error = ValueError("test error")
        result = handle_node_fallback(
            "test_node", error, FallbackStrategy.SUBSTITUTE_AGENT
        )

        assert isinstance(result, dict)
        assert result["fallback"] == "substitute_agent"
        assert result["node"] == "test_node"

    def test_unknown_fallback_strategy(self) -> None:
        """Test unknown fallback strategy."""
        error = ValueError("test error")
        # Create a mock fallback strategy that's not in the enum
        unknown_strategy: Mock = Mock()
        unknown_strategy.value = "unknown"

        result = handle_node_fallback("test_node", error, unknown_strategy)

        assert result is None


class TestErrorStatistics:
    """Test error statistics collection."""

    def test_get_error_statistics(self) -> None:
        """Test error statistics collection."""
        stats = get_error_statistics()

        assert "policies_configured" in stats
        assert "circuit_breakers_active" in stats
        assert "circuit_breaker_states" in stats

        assert isinstance(stats["policies_configured"], int)
        assert isinstance(stats["circuit_breakers_active"], int)
        assert isinstance(stats["circuit_breaker_states"], dict)

        # Should have at least the default policies
        assert (
            stats["policies_configured"] >= 4
        )  # refiner, critic, historian, synthesis

    def test_circuit_breaker_state_details(self) -> None:
        """Test detailed circuit breaker state information."""
        stats = get_error_statistics()

        # Historian should have circuit breaker by default
        if "historian" in stats["circuit_breaker_states"]:
            historian_state = stats["circuit_breaker_states"]["historian"]

            assert "state" in historian_state
            assert "failure_count" in historian_state
            assert "success_count" in historian_state
            assert "last_failure_time" in historian_state

            assert isinstance(historian_state["failure_count"], int)
            assert isinstance(historian_state["success_count"], int)
            assert isinstance(historian_state["last_failure_time"], float)


if __name__ == "__main__":
    pytest.main([__file__])
