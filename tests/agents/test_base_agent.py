import pytest
import asyncio
from unittest.mock import patch
from cognivault.agents.base_agent import BaseAgent, RetryConfig, CircuitBreakerState
from cognivault.context import AgentContext
from cognivault.exceptions import (
    AgentExecutionError,
    AgentTimeoutError,
    LLMError,
    RetryPolicy,
)


# Test fixtures and helper classes
class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing BaseAgent functionality."""

    def __init__(self, name: str = "TestAgent", **kwargs):
        super().__init__(name, **kwargs)
        self.run_called = False

    async def run(self, context: AgentContext) -> AgentContext:
        self.run_called = True
        context.agent_outputs[self.name] = "test output"
        return context


class FailingAgent(BaseAgent):
    """Agent that always fails for testing error handling."""

    def __init__(
        self,
        name: str = "FailingAgent",
        failure_type: Exception = Exception("Test failure"),
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.failure_type = failure_type
        self.attempts = 0

    async def run(self, context: AgentContext) -> AgentContext:
        self.attempts += 1
        raise self.failure_type


class TimeoutAgent(BaseAgent):
    """Agent that times out for testing timeout handling."""

    def __init__(self, name: str = "TimeoutAgent", delay: float = 5.0, **kwargs):
        super().__init__(name, **kwargs)
        self.delay = delay

    async def run(self, context: AgentContext) -> AgentContext:
        await asyncio.sleep(self.delay)
        context.agent_outputs[self.name] = "Should not reach here"
        return context


class FlakyAgent(BaseAgent):
    """Agent that fails N times before succeeding."""

    def __init__(
        self, name: str = "FlakyAgent", failures_before_success: int = 2, **kwargs
    ):
        super().__init__(name, **kwargs)
        self.failures_before_success = failures_before_success
        self.attempts = 0

    async def run(self, context: AgentContext) -> AgentContext:
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise Exception(f"Attempt {self.attempts} failed")
        context.agent_outputs[self.name] = f"Success after {self.attempts} attempts"
        return context


# Basic functionality tests


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_base_agent_run_concrete(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent(name="TestAgent")
    context = AgentContext(query="test")
    result = await agent.run(context)

    assert isinstance(result, AgentContext)
    assert result.agent_outputs["TestAgent"] == "test output"
    assert agent.run_called


def test_base_agent_run_abstract_error():
    with pytest.raises(TypeError):
        BaseAgent(name="Test")  # Directly instantiating BaseAgent should fail


# RetryConfig tests
def test_retry_config_defaults():
    config = RetryConfig()
    assert config.max_retries == 3
    assert config.base_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_backoff is True
    assert config.jitter is True


def test_retry_config_custom():
    config = RetryConfig(
        max_retries=5,
        base_delay=2.0,
        max_delay=120.0,
        exponential_backoff=False,
        jitter=False,
    )
    assert config.max_retries == 5
    assert config.base_delay == 2.0
    assert config.max_delay == 120.0
    assert config.exponential_backoff is False
    assert config.jitter is False


# CircuitBreakerState tests
def test_circuit_breaker_initialization():
    cb = CircuitBreakerState()
    assert cb.failure_threshold == 5
    assert cb.recovery_timeout == 300.0
    assert cb.failure_count == 0
    assert cb.last_failure_time is None
    assert cb.is_open is False


def test_circuit_breaker_custom_initialization():
    cb = CircuitBreakerState(failure_threshold=3, recovery_timeout=60.0)
    assert cb.failure_threshold == 3
    assert cb.recovery_timeout == 60.0


def test_circuit_breaker_record_success():
    cb = CircuitBreakerState()
    cb.failure_count = 3
    cb.is_open = True

    cb.record_success()

    assert cb.failure_count == 0
    assert cb.is_open is False
    assert cb.last_failure_time is None


def test_circuit_breaker_record_failure():
    cb = CircuitBreakerState(failure_threshold=2)

    # First failure
    cb.record_failure()
    assert cb.failure_count == 1
    assert cb.is_open is False
    assert cb.last_failure_time is not None

    # Second failure - should open circuit
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.is_open is True


def test_circuit_breaker_can_execute():
    cb = CircuitBreakerState()

    # Initially can execute
    assert cb.can_execute() is True

    # After opening, cannot execute
    cb.is_open = True
    assert cb.can_execute() is False


@patch("cognivault.agents.base_agent.datetime")
def test_circuit_breaker_recovery_timeout(mock_datetime):
    from datetime import datetime, timezone, timedelta

    # Mock time progression
    start_time = datetime.now(timezone.utc)
    mock_datetime.now.return_value = start_time

    cb = CircuitBreakerState(failure_threshold=2, recovery_timeout=60.0)
    cb.record_failure()
    cb.record_failure()  # Should open circuit now (threshold=2)

    assert cb.is_open is True
    assert cb.can_execute() is False

    # Time passes but not enough
    mock_datetime.now.return_value = start_time + timedelta(seconds=30)
    assert cb.can_execute() is False

    # Enough time passes
    mock_datetime.now.return_value = start_time + timedelta(seconds=70)
    assert cb.can_execute() is True
    assert cb.is_open is False
    assert cb.failure_count == 0


# BaseAgent initialization tests
def test_base_agent_initialization_defaults():
    agent = ConcreteAgent("TestAgent")
    assert agent.name == "TestAgent"
    assert agent.timeout_seconds == 30.0
    assert agent.execution_count == 0
    assert agent.success_count == 0
    assert agent.failure_count == 0
    assert agent.circuit_breaker is not None
    assert isinstance(agent.retry_config, RetryConfig)


def test_base_agent_initialization_custom():
    retry_config = RetryConfig(max_retries=5)
    agent = ConcreteAgent(
        "CustomAgent",
        retry_config=retry_config,
        timeout_seconds=60.0,
        enable_circuit_breaker=False,
    )
    assert agent.name == "CustomAgent"
    assert agent.timeout_seconds == 60.0
    assert agent.retry_config == retry_config
    assert agent.circuit_breaker is None


def test_generate_step_id():
    agent = ConcreteAgent("TestAgent")
    step_id = agent.generate_step_id()
    assert step_id.startswith("testagent_")
    assert len(step_id) == 18  # "testagent_" + 8 hex chars


# run_with_retry tests
@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_success(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    context = AgentContext(query="test")

    result = await agent.run_with_retry(context)

    assert result is not None
    assert agent.execution_count == 1
    assert agent.success_count == 1
    assert agent.failure_count == 0
    assert agent.run_called


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_with_step_id(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    context = AgentContext(query="test")
    step_id = "custom_step_123"

    await agent.run_with_retry(context, step_id=step_id)

    # Check that step metadata was added to context
    metadata_key = f"{agent.name}_step_metadata"
    assert metadata_key in context.execution_state
    assert context.execution_state[metadata_key]["step_id"] == step_id
    assert context.execution_state[metadata_key]["agent_id"] == agent.name


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_circuit_breaker_open(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    agent.circuit_breaker.is_open = True
    agent.circuit_breaker.failure_count = 5
    context = AgentContext(query="test")

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert "Circuit breaker open" in str(exc_info.value)
    assert exc_info.value.error_code == "circuit_breaker_open"


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_timeout(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = TimeoutAgent("TimeoutAgent", delay=2.0, timeout_seconds=0.1)
    context = AgentContext(query="test")

    with pytest.raises(AgentTimeoutError) as exc_info:
        await agent.run_with_retry(context)

    assert exc_info.value.agent_name == "TimeoutAgent"
    assert exc_info.value.timeout_seconds == 0.1
    assert agent.execution_count == 1
    assert agent.failure_count == 1


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_retries_on_failure(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = FlakyAgent("FlakyAgent", failures_before_success=2)
    agent.retry_config = RetryConfig(max_retries=3, base_delay=0.01)
    context = AgentContext(query="test")

    result = await agent.run_with_retry(context)

    assert agent.attempts == 3  # Failed twice, succeeded on third
    assert agent.execution_count == 1
    assert agent.success_count == 1
    assert result.agent_outputs["FlakyAgent"] == "Success after 3 attempts"


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_max_retries_exceeded(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = FailingAgent("FailingAgent")
    agent.retry_config = RetryConfig(max_retries=2, base_delay=0.01)
    context = AgentContext(query="test")

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert agent.attempts == 3  # Initial + 2 retries
    assert agent.execution_count == 1
    assert agent.failure_count == 1
    assert "Agent execution failed" in str(exc_info.value)


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_llm_error_passthrough(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    llm_error = LLMError("Test LLM error", "llm_test_error")
    agent = FailingAgent("FailingAgent", failure_type=llm_error)
    context = AgentContext(query="test")

    with pytest.raises(LLMError):
        await agent.run_with_retry(context)


# _should_retry_exception tests
def test_should_retry_exception_with_retry_policy():
    agent = ConcreteAgent("TestAgent")

    # Create mock exception with retry policy
    class RetryableException(Exception):
        def __init__(self):
            self.retry_policy = RetryPolicy.BACKOFF

    class NonRetryableException(Exception):
        def __init__(self):
            self.retry_policy = RetryPolicy.NEVER

    assert agent._should_retry_exception(RetryableException()) is True
    assert agent._should_retry_exception(NonRetryableException()) is False


def test_should_retry_exception_standard_exceptions():
    agent = ConcreteAgent("TestAgent")

    # Should retry
    assert agent._should_retry_exception(asyncio.TimeoutError()) is True
    assert agent._should_retry_exception(ConnectionError()) is True
    assert agent._should_retry_exception(Exception("Unknown")) is True

    # Should not retry
    assert agent._should_retry_exception(ValueError()) is False
    assert agent._should_retry_exception(TypeError()) is False
    assert agent._should_retry_exception(AttributeError()) is False


# _handle_retry_delay tests
@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_handle_retry_delay_zero_delay(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    agent.retry_config.base_delay = 0

    # Should return immediately
    start_time = asyncio.get_event_loop().time()
    await agent._handle_retry_delay(0)
    end_time = asyncio.get_event_loop().time()

    assert end_time - start_time < 0.01  # Very fast


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_handle_retry_delay_exponential_backoff(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    agent.retry_config = RetryConfig(
        base_delay=0.01, exponential_backoff=True, jitter=False, max_delay=1.0
    )

    # Test exponential increase
    with patch("asyncio.sleep") as mock_sleep:
        await agent._handle_retry_delay(0)
        mock_sleep.assert_called_with(0.01)

        await agent._handle_retry_delay(1)
        mock_sleep.assert_called_with(0.02)

        await agent._handle_retry_delay(2)
        mock_sleep.assert_called_with(0.04)


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_handle_retry_delay_with_jitter(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    agent.retry_config = RetryConfig(
        base_delay=0.1, exponential_backoff=False, jitter=True
    )

    with patch("asyncio.sleep") as mock_sleep:
        await agent._handle_retry_delay(0)
        # Should be called with base_delay ± 25%
        call_args = mock_sleep.call_args[0][0]
        assert 0.075 <= call_args <= 0.125


# get_execution_stats tests
def test_get_execution_stats():
    agent = ConcreteAgent("TestAgent")
    agent.execution_count = 10
    agent.success_count = 8
    agent.failure_count = 2

    stats = agent.get_execution_stats()

    assert stats["agent_name"] == "TestAgent"
    assert stats["execution_count"] == 10
    assert stats["success_count"] == 8
    assert stats["failure_count"] == 2
    assert stats["success_rate"] == 0.8
    assert "retry_config" in stats
    assert "circuit_breaker" in stats


def test_get_execution_stats_no_executions():
    agent = ConcreteAgent("TestAgent")

    stats = agent.get_execution_stats()

    assert stats["success_rate"] == 0.0


def test_get_execution_stats_no_circuit_breaker():
    agent = ConcreteAgent("TestAgent", enable_circuit_breaker=False)

    stats = agent.get_execution_stats()

    assert "circuit_breaker" not in stats


# Additional coverage tests for missing lines
def test_circuit_breaker_threshold_behavior():
    cb = CircuitBreakerState(failure_threshold=3)

    # Test threshold exactly
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False

    cb.record_failure()  # Should reach threshold and open
    assert cb.is_open is True


def test_circuit_breaker_time_calculation_edge_cases():
    cb = CircuitBreakerState(recovery_timeout=60.0)
    cb.is_open = True
    cb.last_failure_time = None  # Edge case

    # Should not execute when last_failure_time is None but circuit is open
    assert cb.can_execute() is False


def test_should_retry_connection_error():
    agent = ConcreteAgent("TestAgent")

    # Test ConnectionError specifically
    assert agent._should_retry_exception(ConnectionError("Network error")) is True


def test_should_retry_default_exception_behavior():
    agent = ConcreteAgent("TestAgent")

    # Test default behavior for unknown exception types
    class CustomException(Exception):
        pass

    assert agent._should_retry_exception(CustomException("Unknown error")) is True


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_circuit_breaker_time_calculation(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")

    # Force circuit breaker to have a failure time for time calculation
    from datetime import datetime, timezone, timedelta

    agent.circuit_breaker.is_open = True
    agent.circuit_breaker.failure_count = 5
    agent.circuit_breaker.last_failure_time = datetime.now(timezone.utc) - timedelta(
        seconds=100
    )

    context = AgentContext(query="test")

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert "Circuit breaker open" in str(exc_info.value)
    # Should calculate time_remaining_seconds
    assert "time_remaining_seconds" in exc_info.value.context


# Integration tests
@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_context_execution_tracking(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    context = AgentContext(query="test")

    await agent.run_with_retry(context)

    # Check that context execution tracking was called
    assert f"{agent.name}_start_time" in context.execution_state
    assert f"{agent.name}_end_time" in context.execution_state

    # Check step metadata
    metadata_key = f"{agent.name}_step_metadata"
    assert metadata_key in context.execution_state
    metadata = context.execution_state[metadata_key]
    assert metadata["agent_id"] == agent.name
    assert metadata["completed"] is True
    assert "step_id" in metadata
    assert "start_time" in metadata
    assert "end_time" in metadata


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_context_trace_logging(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")
    agent = ConcreteAgent("TestAgent")
    context = AgentContext(query="test")

    await agent.run_with_retry(context)

    # Check that trace was logged
    assert agent.name in context.agent_trace
    trace_entries = context.agent_trace[agent.name]
    assert len(trace_entries) == 1

    trace = trace_entries[0]
    assert trace["input"]["attempt"] == 1
    assert trace["output"]["success"] is True
    assert trace["output"]["attempts_used"] == 1
    assert "execution_time_seconds" in trace["output"]


@pytest.mark.parametrize("anyio_backend", ["asyncio", "trio"])
@pytest.mark.anyio
async def test_run_with_retry_unreachable_fallback(anyio_backend):
    if anyio_backend == "trio":
        pytest.skip("Trio not supported due to asyncio-specific constructs.")

    class NoExceptionAgent(BaseAgent):
        def __init__(self):
            super().__init__("NoExceptionAgent")

        async def run(self, context: AgentContext) -> AgentContext:
            raise Exception("non-retryable")  # Non-agent exception not caught

    agent = NoExceptionAgent()
    agent.retry_config = RetryConfig(max_retries=-1)  # Forces loop to skip entirely

    context = AgentContext(query="force fallback path")
    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert "Agent execution failed after" in str(exc_info.value)
