import pytest
from typing import Any, Dict, Generator
import asyncio
from unittest.mock import patch
from cognivault.agents.base_agent import (
    BaseAgent,
    RetryConfig,
    CircuitBreakerState,
    NodeType,
    NodeInputSchema,
    NodeOutputSchema,
    LangGraphNodeDefinition,
)
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)
from cognivault.exceptions import (
    AgentExecutionError,
    AgentTimeoutError,
    LLMError,
    RetryPolicy,
)

from tests.factories import AgentContextFactory, AgentContextPatterns


@pytest.fixture(autouse=True)
def enable_events(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Ensure events are enabled for all BaseAgent tests."""
    monkeypatch.setenv("COGNIVAULT_EVENTS_ENABLED", "true")
    monkeypatch.setenv("COGNIVAULT_EVENTS_IN_MEMORY", "true")

    # Reset and re-enable global event emitter for each test
    from cognivault.events import reset_global_event_emitter, get_global_event_emitter

    reset_global_event_emitter()
    emitter = get_global_event_emitter()
    emitter.enable()

    yield  # Run the test

    # Reset again after the test to avoid interference with subsequent tests
    reset_global_event_emitter()


# Test fixtures and helper classes
class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing BaseAgent functionality."""

    def __init__(self, name: str = "TestAgent", **kwargs: Any) -> None:
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
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.failure_type = failure_type
        self.attempts = 0

    async def run(self, context: AgentContext) -> AgentContext:
        self.attempts += 1
        raise self.failure_type


class TimeoutAgent(BaseAgent):
    """Agent that times out for testing timeout handling."""

    def __init__(
        self, name: str = "TimeoutAgent", delay: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name, **kwargs)
        self.delay = delay

    async def run(self, context: AgentContext) -> AgentContext:
        await asyncio.sleep(self.delay)
        context.agent_outputs[self.name] = "Should not reach here"
        return context


class FlakyAgent(BaseAgent):
    """Agent that fails N times before succeeding."""

    def __init__(
        self, name: str = "FlakyAgent", failures_before_success: int = 2, **kwargs: Any
    ) -> None:
        super().__init__(name, **kwargs)
        self.failures_before_success = failures_before_success
        self.attempts = 0

    async def run(self, context: AgentContext) -> AgentContext:
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise Exception(f"Attempt {self.attempts} failed")
        context.agent_outputs[self.name] = f"Success after {self.attempts} attempts"
        return context


class CustomNodeAgent(BaseAgent):
    """Agent with custom node metadata for testing."""

    def __init__(self, name: str = "CustomNodeAgent", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

    async def run(self, context: AgentContext) -> AgentContext:
        context.agent_outputs[self.name] = "custom node output"
        return context

    def define_node_metadata(self) -> Dict[str, Any]:
        """Override to provide custom node metadata."""
        return {
            "node_type": NodeType.DECISION,
            "description": "Custom decision-making agent for testing",
            "inputs": [
                NodeInputSchema(
                    name="context",
                    description="Input context with decision criteria",
                    required=True,
                    type_hint="AgentContext",
                ),
                NodeInputSchema(
                    name="threshold",
                    description="Decision threshold parameter",
                    required=False,
                    type_hint="float",
                ),
            ],
            "outputs": [
                NodeOutputSchema(
                    name="context",
                    description="Updated context with decision result",
                    type_hint="AgentContext",
                ),
                NodeOutputSchema(
                    name="decision",
                    description="The decision made by the agent",
                    type_hint="bool",
                ),
            ],
            "dependencies": ["refiner", "historian"],
            "tags": ["decision", "custom", "test"],
        }


# Basic functionality tests


@pytest.mark.asyncio
async def test_base_agent_run_concrete() -> None:
    agent = ConcreteAgent(name="TestAgent")
    context = AgentContextPatterns.simple_query("test")
    result = await agent.run(context)

    assert isinstance(result, AgentContext)
    assert result.agent_outputs["TestAgent"] == "test output"
    assert agent.run_called


def test_base_agent_run_abstract_error() -> None:
    with pytest.raises(TypeError):
        BaseAgent(name="Test")  # type: ignore[abstract]  # Directly instantiating BaseAgent should fail


# RetryConfig tests
def test_retry_config_defaults() -> None:
    config = RetryConfig()
    assert config.max_retries == 3
    assert config.base_delay == 1.0
    assert config.max_delay == 60.0
    assert config.exponential_backoff is True
    assert config.jitter is True


def test_retry_config_custom() -> None:
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
def test_circuit_breaker_initialization() -> None:
    cb = CircuitBreakerState()
    assert cb.failure_threshold == 5
    assert cb.recovery_timeout == 300.0
    assert cb.failure_count == 0
    assert cb.last_failure_time is None
    assert cb.is_open is False


def test_circuit_breaker_custom_initialization() -> None:
    cb = CircuitBreakerState(failure_threshold=3, recovery_timeout=60.0)
    assert cb.failure_threshold == 3
    assert cb.recovery_timeout == 60.0


def test_circuit_breaker_record_success() -> None:
    cb = CircuitBreakerState()
    cb.failure_count = 3
    cb.is_open = True

    cb.record_success()

    assert cb.failure_count == 0
    assert cb.is_open is False
    assert cb.last_failure_time is None


def test_circuit_breaker_record_failure() -> None:
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


def test_circuit_breaker_can_execute() -> None:
    cb = CircuitBreakerState()

    # Initially can execute
    assert cb.can_execute() is True

    # After opening, cannot execute
    cb.is_open = True
    assert cb.can_execute() is False


@patch("cognivault.agents.base_agent.datetime")
def test_circuit_breaker_recovery_timeout(mock_datetime: Any) -> None:
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
def test_base_agent_initialization_defaults() -> None:
    agent = ConcreteAgent("TestAgent")
    assert agent.name == "TestAgent"
    assert agent.timeout_seconds == 30.0
    assert agent.execution_count == 0
    assert agent.success_count == 0
    assert agent.failure_count == 0
    assert agent.circuit_breaker is not None
    assert isinstance(agent.retry_config, RetryConfig)


def test_base_agent_initialization_custom() -> None:
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


def test_generate_step_id() -> None:
    agent = ConcreteAgent("TestAgent")
    step_id = agent.generate_step_id()
    assert step_id.startswith("testagent_")
    assert len(step_id) == 18  # "testagent_" + 8 hex chars


# run_with_retry tests
@pytest.mark.asyncio
async def test_run_with_retry_success() -> None:
    """Test successful retry operation using factory patterns."""
    agent = ConcreteAgent("TestAgent")
    context = AgentContextPatterns.simple_query("test")  # Using factory pattern

    result = await agent.run_with_retry(context)

    assert result is not None
    assert agent.execution_count == 1
    assert agent.success_count == 1
    assert agent.failure_count == 0
    assert agent.run_called


@pytest.mark.asyncio
async def test_run_with_retry_with_step_id() -> None:
    agent = ConcreteAgent("TestAgent")
    context = AgentContextPatterns.simple_query("test")
    step_id = "custom_step_123"

    await agent.run_with_retry(context, step_id=step_id)

    # Check that step metadata was added to context
    metadata_key = f"{agent.name}_step_metadata"
    assert metadata_key in context.execution_state
    assert context.execution_state[metadata_key]["step_id"] == step_id
    assert context.execution_state[metadata_key]["agent_id"] == agent.name


@pytest.mark.asyncio
async def test_run_with_retry_circuit_breaker_open() -> None:
    agent = ConcreteAgent("TestAgent")
    assert agent.circuit_breaker is not None
    circuit_breaker = agent.circuit_breaker
    circuit_breaker.is_open = True
    circuit_breaker.failure_count = 5
    context = AgentContextPatterns.simple_query("test")

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert "Circuit breaker open" in str(exc_info.value)
    assert exc_info.value.error_code == "circuit_breaker_open"


@pytest.mark.asyncio
async def test_run_with_retry_timeout() -> None:
    agent = TimeoutAgent("TimeoutAgent", delay=2.0, timeout_seconds=0.1)
    context = AgentContextPatterns.simple_query("test")

    with pytest.raises(AgentTimeoutError) as exc_info:
        await agent.run_with_retry(context)

    assert exc_info.value.agent_name == "TimeoutAgent"
    assert exc_info.value.timeout_seconds == 0.1
    assert agent.execution_count == 1
    assert agent.failure_count == 1


@pytest.mark.asyncio
async def test_run_with_retry_retries_on_failure() -> None:
    agent = FlakyAgent("FlakyAgent", failures_before_success=2)
    agent.retry_config = RetryConfig(max_retries=3, base_delay=0.01)
    context = AgentContextPatterns.simple_query("test")

    result = await agent.run_with_retry(context)

    assert agent.attempts == 3  # Failed twice, succeeded on third
    assert agent.execution_count == 1
    assert agent.success_count == 1
    assert result.agent_outputs["FlakyAgent"] == "Success after 3 attempts"


@pytest.mark.asyncio
async def test_run_with_retry_max_retries_exceeded() -> None:
    agent = FailingAgent("FailingAgent")
    agent.retry_config = RetryConfig(max_retries=2, base_delay=0.01)
    context = AgentContextPatterns.simple_query("test")

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert agent.attempts == 3  # Initial + 2 retries
    assert agent.execution_count == 1
    assert agent.failure_count == 1
    assert "Agent execution failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_with_retry_llm_error_passthrough() -> None:
    llm_error = LLMError("Test LLM error", "llm_test_error")
    agent = FailingAgent("FailingAgent", failure_type=llm_error)
    context = AgentContextPatterns.simple_query("test")

    with pytest.raises(LLMError):
        await agent.run_with_retry(context)


# _should_retry_exception tests
def test_should_retry_exception_with_retry_policy() -> None:
    agent = ConcreteAgent("TestAgent")

    # Create mock exception with retry policy
    class RetryableException(Exception):
        def __init__(self) -> None:
            self.retry_policy = RetryPolicy.BACKOFF

    class NonRetryableException(Exception):
        def __init__(self) -> None:
            self.retry_policy = RetryPolicy.NEVER

    assert agent._should_retry_exception(RetryableException()) is True
    assert agent._should_retry_exception(NonRetryableException()) is False


def test_should_retry_exception_standard_exceptions() -> None:
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
@pytest.mark.asyncio
async def test_handle_retry_delay_zero_delay() -> None:
    agent = ConcreteAgent("TestAgent")
    agent.retry_config.base_delay = 0

    # Should return immediately
    start_time = asyncio.get_event_loop().time()
    await agent._handle_retry_delay(0)
    end_time = asyncio.get_event_loop().time()

    assert end_time - start_time < 0.01  # Very fast


@pytest.mark.asyncio
async def test_handle_retry_delay_exponential_backoff() -> None:
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


@pytest.mark.asyncio
async def test_handle_retry_delay_with_jitter() -> None:
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
def test_get_execution_stats() -> None:
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


def test_get_execution_stats_no_executions() -> None:
    agent = ConcreteAgent("TestAgent")

    stats = agent.get_execution_stats()

    assert stats["success_rate"] == 0.0


def test_get_execution_stats_no_circuit_breaker() -> None:
    agent = ConcreteAgent("TestAgent", enable_circuit_breaker=False)

    stats = agent.get_execution_stats()

    assert "circuit_breaker" not in stats


# Additional coverage tests for missing lines
def test_circuit_breaker_threshold_behavior() -> None:
    cb = CircuitBreakerState(failure_threshold=3)

    # Test threshold exactly
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False

    cb.record_failure()  # Should reach threshold and open
    assert cb.is_open is True


def test_circuit_breaker_time_calculation_edge_cases() -> None:
    cb = CircuitBreakerState(recovery_timeout=60.0)
    cb.is_open = True
    cb.last_failure_time = None  # Edge case

    # Should not execute when last_failure_time is None but circuit is open
    assert cb.can_execute() is False


def test_should_retry_connection_error() -> None:
    agent = ConcreteAgent("TestAgent")

    # Test ConnectionError specifically
    assert agent._should_retry_exception(ConnectionError("Network error")) is True


def test_should_retry_default_exception_behavior() -> None:
    agent = ConcreteAgent("TestAgent")

    # Test default behavior for unknown exception types
    class CustomException(Exception):
        pass

    assert agent._should_retry_exception(CustomException("Unknown error")) is True


@pytest.mark.asyncio
async def test_run_with_retry_circuit_breaker_time_calculation() -> None:
    agent = ConcreteAgent("TestAgent")

    # Force circuit breaker to have a failure time for time calculation
    from datetime import datetime, timezone, timedelta

    assert agent.circuit_breaker is not None
    circuit_breaker = agent.circuit_breaker
    circuit_breaker.is_open = True
    circuit_breaker.failure_count = 5
    circuit_breaker.last_failure_time = datetime.now(timezone.utc) - timedelta(
        seconds=100
    )

    context = AgentContextPatterns.simple_query("test")

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert "Circuit breaker open" in str(exc_info.value)
    # Should calculate time_remaining_seconds
    assert "time_remaining_seconds" in exc_info.value.context


# Integration tests
@pytest.mark.asyncio
async def test_context_execution_tracking() -> None:
    agent = ConcreteAgent("TestAgent")
    context = AgentContextPatterns.simple_query("test")

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


@pytest.mark.asyncio
async def test_context_trace_logging() -> None:
    agent = ConcreteAgent("TestAgent")
    context = AgentContextPatterns.simple_query("test")

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


@pytest.mark.asyncio
async def test_run_with_retry_unreachable_fallback() -> None:
    class NoExceptionAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__("NoExceptionAgent")

        async def run(self, context: AgentContext) -> AgentContext:
            raise Exception("non-retryable")  # Non-agent exception not caught

    agent = NoExceptionAgent()
    agent.retry_config = RetryConfig(max_retries=-1)  # Forces loop to skip entirely

    context = AgentContextPatterns.force_fallback()
    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.run_with_retry(context)

    assert "Agent execution failed after" in str(exc_info.value)


# LangGraph invoke() method tests
@pytest.mark.asyncio
async def test_invoke_method_basic() -> None:
    """Test basic invoke() method functionality."""

    agent = ConcreteAgent("TestAgent")
    context = AgentContextPatterns.invoke_test()

    result = await agent.invoke(context)

    assert isinstance(result, AgentContext)
    assert result.agent_outputs["TestAgent"] == "test output"
    assert agent.run_called
    assert agent.execution_count == 1
    assert agent.success_count == 1


@pytest.mark.asyncio
async def test_invoke_method_with_config() -> None:
    """Test invoke() method with configuration parameters."""

    agent = ConcreteAgent("TestAgent")
    context = AgentContextPatterns.invoke_with_config()
    config = {"step_id": "custom_invoke_step_123", "timeout_seconds": 45.0}

    result = await agent.invoke(context, config=config)

    assert isinstance(result, AgentContext)
    assert agent.run_called

    # Check that step metadata was added with custom step_id
    metadata_key = f"{agent.name}_step_metadata"
    assert metadata_key in context.execution_state
    assert context.execution_state[metadata_key]["step_id"] == "custom_invoke_step_123"

    # Verify timeout was restored
    assert agent.timeout_seconds == 30.0  # Should be back to default


@pytest.mark.asyncio
async def test_invoke_method_timeout_override() -> None:
    """Test that invoke() method can override timeout and restores it properly."""

    agent = ConcreteAgent("TestAgent", timeout_seconds=30.0)
    context = AgentContextPatterns.simple_query("test timeout override")
    config = {"timeout_seconds": 60.0}

    # Capture the timeout during execution
    original_timeout = agent.timeout_seconds

    await agent.invoke(context, config=config)

    # Verify timeout was restored to original value
    assert agent.timeout_seconds == original_timeout


@pytest.mark.asyncio
async def test_invoke_method_no_config() -> None:
    """Test invoke() method without config parameter."""

    agent = ConcreteAgent("TestAgent")
    context = AgentContextPatterns.simple_query("test invoke no config")

    result = await agent.invoke(context, config=None)

    assert isinstance(result, AgentContext)
    assert agent.run_called
    assert agent.execution_count == 1


@pytest.mark.asyncio
async def test_invoke_method_error_propagation() -> None:
    """Test that invoke() method properly propagates errors."""

    agent = FailingAgent("FailingAgent")
    agent.retry_config = RetryConfig(max_retries=1, base_delay=0.01)
    context = AgentContextPatterns.simple_query("test error propagation")

    with pytest.raises(AgentExecutionError):
        await agent.invoke(context)

    assert agent.execution_count == 1
    assert agent.failure_count == 1


@pytest.mark.asyncio
async def test_invoke_method_timeout_override_with_exception() -> None:
    """Test that timeout is restored even when an exception occurs."""

    agent = FailingAgent("FailingAgent", timeout_seconds=30.0)
    agent.retry_config = RetryConfig(max_retries=0, base_delay=0.01)  # Fail immediately
    context = AgentContextPatterns.simple_query("test timeout restore on exception")
    config = {"timeout_seconds": 60.0}

    original_timeout = agent.timeout_seconds

    with pytest.raises(AgentExecutionError):
        await agent.invoke(context, config=config)

    # Verify timeout was restored even after exception
    assert agent.timeout_seconds == original_timeout


@pytest.mark.asyncio
async def test_invoke_method_preserves_all_functionality() -> None:
    """Test that invoke() preserves all BaseAgent functionality like retry, circuit breaker, etc."""

    # Use FlakyAgent to test retry behavior through invoke()
    agent = FlakyAgent("FlakyAgent", failures_before_success=1)
    agent.retry_config = RetryConfig(max_retries=2, base_delay=0.01)
    context = AgentContextPatterns.simple_query("test invoke preserves functionality")

    result = await agent.invoke(context)

    assert agent.attempts == 2  # Failed once, succeeded on second
    assert agent.execution_count == 1
    assert agent.success_count == 1
    assert result.agent_outputs["FlakyAgent"] == "Success after 2 attempts"


# LangGraph Node Metadata Tests
def test_node_type_enum() -> None:
    """Test NodeType enum values."""
    assert NodeType.PROCESSOR.value == "processor"
    assert NodeType.DECISION.value == "decision"
    assert NodeType.TERMINATOR.value == "terminator"
    assert NodeType.AGGREGATOR.value == "aggregator"


def test_node_input_schema() -> None:
    """Test NodeInputSchema creation and defaults."""
    # Required input
    input_schema = NodeInputSchema(
        name="test_input", description="Test input description"
    )
    assert input_schema.name == "test_input"
    assert input_schema.description == "Test input description"
    assert input_schema.required is True  # Default
    assert input_schema.type_hint == "Any"  # Default

    # Optional input with custom type
    optional_input = NodeInputSchema(
        name="optional_input",
        description="Optional input",
        required=False,
        type_hint="str",
    )
    assert optional_input.required is False
    assert optional_input.type_hint == "str"


def test_node_output_schema() -> None:
    """Test NodeOutputSchema creation."""
    output_schema = NodeOutputSchema(
        name="test_output",
        description="Test output description",
        type_hint="AgentContext",
    )
    assert output_schema.name == "test_output"
    assert output_schema.description == "Test output description"
    assert output_schema.type_hint == "AgentContext"


def test_langgraph_node_definition() -> None:
    """Test LangGraphNodeDefinition creation and to_dict method."""
    inputs = [
        NodeInputSchema(
            name="context",
            description="Input context",
            required=True,
            type_hint="AgentContext",
        )
    ]
    outputs = [
        NodeOutputSchema(
            name="context", description="Output context", type_hint="AgentContext"
        )
    ]

    node_def = LangGraphNodeDefinition(
        node_id="test_node",
        node_type=NodeType.PROCESSOR,
        description="Test node description",
        inputs=inputs,
        outputs=outputs,
        dependencies=["node1", "node2"],
        tags=["test", "processor"],
    )

    assert node_def.node_id == "test_node"
    assert node_def.node_type == NodeType.PROCESSOR
    assert node_def.description == "Test node description"
    assert len(node_def.inputs) == 1
    assert len(node_def.outputs) == 1
    assert node_def.dependencies == ["node1", "node2"]
    assert node_def.tags == ["test", "processor"]

    # Test to_dict conversion
    node_dict = node_def.to_dict()
    assert node_dict["node_id"] == "test_node"
    assert node_dict["node_type"] == "processor"
    assert node_dict["description"] == "Test node description"
    assert len(node_dict["inputs"]) == 1
    assert node_dict["inputs"][0]["name"] == "context"
    assert node_dict["inputs"][0]["required"] is True
    assert len(node_dict["outputs"]) == 1
    assert node_dict["outputs"][0]["name"] == "context"
    assert node_dict["dependencies"] == ["node1", "node2"]
    assert node_dict["tags"] == ["test", "processor"]


def test_base_agent_default_node_definition() -> None:
    """Test default node definition generation."""
    agent = ConcreteAgent("TestAgent")

    node_def = agent.get_node_definition()

    assert isinstance(node_def, LangGraphNodeDefinition)
    assert node_def.node_id == "testagent"  # lowercase name
    assert node_def.node_type == NodeType.PROCESSOR  # default
    assert "TestAgent agent for processing context" in node_def.description

    # Check default input
    assert len(node_def.inputs) == 1
    input_schema = node_def.inputs[0]
    assert input_schema.name == "context"
    assert input_schema.required is True
    assert input_schema.type_hint == "AgentContext"

    # Check default output
    assert len(node_def.outputs) == 1
    output_schema = node_def.outputs[0]
    assert output_schema.name == "context"
    assert output_schema.type_hint == "AgentContext"

    # Check defaults
    assert node_def.dependencies == []
    assert "testagent" in node_def.tags
    assert "agent" in node_def.tags


def test_base_agent_node_definition_caching() -> None:
    """Test that node definition is cached after first creation."""
    agent = ConcreteAgent("TestAgent")

    # First call creates and caches
    node_def1 = agent.get_node_definition()

    # Second call returns same object
    node_def2 = agent.get_node_definition()

    assert node_def1 is node_def2  # Same object reference


def test_custom_node_agent_metadata() -> None:
    """Test custom node metadata override."""
    agent = CustomNodeAgent("CustomAgent")

    node_def = agent.get_node_definition()

    assert node_def.node_type == NodeType.DECISION
    assert node_def.description == "Custom decision-making agent for testing"

    # Check custom inputs
    assert len(node_def.inputs) == 2
    context_input = next(inp for inp in node_def.inputs if inp.name == "context")
    assert context_input.description == "Input context with decision criteria"

    threshold_input = next(inp for inp in node_def.inputs if inp.name == "threshold")
    assert threshold_input.required is False
    assert threshold_input.type_hint == "float"

    # Check custom outputs
    assert len(node_def.outputs) == 2
    decision_output = next(out for out in node_def.outputs if out.name == "decision")
    assert decision_output.type_hint == "bool"

    # Check custom dependencies and tags
    assert node_def.dependencies == ["refiner", "historian"]
    assert "decision" in node_def.tags
    assert "custom" in node_def.tags


def test_set_node_definition() -> None:
    """Test setting a custom node definition."""
    agent = ConcreteAgent("TestAgent")

    # Create custom definition
    custom_def = LangGraphNodeDefinition(
        node_id="custom_test",
        node_type=NodeType.TERMINATOR,
        description="Custom terminator node",
        inputs=[],
        outputs=[],
        dependencies=[],
        tags=["custom"],
    )

    agent.set_node_definition(custom_def)

    retrieved_def = agent.get_node_definition()
    assert retrieved_def is custom_def
    assert retrieved_def.node_type == NodeType.TERMINATOR
    assert retrieved_def.description == "Custom terminator node"


def test_validate_node_compatibility() -> None:
    """Test node compatibility validation."""
    agent = ConcreteAgent("TestAgent")

    # Valid context should pass
    valid_context = AgentContextPatterns.simple_query("test")
    assert agent.validate_node_compatibility(valid_context) is True

    # Invalid input should fail
    invalid_input = "not a context"
    assert agent.validate_node_compatibility(invalid_input) is False  # type: ignore[arg-type]


def test_node_definition_to_dict_comprehensive() -> None:
    """Test comprehensive node definition dictionary conversion."""
    agent = CustomNodeAgent("CustomAgent")
    node_def = agent.get_node_definition()

    node_dict = node_def.to_dict()

    # Verify structure
    assert "node_id" in node_dict
    assert "node_type" in node_dict
    assert "description" in node_dict
    assert "inputs" in node_dict
    assert "outputs" in node_dict
    assert "dependencies" in node_dict
    assert "tags" in node_dict

    # Verify input structure
    for inp in node_dict["inputs"]:
        assert "name" in inp
        assert "description" in inp
        assert "required" in inp
        assert "type" in inp

    # Verify output structure
    for out in node_dict["outputs"]:
        assert "name" in out
        assert "description" in out
        assert "type" in out


def test_node_definition_with_empty_override() -> None:
    """Test that empty define_node_metadata override uses defaults."""

    class EmptyMetadataAgent(BaseAgent):
        async def run(self, context: AgentContext) -> AgentContext:
            return context

        def define_node_metadata(self) -> Dict[str, Any]:
            return {}  # Empty override

    agent = EmptyMetadataAgent("EmptyAgent")
    node_def = agent.get_node_definition()

    # Should use defaults
    assert node_def.node_type == NodeType.PROCESSOR
    assert len(node_def.inputs) == 1
    assert len(node_def.outputs) == 1
    assert node_def.dependencies == []


def test_node_definition_partial_override() -> None:
    """Test partial override of node metadata."""

    class PartialMetadataAgent(BaseAgent):
        async def run(self, context: AgentContext) -> AgentContext:
            return context

        def define_node_metadata(self) -> Dict[str, Any]:
            return {
                "node_type": NodeType.AGGREGATOR,
                "tags": ["partial", "test"],
                # Missing other fields - should use defaults
            }

    agent = PartialMetadataAgent("PartialAgent")
    node_def = agent.get_node_definition()

    # Should use overridden values
    assert node_def.node_type == NodeType.AGGREGATOR
    assert "partial" in node_def.tags
    assert "test" in node_def.tags

    # Should use defaults for missing fields
    assert len(node_def.inputs) == 1  # Default input
    assert len(node_def.outputs) == 1  # Default output
    assert node_def.dependencies == []  # Default empty


@pytest.mark.asyncio
async def test_agent_event_emission_includes_actual_content() -> None:
    """
    Test PATTERN 2 fix: Agent-level events should include actual agent output content.

    This test verifies that when a base agent emits an agent_execution_completed event,
    the output_context includes the actual content produced by the agent, not just metadata.
    """
    from unittest.mock import patch, AsyncMock
    from cognivault.context import AgentContext

    class ContentProducingAgent(BaseAgent):
        """Agent that produces specific content for testing."""

        def __init__(self, name: str = "content_agent") -> None:
            super().__init__(name)

        async def run(self, context: AgentContext) -> AgentContext:
            """Produce specific test content."""
            test_content = (
                "This is the actual agent output content that should appear in events"
            )
            context.add_agent_output(self.name, test_content)

            # Add token usage for comprehensive testing
            context.add_agent_token_usage(
                agent_name=self.name,
                input_tokens=150,
                output_tokens=75,
                total_tokens=225,
            )

            return context

    # Create agent and context
    agent = ContentProducingAgent("test_content_agent")
    context = AgentContextFactory.basic(
        query="Test query for content verification",
        user_config={"user_id": "test_user", "session_id": "test_session"},
        metadata={"workflow_metadata": {}},
    )

    # Mock the event emission to capture what gets emitted
    with patch(
        "cognivault.agents.base_agent.emit_agent_execution_completed"
    ) as mock_emit:
        mock_emit.return_value = AsyncMock()

        # Mock the global functions that provide workflow context
        with patch(
            "cognivault.agents.base_agent.get_workflow_id",
            return_value="test-workflow-123",
        ):
            with patch(
                "cognivault.agents.base_agent.get_correlation_id",
                return_value="test-correlation-456",
            ):
                # Execute the agent
                result_context = await agent.run_with_retry(context)

                # Verify the agent ran successfully and produced content
                assert agent.name in result_context.agent_outputs
                assert (
                    result_context.agent_outputs[agent.name]
                    == "This is the actual agent output content that should appear in events"
                )

                # Verify event emission was called
                mock_emit.assert_called_once()

                # Extract the call arguments to verify content inclusion
                call_args, call_kwargs = mock_emit.call_args

                # Verify that agent-level event includes actual content (PATTERN 2 fix)
                output_context = call_kwargs["output_context"]

                # Key assertions for PATTERN 2 fix
                assert "agent_output" in output_context, (
                    "Agent-level event must include actual agent output"
                )
                assert (
                    output_context["agent_output"]
                    == "This is the actual agent output content that should appear in events"
                ), "Agent output content must match what the agent produced"
                assert output_context["output_length"] == 68, (
                    "Output length must reflect actual content length (68 chars)"
                )

                # Verify token usage is included (from PATTERN 1 fix)
                assert output_context["input_tokens"] == 150, (
                    "Input token count must be included"
                )
                assert output_context["output_tokens"] == 75, (
                    "Output token count must be included"
                )
                assert output_context["total_tokens"] == 225, (
                    "Total token count must be included"
                )

                # Verify other expected metadata is still present
                assert "step_id" in output_context
                assert "execution_time_seconds" in output_context
                assert "attempts_used" in output_context
                assert output_context["attempts_used"] == 1  # First attempt succeeded

                # Verify event parameters
                assert call_kwargs["workflow_id"] == "test-workflow-123"
                assert call_kwargs["agent_name"] == "test_content_agent"
                assert call_kwargs["success"] is True
                assert call_kwargs["correlation_id"] == "test-correlation-456"


@pytest.mark.asyncio
async def test_agent_event_emission_with_empty_content() -> None:
    """
    Test edge case: Agent that produces no output should handle gracefully.
    """
    from unittest.mock import patch, AsyncMock
    from cognivault.context import AgentContext

    class EmptyContentAgent(BaseAgent):
        """Agent that produces no content for edge case testing."""

        def __init__(self, name: str = "empty_agent") -> None:
            super().__init__(name)

        async def run(self, context: AgentContext) -> AgentContext:
            """Run without producing any agent output."""
            # Don't add any agent output - test empty content handling
            return context

    # Create agent and context
    agent = EmptyContentAgent("test_empty_agent")
    context = AgentContextFactory.basic(
        query="Test query for empty content",
        user_config={"user_id": "test_user", "session_id": "test_session"},
        metadata={"workflow_metadata": {}},
    )

    # Mock the event emission to capture what gets emitted
    with patch(
        "cognivault.agents.base_agent.emit_agent_execution_completed"
    ) as mock_emit:
        mock_emit.return_value = AsyncMock()

        with patch(
            "cognivault.agents.base_agent.get_workflow_id",
            return_value="test-workflow-empty",
        ):
            with patch(
                "cognivault.agents.base_agent.get_correlation_id",
                return_value="test-correlation-empty",
            ):
                # Execute the agent
                result_context = await agent.run_with_retry(context)

                # Verify event emission was called
                mock_emit.assert_called_once()

                # Extract the call arguments
                call_args, call_kwargs = mock_emit.call_args
                output_context = call_kwargs["output_context"]

                # Verify graceful handling of empty content
                assert "agent_output" in output_context
                assert output_context["agent_output"] == "", (
                    "Empty content should be empty string"
                )
                assert output_context["output_length"] == 0, (
                    "Empty content length should be 0"
                )

                # Token usage should still be properly handled (defaults to 0)
                assert output_context["input_tokens"] == 0
                assert output_context["output_tokens"] == 0
                assert output_context["total_tokens"] == 0
