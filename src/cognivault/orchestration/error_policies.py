"""
Error policies for LangGraph nodes and CogniVault execution.

This module provides centralized error handling policies, retry configurations,
and circuit breaker decorators for robust LangGraph node execution.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorPolicyType(Enum):
    """Types of error handling policies."""

    FAIL_FAST = "fail_fast"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class FallbackStrategy(Enum):
    """Fallback strategies for failed nodes."""

    SKIP_NODE = "skip_node"
    USE_CACHED_RESULT = "use_cached_result"
    SUBSTITUTE_AGENT = "substitute_agent"
    PARTIAL_RESULT = "partial_result"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_types: List[type] = field(default_factory=lambda: [Exception])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class ErrorPolicy:
    """Comprehensive error handling policy."""

    policy_type: ErrorPolicyType
    retry_config: Optional[RetryConfig] = None
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    fallback_strategy: Optional[FallbackStrategy] = None
    timeout_seconds: Optional[float] = None
    critical_errors: List[type] = field(default_factory=list)
    recoverable_errors: List[type] = field(default_factory=lambda: [Exception])


class LangGraphExecutionError(Exception):
    """Wrapper for LangGraph node execution errors."""

    def __init__(
        self,
        message: str,
        node_name: str,
        original_error: Optional[Exception] = None,
        retry_count: int = 0,
        execution_context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.node_name = node_name
        self.original_error = original_error
        self.retry_count = retry_count
        self.execution_context = execution_context or {}
        self.timestamp = time.time()


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker implementation for node resilience."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time >= self.config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 1  # Count this transition as the first call
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            can_proceed = self.half_open_calls < self.config.half_open_max_calls
            if can_proceed:
                self.half_open_calls += 1
            return can_proceed

        return False

    def record_success(self):
        """Record a successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success

    def record_failure(self):
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.half_open_calls = 0


class ErrorPolicyManager:
    """Centralized manager for error policies."""

    def __init__(self) -> None:
        self.policies: Dict[str, ErrorPolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._setup_default_policies()

    def _setup_default_policies(self):
        """Set up default error policies for common scenarios."""

        # Refiner agent policy - fail fast for critical path
        self.policies["refiner"] = ErrorPolicy(
            policy_type=ErrorPolicyType.RETRY_WITH_BACKOFF,
            retry_config=RetryConfig(
                max_attempts=2,
                base_delay_seconds=0.5,
                max_delay_seconds=5.0,
            ),
            timeout_seconds=30.0,
            critical_errors=[MemoryError, KeyboardInterrupt],
        )

        # Critic agent policy - graceful degradation
        self.policies["critic"] = ErrorPolicy(
            policy_type=ErrorPolicyType.GRACEFUL_DEGRADATION,
            retry_config=RetryConfig(max_attempts=1),
            fallback_strategy=FallbackStrategy.PARTIAL_RESULT,
            timeout_seconds=20.0,
        )

        # Historian agent policy - circuit breaker for external calls
        historian_cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
        )
        self.policies["historian"] = ErrorPolicy(
            policy_type=ErrorPolicyType.CIRCUIT_BREAKER,
            circuit_breaker_config=historian_cb_config,
            fallback_strategy=FallbackStrategy.USE_CACHED_RESULT,
            timeout_seconds=25.0,
        )
        # Initialize circuit breaker for historian
        self.circuit_breakers["historian"] = CircuitBreaker(historian_cb_config)

        # Synthesis agent policy - critical but with retries
        self.policies["synthesis"] = ErrorPolicy(
            policy_type=ErrorPolicyType.RETRY_WITH_BACKOFF,
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay_seconds=1.0,
                max_delay_seconds=10.0,
            ),
            timeout_seconds=45.0,
        )

    def get_policy(self, node_name: str) -> ErrorPolicy:
        """Get error policy for a node."""
        return self.policies.get(node_name, self._get_default_policy())

    def set_policy(self, node_name: str, policy: ErrorPolicy):
        """Set error policy for a node."""
        self.policies[node_name] = policy

        # Initialize circuit breaker if needed
        if policy.circuit_breaker_config:
            self.circuit_breakers[node_name] = CircuitBreaker(
                policy.circuit_breaker_config
            )

    def get_circuit_breaker(self, node_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a node."""
        return self.circuit_breakers.get(node_name)

    def _get_default_policy(self) -> ErrorPolicy:
        """Get default error policy."""
        return ErrorPolicy(
            policy_type=ErrorPolicyType.RETRY_WITH_BACKOFF,
            retry_config=RetryConfig(max_attempts=3),
            timeout_seconds=30.0,
        )


# Global error policy manager instance
_error_policy_manager = ErrorPolicyManager()


def get_error_policy_manager() -> ErrorPolicyManager:
    """Get the global error policy manager."""
    return _error_policy_manager


def retry_with_policy(node_name: str):
    """Decorator for retry behavior based on error policy."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            policy = get_error_policy_manager().get_policy(node_name)
            retry_config = policy.retry_config or RetryConfig()

            last_exception = None

            for attempt in range(retry_config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    return result

                except Exception as e:
                    last_exception = e

                    # Check if this error type should be retried
                    if not any(
                        isinstance(e, error_type)
                        for error_type in retry_config.retry_on_types
                    ):
                        logger.warning(f"Non-retryable error in {node_name}: {e}")
                        raise LangGraphExecutionError(
                            f"Non-retryable error in {node_name}: {e}",
                            node_name=node_name,
                            original_error=e,
                            retry_count=attempt,
                        )

                    # Don't retry on last attempt
                    if attempt == retry_config.max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay_seconds
                        * (retry_config.exponential_base**attempt),
                        retry_config.max_delay_seconds,
                    )

                    # Add jitter if enabled
                    if retry_config.jitter:
                        import random

                        delay *= (
                            0.5 + random.random() * 0.5
                        )  # 50-100% of calculated delay

                    logger.info(
                        f"Retrying {node_name} after {delay:.2f}s "
                        f"(attempt {attempt + 1}/{retry_config.max_attempts}): {e}"
                    )

                    await asyncio.sleep(delay)

            # All retries exhausted
            raise LangGraphExecutionError(
                f"All retries exhausted for {node_name}: {last_exception}",
                node_name=node_name,
                original_error=last_exception,
                retry_count=retry_config.max_attempts,
            )

        return wrapper

    return decorator


def circuit_breaker_policy(node_name: str):
    """Decorator for circuit breaker behavior."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            policy_manager = get_error_policy_manager()
            circuit_breaker = policy_manager.get_circuit_breaker(node_name)

            if not circuit_breaker:
                # No circuit breaker configured, execute normally
                return await func(*args, **kwargs)

            if not circuit_breaker.can_execute():
                raise LangGraphExecutionError(
                    f"Circuit breaker open for {node_name} "
                    f"(state: {circuit_breaker.state.value}, "
                    f"failures: {circuit_breaker.failure_count})",
                    node_name=node_name,
                )

            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result

            except Exception as e:
                circuit_breaker.record_failure()
                raise LangGraphExecutionError(
                    f"Execution failed in {node_name} (circuit breaker): {e}",
                    node_name=node_name,
                    original_error=e,
                )

        return wrapper

    return decorator


def timeout_policy(node_name: str):
    """Decorator for timeout handling."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            policy = get_error_policy_manager().get_policy(node_name)
            timeout_seconds = policy.timeout_seconds

            if not timeout_seconds:
                return await func(*args, **kwargs)

            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_seconds
                )
                return result

            except asyncio.TimeoutError:
                raise LangGraphExecutionError(
                    f"Timeout after {timeout_seconds}s in {node_name}",
                    node_name=node_name,
                )

        return wrapper

    return decorator


def comprehensive_error_policy(node_name: str):
    """Comprehensive decorator combining all error policies."""

    def decorator(func: Callable):
        # Apply decorators in order: timeout -> circuit_breaker -> retry
        wrapped_func = timeout_policy(node_name)(func)
        wrapped_func = circuit_breaker_policy(node_name)(wrapped_func)
        wrapped_func = retry_with_policy(node_name)(wrapped_func)
        return wrapped_func

    return decorator


def handle_node_fallback(
    node_name: str,
    error: Exception,
    fallback_strategy: FallbackStrategy,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Handle fallback behavior for failed nodes."""

    logger.warning(
        f"Executing fallback for {node_name} with strategy {fallback_strategy.value}: {error}"
    )

    if fallback_strategy == FallbackStrategy.SKIP_NODE:
        logger.info(f"Skipping {node_name} due to error")
        return None

    elif fallback_strategy == FallbackStrategy.USE_CACHED_RESULT:
        # TODO: Implement cache lookup
        logger.info(f"Using cached result for {node_name}")
        return {"fallback": "cached_result", "node": node_name}

    elif fallback_strategy == FallbackStrategy.PARTIAL_RESULT:
        logger.info(f"Returning partial result for {node_name}")
        return {"fallback": "partial_result", "node": node_name, "error": str(error)}

    elif fallback_strategy == FallbackStrategy.SUBSTITUTE_AGENT:
        # TODO: Implement agent substitution
        logger.info(f"Substituting agent for {node_name}")
        return {"fallback": "substitute_agent", "node": node_name}

    else:
        logger.warning(f"Unknown fallback strategy: {fallback_strategy}")
        return None


def get_error_statistics() -> Dict[str, Any]:
    """Get error handling statistics."""
    policy_manager = get_error_policy_manager()

    stats: Dict[str, Any] = {
        "policies_configured": len(policy_manager.policies),
        "circuit_breakers_active": len(policy_manager.circuit_breakers),
        "circuit_breaker_states": {},
    }

    for node_name, circuit_breaker in policy_manager.circuit_breakers.items():
        stats["circuit_breaker_states"][node_name] = {
            "state": circuit_breaker.state.value,
            "failure_count": circuit_breaker.failure_count,
            "success_count": circuit_breaker.success_count,
            "last_failure_time": circuit_breaker.last_failure_time,
        }

    return stats
