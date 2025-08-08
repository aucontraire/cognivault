"""
Runtime validation decorators for API boundary implementation.

This module provides decorators for ensuring API initialization, rate limiting,
and circuit breaker patterns for API resilience.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Callable, Dict, Any, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator
from cognivault.observability import get_logger

logger = get_logger(__name__)


def ensure_initialized(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure API is initialized before method execution.

    Prevents calls to uninitialized APIs and provides clear error messages.
    """

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "_initialized", False):
            raise RuntimeError(
                f"{self.__class__.__name__} must be initialized before calling {func.__name__}. "
                f"Call await {self.__class__.__name__}.initialize() first."
            )
        return await func(self, *args, **kwargs)

    return wrapper


class TokenBucket(BaseModel):
    """
    Token bucket for rate limiting implementation.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    capacity: int = Field(
        ...,
        description="Maximum number of tokens the bucket can hold",
        gt=0,
        le=10000,
        json_schema_extra={"example": 100},
    )
    refill_rate: float = Field(
        ...,
        description="Tokens refilled per second",
        gt=0.0,
        le=1000.0,
        json_schema_extra={"example": 10.0},
    )
    tokens: float = Field(
        default=0.0,
        description="Current number of tokens available",
        ge=0.0,
        json_schema_extra={"example": 100.0},
    )
    last_refill: float = Field(
        default=0.0,
        description="Timestamp of last refill operation",
        ge=0.0,
        json_schema_extra={"example": 1640995200.0},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )

    @model_validator(mode="after")
    def initialize_tokens_and_time(self) -> "TokenBucket":
        """Initialize tokens and last_refill time if not set."""
        if self.tokens == 0.0:
            self.tokens = float(self.capacity)
        if self.last_refill == 0.0:
            self.last_refill = time.time()
        return self

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Returns:
            True if tokens were consumed, False if not enough tokens available
        """
        now = time.time()
        elapsed = now - self.last_refill

        # Refill tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


# Global rate limiting state
_rate_limiters: Dict[str, TokenBucket] = {}


def rate_limited(
    calls_per_second: int = 10, burst_size: Optional[int] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Rate limiting decorator for API methods using token bucket algorithm.

    Args:
        calls_per_second: Maximum calls allowed per second
        burst_size: Maximum burst size (defaults to calls_per_second * 2)
    """
    burst_size = burst_size or calls_per_second * 2

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Create unique key for this API method
            limiter_key = f"{self.__class__.__name__}.{func.__name__}"

            # Get or create token bucket for this method
            if limiter_key not in _rate_limiters:
                _rate_limiters[limiter_key] = TokenBucket(
                    capacity=burst_size, refill_rate=calls_per_second
                )

            bucket = _rate_limiters[limiter_key]

            # Try to consume a token
            if not bucket.consume():
                logger.warning(
                    f"Rate limit exceeded for {limiter_key}: "
                    f"{calls_per_second} calls/sec, burst: {burst_size}"
                )
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {calls_per_second} calls per second allowed"
                )

            logger.debug(f"Rate limit check passed for {limiter_key}")
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class APICircuitBreaker(BaseModel):
    """
    Circuit breaker implementation for API resilience.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    failure_threshold: int = Field(
        ...,
        description="Number of failures before opening the circuit",
        gt=0,
        le=100,
        json_schema_extra={"example": 5},
    )
    recovery_timeout: int = Field(
        ...,
        description="Seconds to wait before attempting recovery",
        gt=0,
        le=3600,
        json_schema_extra={"example": 60},
    )
    failure_count: int = Field(
        default=0,
        description="Current count of consecutive failures",
        ge=0,
        json_schema_extra={"example": 0},
    )
    last_failure_time: Optional[float] = Field(
        default=None,
        description="Timestamp of the last failure",
        ge=0.0,
        json_schema_extra={"example": 1640995200.0},
    )
    state: CircuitState = Field(
        default=CircuitState.CLOSED,
        description="Current state of the circuit breaker",
        json_schema_extra={"example": "closed"},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.recovery_timeout
            ):
                # Move to half-open to test recovery
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker moving to HALF_OPEN state for recovery test"
                )
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # Allow one call to test recovery
            return True
        return False

    def record_success(self) -> None:
        """Record a successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            # Recovery successful, close the circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker recovered: moving to CLOSED state")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Recovery test failed, go back to open
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker recovery test failed: moving to OPEN state")
        elif (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.failure_threshold
        ):
            # Too many failures, open the circuit
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker OPENED: {self.failure_count} failures "
                f"exceeded threshold of {self.failure_threshold}"
            )


# Global circuit breaker state
_circuit_breakers: Dict[str, APICircuitBreaker] = {}


def circuit_breaker(
    failure_threshold: int = 5, recovery_timeout: int = 60
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Circuit breaker pattern decorator for API resilience.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Create unique key for this API method
            breaker_key = f"{self.__class__.__name__}.{func.__name__}"

            # Get or create circuit breaker for this method
            if breaker_key not in _circuit_breakers:
                _circuit_breakers[breaker_key] = APICircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )

            breaker = _circuit_breakers[breaker_key]

            # Check if circuit allows execution
            if not breaker.can_execute():
                logger.warning(f"Circuit breaker OPEN for {breaker_key}: failing fast")
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for {breaker_key}. "
                    f"Service may be experiencing issues."
                )

            try:
                # Execute the function
                result = await func(self, *args, **kwargs)

                # Record success
                breaker.record_success()
                logger.debug(f"Circuit breaker success recorded for {breaker_key}")

                return result

            except Exception as e:
                # Record failure
                breaker.record_failure()
                logger.warning(
                    f"Circuit breaker failure recorded for {breaker_key}: {e}"
                )

                # Re-raise the original exception
                raise

        return wrapper

    return decorator


# Custom exceptions
class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# Utility functions for monitoring and management
def get_rate_limiter_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all rate limiters."""
    stats = {}
    for key, bucket in _rate_limiters.items():
        stats[key] = {
            "capacity": bucket.capacity,
            "refill_rate": bucket.refill_rate,
            "current_tokens": bucket.tokens,
            "last_refill": bucket.last_refill,
        }
    return stats


def get_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    stats = {}
    for key, breaker in _circuit_breakers.items():
        stats[key] = {
            "state": breaker.state.value,
            "failure_threshold": breaker.failure_threshold,
            "recovery_timeout": breaker.recovery_timeout,
            "failure_count": breaker.failure_count,
            "last_failure_time": breaker.last_failure_time,
        }
    return stats


def reset_rate_limiters() -> None:
    """Reset all rate limiters (useful for testing)."""
    global _rate_limiters
    _rate_limiters.clear()


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (useful for testing)."""
    global _circuit_breakers
    _circuit_breakers.clear()
