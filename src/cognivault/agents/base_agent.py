from abc import ABC, abstractmethod
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from cognivault.context import AgentContext
from cognivault.exceptions import (
    AgentExecutionError,
    AgentTimeoutError,
    LLMError,
    RetryPolicy,
)


class RetryConfig:
    """Configuration for agent retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter


class CircuitBreakerState:
    """Simple circuit breaker state for agent execution."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False

    def record_success(self):
        """Record a successful execution."""
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

    def record_failure(self):
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self.is_open = True

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if not self.is_open:
            return True

        if self.last_failure_time:
            time_since_failure = (
                datetime.now(timezone.utc) - self.last_failure_time
            ).total_seconds()
            if time_since_failure >= self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
                return True

        return False


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Cognivault system.

    Enhanced with LangGraph-compatible features including agent-local retry policies,
    circuit breaker patterns, trace metadata, and error handling designed for
    future DAG-based orchestration.

    Parameters
    ----------
    name : str
        The name of the agent.
    retry_config : RetryConfig, optional
        Retry configuration for this agent. If None, uses default settings.
    timeout_seconds : float, optional
        Agent execution timeout in seconds. Default is 30.0.
    enable_circuit_breaker : bool, optional
        Whether to enable circuit breaker pattern. Default is True.
    """

    def __init__(
        self,
        name: str,
        retry_config: Optional[RetryConfig] = None,
        timeout_seconds: float = 30.0,
        enable_circuit_breaker: bool = True,
    ):
        self.name: str = name
        self.retry_config = retry_config or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Circuit breaker for this agent
        self.circuit_breaker: Optional[CircuitBreakerState] = None
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreakerState()

        # Agent execution statistics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

    def generate_step_id(self) -> str:
        """Generate a unique step ID for this execution."""
        return f"{self.name.lower()}_{uuid.uuid4().hex[:8]}"

    async def run_with_retry(
        self, context: AgentContext, step_id: Optional[str] = None
    ) -> AgentContext:
        """
        Execute the agent with retry logic, timeout, and circuit breaker pattern.

        This method implements LangGraph-compatible node behavior with agent-local
        error handling, making each agent suitable for future DAG orchestration.

        Parameters
        ----------
        context : AgentContext
            The context object containing state and input information for the agent.
        step_id : str, optional
            Step identifier for trace tracking. If None, generates a new one.

        Returns
        -------
        AgentContext
            The updated context after agent processing.

        Raises
        ------
        AgentExecutionError
            When agent execution fails after all retries
        AgentTimeoutError
            When agent execution times out
        """
        step_id = step_id or self.generate_step_id()
        start_time = datetime.now(timezone.utc)

        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            failure_time = self.circuit_breaker.last_failure_time
            time_remaining = self.circuit_breaker.recovery_timeout
            if failure_time:
                elapsed = (datetime.now(timezone.utc) - failure_time).total_seconds()
                time_remaining = max(0, self.circuit_breaker.recovery_timeout - elapsed)

            raise AgentExecutionError(
                message=f"Circuit breaker open for agent '{self.name}'",
                agent_name=self.name,
                error_code="circuit_breaker_open",
                context={
                    "failure_count": self.circuit_breaker.failure_count,
                    "time_remaining_seconds": time_remaining,
                },
                step_id=step_id,
            )

        self.execution_count += 1
        retries = 0
        last_exception: Optional[Exception] = None

        while retries <= self.retry_config.max_retries:
            try:
                self.logger.info(
                    f"[{self.name}] Starting execution (step: {step_id}, attempt: {retries + 1})"
                )

                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_with_context(context, step_id),
                    timeout=self.timeout_seconds,
                )

                # Success - record metrics and return
                execution_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                self.success_count += 1

                if self.circuit_breaker:
                    self.circuit_breaker.record_success()

                self.logger.info(
                    f"[{self.name}] Execution successful "
                    f"(step: {step_id}, time: {execution_time:.2f}s, attempt: {retries + 1})"
                )

                # Add execution metadata to context
                context.log_trace(
                    self.name,
                    input_data={"step_id": step_id, "attempt": retries + 1},
                    output_data={
                        "success": True,
                        "execution_time_seconds": execution_time,
                        "attempts_used": retries + 1,
                    },
                )

                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                self.logger.warning(
                    f"[{self.name}] Timeout after {self.timeout_seconds}s (step: {step_id})"
                )

                # Timeout - decide if retryable
                if retries < self.retry_config.max_retries:
                    await self._handle_retry_delay(retries)
                    retries += 1
                    continue
                else:
                    # Final timeout failure
                    self.failure_count += 1
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()

                    raise AgentTimeoutError(
                        agent_name=self.name,
                        timeout_seconds=self.timeout_seconds,
                        step_id=step_id,
                        context={
                            "attempts_made": retries + 1,
                            "max_retries": self.retry_config.max_retries,
                        },
                        cause=e,
                    )

            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"[{self.name}] Execution failed: {e} (step: {step_id})"
                )

                # Check if this exception is retryable
                should_retry = self._should_retry_exception(e)

                if should_retry and retries < self.retry_config.max_retries:
                    await self._handle_retry_delay(retries)
                    retries += 1
                    continue
                else:
                    # Final failure
                    self.failure_count += 1
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()

                    # Convert to appropriate agent exception
                    if isinstance(e, (AgentExecutionError, LLMError)):
                        # Already a proper exception, just re-raise
                        raise
                    else:
                        # Wrap in AgentExecutionError
                        raise AgentExecutionError(
                            message=f"Agent execution failed: {str(e)}",
                            agent_name=self.name,
                            error_code="agent_execution_failed",
                            step_id=step_id,
                            context={
                                "attempts_made": retries + 1,
                                "max_retries": self.retry_config.max_retries,
                                "original_exception": str(e),
                            },
                            cause=e,
                        )

        # Should never reach here, but just in case
        raise AgentExecutionError(
            message=f"Agent execution failed after {retries} attempts",
            agent_name=self.name,
            step_id=step_id,
            cause=last_exception,
        )

    async def _execute_with_context(
        self, context: AgentContext, step_id: str
    ) -> AgentContext:
        """
        Internal method that wraps the actual agent execution with context metadata.

        This method adds step_id and agent_id metadata to the context before
        calling the abstract run method, and integrates with the context's
        execution state tracking for LangGraph compatibility.
        """
        # Start execution tracking in context
        context.start_agent_execution(self.name, step_id)

        # Add step metadata to execution_state for trace tracking
        step_metadata_key = f"{self.name}_step_metadata"
        context.execution_state[step_metadata_key] = {
            "step_id": step_id,
            "agent_id": self.name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "execution_count": self.execution_count,
        }

        try:
            # Call the actual agent implementation
            result = await self.run(context)

            # Mark execution as successful
            context.complete_agent_execution(self.name, success=True)

            # Update metadata with completion info
            step_metadata_key = f"{self.name}_step_metadata"
            if step_metadata_key in context.execution_state:
                context.execution_state[step_metadata_key]["end_time"] = datetime.now(
                    timezone.utc
                ).isoformat()
                context.execution_state[step_metadata_key]["completed"] = True

            return result

        except Exception as e:
            # Mark execution as failed
            context.complete_agent_execution(self.name, success=False)

            # Update metadata with failure info
            step_metadata_key = f"{self.name}_step_metadata"
            if step_metadata_key in context.execution_state:
                context.execution_state[step_metadata_key]["end_time"] = datetime.now(
                    timezone.utc
                ).isoformat()
                context.execution_state[step_metadata_key]["completed"] = False
                context.execution_state[step_metadata_key]["error"] = str(e)

            # Re-raise the exception to be handled by retry logic
            raise

    def _should_retry_exception(self, exception: Exception) -> bool:
        """
        Determine if an exception should be retried based on the agent's retry policy.

        Parameters
        ----------
        exception : Exception
            The exception that occurred during execution

        Returns
        -------
        bool
            True if the exception should be retried, False otherwise
        """
        # Check if it's a CogniVault exception with retry policy
        if hasattr(exception, "retry_policy"):
            retry_policy = exception.retry_policy
            return retry_policy in [
                RetryPolicy.IMMEDIATE,
                RetryPolicy.BACKOFF,
                RetryPolicy.CIRCUIT_BREAKER,
            ]

        # Default behavior for non-CogniVault exceptions
        # Retry on common transient errors
        if isinstance(exception, (asyncio.TimeoutError, ConnectionError)):
            return True

        # Don't retry on validation, configuration, or authentication errors
        if isinstance(exception, (ValueError, TypeError, AttributeError)):
            return False

        # Default to retry for unknown exceptions (conservative approach)
        return True

    async def _handle_retry_delay(self, retry_attempt: int):
        """
        Handle delay between retry attempts with exponential backoff and jitter.

        Parameters
        ----------
        retry_attempt : int
            The current retry attempt number (0-based)
        """
        if self.retry_config.base_delay <= 0:
            return

        delay = self.retry_config.base_delay

        if self.retry_config.exponential_backoff:
            delay = min(
                self.retry_config.base_delay * (2**retry_attempt),
                self.retry_config.max_delay,
            )

        if self.retry_config.jitter:
            import random

            # Add ±25% jitter to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative

        self.logger.debug(
            f"[{self.name}] Retrying in {delay:.2f}s (attempt {retry_attempt + 1})"
        )
        await asyncio.sleep(delay)

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for this agent.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing execution statistics
        """
        success_rate = (
            (self.success_count / self.execution_count)
            if self.execution_count > 0
            else 0.0
        )

        stats = {
            "agent_name": self.name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "exponential_backoff": self.retry_config.exponential_backoff,
            },
        }

        if self.circuit_breaker:
            stats["circuit_breaker"] = {
                "is_open": self.circuit_breaker.is_open,
                "failure_count": self.circuit_breaker.failure_count,
                "failure_threshold": self.circuit_breaker.failure_threshold,
            }

        return stats

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentContext:
        """
        Execute the agent asynchronously using the provided context.

        This method should be implemented by concrete agent classes to define
        their specific behavior. The base class handles retry logic, timeouts,
        and error handling around this method.

        Parameters
        ----------
        context : AgentContext
            The context object containing state and input information for the agent.

        Returns
        -------
        AgentContext
            The updated context after agent processing.
        """
        pass  # pragma: no cover
