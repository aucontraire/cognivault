from abc import ABC, abstractmethod
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Coroutine
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from cognivault.context import AgentContext
from cognivault.exceptions import (
    AgentExecutionError,
    AgentTimeoutError,
    LLMError,
    RetryPolicy,
)
from cognivault.correlation import (
    get_correlation_id,
    get_workflow_id,
)

# Lazy event emission imports to avoid circular import
# Events will be imported at runtime when needed
EVENTS_AVAILABLE = True


async def _emit_agent_execution_started(
    workflow_id: str,
    agent_name: str,
    input_context: Dict[str, Any],
    agent_metadata: Optional[Any] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    event_category: Optional[Any] = None,
) -> None:
    """Lazily import and emit agent execution started event."""
    try:
        from cognivault.events import emit_agent_execution_started
        from cognivault.events.types import EventCategory

        await emit_agent_execution_started(
            workflow_id=workflow_id,
            agent_name=agent_name,
            input_context=input_context,
            agent_metadata=agent_metadata,
            correlation_id=correlation_id,
            metadata=metadata,
            event_category=event_category or EventCategory.EXECUTION,
        )
    except ImportError as e:
        # Events not available, skip silently
        pass
    except Exception as e:
        # Log error but don't fail execution
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to emit agent execution started event: {e}")


async def _emit_agent_execution_completed(
    workflow_id: str,
    agent_name: str,
    success: bool,
    output_context: Dict[str, Any],
    agent_metadata: Optional[Any] = None,
    execution_time_ms: Optional[float] = None,
    error_message: Optional[str] = None,
    error_type: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    event_category: Optional[Any] = None,
) -> None:
    """Lazily import and emit agent execution completed event."""
    try:
        from cognivault.events import emit_agent_execution_completed
        from cognivault.events.types import EventCategory

        await emit_agent_execution_completed(
            workflow_id=workflow_id,
            agent_name=agent_name,
            success=success,
            output_context=output_context,
            agent_metadata=agent_metadata,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
            error_type=error_type,
            correlation_id=correlation_id,
            metadata=metadata,
            event_category=event_category or EventCategory.EXECUTION,
        )
    except ImportError as e:
        # Events not available, skip silently
        pass
    except Exception as e:
        # Log error but don't fail execution
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to emit agent execution completed event: {e}")


# For backward compatibility, make the lazy functions available
emit_agent_execution_started = _emit_agent_execution_started
emit_agent_execution_completed = _emit_agent_execution_completed

# Make event emission functions available at module level for testing
__all__ = [
    "BaseAgent",
    "AgentRetryConfig",
    "CircuitBreakerState",
    "NodeType",
    "NodeInputSchema",
    "NodeOutputSchema",
    "LangGraphNodeDefinition",
    "emit_agent_execution_started",
    "emit_agent_execution_completed",
]


class AgentRetryConfig:
    """Configuration for agent retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter


class CircuitBreakerState:
    """Simple circuit breaker state for agent execution."""

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: float = 300.0
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False

    def record_success(self) -> None:
        """Record a successful execution."""
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

    def record_failure(self) -> None:
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


class NodeType(Enum):
    """Types of nodes in a LangGraph DAG."""

    PROCESSOR = "processor"  # Standard processing node
    DECISION = "decision"  # Decision/routing node
    TERMINATOR = "terminator"  # End/output node
    AGGREGATOR = "aggregator"  # Combines multiple inputs
    VALIDATOR = "validator"  # Quality assurance checkpoint


class NodeInputSchema(BaseModel):
    """
    Schema definition for node inputs.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    name: str = Field(
        ...,
        description="Name of the input parameter",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "context"},
    )
    description: str = Field(
        ...,
        description="Human-readable description of the input",
        min_length=1,
        max_length=500,
        json_schema_extra={"example": "Agent context containing query and state"},
    )
    required: bool = Field(
        default=True,
        description="Whether this input is required for node execution",
        json_schema_extra={"example": True},
    )
    type_hint: str = Field(
        default="Any",
        description="Type hint for the input parameter",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "AgentContext"},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class NodeOutputSchema(BaseModel):
    """
    Schema definition for node outputs.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    name: str = Field(
        ...,
        description="Name of the output parameter",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "context"},
    )
    description: str = Field(
        ...,
        description="Human-readable description of the output",
        min_length=1,
        max_length=500,
        json_schema_extra={"example": "Updated context after agent processing"},
    )
    type_hint: str = Field(
        default="Any",
        description="Type hint for the output parameter",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "AgentContext"},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class LangGraphNodeDefinition(BaseModel):
    """
    Complete LangGraph node definition for an agent.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    node_id: str = Field(
        ...,
        description="Unique identifier for the node",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "refiner"},
    )
    node_type: NodeType = Field(
        ...,
        description="Type of node in the LangGraph DAG",
        json_schema_extra={"example": "processor"},
    )
    description: str = Field(
        ...,
        description="Human-readable description of the node",
        min_length=1,
        max_length=500,
        json_schema_extra={"example": "Refiner agent for query processing"},
    )
    inputs: List[NodeInputSchema] = Field(
        ...,
        description="List of input schemas for this node",
        json_schema_extra={
            "example": [{"name": "context", "description": "Input context"}]
        },
    )
    outputs: List[NodeOutputSchema] = Field(
        ...,
        description="List of output schemas for this node",
        json_schema_extra={
            "example": [{"name": "context", "description": "Output context"}]
        },
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of node IDs this node depends on",
        json_schema_extra={"example": ["refiner", "critic"]},
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorization",
        json_schema_extra={"example": ["agent", "processor"]},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Maintains backward compatibility with existing code while leveraging
        Pydantic's enhanced serialization capabilities.
        """
        # Use Pydantic's model_dump for better serialization
        data = self.model_dump()

        # Convert node_type enum to string value for compatibility
        data["node_type"] = self.node_type.value

        # Convert input/output schemas to the expected format
        data["inputs"] = [
            {
                "name": inp.name,
                "description": inp.description,
                "required": inp.required,
                "type": inp.type_hint,
            }
            for inp in self.inputs
        ]
        data["outputs"] = [
            {
                "name": out.name,
                "description": out.description,
                "type": out.type_hint,
            }
            for out in self.outputs
        ]

        return data


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
    retry_config : AgentRetryConfig, optional
        Retry configuration for this agent. If None, uses default settings.
    timeout_seconds : float, optional
        Agent execution timeout in seconds. Default is 30.0.
    enable_circuit_breaker : bool, optional
        Whether to enable circuit breaker pattern. Default is True.
    """

    def __init__(
        self,
        name: str,
        retry_config: Optional[AgentRetryConfig] = None,
        timeout_seconds: float = 30.0,
        enable_circuit_breaker: bool = True,
    ) -> None:
        self.name: str = name
        self.retry_config = retry_config or AgentRetryConfig()
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

        # LangGraph node metadata
        self._node_definition: Optional[LangGraphNodeDefinition] = None

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
                time_remaining = max(
                    0.0, self.circuit_breaker.recovery_timeout - elapsed
                )

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

        # Emit agent execution started event if available
        if True:  # Events always available with lazy loading
            try:
                from cognivault.agents.registry import get_agent_registry

                registry = get_agent_registry()
                try:
                    agent_metadata = registry.get_metadata(self.name)
                except ValueError:
                    # Agent not registered in registry, use None metadata
                    agent_metadata = None

                await emit_agent_execution_started(
                    workflow_id=get_workflow_id() or step_id,
                    agent_name=self.name,
                    input_context={
                        "step_id": step_id,
                        "execution_count": self.execution_count,
                        "input_tokens": getattr(context, "token_count", 0),
                        "context_size": len(str(context)),
                    },
                    agent_metadata=agent_metadata,
                    correlation_id=get_correlation_id(),
                    metadata={
                        "retry_config": {
                            "max_retries": self.retry_config.max_retries,
                            "timeout_seconds": self.timeout_seconds,
                        },
                        "circuit_breaker_enabled": self.circuit_breaker is not None,
                    },
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to emit agent execution started event: {e}"
                )

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

                # Emit agent execution completed event if available
                if True:  # Events always available with lazy loading
                    try:
                        from cognivault.agents.registry import get_agent_registry

                        registry = get_agent_registry()
                        try:
                            agent_metadata = registry.get_metadata(self.name)
                        except ValueError:
                            # Agent not registered in registry, use None metadata
                            agent_metadata = None

                        # Extract actual agent output content from the result context
                        agent_output: str = result.agent_outputs.get(self.name, "")

                        # Get token usage information for this agent if available
                        token_usage = result.get_agent_token_usage(self.name)

                        await emit_agent_execution_completed(
                            workflow_id=get_workflow_id() or step_id,
                            agent_name=self.name,
                            success=True,
                            output_context={
                                "step_id": step_id,
                                "execution_time_seconds": execution_time,
                                "attempts_used": retries + 1,
                                "agent_output": (
                                    agent_output[:1000] if agent_output else ""
                                ),  # Include actual content, truncated for events
                                "output_length": (
                                    len(agent_output) if agent_output else 0
                                ),
                                "input_tokens": token_usage["input_tokens"],
                                "output_tokens": token_usage["output_tokens"],
                                "total_tokens": token_usage["total_tokens"],
                                "context_size": len(str(result)),
                            },
                            agent_metadata=agent_metadata,
                            correlation_id=get_correlation_id(),
                            execution_time_ms=execution_time * 1000,
                            metadata={
                                "success_count": self.success_count,
                                "total_executions": self.execution_count,
                                "circuit_breaker_state": (
                                    "closed"
                                    if not self.circuit_breaker
                                    or not self.circuit_breaker.is_open
                                    else "open"
                                ),
                            },
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to emit agent execution completed event: {e}"
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

                    # Emit agent execution completed event for timeout failure if available
                    if True:  # Events always available with lazy loading
                        try:
                            from cognivault.agents.registry import get_agent_registry

                            registry = get_agent_registry()
                            try:
                                agent_metadata = registry.get_metadata(self.name)
                            except ValueError:
                                # Agent not registered in registry, use None metadata
                                agent_metadata = None

                            await emit_agent_execution_completed(
                                workflow_id=get_workflow_id() or step_id,
                                agent_name=self.name,
                                success=False,
                                output_context={
                                    "step_id": step_id,
                                    "attempts_made": retries + 1,
                                    "max_retries": self.retry_config.max_retries,
                                    "timeout_seconds": self.timeout_seconds,
                                    "error_type": "AgentTimeoutError",
                                    "error_message": f"Agent timed out after {self.timeout_seconds}s",
                                },
                                agent_metadata=agent_metadata,
                                correlation_id=get_correlation_id(),
                                error_message=f"Agent timed out after {self.timeout_seconds}s",
                                error_type="AgentTimeoutError",
                                metadata={
                                    "failure_count": self.failure_count,
                                    "total_executions": self.execution_count,
                                    "circuit_breaker_state": (
                                        "open"
                                        if self.circuit_breaker
                                        and self.circuit_breaker.is_open
                                        else "closed"
                                    ),
                                },
                            )
                        except Exception as emit_e:
                            self.logger.warning(
                                f"Failed to emit agent timeout event: {emit_e}"
                            )

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

                    # Emit agent execution completed event for failure if available
                    if True:  # Events always available with lazy loading
                        try:
                            from cognivault.agents.registry import get_agent_registry

                            registry = get_agent_registry()
                            try:
                                agent_metadata = registry.get_metadata(self.name)
                            except ValueError:
                                # Agent not registered in registry, use None metadata
                                agent_metadata = None

                            await emit_agent_execution_completed(
                                workflow_id=get_workflow_id() or step_id,
                                agent_name=self.name,
                                success=False,
                                output_context={
                                    "step_id": step_id,
                                    "attempts_made": retries + 1,
                                    "max_retries": self.retry_config.max_retries,
                                    "error_type": type(e).__name__,
                                    "error_message": str(e),
                                },
                                agent_metadata=agent_metadata,
                                correlation_id=get_correlation_id(),
                                error_message=str(e),
                                error_type=type(e).__name__,
                                metadata={
                                    "failure_count": self.failure_count,
                                    "total_executions": self.execution_count,
                                    "circuit_breaker_state": (
                                        "open"
                                        if self.circuit_breaker
                                        and self.circuit_breaker.is_open
                                        else "closed"
                                    ),
                                },
                            )
                        except Exception as emit_e:
                            self.logger.warning(
                                f"Failed to emit agent execution failed event: {emit_e}"
                            )

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

    async def _handle_retry_delay(self, retry_attempt: int) -> None:
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

    def get_node_definition(self) -> LangGraphNodeDefinition:
        """
        Get the LangGraph node definition for this agent.

        This method creates or returns the cached node definition that describes
        this agent as a node in a LangGraph DAG. Subclasses can override
        define_node_metadata() to customize the definition.

        Returns
        -------
        LangGraphNodeDefinition
            Complete node definition including inputs, outputs, and metadata
        """
        if self._node_definition is None:
            self._node_definition = self._create_default_node_definition()
        return self._node_definition

    def _create_default_node_definition(self) -> LangGraphNodeDefinition:
        """Create default node definition for this agent."""
        # Default input/output schemas
        inputs = [
            NodeInputSchema(
                name="context",
                description=f"Agent context containing query and state for {self.name}",
                required=True,
                type_hint="AgentContext",
            )
        ]

        outputs = [
            NodeOutputSchema(
                name="context",
                description=f"Updated context after {self.name} processing",
                type_hint="AgentContext",
            )
        ]

        # Allow subclasses to customize
        node_metadata = self.define_node_metadata()

        return LangGraphNodeDefinition(
            node_id=self.name.lower(),
            node_type=node_metadata.get("node_type", NodeType.PROCESSOR),
            description=node_metadata.get(
                "description", f"{self.name} agent for processing context"
            ),
            inputs=node_metadata.get("inputs", inputs),
            outputs=node_metadata.get("outputs", outputs),
            dependencies=node_metadata.get("dependencies", []),
            tags=node_metadata.get("tags", [self.name.lower(), "agent"]),
        )

    def define_node_metadata(self) -> Dict[str, Any]:
        """
        Define LangGraph node metadata for this agent.

        Subclasses should override this method to provide specific metadata
        for their node type, inputs, outputs, and dependencies.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing node metadata with keys:
            - node_type: NodeType enum value
            - description: Human-readable description
            - inputs: List of NodeInputSchema objects
            - outputs: List of NodeOutputSchema objects
            - dependencies: List of agent names this depends on
            - tags: List of string tags for categorization
        """
        # Default implementation - subclasses can override
        return {}

    def set_node_definition(self, node_definition: LangGraphNodeDefinition) -> None:
        """
        Set a custom node definition for this agent.

        This allows external configuration of the node definition,
        useful for dynamic graph construction.

        Parameters
        ----------
        node_definition : LangGraphNodeDefinition
            Complete node definition to use for this agent
        """
        self._node_definition = node_definition

    def validate_node_compatibility(self, input_context: AgentContext) -> bool:
        """
        Validate that the input context is compatible with this node's requirements.

        This method checks the input context against the node's input schema
        to ensure compatibility before execution.

        Parameters
        ----------
        input_context : AgentContext
            The context to validate

        Returns
        -------
        bool
            True if the context is compatible, False otherwise
        """
        node_def = self.get_node_definition()

        # Basic validation - check that required inputs are present
        for input_schema in node_def.inputs:
            if input_schema.required and input_schema.name == "context":
                # For now, just check that we have a valid AgentContext
                if not isinstance(input_context, AgentContext):
                    return False

        return True

    async def invoke(
        self, state: AgentContext, config: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """
        LangGraph-compatible node interface for agent execution.

        This method provides the standard LangGraph node signature while maintaining
        compatibility with our existing agent architecture. It delegates to the
        run_with_retry method to preserve all existing error handling and retry logic.

        Parameters
        ----------
        state : AgentContext
            The current state/context for the graph execution.
        config : Dict[str, Any], optional
            Optional configuration parameters for this invocation.
            Can include step_id, timeout overrides, or other execution parameters.

        Returns
        -------
        AgentContext
            The updated state/context after agent processing.
        """
        # Extract configuration parameters if provided
        step_id = None
        if config:
            step_id = config.get("step_id")

            # Allow config to override timeout for this specific invocation
            original_timeout = self.timeout_seconds
            if "timeout_seconds" in config:
                self.timeout_seconds = config["timeout_seconds"]

            try:
                result = await self.run_with_retry(state, step_id=step_id)
                return result
            finally:
                # Restore original timeout
                self.timeout_seconds = original_timeout
        else:
            # Standard execution without config overrides
            return await self.run_with_retry(state, step_id=step_id)

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
