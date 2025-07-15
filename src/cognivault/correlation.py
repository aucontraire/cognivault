"""
Correlation Context Management for CogniVault.

Provides correlation ID propagation through async execution chains using
contextvars. This enables tracing of requests through the orchestrator,
agents, and event system for debugging and observability.
"""

import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Dict, Any
from datetime import datetime, timezone

from cognivault.observability import get_logger

logger = get_logger(__name__)

# Context variables for correlation tracking
context_correlation_id: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
context_workflow_id: ContextVar[Optional[str]] = ContextVar("workflow_id", default=None)
context_parent_span_id: ContextVar[Optional[str]] = ContextVar(
    "parent_span_id", default=None
)
context_trace_metadata: ContextVar[Dict[str, Any]] = ContextVar(
    "trace_metadata", default={}
)


@dataclass
class CorrelationContext:
    """
    Correlation context for tracing requests through the system.

    Provides access to correlation information that automatically
    propagates through async execution chains.
    """

    correlation_id: str
    workflow_id: str
    parent_span_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert correlation context to dictionary for serialization."""
        return {
            "correlation_id": self.correlation_id,
            "workflow_id": self.workflow_id,
            "parent_span_id": self.parent_span_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrelationContext":
        """Create correlation context from dictionary."""
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None
        )
        return cls(
            correlation_id=data["correlation_id"],
            workflow_id=data["workflow_id"],
            parent_span_id=data.get("parent_span_id"),
            metadata=data.get("metadata", {}),
            created_at=created_at or datetime.now(timezone.utc),
        )


@asynccontextmanager
async def trace(
    correlation_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[CorrelationContext, None]:
    """
    Context manager for explicit correlation control.

    Sets correlation context for the duration of the async context and
    automatically restores the previous context when exiting.

    Args:
        correlation_id: Explicit correlation ID, or generates new one
        workflow_id: Explicit workflow ID, or generates new one
        parent_span_id: Parent span for nested operations
        metadata: Additional trace metadata

    Yields:
        CorrelationContext with current trace information

    Example:
        async with trace(correlation_id="custom-trace-123") as ctx:
            result = await orchestrator.run(query, config)
            # All events emitted within this context use custom-trace-123
    """
    # Generate IDs if not provided
    current_correlation = correlation_id or str(uuid.uuid4())
    current_workflow = workflow_id or str(uuid.uuid4())
    current_metadata = metadata or {}

    # Get current context values to restore later
    prev_correlation = context_correlation_id.get(None)
    prev_workflow = context_workflow_id.get(None)
    prev_parent_span = context_parent_span_id.get(None)
    prev_metadata = context_trace_metadata.get({})

    # Set new context
    correlation_token = context_correlation_id.set(current_correlation)
    workflow_token = context_workflow_id.set(current_workflow)
    parent_span_token = context_parent_span_id.set(parent_span_id)
    metadata_token = context_trace_metadata.set(current_metadata)

    context = CorrelationContext(
        correlation_id=current_correlation,
        workflow_id=current_workflow,
        parent_span_id=parent_span_id,
        metadata=current_metadata,
    )

    logger.debug(
        f"Starting trace context: correlation_id={current_correlation}, "
        f"workflow_id={current_workflow}, parent_span_id={parent_span_id}"
    )

    try:
        yield context
    finally:
        # Restore previous context
        context_correlation_id.reset(correlation_token)
        context_workflow_id.reset(workflow_token)
        context_parent_span_id.reset(parent_span_token)
        context_trace_metadata.reset(metadata_token)

        logger.debug(f"Restored trace context: correlation_id={prev_correlation}")


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return context_correlation_id.get(None)


def get_workflow_id() -> Optional[str]:
    """Get the current workflow ID from context."""
    return context_workflow_id.get(None)


def get_parent_span_id() -> Optional[str]:
    """Get the current parent span ID from context."""
    return context_parent_span_id.get(None)


def get_trace_metadata() -> Dict[str, Any]:
    """Get the current trace metadata from context."""
    return context_trace_metadata.get({})


def get_current_context() -> Optional[CorrelationContext]:
    """
    Get the current correlation context if available.

    Returns:
        CorrelationContext if context is active, None otherwise
    """
    correlation_id = get_correlation_id()
    workflow_id = get_workflow_id()

    if not correlation_id or not workflow_id:
        return None

    return CorrelationContext(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        parent_span_id=get_parent_span_id(),
        metadata=get_trace_metadata(),
    )


def ensure_correlation_context() -> CorrelationContext:
    """
    Ensure a correlation context exists, creating one if necessary.

    This is useful for operations that should always have trace context,
    even if not explicitly created by the caller.

    Returns:
        CorrelationContext (existing or newly created)
    """
    current = get_current_context()
    if current:
        return current

    # Create new context
    correlation_id = str(uuid.uuid4())
    workflow_id = str(uuid.uuid4())

    # Set in context vars
    context_correlation_id.set(correlation_id)
    context_workflow_id.set(workflow_id)
    context_trace_metadata.set({})

    logger.debug(f"Created new correlation context: {correlation_id}")

    return CorrelationContext(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        metadata={},
    )


def create_child_span(
    operation_name: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a child span ID for nested operations.

    Args:
        operation_name: Name of the operation for the child span
        metadata: Additional metadata for the child span

    Returns:
        Child span ID that can be used as parent_span_id for further nesting
    """
    current_correlation = get_correlation_id()
    current_span = get_parent_span_id()

    # Generate child span ID
    child_span_id = f"{operation_name}:{uuid.uuid4().hex[:8]}"

    # Update metadata with span information
    current_metadata = get_trace_metadata().copy()
    current_metadata.update(
        {
            "current_span": child_span_id,
            "parent_span": current_span,
            "operation": operation_name,
            **(metadata or {}),
        }
    )

    # Update context
    context_parent_span_id.set(child_span_id)
    context_trace_metadata.set(current_metadata)

    logger.debug(
        f"Created child span: {child_span_id} (parent: {current_span}, "
        f"correlation: {current_correlation})"
    )

    return child_span_id


def add_trace_metadata(key: str, value: Any) -> None:
    """
    Add metadata to the current trace context.

    Args:
        key: Metadata key
        value: Metadata value
    """
    current_metadata = get_trace_metadata().copy()
    current_metadata[key] = value
    context_trace_metadata.set(current_metadata)


def get_correlation_headers() -> Dict[str, str]:
    """
    Get correlation information as HTTP headers.

    Useful for propagating correlation across service boundaries.

    Returns:
        Dictionary of correlation headers
    """
    correlation_id = get_correlation_id()
    workflow_id = get_workflow_id()
    parent_span_id = get_parent_span_id()

    headers = {}

    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    if workflow_id:
        headers["X-Workflow-ID"] = workflow_id
    if parent_span_id:
        headers["X-Parent-Span-ID"] = parent_span_id

    return headers


async def propagate_correlation(func, *args, **kwargs):
    """
    Decorator-like function to ensure correlation context propagates to async functions.

    Args:
        func: Async function to call
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call with correlation context preserved
    """
    current_context = get_current_context()

    if current_context:
        # Context already exists, just call the function
        return await func(*args, **kwargs)
    else:
        # No context, create one for this operation
        async with trace() as ctx:
            return await func(*args, **kwargs)


# Convenience functions for common patterns
async def with_correlation(
    correlation_id: str, workflow_id: Optional[str] = None, func=None, *args, **kwargs
):
    """
    Execute a function with explicit correlation context.

    Args:
        correlation_id: Correlation ID to use
        workflow_id: Workflow ID to use (optional, generates if not provided)
        func: Async function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function execution
    """
    async with trace(correlation_id=correlation_id, workflow_id=workflow_id):
        if func:
            return await func(*args, **kwargs)
        else:
            return get_current_context()


def is_traced() -> bool:
    """
    Check if we're currently in a trace context.

    Returns:
        True if correlation context is active
    """
    return get_correlation_id() is not None
