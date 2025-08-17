"""
Observability context management for correlation tracking.

This module provides thread-local context management for correlation IDs
and observability metadata throughout the request lifecycle.
"""

import threading
import uuid
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class ObservabilityContext(BaseModel):
    """
    Observability context for tracking execution state.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.

    Contains correlation information and metadata that flows
    through the entire execution pipeline.
    """

    # Fields with defaults - UUID generation and optional fields
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique correlation identifier for tracing requests",
        min_length=1,
        max_length=200,
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    agent_name: Optional[str] = Field(
        None,
        description="Name of the agent currently executing",
        max_length=100,
        json_schema_extra={"example": "critic"},
    )
    step_id: Optional[str] = Field(
        None,
        description="Identifier for the current execution step",
        max_length=100,
        json_schema_extra={"example": "step_001_analyze"},
    )
    pipeline_id: Optional[str] = Field(
        None,
        description="Identifier for the execution pipeline",
        max_length=100,
        json_schema_extra={"example": "pipeline_main"},
    )
    execution_phase: Optional[str] = Field(
        None,
        description="Current phase of execution",
        max_length=50,
        json_schema_extra={"example": "analysis"},
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional observability metadata",
        json_schema_extra={
            "example": {"source": "api", "version": "1.0", "trace_level": "debug"}
        },
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    def with_agent(
        self, agent_name: str, step_id: Optional[str] = None
    ) -> "ObservabilityContext":
        """Create new context with agent information."""
        return ObservabilityContext(
            correlation_id=self.correlation_id,
            agent_name=agent_name,
            step_id=step_id or self.step_id,
            pipeline_id=self.pipeline_id,
            execution_phase=self.execution_phase,
            metadata=self.metadata.copy(),
        )

    def with_step(self, step_id: str) -> "ObservabilityContext":
        """Create new context with step information."""
        return ObservabilityContext(
            correlation_id=self.correlation_id,
            agent_name=self.agent_name,
            step_id=step_id,
            pipeline_id=self.pipeline_id,
            execution_phase=self.execution_phase,
            metadata=self.metadata.copy(),
        )

    def with_phase(self, execution_phase: str) -> "ObservabilityContext":
        """Create new context with execution phase."""
        return ObservabilityContext(
            correlation_id=self.correlation_id,
            agent_name=self.agent_name,
            step_id=self.step_id,
            pipeline_id=self.pipeline_id,
            execution_phase=execution_phase,
            metadata=self.metadata.copy(),
        )

    def with_metadata(self, **metadata: Any) -> "ObservabilityContext":
        """Create new context with additional metadata."""
        new_metadata = self.metadata.copy()
        new_metadata.update(metadata)

        return ObservabilityContext(
            correlation_id=self.correlation_id,
            agent_name=self.agent_name,
            step_id=self.step_id,
            pipeline_id=self.pipeline_id,
            execution_phase=self.execution_phase,
            metadata=new_metadata,
        )


# Thread-local storage for observability context
_context_storage = threading.local()


def get_observability_context() -> Optional[ObservabilityContext]:
    """
    Get current observability context.

    Returns
    -------
    ObservabilityContext or None
        Current context if available
    """
    return getattr(_context_storage, "context", None)


def set_observability_context(context: ObservabilityContext) -> None:
    """
    Set observability context for current thread.

    Parameters
    ----------
    context : ObservabilityContext
        Context to set
    """
    _context_storage.context = context


def clear_observability_context() -> None:
    """Clear observability context for current thread."""
    if hasattr(_context_storage, "context"):
        delattr(_context_storage, "context")


def get_correlation_id() -> Optional[str]:
    """
    Get current correlation ID.

    Returns
    -------
    str or None
        Current correlation ID if available
    """
    context = get_observability_context()
    return context.correlation_id if context else None


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for current thread.

    Parameters
    ----------
    correlation_id : str
        Correlation ID to set
    """
    context = get_observability_context()
    if context:
        context.correlation_id = correlation_id
    else:
        new_context = ObservabilityContext(correlation_id=correlation_id)
        set_observability_context(new_context)


def clear_correlation_id() -> None:
    """Clear correlation ID for current thread."""
    clear_observability_context()


class ObservabilityContextManager:
    """
    Context manager for observability context.

    Provides convenient management of observability context
    within a specific scope.
    """

    def __init__(self, context: ObservabilityContext) -> None:
        """
        Initialize context manager.

        Parameters
        ----------
        context : ObservabilityContext
            Context to set during scope
        """
        self.context = context
        self.previous_context: Optional[ObservabilityContext] = None

    def __enter__(self) -> ObservabilityContext:
        """Enter context scope."""
        self.previous_context = get_observability_context()
        set_observability_context(self.context)
        return self.context

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context scope."""
        if self.previous_context:
            set_observability_context(self.previous_context)
        else:
            clear_observability_context()


def observability_context(
    correlation_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    step_id: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    execution_phase: Optional[str] = None,
    **metadata: Any,
) -> ObservabilityContextManager:
    """
    Create observability context manager.

    Parameters
    ----------
    correlation_id : str, optional
        Correlation ID (generates new one if not provided)
    agent_name : str, optional
        Name of the agent
    step_id : str, optional
        Step identifier
    pipeline_id : str, optional
        Pipeline identifier
    execution_phase : str, optional
        Current execution phase
    **metadata
        Additional metadata

    Returns
    -------
    ObservabilityContextManager
        Context manager for the observability context
    """
    context = ObservabilityContext(
        correlation_id=correlation_id or str(uuid.uuid4()),
        agent_name=agent_name,
        step_id=step_id,
        pipeline_id=pipeline_id,
        execution_phase=execution_phase,
        metadata=metadata,
    )

    return ObservabilityContextManager(context)
