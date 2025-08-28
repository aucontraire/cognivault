"""
LangGraph node wrappers for CogniVault agents.

This module provides wrapper functions that convert CogniVault agents
into LangGraph-compatible node functions. Each wrapper handles:
- State conversion between AgentContext and LangGraph state
- Error handling with circuit breaker patterns
- Async execution with proper timeout handling
- Logging and metrics integration
- Output formatting and validation

Design Principles:
- Preserve agent autonomy while enabling DAG execution
- Maintain backward compatibility with existing agents
- Provide robust error handling and recovery
- Enable comprehensive observability and debugging
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Callable,
    Coroutine,
    TypeVar,
    Union,
    cast,
    Protocol,
    Generator,
)
from functools import wraps

from langgraph.runtime import Runtime

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.orchestration.state_bridge import AgentContextStateBridge
from cognivault.orchestration.state_schemas import (
    CogniVaultState,
    CogniVaultContext,
    RefinerState,
    CriticState,
    HistorianState,
    SynthesisState,
    record_agent_error,
)
from cognivault.observability import get_logger
from cognivault.config.openai_config import OpenAIConfig
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.correlation import get_correlation_id, get_workflow_id
from cognivault.events import (
    emit_agent_execution_started,
    emit_agent_execution_completed,
)
from cognivault.events.types import EventCategory
from cognivault.utils.content_truncation import truncate_for_websocket_event

logger = get_logger(__name__)


# Global timing registry to store node execution times
# This allows timing data to be collected across node executions
# and retrieved when creating workflow results
_TIMING_REGISTRY: Dict[str, Dict[str, float]] = {}


def get_timing_registry() -> Dict[str, Dict[str, float]]:
    """Get the current timing registry."""
    return _TIMING_REGISTRY.copy()


def clear_timing_registry() -> None:
    """Clear the timing registry for a new workflow execution."""
    global _TIMING_REGISTRY
    _TIMING_REGISTRY.clear()


def register_node_timing(
    execution_id: str, node_name: str, execution_time_seconds: float
) -> None:
    """Register node execution timing data."""
    global _TIMING_REGISTRY
    if execution_id not in _TIMING_REGISTRY:
        _TIMING_REGISTRY[execution_id] = {}
    _TIMING_REGISTRY[execution_id][node_name] = execution_time_seconds


class NodeExecutionError(Exception):
    """Raised when a node execution fails."""

    pass


F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


class CircuitBreakerFunction(Protocol):
    """Protocol for functions decorated with @circuit_breaker.

    This protocol defines the attributes that are dynamically added
    by the circuit_breaker decorator, making them visible to type checkers.
    """

    _failure_count: int
    _circuit_open: bool
    _last_failure_time: Optional[float]

    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        """Function call signature for circuit breaker decorated functions."""
        ...


def circuit_breaker(
    max_failures: int = 3, reset_timeout: float = 300.0
) -> Callable[[F], F]:
    """
    Circuit breaker decorator for node functions.

    Prevents cascade failures by stopping execution after repeated failures.

    Parameters
    ----------
    max_failures : int
        Maximum number of failures before circuit opens
    reset_timeout : float
        Time in seconds before circuit can be retried
    """

    def decorator(func: F) -> F:
        # Store state in a dictionary to avoid attribute issues
        circuit_state: Dict[str, Union[int, float, bool, None]] = {
            "failure_count": 0,
            "last_failure_time": None,
            "circuit_open": False,
        }

        def _sync_state() -> None:
            """Sync internal state with exposed attributes"""
            # Set attributes directly on the function object for testing access
            setattr(wrapper, "_failure_count", circuit_state["failure_count"])
            setattr(wrapper, "_last_failure_time", circuit_state["last_failure_time"])
            setattr(wrapper, "_circuit_open", circuit_state["circuit_open"])

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Sync state from external attributes if they were manually set
            if hasattr(wrapper, "_failure_count"):
                circuit_state["failure_count"] = getattr(wrapper, "_failure_count")
            if hasattr(wrapper, "_last_failure_time"):
                circuit_state["last_failure_time"] = getattr(
                    wrapper, "_last_failure_time"
                )
            if hasattr(wrapper, "_circuit_open"):
                circuit_state["circuit_open"] = getattr(wrapper, "_circuit_open")

            # Check if circuit is open
            if circuit_state["circuit_open"]:
                if circuit_state["last_failure_time"]:
                    time_since_failure = (
                        time.time() - circuit_state["last_failure_time"]
                    )
                    if time_since_failure < reset_timeout:
                        raise NodeExecutionError(
                            f"Circuit breaker open for {func.__name__}. "
                            f"Retry in {reset_timeout - time_since_failure:.1f}s"
                        )
                    else:
                        # Reset circuit
                        circuit_state["circuit_open"] = False
                        circuit_state["failure_count"] = 0
                        logger.info(f"Circuit breaker reset for {func.__name__}")

            try:
                result = await func(*args, **kwargs)
                # Reset on success
                circuit_state["failure_count"] = 0
                circuit_state["circuit_open"] = False
                _sync_state()
                return result

            except Exception:
                # Type-safe increment of failure count
                current_count = cast(int, circuit_state["failure_count"])
                circuit_state["failure_count"] = current_count + 1
                circuit_state["last_failure_time"] = time.time()

                if cast(int, circuit_state["failure_count"]) >= max_failures:
                    circuit_state["circuit_open"] = True
                    logger.error(
                        f"Circuit breaker opened for {func.__name__} "
                        f"after {circuit_state['failure_count']} failures"
                    )

                _sync_state()
                raise

        # Initialize attributes on wrapper before returning
        setattr(wrapper, "_failure_count", 0)
        setattr(wrapper, "_last_failure_time", None)
        setattr(wrapper, "_circuit_open", False)

        return cast(F, wrapper)

    return decorator


def node_metrics(func: F) -> F:
    """
    Decorator to add metrics collection to node functions.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        node_name = func.__name__.replace("_node", "")

        try:
            logger.info(f"Starting execution of {node_name} node")
            result = await func(*args, **kwargs)

            execution_time_ms = (time.time() - start_time) * 1000
            execution_time_seconds = execution_time_ms / 1000
            logger.info(
                f"Completed {node_name} node execution in {execution_time_ms:.2f}ms"
            )

            # Store timing data directly in the result context if available
            # This ensures timing data is available regardless of the execution path
            context_updated = False

            # Try to find and update AgentContext in the result
            if hasattr(result, "get") and "final_context" in result:
                # If result has a final_context, store timing there
                final_context = result["final_context"]
                if hasattr(final_context, "execution_state"):
                    if "_node_execution_times" not in final_context.execution_state:
                        final_context.execution_state["_node_execution_times"] = {}
                    final_context.execution_state["_node_execution_times"][
                        node_name
                    ] = execution_time_seconds
                    context_updated = True

            # Also try to extract execution_id for registry (backup approach)
            execution_id = None
            if args and isinstance(args[0], dict):
                state = args[0]

                # Try various ways to get execution_id
                if "execution_metadata" in state:
                    execution_id = state["execution_metadata"].get("execution_id")
                elif "execution_id" in state:
                    execution_id = state["execution_id"]
                else:
                    # Check nested structures
                    for key, value in state.items():
                        if isinstance(value, dict) and "execution_id" in value:
                            execution_id = value["execution_id"]
                            break

            if execution_id:
                register_node_timing(execution_id, node_name, execution_time_seconds)
            elif not context_updated:
                logger.debug(
                    f"Could not store timing data for {node_name} node - no execution_id and no context access"
                )

            # Also store timing data in result for immediate access
            if isinstance(result, dict):
                if "_node_execution_times" not in result:
                    result["_node_execution_times"] = {}
                result["_node_execution_times"][node_name] = {
                    "execution_time_seconds": execution_time_seconds,
                    "execution_time_ms": execution_time_ms,
                    "completed": True,
                }

            return result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Failed {node_name} node execution after {execution_time_ms:.2f}ms: {e}"
            )
            raise

    return cast(F, wrapper)


async def create_agent_with_llm(agent_name: str) -> BaseAgent:
    """
    Create an agent instance with LLM configuration.

    Parameters
    ----------
    agent_name : str
        Name of the agent to create

    Returns
    -------
    BaseAgent
        Configured agent instance
    """
    registry = get_agent_registry()

    # Initialize LLM
    llm_config = OpenAIConfig.load()
    llm = OpenAIChatLLM(
        api_key=llm_config.api_key,
        model=llm_config.model,
        base_url=llm_config.base_url,
    )

    # Create agent
    agent = registry.create_agent(agent_name.lower(), llm=llm)
    return agent


async def convert_state_to_context(state: CogniVaultState) -> AgentContext:
    """
    Convert LangGraph state to AgentContext for agent execution.

    Handles both complete and partial state objects gracefully.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (may be partial)

    Returns
    -------
    AgentContext
        AgentContext for agent execution
    """
    bridge = AgentContextStateBridge()

    # Create AgentContext from query - handle partial state
    query = state.get("query", "")
    if "query" not in state:
        raise ValueError("State must contain a query field")

    context = AgentContext(query=query)

    # Add any existing agent outputs to context with proper keys
    if state.get("refiner"):
        refiner_state: Optional[RefinerState] = state["refiner"]
        if refiner_state is not None:
            refined_question = refiner_state.get("refined_question", "")
            # Add with both 'refiner' and 'Refiner' keys for compatibility
            if refined_question:
                context.add_agent_output("refiner", refined_question)
                context.add_agent_output("Refiner", refined_question)
                context.execution_state["refiner_topics"] = refiner_state.get(
                    "topics", []
                )
                context.execution_state["refiner_confidence"] = refiner_state.get(
                    "confidence", 0.8
                )
                logger.info(
                    f"Added refiner output to context: {str(refined_question)[:100]}..."
                )
            else:
                logger.warning(
                    "Refiner output found in state but refined_question is empty"
                )

    if state.get("critic"):
        critic_state: Optional[CriticState] = state["critic"]
        if critic_state is not None:
            critique = critic_state.get("critique", "")
            # Add with both 'critic' and 'Critic' keys for compatibility
            if critique:
                context.add_agent_output("critic", critique)
                context.add_agent_output("Critic", critique)
                context.execution_state["critic_suggestions"] = critic_state.get(
                    "suggestions", []
                )
                context.execution_state["critic_severity"] = critic_state.get(
                    "severity", "medium"
                )
                logger.info(f"Added critic output to context: {str(critique)[:100]}...")
            else:
                logger.warning("Critic output found in state but critique is empty")

    if state.get("historian"):
        historian_state: Optional[HistorianState] = state["historian"]
        if historian_state is not None:
            historical_summary = historian_state.get("historical_summary", "")
            # Add with both 'historian' and 'Historian' keys for compatibility
            if historical_summary:
                context.add_agent_output("historian", historical_summary)
                context.add_agent_output("Historian", historical_summary)
                context.execution_state["historian_retrieved_notes"] = (
                    historian_state.get("retrieved_notes", [])
                )
                context.execution_state["historian_search_strategy"] = (
                    historian_state.get("search_strategy", "hybrid")
                )
                context.execution_state["historian_topics_found"] = historian_state.get(
                    "topics_found", []
                )
                context.execution_state["historian_confidence"] = historian_state.get(
                    "confidence", 0.8
                )
                logger.info(
                    f"Added historian output to context: {str(historical_summary)[:100]}..."
                )
            else:
                logger.warning(
                    "Historian output found in state but historical_summary is empty"
                )

    # Add execution metadata - handle partial state
    execution_metadata = state.get("execution_metadata", {})
    if execution_metadata:
        context.execution_state.update(
            {
                "execution_id": execution_metadata.get("execution_id", ""),
                "orchestrator_type": "langgraph-real",
                "successful_agents": (
                    state.get("successful_agents", []).copy()
                    if isinstance(state.get("successful_agents"), list)
                    else []
                ),
                "failed_agents": (
                    state.get("failed_agents", []).copy()
                    if isinstance(state.get("failed_agents"), list)
                    else []
                ),
            }
        )

    return context


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def refiner_node(
    state: CogniVaultState, runtime: Runtime[CogniVaultContext]
) -> Dict[str, Any]:
    """
    LangGraph node wrapper for RefinerAgent.

    Transforms the raw user query into a structured, clarified prompt
    and updates the state with typed RefinerState.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state
    runtime : Runtime[CogniVaultContext]
        LangGraph runtime context with thread and execution information

    Returns
    -------
    CogniVaultState
        Updated state with RefinerState
    """
    # ✅ Type-safe context access (LangGraph 0.6.x Runtime API)
    thread_id = runtime.context.thread_id
    execution_id = runtime.context.execution_id
    original_query = runtime.context.query
    correlation_id = runtime.context.correlation_id
    checkpoint_enabled = runtime.context.enable_checkpoints

    logger.info(
        f"Executing refiner node in thread {thread_id}",
        extra={
            "thread_id": thread_id,
            "execution_id": execution_id,
            "correlation_id": correlation_id,
            "query_length": len(original_query or ""),
            "checkpoint_enabled": checkpoint_enabled,
        },
    )

    # Backward compatibility: use execution_id as workflow_id
    workflow_id = execution_id

    try:
        # ✅ Enhanced event emission with runtime context
        await emit_agent_execution_started(
            event_category=EventCategory.EXECUTION,
            workflow_id=workflow_id,
            agent_name="refiner",
            input_context={
                "query": original_query or state.get("query", ""),
                "node_type": "refiner",
                "thread_id": thread_id,
                "execution_id": execution_id,
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_context": {
                    "thread_id": thread_id,
                    "execution_id": execution_id,
                    "checkpoint_enabled": checkpoint_enabled,
                    "query_length": len(original_query or ""),
                },
            },
        )

        # Create agent
        agent = await create_agent_with_llm("refiner")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract refiner output (using "Refiner" key, not "refiner")
        refiner_raw_output = result_context.agent_outputs.get("refiner", "")

        # Create typed output
        refiner_state = RefinerState(
            refined_question=refiner_raw_output,
            topics=result_context.execution_state.get("topics", []),
            confidence=result_context.execution_state.get("confidence", 0.8),
            processing_notes=result_context.execution_state.get("processing_notes"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Get token usage information from agent context
        refiner_token_usage = result_context.get_agent_token_usage("refiner")

        # ✅ Enhanced completion event emission with runtime context and token usage
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=workflow_id,
            agent_name="refiner",
            success=True,
            output_context={
                "refined_question": truncate_for_websocket_event(
                    str(refiner_raw_output), "refined_question"
                ),
                "node_type": "refiner",
                "thread_id": thread_id,
                "execution_id": execution_id,
                "confidence": refiner_state["confidence"],
                "topics_count": len(refiner_state["topics"]),
                # Add token usage to output context for WebSocket events
                "input_tokens": refiner_token_usage["input_tokens"],
                "output_tokens": refiner_token_usage["output_tokens"],
                "total_tokens": refiner_token_usage["total_tokens"],
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_context": {
                    "thread_id": thread_id,
                    "execution_id": execution_id,
                    "checkpoint_enabled": checkpoint_enabled,
                    "original_query": original_query,
                },
                "output_metadata": {
                    "confidence": refiner_state["confidence"],
                    "topics_identified": len(refiner_state["topics"]),
                    "processing_notes": bool(refiner_state["processing_notes"]),
                },
                # Add token usage to metadata as well for comprehensive tracking
                "token_usage": refiner_token_usage,
            },
        )

        # ✅ Enhanced success logging with runtime context
        logger.info(
            f"Refiner node completed successfully for thread {thread_id}",
            extra={
                "thread_id": thread_id,
                "execution_id": execution_id,
                "correlation_id": correlation_id,
                "confidence": refiner_state["confidence"],
                "topics_count": len(refiner_state["topics"]),
                "query_length": len(original_query or ""),
                "checkpoint_enabled": checkpoint_enabled,
            },
        )
        return {
            "refiner": refiner_state,
            "successful_agents": ["refiner"],
        }

    except Exception as e:
        # ✅ Enhanced error logging with runtime context
        logger.error(
            f"Refiner node failed for thread {thread_id}: {e}",
            extra={
                "thread_id": thread_id,
                "execution_id": execution_id,
                "correlation_id": correlation_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "checkpoint_enabled": checkpoint_enabled,
                "query_length": len(original_query or ""),
            },
        )

        # ✅ Enhanced error event emission with runtime context
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=workflow_id,
            agent_name="refiner",
            success=False,
            output_context={
                "error": str(e),
                "node_type": "refiner",
                "thread_id": thread_id,
                "execution_id": execution_id,
            },
            error_message=str(e),
            error_type=type(e).__name__,
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_context": {
                    "thread_id": thread_id,
                    "execution_id": execution_id,
                    "checkpoint_enabled": checkpoint_enabled,
                    "original_query": original_query,
                },
                "error_context": {
                    "error_type": type(e).__name__,
                    "query_length": len(original_query or ""),
                },
            },
        )

        error_state = record_agent_error(state, "refiner", e)
        raise NodeExecutionError(f"Refiner execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def critic_node(
    state: CogniVaultState, runtime: Runtime[CogniVaultContext]
) -> Dict[str, Any]:
    """
    LangGraph node wrapper for CriticAgent.

    Provides analytical critique of the refined query and adds
    suggestions for improvement to the state.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (must contain refiner output)
    runtime : Runtime[CogniVaultContext]
        LangGraph 0.6.x runtime context providing type-safe access to execution metadata

    Returns
    -------
    CogniVaultState
        Updated state with CriticState
    """
    # Extract context data from LangGraph 0.6.x Runtime Context API
    thread_id = runtime.context.thread_id
    execution_id = runtime.context.execution_id
    original_query = runtime.context.query
    correlation_id = runtime.context.correlation_id
    checkpoint_enabled = runtime.context.enable_checkpoints

    logger.info(
        f"Executing critic node in thread {thread_id}",
        extra={
            "thread_id": thread_id,
            "execution_id": execution_id,
            "correlation_id": correlation_id,
            "query_length": len(original_query or ""),
            "checkpoint_enabled": checkpoint_enabled,
        },
    )

    try:
        # Validate dependencies
        if not state.get("refiner"):
            raise NodeExecutionError("Critic node requires refiner output")

        # Emit agent execution started event with runtime context
        await emit_agent_execution_started(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="critic",
            input_context={
                "query": original_query or "",
                "has_refiner_output": True,
                "node_type": "critic",
                "thread_id": thread_id,
                "runtime_context": True,
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        # Create agent
        agent = await create_agent_with_llm("critic")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract critic output (using "Critic" key, not "critic")
        critic_raw_output = result_context.agent_outputs.get("critic", "")

        # Create typed output
        critic_state = CriticState(
            critique=critic_raw_output,
            suggestions=result_context.execution_state.get("suggestions", []),
            severity=result_context.execution_state.get("severity", "medium"),
            strengths=result_context.execution_state.get("strengths", []),
            weaknesses=result_context.execution_state.get("weaknesses", []),
            confidence=result_context.execution_state.get("confidence", 0.7),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Get token usage information from agent context
        critic_token_usage = result_context.get_agent_token_usage("critic")

        # Emit agent execution completed event with runtime context and token usage
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="critic",
            success=True,
            output_context={
                "critique": truncate_for_websocket_event(
                    str(critic_raw_output), "critique"
                ),
                "node_type": "critic",
                "thread_id": thread_id,
                "runtime_context": True,
                "confidence": critic_state["confidence"],
                # Add token usage to output context for WebSocket events
                "input_tokens": critic_token_usage["input_tokens"],
                "output_tokens": critic_token_usage["output_tokens"],
                "total_tokens": critic_token_usage["total_tokens"],
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        # Return state update with agent output and success tracking
        logger.info("Critic node completed successfully")
        return {
            "critic": critic_state,
            "successful_agents": ["critic"],
        }

    except Exception as e:
        logger.error(f"Critic node failed: {e}")

        # Emit agent execution failed event with runtime context
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="critic",
            success=False,
            output_context={
                "error": str(e),
                "node_type": "critic",
                "thread_id": thread_id,
                "runtime_context": True,
            },
            error_message=str(e),
            error_type=type(e).__name__,
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        error_state = record_agent_error(state, "critic", e)
        raise NodeExecutionError(f"Critic execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def historian_node(
    state: CogniVaultState, runtime: Runtime[CogniVaultContext]
) -> Dict[str, Any]:
    """
    LangGraph node wrapper for HistorianAgent.

    Retrieves and analyzes historical context using intelligent search
    and LLM-powered relevance analysis.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (must contain refiner output)
    runtime : Runtime[CogniVaultContext]
        LangGraph 0.6.x runtime context providing type-safe access to execution metadata

    Returns
    -------
    CogniVaultState
        Updated state with HistorianState
    """
    # Extract context data from LangGraph 0.6.x Runtime Context API
    thread_id = runtime.context.thread_id
    execution_id = runtime.context.execution_id
    original_query = runtime.context.query
    correlation_id = runtime.context.correlation_id
    checkpoint_enabled = runtime.context.enable_checkpoints

    logger.info(
        f"Executing historian node in thread {thread_id}",
        extra={
            "thread_id": thread_id,
            "execution_id": execution_id,
            "correlation_id": correlation_id,
            "query_length": len(original_query or ""),
            "checkpoint_enabled": checkpoint_enabled,
        },
    )

    try:
        # Validate dependencies
        if not state.get("refiner"):
            raise NodeExecutionError("Historian node requires refiner output")

        # Emit agent execution started event with runtime context
        await emit_agent_execution_started(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="historian",
            input_context={
                "query": original_query or "",
                "has_refiner_output": True,
                "node_type": "historian",
                "thread_id": thread_id,
                "runtime_context": True,
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        # Create agent
        agent = await create_agent_with_llm("historian")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract historian output (using "Historian" key, not "historian")
        historian_raw_output = result_context.agent_outputs.get("historian", "")

        # Extract retrieved notes from context
        retrieved_notes = getattr(result_context, "retrieved_notes", [])

        # Determine topics found from retrieved notes context
        topics_found = []
        if hasattr(result_context, "execution_state"):
            topics_found = result_context.execution_state.get("topics_found", [])

        # Create typed output
        historian_state = HistorianState(
            historical_summary=historian_raw_output,
            retrieved_notes=retrieved_notes,
            search_results_count=result_context.execution_state.get(
                "search_results_count", 0
            ),
            filtered_results_count=result_context.execution_state.get(
                "filtered_results_count", 0
            ),
            search_strategy=result_context.execution_state.get(
                "search_strategy", "hybrid"
            ),
            topics_found=topics_found,
            confidence=result_context.execution_state.get("confidence", 0.8),
            llm_analysis_used=result_context.execution_state.get(
                "llm_analysis_used", True
            ),
            metadata=result_context.execution_state.get("historian_metadata", {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Get token usage information from agent context
        historian_token_usage = result_context.get_agent_token_usage("historian")

        # Emit agent execution completed event with runtime context and token usage
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="historian",
            success=True,
            output_context={
                "historical_summary": truncate_for_websocket_event(
                    str(historian_raw_output), "historical_summary"
                ),
                "node_type": "historian",
                "thread_id": thread_id,
                "runtime_context": True,
                "confidence": historian_state["confidence"],
                "search_strategy": historian_state["search_strategy"],
                # Add token usage to output context for WebSocket events
                "input_tokens": historian_token_usage["input_tokens"],
                "output_tokens": historian_token_usage["output_tokens"],
                "total_tokens": historian_token_usage["total_tokens"],
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        # Return state update with agent output and success tracking
        logger.info("Historian node completed successfully")
        return {
            "historian": historian_state,
            "successful_agents": ["historian"],
        }

    except Exception as e:
        logger.error(f"Historian node failed: {e}")

        # Emit agent execution failed event with runtime context
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="historian",
            success=False,
            output_context={
                "error": str(e),
                "node_type": "historian",
                "thread_id": thread_id,
                "runtime_context": True,
            },
            error_message=str(e),
            error_type=type(e).__name__,
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        error_state = record_agent_error(state, "historian", e)
        raise NodeExecutionError(f"Historian execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def synthesis_node(
    state: CogniVaultState, runtime: Runtime[CogniVaultContext]
) -> Dict[str, Any]:
    """
    LangGraph node wrapper for SynthesisAgent.

    Generates final synthesis from multiple agent outputs,
    creating coherent and comprehensive analysis.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (must contain refiner, critic, and historian outputs)
    runtime : Runtime[CogniVaultContext]
        LangGraph 0.6.x runtime context providing type-safe access to execution metadata

    Returns
    -------
    CogniVaultState
        Updated state with SynthesisState
    """
    # Extract context data from LangGraph 0.6.x Runtime Context API
    thread_id = runtime.context.thread_id
    execution_id = runtime.context.execution_id
    original_query = runtime.context.query
    correlation_id = runtime.context.correlation_id
    checkpoint_enabled = runtime.context.enable_checkpoints

    logger.info(
        f"Executing synthesis node in thread {thread_id}",
        extra={
            "thread_id": thread_id,
            "execution_id": execution_id,
            "correlation_id": correlation_id,
            "query_length": len(original_query or ""),
            "checkpoint_enabled": checkpoint_enabled,
        },
    )

    try:
        # Validate dependencies
        if not state.get("refiner"):
            raise NodeExecutionError("Synthesis node requires refiner output")
        if not state.get("critic"):
            raise NodeExecutionError("Synthesis node requires critic output")
        if not state.get("historian"):
            raise NodeExecutionError("Synthesis node requires historian output")

        # Emit agent execution started event with runtime context
        await emit_agent_execution_started(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="synthesis",
            input_context={
                "query": original_query or "",
                "has_all_inputs": True,
                "node_type": "synthesis",
                "thread_id": thread_id,
                "runtime_context": True,
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        # Create agent
        agent = await create_agent_with_llm("synthesis")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract synthesis output (using "Synthesis" key, not "synthesis")
        synthesis_raw_output = result_context.agent_outputs.get("synthesis", "")

        # Determine sources used
        sources_used = []
        if state.get("refiner"):
            sources_used.append("refiner")
        if state.get("critic"):
            sources_used.append("critic")
        if state.get("historian"):
            sources_used.append("historian")

        # Create typed output
        synthesis_state = SynthesisState(
            final_analysis=synthesis_raw_output,
            key_insights=result_context.execution_state.get("key_insights", []),
            sources_used=sources_used,
            themes_identified=result_context.execution_state.get("themes", []),
            conflicts_resolved=result_context.execution_state.get(
                "conflicts_resolved", 0
            ),
            confidence=result_context.execution_state.get("confidence", 0.8),
            metadata=result_context.execution_state.get("synthesis_metadata", {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Get token usage information from agent context
        synthesis_token_usage = result_context.get_agent_token_usage("synthesis")

        # Emit agent execution completed event with runtime context and token usage
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="synthesis",
            success=True,
            output_context={
                "final_analysis": truncate_for_websocket_event(
                    str(synthesis_raw_output), "final_analysis"
                ),
                "node_type": "synthesis",
                "thread_id": thread_id,
                "runtime_context": True,
                "confidence": synthesis_state["confidence"],
                "sources_used": synthesis_state["sources_used"],
                # Add token usage to output context for WebSocket events
                "input_tokens": synthesis_token_usage["input_tokens"],
                "output_tokens": synthesis_token_usage["output_tokens"],
                "total_tokens": synthesis_token_usage["total_tokens"],
            },
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        # Return state update with agent output and success tracking
        logger.info("Synthesis node completed successfully")
        return {
            "synthesis": synthesis_state,
            "successful_agents": ["synthesis"],
        }

    except Exception as e:
        logger.error(f"Synthesis node failed: {e}")

        # Emit agent execution failed event with runtime context
        await emit_agent_execution_completed(
            event_category=EventCategory.EXECUTION,
            workflow_id=execution_id,
            agent_name="synthesis",
            success=False,
            output_context={
                "error": str(e),
                "node_type": "synthesis",
                "thread_id": thread_id,
                "runtime_context": True,
            },
            error_message=str(e),
            error_type=type(e).__name__,
            correlation_id=correlation_id,
            metadata={
                "node_execution": True,
                "orchestrator_type": "langgraph-real",
                "runtime_api_version": "0.6.4",
                "thread_id": thread_id,
                "checkpoint_enabled": checkpoint_enabled,
            },
        )

        error_state = record_agent_error(state, "synthesis", e)
        raise NodeExecutionError(f"Synthesis execution failed: {e}") from e


async def handle_node_timeout(
    coro: Coroutine[Any, Any, Any], timeout_seconds: float = 30.0
) -> Any:
    """
    Handle node execution with timeout.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to execute
    timeout_seconds : float
        Timeout in seconds

    Returns
    -------
    Any
        Coroutine result

    Raises
    ------
    NodeExecutionError
        If execution times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise NodeExecutionError(f"Node execution timed out after {timeout_seconds}s")


def get_node_dependencies() -> Dict[str, List[str]]:
    """
    Get node dependency mapping for DAG construction.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of node name to list of required predecessor nodes
    """
    return {
        "refiner": [],  # No dependencies
        "critic": ["refiner"],  # Requires refiner output
        "historian": ["refiner"],  # Requires refiner output
        "synthesis": ["critic", "historian"],  # Requires critic and historian outputs
    }


def validate_node_input(state: CogniVaultState, node_name: str) -> bool:
    """
    Validate that a node has all required inputs.

    Parameters
    ----------
    state : CogniVaultState
        Current state
    node_name : str
        Name of node to validate

    Returns
    -------
    bool
        True if node can execute, False otherwise
    """
    dependencies = get_node_dependencies()
    required_deps = dependencies.get(node_name, [])

    missing_deps = []
    for dep in required_deps:
        if not state.get(dep):
            missing_deps.append(dep)
            logger.warning(f"Node {node_name} missing required dependency: {dep}")

    return len(missing_deps) == 0


# Export node functions and utilities
__all__ = [
    "refiner_node",
    "critic_node",
    "historian_node",
    "synthesis_node",
    "NodeExecutionError",
    "circuit_breaker",
    "node_metrics",
    "handle_node_timeout",
    "get_node_dependencies",
    "validate_node_input",
    "create_agent_with_llm",
    "convert_state_to_context",
    "CircuitBreakerFunction",
]
