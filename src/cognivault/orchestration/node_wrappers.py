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
from typing import Dict, Any, Optional, List
from functools import wraps

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.orchestration.state_bridge import AgentContextStateBridge
from cognivault.orchestration.state_schemas import (
    CogniVaultState,
    RefinerOutput,
    CriticOutput,
    HistorianOutput,
    SynthesisOutput,
    record_agent_error,
)
from cognivault.observability import get_logger
from cognivault.config.openai_config import OpenAIConfig
from cognivault.llm.openai import OpenAIChatLLM

logger = get_logger(__name__)


class NodeExecutionError(Exception):
    """Raised when a node execution fails."""

    pass


def circuit_breaker(max_failures: int = 3, reset_timeout: float = 300.0):
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

    def decorator(func):
        # Store state in a dictionary to avoid attribute issues
        circuit_state = {
            "failure_count": 0,
            "last_failure_time": None,
            "circuit_open": False,
        }

        def _sync_state():
            """Sync internal state with exposed attributes"""
            wrapper._failure_count = circuit_state["failure_count"]
            wrapper._last_failure_time = circuit_state["last_failure_time"]
            wrapper._circuit_open = circuit_state["circuit_open"]

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Sync state from external attributes if they were manually set
            if hasattr(wrapper, "_failure_count"):
                circuit_state["failure_count"] = wrapper._failure_count
            if hasattr(wrapper, "_last_failure_time"):
                circuit_state["last_failure_time"] = wrapper._last_failure_time
            if hasattr(wrapper, "_circuit_open"):
                circuit_state["circuit_open"] = wrapper._circuit_open

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
                circuit_state["failure_count"] += 1
                circuit_state["last_failure_time"] = time.time()

                if circuit_state["failure_count"] >= max_failures:
                    circuit_state["circuit_open"] = True
                    logger.error(
                        f"Circuit breaker opened for {func.__name__} "
                        f"after {circuit_state['failure_count']} failures"
                    )

                _sync_state()
                raise

        # Initialize attributes on wrapper before returning
        wrapper._failure_count = 0
        wrapper._last_failure_time = None
        wrapper._circuit_open = False

        return wrapper

    return decorator


def node_metrics(func):
    """
    Decorator to add metrics collection to node functions.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        node_name = func.__name__.replace("_node", "")

        try:
            logger.info(f"Starting execution of {node_name} node")
            result = await func(*args, **kwargs)

            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Completed {node_name} node execution in {execution_time:.2f}ms"
            )

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Failed {node_name} node execution after {execution_time:.2f}ms: {e}"
            )
            raise

    return wrapper


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
        refiner_output: Optional[RefinerOutput] = state["refiner"]
        if refiner_output is not None:
            refined_question = refiner_output.get("refined_question", "")
            # Add with both 'refiner' and 'Refiner' keys for compatibility
            if refined_question:
                context.add_agent_output("refiner", refined_question)
                context.add_agent_output("Refiner", refined_question)
                context.execution_state["refiner_topics"] = refiner_output.get(
                    "topics", []
                )
                context.execution_state["refiner_confidence"] = refiner_output.get(
                    "confidence", 0.8
                )
                logger.info(
                    f"Added refiner output to context: {refined_question[:100]}..."
                )
            else:
                logger.warning(
                    "Refiner output found in state but refined_question is empty"
                )

    if state.get("critic"):
        critic_output: Optional[CriticOutput] = state["critic"]
        if critic_output is not None:
            critique = critic_output.get("critique", "")
            # Add with both 'critic' and 'Critic' keys for compatibility
            if critique:
                context.add_agent_output("critic", critique)
                context.add_agent_output("Critic", critique)
                context.execution_state["critic_suggestions"] = critic_output.get(
                    "suggestions", []
                )
                context.execution_state["critic_severity"] = critic_output.get(
                    "severity", "medium"
                )
                logger.info(f"Added critic output to context: {critique[:100]}...")
            else:
                logger.warning("Critic output found in state but critique is empty")

    if state.get("historian"):
        historian_output: Optional[HistorianOutput] = state["historian"]
        if historian_output is not None:
            historical_summary = historian_output.get("historical_summary", "")
            # Add with both 'historian' and 'Historian' keys for compatibility
            if historical_summary:
                context.add_agent_output("historian", historical_summary)
                context.add_agent_output("Historian", historical_summary)
                context.execution_state["historian_retrieved_notes"] = (
                    historian_output.get("retrieved_notes", [])
                )
                context.execution_state["historian_search_strategy"] = (
                    historian_output.get("search_strategy", "hybrid")
                )
                context.execution_state["historian_topics_found"] = (
                    historian_output.get("topics_found", [])
                )
                context.execution_state["historian_confidence"] = historian_output.get(
                    "confidence", 0.8
                )
                logger.info(
                    f"Added historian output to context: {historical_summary[:100]}..."
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
async def refiner_node(state: CogniVaultState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for RefinerAgent.

    Transforms the raw user query into a structured, clarified prompt
    and updates the state with typed RefinerOutput.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state

    Returns
    -------
    CogniVaultState
        Updated state with RefinerOutput
    """
    logger.info("Executing refiner node")

    try:
        # Create agent
        agent = await create_agent_with_llm("refiner")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract refiner output (using "Refiner" key, not "refiner")
        refiner_raw_output = result_context.agent_outputs.get("Refiner", "")

        # Create typed output
        refiner_output = RefinerOutput(
            refined_question=refiner_raw_output,
            topics=result_context.execution_state.get("topics", []),
            confidence=result_context.execution_state.get("confidence", 0.8),
            processing_notes=result_context.execution_state.get("processing_notes"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Return state update with agent output and success tracking
        logger.info("Refiner node completed successfully")
        return {
            "refiner": refiner_output,
            "successful_agents": ["refiner"],
        }

    except Exception as e:
        logger.error(f"Refiner node failed: {e}")
        error_state = record_agent_error(state, "refiner", e)
        raise NodeExecutionError(f"Refiner execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def critic_node(state: CogniVaultState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for CriticAgent.

    Provides analytical critique of the refined query and adds
    suggestions for improvement to the state.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (must contain refiner output)

    Returns
    -------
    CogniVaultState
        Updated state with CriticOutput
    """
    logger.info("Executing critic node")

    try:
        # Validate dependencies
        if not state.get("refiner"):
            raise NodeExecutionError("Critic node requires refiner output")

        # Create agent
        agent = await create_agent_with_llm("critic")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract critic output (using "Critic" key, not "critic")
        critic_raw_output = result_context.agent_outputs.get("Critic", "")

        # Create typed output
        critic_output = CriticOutput(
            critique=critic_raw_output,
            suggestions=result_context.execution_state.get("suggestions", []),
            severity=result_context.execution_state.get("severity", "medium"),
            strengths=result_context.execution_state.get("strengths", []),
            weaknesses=result_context.execution_state.get("weaknesses", []),
            confidence=result_context.execution_state.get("confidence", 0.7),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Return state update with agent output and success tracking
        logger.info("Critic node completed successfully")
        return {
            "critic": critic_output,
            "successful_agents": ["critic"],
        }

    except Exception as e:
        logger.error(f"Critic node failed: {e}")
        error_state = record_agent_error(state, "critic", e)
        raise NodeExecutionError(f"Critic execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def historian_node(state: CogniVaultState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for HistorianAgent.

    Retrieves and analyzes historical context using intelligent search
    and LLM-powered relevance analysis.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (must contain refiner output)

    Returns
    -------
    CogniVaultState
        Updated state with HistorianOutput
    """
    logger.info("Executing historian node")

    try:
        # Validate dependencies
        if not state.get("refiner"):
            raise NodeExecutionError("Historian node requires refiner output")

        # Create agent
        agent = await create_agent_with_llm("historian")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract historian output (using "Historian" key, not "historian")
        historian_raw_output = result_context.agent_outputs.get("Historian", "")

        # Extract retrieved notes from context
        retrieved_notes = getattr(result_context, "retrieved_notes", [])

        # Determine topics found from retrieved notes context
        topics_found = []
        if hasattr(result_context, "execution_state"):
            topics_found = result_context.execution_state.get("topics_found", [])

        # Create typed output
        historian_output = HistorianOutput(
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

        # Return state update with agent output and success tracking
        logger.info("Historian node completed successfully")
        return {
            "historian": historian_output,
            "successful_agents": ["historian"],
        }

    except Exception as e:
        logger.error(f"Historian node failed: {e}")
        error_state = record_agent_error(state, "historian", e)
        raise NodeExecutionError(f"Historian execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def synthesis_node(state: CogniVaultState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for SynthesisAgent.

    Generates final synthesis from multiple agent outputs,
    creating coherent and comprehensive analysis.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (must contain refiner, critic, and historian outputs)

    Returns
    -------
    CogniVaultState
        Updated state with SynthesisOutput
    """
    logger.info("Executing synthesis node")

    try:
        # Validate dependencies
        if not state.get("refiner"):
            raise NodeExecutionError("Synthesis node requires refiner output")
        if not state.get("critic"):
            raise NodeExecutionError("Synthesis node requires critic output")
        if not state.get("historian"):
            raise NodeExecutionError("Synthesis node requires historian output")

        # Create agent
        agent = await create_agent_with_llm("synthesis")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent with event emission and retry logic
        result_context = await agent.run_with_retry(context)

        # Extract synthesis output (using "Synthesis" key, not "synthesis")
        synthesis_raw_output = result_context.agent_outputs.get("Synthesis", "")

        # Determine sources used
        sources_used = []
        if state.get("refiner"):
            sources_used.append("refiner")
        if state.get("critic"):
            sources_used.append("critic")
        if state.get("historian"):
            sources_used.append("historian")

        # Create typed output
        synthesis_output = SynthesisOutput(
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

        # Return state update with agent output and success tracking
        logger.info("Synthesis node completed successfully")
        return {
            "synthesis": synthesis_output,
            "successful_agents": ["synthesis"],
        }

    except Exception as e:
        logger.error(f"Synthesis node failed: {e}")
        error_state = record_agent_error(state, "synthesis", e)
        raise NodeExecutionError(f"Synthesis execution failed: {e}") from e


async def handle_node_timeout(coro, timeout_seconds: float = 30.0):
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
]
