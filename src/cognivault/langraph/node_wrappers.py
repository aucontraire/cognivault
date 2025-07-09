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
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import wraps

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.langraph.state_bridge import AgentContextStateBridge
from cognivault.langraph.state_schemas import (
    CogniVaultState,
    RefinerOutput,
    CriticOutput,
    SynthesisOutput,
    set_agent_output,
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

            except Exception as e:
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

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state

    Returns
    -------
    AgentContext
        AgentContext for agent execution
    """
    bridge = AgentContextStateBridge()

    # Create AgentContext from query
    context = AgentContext(query=state["query"])

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

    # Add execution metadata
    context.execution_state.update(
        {
            "execution_id": state["execution_metadata"]["execution_id"],
            "orchestrator_type": "langgraph-real",
            "successful_agents": state["successful_agents"].copy(),
            "failed_agents": state["failed_agents"].copy(),
        }
    )

    return context


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def refiner_node(state: CogniVaultState) -> CogniVaultState:
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

        # Execute agent
        result_context = await agent.run(context)

        # Extract refiner output (using "Refiner" key, not "refiner")
        refiner_raw_output = result_context.agent_outputs.get("Refiner", "")

        # Create typed output
        refiner_output = RefinerOutput(
            refined_question=refiner_raw_output,
            topics=result_context.execution_state.get("topics", []),
            confidence=result_context.execution_state.get("confidence", 0.8),
            processing_notes=result_context.execution_state.get("processing_notes"),
            timestamp=datetime.utcnow().isoformat(),
        )

        # Update state with typed output
        new_state = set_agent_output(state, "refiner", refiner_output)

        logger.info("Refiner node completed successfully")
        return new_state

    except Exception as e:
        logger.error(f"Refiner node failed: {e}")
        error_state = record_agent_error(state, "refiner", e)
        raise NodeExecutionError(f"Refiner execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def critic_node(state: CogniVaultState) -> CogniVaultState:
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

        # Execute agent
        result_context = await agent.run(context)

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
            timestamp=datetime.utcnow().isoformat(),
        )

        # Update state with typed output
        new_state = set_agent_output(state, "critic", critic_output)

        logger.info("Critic node completed successfully")
        return new_state

    except Exception as e:
        logger.error(f"Critic node failed: {e}")
        error_state = record_agent_error(state, "critic", e)
        raise NodeExecutionError(f"Critic execution failed: {e}") from e


@circuit_breaker(max_failures=3, reset_timeout=300.0)
@node_metrics
async def synthesis_node(state: CogniVaultState) -> CogniVaultState:
    """
    LangGraph node wrapper for SynthesisAgent.

    Generates final synthesis from multiple agent outputs,
    creating coherent and comprehensive analysis.

    Parameters
    ----------
    state : CogniVaultState
        Current LangGraph state (must contain refiner and critic outputs)

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

        # Create agent
        agent = await create_agent_with_llm("synthesis")

        # Convert state to context
        context = await convert_state_to_context(state)

        # Execute agent
        result_context = await agent.run(context)

        # Extract synthesis output (using "Synthesis" key, not "synthesis")
        synthesis_raw_output = result_context.agent_outputs.get("Synthesis", "")

        # Determine sources used
        sources_used = []
        if state.get("refiner"):
            sources_used.append("refiner")
        if state.get("critic"):
            sources_used.append("critic")

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
            timestamp=datetime.utcnow().isoformat(),
        )

        # Update state with typed output
        new_state = set_agent_output(state, "synthesis", synthesis_output)

        logger.info("Synthesis node completed successfully")
        return new_state

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
        "synthesis": ["refiner", "critic"],  # Requires both refiner and critic
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
