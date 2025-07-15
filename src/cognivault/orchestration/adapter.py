"""
LangGraph Node Adapter for CogniVault agents.

This module provides the LangGraphNodeAdapter class which serves as a bridge
between CogniVault agents and LangGraph's execution model, enabling seamless
conversion of agents into LangGraph-compatible nodes.
"""

import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from abc import ABC

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent, LangGraphNodeDefinition
from cognivault.exceptions import StateTransitionError
from cognivault.observability import get_logger

logger = get_logger(__name__)


@dataclass
class NodeExecutionResult:
    """Result of a node execution with metadata."""

    context: AgentContext
    success: bool
    execution_time_ms: float
    node_id: str
    step_id: Optional[str] = None
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NodeConfiguration:
    """Configuration for node execution."""

    timeout_seconds: Optional[float] = None
    retry_enabled: bool = True
    step_id: Optional[str] = None
    custom_config: Optional[Dict[str, Any]] = None


class LangGraphNodeAdapter(ABC):
    """
    Abstract base adapter for converting CogniVault agents into LangGraph nodes.

    This adapter provides the interface and utilities needed to execute
    CogniVault agents within a LangGraph DAG context, handling state
    transitions, error propagation, and execution metadata.
    """

    def __init__(self, agent: BaseAgent, node_id: Optional[str] = None):
        """
        Initialize the node adapter.

        Parameters
        ----------
        agent : BaseAgent
            The CogniVault agent to adapt
        node_id : str, optional
            Custom node ID (defaults to agent name)
        """
        self.agent = agent
        self.node_id = node_id or agent.name.lower()
        self.logger = get_logger(f"{__name__}.{self.node_id}")

        # Execution statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time_ms = 0.0

    @property
    def node_definition(self) -> LangGraphNodeDefinition:
        """Get the LangGraph node definition for this adapter."""
        return self.agent.get_node_definition()

    @property
    def average_execution_time_ms(self) -> float:
        """Get average execution time in milliseconds."""
        if self.total_executions == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_executions

    @property
    def success_rate(self) -> float:
        """Get success rate as a percentage (0-100)."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    async def __call__(
        self, state: AgentContext, config: Optional[NodeConfiguration] = None
    ) -> AgentContext:
        """
        Execute the node with the given state and configuration.

        This is the main LangGraph-compatible interface.

        Parameters
        ----------
        state : AgentContext
            Current graph state
        config : NodeConfiguration, optional
            Node execution configuration

        Returns
        -------
        AgentContext
            Updated state after node execution
        """
        result = await self.execute(state, config)
        if not result.success and result.error:
            raise result.error
        return result.context

    async def execute(
        self, state: AgentContext, config: Optional[NodeConfiguration] = None
    ) -> NodeExecutionResult:
        """
        Execute the adapted agent and return detailed results.

        Parameters
        ----------
        state : AgentContext
            Current execution state
        config : NodeConfiguration, optional
            Execution configuration

        Returns
        -------
        NodeExecutionResult
            Detailed execution result with metadata
        """
        config = config or NodeConfiguration()
        start_time = time.time()

        self.logger.debug(f"Executing node {self.node_id}")
        self.total_executions += 1

        # Prepare execution context
        execution_context = await self._prepare_execution_context(state, config)

        try:
            # Validate input state (for StandardNodeAdapter, check enable_state_validation flag)
            should_validate = getattr(self, "enable_state_validation", True)
            if should_validate and not self.agent.validate_node_compatibility(
                execution_context
            ):
                raise StateTransitionError(
                    transition_type="validation_failed",
                    state_details="Input state incompatible with node requirements",
                    step_id=config.step_id,
                    agent_id=self.node_id,
                )

            # Execute the agent
            result_context = await self._execute_agent(execution_context, config)

            # Post-process results
            final_context = await self._post_process_results(result_context, config)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            self.total_execution_time_ms += execution_time_ms
            self.successful_executions += 1

            self.logger.debug(
                f"Node {self.node_id} completed successfully in {execution_time_ms:.2f}ms"
            )

            return NodeExecutionResult(
                context=final_context,
                success=True,
                execution_time_ms=execution_time_ms,
                node_id=self.node_id,
                step_id=config.step_id,
                metadata=self._get_execution_metadata(final_context, config),
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.total_execution_time_ms += execution_time_ms
            self.failed_executions += 1

            self.logger.error(
                f"Node {self.node_id} failed after {execution_time_ms:.2f}ms: {str(e)}"
            )

            # Update state with failure information
            state.complete_agent_execution(self.node_id, success=False)
            state.add_execution_edge(
                from_agent=self.node_id,
                to_agent="ERROR",
                edge_type="error",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

            return NodeExecutionResult(
                context=state,
                success=False,
                execution_time_ms=execution_time_ms,
                node_id=self.node_id,
                step_id=config.step_id,
                error=e,
                metadata=self._get_execution_metadata(state, config),
            )

    async def _prepare_execution_context(
        self, state: AgentContext, config: NodeConfiguration
    ) -> AgentContext:
        """
        Prepare the execution context for the agent.

        Subclasses can override this to customize context preparation.
        """
        # Create a snapshot for rollback capability
        snapshot_id = state.create_execution_snapshot(f"before_{self.node_id}")

        # Add node execution metadata
        state.set_path_metadata(f"{self.node_id}_snapshot_id", str(snapshot_id))
        state.set_path_metadata(f"{self.node_id}_config", config.custom_config or {})

        return state

    async def _execute_agent(
        self, context: AgentContext, config: NodeConfiguration
    ) -> AgentContext:
        """Execute the underlying agent."""
        # Build agent config from node configuration
        agent_config: Dict[str, Any] = {}
        if config.step_id:
            agent_config["step_id"] = config.step_id
        if config.timeout_seconds:
            agent_config["timeout_seconds"] = config.timeout_seconds
        if config.custom_config:
            agent_config.update(config.custom_config)

        return await self.agent.invoke(context, agent_config)

    async def _post_process_results(
        self, context: AgentContext, config: NodeConfiguration
    ) -> AgentContext:
        """
        Post-process execution results.

        Subclasses can override this to add custom post-processing.
        """
        # Add execution edge for successful completion
        context.add_execution_edge(
            from_agent=self.node_id,
            to_agent="NEXT",
            edge_type="normal",
            metadata={
                "success": True,
                "execution_time_ms": self.average_execution_time_ms,
            },
        )

        return context

    def _get_execution_metadata(
        self, context: AgentContext, config: NodeConfiguration
    ) -> Dict[str, Any]:
        """Get execution metadata for the result."""
        return {
            "node_id": self.node_id,
            "agent_name": self.agent.name,
            "total_executions": self.total_executions,
            "success_rate": self.success_rate,
            "average_execution_time_ms": self.average_execution_time_ms,
            "context_size_bytes": context.current_size,
            "config": {
                "timeout_seconds": config.timeout_seconds,
                "retry_enabled": config.retry_enabled,
                "step_id": config.step_id,
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics for this node."""
        return {
            "node_id": self.node_id,
            "agent_name": self.agent.name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
            "total_execution_time_ms": self.total_execution_time_ms,
            "average_execution_time_ms": self.average_execution_time_ms,
            "agent_stats": self.agent.get_execution_stats(),
            "node_definition": self.node_definition.to_dict(),
        }


class StandardNodeAdapter(LangGraphNodeAdapter):
    """
    Standard implementation of LangGraphNodeAdapter for typical agent conversions.

    This provides a complete, ready-to-use adapter for most CogniVault agents
    without requiring custom implementations.
    """

    def __init__(
        self,
        agent: BaseAgent,
        node_id: Optional[str] = None,
        enable_state_validation: bool = True,
        enable_rollback: bool = True,
    ):
        """
        Initialize the standard node adapter.

        Parameters
        ----------
        agent : BaseAgent
            The agent to adapt
        node_id : str, optional
            Custom node ID
        enable_state_validation : bool
            Whether to validate state transitions
        enable_rollback : bool
            Whether to enable automatic rollback on failure
        """
        super().__init__(agent, node_id)
        self.enable_state_validation = enable_state_validation
        self.enable_rollback = enable_rollback

    async def _prepare_execution_context(
        self, state: AgentContext, config: NodeConfiguration
    ) -> AgentContext:
        """Enhanced context preparation with validation and rollback setup."""
        context = await super()._prepare_execution_context(state, config)

        # Note: State validation is handled in the main execute method
        # to ensure consistent error handling and result formatting
        return context

    async def _post_process_results(
        self, context: AgentContext, config: NodeConfiguration
    ) -> AgentContext:
        """Enhanced post-processing with state validation."""
        result_context = await super()._post_process_results(context, config)

        # Validate output state if enabled
        if self.enable_state_validation:
            # Check that the agent produced valid output
            agent_output = result_context.get_output(self.agent.name)
            if agent_output is None:
                self.logger.warning(f"Agent {self.node_id} produced no output")

        return result_context


class ConditionalNodeAdapter(LangGraphNodeAdapter):
    """
    Node adapter with conditional routing capabilities.

    This adapter can determine the next node(s) to execute based on
    the agent's output and custom routing logic.
    """

    def __init__(
        self,
        agent: BaseAgent,
        routing_function: Callable[[AgentContext], List[str]],
        node_id: Optional[str] = None,
    ):
        """
        Initialize the conditional node adapter.

        Parameters
        ----------
        agent : BaseAgent
            The agent to adapt
        routing_function : Callable[[AgentContext], List[str]]
            Function that returns list of next node IDs based on context
        node_id : str, optional
            Custom node ID
        """
        super().__init__(agent, node_id)
        self.routing_function = routing_function

    async def _post_process_results(
        self, context: AgentContext, config: NodeConfiguration
    ) -> AgentContext:
        """Post-processing with conditional routing."""
        result_context = await super()._post_process_results(context, config)

        # Determine next nodes using routing function
        try:
            next_nodes = self.routing_function(result_context)

            # Record conditional routing decision
            result_context.record_conditional_routing(
                decision_point=self.node_id,
                condition="routing_function_evaluation",
                chosen_path=",".join(next_nodes),
                alternative_paths=[],
                metadata={
                    "routing_function": str(self.routing_function),
                    "agent_output_length": len(
                        str(result_context.get_output(self.agent.name) or "")
                    ),
                },
            )

            # Add execution edges to next nodes
            for next_node in next_nodes:
                result_context.add_execution_edge(
                    from_agent=self.node_id,
                    to_agent=next_node,
                    edge_type="conditional",
                    metadata={"routing_decision": True},
                )

        except Exception as e:
            self.logger.error(f"Routing function failed for {self.node_id}: {e}")
            # Fallback to standard routing
            result_context.add_execution_edge(
                from_agent=self.node_id,
                to_agent="ERROR",
                edge_type="error",
                metadata={"routing_error": str(e)},
            )

        return result_context


def create_node_adapter(
    agent: BaseAgent, adapter_type: str = "standard", **kwargs
) -> LangGraphNodeAdapter:
    """
    Factory function to create appropriate node adapter for an agent.

    Parameters
    ----------
    agent : BaseAgent
        The agent to adapt
    adapter_type : str
        Type of adapter ("standard", "conditional")
    **kwargs
        Additional arguments for adapter initialization

    Returns
    -------
    LangGraphNodeAdapter
        Configured node adapter
    """
    if adapter_type == "standard":
        return StandardNodeAdapter(agent, **kwargs)
    elif adapter_type == "conditional":
        if "routing_function" not in kwargs:
            raise ValueError(
                "ConditionalNodeAdapter requires 'routing_function' parameter"
            )
        return ConditionalNodeAdapter(agent, **kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
