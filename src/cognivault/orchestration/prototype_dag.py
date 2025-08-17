"""
Prototype DAG implementation for CogniVault LangGraph integration.

This module demonstrates the conversion of CogniVault's sequential agent
execution model into a LangGraph-compatible DAG structure, starting with
a simple Refiner → Critic flow.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from cognivault.context import AgentContext
from cognivault.agents.registry import get_agent_registry
from cognivault.config.openai_config import OpenAIConfig
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.observability import get_logger

from .adapter import (
    LangGraphNodeAdapter,
    StandardNodeAdapter,
    ConditionalNodeAdapter,
    ExecutionNodeConfiguration,
)
from .graph_builder import GraphBuilder, GraphEdge, EdgeType, GraphDefinition

logger = get_logger(__name__)


@dataclass
class DAGExecutionResult:
    """Result of DAG execution with comprehensive metadata."""

    final_context: AgentContext
    success: bool
    total_execution_time_ms: float
    nodes_executed: List[str]
    edges_traversed: List[Tuple[str, str]]
    errors: List[Exception]
    execution_path: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class PrototypeDAGExecutor:
    """
    Prototype DAG executor demonstrating LangGraph integration.

    This executor implements a simplified version of LangGraph's execution
    model, focusing on the Refiner → Critic flow as a proof of concept
    for the full CogniVault → LangGraph transition.
    """

    def __init__(
        self,
        enable_parallel_execution: bool = False,
        max_execution_time_seconds: float = 300.0,
    ) -> None:
        """
        Initialize the prototype DAG executor.

        Parameters
        ----------
        enable_parallel_execution : bool
            Whether to enable parallel execution of independent nodes
        max_execution_time_seconds : float
            Maximum total execution time for the DAG
        """
        self.enable_parallel_execution = enable_parallel_execution
        self.max_execution_time_seconds = max_execution_time_seconds
        self.logger = get_logger(f"{__name__}.PrototypeDAGExecutor")

        # Execution statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

    async def execute_refiner_critic_dag(
        self, query: str, config: Optional[Dict[str, Any]] = None
    ) -> DAGExecutionResult:
        """
        Execute the prototype Refiner → Critic DAG.

        This method demonstrates the core LangGraph execution pattern
        with CogniVault agents converted to LangGraph nodes.

        Parameters
        ----------
        query : str
            Input query for the DAG
        config : Dict[str, Any], optional
            Execution configuration

        Returns
        -------
        DAGExecutionResult
            Comprehensive execution result
        """
        start_time = time.time()
        config = config or {}

        self.logger.info(
            f"Starting Refiner → Critic DAG execution for query: {query[:100]}..."
        )
        self.total_executions += 1

        try:
            # Initialize components
            context = AgentContext(query=query)
            registry = get_agent_registry()
            llm = self._initialize_llm()

            # Create agents
            refiner_agent = registry.create_agent("refiner", llm=llm)
            critic_agent = registry.create_agent("critic", llm=llm)

            # Create node adapters
            refiner_node = StandardNodeAdapter(
                agent=refiner_agent,
                node_id="refiner",
                enable_state_validation=config.get("enable_validation", True),
            )

            # Create conditional routing for critic based on refiner success
            def route_after_refiner(context: AgentContext) -> List[str]:
                """Route to critic if refiner succeeded, end if failed."""
                refiner_output = context.get_output("Refiner")
                if refiner_output and len(str(refiner_output).strip()) > 0:
                    return ["critic"]
                else:
                    return ["end"]

            critic_node = ConditionalNodeAdapter(
                agent=critic_agent,
                routing_function=lambda ctx: ["end"],  # Always end after critic
                node_id="critic",
            )

            # Build graph definition
            graph_def = self._build_refiner_critic_graph(refiner_node, critic_node)

            # Execute the DAG
            execution_result = await self._execute_dag(
                graph_def=graph_def,
                initial_context=context,
                node_adapters={"refiner": refiner_node, "critic": critic_node},
                config=config,
            )

            # Calculate total execution time
            total_time_ms = (time.time() - start_time) * 1000

            # Determine overall success
            success = (
                len(execution_result.errors) == 0
                and len(execution_result.nodes_executed)
                >= 1  # At least refiner should execute
            )

            if success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1

            # Create comprehensive result
            result = DAGExecutionResult(
                final_context=execution_result.final_context,
                success=success,
                total_execution_time_ms=total_time_ms,
                nodes_executed=execution_result.nodes_executed,
                edges_traversed=execution_result.edges_traversed,
                errors=execution_result.errors,
                execution_path=execution_result.execution_path,
                performance_metrics=self._calculate_performance_metrics(
                    execution_result, total_time_ms
                ),
            )

            self.logger.info(
                f"DAG execution completed: success={success}, "
                f"time={total_time_ms:.2f}ms, nodes={len(execution_result.nodes_executed)}"
            )

            return result

        except Exception as e:
            self.failed_executions += 1
            total_time_ms = (time.time() - start_time) * 1000

            self.logger.error(f"DAG execution failed after {total_time_ms:.2f}ms: {e}")

            return DAGExecutionResult(
                final_context=AgentContext(query=query),
                success=False,
                total_execution_time_ms=total_time_ms,
                nodes_executed=[],
                edges_traversed=[],
                errors=[e],
                execution_path=[],
                performance_metrics={
                    "error": str(e),
                    "execution_time_ms": total_time_ms,
                },
            )

    def _build_refiner_critic_graph(
        self, refiner_node: StandardNodeAdapter, critic_node: ConditionalNodeAdapter
    ) -> GraphDefinition:
        """Build the graph definition for Refiner → Critic flow."""
        builder = GraphBuilder()

        # Add agents (the builder will get node definitions from them)
        builder.add_agent(refiner_node.agent)
        builder.add_agent(critic_node.agent)

        # Add explicit edge from refiner to critic
        refiner_to_critic_edge = GraphEdge(
            from_node="refiner",
            to_node="critic",
            edge_type=EdgeType.CONDITIONAL,
            condition_name="refiner_success",
            metadata={
                "description": "Execute critic if refiner succeeds",
                "condition": "refiner_output_exists",
            },
        )
        builder.add_edge(refiner_to_critic_edge)

        # Add conditional routing for refiner
        def route_from_refiner(context: AgentContext) -> str:
            """Route to critic if refiner succeeded."""
            refiner_output = context.get_output("Refiner")
            if refiner_output and len(str(refiner_output).strip()) > 0:
                return "critic"
            return "END"

        builder.add_conditional_routing(
            from_node="refiner",
            routing_func=route_from_refiner,
            condition_name="refiner_success_routing",
        )

        return builder.build()

    async def _execute_dag(
        self,
        graph_def: GraphDefinition,
        initial_context: AgentContext,
        node_adapters: Dict[str, LangGraphNodeAdapter],
        config: Dict[str, Any],
    ) -> DAGExecutionResult:
        """Execute the DAG with the given configuration."""
        current_context = initial_context
        nodes_executed = []
        edges_traversed = []
        errors = []
        execution_path = []

        # Start from entry points
        current_nodes = graph_def.entry_points.copy()
        visited_nodes = set()
        max_iterations = len(graph_def.nodes) * 2  # Prevent infinite loops
        iteration = 0

        while current_nodes and iteration < max_iterations:
            iteration += 1
            next_nodes = []

            for node_id in current_nodes:
                if node_id in visited_nodes:
                    continue

                if node_id not in node_adapters:
                    self.logger.warning(f"No adapter found for node: {node_id}")
                    continue

                self.logger.debug(f"Executing node: {node_id}")

                try:
                    # Execute the node
                    node_adapter = node_adapters[node_id]

                    node_config = ExecutionNodeConfiguration(
                        timeout_seconds=config.get("node_timeout_seconds"),
                        retry_enabled=config.get("retry_enabled", True),
                        step_id=f"{node_id}_{iteration}",
                        custom_config=config.get("node_configs", {}).get(node_id, {}),
                    )

                    execution_start = time.time()
                    result = await node_adapter.execute(current_context, node_config)
                    execution_time_ms = (time.time() - execution_start) * 1000

                    if result.success:
                        current_context = result.context
                        nodes_executed.append(node_id)
                        visited_nodes.add(node_id)

                        # Record execution in path
                        execution_path.append(
                            {
                                "node_id": node_id,
                                "success": True,
                                "execution_time_ms": execution_time_ms,
                                "iteration": iteration,
                                "output_length": len(
                                    str(
                                        current_context.get_output(
                                            node_adapter.agent.name
                                        )
                                        or ""
                                    )
                                ),
                            }
                        )

                        # Determine next nodes based on graph edges and routing
                        node_next_nodes = self._get_next_nodes(
                            node_id, current_context, graph_def, node_adapters
                        )
                        next_nodes.extend(node_next_nodes)

                        # Record edges traversed
                        for next_node in node_next_nodes:
                            edges_traversed.append((node_id, next_node))

                        self.logger.debug(
                            f"Node {node_id} completed successfully, next nodes: {node_next_nodes}"
                        )
                    else:
                        # Node execution failed
                        if result.error:
                            errors.append(result.error)

                        execution_path.append(
                            {
                                "node_id": node_id,
                                "success": False,
                                "execution_time_ms": execution_time_ms,
                                "iteration": iteration,
                                "error": (
                                    str(result.error)
                                    if result.error
                                    else "Unknown error"
                                ),
                            }
                        )

                        self.logger.error(
                            f"Node {node_id} execution failed: {result.error}"
                        )

                        # Handle failure based on configuration
                        if config.get("fail_fast", True):
                            break

                except Exception as e:
                    errors.append(e)
                    execution_path.append(
                        {
                            "node_id": node_id,
                            "success": False,
                            "iteration": iteration,
                            "error": str(e),
                        }
                    )

                    self.logger.error(f"Unexpected error executing node {node_id}: {e}")

                    if config.get("fail_fast", True):
                        break

            # Update current nodes for next iteration
            current_nodes = list(set(next_nodes))  # Remove duplicates

            # Check for completion conditions
            if not current_nodes:
                self.logger.debug("No more nodes to execute - DAG complete")
                break

            # Prevent runaway execution
            if iteration >= max_iterations:
                self.logger.warning(f"Maximum iterations ({max_iterations}) reached")
                break

        return DAGExecutionResult(
            final_context=current_context,
            success=len(errors) == 0,
            total_execution_time_ms=0,  # Will be calculated by caller
            nodes_executed=nodes_executed,
            edges_traversed=edges_traversed,
            errors=errors,
            execution_path=execution_path,
            performance_metrics={},  # Will be calculated by caller
        )

    def _get_next_nodes(
        self,
        current_node: str,
        context: AgentContext,
        graph_def: GraphDefinition,
        node_adapters: Dict[str, LangGraphNodeAdapter],
    ) -> List[str]:
        """Determine next nodes to execute after current node."""
        next_nodes = []

        # Check graph edges
        for edge in graph_def.edges:
            if edge.from_node == current_node:
                if edge.edge_type == EdgeType.SEQUENTIAL:
                    next_nodes.append(edge.to_node)
                elif edge.edge_type == EdgeType.CONDITIONAL:
                    # For conditional edges, we need to evaluate the condition
                    # For now, we'll add all conditional targets and let routing decide
                    next_nodes.append(edge.to_node)

        # Check if current node has conditional routing
        if current_node in node_adapters:
            adapter = node_adapters[current_node]
            if hasattr(adapter, "routing_function"):
                try:
                    routed_nodes = adapter.routing_function(context)
                    next_nodes.extend(routed_nodes)
                except Exception as e:
                    self.logger.error(
                        f"Routing function failed for {current_node}: {e}"
                    )

        # Filter out invalid nodes and special nodes
        valid_next_nodes = []
        for node in next_nodes:
            if node in ["END", "end", "ERROR", "error"]:
                continue  # Skip terminal nodes
            if node in graph_def.nodes:
                valid_next_nodes.append(node)

        return list(set(valid_next_nodes))  # Remove duplicates

    def _initialize_llm(self) -> OpenAIChatLLM:
        """Initialize LLM for agent creation."""
        llm_config = OpenAIConfig.load()
        return OpenAIChatLLM(
            api_key=llm_config.api_key,
            model=llm_config.model,
            base_url=llm_config.base_url,
        )

    def _calculate_performance_metrics(
        self, execution_result: DAGExecutionResult, total_time_ms: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        nodes_count = len(execution_result.nodes_executed)
        edges_count = len(execution_result.edges_traversed)

        # Calculate per-node timing from execution path
        node_timings = {}
        total_node_time = 0
        for step in execution_result.execution_path:
            if step.get("success", False):
                node_id = step["node_id"]
                exec_time = step.get("execution_time_ms", 0)
                node_timings[node_id] = exec_time
                total_node_time += exec_time

        return {
            "total_execution_time_ms": total_time_ms,
            "total_node_execution_time_ms": total_node_time,
            "overhead_time_ms": total_time_ms - total_node_time,
            "nodes_executed": nodes_count,
            "edges_traversed": edges_count,
            "errors_count": len(execution_result.errors),
            "success_rate": 1.0 if len(execution_result.errors) == 0 else 0.0,
            "average_node_time_ms": (
                total_node_time / nodes_count if nodes_count > 0 else 0
            ),
            "node_timings": node_timings,
            "execution_efficiency": (
                total_node_time / total_time_ms if total_time_ms > 0 else 0
            ),
        }

    def get_executor_statistics(self) -> Dict[str, Any]:
        """Get comprehensive executor statistics."""
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0
            else 0
        )

        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "configuration": {
                "enable_parallel_execution": self.enable_parallel_execution,
                "max_execution_time_seconds": self.max_execution_time_seconds,
            },
        }


async def run_prototype_demo(
    query: str = "Analyze the benefits of renewable energy",
) -> DAGExecutionResult:
    """
    Run a demonstration of the prototype DAG execution.

    This function provides a simple interface to test the Refiner → Critic
    DAG execution with a sample query.

    Parameters
    ----------
    query : str
        Query to process through the DAG

    Returns
    -------
    DAGExecutionResult
        Complete execution result
    """
    executor = PrototypeDAGExecutor(
        enable_parallel_execution=False,  # Start with sequential for simplicity
        max_execution_time_seconds=120.0,
    )

    config = {
        "enable_validation": True,
        "fail_fast": True,
        "retry_enabled": True,
        "node_timeout_seconds": 30.0,
    }

    result = await executor.execute_refiner_critic_dag(query, config)

    # Log summary
    logger.info("Demo execution completed:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Execution time: {result.total_execution_time_ms:.2f}ms")
    logger.info(f"  Nodes executed: {result.nodes_executed}")
    logger.info(f"  Edges traversed: {result.edges_traversed}")
    logger.info(f"  Errors: {len(result.errors)}")

    return result


if __name__ == "__main__":
    # Run the demo if this file is executed directly
    import asyncio

    async def main() -> None:
        result = await run_prototype_demo()
        print(f"Demo result: {result.success}")
        print(f"Performance: {result.performance_metrics}")

    asyncio.run(main())
