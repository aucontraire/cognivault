"""
LangGraph-based orchestrator for CogniVault agents.

This module provides a LangGraph-compatible orchestrator that converts CogniVault's
sequential agent execution into DAG-based orchestration using the LangGraph framework.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.config.openai_config import OpenAIConfig
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.observability import get_logger
from cognivault.exceptions import PipelineExecutionError

from .adapter import create_node_adapter, StandardNodeAdapter, ConditionalNodeAdapter
from .graph_builder import GraphBuilder, GraphEdge, EdgeType
from .prototype_dag import PrototypeDAGExecutor, DAGExecutionResult

logger = get_logger(__name__)


@dataclass
class LangGraphExecutionResult:
    """Result of LangGraph-based execution with performance metrics."""

    final_context: AgentContext
    success: bool
    execution_time_ms: float
    nodes_executed: List[str]
    edges_traversed: List[tuple]
    performance_metrics: Dict[str, Any]
    dag_result: Optional[DAGExecutionResult] = None


class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator for CogniVault agents.

    This orchestrator converts CogniVault agents into LangGraph-compatible DAG execution,
    providing improved parallelization, dependency management, and conditional routing.
    """

    def __init__(self, agents_to_run: Optional[List[str]] = None):
        """
        Initialize the LangGraph orchestrator.

        Parameters
        ----------
        agents_to_run : List[str], optional
            List of agent names to run. If None, runs all default agents.
        """
        self.agents_to_run = agents_to_run or [
            "refiner",
            "historian",
            "critic",
            "synthesis",
        ]
        self.registry = get_agent_registry()
        self.logger = get_logger(f"{__name__}.LangGraphOrchestrator")

        # Add agents property for compatibility with health checks and dry runs
        self.agents: List[BaseAgent] = []  # Will be populated when agents are created

        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

    async def run(
        self, query: str, config: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """
        Execute agents using LangGraph DAG orchestration.

        Parameters
        ----------
        query : str
            The query to process
        config : Dict[str, Any], optional
            Execution configuration options

        Returns
        -------
        AgentContext
            The final context after DAG execution
        """
        config = config or {}
        start_time = time.time()

        self.logger.info(f"Starting LangGraph execution for query: {query[:100]}...")
        self.total_executions += 1

        try:
            # Initialize context and LLM
            context = AgentContext(query=query)
            llm = self._initialize_llm()

            # Create agents
            agents = self._create_agents(llm)

            # Execute using LangGraph DAG
            execution_result = await self._execute_langgraph_dag(
                context, agents, config
            )

            # Calculate performance metrics
            total_time_ms = (time.time() - start_time) * 1000

            if execution_result.success:
                self.successful_executions += 1
                self.logger.info(
                    f"LangGraph execution completed successfully in {total_time_ms:.2f}ms"
                )
            else:
                self.failed_executions += 1
                self.logger.error(
                    f"LangGraph execution failed after {total_time_ms:.2f}ms"
                )

            return execution_result.final_context

        except Exception as e:
            self.failed_executions += 1
            total_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"LangGraph orchestration failed after {total_time_ms:.2f}ms: {e}"
            )

            # Return context with error information
            context = AgentContext(query=query)
            context.execution_state["langgraph_error"] = str(e)
            context.execution_state["execution_time_ms"] = total_time_ms
            return context

    def _create_agents(self, llm) -> Dict[str, BaseAgent]:
        """Create agents for the specified agent list."""
        agents = {}
        self.agents = []  # Reset agents list

        for agent_name in self.agents_to_run:
            try:
                agent = self.registry.create_agent(agent_name.lower(), llm=llm)
                agents[agent_name.lower()] = agent
                self.agents.append(agent)  # Add to agents list for compatibility
                self.logger.debug(f"Created agent: {agent_name}")
            except Exception as e:
                self.logger.error(f"Failed to create agent {agent_name}: {e}")
                # Continue with other agents

        return agents

    async def _execute_langgraph_dag(
        self,
        context: AgentContext,
        agents: Dict[str, BaseAgent],
        config: Dict[str, Any],
    ) -> LangGraphExecutionResult:
        """Execute agents using LangGraph DAG orchestration."""

        # For now, use the prototype DAG executor
        # In the future, this would be replaced with full LangGraph integration
        executor = PrototypeDAGExecutor(
            enable_parallel_execution=config.get("enable_parallel_execution", False),
            max_execution_time_seconds=config.get("max_execution_time_seconds", 300.0),
        )

        # Execute based on agent selection
        if set(self.agents_to_run) == {"refiner", "critic"}:
            # Use the prototype refiner->critic DAG
            dag_result = await executor.execute_refiner_critic_dag(
                context.query, config
            )
        else:
            # For other combinations, fall back to prototype DAG with available agents
            dag_result = await self._execute_general_dag(
                executor, context, agents, config
            )

        return LangGraphExecutionResult(
            final_context=dag_result.final_context,
            success=dag_result.success,
            execution_time_ms=dag_result.total_execution_time_ms,
            nodes_executed=dag_result.nodes_executed,
            edges_traversed=dag_result.edges_traversed,
            performance_metrics=dag_result.performance_metrics,
            dag_result=dag_result,
        )

    async def _execute_general_dag(
        self,
        executor: PrototypeDAGExecutor,
        context: AgentContext,
        agents: Dict[str, BaseAgent],
        config: Dict[str, Any],
    ) -> DAGExecutionResult:
        """Execute a general DAG with available agents."""

        # Create a simple sequential DAG for now
        # In the future, this would use the graph builder to create optimal DAGs

        execution_time_start = time.time()
        nodes_executed = []
        edges_traversed = []
        errors = []

        try:
            # Execute agents sequentially for now
            # TODO: Replace with actual DAG execution using graph builder
            current_context = context

            for agent_name in self.agents_to_run:
                if agent_name in agents:
                    agent = agents[agent_name]

                    try:
                        self.logger.debug(f"Executing agent: {agent_name}")
                        current_context = await agent.run(current_context)
                        nodes_executed.append(agent_name)

                        # Add execution edge
                        if len(nodes_executed) > 1:
                            edges_traversed.append((nodes_executed[-2], agent_name))

                    except Exception as e:
                        self.logger.error(f"Agent {agent_name} failed: {e}")
                        errors.append(e)

                        # For now, continue with other agents
                        # TODO: Implement proper failure handling strategies
                        continue

            total_time_ms = (time.time() - execution_time_start) * 1000

            return DAGExecutionResult(
                final_context=current_context,
                success=len(errors) == 0,
                total_execution_time_ms=total_time_ms,
                nodes_executed=nodes_executed,
                edges_traversed=edges_traversed,
                errors=errors,
                execution_path=[],  # TODO: Add execution path tracking
                performance_metrics={
                    "total_execution_time_ms": total_time_ms,
                    "nodes_executed": len(nodes_executed),
                    "edges_traversed": len(edges_traversed),
                    "errors_count": len(errors),
                    "success_rate": 1.0 if len(errors) == 0 else 0.0,
                },
            )

        except Exception as e:
            total_time_ms = (time.time() - execution_time_start) * 1000
            self.logger.error(f"General DAG execution failed: {e}")

            return DAGExecutionResult(
                final_context=context,
                success=False,
                total_execution_time_ms=total_time_ms,
                nodes_executed=nodes_executed,
                edges_traversed=edges_traversed,
                errors=[e],
                execution_path=[],
                performance_metrics={
                    "error": str(e),
                    "execution_time_ms": total_time_ms,
                },
            )

    def _initialize_llm(self):
        """Initialize LLM for agent creation."""
        llm_config = OpenAIConfig.load()
        return OpenAIChatLLM(
            api_key=llm_config.api_key,
            model=llm_config.model,
            base_url=llm_config.base_url,
        )

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get orchestrator execution statistics."""
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0
            else 0
        )

        return {
            "orchestrator_type": "langgraph",
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "agents_to_run": self.agents_to_run,
        }
