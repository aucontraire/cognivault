"""
Real LangGraph orchestrator for CogniVault agents.

This module provides the actual LangGraph integration using the real LangGraph library,
implementing production-ready DAG execution with StateGraph orchestration.

Features:
- True DAG-based execution using LangGraph StateGraph
- Parallel execution where dependencies allow
- Type-safe state management with TypedDict schemas
- Circuit breaker patterns for error handling
- Comprehensive logging and metrics
- State bridge integration for AgentContext compatibility
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Union

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.observability import get_logger
from cognivault.langraph.state_bridge import AgentContextStateBridge
from cognivault.langraph.state_schemas import (
    CogniVaultState,
    RefinerOutput,
    CriticOutput,
    HistorianOutput,
    SynthesisOutput,
    create_initial_state,
    validate_state_integrity,
)
from cognivault.langraph.node_wrappers import (
    refiner_node,
    critic_node,
    historian_node,
    synthesis_node,
    NodeExecutionError,
    get_node_dependencies,
)
from cognivault.langraph.memory_manager import (
    CogniVaultMemoryManager,
    CheckpointConfig,
    create_memory_manager,
)

logger = get_logger(__name__)


class RealLangGraphOrchestrator:
    """
    Real LangGraph orchestrator for CogniVault agents.

    This orchestrator uses the actual LangGraph library to provide production-ready
    DAG-based execution with advanced state management, parallel processing, and
    conditional routing capabilities.

    Features:
    - StateGraph-based DAG execution with proper dependencies
    - Parallel execution of independent agents (Refiner → [Critic, Historian] → Synthesis)
    - Type-safe state management with comprehensive validation
    - Circuit breaker patterns for robust error handling
    - Optional memory checkpointing for stateful conversations
    - Comprehensive logging and performance metrics
    """

    def __init__(
        self,
        agents_to_run: Optional[List[str]] = None,
        enable_checkpoints: bool = False,
        thread_id: Optional[str] = None,
        memory_manager: Optional[CogniVaultMemoryManager] = None,
    ):
        """
        Initialize the real LangGraph orchestrator.

        Parameters
        ----------
        agents_to_run : List[str], optional
            List of agent names to run. For Phase 2.1, defaults to refiner, critic, historian, synthesis.
        enable_checkpoints : bool, optional
            Whether to enable memory checkpointing for stateful conversations.
        thread_id : str, optional
            Thread ID for conversation scoping (auto-generated if not provided).
        memory_manager : CogniVaultMemoryManager, optional
            Custom memory manager instance. If None, one will be created.
        """
        # For Phase 2.1, we include all four agents with historian
        self.agents_to_run = agents_to_run or [
            "refiner",
            "critic",
            "historian",
            "synthesis",
        ]
        self.enable_checkpoints = enable_checkpoints
        self.thread_id = thread_id
        self.registry = get_agent_registry()
        self.logger = get_logger(f"{__name__}.RealLangGraphOrchestrator")

        # Initialize memory manager
        if memory_manager:
            self.memory_manager = memory_manager
        else:
            self.memory_manager = create_memory_manager(
                enable_checkpoints=enable_checkpoints,
                thread_id=thread_id,
            )

        # Add agents property for compatibility with health checks and dry runs
        self.agents: List[BaseAgent] = []  # Will be populated when agents are created

        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        # State bridge for AgentContext <-> LangGraph state conversion
        self.state_bridge = AgentContextStateBridge()

        # LangGraph components (initialized lazily)
        self._graph = None
        self._compiled_graph = None

        self.logger.info(
            f"Initialized RealLangGraphOrchestrator with agents: {self.agents_to_run}, "
            f"checkpoints: {self.enable_checkpoints}, thread_id: {self.thread_id}"
        )

    async def run(
        self, query: str, config: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """
        Execute agents using real LangGraph StateGraph orchestration.

        This method implements true DAG-based execution with:
        - Refiner → [Critic, Historian] → Synthesis pipeline
        - Parallel execution of Critic and Historian after Refiner
        - Type-safe state management
        - Comprehensive error handling and recovery

        Parameters
        ----------
        query : str
            The query to process
        config : Dict[str, Any], optional
            Execution configuration options

        Returns
        -------
        AgentContext
            The final context after LangGraph execution

        Raises
        ------
        NodeExecutionError
            If LangGraph execution fails
        """
        config = config or {}
        start_time = time.time()
        execution_id = str(uuid.uuid4())

        self.logger.info(
            f"Starting real LangGraph execution for query: {query[:100]}... "
            f"(execution_id: {execution_id})"
        )
        self.logger.info(f"Execution mode: langgraph-real")
        self.logger.info(f"Agents to run: {self.agents_to_run}")
        self.logger.info(f"Config: {config}")

        self.total_executions += 1

        try:
            # Get or generate thread ID for this execution
            thread_id = self.memory_manager.get_thread_id(config.get("thread_id"))

            # Create initial LangGraph state
            initial_state = create_initial_state(query, execution_id)

            # Validate initial state
            if not validate_state_integrity(initial_state):
                raise NodeExecutionError("Initial state validation failed")

            # Create initial checkpoint if enabled
            if self.memory_manager.is_enabled():
                self.memory_manager.create_checkpoint(
                    thread_id=thread_id,
                    state=initial_state,
                    agent_step="initialization",
                    metadata={"execution_id": execution_id, "query": query},
                )

            # Build and compile StateGraph if not already done
            compiled_graph = await self._get_compiled_graph()

            # Execute the StateGraph
            self.logger.info(
                f"Executing LangGraph StateGraph with thread_id: {thread_id}"
            )

            # Prepare invocation config with thread ID
            invocation_config = {"configurable": {"thread_id": thread_id}}

            # Run the StateGraph
            final_state = await compiled_graph.ainvoke(
                initial_state, config=invocation_config
            )

            # Validate final state
            if not validate_state_integrity(final_state):
                self.logger.warning("Final state validation failed, but proceeding")

            # Create final checkpoint if enabled
            if self.memory_manager.is_enabled():
                self.memory_manager.create_checkpoint(
                    thread_id=thread_id,
                    state=final_state,
                    agent_step="completion",
                    metadata={
                        "execution_id": execution_id,
                        "query": query,
                        "successful_agents": final_state["successful_agents"],
                        "failed_agents": final_state["failed_agents"],
                        "completion_status": "success",
                    },
                )

            # Convert LangGraph state back to AgentContext
            context = await self._convert_state_to_context(final_state)

            # Add execution metadata
            total_time_ms = (time.time() - start_time) * 1000
            context.execution_state.update(
                {
                    "orchestrator_type": "langgraph-real",
                    "phase": "phase2_1",
                    "execution_id": execution_id,
                    "thread_id": thread_id,
                    "agents_requested": self.agents_to_run,
                    "config": config,
                    "execution_time_ms": total_time_ms,
                    "langgraph_execution": True,
                    "checkpoints_enabled": self.memory_manager.is_enabled(),
                    "successful_agents_count": len(final_state["successful_agents"]),
                    "failed_agents_count": len(final_state["failed_agents"]),
                    "errors_count": len(final_state["errors"]),
                }
            )

            # Update statistics
            if final_state["failed_agents"]:
                self.failed_executions += 1
                self.logger.warning(
                    f"LangGraph execution completed with failures: {final_state['failed_agents']}"
                )
            else:
                self.successful_executions += 1

            self.logger.info(
                f"Real LangGraph execution completed in {total_time_ms:.2f}ms "
                f"(successful: {len(final_state['successful_agents'])}, "
                f"failed: {len(final_state['failed_agents'])})"
            )

            return context

        except Exception as e:
            self.failed_executions += 1
            total_time_ms = (time.time() - start_time) * 1000

            self.logger.error(
                f"LangGraph execution failed after {total_time_ms:.2f}ms: {e}"
            )

            # Create fallback context with error information
            context = AgentContext(query=query)
            context.execution_state.update(
                {
                    "orchestrator_type": "langgraph-real",
                    "phase": "phase2_1",
                    "execution_id": execution_id,
                    "agents_requested": self.agents_to_run,
                    "config": config,
                    "execution_time_ms": total_time_ms,
                    "langgraph_execution": True,
                    "execution_error": str(e),
                    "execution_error_type": type(e).__name__,
                }
            )

            # Add error output
            context.add_agent_output(
                "langgraph_error",
                f"LangGraph execution failed: {e}\n"
                f"Execution ID: {execution_id}\n"
                f"Requested agents: {', '.join(self.agents_to_run)}\n"
                f"This indicates an issue with the DAG execution pipeline.",
            )

            raise NodeExecutionError(f"LangGraph execution failed: {e}") from e

    async def _get_compiled_graph(self):
        """
        Get or create compiled LangGraph StateGraph.

        Returns
        -------
        CompiledGraph
            Compiled LangGraph StateGraph ready for execution
        """
        if self._compiled_graph is None:
            self.logger.info("Building LangGraph StateGraph...")

            # Create StateGraph with CogniVaultState schema
            graph = StateGraph(CogniVaultState)

            # Add nodes for Phase 2.1 pipeline
            graph.add_node("refiner", refiner_node)
            graph.add_node("critic", critic_node)
            graph.add_node("historian", historian_node)
            graph.add_node("synthesis", synthesis_node)

            # Define the DAG structure: START → refiner → [critic, historian] → synthesis → END
            graph.set_entry_point("refiner")
            graph.add_edge("refiner", "critic")
            graph.add_edge("refiner", "historian")
            graph.add_edge("critic", "synthesis")
            graph.add_edge("historian", "synthesis")
            graph.add_edge("synthesis", END)

            # Set up checkpointer if enabled
            if self.memory_manager.is_enabled():
                checkpointer = self.memory_manager.get_memory_saver()
                if checkpointer:
                    self._compiled_graph = graph.compile(checkpointer=checkpointer)
                    self.logger.info(
                        "StateGraph compiled with memory checkpointing enabled"
                    )
                else:
                    self._compiled_graph = graph.compile()
                    self.logger.warning(
                        "Checkpointing enabled but no MemorySaver available, compiling without checkpointing"
                    )
            else:
                self._compiled_graph = graph.compile()
                self.logger.info("StateGraph compiled without checkpointing")

            self._graph = graph

        return self._compiled_graph

    async def _convert_state_to_context(
        self, final_state: CogniVaultState
    ) -> AgentContext:
        """
        Convert final LangGraph state back to AgentContext.

        Parameters
        ----------
        final_state : CogniVaultState
            Final state from LangGraph execution

        Returns
        -------
        AgentContext
            AgentContext with all agent outputs
        """
        # Create AgentContext
        context = AgentContext(query=final_state["query"])

        # Add agent outputs
        if final_state.get("refiner"):
            refiner_output: Optional[RefinerOutput] = final_state["refiner"]
            if refiner_output is not None:
                context.add_agent_output("refiner", refiner_output["refined_question"])
                context.execution_state["refiner_topics"] = refiner_output["topics"]
                context.execution_state["refiner_confidence"] = refiner_output[
                    "confidence"
                ]

        if final_state.get("critic"):
            critic_output: Optional[CriticOutput] = final_state["critic"]
            if critic_output is not None:
                context.add_agent_output("critic", critic_output["critique"])
                context.execution_state["critic_suggestions"] = critic_output[
                    "suggestions"
                ]
                context.execution_state["critic_severity"] = critic_output["severity"]

        if final_state.get("historian"):
            historian_output: Optional[HistorianOutput] = final_state["historian"]
            if historian_output is not None:
                context.add_agent_output(
                    "historian", historian_output["historical_summary"]
                )
                context.execution_state["historian_retrieved_notes"] = historian_output[
                    "retrieved_notes"
                ]
                context.execution_state["historian_search_strategy"] = historian_output[
                    "search_strategy"
                ]
                context.execution_state["historian_topics_found"] = historian_output[
                    "topics_found"
                ]
                context.execution_state["historian_confidence"] = historian_output[
                    "confidence"
                ]

        if final_state.get("synthesis"):
            synthesis_output: Optional[SynthesisOutput] = final_state["synthesis"]
            if synthesis_output is not None:
                context.add_agent_output(
                    "synthesis", synthesis_output["final_analysis"]
                )
                context.execution_state["synthesis_insights"] = synthesis_output[
                    "key_insights"
                ]
                context.execution_state["synthesis_themes"] = synthesis_output[
                    "themes_identified"
                ]

        # Track successful and failed agents
        for agent in final_state["successful_agents"]:
            context.successful_agents.add(agent)

        # Add error information if any
        if final_state["errors"]:
            context.execution_state["langgraph_errors"] = final_state["errors"]

        return context

    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get orchestrator execution statistics.

        Returns
        -------
        Dict[str, Any]
            Execution statistics
        """
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0
            else 0
        )

        return {
            "orchestrator_type": "langgraph-real",
            "implementation_status": "phase2_1_production",
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "agents_to_run": self.agents_to_run,
            "state_bridge_available": True,
            "checkpoints_enabled": self.enable_checkpoints,
            "dag_structure": "refiner → [critic, historian] → synthesis",
        }

    def get_dag_structure(self) -> Dict[str, Any]:
        """
        Get information about the DAG structure.

        Returns
        -------
        Dict[str, Any]
            DAG structure information
        """
        dependencies = get_node_dependencies()

        return {
            "nodes": self.agents_to_run,
            "dependencies": dependencies,
            "execution_order": ["refiner", "critic", "historian", "synthesis"],
            "parallel_capable": [
                "critic",
                "historian",
            ],  # Can run in parallel after refiner
            "entry_point": "refiner",
            "terminal_nodes": ["synthesis"],
        }

    async def rollback_to_checkpoint(
        self, thread_id: Optional[str] = None, checkpoint_id: Optional[str] = None
    ) -> Optional[AgentContext]:
        """
        Rollback to a specific checkpoint and return the restored context.

        Parameters
        ----------
        thread_id : str, optional
            Thread ID for conversation. If None, uses current thread_id.
        checkpoint_id : str, optional
            Specific checkpoint ID. If None, uses latest checkpoint.

        Returns
        -------
        AgentContext, optional
            Restored context from checkpoint, or None if not found
        """
        if not self.memory_manager.is_enabled():
            self.logger.warning("Rollback requested but checkpointing is disabled")
            return None

        target_thread_id = thread_id or self.thread_id
        if not target_thread_id:
            self.logger.error("No thread ID available for rollback")
            return None

        # Attempt rollback through memory manager
        restored_state = self.memory_manager.rollback_to_checkpoint(
            thread_id=target_thread_id, checkpoint_id=checkpoint_id
        )

        if restored_state:
            # Convert restored state back to AgentContext
            context = await self._convert_state_to_context(restored_state)
            context.execution_state["rollback_performed"] = True
            context.execution_state["rollback_thread_id"] = target_thread_id
            context.execution_state["rollback_checkpoint_id"] = checkpoint_id

            self.logger.info(
                f"Successfully rolled back to checkpoint for thread {target_thread_id}"
            )
            return context
        else:
            self.logger.warning(
                f"Rollback failed - no checkpoint found for thread {target_thread_id}"
            )
            return None

    def get_checkpoint_history(
        self, thread_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get checkpoint history for a thread.

        Parameters
        ----------
        thread_id : str, optional
            Thread ID to get history for. If None, uses current thread_id.

        Returns
        -------
        List[Dict[str, Any]]
            List of checkpoint information dictionaries
        """
        target_thread_id = thread_id or self.thread_id
        if not target_thread_id:
            return []

        checkpoints = self.memory_manager.get_checkpoint_history(target_thread_id)
        return [
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "agent_step": checkpoint.agent_step,
                "state_size_bytes": checkpoint.state_size_bytes,
                "success": checkpoint.success,
                "metadata": checkpoint.metadata,
            }
            for checkpoint in checkpoints
        ]

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory and checkpoint statistics.

        Returns
        -------
        Dict[str, Any]
            Memory usage and checkpoint statistics
        """
        memory_stats = self.memory_manager.get_memory_stats()

        # Add orchestrator-specific stats
        orchestrator_stats = {
            "orchestrator_type": "langgraph-real",
            "checkpointing_enabled": self.memory_manager.is_enabled(),
            "current_thread_id": self.thread_id,
            "execution_statistics": self.get_execution_statistics(),
        }

        return {**memory_stats, **orchestrator_stats}

    def cleanup_expired_checkpoints(self) -> int:
        """
        Clean up expired checkpoints.

        Returns
        -------
        int
            Number of checkpoints removed
        """
        return self.memory_manager.cleanup_expired_checkpoints()

    # Phase 2.0 Implementation Complete ✅
    # ✅ Add real LangGraph dependency to requirements.txt (done in Phase 1)
    # ✅ Import real LangGraph StateGraph and related classes
    # ✅ Convert agents to LangGraph StateGraph nodes (node_wrappers.py)
    # ✅ Implement actual StateGraph execution with typed state management
    # ✅ Add comprehensive error handling with circuit breakers
    # ✅ Performance tracking and execution statistics

    # Phase 2.1 Next Steps:
    # 🔜 Add Historian agent back into pipeline
    # 🔜 Implement conditional routing and parallel execution optimization
    # 🔜 Enhanced CLI integration with visualization tools
    # 🔜 Performance optimization and benchmarking vs legacy mode
