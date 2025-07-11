import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from cognivault.config.logging_config import setup_logging
from cognivault.config.app_config import get_config
from cognivault.config.openai_config import OpenAIConfig
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.context import AgentContext
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMInterface
from cognivault.exceptions import (
    FailurePropagationStrategy,
    ExecutionPath,
    GracefulDegradationWarning,
    PipelineExecutionError,
)
from cognivault.observability import (
    get_logger,
    observability_context,
)
from cognivault.diagnostics.metrics import get_metrics_collector

setup_logging()
logger = get_logger(__name__)


# DEPRECATED: Target Removal v1.1.0
# This class will be removed after the 2-3 week safety period.
# Use RealLangGraphOrchestrator (--execution-mode=langgraph-real) instead.
class AgentOrchestrator:
    """
    DEPRECATED: Coordinates and runs a set of AI agents to process a user query.

    WARNING: This orchestrator is deprecated and will be removed in v1.1.0.
    Use RealLangGraphOrchestrator with --execution-mode=langgraph-real instead.

    This orchestrator initializes the appropriate agents based on the specified configuration,
    then executes them sequentially to handle agent dependencies. The results are merged into a shared context.

    Parameters
    ----------
    critic_enabled : bool, optional
        Whether to include the CriticAgent in the execution pipeline. Default is True.
    agents_to_run : list of str, optional
        A list of agent names to run explicitly. If None, a default pipeline is used.

    Attributes
    ----------
    agents : list of BaseAgent
        The list of initialized agents that will be run.
    critic_enabled : bool
        Flag indicating whether the CriticAgent is enabled.
    agents_to_run : list of str or None
        Custom list of agents to run, or None if using the default pipeline.
    """

    def __init__(
        self,
        critic_enabled: Optional[bool] = None,
        agents_to_run: Optional[list[str]] = None,
    ):
        """
        DEPRECATED: Initialize the AgentOrchestrator with optional critic and custom agent list.

        WARNING: This method is deprecated and will be removed in v1.1.0.
        Use RealLangGraphOrchestrator instead.

        Parameters
        ----------
        critic_enabled : bool, optional
            Whether to enable the CriticAgent in the pipeline. If None, uses configuration default.
        agents_to_run : list of str, optional
            Explicit list of agent names to run. If None, the default pipeline is used.
        """
        # Get application configuration
        config = get_config()

        # Use configuration defaults if not explicitly provided
        self.critic_enabled = (
            critic_enabled
            if critic_enabled is not None
            else config.execution.critic_enabled
        )
        self.agents_to_run = (
            [a.lower() for a in agents_to_run] if agents_to_run else None
        )
        logger.debug(
            f"Initializing AgentOrchestrator with agents_to_run={self.agents_to_run} and critic_enabled={self.critic_enabled}"
        )
        self.agents: list[BaseAgent] = []
        self.registry = get_agent_registry()

        # Execution state tracking for conditional execution
        self.execution_state: Dict[str, Any] = {
            "completed_agents": [],
            "failed_agents": [],
            "skipped_agents": [],
            "degraded_functionality": [],
            "execution_path": ExecutionPath.NORMAL,
        }

        # Initialize LLM for agents that require it
        llm_config = OpenAIConfig.load()
        llm: LLMInterface = OpenAIChatLLM(
            api_key=llm_config.api_key,
            model=llm_config.model,
            base_url=llm_config.base_url,
        )

        if self.agents_to_run:
            logger.debug("Custom agent list specified.")
            # Validate and resolve dependencies for custom agent list
            try:
                resolved_order = self.registry.resolve_dependencies(self.agents_to_run)
                logger.debug(f"Resolved agent execution order: {resolved_order}")

                for agent_name in resolved_order:
                    try:
                        agent = self.registry.create_agent(agent_name, llm=llm)
                        self.agents.append(agent)
                        logger.debug(f"Added agent: {agent_name}")
                    except ValueError as e:
                        logger.warning(f"Failed to create agent '{agent_name}': {e}")
                        print(f"[DEBUG] Unknown agent name: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to resolve agent dependencies: {e}")
                # Fallback to original order if dependency resolution fails
                for agent_name in self.agents_to_run:
                    try:
                        agent = self.registry.create_agent(agent_name, llm=llm)
                        self.agents.append(agent)
                        logger.debug(
                            f"Added agent (no dependency resolution): {agent_name}"
                        )
                    except ValueError as e:
                        logger.warning(f"Failed to create agent '{agent_name}': {e}")
        else:
            # Use default pipeline from configuration
            default_agents = config.execution.default_agents.copy()
            if self.critic_enabled and "critic" not in default_agents:
                # Insert critic after refiner if it exists, otherwise add to end
                if "refiner" in default_agents:
                    refiner_index = default_agents.index("refiner")
                    default_agents.insert(refiner_index + 1, "critic")
                else:
                    default_agents.append("critic")
            elif not self.critic_enabled and "critic" in default_agents:
                default_agents.remove("critic")

            # Resolve dependencies for default pipeline
            try:
                resolved_order = self.registry.resolve_dependencies(default_agents)
                logger.debug(f"Resolved default agent order: {resolved_order}")
                default_agents = resolved_order
            except Exception as e:
                logger.warning(
                    f"Failed to resolve dependencies for default pipeline: {e}"
                )

            for agent_name in default_agents:
                try:
                    agent = self.registry.create_agent(agent_name, llm=llm)
                    self.agents.append(agent)
                except ValueError as e:
                    logger.error(f"Failed to create core agent '{agent_name}': {e}")
                    raise

            logger.debug(
                f"Default agent order: {[agent.__class__.__name__ for agent in self.agents]}"
            )

    # DEPRECATED: Target Removal v1.1.0
    async def run(self, query: str) -> AgentContext:
        """
        DEPRECATED: Run all agents with conditional execution and graceful degradation.

        WARNING: This method is deprecated and will be removed in v1.1.0.
        Use RealLangGraphOrchestrator.run() instead.

        Implements LangGraph-compatible failure propagation strategies:
        - FAIL_FAST: Stop immediately on critical agent failure
        - WARN_CONTINUE: Log warning but continue execution
        - CONDITIONAL_FALLBACK: Try alternative agents or paths
        - GRACEFUL_DEGRADATION: Skip non-critical agents

        Parameters
        ----------
        query : str
            The user query to be processed by the agents.

        Returns
        -------
        AgentContext
            The updated agent context after all agents have completed execution.
        """
        # Generate pipeline ID and set up observability context
        pipeline_id = str(uuid.uuid4())
        pipeline_start_time = time.time()

        # Initialize metrics collector
        metrics = get_metrics_collector()

        with observability_context(
            pipeline_id=pipeline_id, execution_phase="pipeline_start"
        ):
            logger.log_pipeline_start(
                pipeline_id=pipeline_id,
                agents=[agent.name for agent in self.agents],
                query_length=len(query),
                agent_count=len(self.agents),
            )

            logger.info(f"[AgentOrchestrator] Running orchestrator with query: {query}")
            context = AgentContext(query=query)

            # Store pipeline ID in context for tracking
            context.set_path_metadata("pipeline_id", pipeline_id)

        # Reset execution state
        self.execution_state = {
            "completed_agents": [],
            "failed_agents": [],
            "skipped_agents": [],
            "degraded_functionality": [],
            "execution_path": ExecutionPath.NORMAL,
        }

        # Add execution metadata to context
        execution_path = self.execution_state["execution_path"]
        assert isinstance(execution_path, ExecutionPath)
        context.execution_state["orchestrator_metadata"] = {
            "total_agents": len(self.agents),
            "execution_path": execution_path.value,
            "conditional_execution": True,
        }

        # Initialize execution path tracing
        context.set_path_metadata(
            "pipeline_start", datetime.now(timezone.utc).isoformat()
        )

        previous_agent = None
        for agent in self.agents:
            if agent is None:
                continue

            agent_name = agent.name
            logger.info(f"[AgentOrchestrator] Processing agent: {agent_name}")

            # Add execution edge from previous agent
            if previous_agent:
                completed_agents = self.execution_state["completed_agents"]
                assert isinstance(completed_agents, list)
                context.add_execution_edge(
                    from_agent=previous_agent,
                    to_agent=agent_name,
                    edge_type="normal",
                    metadata={"execution_order": len(completed_agents)},
                )

            # Check if agent should be skipped due to dependencies
            if await self._should_skip_agent(agent_name, context):
                logger.info(
                    f"[AgentOrchestrator] Skipping agent {agent_name} due to dependency failure"
                )
                skipped_agents = self.execution_state["skipped_agents"]
                assert isinstance(skipped_agents, list)
                skipped_agents.append(agent_name)

                # Record conditional routing decision
                context.record_conditional_routing(
                    decision_point=f"dependency_check_{agent_name}",
                    condition="dependency_failed",
                    chosen_path="skip_agent",
                    alternative_paths=["execute_agent"],
                    metadata={"reason": "dependency_failure"},
                )
                continue

            # Perform health checks (convert to lowercase to match registry keys)
            registry_key = agent_name.lower()
            if not self.registry.check_health(registry_key):
                logger.warning(
                    f"[AgentOrchestrator] Health check failed for agent {agent_name}"
                )

                # Record health check failure routing
                context.record_conditional_routing(
                    decision_point=f"health_check_{agent_name}",
                    condition="health_check_failed",
                    chosen_path="handle_failure",
                    alternative_paths=["execute_agent"],
                    metadata={"reason": "health_check_failure"},
                )

                await self._handle_agent_failure(
                    agent_name, "Health check failed", context
                )
                continue

            # Mark agent as starting execution
            context.start_agent_execution(agent_name)

            # Run agent with conditional execution
            agent_start_time = time.time()

            with observability_context(
                pipeline_id=pipeline_id,
                agent_name=agent_name,
                execution_phase="agent_execution",
            ):
                try:
                    logger.log_agent_start(
                        agent_name, step_id=f"exec_{len(context.agent_outputs)}"
                    )
                    logger.info(f"[AgentOrchestrator] Running agent: {agent_name}")

                    await agent.run_with_retry(context)

                    # Calculate execution time
                    agent_duration_ms = (time.time() - agent_start_time) * 1000

                    # Record successful agent execution metrics
                    metrics.record_agent_execution(
                        agent_name=agent_name,
                        success=True,
                        duration_ms=agent_duration_ms,
                        tokens_used=0,  # Will be updated by LLM calls
                    )

                    # Agent succeeded
                    completed_agents = self.execution_state["completed_agents"]
                    assert isinstance(completed_agents, list)
                    completed_agents.append(agent_name)
                    context.complete_agent_execution(agent_name, success=True)

                    logger.log_agent_end(
                        agent_name,
                        success=True,
                        duration_ms=agent_duration_ms,
                        output_length=len(context.get_output(agent_name) or ""),
                    )
                    logger.info(f"[AgentOrchestrator] Completed agent: {agent_name}")

                    # Handle synthesis agent special case
                    if agent_name == "Synthesis":
                        logger.debug(f"Setting final_synthesis from {agent_name}")
                        output = context.get_output(agent_name)
                        if isinstance(output, str):
                            context.set_final_synthesis(output)

                    previous_agent = agent_name

                except Exception as e:
                    # Calculate execution time for failed attempt
                    agent_duration_ms = (time.time() - agent_start_time) * 1000

                    # Record failed agent execution metrics
                    metrics.record_agent_execution(
                        agent_name=agent_name,
                        success=False,
                        duration_ms=agent_duration_ms,
                        error_type=type(e).__name__,
                    )

                    logger.log_agent_end(
                        agent_name,
                        success=False,
                        duration_ms=agent_duration_ms,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    logger.error(f"[AgentOrchestrator] Agent {agent_name} failed: {e}")
                    context.complete_agent_execution(agent_name, success=False)

                    # Record failure routing decision
                    context.record_conditional_routing(
                        decision_point=f"agent_failure_{agent_name}",
                        condition="agent_execution_failed",
                        chosen_path="handle_failure",
                        alternative_paths=["continue_execution", "fallback_agent"],
                        metadata={"error": str(e), "failure_type": "execution_error"},
                    )

                    await self._handle_agent_failure(
                        agent_name, str(e), context, cause=e
                    )

        # Set final execution path metadata
        context.set_path_metadata(
            "pipeline_end", datetime.now(timezone.utc).isoformat()
        )
        context.set_path_metadata("total_agents", len(self.agents))

        # Get execution state with type assertions
        completed_agents = self.execution_state["completed_agents"]
        failed_agents = self.execution_state["failed_agents"]
        skipped_agents = self.execution_state["skipped_agents"]
        degraded_functionality = self.execution_state["degraded_functionality"]
        execution_path = self.execution_state["execution_path"]

        assert isinstance(completed_agents, list)
        assert isinstance(failed_agents, list)
        assert isinstance(skipped_agents, list)
        assert isinstance(degraded_functionality, list)
        assert isinstance(execution_path, ExecutionPath)

        context.set_path_metadata(
            "execution_summary",
            {
                "completed_count": len(completed_agents),
                "failed_count": len(failed_agents),
                "skipped_count": len(skipped_agents),
                "degraded_count": len(degraded_functionality),
                "success_rate": (
                    len(completed_agents) / len(self.agents) if self.agents else 0
                ),
                "final_execution_path": execution_path.value,
            },
        )

        # Update final execution metadata
        context.execution_state["orchestrator_metadata"].update(
            {
                "completed_agents": completed_agents,
                "failed_agents": failed_agents,
                "skipped_agents": skipped_agents,
                "degraded_functionality": degraded_functionality,
                "execution_path": execution_path.value,
                "success_rate": (
                    len(completed_agents) / len(self.agents) if self.agents else 0
                ),
            }
        )

        # Calculate total pipeline execution time and record metrics
        pipeline_duration_ms = (time.time() - pipeline_start_time) * 1000
        pipeline_success = len(failed_agents) == 0 or len(completed_agents) > 0

        # Record pipeline metrics
        with observability_context(
            pipeline_id=pipeline_id, execution_phase="pipeline_end"
        ):
            metrics.record_pipeline_execution(
                pipeline_id=pipeline_id,
                success=pipeline_success,
                duration_ms=pipeline_duration_ms,
                agents_executed=completed_agents,
                total_tokens=0,  # Will be aggregated from agent metrics
            )

            logger.log_pipeline_end(
                pipeline_id=pipeline_id,
                success=pipeline_success,
                duration_ms=pipeline_duration_ms,
                completed_agents=len(completed_agents),
                failed_agents=len(failed_agents),
                skipped_agents=len(skipped_agents),
                success_rate=(
                    len(completed_agents) / len(self.agents) if self.agents else 0
                ),
            )

        logger.info(
            f"[AgentOrchestrator] Pipeline completed: "
            f"{len(completed_agents)}/{len(self.agents)} agents succeeded"
        )

        return context

    # DEPRECATED: Target Removal v1.1.0
    async def _should_skip_agent(self, agent_name: str, context: AgentContext) -> bool:
        """
        DEPRECATED: Check if an agent should be skipped due to dependency failures.

        WARNING: This method is deprecated and will be removed in v1.1.0.

        Parameters
        ----------
        agent_name : str
            Name of the agent to check
        context : AgentContext
            Current execution context

        Returns
        -------
        bool
            True if agent should be skipped
        """
        registry_key = agent_name.lower()
        if registry_key not in self.registry._agents:
            return False

        metadata = self.registry._agents[registry_key]

        # Check if any dependencies failed
        dependencies = metadata.dependencies or []
        for dependency in dependencies:
            failed_agents = self.execution_state["failed_agents"]
            assert isinstance(failed_agents, list)
            if dependency in failed_agents:
                logger.warning(
                    f"Agent {agent_name} dependency '{dependency}' failed, checking failure strategy"
                )

                # Check if we can continue without this dependency
                dep_strategy = self.registry.get_failure_strategy(dependency.lower())
                if dep_strategy == FailurePropagationStrategy.FAIL_FAST:
                    return True  # Skip dependent agent
                elif dep_strategy == FailurePropagationStrategy.GRACEFUL_DEGRADATION:
                    # Continue but mark as degraded
                    degraded_functionality = self.execution_state[
                        "degraded_functionality"
                    ]
                    assert isinstance(degraded_functionality, list)
                    if dependency not in degraded_functionality:
                        degraded_functionality.append(dependency)
                    return False  # Don't skip, but execution is degraded

        return False

    # DEPRECATED: Target Removal v1.1.0
    async def _handle_agent_failure(
        self,
        agent_name: str,
        error_message: str,
        context: AgentContext,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        DEPRECATED: Handle agent failure according to its failure propagation strategy.

        WARNING: This method is deprecated and will be removed in v1.1.0.

        Parameters
        ----------
        agent_name : str
            Name of the failed agent
        error_message : str
            Error message from the failure
        context : AgentContext
            Current execution context
        cause : Optional[Exception]
            Original exception that caused the failure
        """
        failed_agents = self.execution_state["failed_agents"]
        assert isinstance(failed_agents, list)
        failed_agents.append(agent_name)

        registry_key = agent_name.lower()
        failure_strategy = self.registry.get_failure_strategy(registry_key)
        is_critical = self.registry.is_critical_agent(registry_key)

        logger.info(
            f"Handling failure for {agent_name}: strategy={failure_strategy.value}, critical={is_critical}"
        )

        if failure_strategy == FailurePropagationStrategy.FAIL_FAST:
            # Critical failure - stop execution
            logger.error(
                f"FAIL_FAST: Critical agent {agent_name} failed, stopping pipeline"
            )
            completed_agents = self.execution_state["completed_agents"]
            assert isinstance(completed_agents, list)
            raise PipelineExecutionError(
                failed_agents=[agent_name],
                successful_agents=completed_agents,
                pipeline_stage="agent_execution",
                failure_reason=error_message,
                cause=cause,
            )

        elif failure_strategy == FailurePropagationStrategy.WARN_CONTINUE:
            # Log warning but continue
            logger.warning(
                f"WARN_CONTINUE: Agent {agent_name} failed, continuing pipeline: {error_message}"
            )
            # Don't raise exception, let execution continue

        elif failure_strategy == FailurePropagationStrategy.GRACEFUL_DEGRADATION:
            # Skip non-critical functionality
            logger.warning(
                f"GRACEFUL_DEGRADATION: Skipping non-critical agent {agent_name}"
            )
            degraded_functionality = self.execution_state["degraded_functionality"]
            assert isinstance(degraded_functionality, list)
            degraded_functionality.append(agent_name)
            self.execution_state["execution_path"] = ExecutionPath.DEGRADED

            # Create warning for degraded functionality
            degradation_warning = GracefulDegradationWarning(
                degraded_functionality=f"{agent_name} functionality unavailable",
                skipped_agents=[agent_name],
                continuing_agents=[
                    a
                    for a in [agent.name for agent in self.agents]
                    if a not in self.execution_state["failed_agents"]
                ],
                degradation_reason=error_message,
            )
            logger.warning(degradation_warning.get_user_message())

        elif failure_strategy == FailurePropagationStrategy.CONDITIONAL_FALLBACK:
            # Try fallback agents or alternative paths
            logger.info(f"CONDITIONAL_FALLBACK: Trying fallback for {agent_name}")
            fallback_agents = self.registry.get_fallback_agents(registry_key)

            if fallback_agents:
                logger.info(
                    f"Found fallback agents for {agent_name}: {fallback_agents}"
                )
                self.execution_state["execution_path"] = ExecutionPath.FALLBACK

                # Record fallback routing decision
                context.record_conditional_routing(
                    decision_point=f"fallback_{agent_name}",
                    condition="agent_failed_with_fallback",
                    chosen_path="execute_fallback",
                    alternative_paths=["fail_pipeline"],
                    metadata={
                        "failed_agent": agent_name,
                        "fallback_agents": fallback_agents,
                        "reason": error_message,
                    },
                )

                # Add fallback execution edges
                for fallback_agent in fallback_agents:
                    context.add_execution_edge(
                        from_agent=agent_name,
                        to_agent=fallback_agent,
                        edge_type="fallback",
                        condition="failure_fallback",
                        metadata={
                            "original_agent": agent_name,
                            "fallback_reason": error_message,
                        },
                    )

                # For now, just log - fallback execution would be implemented here
            else:
                logger.warning(f"No fallback agents configured for {agent_name}")

                # Record no fallback routing decision
                context.record_conditional_routing(
                    decision_point=f"no_fallback_{agent_name}",
                    condition="agent_failed_no_fallback",
                    chosen_path="fail_pipeline" if is_critical else "continue_degraded",
                    alternative_paths=["execute_fallback"],
                    metadata={
                        "failed_agent": agent_name,
                        "is_critical": is_critical,
                        "reason": error_message,
                    },
                )

                if is_critical:
                    completed_agents = self.execution_state["completed_agents"]
                    assert isinstance(completed_agents, list)
                    raise PipelineExecutionError(
                        failed_agents=[agent_name],
                        successful_agents=completed_agents,
                        pipeline_stage="fallback_execution",
                        failure_reason=f"No fallback available for critical agent: {error_message}",
                        cause=cause,
                    )
