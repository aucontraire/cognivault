"""
Advanced orchestrator that integrates all dependency management features.

This module provides an enhanced orchestrator that combines the dependency graph engine,
execution planner, failure manager, resource scheduler, and dynamic composition
for sophisticated agent execution management.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.agents.registry import get_agent_registry
from cognivault.config.openai_config import OpenAIConfig
from cognivault.llm.openai import OpenAIChatLLM
from cognivault.llm.llm_interface import LLMInterface
from cognivault.observability import get_logger, observability_context
from cognivault.diagnostics.metrics import get_metrics_collector

from .graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
    DependencyType,
    ExecutionPriority,
    ResourceConstraint,
)
from .execution_planner import ExecutionPlanner, ExecutionStrategy, ExecutionPlan
from .failure_manager import (
    FailureManager,
    CascadePreventionStrategy,
    RetryConfiguration,
    RetryStrategy,
)
from .resource_scheduler import ResourceScheduler, ResourceType
from .dynamic_composition import (
    DynamicAgentComposer,
    RegistryDiscoverer,
    create_version_upgrade_rule,
    create_failure_recovery_rule,
)

logger = get_logger(__name__)


class ExecutionPhase(Enum):
    """Enumeration of execution phases in the orchestration pipeline."""

    PREPARATION = "preparation"
    RESOURCE_ALLOCATION = "resource_allocation"
    EXECUTION = "execution"
    CLEANUP = "cleanup"


class OrchestratorConfig(BaseModel):
    """
    Configuration for the advanced orchestrator.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    max_concurrent_agents: int = Field(
        4,
        description="Maximum number of agents that can execute concurrently",
        ge=1,
        le=100,
        json_schema_extra={"example": 4},
    )
    enable_failure_recovery: bool = Field(
        True,
        description="Enable failure recovery mechanisms",
        json_schema_extra={"example": True},
    )
    enable_resource_scheduling: bool = Field(
        True,
        description="Enable resource scheduling and allocation",
        json_schema_extra={"example": True},
    )
    enable_dynamic_composition: bool = Field(
        False,
        description="Enable dynamic agent composition",
        json_schema_extra={"example": False},
    )
    default_execution_strategy: ExecutionStrategy = Field(
        ExecutionStrategy.ADAPTIVE,
        description="Default strategy for execution planning",
        json_schema_extra={"example": "adaptive"},
    )
    cascade_prevention_strategy: CascadePreventionStrategy = Field(
        CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        description="Strategy for preventing failure cascades",
        json_schema_extra={"example": "graceful_degradation"},
    )
    pipeline_timeout_ms: int = Field(
        60000,
        description="Pipeline execution timeout in milliseconds",
        ge=1000,
        le=600000,  # 10 minutes max
        json_schema_extra={"example": 60000},
    )
    resource_allocation_timeout_ms: int = Field(
        10000,
        description="Resource allocation timeout in milliseconds",
        ge=100,
        le=60000,  # 1 minute max
        json_schema_extra={"example": 10000},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects, not string values
    )


class ResourceAllocationResult(BaseModel):
    """
    Result of resource allocation for an agent.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    agent_id: str = Field(
        ...,
        description="Unique identifier for the agent",
        min_length=1,
        max_length=200,
        json_schema_extra={"example": "critic_agent_001"},
    )
    resource_type: ResourceType = Field(
        ...,
        description="Type of resource that was allocated",
        json_schema_extra={"example": "memory"},
    )
    requested_amount: float = Field(
        ...,
        description="Amount of resource that was requested",
        ge=0.0,
        json_schema_extra={"example": 1.5},
    )
    allocated_amount: float = Field(
        ...,
        description="Amount of resource that was actually allocated",
        ge=0.0,
        json_schema_extra={"example": 1.2},
    )
    allocation_time_ms: float = Field(
        ...,
        description="Time taken to allocate resource in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 125.5},
    )
    success: bool = Field(
        ...,
        description="Whether the allocation was successful",
        json_schema_extra={"example": True},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects
    )


class PipelineStage(BaseModel):
    """
    Information about a stage in the execution pipeline.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    stage_id: str = Field(
        ...,
        description="Unique identifier for the pipeline stage",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "stage_001_execution"},
    )
    phase: ExecutionPhase = Field(
        ...,
        description="Execution phase for this stage",
        json_schema_extra={"example": "execution"},
    )
    agents_executed: List[str] = Field(
        ...,
        description="List of agent names executed in this stage",
        json_schema_extra={"example": ["critic", "historian", "refiner"]},
    )
    stage_duration_ms: float = Field(
        ...,
        description="Duration of this stage in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 2500.0},
    )
    success: bool = Field(
        ...,
        description="Whether this stage completed successfully",
        json_schema_extra={"example": True},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects
    )


class ExecutionResults(BaseModel):
    """
    Results of pipeline execution.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    success: bool = Field(
        ...,
        description="Whether the overall pipeline execution was successful",
        json_schema_extra={"example": True},
    )
    total_agents_executed: int = Field(
        ...,
        description="Total number of agents that were executed",
        ge=0,
        json_schema_extra={"example": 4},
    )
    successful_agents: int = Field(
        ...,
        description="Number of agents that executed successfully",
        ge=0,
        json_schema_extra={"example": 3},
    )
    failed_agents: int = Field(
        ...,
        description="Number of agents that failed during execution",
        ge=0,
        json_schema_extra={"example": 1},
    )
    execution_time_ms: float = Field(
        ...,
        description="Total execution time in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 5250.0},
    )
    pipeline_stages: List[PipelineStage] = Field(
        default_factory=list,
        description="List of pipeline stages that were executed",
    )
    resource_allocation_results: List[ResourceAllocationResult] = Field(
        default_factory=list,
        description="Results of resource allocations during execution",
    )
    failure_recovery_actions: List[str] = Field(
        default_factory=list,
        description="List of failure recovery actions that were taken",
        json_schema_extra={
            "example": ["retry_failed_agent", "fallback_to_alternative"]
        },
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects
    )

    def get_success_rate(self) -> float:
        """Calculate success rate as a fraction."""
        if self.total_agents_executed == 0:
            return 0.0
        return self.successful_agents / self.total_agents_executed

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Maintained for backward compatibility. Uses Pydantic's model_dump()
        internally for consistent serialization with calculated fields.
        """
        # Use model_dump with mode='json' to properly serialize enums
        data = self.model_dump(mode="json")

        # Add calculated fields
        data["success_rate"] = self.get_success_rate()

        return data


class AdvancedOrchestrator:
    """
    Advanced orchestrator with sophisticated dependency management.

    Integrates dependency graph execution, failure management, resource scheduling,
    and dynamic composition for comprehensive agent orchestration capabilities.
    """

    def __init__(
        self,
        graph_engine: DependencyGraphEngine,
        config: OrchestratorConfig,
    ) -> None:
        # Core components
        self.graph_engine = graph_engine
        self.config = config
        self.execution_planner = ExecutionPlanner()
        self.failure_manager = FailureManager(self.graph_engine)
        self.resource_scheduler = ResourceScheduler()
        self.dynamic_composer = (
            DynamicAgentComposer(self.graph_engine)
            if config.enable_dynamic_composition
            else None
        )

        # Execution state
        self._current_execution_plan: Optional[ExecutionPlan] = None

        # Agent management
        self.registry = get_agent_registry()
        self.loaded_agents: Dict[str, BaseAgent] = {}
        self.current_plan: Optional[ExecutionPlan] = None

        # Execution state
        self.execution_id: Optional[str] = None
        self.pipeline_start_time: Optional[float] = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize orchestrator components."""
        # Configure resource scheduler
        if self.config.enable_resource_scheduling:
            self.resource_scheduler.create_standard_pools()

        # Configure failure manager
        self.failure_manager.set_cascade_prevention(
            self.config.cascade_prevention_strategy
        )

        # Configure dynamic composition
        if self.dynamic_composer:
            # Add discoverers
            registry_discoverer = RegistryDiscoverer(self.registry)
            self.dynamic_composer.add_discoverer(registry_discoverer)

            # Add composition rules
            self.dynamic_composer.add_composition_rule(create_version_upgrade_rule())
            self.dynamic_composer.add_composition_rule(create_failure_recovery_rule())

    async def initialize_agents(self) -> None:
        """Initialize and configure agents for execution."""
        logger.info("Initializing agents for advanced orchestration")

        # Discover agents if dynamic composition is enabled
        if self.dynamic_composer:
            await self.dynamic_composer.discover_agents()

        # Get agent list
        agent_list = list(self.graph_engine.nodes.keys())
        if not agent_list:
            # Use all available agents from registry
            agent_list = list(self.registry._agents.keys())

        # Initialize LLM
        llm_config = OpenAIConfig.load()
        llm: LLMInterface = OpenAIChatLLM(
            api_key=llm_config.api_key,
            model=llm_config.model,
            base_url=llm_config.base_url,
        )

        # Load and configure agents
        for agent_name in agent_list:
            try:
                # Load agent
                if self.dynamic_composer:
                    agent = await self.dynamic_composer.load_agent(agent_name)
                else:
                    agent = self.registry.create_agent(agent_name, llm=llm)

                if agent is None:
                    logger.warning(f"Failed to load agent: {agent_name}")
                    continue

                self.loaded_agents[agent_name] = agent

                # Create dependency node
                node = self._create_dependency_node(agent_name, agent)
                self.graph_engine.add_node(node)

                # Configure failure handling
                self._configure_agent_failure_handling(agent_name, agent)

                logger.debug(f"Initialized agent: {agent_name}")

            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")

        # Add dependencies based on registry metadata
        self._add_agent_dependencies()

        # Validate graph
        validation_issues = self.graph_engine.validate_graph()
        if validation_issues:
            logger.warning(f"Dependency graph validation issues: {validation_issues}")

        logger.info(f"Initialized {len(self.loaded_agents)} agents")

    async def run(self, query: str) -> AgentContext:
        """
        Run the advanced orchestration pipeline.

        Parameters
        ----------
        query : str
            The query to process

        Returns
        -------
        AgentContext
            Final execution context
        """
        # Initialize execution
        self.execution_id = str(uuid.uuid4())
        self.pipeline_start_time = time.time()

        logger.info(f"Starting advanced orchestration pipeline: {self.execution_id}")

        # Initialize context
        context = AgentContext(query=query)
        context.set_path_metadata("execution_id", self.execution_id)
        context.set_path_metadata("orchestration_type", "advanced")

        # Initialize metrics
        metrics = get_metrics_collector()

        with observability_context(
            pipeline_id=self.execution_id, execution_phase="advanced_pipeline_start"
        ):
            try:
                # Phase 1: Agent Discovery and Composition
                await self._phase_discovery_and_composition(context)

                # Phase 2: Execution Planning
                await self._phase_execution_planning(context)

                # Phase 3: Resource Allocation
                await self._phase_resource_allocation(context)

                # Phase 4: Agent Execution
                await self._phase_agent_execution(context)

                # Phase 5: Cleanup and Finalization
                await self._phase_cleanup_and_finalization(context)

            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                await self._handle_pipeline_failure(e, context)
                raise

            finally:
                # Record final metrics
                start_time = self.pipeline_start_time or time.time()
                pipeline_duration = (time.time() - start_time) * 1000
                pipeline_success = len(context.agent_outputs) > 0

                metrics.record_pipeline_execution(
                    pipeline_id=self.execution_id,
                    success=pipeline_success,
                    duration_ms=pipeline_duration,
                    agents_executed=list(context.agent_outputs.keys()),
                    total_tokens=0,  # Would be aggregated from agent executions
                )

        logger.info(f"Advanced orchestration pipeline completed: {self.execution_id}")
        return context

    async def _phase_discovery_and_composition(self, context: AgentContext) -> None:
        """Phase 1: Agent discovery and dynamic composition."""
        logger.info("Phase 1: Agent Discovery and Composition")

        if not self.dynamic_composer:
            return

        # Auto-discovery and optimization
        discovery_results = await self.dynamic_composer.auto_discover_and_swap(context)
        context.execution_state["discovery_results"] = discovery_results

        # Apply composition optimization
        optimization_results = await self.dynamic_composer.optimize_composition(context)
        context.execution_state["composition_optimization"] = optimization_results

        logger.debug(f"Discovery results: {discovery_results}")
        logger.debug(f"Composition optimization: {optimization_results}")

    async def _phase_execution_planning(self, context: AgentContext) -> None:
        """Phase 2: Execution planning with dependency analysis."""
        logger.info("Phase 2: Execution Planning")

        # Create execution plan
        self.current_plan = self.execution_planner.create_plan(
            self.graph_engine,
            strategy=self.config.default_execution_strategy,
            context=context,
        )

        context.execution_state["execution_plan"] = (
            self.current_plan.get_execution_summary()
        )

        logger.info(
            f"Created execution plan: {self.current_plan.get_total_stages()} stages, "
            f"parallelism factor: {self.current_plan.parallelism_factor:.2f}"
        )

    async def _phase_resource_allocation(self, context: AgentContext) -> None:
        """Phase 3: Resource allocation and scheduling."""
        logger.info("Phase 3: Resource Allocation")

        if not self.config.enable_resource_scheduling:
            return

        # Allocate resources for all agents
        allocation_requests = []

        for agent_name, agent in self.loaded_agents.items():
            if agent_name in self.graph_engine.nodes:
                node = self.graph_engine.nodes[agent_name]

                if node.resource_constraints:
                    request_ids = await self.resource_scheduler.request_resources(
                        agent_id=agent_name,
                        resources=node.resource_constraints,
                        priority=node.priority,
                        estimated_duration_ms=node.timeout_ms,
                    )
                    allocation_requests.extend(request_ids)

        # Record resource allocation status
        utilization = self.resource_scheduler.get_resource_utilization()
        context.execution_state["resource_utilization"] = utilization

        logger.debug(f"Resource allocation requests: {len(allocation_requests)}")

    async def _phase_agent_execution(self, context: AgentContext) -> None:
        """Phase 4: Agent execution with advanced dependency management."""
        logger.info("Phase 4: Agent Execution")

        if not self.current_plan:
            raise RuntimeError("No execution plan available")

        # Execute stages according to plan
        for stage_index, stage in enumerate(self.current_plan.stages):
            logger.info(
                f"Executing stage {stage_index + 1}/{len(self.current_plan.stages)}: {stage.stage_id}"
            )

            # Update plan progress
            self.current_plan.current_stage_index = stage_index

            try:
                if stage.is_parallel():
                    await self._execute_parallel_stage(stage, context)
                else:
                    await self._execute_sequential_stage(stage, context)

                # Create checkpoint after each stage
                self.failure_manager.create_checkpoint(f"stage_{stage_index}", context)

            except Exception as e:
                logger.error(f"Stage {stage.stage_id} failed: {e}")

                # Attempt stage recovery
                recovery_successful = await self._attempt_stage_recovery(
                    stage, e, context
                )
                if not recovery_successful:
                    raise

        # Mark plan as completed
        self.current_plan.completed_at = time.time()

    async def _phase_cleanup_and_finalization(self, context: AgentContext) -> None:
        """Phase 5: Cleanup and finalization."""
        logger.info("Phase 5: Cleanup and Finalization")

        # Release all resources
        if self.config.enable_resource_scheduling:
            for agent_name in self.loaded_agents:
                await self.resource_scheduler.release_resources(agent_name)

        # Get final statistics
        if self.dynamic_composer:
            composition_status = self.dynamic_composer.get_composition_status()
            context.execution_state["final_composition_status"] = composition_status

        failure_stats = self.failure_manager.get_failure_statistics()
        context.execution_state["failure_statistics"] = failure_stats

        if self.config.enable_resource_scheduling:
            scheduling_stats = self.resource_scheduler.get_scheduling_statistics()
            context.execution_state["resource_scheduling_statistics"] = scheduling_stats

        # Set final execution metadata
        context.set_path_metadata(
            "pipeline_end", datetime.now(timezone.utc).isoformat()
        )
        start_time = self.pipeline_start_time or time.time()
        context.set_path_metadata(
            "total_duration_ms", (time.time() - start_time) * 1000
        )

    async def _execute_parallel_stage(self, stage: Any, context: AgentContext) -> None:
        """Execute a parallel stage with concurrent agent execution."""
        tasks = []

        # Create tasks for all parallel groups
        for group in stage.parallel_groups:
            for agent_id in group.agents:
                if agent_id in self.loaded_agents:
                    task = asyncio.create_task(
                        self._execute_agent_with_failure_handling(agent_id, context)
                    )
                    tasks.append(task)

        # Execute all agents in parallel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_sequential_stage(
        self, stage: Any, context: AgentContext
    ) -> None:
        """Execute a sequential stage with one-by-one agent execution."""
        for agent_id in stage.agents:
            if agent_id in self.loaded_agents:
                await self._execute_agent_with_failure_handling(agent_id, context)

    async def _execute_agent_with_failure_handling(
        self, agent_id: str, context: AgentContext
    ) -> None:
        """Execute an agent with comprehensive failure handling."""
        agent = self.loaded_agents[agent_id]
        attempt_number = 1
        max_attempts = 3

        while attempt_number <= max_attempts:
            try:
                # Check if agent can execute
                can_execute, reason = self.failure_manager.can_execute_agent(
                    agent_id, context
                )
                if not can_execute:
                    logger.warning(f"Agent {agent_id} cannot execute: {reason}")
                    return

                # Execute agent
                logger.info(f"Executing agent: {agent_id} (attempt {attempt_number})")
                await agent.run(context)

                # Record success
                if agent_id in self.failure_manager.circuit_breakers:
                    self.failure_manager.circuit_breakers[agent_id].record_success()

                logger.info(f"Agent {agent_id} completed successfully")
                return

            except Exception as e:
                logger.error(f"Agent {agent_id} failed (attempt {attempt_number}): {e}")

                # Handle failure
                (
                    should_retry,
                    recovery_action,
                ) = await self.failure_manager.handle_agent_failure(
                    agent_id, e, context, attempt_number
                )

                if recovery_action and recovery_action != "no_action":
                    recovery_success = await self.failure_manager.attempt_recovery(
                        agent_id, recovery_action, context
                    )
                    if recovery_success:
                        logger.info(f"Recovery successful for {agent_id}")
                        return

                if not should_retry or attempt_number >= max_attempts:
                    logger.error(
                        f"Agent {agent_id} failed permanently after {attempt_number} attempts"
                    )

                    # Try hot-swapping if dynamic composition is enabled
                    if self.dynamic_composer:
                        swap_success = await self._attempt_agent_hot_swap(
                            agent_id, context
                        )
                        if swap_success:
                            return

                    raise

                attempt_number += 1

                # Wait before retry
                if attempt_number <= max_attempts:
                    retry_config = self.failure_manager.retry_configs.get(
                        agent_id,
                        RetryConfiguration(
                            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                            max_attempts=3,
                            base_delay_ms=1000.0,
                            max_delay_ms=30000.0,
                            backoff_multiplier=2.0,
                            jitter=True,
                            reset_on_success=True,
                            success_rate_threshold=0.7,
                            failure_window_size=10,
                        ),
                    )
                    delay_ms = retry_config.calculate_delay(attempt_number)
                    await asyncio.sleep(delay_ms / 1000)

    async def _attempt_agent_hot_swap(
        self, failed_agent_id: str, context: AgentContext
    ) -> bool:
        """Attempt to hot-swap a failed agent."""
        if not self.dynamic_composer:
            return False

        logger.info(f"Attempting hot-swap for failed agent: {failed_agent_id}")

        # Find swap opportunities
        opportunities = await self.dynamic_composer._find_swap_opportunities(context)

        for opportunity in opportunities:
            if opportunity["old_agent"] == failed_agent_id:
                new_agent_id = opportunity["new_agent"]
                swap_success = await self.dynamic_composer.hot_swap_agent(
                    failed_agent_id, new_agent_id, context
                )

                if swap_success:
                    # Update loaded agents
                    if new_agent_id in self.dynamic_composer.loaded_agents:
                        self.loaded_agents[new_agent_id] = (
                            self.dynamic_composer.loaded_agents[new_agent_id]
                        )
                        if failed_agent_id in self.loaded_agents:
                            del self.loaded_agents[failed_agent_id]

                    logger.info(
                        f"Hot-swap successful: {failed_agent_id} -> {new_agent_id}"
                    )
                    return True

        return False

    async def _attempt_stage_recovery(
        self, stage: Any, error: Exception, context: AgentContext
    ) -> bool:
        """Attempt to recover from stage failure."""
        logger.info(f"Attempting stage recovery for: {stage.stage_id}")

        # Try fallback plan if available
        if self.current_plan and self.current_plan.fallback_plan:
            logger.info("Switching to fallback execution plan")
            self.current_plan = self.current_plan.fallback_plan
            return True

        # Try checkpoint rollback
        if self.failure_manager.recovery_checkpoints:
            logger.info("Attempting checkpoint rollback")
            return self.failure_manager._rollback_to_checkpoint("latest", context)

        return False

    async def _handle_pipeline_failure(
        self, error: Exception, context: AgentContext
    ) -> None:
        """Handle complete pipeline failure."""
        logger.error(f"Pipeline failure: {error}")

        # Record failure details
        context.set_path_metadata(
            "pipeline_failure",
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "timestamp": time.time(),
            },
        )

        # Attempt emergency recovery
        if self.failure_manager.recovery_checkpoints:
            logger.info("Attempting emergency recovery")
            self.failure_manager._rollback_to_checkpoint("earliest", context)

    def _create_dependency_node(
        self, agent_name: str, agent: BaseAgent
    ) -> DependencyNode:
        """Create a dependency node for an agent."""
        # Get metadata from registry if available
        registry_key = agent_name.lower()
        metadata = self.registry._agents.get(registry_key, None)

        # Determine priority
        priority = ExecutionPriority.NORMAL
        if hasattr(agent, "priority"):
            priority = agent.priority

        # Create resource constraints
        resource_constraints = []
        if hasattr(agent, "resource_requirements"):
            for req_type, req_amount in agent.resource_requirements.items():
                constraint = ResourceConstraint(
                    resource_type=req_type,
                    max_usage=req_amount,
                    units="units",
                    shared=False,
                    renewable=True,
                )
                resource_constraints.append(constraint)

        return DependencyNode(
            agent_id=agent_name,
            agent=agent,
            priority=priority,
            resource_constraints=resource_constraints,
            max_retries=3,
            timeout_ms=30000,
        )

    def _configure_agent_failure_handling(
        self, agent_name: str, agent: BaseAgent
    ) -> None:
        """Configure failure handling for an agent."""
        # Configure retry behavior
        retry_config = RetryConfiguration(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay_ms=1000.0,
            max_delay_ms=30000.0,
            backoff_multiplier=2.0,
            jitter=True,
            reset_on_success=True,
            success_rate_threshold=0.7,
            failure_window_size=10,
        )
        self.failure_manager.configure_retry(agent_name, retry_config)

        # Configure circuit breaker
        self.failure_manager.configure_circuit_breaker(
            agent_name, failure_threshold=3, recovery_timeout_ms=60000
        )

    def _add_agent_dependencies(self) -> None:
        """Add agent dependencies based on registry metadata."""
        for agent_name in self.loaded_agents:
            registry_key = agent_name.lower()
            if registry_key in self.registry._agents:
                metadata = self.registry._agents[registry_key]
                dependencies = metadata.dependencies or []

                for dependency in dependencies:
                    if dependency in self.loaded_agents:
                        self.graph_engine.add_dependency(
                            from_agent=dependency,
                            to_agent=agent_name,
                            dependency_type=DependencyType.HARD,
                        )

    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status."""
        status: Dict[str, Any] = {
            "execution_id": self.execution_id,
            "pipeline_start_time": self.pipeline_start_time,
            "loaded_agents": len(self.loaded_agents),
            "execution_strategy": self.config.default_execution_strategy.value,
            "enable_dynamic_composition": self.config.enable_dynamic_composition,
            "enable_resource_scheduling": self.config.enable_resource_scheduling,
        }

        if self.current_plan:
            status["execution_plan"] = self.current_plan.get_execution_summary()

        if self.dynamic_composer:
            status["composition_status"] = (
                self.dynamic_composer.get_composition_status()
            )

        status["graph_statistics"] = self.graph_engine.get_execution_statistics()
        status["failure_statistics"] = self.failure_manager.get_failure_statistics()

        if self.config.enable_resource_scheduling:
            status["resource_utilization"] = (
                self.resource_scheduler.get_resource_utilization()
            )
            status["scheduling_statistics"] = (
                self.resource_scheduler.get_scheduling_statistics()
            )

        return status

    async def execute_pipeline(self, context: AgentContext) -> ExecutionResults:
        """Execute the complete orchestration pipeline."""
        start_time = time.time()
        pipeline_stages: List[PipelineStage] = []
        resource_allocation_results: List[ResourceAllocationResult] = []
        failure_recovery_actions: List[str] = []
        agent_results: List[Dict[str, Any]] = []

        try:
            # Check for timeout
            if self.config.pipeline_timeout_ms:
                timeout_seconds = self.config.pipeline_timeout_ms / 1000.0
                return await asyncio.wait_for(
                    self._execute_pipeline_impl(
                        context,
                        start_time,
                        pipeline_stages,
                        resource_allocation_results,
                        failure_recovery_actions,
                        agent_results,
                    ),
                    timeout=timeout_seconds,
                )
            else:
                return await self._execute_pipeline_impl(
                    context,
                    start_time,
                    pipeline_stages,
                    resource_allocation_results,
                    failure_recovery_actions,
                    agent_results,
                )
        except asyncio.TimeoutError:
            # Handle timeout
            return ExecutionResults(
                success=False,
                total_agents_executed=0,
                successful_agents=0,
                failed_agents=len(self.graph_engine.nodes),
                execution_time_ms=(time.time() - start_time) * 1000,
                pipeline_stages=pipeline_stages,
                resource_allocation_results=resource_allocation_results,
                failure_recovery_actions=["pipeline_timeout"],
            )
        except Exception as e:
            # Handle pipeline failure
            logger.error(f"Pipeline execution failed: {e}")
            return ExecutionResults(
                success=False,
                total_agents_executed=0,
                successful_agents=0,
                failed_agents=len(self.graph_engine.nodes),
                execution_time_ms=(time.time() - start_time) * 1000,
                pipeline_stages=pipeline_stages,
                resource_allocation_results=resource_allocation_results,
                failure_recovery_actions=failure_recovery_actions,
            )

    async def _execute_pipeline_impl(
        self,
        context: AgentContext,
        start_time: float,
        pipeline_stages: List[PipelineStage],
        resource_allocation_results: List[ResourceAllocationResult],
        failure_recovery_actions: List[str],
        agent_results: List[Dict[str, Any]],
    ) -> ExecutionResults:
        """Internal implementation of pipeline execution."""
        # Phase 1: Preparation
        prep_stage = await self._prepare_execution(context)
        pipeline_stages.append(prep_stage)

        # Phase 2: Resource Allocation
        if self.config.enable_resource_scheduling:
            alloc_stage = await self._allocate_resources(context)
            pipeline_stages.append(alloc_stage)

        # Phase 3: Agent Execution
        exec_stage = await self._execute_agents(context)
        pipeline_stages.append(exec_stage)

        # Phase 4: Cleanup
        cleanup_stage = await self._cleanup_resources(context)
        pipeline_stages.append(cleanup_stage)

        # Calculate results
        successful_agents = len(context.agent_outputs)
        total_agents = len(self.graph_engine.nodes)
        failed_agents = total_agents - successful_agents

        # Collect failure recovery actions from context
        context_recovery_actions = context.execution_state.get(
            "failure_recovery_actions", []
        )
        failure_recovery_actions.extend(context_recovery_actions)

        # Collect resource allocation results from context
        context_resource_results = context.execution_state.get(
            "resource_allocation_results", []
        )
        resource_allocation_results.extend(context_resource_results)

        # Pipeline is successful only if all stages succeeded and no agents failed
        overall_success = (
            all(stage.success for stage in pipeline_stages) and failed_agents == 0
        )

        return ExecutionResults(
            success=overall_success,
            total_agents_executed=total_agents,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            execution_time_ms=(time.time() - start_time) * 1000,
            pipeline_stages=pipeline_stages,
            resource_allocation_results=resource_allocation_results,
            failure_recovery_actions=failure_recovery_actions,
        )

    async def _prepare_execution(self, context: AgentContext) -> PipelineStage:
        """Prepare for execution."""
        stage_start = time.time()

        try:
            # Create execution plan
            self._current_execution_plan = self.execution_planner.create_plan(
                self.graph_engine,
                strategy=self.config.default_execution_strategy,
                context=context,
            )

            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="preparation",
                phase=ExecutionPhase.PREPARATION,
                agents_executed=[],
                stage_duration_ms=stage_duration,
                success=True,
            )
        except Exception:
            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="preparation",
                phase=ExecutionPhase.PREPARATION,
                agents_executed=[],
                stage_duration_ms=stage_duration,
                success=False,
            )

    async def _allocate_resources(self, context: AgentContext) -> PipelineStage:
        """Allocate resources for agents."""
        stage_start = time.time()
        agents_processed = []

        try:
            for agent_id, node in self.graph_engine.nodes.items():
                agents_processed.append(agent_id)
                if node.resource_constraints:
                    # Request resources
                    request_ids = await self.resource_scheduler.request_resources(
                        agent_id=agent_id,
                        resources=node.resource_constraints,
                        priority=node.priority,
                        estimated_duration_ms=node.timeout_ms,
                    )

                    # Create resource allocation results for tracking
                    allocation_results = context.execution_state.get(
                        "resource_allocation_results", []
                    )
                    for constraint in node.resource_constraints:
                        result = ResourceAllocationResult(
                            agent_id=agent_id,
                            resource_type=self.resource_scheduler._map_constraint_to_type(
                                constraint
                            ),
                            requested_amount=constraint.max_usage,
                            allocated_amount=constraint.max_usage,  # Assume successful allocation
                            allocation_time_ms=10.0,  # Placeholder timing
                            success=True,
                        )
                        allocation_results.append(result)
                    context.execution_state["resource_allocation_results"] = (
                        allocation_results
                    )

            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="resource_allocation",
                phase=ExecutionPhase.RESOURCE_ALLOCATION,
                agents_executed=agents_processed,
                stage_duration_ms=stage_duration,
                success=True,
            )
        except Exception:
            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="resource_allocation",
                phase=ExecutionPhase.RESOURCE_ALLOCATION,
                agents_executed=agents_processed,
                stage_duration_ms=stage_duration,
                success=False,
            )

    async def _execute_agents(self, context: AgentContext) -> PipelineStage:
        """Execute agents according to the plan."""
        stage_start = time.time()
        agents_executed = []
        execution_results = []
        has_failures = False

        try:
            # Get execution order
            execution_order = self.graph_engine.get_execution_order()

            # Execute agents in order
            for agent_id in execution_order:
                if agent_id in self.graph_engine.nodes:
                    node = self.graph_engine.nodes[agent_id]
                    result = await self._handle_agent_execution(
                        agent_id, node.agent, context
                    )
                    agents_executed.append(agent_id)
                    execution_results.append(result)

                    # Track failures and collect recovery actions
                    if not result.get("success", False):
                        has_failures = True
                        # Store recovery action in context for collection later
                        if "recovery_action" in result and result["recovery_action"]:
                            recovery_actions = context.execution_state.get(
                                "failure_recovery_actions", []
                            )
                            recovery_actions.append(result["recovery_action"])
                            context.execution_state["failure_recovery_actions"] = (
                                recovery_actions
                            )

            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="execution",
                phase=ExecutionPhase.EXECUTION,
                agents_executed=agents_executed,
                stage_duration_ms=stage_duration,
                success=not has_failures,  # Stage succeeds only if no agent failures
            )
        except Exception:
            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="execution",
                phase=ExecutionPhase.EXECUTION,
                agents_executed=agents_executed,
                stage_duration_ms=stage_duration,
                success=False,
            )

    async def _cleanup_resources(self, context: AgentContext) -> PipelineStage:
        """Cleanup resources after execution."""
        stage_start = time.time()

        try:
            # Release resources for all agents
            for agent_id in self.graph_engine.nodes:
                await self.resource_scheduler.release_resources(agent_id)

            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="cleanup",
                phase=ExecutionPhase.CLEANUP,
                agents_executed=[],
                stage_duration_ms=stage_duration,
                success=True,
            )
        except Exception:
            stage_duration = (time.time() - stage_start) * 1000
            return PipelineStage(
                stage_id="cleanup",
                phase=ExecutionPhase.CLEANUP,
                agents_executed=[],
                stage_duration_ms=stage_duration,
                success=False,
            )

    async def _handle_agent_execution(
        self, agent_id: str, agent: BaseAgent, context: AgentContext
    ) -> Dict[str, Any]:
        """Handle execution of a single agent."""
        start_time = time.time()

        try:
            # Check if agent can execute
            can_execute, reason = self.failure_manager.can_execute_agent(
                agent_id, context
            )
            if not can_execute:
                return {
                    "agent_id": agent_id,
                    "success": False,
                    "error": f"Agent blocked: {reason}",
                    "execution_time_ms": 0,
                }

            # Get timeout from node
            timeout_ms = None
            if agent_id in self.graph_engine.nodes:
                timeout_ms = self.graph_engine.nodes[agent_id].timeout_ms

            # Execute with timeout
            if timeout_ms:
                timeout_seconds = timeout_ms / 1000.0
                try:
                    result_context = await asyncio.wait_for(
                        agent.run(context), timeout=timeout_seconds
                    )
                    # Update context with result
                    for key, value in result_context.agent_outputs.items():
                        context.agent_outputs[key] = value
                except asyncio.TimeoutError:
                    return {
                        "agent_id": agent_id,
                        "success": False,
                        "error": f"Agent execution timeout after {timeout_seconds}s",
                        "execution_time_ms": (time.time() - start_time) * 1000,
                    }
            else:
                await agent.run(context)

            execution_time = (time.time() - start_time) * 1000
            return {
                "agent_id": agent_id,
                "success": True,
                "execution_time_ms": execution_time,
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            # Handle failure through failure manager
            (
                should_retry,
                recovery_action,
            ) = await self.failure_manager.handle_agent_failure(
                agent_id, e, context, attempt_number=1
            )

            return {
                "agent_id": agent_id,
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "should_retry": should_retry,
                "recovery_action": recovery_action,
            }

    def _create_resource_allocation_result(
        self,
        agent_id: str,
        resource_type: ResourceType,
        requested_amount: float,
        allocated_amount: float,
        allocation_time_ms: float,
        success: bool,
    ) -> ResourceAllocationResult:
        """Create a resource allocation result."""
        return ResourceAllocationResult(
            agent_id=agent_id,
            resource_type=resource_type,
            requested_amount=requested_amount,
            allocated_amount=allocated_amount,
            allocation_time_ms=allocation_time_ms,
            success=success,
        )

    def _aggregate_execution_results(
        self,
        agent_results: List[Dict[str, Any]],
        pipeline_stages: List[PipelineStage],
        resource_results: List[ResourceAllocationResult],
        failure_actions: List[str],
        start_time: float,
    ) -> ExecutionResults:
        """Aggregate execution results."""
        successful_agents = sum(
            1 for result in agent_results if result.get("success", False)
        )
        failed_agents = len(agent_results) - successful_agents
        overall_success = failed_agents == 0 and all(
            stage.success for stage in pipeline_stages
        )

        return ExecutionResults(
            success=overall_success,
            total_agents_executed=len(agent_results),
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            execution_time_ms=(time.time() - start_time) * 1000,
            pipeline_stages=pipeline_stages,
            resource_allocation_results=resource_results,
            failure_recovery_actions=failure_actions,
        )

    def _create_pipeline_stage(
        self,
        stage_id: str,
        phase: ExecutionPhase,
        agents_executed: List[str],
        stage_duration_ms: float,
        success: bool,
    ) -> PipelineStage:
        """Create a pipeline stage."""
        return PipelineStage(
            stage_id=stage_id,
            phase=phase,
            agents_executed=agents_executed,
            stage_duration_ms=stage_duration_ms,
            success=success,
        )

    def _calculate_execution_statistics(
        self, agent_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate execution statistics."""
        total_agents = len(agent_results)
        successful_agents = sum(
            1 for result in agent_results if result.get("success", False)
        )
        failed_agents = total_agents - successful_agents

        success_rate = successful_agents / total_agents if total_agents > 0 else 0.0

        execution_times = [
            result.get("execution_time_ms", 0) for result in agent_results
        ]
        avg_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0.0
        )
        total_execution_time = sum(execution_times)

        return {
            "total_agents": total_agents,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_execution_time,
            "total_execution_time_ms": total_execution_time,
        }
