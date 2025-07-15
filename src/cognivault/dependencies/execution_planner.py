"""
Execution planner for optimized agent execution strategies.

This module provides execution planning capabilities that work with the dependency
graph engine to create optimal execution plans considering parallelism, resource
constraints, and failure scenarios.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from cognivault.context import AgentContext
from cognivault.observability import get_logger
from .graph_engine import DependencyGraphEngine, DependencyNode

logger = get_logger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategy types."""

    SEQUENTIAL = "sequential"  # Execute agents one by one
    PARALLEL_BATCHED = "parallel_batched"  # Execute in parallel batches
    ADAPTIVE = "adaptive"  # Dynamically adapt based on conditions
    PRIORITY_FIRST = "priority_first"  # Execute high priority agents first
    RESOURCE_OPTIMIZED = "resource_optimized"  # Optimize for resource usage


class StageType(Enum):
    """Types of execution stages."""

    PARALLEL = "parallel"  # All agents in stage run in parallel
    SEQUENTIAL = "sequential"  # Agents run one after another
    CONDITIONAL = "conditional"  # Execution depends on conditions
    FALLBACK = "fallback"  # Fallback execution path


@dataclass
class ParallelGroup:
    """Group of agents that can execute in parallel."""

    agents: List[str]
    max_concurrency: Optional[int] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    estimated_duration_ms: Optional[float] = None
    dependencies_satisfied: bool = True

    def can_add_agent(self, agent_id: str, node: DependencyNode) -> bool:
        """Check if an agent can be added to this parallel group."""
        if self.max_concurrency and len(self.agents) >= self.max_concurrency:
            return False

        # Check for exclusive access conflicts
        if node.requires_exclusive_access and len(self.agents) > 0:
            return False

        for existing_agent in self.agents:
            # This would check for resource conflicts
            # Implementation would depend on resource scheduler
            pass

        return True


@dataclass
class ExecutionStage:
    """A stage in the execution plan."""

    stage_id: str
    stage_type: StageType
    agents: List[str] = field(default_factory=list)
    parallel_groups: List[ParallelGroup] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Stage dependencies
    conditions: List[str] = field(default_factory=list)  # Execution conditions
    estimated_duration_ms: Optional[float] = None
    max_retries: int = 3
    timeout_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_agents(self) -> List[str]:
        """Get all agents in this stage."""
        all_agents = self.agents.copy()
        for group in self.parallel_groups:
            all_agents.extend(group.agents)
        return list(set(all_agents))

    def get_execution_count(self) -> int:
        """Get total number of agents to execute in this stage."""
        return len(self.get_all_agents())

    def is_parallel(self) -> bool:
        """Check if this stage involves parallel execution."""
        return (
            self.stage_type == StageType.PARALLEL
            or len(self.parallel_groups) > 0
            or len(self.agents) > 1
        )


@dataclass
class ExecutionPlan:
    """Complete execution plan with stages and metadata."""

    plan_id: str
    stages: List[ExecutionStage]
    strategy: ExecutionStrategy
    total_agents: int
    estimated_total_duration_ms: Optional[float] = None
    parallelism_factor: float = 1.0  # Average parallelism achieved
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    fallback_plan: Optional["ExecutionPlan"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime tracking
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_stage_index: int = 0

    def get_total_stages(self) -> int:
        """Get total number of stages in the plan."""
        return len(self.stages)

    def get_current_stage(self) -> Optional[ExecutionStage]:
        """Get the current execution stage."""
        if 0 <= self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None

    def advance_stage(self) -> bool:
        """Advance to the next stage. Returns True if successful."""
        if self.current_stage_index < len(self.stages):
            self.current_stage_index += 1
            return True
        return False

    def get_remaining_stages(self) -> List[ExecutionStage]:
        """Get remaining stages to execute."""
        return self.stages[self.current_stage_index + 1 :]

    def get_completion_percentage(self) -> float:
        """Get completion percentage (0-100)."""
        if not self.stages:
            return 100.0
        return (self.current_stage_index / len(self.stages)) * 100

    def is_completed(self) -> bool:
        """Check if execution plan is completed."""
        return self.current_stage_index >= len(self.stages)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        duration = None
        if self.started_at and self.completed_at:
            duration = self.completed_at - self.started_at
        elif self.started_at:
            duration = time.time() - self.started_at

        return {
            "plan_id": self.plan_id,
            "strategy": self.strategy.value,
            "total_stages": self.get_total_stages(),
            "total_agents": self.total_agents,
            "current_stage": self.current_stage_index,
            "completion_percentage": self.get_completion_percentage(),
            "estimated_duration_ms": self.estimated_total_duration_ms,
            "actual_duration_ms": duration * 1000 if duration else None,
            "parallelism_factor": self.parallelism_factor,
            "is_completed": self.is_completed(),
            "has_fallback": self.fallback_plan is not None,
        }


class ExecutionPlanBuilder(ABC):
    """Abstract base for execution plan builders."""

    @abstractmethod
    def build_plan(
        self,
        graph_engine: DependencyGraphEngine,
        context: Optional[AgentContext] = None,
    ) -> ExecutionPlan:
        """Build an execution plan from the dependency graph."""
        pass


class SequentialPlanBuilder(ExecutionPlanBuilder):
    """Builder for sequential execution plans."""

    def build_plan(
        self,
        graph_engine: DependencyGraphEngine,
        context: Optional[AgentContext] = None,
    ) -> ExecutionPlan:
        """Build a sequential execution plan."""
        execution_order = graph_engine.get_execution_order(context)
        stages = []

        for i, agent_id in enumerate(execution_order):
            stage = ExecutionStage(
                stage_id=f"seq_stage_{i}",
                stage_type=StageType.SEQUENTIAL,
                agents=[agent_id],
                metadata={"order": i},
            )
            stages.append(stage)

        plan = ExecutionPlan(
            plan_id=f"sequential_{int(time.time())}",
            stages=stages,
            strategy=ExecutionStrategy.SEQUENTIAL,
            total_agents=len(execution_order),
            parallelism_factor=1.0,
        )

        return plan


class ParallelBatchedPlanBuilder(ExecutionPlanBuilder):
    """Builder for parallel batched execution plans."""

    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size

    def build_plan(
        self,
        graph_engine: DependencyGraphEngine,
        context: Optional[AgentContext] = None,
    ) -> ExecutionPlan:
        """Build a parallel batched execution plan."""
        parallel_groups = graph_engine.get_parallel_groups(context)
        stages = []
        total_parallelism = 0

        for i, group in enumerate(parallel_groups):
            # Split large groups into smaller batches
            batches = self._split_into_batches(group, graph_engine)

            for j, batch in enumerate(batches):
                parallel_group = ParallelGroup(
                    agents=batch, max_concurrency=self.max_batch_size
                )

                stage = ExecutionStage(
                    stage_id=f"parallel_stage_{i}_{j}",
                    stage_type=StageType.PARALLEL,
                    parallel_groups=[parallel_group],
                    metadata={"level": i, "batch": j},
                )
                stages.append(stage)
                total_parallelism += len(batch)

        # Calculate parallelism factor (average agents per stage)
        total_agents = len(graph_engine.nodes)
        num_stages = len(stages) if stages else 1
        parallelism_factor = total_agents / num_stages if num_stages > 0 else 1.0

        plan = ExecutionPlan(
            plan_id=f"parallel_batched_{int(time.time())}",
            stages=stages,
            strategy=ExecutionStrategy.PARALLEL_BATCHED,
            total_agents=total_agents,
            parallelism_factor=parallelism_factor,
        )

        return plan

    def _split_into_batches(
        self, agents: List[str], graph_engine: DependencyGraphEngine
    ) -> List[List[str]]:
        """Split a large group into smaller batches based on constraints."""
        if len(agents) <= self.max_batch_size:
            return [agents]

        batches = []
        current_batch: List[str] = []

        # Sort by priority first
        agent_priorities = [
            (agent_id, graph_engine.nodes[agent_id].priority.value)
            for agent_id in agents
        ]
        agent_priorities.sort(key=lambda x: x[1])  # Higher priority first

        for agent_id, _ in agent_priorities:
            node = graph_engine.nodes[agent_id]

            # Check if agent can be added to current batch
            if (
                len(current_batch) < self.max_batch_size
                and not node.requires_exclusive_access
            ):
                current_batch.append(agent_id)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [agent_id]

        if current_batch:
            batches.append(current_batch)

        return batches


class AdaptivePlanBuilder(ExecutionPlanBuilder):
    """Builder for adaptive execution plans that adjust based on conditions."""

    def __init__(self):
        self.sequential_builder = SequentialPlanBuilder()
        self.parallel_builder = ParallelBatchedPlanBuilder()

    def build_plan(
        self,
        graph_engine: DependencyGraphEngine,
        context: Optional[AgentContext] = None,
    ) -> ExecutionPlan:
        """Build an adaptive execution plan."""
        # Analyze graph characteristics to choose strategy
        stats = graph_engine.get_execution_statistics()
        parallel_groups = graph_engine.get_parallel_groups(context)

        # Decision logic for strategy selection
        max_parallel_size = (
            max(len(group) for group in parallel_groups) if parallel_groups else 1
        )
        avg_parallel_size = (
            sum(len(group) for group in parallel_groups) / len(parallel_groups)
            if parallel_groups
            else 1
        )

        # Use parallel strategy if there's significant parallelism opportunity
        if max_parallel_size >= 3 or avg_parallel_size >= 2:
            base_plan = self.parallel_builder.build_plan(graph_engine, context)
            strategy = ExecutionStrategy.PARALLEL_BATCHED
        else:
            base_plan = self.sequential_builder.build_plan(graph_engine, context)
            strategy = ExecutionStrategy.SEQUENTIAL

        # Create adaptive plan with fallback
        fallback_plan = None
        if strategy == ExecutionStrategy.PARALLEL_BATCHED:
            fallback_plan = self.sequential_builder.build_plan(graph_engine, context)

        adaptive_plan = ExecutionPlan(
            plan_id=f"adaptive_{int(time.time())}",
            stages=base_plan.stages,
            strategy=ExecutionStrategy.ADAPTIVE,
            total_agents=base_plan.total_agents,
            parallelism_factor=base_plan.parallelism_factor,
            fallback_plan=fallback_plan,
            metadata={
                "base_strategy": strategy.value,
                "max_parallel_size": max_parallel_size,
                "avg_parallel_size": avg_parallel_size,
                "adaptation_criteria": "parallelism_analysis",
            },
        )

        return adaptive_plan


class PriorityFirstPlanBuilder(ExecutionPlanBuilder):
    """Builder for priority-first execution plans."""

    def build_plan(
        self,
        graph_engine: DependencyGraphEngine,
        context: Optional[AgentContext] = None,
    ) -> ExecutionPlan:
        """Build a priority-first execution plan that respects dependencies."""
        # Get base execution order (already respects dependencies)
        execution_order = graph_engine.get_execution_order(context)

        # Get parallel groups to understand what can run together
        parallel_groups = graph_engine.get_parallel_groups(context)

        # Create stages, prioritizing higher priority agents within each parallel group
        stages = []

        for i, group in enumerate(parallel_groups):
            # Sort group by priority (lower value = higher priority)
            group_with_priority = [
                (agent_id, graph_engine.nodes[agent_id].priority.value)
                for agent_id in group
            ]
            group_with_priority.sort(key=lambda x: x[1])
            sorted_group = [agent_id for agent_id, _ in group_with_priority]

            if len(sorted_group) == 1:
                # Single agent stage
                priority = graph_engine.nodes[sorted_group[0]].priority
                stage = ExecutionStage(
                    stage_id=f"priority_stage_{i}",
                    stage_type=StageType.SEQUENTIAL,
                    agents=sorted_group,
                    metadata={"priority": priority.name, "level": i},
                )
            else:
                # Parallel group stage
                parallel_group = ParallelGroup(agents=sorted_group)
                stage = ExecutionStage(
                    stage_id=f"priority_stage_{i}",
                    stage_type=StageType.PARALLEL,
                    parallel_groups=[parallel_group],
                    metadata={"level": i},
                )

            stages.append(stage)

        # Calculate parallelism factor
        total_agents = len(execution_order)
        num_stages = len(stages) if stages else 1
        parallelism_factor = total_agents / num_stages if num_stages > 0 else 1.0

        plan = ExecutionPlan(
            plan_id=f"priority_first_{int(time.time())}",
            stages=stages,
            strategy=ExecutionStrategy.PRIORITY_FIRST,
            total_agents=total_agents,
            parallelism_factor=parallelism_factor,
            metadata={"parallel_groups": len(parallel_groups)},
        )

        return plan


class ExecutionPlanner:
    """
    Main execution planner that coordinates different planning strategies.

    Provides a unified interface for creating optimized execution plans based on
    dependency graphs, execution strategies, and runtime conditions.
    """

    def __init__(self):
        self.builders = {
            ExecutionStrategy.SEQUENTIAL: SequentialPlanBuilder(),
            ExecutionStrategy.PARALLEL_BATCHED: ParallelBatchedPlanBuilder(),
            ExecutionStrategy.ADAPTIVE: AdaptivePlanBuilder(),
            ExecutionStrategy.PRIORITY_FIRST: PriorityFirstPlanBuilder(),
        }
        self.default_strategy = ExecutionStrategy.ADAPTIVE

    def create_plan(
        self,
        graph_engine: DependencyGraphEngine,
        strategy: Optional[ExecutionStrategy] = None,
        context: Optional[AgentContext] = None,
        **kwargs,
    ) -> ExecutionPlan:
        """
        Create an execution plan using the specified strategy.

        Parameters
        ----------
        graph_engine : DependencyGraphEngine
            The dependency graph to plan execution for
        strategy : Optional[ExecutionStrategy]
            Execution strategy to use (defaults to adaptive)
        context : Optional[AgentContext]
            Current execution context
        **kwargs
            Additional parameters for specific builders

        Returns
        -------
        ExecutionPlan
            Optimized execution plan
        """
        if strategy is None:
            strategy = self.default_strategy

        if strategy not in self.builders:
            logger.warning(
                f"Unknown strategy {strategy}, using {self.default_strategy}"
            )
            strategy = self.default_strategy

        builder = self.builders[strategy]

        # Configure builder with kwargs if needed
        if (
            strategy == ExecutionStrategy.PARALLEL_BATCHED
            and "max_batch_size" in kwargs
        ):
            builder.max_batch_size = kwargs["max_batch_size"]

        logger.info(f"Creating execution plan with strategy: {strategy.value}")
        plan = builder.build_plan(graph_engine, context)

        # Add timing estimates if available
        self._add_timing_estimates(plan, graph_engine)

        logger.info(
            f"Created execution plan: {len(plan.stages)} stages, "
            f"{plan.total_agents} agents, "
            f"parallelism factor: {plan.parallelism_factor:.2f}"
        )

        return plan

    def compare_strategies(
        self,
        graph_engine: DependencyGraphEngine,
        strategies: Optional[List[ExecutionStrategy]] = None,
        context: Optional[AgentContext] = None,
    ) -> Dict[ExecutionStrategy, ExecutionPlan]:
        """
        Compare multiple execution strategies and return all plans.

        Parameters
        ----------
        graph_engine : DependencyGraphEngine
            The dependency graph to plan for
        strategies : Optional[List[ExecutionStrategy]]
            List of strategies to compare (defaults to all available)
        context : Optional[AgentContext]
            Current execution context

        Returns
        -------
        Dict[ExecutionStrategy, ExecutionPlan]
            Mapping of strategies to their execution plans
        """
        if strategies is None:
            strategies = list(self.builders.keys())

        plans = {}
        for strategy in strategies:
            try:
                plan = self.create_plan(graph_engine, strategy, context)
                plans[strategy] = plan
            except Exception as e:
                logger.warning(f"Failed to create plan for strategy {strategy}: {e}")

        return plans

    def recommend_strategy(
        self,
        graph_engine: DependencyGraphEngine,
        context: Optional[AgentContext] = None,
        optimization_goal: str = "balanced",
    ) -> ExecutionStrategy:
        """
        Recommend the best execution strategy based on graph characteristics.

        Parameters
        ----------
        graph_engine : DependencyGraphEngine
            The dependency graph to analyze
        context : Optional[AgentContext]
            Current execution context
        optimization_goal : str
            Optimization goal: "speed", "reliability", "resource", "balanced"

        Returns
        -------
        ExecutionStrategy
            Recommended execution strategy
        """
        stats = graph_engine.get_execution_statistics()
        parallel_groups = graph_engine.get_parallel_groups(context)

        # Analysis metrics
        total_agents = stats["total_nodes"]
        avg_dependencies = stats["average_dependencies_per_node"]
        max_parallel_size = (
            max(len(group) for group in parallel_groups) if parallel_groups else 1
        )

        # Strategy recommendation logic
        if optimization_goal == "speed":
            if max_parallel_size >= 3:
                return ExecutionStrategy.PARALLEL_BATCHED
            else:
                return ExecutionStrategy.PRIORITY_FIRST

        elif optimization_goal == "reliability":
            if avg_dependencies > 2:
                return ExecutionStrategy.SEQUENTIAL
            else:
                return ExecutionStrategy.PRIORITY_FIRST

        elif optimization_goal == "resource":
            return ExecutionStrategy.SEQUENTIAL

        else:  # balanced
            return ExecutionStrategy.ADAPTIVE

    def _add_timing_estimates(
        self, plan: ExecutionPlan, graph_engine: DependencyGraphEngine
    ) -> None:
        """Add timing estimates to the execution plan."""
        total_estimated_duration = 0

        for stage in plan.stages:
            # Estimate stage duration based on agent characteristics
            stage_duration = 0

            if stage.is_parallel():
                # For parallel stages, duration is the max of all parallel groups
                max_group_duration = 0
                for group in stage.parallel_groups:
                    group_duration = 0
                    for agent_id in group.agents:
                        node = graph_engine.nodes[agent_id]
                        agent_duration = node.timeout_ms or 30000  # Default 30s
                        group_duration = max(group_duration, agent_duration)
                    max_group_duration = max(max_group_duration, group_duration)
                stage_duration = max_group_duration
            else:
                # For sequential stages, sum all agent durations
                for agent_id in stage.agents:
                    node = graph_engine.nodes[agent_id]
                    stage_duration += node.timeout_ms or 30000

            stage.estimated_duration_ms = stage_duration
            total_estimated_duration += stage_duration

        plan.estimated_total_duration_ms = total_estimated_duration
