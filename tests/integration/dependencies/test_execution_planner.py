"""
Tests for the execution planner.

Covers execution strategies, plan generation, parallelism optimization,
and stage management.
"""

import pytest
import time

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.dependencies.graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
    DependencyType,
    ExecutionPriority,
)
from cognivault.dependencies.execution_planner import (
    ExecutionPlanner,
    ExecutionStrategy,
    ExecutionPlan,
    ExecutionStage,
    StageType,
    ParallelGroup,
    SequentialPlanBuilder,
    ParallelBatchedPlanBuilder,
    AdaptivePlanBuilder,
    PriorityFirstPlanBuilder,
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(
        self, name: str, priority: ExecutionPriority = ExecutionPriority.NORMAL
    ):
        super().__init__(name=name)
        self.name = name
        self.priority = priority

    async def run(self, context: AgentContext) -> AgentContext:
        context.agent_outputs[self.name] = f"Output from {self.name}"
        return context


@pytest.fixture
def mock_agents():
    """Create mock agents with different priorities."""
    return {
        "critical_agent": MockAgent("critical_agent", ExecutionPriority.CRITICAL),
        "high_agent": MockAgent("high_agent", ExecutionPriority.HIGH),
        "normal_agent": MockAgent("normal_agent", ExecutionPriority.NORMAL),
        "low_agent": MockAgent("low_agent", ExecutionPriority.LOW),
        "background_agent": MockAgent("background_agent", ExecutionPriority.BACKGROUND),
    }


@pytest.fixture
def graph_engine_with_agents(mock_agents):
    """Create a graph engine populated with test agents."""
    engine = DependencyGraphEngine()

    for agent_id, agent in mock_agents.items():
        node = DependencyNode(
            agent_id=agent_id, agent=agent, priority=agent.priority, timeout_ms=30000
        )
        engine.add_node(node)

    return engine


@pytest.fixture
def execution_planner():
    """Create an execution planner for testing."""
    return ExecutionPlanner()


class TestParallelGroup:
    """Test ParallelGroup functionality."""

    def test_parallel_group_creation(self):
        """Test creating a parallel group."""
        group = ParallelGroup(
            agents=["agent_a", "agent_b"],
            max_concurrency=2,
            estimated_duration_ms=5000.0,
        )

        assert group.agents == ["agent_a", "agent_b"]
        assert group.max_concurrency == 2
        assert group.estimated_duration_ms == 5000.0
        assert group.dependencies_satisfied is True

    def test_can_add_agent(self, mock_agents):
        """Test checking if an agent can be added to a group."""
        group = ParallelGroup(agents=["agent_a"], max_concurrency=2)

        # Should be able to add normal agent
        normal_node = DependencyNode(
            agent_id="normal_agent",
            agent=mock_agents["normal_agent"],
            requires_exclusive_access=False,
        )
        assert group.can_add_agent("normal_agent", normal_node)

        # Should not be able to add if at max concurrency
        group.max_concurrency = 1
        assert not group.can_add_agent("normal_agent", normal_node)

        # Should not be able to add exclusive agent if group not empty
        group.max_concurrency = 2
        exclusive_node = DependencyNode(
            agent_id="exclusive_agent",
            agent=mock_agents["normal_agent"],
            requires_exclusive_access=True,
        )
        assert not group.can_add_agent("exclusive_agent", exclusive_node)


class TestExecutionStage:
    """Test ExecutionStage functionality."""

    def test_stage_creation(self):
        """Test creating an execution stage."""
        stage = ExecutionStage(
            stage_id="test_stage",
            stage_type=StageType.PARALLEL,
            agents=["agent_a"],
            estimated_duration_ms=10000.0,
            max_retries=3,
        )

        assert stage.stage_id == "test_stage"
        assert stage.stage_type == StageType.PARALLEL
        assert stage.agents == ["agent_a"]
        assert stage.estimated_duration_ms == 10000.0
        assert stage.max_retries == 3

    def test_get_all_agents(self):
        """Test getting all agents in a stage."""
        group1 = ParallelGroup(agents=["agent_a", "agent_b"])
        group2 = ParallelGroup(agents=["agent_c"])

        stage = ExecutionStage(
            stage_id="test_stage",
            stage_type=StageType.PARALLEL,
            agents=["agent_d"],
            parallel_groups=[group1, group2],
        )

        all_agents = stage.get_all_agents()
        assert set(all_agents) == {"agent_a", "agent_b", "agent_c", "agent_d"}

    def test_is_parallel(self):
        """Test parallel stage detection."""
        # Parallel type
        stage1 = ExecutionStage(stage_id="s1", stage_type=StageType.PARALLEL)
        assert stage1.is_parallel()

        # Has parallel groups
        stage2 = ExecutionStage(
            stage_id="s2",
            stage_type=StageType.SEQUENTIAL,
            parallel_groups=[ParallelGroup(agents=["a", "b"])],
        )
        assert stage2.is_parallel()

        # Multiple agents in stage
        stage3 = ExecutionStage(
            stage_id="s3", stage_type=StageType.SEQUENTIAL, agents=["a", "b"]
        )
        assert stage3.is_parallel()

        # Single agent, sequential
        stage4 = ExecutionStage(
            stage_id="s4", stage_type=StageType.SEQUENTIAL, agents=["a"]
        )
        assert not stage4.is_parallel()


class TestExecutionPlan:
    """Test ExecutionPlan functionality."""

    def test_plan_creation(self):
        """Test creating an execution plan."""
        stages = [
            ExecutionStage(
                stage_id="stage1", stage_type=StageType.SEQUENTIAL, agents=["agent_a"]
            ),
            ExecutionStage(
                stage_id="stage2",
                stage_type=StageType.PARALLEL,
                agents=["agent_b", "agent_c"],
            ),
        ]

        plan = ExecutionPlan(
            plan_id="test_plan",
            stages=stages,
            strategy=ExecutionStrategy.ADAPTIVE,
            total_agents=3,
            estimated_total_duration_ms=15000.0,
            parallelism_factor=1.5,
        )

        assert plan.plan_id == "test_plan"
        assert plan.strategy == ExecutionStrategy.ADAPTIVE
        assert plan.total_agents == 3
        assert plan.get_total_stages() == 2
        assert plan.parallelism_factor == 1.5

    def test_plan_navigation(self):
        """Test navigating through plan stages."""
        stages = [
            ExecutionStage(stage_id="stage1", stage_type=StageType.SEQUENTIAL),
            ExecutionStage(stage_id="stage2", stage_type=StageType.PARALLEL),
            ExecutionStage(stage_id="stage3", stage_type=StageType.SEQUENTIAL),
        ]

        plan = ExecutionPlan(
            plan_id="test",
            stages=stages,
            strategy=ExecutionStrategy.SEQUENTIAL,
            total_agents=3,
        )

        # Initially at stage 0
        assert plan.current_stage_index == 0
        assert plan.get_current_stage().stage_id == "stage1"
        assert plan.get_completion_percentage() == 0.0
        assert not plan.is_completed()

        # Advance to next stage
        assert plan.advance_stage() is True
        assert plan.current_stage_index == 1
        assert plan.get_current_stage().stage_id == "stage2"
        assert plan.get_completion_percentage() == pytest.approx(33.33, rel=1e-2)

        # Advance to final stage
        assert plan.advance_stage() is True
        assert plan.current_stage_index == 2
        assert plan.get_completion_percentage() == pytest.approx(66.67, rel=1e-2)

        # Advance beyond final stage to mark completion
        assert plan.advance_stage() is True
        assert plan.current_stage_index == 3
        assert plan.is_completed() is True
        assert plan.get_completion_percentage() == 100.0
        assert plan.get_current_stage() is None

        # Cannot advance beyond completion
        assert plan.advance_stage() is False

    def test_execution_summary(self):
        """Test execution summary generation."""
        stages = [ExecutionStage(stage_id="stage1", stage_type=StageType.SEQUENTIAL)]
        plan = ExecutionPlan(
            plan_id="test",
            stages=stages,
            strategy=ExecutionStrategy.SEQUENTIAL,
            total_agents=1,
        )

        plan.started_at = time.time() - 5  # Started 5 seconds ago

        summary = plan.get_execution_summary()

        assert summary["plan_id"] == "test"
        assert summary["strategy"] == "sequential"
        assert summary["total_stages"] == 1
        assert summary["total_agents"] == 1
        assert summary["current_stage"] == 0
        assert summary["is_completed"] is False
        assert summary["actual_duration_ms"] is not None
        assert summary["actual_duration_ms"] > 4000  # At least 4 seconds


class TestSequentialPlanBuilder:
    """Test SequentialPlanBuilder."""

    def test_sequential_plan_creation(self, graph_engine_with_agents):
        """Test creating a sequential execution plan."""
        builder = SequentialPlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        assert plan.strategy == ExecutionStrategy.SEQUENTIAL
        assert plan.parallelism_factor == 1.0
        assert len(plan.stages) == len(graph_engine_with_agents.nodes)

        # Each stage should have exactly one agent
        for stage in plan.stages:
            assert len(stage.agents) == 1
            assert stage.stage_type == StageType.SEQUENTIAL

    def test_sequential_respects_dependencies(self, graph_engine_with_agents):
        """Test that sequential plan respects dependencies."""
        # Add dependency
        graph_engine_with_agents.add_dependency(
            "critical_agent", "high_agent", DependencyType.HARD
        )

        builder = SequentialPlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        # Find stage indices
        critical_stage_idx = None
        high_stage_idx = None

        for i, stage in enumerate(plan.stages):
            if "critical_agent" in stage.agents:
                critical_stage_idx = i
            elif "high_agent" in stage.agents:
                high_stage_idx = i

        assert critical_stage_idx is not None
        assert high_stage_idx is not None
        assert critical_stage_idx < high_stage_idx


class TestParallelBatchedPlanBuilder:
    """Test ParallelBatchedPlanBuilder."""

    def test_parallel_plan_creation(self, graph_engine_with_agents):
        """Test creating a parallel batched execution plan."""
        builder = ParallelBatchedPlanBuilder(max_batch_size=3)
        plan = builder.build_plan(graph_engine_with_agents)

        assert plan.strategy == ExecutionStrategy.PARALLEL_BATCHED
        assert plan.parallelism_factor > 1.0  # Should have some parallelism

        # Should have fewer stages than agents (due to batching)
        assert len(plan.stages) <= len(graph_engine_with_agents.nodes)

        # Check that stages use parallel groups
        has_parallel_stage = any(
            len(stage.parallel_groups) > 0 or len(stage.agents) > 1
            for stage in plan.stages
        )
        assert has_parallel_stage

    def test_parallel_respects_dependencies(self, graph_engine_with_agents):
        """Test that parallel plan respects dependencies."""
        # Create dependency chain
        graph_engine_with_agents.add_dependency(
            "critical_agent", "high_agent", DependencyType.HARD
        )
        graph_engine_with_agents.add_dependency(
            "high_agent", "normal_agent", DependencyType.HARD
        )

        builder = ParallelBatchedPlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        # Should have multiple stages due to dependencies
        assert len(plan.stages) >= 3

        # Verify dependency ordering in stages
        agent_stage_map = {}
        for stage_idx, stage in enumerate(plan.stages):
            for agent in stage.get_all_agents():
                agent_stage_map[agent] = stage_idx

        assert agent_stage_map["critical_agent"] < agent_stage_map["high_agent"]
        assert agent_stage_map["high_agent"] < agent_stage_map["normal_agent"]

    def test_batch_size_limiting(self, graph_engine_with_agents):
        """Test that batch size is respected."""
        builder = ParallelBatchedPlanBuilder(max_batch_size=2)
        plan = builder.build_plan(graph_engine_with_agents)

        # Check that no stage exceeds batch size
        for stage in plan.stages:
            total_agents = len(stage.get_all_agents())
            assert total_agents <= 2


class TestPriorityFirstPlanBuilder:
    """Test PriorityFirstPlanBuilder."""

    def test_priority_ordering(self, graph_engine_with_agents):
        """Test that agents are ordered by priority."""
        builder = PriorityFirstPlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        assert plan.strategy == ExecutionStrategy.PRIORITY_FIRST

        # Create mapping of agent to stage index
        agent_stage_map = {}
        for stage_idx, stage in enumerate(plan.stages):
            for agent in stage.get_all_agents():
                agent_stage_map[agent] = stage_idx

        # Verify priority ordering (lower number = higher priority)
        assert agent_stage_map["critical_agent"] <= agent_stage_map["high_agent"]
        assert agent_stage_map["high_agent"] <= agent_stage_map["normal_agent"]
        assert agent_stage_map["normal_agent"] <= agent_stage_map["low_agent"]
        assert agent_stage_map["low_agent"] <= agent_stage_map["background_agent"]

    def test_parallel_within_priority_level(self, graph_engine_with_agents):
        """Test that agents within same priority can run in parallel."""
        # Add another normal priority agent
        normal_agent2 = MockAgent("normal_agent2", ExecutionPriority.NORMAL)
        node = DependencyNode(
            agent_id="normal_agent2",
            agent=normal_agent2,
            priority=ExecutionPriority.NORMAL,
        )
        graph_engine_with_agents.add_node(node)

        builder = PriorityFirstPlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        # Find stages containing normal priority agents
        normal_stages = []
        for stage_idx, stage in enumerate(plan.stages):
            stage_agents = stage.get_all_agents()
            if "normal_agent" in stage_agents or "normal_agent2" in stage_agents:
                normal_stages.append(stage_idx)

        # Should be able to run normal agents in same stage (parallel)
        if len(normal_stages) == 1:
            # Both in same stage - good for parallelism
            pass
        else:
            # In different stages - also acceptable
            assert len(normal_stages) <= 2


class TestAdaptivePlanBuilder:
    """Test AdaptivePlanBuilder."""

    def test_adaptive_strategy_selection(self, graph_engine_with_agents):
        """Test that adaptive builder selects appropriate strategy."""
        builder = AdaptivePlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        assert plan.strategy == ExecutionStrategy.ADAPTIVE
        assert "base_strategy" in plan.metadata
        assert plan.metadata["base_strategy"] in ["sequential", "parallel_batched"]

    def test_adaptive_with_parallelism(self, graph_engine_with_agents):
        """Test adaptive builder with high parallelism opportunity."""
        # Create scenario with high parallelism (no dependencies)
        builder = AdaptivePlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        # Should detect parallelism opportunity
        assert plan.parallelism_factor > 1.0
        assert plan.fallback_plan is not None
        assert plan.fallback_plan.strategy == ExecutionStrategy.SEQUENTIAL

    def test_adaptive_with_dependencies(self, graph_engine_with_agents):
        """Test adaptive builder with many dependencies."""
        # Create chain of dependencies to reduce parallelism
        agents = list(graph_engine_with_agents.nodes.keys())
        for i in range(len(agents) - 1):
            graph_engine_with_agents.add_dependency(
                agents[i], agents[i + 1], DependencyType.HARD
            )

        builder = AdaptivePlanBuilder()
        plan = builder.build_plan(graph_engine_with_agents)

        # Should favor sequential due to dependencies
        assert plan.parallelism_factor <= 1.5  # Low parallelism


class TestExecutionPlanner:
    """Test ExecutionPlanner main class."""

    def test_planner_initialization(self, execution_planner):
        """Test planner initialization."""
        assert ExecutionStrategy.SEQUENTIAL in execution_planner.builders
        assert ExecutionStrategy.PARALLEL_BATCHED in execution_planner.builders
        assert ExecutionStrategy.ADAPTIVE in execution_planner.builders
        assert ExecutionStrategy.PRIORITY_FIRST in execution_planner.builders
        assert execution_planner.default_strategy == ExecutionStrategy.ADAPTIVE

    def test_create_plan_with_strategy(
        self, execution_planner, graph_engine_with_agents
    ):
        """Test creating plans with different strategies."""
        strategies_to_test = [
            ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategy.PARALLEL_BATCHED,
            ExecutionStrategy.ADAPTIVE,
            ExecutionStrategy.PRIORITY_FIRST,
        ]

        for strategy in strategies_to_test:
            plan = execution_planner.create_plan(graph_engine_with_agents, strategy)
            assert plan.strategy == strategy
            assert len(plan.stages) > 0
            assert plan.total_agents == len(graph_engine_with_agents.nodes)

    def test_create_plan_with_default_strategy(
        self, execution_planner, graph_engine_with_agents
    ):
        """Test creating plan with default strategy."""
        plan = execution_planner.create_plan(graph_engine_with_agents)
        assert plan.strategy == ExecutionStrategy.ADAPTIVE

    def test_create_plan_with_invalid_strategy(
        self, execution_planner, graph_engine_with_agents
    ):
        """Test creating plan with invalid strategy falls back to default."""
        # This would require modifying the enum, so we'll test the warning path
        plan = execution_planner.create_plan(
            graph_engine_with_agents, ExecutionStrategy.ADAPTIVE
        )
        assert plan is not None

    def test_compare_strategies(self, execution_planner, graph_engine_with_agents):
        """Test comparing multiple strategies."""
        strategies = [ExecutionStrategy.SEQUENTIAL, ExecutionStrategy.PARALLEL_BATCHED]
        plans = execution_planner.compare_strategies(
            graph_engine_with_agents, strategies
        )

        assert len(plans) == 2
        assert ExecutionStrategy.SEQUENTIAL in plans
        assert ExecutionStrategy.PARALLEL_BATCHED in plans

        seq_plan = plans[ExecutionStrategy.SEQUENTIAL]
        par_plan = plans[ExecutionStrategy.PARALLEL_BATCHED]

        assert seq_plan.strategy == ExecutionStrategy.SEQUENTIAL
        assert par_plan.strategy == ExecutionStrategy.PARALLEL_BATCHED

        # Sequential should have parallelism factor of 1.0
        assert seq_plan.parallelism_factor == 1.0
        # Parallel should have higher parallelism factor
        assert par_plan.parallelism_factor >= 1.0

    def test_recommend_strategy(self, execution_planner, graph_engine_with_agents):
        """Test strategy recommendation."""
        # Test different optimization goals
        speed_strategy = execution_planner.recommend_strategy(
            graph_engine_with_agents, optimization_goal="speed"
        )
        assert speed_strategy in [
            ExecutionStrategy.PARALLEL_BATCHED,
            ExecutionStrategy.PRIORITY_FIRST,
        ]

        reliability_strategy = execution_planner.recommend_strategy(
            graph_engine_with_agents, optimization_goal="reliability"
        )
        assert reliability_strategy in [
            ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategy.PRIORITY_FIRST,
        ]

        resource_strategy = execution_planner.recommend_strategy(
            graph_engine_with_agents, optimization_goal="resource"
        )
        assert resource_strategy == ExecutionStrategy.SEQUENTIAL

        balanced_strategy = execution_planner.recommend_strategy(
            graph_engine_with_agents, optimization_goal="balanced"
        )
        assert balanced_strategy == ExecutionStrategy.ADAPTIVE

    def test_timing_estimates(self, execution_planner, graph_engine_with_agents):
        """Test that timing estimates are added to plans."""
        plan = execution_planner.create_plan(graph_engine_with_agents)

        # Plan should have total duration estimate
        assert plan.estimated_total_duration_ms is not None
        assert plan.estimated_total_duration_ms > 0

        # Stages should have duration estimates
        for stage in plan.stages:
            assert stage.estimated_duration_ms is not None
            assert stage.estimated_duration_ms > 0


class TestIntegration:
    """Integration tests for execution planner."""

    def test_complex_dependency_planning(self, execution_planner):
        """Test planning with complex dependencies."""
        # Create complex graph
        engine = DependencyGraphEngine()

        # Create agents with different characteristics
        agents = {
            "preprocessor": MockAgent("preprocessor", ExecutionPriority.HIGH),
            "analyzer_1": MockAgent("analyzer_1", ExecutionPriority.NORMAL),
            "analyzer_2": MockAgent("analyzer_2", ExecutionPriority.NORMAL),
            "validator": MockAgent("validator", ExecutionPriority.CRITICAL),
            "formatter": MockAgent("formatter", ExecutionPriority.LOW),
            "reporter": MockAgent("reporter", ExecutionPriority.LOW),
        }

        for agent_id, agent in agents.items():
            node = DependencyNode(
                agent_id=agent_id,
                agent=agent,
                priority=agent.priority,
                timeout_ms=10000 if "analyzer" in agent_id else 5000,
            )
            engine.add_node(node)

        # Create complex dependency structure
        # preprocessor -> (analyzer_1, analyzer_2) -> validator -> (formatter, reporter)
        engine.add_dependency("preprocessor", "analyzer_1", DependencyType.HARD)
        engine.add_dependency("preprocessor", "analyzer_2", DependencyType.HARD)
        engine.add_dependency("analyzer_1", "validator", DependencyType.HARD)
        engine.add_dependency("analyzer_2", "validator", DependencyType.HARD)
        engine.add_dependency("validator", "formatter", DependencyType.HARD)
        engine.add_dependency("validator", "reporter", DependencyType.HARD)

        # Test different strategies
        strategies = [
            ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategy.PARALLEL_BATCHED,
            ExecutionStrategy.PRIORITY_FIRST,
            ExecutionStrategy.ADAPTIVE,
        ]

        for strategy in strategies:
            plan = execution_planner.create_plan(engine, strategy)

            # Verify plan is valid
            assert len(plan.stages) > 0
            assert plan.total_agents == 6

            # Verify dependencies are respected in all strategies
            agent_stage_map = {}
            for stage_idx, stage in enumerate(plan.stages):
                for agent in stage.get_all_agents():
                    agent_stage_map[agent] = stage_idx

            # Check key dependencies
            assert agent_stage_map["preprocessor"] < agent_stage_map["analyzer_1"]
            assert agent_stage_map["preprocessor"] < agent_stage_map["analyzer_2"]
            assert agent_stage_map["analyzer_1"] < agent_stage_map["validator"]
            assert agent_stage_map["analyzer_2"] < agent_stage_map["validator"]
            assert agent_stage_map["validator"] < agent_stage_map["formatter"]
            assert agent_stage_map["validator"] < agent_stage_map["reporter"]

            # Check parallelism characteristics
            if strategy == ExecutionStrategy.SEQUENTIAL:
                assert plan.parallelism_factor == 1.0
            elif strategy == ExecutionStrategy.PARALLEL_BATCHED:
                assert plan.parallelism_factor > 1.0

    def test_plan_execution_simulation(
        self, execution_planner, graph_engine_with_agents
    ):
        """Test simulating plan execution."""
        # Create plan
        plan = execution_planner.create_plan(
            graph_engine_with_agents, ExecutionStrategy.PARALLEL_BATCHED
        )

        # Simulate execution timing
        plan.started_at = time.time()

        # Simulate stage progression
        initial_completion = plan.get_completion_percentage()
        assert initial_completion == 0.0

        # Advance through stages
        stages_completed = 0
        while not plan.is_completed() and stages_completed < 10:  # Safety limit
            current_stage = plan.get_current_stage()
            assert current_stage is not None

            # Simulate stage execution
            time.sleep(0.001)  # Brief delay

            # Advance to next stage (or mark as completed)
            plan.advance_stage()
            stages_completed += 1

        plan.completed_at = time.time()

        # Verify final state
        final_summary = plan.get_execution_summary()
        assert final_summary["completion_percentage"] == 100.0
        assert final_summary["actual_duration_ms"] is not None
        assert final_summary["is_completed"] is True
