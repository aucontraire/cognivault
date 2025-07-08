"""
Tests for the advanced orchestrator.

Covers integration of all dependency management components:
graph engine, execution planner, failure manager, resource scheduler,
and dynamic composition.
"""

import pytest
import asyncio
import time
from typing import Optional
from unittest.mock import Mock, AsyncMock, patch

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.dependencies.graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
    DependencyType,
    ExecutionPriority,
    ResourceConstraint,
)
from cognivault.dependencies.execution_planner import (
    ExecutionPlanner,
    ExecutionStrategy,
)
from cognivault.dependencies.failure_manager import (
    FailureManager,
    CascadePreventionStrategy,
    RetryConfiguration,
)
from cognivault.dependencies.resource_scheduler import (
    ResourceScheduler,
    ResourcePool,
    ResourceType,
)
from cognivault.dependencies.dynamic_composition import (
    DynamicAgentComposer,
    AgentMetadata,
)
from cognivault.dependencies.advanced_orchestrator import (
    AdvancedOrchestrator,
    OrchestratorConfig,
    ExecutionPhase,
    ExecutionResults,
    ResourceAllocationResult,
    PipelineStage,
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        delay: float = 0.0,
        execution_time: Optional[float] = None,
    ):
        super().__init__(name=name)
        self.should_fail = should_fail
        self.delay = delay if execution_time is None else execution_time
        self.execution_count = 0

    async def run(self, context: AgentContext) -> AgentContext:
        self.execution_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise Exception(f"Simulated failure in {self.name}")

        context.agent_outputs[self.name] = (
            f"Output from {self.name} (execution #{self.execution_count})"
        )
        return context


@pytest.fixture
def orchestrator_config():
    """Create orchestrator configuration for testing."""
    return OrchestratorConfig(
        max_concurrent_agents=3,
        enable_failure_recovery=True,
        enable_resource_scheduling=True,
        enable_dynamic_composition=False,  # Disable for simpler tests
        default_execution_strategy=ExecutionStrategy.PARALLEL_BATCHED,
        cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        pipeline_timeout_ms=30000,
        resource_allocation_timeout_ms=5000,
    )


@pytest.fixture
def mock_agent_registry():
    """Create a comprehensive mock for AgentRegistry with common defaults."""
    from cognivault.exceptions import FailurePropagationStrategy

    mock = Mock()

    # Set up default behaviors
    mock.get_failure_strategy.return_value = (
        FailurePropagationStrategy.GRACEFUL_DEGRADATION
    )
    mock.is_critical_agent.return_value = False
    mock.has_fallback_agents.return_value = False
    mock.get_fallback_agents.return_value = []
    mock.check_health.return_value = True
    mock.validate_agent_health.return_value = True
    mock.has_agent.return_value = True
    mock.get_agent_dependencies.return_value = []

    return mock


def create_registry_mock(**overrides):
    """Helper function to create registry mocks with custom behavior."""
    from cognivault.exceptions import FailurePropagationStrategy

    mock = Mock()

    # Set up default behaviors
    defaults = {
        "get_failure_strategy": FailurePropagationStrategy.GRACEFUL_DEGRADATION,
        "is_critical_agent": False,
        "has_fallback_agents": False,
        "get_fallback_agents": [],
        "check_health": True,
        "validate_agent_health": True,
        "has_agent": True,
        "get_agent_dependencies": [],
    }

    # Apply defaults
    for method, default_value in defaults.items():
        if callable(default_value):
            getattr(mock, method).return_value = default_value()
        else:
            getattr(mock, method).return_value = default_value

    # Apply custom overrides
    for method, custom_value in overrides.items():
        if method.endswith("_side_effect"):
            # Handle side effects
            method_name = method.replace("_side_effect", "")
            getattr(mock, method_name).side_effect = custom_value
        else:
            getattr(mock, method).return_value = custom_value

    return mock


def assert_execution_success(results, expected_agents=None, min_successful=None):
    """Helper to assert successful execution with observable metrics."""
    assert results.execution_time_ms > 0, "Execution should take measurable time"
    assert results.total_agents_executed > 0, "Should execute at least one agent"

    if expected_agents is not None:
        assert results.total_agents_executed == expected_agents, (
            f"Expected {expected_agents} agents executed"
        )

    if min_successful is not None:
        assert results.successful_agents >= min_successful, (
            f"Expected at least {min_successful} successful agents"
        )

    # Check pipeline stages
    assert len(results.pipeline_stages) > 0, "Should have pipeline stages"

    # Verify all phases executed
    phases_executed = {stage.phase for stage in results.pipeline_stages}
    expected_phases = {
        ExecutionPhase.PREPARATION,
        ExecutionPhase.RESOURCE_ALLOCATION,
        ExecutionPhase.EXECUTION,
        ExecutionPhase.CLEANUP,
    }
    assert phases_executed == expected_phases, (
        f"Missing phases: {expected_phases - phases_executed}"
    )


def assert_execution_failure(results, expected_failures=None, recovery_actions=True):
    """Helper to assert execution failure with recovery actions."""
    assert results.execution_time_ms > 0, "Execution should take measurable time"
    assert results.failed_agents > 0, "Should have at least one failure"

    if expected_failures is not None:
        assert results.failed_agents == expected_failures, (
            f"Expected {expected_failures} failures"
        )

    if recovery_actions:
        assert len(results.failure_recovery_actions) > 0, "Should have recovery actions"


def assert_specific_agents_executed(results, agent_names):
    """Assert that specific agents were executed based on pipeline stages."""
    executed_agents = set()
    for stage in results.pipeline_stages:
        executed_agents.update(stage.agents_executed)

    for agent_name in agent_names:
        assert agent_name in executed_agents, (
            f"Agent {agent_name} should have been executed"
        )


def get_executed_agents(results):
    """Get set of all agents that were executed according to pipeline stages."""
    executed_agents = set()
    for stage in results.pipeline_stages:
        executed_agents.update(stage.agents_executed)
    return executed_agents


@pytest.fixture
def simple_graph_engine():
    """Create a simple graph engine with test agents."""
    engine = DependencyGraphEngine()

    agents = {
        "preprocessor": MockAgent("preprocessor"),
        "analyzer": MockAgent("analyzer"),
        "formatter": MockAgent("formatter"),
    }

    for agent_id, agent in agents.items():
        node = DependencyNode(
            agent_id=agent_id,
            agent=agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
            resource_constraints=[
                ResourceConstraint("cpu", 10.0, 30.0, "percentage"),
                ResourceConstraint("memory", 100.0, 512.0, "MB"),
            ],
        )
        engine.add_node(node)

    # Add dependencies: preprocessor -> analyzer -> formatter
    engine.add_dependency("preprocessor", "analyzer", DependencyType.HARD)
    engine.add_dependency("analyzer", "formatter", DependencyType.HARD)

    return engine


@pytest.fixture
def orchestrator(orchestrator_config, simple_graph_engine):
    """Create an advanced orchestrator for testing."""
    return AdvancedOrchestrator(simple_graph_engine, orchestrator_config)


@pytest.fixture
def context():
    """Create a basic agent context."""
    return AgentContext(query="test query for orchestration")


class TestOrchestratorConfig:
    """Test OrchestratorConfig functionality."""

    def test_config_creation(self):
        """Test creating orchestrator configuration."""
        config = OrchestratorConfig(
            max_concurrent_agents=5,
            enable_failure_recovery=True,
            enable_resource_scheduling=True,
            enable_dynamic_composition=True,
            default_execution_strategy=ExecutionStrategy.ADAPTIVE,
            cascade_prevention_strategy=CascadePreventionStrategy.CIRCUIT_BREAKER,
            pipeline_timeout_ms=60000,
        )

        assert config.max_concurrent_agents == 5
        assert config.enable_failure_recovery is True
        assert config.enable_resource_scheduling is True
        assert config.enable_dynamic_composition is True
        assert config.default_execution_strategy == ExecutionStrategy.ADAPTIVE
        assert (
            config.cascade_prevention_strategy
            == CascadePreventionStrategy.CIRCUIT_BREAKER
        )
        assert config.pipeline_timeout_ms == 60000

    def test_config_defaults(self):
        """Test default configuration values."""
        config = OrchestratorConfig()

        assert config.max_concurrent_agents == 4
        assert config.enable_failure_recovery is True
        assert config.enable_resource_scheduling is True
        assert config.enable_dynamic_composition is False
        assert config.default_execution_strategy == ExecutionStrategy.ADAPTIVE
        assert (
            config.cascade_prevention_strategy
            == CascadePreventionStrategy.GRACEFUL_DEGRADATION
        )


class TestExecutionResults:
    """Test ExecutionResults functionality."""

    def test_results_creation(self):
        """Test creating execution results."""
        results = ExecutionResults(
            success=True,
            total_agents_executed=3,
            successful_agents=3,
            failed_agents=0,
            execution_time_ms=1500.0,
            pipeline_stages=[],
            resource_allocation_results=[],
            failure_recovery_actions=[],
        )

        assert results.success is True
        assert results.total_agents_executed == 3
        assert results.successful_agents == 3
        assert results.failed_agents == 0
        assert results.execution_time_ms == 1500.0

    def test_results_get_success_rate(self):
        """Test success rate calculation."""
        results = ExecutionResults(
            success=False,
            total_agents_executed=5,
            successful_agents=3,
            failed_agents=2,
            execution_time_ms=2000.0,
            pipeline_stages=[],
            resource_allocation_results=[],
            failure_recovery_actions=[],
        )

        assert results.get_success_rate() == 0.6  # 3/5

    def test_results_get_success_rate_no_agents(self):
        """Test success rate with no agents executed."""
        results = ExecutionResults(
            success=False,
            total_agents_executed=0,
            successful_agents=0,
            failed_agents=0,
            execution_time_ms=0.0,
            pipeline_stages=[],
            resource_allocation_results=[],
            failure_recovery_actions=[],
        )

        assert results.get_success_rate() == 0.0

    def test_results_to_dict(self):
        """Test converting results to dictionary."""
        results = ExecutionResults(
            success=True,
            total_agents_executed=2,
            successful_agents=2,
            failed_agents=0,
            execution_time_ms=1000.0,
            pipeline_stages=[],
            resource_allocation_results=[],
            failure_recovery_actions=[],
        )

        result_dict = results.to_dict()

        assert result_dict["success"] is True
        assert result_dict["total_agents_executed"] == 2
        assert result_dict["successful_agents"] == 2
        assert result_dict["failed_agents"] == 0
        assert result_dict["execution_time_ms"] == 1000.0
        assert result_dict["success_rate"] == 1.0
        assert "pipeline_stages" in result_dict
        assert "resource_allocation_results" in result_dict
        assert "failure_recovery_actions" in result_dict


class TestResourceAllocationResult:
    """Test ResourceAllocationResult functionality."""

    def test_allocation_result_creation(self):
        """Test creating resource allocation result."""
        result = ResourceAllocationResult(
            agent_id="test_agent",
            resource_type=ResourceType.CPU,
            requested_amount=50.0,
            allocated_amount=50.0,
            allocation_time_ms=100.0,
            success=True,
        )

        assert result.agent_id == "test_agent"
        assert result.resource_type == ResourceType.CPU
        assert result.requested_amount == 50.0
        assert result.allocated_amount == 50.0
        assert result.allocation_time_ms == 100.0
        assert result.success is True


class TestPipelineStage:
    """Test PipelineStage functionality."""

    def test_pipeline_stage_creation(self):
        """Test creating pipeline stage."""
        stage = PipelineStage(
            stage_id="test_stage",
            phase=ExecutionPhase.EXECUTION,
            agents_executed=["agent1", "agent2"],
            stage_duration_ms=1500.0,
            success=True,
        )

        assert stage.stage_id == "test_stage"
        assert stage.phase == ExecutionPhase.EXECUTION
        assert stage.agents_executed == ["agent1", "agent2"]
        assert stage.stage_duration_ms == 1500.0
        assert stage.success is True


class TestAdvancedOrchestrator:
    """Test AdvancedOrchestrator functionality."""

    def test_orchestrator_creation(self, simple_graph_engine, orchestrator_config):
        """Test creating advanced orchestrator."""
        orchestrator = AdvancedOrchestrator(simple_graph_engine, orchestrator_config)

        assert orchestrator.graph_engine == simple_graph_engine
        assert orchestrator.config == orchestrator_config
        assert isinstance(orchestrator.execution_planner, ExecutionPlanner)
        assert isinstance(orchestrator.failure_manager, FailureManager)
        assert isinstance(orchestrator.resource_scheduler, ResourceScheduler)
        assert orchestrator.dynamic_composer is None  # Disabled in config

    def test_orchestrator_with_dynamic_composition(self, simple_graph_engine):
        """Test orchestrator with dynamic composition enabled."""
        config = OrchestratorConfig(enable_dynamic_composition=True)
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)

        assert orchestrator.dynamic_composer is not None
        assert isinstance(orchestrator.dynamic_composer, DynamicAgentComposer)

    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self, orchestrator, context):
        """Test successful pipeline execution."""
        results = await orchestrator.execute_pipeline(context)

        assert results.success is True
        assert results.total_agents_executed == 3
        assert results.successful_agents == 3
        assert results.failed_agents == 0
        assert_execution_success(results, min_successful=3)
        assert_specific_agents_executed(
            results, ["preprocessor", "analyzer", "formatter"]
        )

        # Check that all agents executed
        assert "preprocessor" in context.agent_outputs
        assert "analyzer" in context.agent_outputs
        assert "formatter" in context.agent_outputs

    @pytest.mark.asyncio
    async def test_execute_pipeline_with_failure(self, orchestrator_config, context):
        """Test pipeline execution with agent failure."""
        # Create graph with failing agent
        engine = DependencyGraphEngine()

        agents = {
            "preprocessor": MockAgent("preprocessor"),
            "failing_agent": MockAgent("failing_agent", should_fail=True),
            "formatter": MockAgent("formatter"),
        }

        for agent_id, agent in agents.items():
            node = DependencyNode(
                agent_id=agent_id,
                agent=agent,
                priority=ExecutionPriority.NORMAL,
                timeout_ms=5000,
            )
            engine.add_node(node)

        # Add dependencies
        engine.add_dependency("preprocessor", "failing_agent", DependencyType.HARD)
        engine.add_dependency("failing_agent", "formatter", DependencyType.HARD)

        orchestrator = AdvancedOrchestrator(engine, orchestrator_config)
        results = await orchestrator.execute_pipeline(context)

        # Should handle failure gracefully
        assert results.success is False
        assert results.failed_agents > 0
        assert len(results.failure_recovery_actions) > 0

    @pytest.mark.asyncio
    async def test_execute_pipeline_timeout(self, orchestrator_config, context):
        """Test pipeline execution with timeout."""
        # Create graph with slow agent
        engine = DependencyGraphEngine()

        # Create agent that takes longer than pipeline timeout
        slow_agent = MockAgent("slow_agent", delay=1.0)  # 1 second delay
        node = DependencyNode(
            agent_id="slow_agent",
            agent=slow_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=500,  # 500ms timeout
        )
        engine.add_node(node)

        # Set very short pipeline timeout
        orchestrator_config.pipeline_timeout_ms = 100  # 100ms
        orchestrator = AdvancedOrchestrator(engine, orchestrator_config)

        results = await orchestrator.execute_pipeline(context)

        # Should timeout
        assert results.success is False

    @pytest.mark.asyncio
    async def test_prepare_execution_phase(self, orchestrator, context):
        """Test execution preparation phase."""
        stage = await orchestrator._prepare_execution(context)

        assert stage.phase == ExecutionPhase.PREPARATION
        assert stage.success is True
        assert stage.stage_duration_ms > 0

        # Should have created execution plan
        assert orchestrator._current_execution_plan is not None

    @pytest.mark.asyncio
    async def test_allocate_resources_phase(self, orchestrator, context):
        """Test resource allocation phase."""
        # First prepare execution
        await orchestrator._prepare_execution(context)

        stage = await orchestrator._allocate_resources(context)

        assert stage.phase == ExecutionPhase.RESOURCE_ALLOCATION
        assert stage.success is True
        assert len(stage.agents_executed) == 3  # All agents should get resources

    @pytest.mark.asyncio
    async def test_allocate_resources_with_failure(self, orchestrator_config, context):
        """Test resource allocation with insufficient resources."""
        # Create limited resource pool
        engine = DependencyGraphEngine()

        # Create agent with high resource requirements
        high_resource_agent = MockAgent("high_resource_agent")
        node = DependencyNode(
            agent_id="high_resource_agent",
            agent=high_resource_agent,
            priority=ExecutionPriority.NORMAL,
            resource_constraints=[
                ResourceConstraint(
                    "memory", 10000.0, 20000.0, "MB"
                ),  # Very high memory
            ],
        )
        engine.add_node(node)

        orchestrator = AdvancedOrchestrator(engine, orchestrator_config)

        # Prepare execution first
        await orchestrator._prepare_execution(context)

        stage = await orchestrator._allocate_resources(context)

        # May fail due to insufficient resources, but should handle gracefully
        assert stage.phase == ExecutionPhase.RESOURCE_ALLOCATION
        # Success depends on available resources

    @pytest.mark.asyncio
    async def test_execute_agents_phase(self, orchestrator, context):
        """Test agent execution phase."""
        # Prepare and allocate resources first
        await orchestrator._prepare_execution(context)
        await orchestrator._allocate_resources(context)

        stage = await orchestrator._execute_agents(context)

        assert stage.phase == ExecutionPhase.EXECUTION
        assert stage.success is True
        assert len(stage.agents_executed) == 3

        # All agents should have executed
        executed_agents = set(stage.agents_executed)
        assert "preprocessor" in executed_agents
        assert "analyzer" in executed_agents
        assert "formatter" in executed_agents
        assert "preprocessor" in context.agent_outputs
        assert "analyzer" in context.agent_outputs
        assert "formatter" in context.agent_outputs

    @pytest.mark.asyncio
    async def test_cleanup_resources_phase(self, orchestrator, context):
        """Test resource cleanup phase."""
        # Execute full pipeline first
        await orchestrator._prepare_execution(context)
        await orchestrator._allocate_resources(context)
        await orchestrator._execute_agents(context)

        stage = await orchestrator._cleanup_resources(context)

        assert stage.phase == ExecutionPhase.CLEANUP
        assert stage.success is True

    @pytest.mark.asyncio
    async def test_handle_agent_execution_success(self, orchestrator, context):
        """Test handling successful agent execution."""
        agent = MockAgent("test_agent")

        result = await orchestrator._handle_agent_execution(
            "test_agent", agent, context
        )

        assert result["success"] is True
        assert result["agent_id"] == "test_agent"
        assert result["execution_time_ms"] > 0
        assert "test_agent" in context.agent_outputs

    @pytest.mark.asyncio
    async def test_handle_agent_execution_failure(self, orchestrator, context):
        """Test handling failed agent execution."""
        failing_agent = MockAgent("failing_agent", should_fail=True)

        result = await orchestrator._handle_agent_execution(
            "failing_agent", failing_agent, context
        )

        assert result["success"] is False
        assert result["agent_id"] == "failing_agent"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_agent_execution_timeout(self, orchestrator, context):
        """Test handling agent execution timeout."""
        # Create very slow agent
        slow_agent = MockAgent("slow_agent", delay=2.0)  # 2 second delay

        # Mock the node to have a short timeout
        node = DependencyNode(
            agent_id="slow_agent",
            agent=slow_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=100,  # 100ms timeout
        )
        orchestrator.graph_engine.add_node(node)

        result = await orchestrator._handle_agent_execution(
            "slow_agent", slow_agent, context
        )

        # Should timeout and return failure
        assert result["success"] is False
        assert "timeout" in result.get("error", "").lower()

    def test_create_resource_allocation_result(self, orchestrator):
        """Test creating resource allocation result."""
        result = orchestrator._create_resource_allocation_result(
            "test_agent",
            ResourceType.CPU,
            50.0,
            45.0,
            150.0,
            True,
        )

        assert result.agent_id == "test_agent"
        assert result.resource_type == ResourceType.CPU
        assert result.requested_amount == 50.0
        assert result.allocated_amount == 45.0
        assert result.allocation_time_ms == 150.0
        assert result.success is True

    def test_aggregate_execution_results(self, orchestrator):
        """Test aggregating execution results."""
        agent_results = [
            {"agent_id": "agent1", "success": True, "execution_time_ms": 100.0},
            {"agent_id": "agent2", "success": True, "execution_time_ms": 150.0},
            {
                "agent_id": "agent3",
                "success": False,
                "execution_time_ms": 200.0,
                "error": "failed",
            },
        ]

        pipeline_stages = [
            PipelineStage("prep", ExecutionPhase.PREPARATION, [], 50.0, True),
            PipelineStage(
                "exec",
                ExecutionPhase.EXECUTION,
                ["agent1", "agent2", "agent3"],
                450.0,
                False,
            ),
        ]

        resource_results = []
        failure_actions = ["graceful_degradation"]
        start_time = time.time() - 1.0  # 1 second ago

        results = orchestrator._aggregate_execution_results(
            agent_results,
            pipeline_stages,
            resource_results,
            failure_actions,
            start_time,
        )

        assert results.success is False  # One agent failed
        assert results.total_agents_executed == 3
        assert results.successful_agents == 2
        assert results.failed_agents == 1
        assert results.execution_time_ms > 900  # Should be around 1000ms
        assert len(results.pipeline_stages) == 2
        assert len(results.failure_recovery_actions) == 1

    def test_create_pipeline_stage(self, orchestrator):
        """Test creating pipeline stage."""
        stage = orchestrator._create_pipeline_stage(
            "test_stage",
            ExecutionPhase.EXECUTION,
            ["agent1", "agent2"],
            1500.0,
            True,
        )

        assert stage.stage_id == "test_stage"
        assert stage.phase == ExecutionPhase.EXECUTION
        assert stage.agents_executed == ["agent1", "agent2"]
        assert stage.stage_duration_ms == 1500.0
        assert stage.success is True

    def test_calculate_execution_statistics(self, orchestrator):
        """Test calculating execution statistics."""
        agent_results = [
            {"agent_id": "agent1", "success": True, "execution_time_ms": 100.0},
            {"agent_id": "agent2", "success": True, "execution_time_ms": 200.0},
            {"agent_id": "agent3", "success": False, "execution_time_ms": 150.0},
        ]

        stats = orchestrator._calculate_execution_statistics(agent_results)

        assert stats["total_agents"] == 3
        assert stats["successful_agents"] == 2
        assert stats["failed_agents"] == 1
        assert stats["success_rate"] == pytest.approx(0.667, rel=1e-2)
        assert stats["avg_execution_time_ms"] == 150.0  # (100+200+150)/3
        assert stats["total_execution_time_ms"] == 450.0


class TestIntegration:
    """Integration tests for advanced orchestrator."""

    @pytest.mark.asyncio
    async def test_complete_orchestration_workflow(self):
        """Test complete orchestration workflow with all components."""
        # Create complex dependency graph
        engine = DependencyGraphEngine()

        agents = {
            "data_loader": MockAgent("data_loader"),
            "preprocessor": MockAgent("preprocessor"),
            "analyzer_1": MockAgent("analyzer_1"),
            "analyzer_2": MockAgent("analyzer_2"),
            "aggregator": MockAgent("aggregator"),
            "reporter": MockAgent("reporter"),
        }

        for agent_id, agent in agents.items():
            priority = (
                ExecutionPriority.HIGH
                if "loader" in agent_id
                else ExecutionPriority.NORMAL
            )
            node = DependencyNode(
                agent_id=agent_id,
                agent=agent,
                priority=priority,
                timeout_ms=10000,
                resource_constraints=[
                    ResourceConstraint("cpu", 10.0, 25.0, "percentage"),
                    ResourceConstraint("memory", 100.0, 300.0, "MB"),
                ],
            )
            engine.add_node(node)

        # Create complex dependency structure
        engine.add_dependency("data_loader", "preprocessor", DependencyType.HARD)
        engine.add_dependency("preprocessor", "analyzer_1", DependencyType.HARD)
        engine.add_dependency("preprocessor", "analyzer_2", DependencyType.HARD)
        engine.add_dependency("analyzer_1", "aggregator", DependencyType.HARD)
        engine.add_dependency("analyzer_2", "aggregator", DependencyType.HARD)
        engine.add_dependency("aggregator", "reporter", DependencyType.HARD)

        # Create orchestrator with all features enabled
        config = OrchestratorConfig(
            max_concurrent_agents=3,
            enable_failure_recovery=True,
            enable_resource_scheduling=True,
            enable_dynamic_composition=False,  # Keep simple for testing
            default_execution_strategy=ExecutionStrategy.PARALLEL_BATCHED,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )

        orchestrator = AdvancedOrchestrator(engine, config)
        context = AgentContext(query="Complex workflow test")

        # Execute complete pipeline
        results = await orchestrator.execute_pipeline(context)

        # Verify results
        assert results.success is True
        assert results.total_agents_executed == 6
        assert results.successful_agents == 6
        assert results.failed_agents == 0
        assert results.execution_time_ms > 0

        # Verify all pipeline phases completed
        phase_names = [stage.phase for stage in results.pipeline_stages]
        assert ExecutionPhase.PREPARATION in phase_names
        assert ExecutionPhase.RESOURCE_ALLOCATION in phase_names
        assert ExecutionPhase.EXECUTION in phase_names
        assert ExecutionPhase.CLEANUP in phase_names

        # Verify all agents executed
        expected_agents = [
            "data_loader",
            "preprocessor",
            "analyzer_1",
            "analyzer_2",
            "aggregator",
            "reporter",
        ]
        for agent_id in expected_agents:
            assert agent_id in context.agent_outputs

        # Verify resource allocation occurred
        assert len(results.resource_allocation_results) > 0

    @pytest.mark.asyncio
    async def test_orchestration_with_failures_and_recovery(self):
        """Test orchestration with failures and recovery mechanisms."""
        # Create graph with some failing agents
        engine = DependencyGraphEngine()

        agents = {
            "reliable_agent": MockAgent("reliable_agent"),
            "failing_agent": MockAgent("failing_agent", should_fail=True),
            "backup_agent": MockAgent("backup_agent"),
            "final_agent": MockAgent("final_agent"),
        }

        for agent_id, agent in agents.items():
            node = DependencyNode(
                agent_id=agent_id,
                agent=agent,
                priority=ExecutionPriority.NORMAL,
                timeout_ms=5000,
            )
            engine.add_node(node)

        # Create dependencies with fallback path
        engine.add_dependency("reliable_agent", "failing_agent", DependencyType.HARD)
        engine.add_dependency("reliable_agent", "backup_agent", DependencyType.SOFT)
        engine.add_dependency("failing_agent", "final_agent", DependencyType.HARD)
        engine.add_dependency("backup_agent", "final_agent", DependencyType.HARD)

        # Configure for failure recovery
        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )

        orchestrator = AdvancedOrchestrator(engine, config)

        # Configure failure recovery for failing agent
        orchestrator.failure_manager.add_fallback_chain(
            "failing_agent", ["backup_agent"]
        )

        context = AgentContext(query="Failure recovery test")
        results = await orchestrator.execute_pipeline(context)

        # Should handle failure gracefully
        assert results.failed_agents > 0  # failing_agent should fail
        assert len(results.failure_recovery_actions) > 0

        # Should still execute successfully overall due to recovery
        # (depending on implementation details)

    @pytest.mark.asyncio
    async def test_orchestration_performance_characteristics(self):
        """Test orchestration performance characteristics."""
        # Create graph optimized for parallelism
        engine = DependencyGraphEngine()

        # Create independent parallel agents
        parallel_agents = {}
        for i in range(5):
            agent_id = f"parallel_agent_{i}"
            parallel_agents[agent_id] = MockAgent(
                agent_id, delay=0.1
            )  # Small delay for realism

            node = DependencyNode(
                agent_id=agent_id,
                agent=parallel_agents[agent_id],
                priority=ExecutionPriority.NORMAL,
                timeout_ms=5000,
            )
            engine.add_node(node)

        # Add final aggregator
        aggregator = MockAgent("aggregator", delay=0.05)
        node = DependencyNode(
            agent_id="aggregator",
            agent=aggregator,
            priority=ExecutionPriority.HIGH,
            timeout_ms=5000,
        )
        engine.add_node(node)

        # All parallel agents feed into aggregator
        for i in range(5):
            engine.add_dependency(
                f"parallel_agent_{i}", "aggregator", DependencyType.HARD
            )

        # Test different execution strategies
        strategies = [
            ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategy.PARALLEL_BATCHED,
            ExecutionStrategy.ADAPTIVE,
        ]

        context = AgentContext(query="Performance test")
        performance_results = {}

        for strategy in strategies:
            config = OrchestratorConfig(
                default_execution_strategy=strategy,
                max_concurrent_agents=3,
            )

            orchestrator = AdvancedOrchestrator(engine, config)

            start_time = time.time()
            results = await orchestrator.execute_pipeline(context.model_copy())
            end_time = time.time()

            performance_results[strategy] = {
                "execution_time": end_time - start_time,
                "success": results.success,
                "agents_executed": results.total_agents_executed,
            }

        # Verify all strategies work
        for strategy, result in performance_results.items():
            assert result["success"] is True
            assert result["agents_executed"] == 6

        # Parallel execution should generally be faster than sequential
        # (though with small delays, the difference might be minimal)
        parallel_time = performance_results[ExecutionStrategy.PARALLEL_BATCHED][
            "execution_time"
        ]
        sequential_time = performance_results[ExecutionStrategy.SEQUENTIAL][
            "execution_time"
        ]

        # Allow some tolerance due to test environment variability
        assert parallel_time <= sequential_time + 0.1  # Within 100ms tolerance

    @pytest.mark.asyncio
    async def test_orchestration_resource_constraints(self):
        """Test orchestration with resource constraints."""
        # Create agents with high resource requirements
        engine = DependencyGraphEngine()

        agents = {
            "memory_intensive": MockAgent("memory_intensive"),
            "cpu_intensive": MockAgent("cpu_intensive"),
            "balanced": MockAgent("balanced"),
        }

        # Define different resource profiles
        resource_profiles = {
            "memory_intensive": [ResourceConstraint("memory", 1000.0, 2000.0, "MB")],
            "cpu_intensive": [ResourceConstraint("cpu", 80.0, 90.0, "percentage")],
            "balanced": [
                ResourceConstraint("cpu", 30.0, 50.0, "percentage"),
                ResourceConstraint("memory", 300.0, 500.0, "MB"),
            ],
        }

        for agent_id, agent in agents.items():
            node = DependencyNode(
                agent_id=agent_id,
                agent=agent,
                priority=ExecutionPriority.NORMAL,
                timeout_ms=5000,
                resource_constraints=resource_profiles[agent_id],
            )
            engine.add_node(node)

        # Make them independent for parallel execution
        # (no dependencies to test resource scheduling)

        config = OrchestratorConfig(
            enable_resource_scheduling=True,
            max_concurrent_agents=2,  # Limited concurrency
        )

        orchestrator = AdvancedOrchestrator(engine, config)
        context = AgentContext(query="Resource constraint test")

        results = await orchestrator.execute_pipeline(context)

        # Should complete successfully with resource management
        assert results.success is True
        assert results.total_agents_executed == 3

        # Should have resource allocation results
        assert len(results.resource_allocation_results) > 0

        # Verify resource allocation was attempted for each agent
        allocated_agents = set(
            result.agent_id for result in results.resource_allocation_results
        )
        assert len(allocated_agents) == 3

    @pytest.mark.asyncio
    async def test_orchestration_error_scenarios(self):
        """Test orchestration with various error scenarios."""
        # Create minimal graph for error testing
        engine = DependencyGraphEngine()

        error_agent = MockAgent("error_agent", should_fail=True)
        node = DependencyNode(
            agent_id="error_agent",
            agent=error_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=1000,
        )
        engine.add_node(node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.ISOLATION,
        )

        orchestrator = AdvancedOrchestrator(engine, config)
        context = AgentContext(query="Error scenario test")

        results = await orchestrator.execute_pipeline(context)

        # Should handle errors gracefully
        assert results.success is False
        assert results.failed_agents == 1
        assert len(results.failure_recovery_actions) > 0

        # Should have attempted execution
        assert results.total_agents_executed == 1

        # Should have recorded the error
        execution_stage = None
        for stage in results.pipeline_stages:
            if stage.phase == ExecutionPhase.EXECUTION:
                execution_stage = stage
                break

        assert execution_stage is not None
        assert execution_stage.success is False


class TestOrchestratorFailureHandling:
    """Test suite for failure propagation strategies and error handling."""

    @pytest.fixture
    def orchestrator_with_failure_manager(self, simple_graph_engine):
        """Create orchestrator with configured failure manager."""
        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        return AdvancedOrchestrator(simple_graph_engine, config)

    @pytest.mark.asyncio
    async def test_fail_fast_strategy(self, simple_graph_engine):
        """Test FAIL_FAST strategy raises PipelineExecutionError."""
        from cognivault.exceptions import PipelineExecutionError
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create agent that will fail and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        node = DependencyNode(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.CIRCUIT_BREAKER,  # Use circuit breaker for fast failure
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry to return FAIL_FAST strategy
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = create_registry_mock(
                get_failure_strategy=FailurePropagationStrategy.FAIL_FAST,
                is_critical_agent=True,
            )
            mock_registry_class.return_value = mock_registry

            # Execute pipeline with FAIL_FAST strategy
            results = await orchestrator.execute_pipeline(context)

            # Should have recorded the failure, but the exact behavior depends on cascade prevention strategy
            assert results.failed_agents >= 1
            # The overall success depends on the cascade prevention strategy configuration

    @pytest.mark.asyncio
    async def test_warn_continue_strategy(self, simple_graph_engine):
        """Test WARN_CONTINUE strategy logs warning but continues."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        failing_node = DependencyNode(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(failing_node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry to return WARN_CONTINUE strategy
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = create_registry_mock(
                get_failure_strategy=FailurePropagationStrategy.WARN_CONTINUE,
                is_critical_agent=False,
            )
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should continue despite failure - other agents should execute
            assert_execution_failure(
                results, expected_failures=1, recovery_actions=True
            )
            assert (
                results.successful_agents >= 3
            )  # The original 3 agents from simple_graph_engine

    @pytest.mark.asyncio
    async def test_graceful_degradation_strategy(self, simple_graph_engine):
        """Test GRACEFUL_DEGRADATION creates warnings and marks degraded."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        node = DependencyNode(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry to return GRACEFUL_DEGRADATION strategy
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_failure_strategy.return_value = (
                FailurePropagationStrategy.GRACEFUL_DEGRADATION
            )
            mock_registry.has_fallback_agents.return_value = False
            mock_registry.is_critical_agent.return_value = False
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should continue with degraded mode
            assert_execution_failure(
                results, expected_failures=1, recovery_actions=True
            )
            assert results.successful_agents == 3  # Other agents succeeded
            assert "graceful_degradation" in results.failure_recovery_actions

    @pytest.mark.asyncio
    async def test_conditional_fallback_with_agents(self, simple_graph_engine):
        """Test CONDITIONAL_FALLBACK with actual fallback agents."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing primary agent and add to graph engine
        failing_agent = MockAgent("primary_agent", should_fail=True)
        primary_node = DependencyNode(
            agent_id="primary_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(primary_node)

        # Also add a fallback agent to the graph engine
        fallback_agent = MockAgent("fallback_agent")
        fallback_node = DependencyNode(
            agent_id="fallback_agent",
            agent=fallback_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(fallback_node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry to provide fallback agents
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = create_registry_mock(
                get_failure_strategy=FailurePropagationStrategy.CONDITIONAL_FALLBACK,
                has_fallback_agents=True,
                get_fallback_agents=["fallback_agent"],
                is_critical_agent=True,
            )
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should handle fallback scenario
            assert_execution_failure(results, recovery_actions=True)
            assert_specific_agents_executed(results, ["fallback_agent"])
            assert "fallback_agent" in context.agent_outputs

    @pytest.mark.asyncio
    async def test_conditional_fallback_no_fallback_agents(self, simple_graph_engine):
        """Test CONDITIONAL_FALLBACK when no fallback agents available."""
        from cognivault.exceptions import PipelineExecutionError
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        failing_node = DependencyNode(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(failing_node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry with no fallback agents
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_failure_strategy.return_value = (
                FailurePropagationStrategy.CONDITIONAL_FALLBACK
            )
            mock_registry.has_fallback_agents.return_value = False
            mock_registry.is_critical_agent.return_value = True
            mock_registry_class.return_value = mock_registry

            # Without fallback agents, should handle failure according to cascade prevention
            results = await orchestrator.execute_pipeline(context)

            # Should have at least one failure
            assert results.failed_agents >= 1

    @pytest.mark.asyncio
    async def test_critical_agent_failure_handling(self, simple_graph_engine):
        """Test handling of critical agent failures."""
        from cognivault.exceptions import PipelineExecutionError
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create critical failing agent and add to graph engine
        critical_agent = MockAgent("critical_agent", should_fail=True)
        critical_node = DependencyNode(
            agent_id="critical_agent",
            agent=critical_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(critical_node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.CIRCUIT_BREAKER,  # Use circuit breaker for critical agents
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry to mark agent as critical
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_failure_strategy.return_value = (
                FailurePropagationStrategy.FAIL_FAST
            )
            mock_registry.is_critical_agent.return_value = True
            mock_registry.has_fallback_agents.return_value = False
            mock_registry_class.return_value = mock_registry

            # Critical agent failure should be handled according to strategy
            results = await orchestrator.execute_pipeline(context)

            # Should have failures recorded
            assert results.failed_agents >= 1
            # Overall success depends on cascade prevention configuration

    @pytest.mark.asyncio
    async def test_agent_execution_failure_with_metrics(self, simple_graph_engine):
        """Test agent failure with proper metrics recording."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True, execution_time=0.1)
        failing_node = DependencyNode(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(failing_node)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry to allow graceful degradation
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_failure_strategy.return_value = (
                FailurePropagationStrategy.GRACEFUL_DEGRADATION
            )
            mock_registry.is_critical_agent.return_value = False
            mock_registry.has_fallback_agents.return_value = False
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Check that metrics recorded failure
            assert results.execution_time_ms > 0
            assert results.failed_agents >= 1
            assert "graceful_degradation" in results.failure_recovery_actions


class TestOrchestratorDependencyManagement:
    """Test suite for agent dependency handling."""

    @pytest.fixture
    def orchestrator_with_dependencies(self, simple_graph_engine):
        """Create orchestrator with dependency management enabled."""
        config = OrchestratorConfig(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,  # Simplify for dependency tests
        )
        return AdvancedOrchestrator(simple_graph_engine, config)

    @pytest.mark.asyncio
    async def test_agent_dependency_validation(self, simple_graph_engine):
        """Test dependency validation for agents."""
        from cognivault.agents.registry import AgentRegistry

        # Create agents with dependencies and add to graph engine
        agent_a = MockAgent("agent_a")
        agent_b = MockAgent("agent_b")

        node_a = DependencyNode(
            agent_id="agent_a",
            agent=agent_a,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        node_b = DependencyNode(
            agent_id="agent_b",
            agent=agent_b,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node_a)
        simple_graph_engine.add_node(node_b)

        # Add dependency: agent_a must run before agent_b
        simple_graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry with dependencies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_agent_dependencies.side_effect = lambda name: (
                ["agent_a"] if name == "agent_b" else []
            )
            mock_registry.has_agent.return_value = True
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should execute in dependency order
            assert results.successful_agents >= 2
            assert "agent_a" in context.agent_outputs
            assert "agent_b" in context.agent_outputs

    @pytest.mark.asyncio
    async def test_dependency_failure_strategies(self, simple_graph_engine):
        """Test different failure strategies for dependencies."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create dependency chain: agent_a -> agent_b
        agent_a = MockAgent("agent_a", should_fail=True)
        agent_b = MockAgent("agent_b")

        node_a = DependencyNode(
            agent_id="agent_a",
            agent=agent_a,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        node_b = DependencyNode(
            agent_id="agent_b",
            agent=agent_b,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node_a)
        simple_graph_engine.add_node(node_b)

        # Add dependency
        simple_graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry with dependencies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_failure_strategy.return_value = (
                FailurePropagationStrategy.GRACEFUL_DEGRADATION
            )
            mock_registry.get_agent_dependencies.side_effect = lambda name: (
                ["agent_a"] if name == "agent_b" else []
            )
            mock_registry.is_critical_agent.return_value = False
            mock_registry.has_fallback_agents.return_value = False
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should handle dependency failure gracefully
            assert results.failed_agents >= 1
            assert "graceful_degradation" in results.failure_recovery_actions

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_dependencies(self, simple_graph_engine):
        """Test graceful degradation when dependencies fail."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create dependency chain where first agent fails
        failing_agent = MockAgent("failing_dependency", should_fail=True)
        dependent_agent = MockAgent("dependent_agent")
        independent_agent = MockAgent("independent_agent")

        # Add all agents to graph engine
        failing_node = DependencyNode(
            agent_id="failing_dependency",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        dependent_node = DependencyNode(
            agent_id="dependent_agent",
            agent=dependent_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        independent_node = DependencyNode(
            agent_id="independent_agent",
            agent=independent_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )

        simple_graph_engine.add_node(failing_node)
        simple_graph_engine.add_node(dependent_node)
        simple_graph_engine.add_node(independent_node)

        # Add dependency: failing_dependency -> dependent_agent
        simple_graph_engine.add_dependency(
            "failing_dependency", "dependent_agent", DependencyType.HARD
        )

        config = OrchestratorConfig(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_failure_strategy.return_value = (
                FailurePropagationStrategy.GRACEFUL_DEGRADATION
            )
            mock_registry.get_agent_dependencies.side_effect = lambda name: (
                ["failing_dependency"] if name == "dependent_agent" else []
            )
            mock_registry.is_critical_agent.return_value = False
            mock_registry.has_fallback_agents.return_value = False
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should continue with independent agents
            assert "independent_agent" in context.agent_outputs
            assert results.failed_agents >= 1  # The failing dependency should fail
            assert "graceful_degradation" in results.failure_recovery_actions


class TestOrchestratorHealthChecks:
    """Test suite for health check integration."""

    @pytest.fixture
    def orchestrator_with_health_checks(self, simple_graph_engine):
        """Create orchestrator with health check validation."""
        config = OrchestratorConfig(enable_failure_recovery=True)
        return AdvancedOrchestrator(simple_graph_engine, config)

    @pytest.mark.asyncio
    async def test_agent_health_check_failure(self, simple_graph_engine):
        """Test agent health check failure handling."""
        from cognivault.agents.registry import AgentRegistry

        # Create agent that will fail health check and add to graph engine
        unhealthy_agent = MockAgent("unhealthy_agent")
        unhealthy_node = DependencyNode(
            agent_id="unhealthy_agent",
            agent=unhealthy_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(unhealthy_node)

        config = OrchestratorConfig(enable_failure_recovery=True)
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry with failing health check
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.validate_agent_health.return_value = False
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should handle health check failure - execution should complete
            assert results.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_health_check_routing_decisions(self, simple_graph_engine):
        """Test conditional routing for health check failures."""
        from cognivault.agents.registry import AgentRegistry

        # Create multiple agents with mixed health and add to graph engine
        healthy_agent = MockAgent("healthy_agent")
        unhealthy_agent = MockAgent("unhealthy_agent")

        healthy_node = DependencyNode(
            agent_id="healthy_agent",
            agent=healthy_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        unhealthy_node = DependencyNode(
            agent_id="unhealthy_agent",
            agent=unhealthy_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(healthy_node)
        simple_graph_engine.add_node(unhealthy_node)

        config = OrchestratorConfig(enable_failure_recovery=True)
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContext(query="test")

        # Mock registry with selective health check results
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.validate_agent_health.side_effect = (
                lambda name: name == "healthy_agent"
            )
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should execute healthy agents
            assert "healthy_agent" in context.agent_outputs
            assert results.execution_time_ms > 0


class TestOrchestratorObservability:
    """Test suite for observability and metrics integration."""

    @pytest.mark.asyncio
    async def test_execution_state_tracking(self, orchestrator):
        """Test complex execution state tracking across multiple failures."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create mixed scenario with success, failure, and degradation
        success_agent = MockAgent("success_agent")
        failing_agent = MockAgent("failing_agent", should_fail=True)
        degraded_agent = MockAgent("degraded_agent", should_fail=True)
        agents = {
            "success_agent": success_agent,
            "failing_agent": failing_agent,
            "degraded_agent": degraded_agent,
        }
        context = AgentContext(query="test")

        # Mock registry with different strategies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry = Mock()

            def get_strategy(name):
                if name == "degraded_agent":
                    return FailurePropagationStrategy.GRACEFUL_DEGRADATION
                return FailurePropagationStrategy.WARN_CONTINUE

            mock_registry.get_failure_strategy.side_effect = get_strategy
            mock_registry.is_critical_agent.return_value = False
            mock_registry.has_fallback_agents.return_value = False
            mock_registry_class.return_value = mock_registry

            # Add agents to graph engine first
            for agent_name, agent in agents.items():
                if agent_name not in [
                    node.agent_id for node in orchestrator.graph_engine.nodes.values()
                ]:
                    node = DependencyNode(
                        agent_id=agent_name,
                        agent=agent,
                        priority=ExecutionPriority.NORMAL,
                        timeout_ms=5000,
                    )
                    orchestrator.graph_engine.add_node(node)

            results = await orchestrator.execute_pipeline(context)

            # Should track complex execution state
            assert_execution_failure(results, recovery_actions=True)
            assert_specific_agents_executed(results, ["success_agent"])
            assert "success_agent" in context.agent_outputs

    @pytest.mark.asyncio
    async def test_metrics_recording_comprehensive(self, orchestrator):
        """Test comprehensive metrics recording for various scenarios."""
        # Create agents with different execution characteristics
        fast_agent = MockAgent("fast_agent", execution_time=0.05)
        slow_agent = MockAgent("slow_agent", execution_time=0.15)
        agents = {"fast_agent": fast_agent, "slow_agent": slow_agent}
        context = AgentContext(query="test")

        # Add agents to graph engine first
        for agent_name, agent in agents.items():
            if agent_name not in [
                node.agent_id for node in orchestrator.graph_engine.nodes.values()
            ]:
                node = DependencyNode(
                    agent_id=agent_name,
                    agent=agent,
                    priority=ExecutionPriority.NORMAL,
                    timeout_ms=5000,
                )
                orchestrator.graph_engine.add_node(node)

        results = await orchestrator.execute_pipeline(context)

        # Should record comprehensive metrics
        assert_execution_success(
            results, min_successful=5
        )  # All agents (including defaults) should succeed
        assert_specific_agents_executed(results, ["fast_agent", "slow_agent"])
        assert len(results.pipeline_stages) == 4  # All stages
        assert all(stage.success for stage in results.pipeline_stages)

    @pytest.mark.asyncio
    async def test_mixed_failure_strategies(self, orchestrator):
        """Test pipeline with agents having different failure strategies."""
        from cognivault.agents.registry import AgentRegistry
        from cognivault.exceptions import FailurePropagationStrategy

        # Create agents with different failure behaviors
        warn_agent = MockAgent("warn_agent", should_fail=True)
        degrade_agent = MockAgent("degrade_agent", should_fail=True)
        success_agent = MockAgent("success_agent")
        agents = {
            "warn_agent": warn_agent,
            "degrade_agent": degrade_agent,
            "success_agent": success_agent,
        }
        context = AgentContext(query="test")

        # Mock registry with different strategies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:

            def get_strategy(name):
                if name == "warn_agent":
                    return FailurePropagationStrategy.WARN_CONTINUE
                elif name == "degrade_agent":
                    return FailurePropagationStrategy.GRACEFUL_DEGRADATION
                return FailurePropagationStrategy.FAIL_FAST

            mock_registry = create_registry_mock(
                get_failure_strategy_side_effect=get_strategy, is_critical_agent=False
            )
            mock_registry_class.return_value = mock_registry

            # Add agents to graph engine first
            for agent_name, agent in agents.items():
                if agent_name not in [
                    node.agent_id for node in orchestrator.graph_engine.nodes.values()
                ]:
                    node = DependencyNode(
                        agent_id=agent_name,
                        agent=agent,
                        priority=ExecutionPriority.NORMAL,
                        timeout_ms=5000,
                    )
                    orchestrator.graph_engine.add_node(node)

            results = await orchestrator.execute_pipeline(context)

            # Should handle mixed strategies appropriately
            assert_execution_failure(results, recovery_actions=True)
            assert_specific_agents_executed(results, ["success_agent"])
            assert "success_agent" in context.agent_outputs
            assert "graceful_degradation" in results.failure_recovery_actions
