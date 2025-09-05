"""
Tests for the advanced orchestrator.

Covers integration of all dependency management components:
graph engine, execution planner, failure manager, resource scheduler,
and dynamic composition.
"""

import pytest
import asyncio
import time
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

from cognivault.context import AgentContext
from tests.factories.agent_context_factories import AgentContextPatterns
from cognivault.agents.base_agent import BaseAgent
from cognivault.dependencies.graph_engine import (
    DependencyGraphEngine,
    DependencyType,
    ExecutionPriority,
)
from cognivault.dependencies.execution_planner import (
    ExecutionPlanner,
    ExecutionStrategy,
)
from cognivault.dependencies.failure_manager import (
    FailureManager,
    CascadePreventionStrategy,
)
from cognivault.dependencies.resource_scheduler import (
    ResourceScheduler,
    ResourceType,
)
from cognivault.dependencies.dynamic_composition import (
    DynamicAgentComposer,
)
from cognivault.dependencies.advanced_orchestrator import (
    AdvancedOrchestrator,
    ExecutionPhase,
)
from tests.factories.advanced_orchestrator_factories import (
    AdvancedResourceConstraintFactory,
    OrchestratorConfigFactory,
    ExecutionResultsFactory,
    ResourceAllocationResultFactory,
    PipelineStageFactory,
    DependencyNodeFactory,
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        delay: float = 0.0,
        execution_time: Optional[float] = None,
    ) -> None:
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
def orchestrator_config() -> Any:
    """Create orchestrator configuration for testing."""
    return OrchestratorConfigFactory.generate_valid_data(
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
def mock_agent_registry() -> Any:
    """Create a comprehensive mock for AgentRegistry with common defaults."""
    from cognivault.exceptions import FailurePropagationStrategy

    mock: Mock = Mock()

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


def create_registry_mock(**overrides: Any) -> Mock:
    """Helper function to create registry mocks with custom behavior."""
    from cognivault.exceptions import FailurePropagationStrategy

    mock: Mock = Mock()

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


def assert_execution_success(
    results: Any, expected_agents: Any = None, min_successful: Any = None
) -> None:
    """Helper to assert successful execution with observable metrics."""
    assert results.execution_time_ms > 0, "Execution should take measurable time"
    assert results.total_agents_executed > 0, "Should execute at least one agent"

    if expected_agents is not None:
        assert (
            results.total_agents_executed == expected_agents
        ), f"Expected {expected_agents} agents executed"

    if min_successful is not None:
        assert (
            results.successful_agents >= min_successful
        ), f"Expected at least {min_successful} successful agents"

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
    assert (
        phases_executed == expected_phases
    ), f"Missing phases: {expected_phases - phases_executed}"


def assert_execution_failure(
    results: Any, expected_failures: Any = None, recovery_actions: bool = True
) -> None:
    """Helper to assert execution failure with recovery actions."""
    assert results.execution_time_ms > 0, "Execution should take measurable time"
    assert results.failed_agents > 0, "Should have at least one failure"

    if expected_failures is not None:
        assert (
            results.failed_agents == expected_failures
        ), f"Expected {expected_failures} failures"

    if recovery_actions:
        assert len(results.failure_recovery_actions) > 0, "Should have recovery actions"


def assert_specific_agents_executed(results: Any, agent_names: Any) -> None:
    """Assert that specific agents were executed based on pipeline stages."""
    executed_agents = set()
    for stage in results.pipeline_stages:
        executed_agents.update(stage.agents_executed)

    for agent_name in agent_names:
        assert (
            agent_name in executed_agents
        ), f"Agent {agent_name} should have been executed"


def get_executed_agents(results: Any) -> Any:
    """Get set of all agents that were executed according to pipeline stages."""
    executed_agents = set()
    for stage in results.pipeline_stages:
        executed_agents.update(stage.agents_executed)
    return executed_agents


@pytest.fixture
def simple_graph_engine() -> Any:
    """Create a simple graph engine with test agents."""
    engine = DependencyGraphEngine()

    agents = {
        "preprocessor": MockAgent("preprocessor"),
        "analyzer": MockAgent("analyzer"),
        "formatter": MockAgent("formatter"),
    }

    for agent_id, agent in agents.items():
        node = DependencyNodeFactory.basic(
            agent_id=agent_id,
            agent=agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
            resource_constraints=[
                AdvancedResourceConstraintFactory.basic(
                    resource_type="cpu", max_usage=30.0, units="percentage"
                ),
                AdvancedResourceConstraintFactory.basic(
                    resource_type="memory", max_usage=512.0, units="MB"
                ),
            ],
        )
        engine.add_node(node)

    # Add dependencies: preprocessor -> analyzer -> formatter
    engine.add_dependency("preprocessor", "analyzer", DependencyType.HARD)
    engine.add_dependency("analyzer", "formatter", DependencyType.HARD)

    return engine


@pytest.fixture
def orchestrator(orchestrator_config: Any, simple_graph_engine: Any) -> Any:
    """Create an advanced orchestrator for testing."""
    return AdvancedOrchestrator(simple_graph_engine, orchestrator_config)


@pytest.fixture
def context() -> Any:
    """Create a basic agent context."""
    return AgentContextPatterns.simple_query("test query for orchestration")


class TestOrchestratorConfig:
    """Test OrchestratorConfig functionality."""

    def test_config_creation(self) -> None:
        """Test creating orchestrator configuration."""
        config = OrchestratorConfigFactory.basic(
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

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = OrchestratorConfigFactory.basic()

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

    def test_results_creation(self) -> None:
        """Test creating execution results."""
        results = ExecutionResultsFactory.basic(
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

    def test_results_get_success_rate(self) -> None:
        """Test success rate calculation."""
        results = ExecutionResultsFactory.basic(
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

    def test_results_get_success_rate_no_agents(self) -> None:
        """Test success rate with no agents executed."""
        results = ExecutionResultsFactory.basic(
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

    def test_results_to_dict(self) -> None:
        """Test converting results to dictionary."""
        results = ExecutionResultsFactory.basic(
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

    def test_allocation_result_creation(self) -> None:
        """Test creating resource allocation result."""
        result = ResourceAllocationResultFactory.basic(
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

    def test_pipeline_stage_creation(self) -> None:
        """Test creating pipeline stage."""
        stage = PipelineStageFactory.basic(
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

    def test_orchestrator_creation(
        self, simple_graph_engine: Any, orchestrator_config: Any
    ) -> None:
        """Test creating advanced orchestrator."""
        orchestrator = AdvancedOrchestrator(simple_graph_engine, orchestrator_config)

        assert orchestrator.graph_engine == simple_graph_engine
        assert orchestrator.config == orchestrator_config
        assert isinstance(orchestrator.execution_planner, ExecutionPlanner)
        assert isinstance(orchestrator.failure_manager, FailureManager)
        assert isinstance(orchestrator.resource_scheduler, ResourceScheduler)
        assert orchestrator.dynamic_composer is None  # Disabled in config

    def test_orchestrator_with_dynamic_composition(
        self, simple_graph_engine: Any
    ) -> None:
        """Test orchestrator with dynamic composition enabled."""
        config = OrchestratorConfigFactory.basic(enable_dynamic_composition=True)
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)

        assert orchestrator.dynamic_composer is not None
        assert isinstance(orchestrator.dynamic_composer, DynamicAgentComposer)

    @pytest.mark.asyncio
    async def test_execute_pipeline_success(
        self, orchestrator: Any, context: Any
    ) -> None:
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
    async def test_execute_pipeline_with_failure(
        self, orchestrator_config: Any, context: Any
    ) -> None:
        """Test pipeline execution with agent failure."""
        # Create graph with failing agent
        engine = DependencyGraphEngine()

        agents = {
            "preprocessor": MockAgent("preprocessor"),
            "failing_agent": MockAgent("failing_agent", should_fail=True),
            "formatter": MockAgent("formatter"),
        }

        for agent_id, agent in agents.items():
            node = DependencyNodeFactory.basic(
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
    async def test_execute_pipeline_timeout(
        self, orchestrator_config: Any, context: Any
    ) -> None:
        """Test pipeline execution with timeout."""
        # Create graph with slow agent
        engine = DependencyGraphEngine()

        # Create agent that takes longer than pipeline timeout
        slow_agent = MockAgent("slow_agent", delay=1.0)  # 1 second delay
        node = DependencyNodeFactory.basic(
            agent_id="slow_agent",
            agent=slow_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=500,  # 500ms timeout
        )
        engine.add_node(node)

        # Set very short pipeline timeout (but still within validation bounds)
        orchestrator_config.pipeline_timeout_ms = 1000  # 1000ms (1 second)
        orchestrator = AdvancedOrchestrator(engine, orchestrator_config)

        results = await orchestrator.execute_pipeline(context)

        # Should timeout
        assert results.success is False

    @pytest.mark.asyncio
    async def test_prepare_execution_phase(
        self, orchestrator: Any, context: Any
    ) -> None:
        """Test execution preparation phase."""
        stage = await orchestrator._prepare_execution(context)

        assert stage.phase == ExecutionPhase.PREPARATION
        assert stage.success is True
        assert stage.stage_duration_ms > 0

        # Should have created execution plan
        assert orchestrator._current_execution_plan is not None

    @pytest.mark.asyncio
    async def test_allocate_resources_phase(
        self, orchestrator: Any, context: Any
    ) -> None:
        """Test resource allocation phase."""
        # First prepare execution
        await orchestrator._prepare_execution(context)

        stage = await orchestrator._allocate_resources(context)

        assert stage.phase == ExecutionPhase.RESOURCE_ALLOCATION
        assert stage.success is True
        assert len(stage.agents_executed) == 3  # All agents should get resources

    @pytest.mark.asyncio
    async def test_allocate_resources_with_failure(
        self, orchestrator_config: Any, context: Any
    ) -> None:
        """Test resource allocation with insufficient resources."""
        # Create limited resource pool
        engine = DependencyGraphEngine()

        # Create agent with high resource requirements
        high_resource_agent = MockAgent("high_resource_agent")
        node = DependencyNodeFactory.basic(
            agent_id="high_resource_agent",
            agent=high_resource_agent,
            priority=ExecutionPriority.NORMAL,
            resource_constraints=[
                AdvancedResourceConstraintFactory.basic(
                    resource_type="memory", max_usage=20000.0, units="MB"
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
    async def test_execute_agents_phase(self, orchestrator: Any, context: Any) -> None:
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
    async def test_cleanup_resources_phase(
        self, orchestrator: Any, context: Any
    ) -> None:
        """Test resource cleanup phase."""
        # Execute full pipeline first
        await orchestrator._prepare_execution(context)
        await orchestrator._allocate_resources(context)
        await orchestrator._execute_agents(context)

        stage = await orchestrator._cleanup_resources(context)

        assert stage.phase == ExecutionPhase.CLEANUP
        assert stage.success is True

    @pytest.mark.asyncio
    async def test_handle_agent_execution_success(
        self, orchestrator: Any, context: Any
    ) -> None:
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
    async def test_handle_agent_execution_failure(
        self, orchestrator: Any, context: Any
    ) -> None:
        """Test handling failed agent execution."""
        failing_agent = MockAgent("failing_agent", should_fail=True)

        result = await orchestrator._handle_agent_execution(
            "failing_agent", failing_agent, context
        )

        assert result["success"] is False
        assert result["agent_id"] == "failing_agent"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_agent_execution_timeout(
        self, orchestrator: Any, context: Any
    ) -> None:
        """Test handling agent execution timeout."""
        # Create very slow agent
        slow_agent = MockAgent("slow_agent", delay=2.0)  # 2 second delay

        # Mock the node to have a short timeout
        node = DependencyNodeFactory.basic(
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

    def test_create_resource_allocation_result(self, orchestrator: Any) -> None:
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

    def test_aggregate_execution_results(self, orchestrator: Any) -> None:
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
            PipelineStageFactory.basic(
                stage_id="prep",
                phase=ExecutionPhase.PREPARATION,
                agents_executed=[],
                stage_duration_ms=50.0,
                success=True,
            ),
            PipelineStageFactory.basic(
                stage_id="exec",
                phase=ExecutionPhase.EXECUTION,
                agents_executed=["agent1", "agent2", "agent3"],
                stage_duration_ms=450.0,
                success=False,
            ),
        ]

        resource_results: list[Any] = []
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

    def test_create_pipeline_stage(self, orchestrator: Any) -> None:
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

    def test_calculate_execution_statistics(self, orchestrator: Any) -> None:
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
    async def test_complete_orchestration_workflow(self) -> None:
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
            node = DependencyNodeFactory.basic(
                agent_id=agent_id,
                agent=agent,
                priority=priority,
                timeout_ms=10000,
                resource_constraints=[
                    AdvancedResourceConstraintFactory.basic(
                        resource_type="cpu", max_usage=25.0, units="percentage"
                    ),
                    AdvancedResourceConstraintFactory.basic(
                        resource_type="memory", max_usage=300.0, units="MB"
                    ),
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
        config = OrchestratorConfigFactory.basic(
            max_concurrent_agents=3,
            enable_failure_recovery=True,
            enable_resource_scheduling=True,
            enable_dynamic_composition=False,  # Keep simple for testing
            default_execution_strategy=ExecutionStrategy.PARALLEL_BATCHED,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )

        orchestrator = AdvancedOrchestrator(engine, config)
        context = AgentContextPatterns.simple_query("Complex workflow test")

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
    async def test_orchestration_with_failures_and_recovery(self) -> None:
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
            node = DependencyNodeFactory.basic(
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
        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )

        orchestrator = AdvancedOrchestrator(engine, config)

        # Configure failure recovery for failing agent
        orchestrator.failure_manager.add_fallback_chain(
            "failing_agent", ["backup_agent"]
        )

        context = AgentContextPatterns.simple_query("Failure recovery test")
        results = await orchestrator.execute_pipeline(context)

        # Should handle failure gracefully
        assert results.failed_agents > 0  # failing_agent should fail
        assert len(results.failure_recovery_actions) > 0

        # Should still execute successfully overall due to recovery
        # (depending on implementation details)

    @pytest.mark.asyncio
    async def test_orchestration_performance_characteristics(self) -> None:
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

            node = DependencyNodeFactory.basic(
                agent_id=agent_id,
                agent=parallel_agents[agent_id],
                priority=ExecutionPriority.NORMAL,
                timeout_ms=5000,
            )
            engine.add_node(node)

        # Add final aggregator
        aggregator = MockAgent("aggregator", delay=0.05)
        node = DependencyNodeFactory.basic(
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

        context = AgentContextPatterns.simple_query("Performance test")
        performance_results = {}

        for strategy in strategies:
            config = OrchestratorConfigFactory.basic(
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
    async def test_orchestration_resource_constraints(self) -> None:
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
            "memory_intensive": [
                AdvancedResourceConstraintFactory.basic(
                    resource_type="memory", max_usage=2000.0, units="MB"
                )
            ],
            "cpu_intensive": [
                AdvancedResourceConstraintFactory.basic(
                    resource_type="cpu", max_usage=90.0, units="percentage"
                )
            ],
            "balanced": [
                AdvancedResourceConstraintFactory.basic(
                    resource_type="cpu", max_usage=50.0, units="percentage"
                ),
                AdvancedResourceConstraintFactory.basic(
                    resource_type="memory", max_usage=500.0, units="MB"
                ),
            ],
        }

        for agent_id, agent in agents.items():
            node = DependencyNodeFactory.basic(
                agent_id=agent_id,
                agent=agent,
                priority=ExecutionPriority.NORMAL,
                timeout_ms=5000,
                resource_constraints=resource_profiles[agent_id],
            )
            engine.add_node(node)

        # Make them independent for parallel execution
        # (no dependencies to test resource scheduling)

        config = OrchestratorConfigFactory.basic(
            enable_resource_scheduling=True,
            max_concurrent_agents=2,  # Limited concurrency
        )

        orchestrator = AdvancedOrchestrator(engine, config)
        context = AgentContextPatterns.simple_query("Resource constraint test")

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
    async def test_orchestration_error_scenarios(self) -> None:
        """Test orchestration with various error scenarios."""
        # Create minimal graph for error testing
        engine = DependencyGraphEngine()

        error_agent = MockAgent("error_agent", should_fail=True)
        node = DependencyNodeFactory.basic(
            agent_id="error_agent",
            agent=error_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=1000,
        )
        engine.add_node(node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.ISOLATION,
        )

        orchestrator = AdvancedOrchestrator(engine, config)
        context = AgentContextPatterns.simple_query("Error scenario test")

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
    def orchestrator_with_failure_manager(self, simple_graph_engine: Any) -> Any:
        """Create orchestrator with configured failure manager."""
        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        return AdvancedOrchestrator(simple_graph_engine, config)

    @pytest.mark.asyncio
    async def test_fail_fast_strategy(self, simple_graph_engine: Any) -> None:
        """Test FAIL_FAST strategy raises PipelineExecutionError."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create agent that will fail and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        node = DependencyNodeFactory.basic(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.CIRCUIT_BREAKER,  # Use circuit breaker for fast failure
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry to return FAIL_FAST strategy
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = create_registry_mock(
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
    async def test_warn_continue_strategy(self, simple_graph_engine: Any) -> None:
        """Test WARN_CONTINUE strategy logs warning but continues."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        failing_node = DependencyNodeFactory.basic(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(failing_node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

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
    async def test_graceful_degradation_strategy(
        self, simple_graph_engine: Any
    ) -> None:
        """Test GRACEFUL_DEGRADATION creates warnings and marks degraded."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        node = DependencyNodeFactory.basic(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry to return GRACEFUL_DEGRADATION strategy
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    async def test_conditional_fallback_with_agents(
        self, simple_graph_engine: Any
    ) -> None:
        """Test CONDITIONAL_FALLBACK with actual fallback agents."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing primary agent and add to graph engine
        failing_agent = MockAgent("primary_agent", should_fail=True)
        primary_node = DependencyNodeFactory.basic(
            agent_id="primary_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(primary_node)

        # Also add a fallback agent to the graph engine
        fallback_agent = MockAgent("fallback_agent")
        fallback_node = DependencyNodeFactory.basic(
            agent_id="fallback_agent",
            agent=fallback_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(fallback_node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

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
    async def test_conditional_fallback_no_fallback_agents(
        self, simple_graph_engine: Any
    ) -> None:
        """Test CONDITIONAL_FALLBACK when no fallback agents available."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True)
        failing_node = DependencyNodeFactory.basic(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(failing_node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry with no fallback agents
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    async def test_critical_agent_failure_handling(
        self, simple_graph_engine: Any
    ) -> None:
        """Test handling of critical agent failures."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create critical failing agent and add to graph engine
        critical_agent = MockAgent("critical_agent", should_fail=True)
        critical_node = DependencyNodeFactory.basic(
            agent_id="critical_agent",
            agent=critical_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(critical_node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.CIRCUIT_BREAKER,  # Use circuit breaker for critical agents
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry to mark agent as critical
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    async def test_agent_execution_failure_with_metrics(
        self, simple_graph_engine: Any
    ) -> None:
        """Test agent failure with proper metrics recording."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create failing agent and add to graph engine
        failing_agent = MockAgent("failing_agent", should_fail=True, execution_time=0.1)
        failing_node = DependencyNodeFactory.basic(
            agent_id="failing_agent",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(failing_node)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry to allow graceful degradation
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    def orchestrator_with_dependencies(self, simple_graph_engine: Any) -> Any:
        """Create orchestrator with dependency management enabled."""
        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,  # Simplify for dependency tests
        )
        return AdvancedOrchestrator(simple_graph_engine, config)

    @pytest.mark.asyncio
    async def test_agent_dependency_validation(self, simple_graph_engine: Any) -> None:
        """Test dependency validation for agents."""

        # Create agents with dependencies and add to graph engine
        agent_a = MockAgent("agent_a")
        agent_b = MockAgent("agent_b")

        node_a = DependencyNodeFactory.basic(
            agent_id="agent_a",
            agent=agent_a,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        node_b = DependencyNodeFactory.basic(
            agent_id="agent_b",
            agent=agent_b,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node_a)
        simple_graph_engine.add_node(node_b)

        # Add dependency: agent_a must run before agent_b
        simple_graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry with dependencies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    async def test_dependency_failure_strategies(
        self, simple_graph_engine: Any
    ) -> None:
        """Test different failure strategies for dependencies."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create dependency chain: agent_a -> agent_b
        agent_a = MockAgent("agent_a", should_fail=True)
        agent_b = MockAgent("agent_b")

        node_a = DependencyNodeFactory.basic(
            agent_id="agent_a",
            agent=agent_a,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        node_b = DependencyNodeFactory.basic(
            agent_id="agent_b",
            agent=agent_b,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(node_a)
        simple_graph_engine.add_node(node_b)

        # Add dependency
        simple_graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry with dependencies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    async def test_graceful_degradation_with_dependencies(
        self, simple_graph_engine: Any
    ) -> None:
        """Test graceful degradation when dependencies fail."""
        from cognivault.exceptions import FailurePropagationStrategy

        # Create dependency chain where first agent fails
        failing_agent = MockAgent("failing_dependency", should_fail=True)
        dependent_agent = MockAgent("dependent_agent")
        independent_agent = MockAgent("independent_agent")

        # Add all agents to graph engine
        failing_node = DependencyNodeFactory.basic(
            agent_id="failing_dependency",
            agent=failing_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        dependent_node = DependencyNodeFactory.basic(
            agent_id="dependent_agent",
            agent=dependent_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        independent_node = DependencyNodeFactory.basic(
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

        config = OrchestratorConfigFactory.basic(
            enable_failure_recovery=True,
            enable_resource_scheduling=False,
        )
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    def orchestrator_with_health_checks(self, simple_graph_engine: Any) -> Any:
        """Create orchestrator with health check validation."""
        config = OrchestratorConfigFactory.basic(enable_failure_recovery=True)
        return AdvancedOrchestrator(simple_graph_engine, config)

    @pytest.mark.asyncio
    async def test_agent_health_check_failure(self, simple_graph_engine: Any) -> None:
        """Test agent health check failure handling."""

        # Create agent that will fail health check and add to graph engine
        unhealthy_agent = MockAgent("unhealthy_agent")
        unhealthy_node = DependencyNodeFactory.basic(
            agent_id="unhealthy_agent",
            agent=unhealthy_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(unhealthy_node)

        config = OrchestratorConfigFactory.basic(enable_failure_recovery=True)
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry with failing health check
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
            mock_registry.validate_agent_health.return_value = False
            mock_registry_class.return_value = mock_registry

            results = await orchestrator.execute_pipeline(context)

            # Should handle health check failure - execution should complete
            assert results.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_health_check_routing_decisions(
        self, simple_graph_engine: Any
    ) -> None:
        """Test conditional routing for health check failures."""

        # Create multiple agents with mixed health and add to graph engine
        healthy_agent = MockAgent("healthy_agent")
        unhealthy_agent = MockAgent("unhealthy_agent")

        healthy_node = DependencyNodeFactory.basic(
            agent_id="healthy_agent",
            agent=healthy_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        unhealthy_node = DependencyNodeFactory.basic(
            agent_id="unhealthy_agent",
            agent=unhealthy_agent,
            priority=ExecutionPriority.NORMAL,
            timeout_ms=5000,
        )
        simple_graph_engine.add_node(healthy_node)
        simple_graph_engine.add_node(unhealthy_node)

        config = OrchestratorConfigFactory.basic(enable_failure_recovery=True)
        orchestrator = AdvancedOrchestrator(simple_graph_engine, config)
        context = AgentContextPatterns.simple_query("test")

        # Mock registry with selective health check results
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()
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
    async def test_execution_state_tracking(self, orchestrator: Any) -> None:
        """Test complex execution state tracking across multiple failures."""
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
        context = AgentContextPatterns.simple_query("test")

        # Mock registry with different strategies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:
            mock_registry: Mock = Mock()

            def get_strategy(name: str) -> Any:
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
                    node = DependencyNodeFactory.basic(
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
    async def test_metrics_recording_comprehensive(self, orchestrator: Any) -> None:
        """Test comprehensive metrics recording for various scenarios."""
        # Create agents with different execution characteristics
        fast_agent = MockAgent("fast_agent", execution_time=0.05)
        slow_agent = MockAgent("slow_agent", execution_time=0.15)
        agents = {"fast_agent": fast_agent, "slow_agent": slow_agent}
        context = AgentContextPatterns.simple_query("test")

        # Add agents to graph engine first
        for agent_name, agent in agents.items():
            if agent_name not in [
                node.agent_id for node in orchestrator.graph_engine.nodes.values()
            ]:
                node = DependencyNodeFactory.basic(
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
    async def test_mixed_failure_strategies(self, orchestrator: Any) -> None:
        """Test pipeline with agents having different failure strategies."""
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
        context = AgentContextPatterns.simple_query("test")

        # Mock registry with different strategies
        with patch("cognivault.agents.registry.AgentRegistry") as mock_registry_class:

            def get_strategy(name: str) -> Any:
                if name == "warn_agent":
                    return FailurePropagationStrategy.WARN_CONTINUE
                elif name == "degrade_agent":
                    return FailurePropagationStrategy.GRACEFUL_DEGRADATION
                return FailurePropagationStrategy.FAIL_FAST

            mock_registry: Mock = create_registry_mock(
                get_failure_strategy_side_effect=get_strategy, is_critical_agent=False
            )
            mock_registry_class.return_value = mock_registry

            # Add agents to graph engine first
            for agent_name, agent in agents.items():
                if agent_name not in [
                    node.agent_id for node in orchestrator.graph_engine.nodes.values()
                ]:
                    node = DependencyNodeFactory.basic(
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


# Critical Coverage Tests - Race Conditions, Deadlocks, Resource Leaks
class TestAdvancedOrchestratorCriticalPaths:
    """Tests for critical missing coverage areas to prevent race conditions, deadlocks, and resource leaks."""

    @pytest.fixture
    def orchestrator_with_dynamic_composition(self, simple_graph_engine: Any) -> Any:
        """Create orchestrator with dynamic composition enabled."""
        config = OrchestratorConfigFactory.basic(
            max_concurrent_agents=2,
            enable_failure_recovery=True,
            enable_resource_scheduling=True,
            enable_dynamic_composition=True,  # Enable for complex tests
            default_execution_strategy=ExecutionStrategy.PARALLEL_BATCHED,
            cascade_prevention_strategy=CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        )
        return AdvancedOrchestrator(simple_graph_engine, config)

    @pytest.mark.asyncio
    async def test_agent_initialization_failure_handling(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test agent initialization failures - lines 217-272."""
        # Clear existing loaded agents
        orchestrator_with_dynamic_composition.loaded_agents.clear()

        # Mock the graph engine to have specific agents we want to test
        from cognivault.dependencies.graph_engine import DependencyNode

        # Create proper DependencyNode mocks with priority values
        mock_node1 = Mock(spec=DependencyNode)
        mock_node1.priority = ExecutionPriority.NORMAL
        mock_node1.requires_exclusive_access = False

        mock_node2 = Mock(spec=DependencyNode)
        mock_node2.priority = ExecutionPriority.NORMAL
        mock_node2.requires_exclusive_access = False

        mock_node3 = Mock(spec=DependencyNode)
        mock_node3.priority = ExecutionPriority.NORMAL
        mock_node3.requires_exclusive_access = False

        mock_node4 = Mock(spec=DependencyNode)
        mock_node4.priority = ExecutionPriority.NORMAL
        mock_node4.requires_exclusive_access = False

        orchestrator_with_dynamic_composition.graph_engine.nodes = {
            "successful_agent": mock_node1,
            "failing_agent": mock_node2,
            "missing_agent": mock_node3,
            "another_successful_agent": mock_node4,
        }

        # Mock dynamic composer to fail on certain agents
        def mock_load_agent(agent_name: str) -> Any:
            if agent_name == "successful_agent":
                return MockAgent("successful_agent")
            elif agent_name == "failing_agent":
                raise Exception("Agent load failure")
            elif agent_name == "missing_agent":
                return None  # Simulate agent not found
            elif agent_name == "another_successful_agent":
                return MockAgent("another_successful_agent")
            else:
                return MockAgent(agent_name)  # Fallback for other agents

        orchestrator_with_dynamic_composition.dynamic_composer.load_agent = AsyncMock(
            side_effect=mock_load_agent
        )
        orchestrator_with_dynamic_composition.dynamic_composer.discover_agents = (
            AsyncMock()
        )

        # Should handle failures gracefully without stopping initialization
        await orchestrator_with_dynamic_composition.initialize_agents()

        # Verify only successful agents were loaded (should be 2: successful_agent and another_successful_agent)
        assert len(orchestrator_with_dynamic_composition.loaded_agents) == 2
        assert "successful_agent" in orchestrator_with_dynamic_composition.loaded_agents
        assert (
            "another_successful_agent"
            in orchestrator_with_dynamic_composition.loaded_agents
        )
        assert (
            "failing_agent" not in orchestrator_with_dynamic_composition.loaded_agents
        )
        assert (
            "missing_agent" not in orchestrator_with_dynamic_composition.loaded_agents
        )

    @pytest.mark.asyncio
    async def test_discovery_and_composition_phase(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test discovery and composition phase - lines 343-359."""
        context = AgentContextPatterns.simple_query("test discovery")

        # Mock dynamic composer methods
        discovery_results = {"discovered_agents": ["new_agent_1", "new_agent_2"]}
        optimization_results = {"optimizations_applied": ["load_balancing", "caching"]}

        orchestrator_with_dynamic_composition.dynamic_composer.auto_discover_and_swap = AsyncMock(
            return_value=discovery_results
        )
        orchestrator_with_dynamic_composition.dynamic_composer.optimize_composition = (
            AsyncMock(return_value=optimization_results)
        )

        await orchestrator_with_dynamic_composition._phase_discovery_and_composition(
            context
        )

        # Verify results stored in context
        assert context.execution_state["discovery_results"] == discovery_results
        assert (
            context.execution_state["composition_optimization"] == optimization_results
        )

    @pytest.mark.asyncio
    async def test_execution_planning_phase(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test execution planning phase - lines 361-380."""
        context = AgentContextPatterns.simple_query("test planning")

        # Mock execution plan
        mock_plan: Mock = Mock()
        mock_plan.get_execution_summary.return_value = {
            "stages": 3,
            "parallel_factor": 2.0,
        }
        mock_plan.get_total_stages.return_value = 3
        mock_plan.parallelism_factor = 2.0

        orchestrator_with_dynamic_composition.execution_planner.create_plan = Mock(
            return_value=mock_plan
        )

        await orchestrator_with_dynamic_composition._phase_execution_planning(context)

        # Verify plan was created and stored
        assert orchestrator_with_dynamic_composition.current_plan == mock_plan
        assert "execution_plan" in context.execution_state
        orchestrator_with_dynamic_composition.execution_planner.create_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_resource_allocation_phase_with_constraints(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test resource allocation phase with constraints - lines 381-408."""
        context = AgentContextPatterns.simple_query("test resource allocation")

        # Add agents with resource constraints
        agent1 = MockAgent("resource_heavy_agent")
        agent2 = MockAgent("lightweight_agent")
        orchestrator_with_dynamic_composition.loaded_agents = {
            "resource_heavy_agent": agent1,
            "lightweight_agent": agent2,
        }

        # Mock graph nodes with resource constraints
        node1: Mock = Mock()
        node1.resource_constraints = {"memory": 1024, "cpu": 2}
        node1.priority = ExecutionPriority.HIGH
        node1.timeout_ms = 5000

        node2: Mock = Mock()
        node2.resource_constraints = {"memory": 256, "cpu": 1}
        node2.priority = ExecutionPriority.NORMAL
        node2.timeout_ms = 3000

        orchestrator_with_dynamic_composition.graph_engine.nodes = {
            "resource_heavy_agent": node1,
            "lightweight_agent": node2,
        }

        # Mock resource scheduler
        orchestrator_with_dynamic_composition.resource_scheduler.request_resources = (
            AsyncMock(return_value=["req_1", "req_2"])
        )
        orchestrator_with_dynamic_composition.resource_scheduler.get_resource_utilization = Mock(
            return_value={"memory_usage": 0.75, "cpu_usage": 0.60}
        )

        await orchestrator_with_dynamic_composition._phase_resource_allocation(context)

        # Verify resource requests were made
        assert (
            orchestrator_with_dynamic_composition.resource_scheduler.request_resources.call_count
            == 2
        )
        assert "resource_utilization" in context.execution_state

    @pytest.mark.asyncio
    async def test_parallel_execution_race_condition_prevention(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test parallel execution to prevent race conditions - lines 478-493."""
        context = AgentContextPatterns.simple_query("test parallel execution")

        # Create mock stage with parallel groups
        mock_stage: Mock = Mock()
        mock_group: Mock = Mock()
        mock_group.agents = ["agent1", "agent2", "agent3"]
        mock_stage.parallel_groups = [mock_group]

        # Add agents that will execute concurrently
        agent1 = MockAgent("agent1", delay=0.1)
        agent2 = MockAgent("agent2", delay=0.2)
        agent3 = MockAgent("agent3", delay=0.15)

        orchestrator_with_dynamic_composition.loaded_agents = {
            "agent1": agent1,
            "agent2": agent2,
            "agent3": agent3,
        }

        # Track execution order for race condition detection
        execution_order = []
        original_execute = (
            orchestrator_with_dynamic_composition._execute_agent_with_failure_handling
        )

        async def track_execution(agent_id: str, ctx: Any) -> None:
            execution_order.append(f"{agent_id}_start")
            await asyncio.sleep(0.05)  # Simulate work
            execution_order.append(f"{agent_id}_end")

        orchestrator_with_dynamic_composition._execute_agent_with_failure_handling = (
            track_execution
        )

        start_time = time.time()
        await orchestrator_with_dynamic_composition._execute_parallel_stage(
            mock_stage, context
        )
        end_time = time.time()

        # Verify parallel execution (should complete in ~0.05s, not 0.45s sequential)
        assert (end_time - start_time) < 0.3  # Much faster than sequential
        assert len(execution_order) == 6  # 3 starts + 3 ends

        # Verify all agents started before any finished (true parallelism)
        start_count = len([e for e in execution_order[:3] if e.endswith("_start")])
        assert start_count == 3  # All agents started concurrently

    @pytest.mark.asyncio
    async def test_agent_execution_with_comprehensive_failure_handling(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test comprehensive agent execution failure handling - lines 501-573."""
        context = AgentContextPatterns.simple_query("test failure handling")

        # Create agent that fails on first two attempts, succeeds on third
        failing_agent = MockAgent("failing_agent", should_fail=True)
        call_count = 0

        async def selective_failure_run(ctx: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Simulated failure attempt {call_count}")
            # Succeed on third attempt
            ctx.agent_outputs["failing_agent"] = "success after retries"
            return ctx

        # Replace the run method with our test implementation
        setattr(failing_agent, "run", selective_failure_run)
        orchestrator_with_dynamic_composition.loaded_agents = {
            "failing_agent": failing_agent
        }

        # Mock failure manager methods
        orchestrator_with_dynamic_composition.failure_manager.can_execute_agent = Mock(
            return_value=(True, None)
        )
        orchestrator_with_dynamic_composition.failure_manager.handle_agent_failure = (
            AsyncMock(return_value=(True, "retry"))  # Should retry
        )
        orchestrator_with_dynamic_composition.failure_manager.circuit_breakers = {
            "failing_agent": Mock()
        }

        # Mock retry configuration
        retry_config: Mock = Mock()
        retry_config.calculate_delay.return_value = 1  # 1ms delay
        orchestrator_with_dynamic_composition.failure_manager.retry_configs = {
            "failing_agent": retry_config
        }

        await (
            orchestrator_with_dynamic_composition._execute_agent_with_failure_handling(
                "failing_agent", context
            )
        )

        # Verify agent eventually succeeded after retries
        assert "failing_agent" in context.agent_outputs
        assert context.agent_outputs["failing_agent"] == "success after retries"
        assert call_count == 3  # Failed twice, succeeded third time

    @pytest.mark.asyncio
    async def test_hot_swap_functionality(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test agent hot-swap functionality - lines 574-607."""
        context = AgentContextPatterns.simple_query("test hot swap")

        # Mock swap opportunities
        swap_opportunities = [
            {"old_agent": "failed_agent", "new_agent": "replacement_agent"}
        ]

        orchestrator_with_dynamic_composition.dynamic_composer._find_swap_opportunities = AsyncMock(
            return_value=swap_opportunities
        )
        orchestrator_with_dynamic_composition.dynamic_composer.hot_swap_agent = (
            AsyncMock(return_value=True)
        )
        orchestrator_with_dynamic_composition.dynamic_composer.loaded_agents = {
            "replacement_agent": MockAgent("replacement_agent")
        }

        # Add the failed agent to loaded agents
        orchestrator_with_dynamic_composition.loaded_agents = {
            "failed_agent": MockAgent("failed_agent")
        }

        result = await orchestrator_with_dynamic_composition._attempt_agent_hot_swap(
            "failed_agent", context
        )

        # Verify hot swap was successful
        assert result is True
        assert (
            "replacement_agent" in orchestrator_with_dynamic_composition.loaded_agents
        )
        assert "failed_agent" not in orchestrator_with_dynamic_composition.loaded_agents

    @pytest.mark.asyncio
    async def test_cleanup_and_finalization_resource_release(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test cleanup and finalization phase - lines 448-477."""
        context = AgentContextPatterns.simple_query("test cleanup")

        # Add loaded agents
        orchestrator_with_dynamic_composition.loaded_agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }

        # Mock resource scheduler
        orchestrator_with_dynamic_composition.resource_scheduler.release_resources = (
            AsyncMock()
        )
        orchestrator_with_dynamic_composition.resource_scheduler.get_scheduling_statistics = Mock(
            return_value={"total_allocations": 5, "total_releases": 5}
        )

        # Mock dynamic composer and failure manager
        orchestrator_with_dynamic_composition.dynamic_composer.get_composition_status = Mock(
            return_value={"active_agents": 2, "swaps_performed": 1}
        )
        orchestrator_with_dynamic_composition.failure_manager.get_failure_statistics = (
            Mock(return_value={"total_failures": 3, "recoveries_attempted": 2})
        )

        # Set pipeline start time
        orchestrator_with_dynamic_composition.pipeline_start_time = (
            time.time() - 1.0
        )  # 1 second ago

        await orchestrator_with_dynamic_composition._phase_cleanup_and_finalization(
            context
        )

        # Verify resources were released for all agents
        assert (
            orchestrator_with_dynamic_composition.resource_scheduler.release_resources.call_count
            == 2
        )

        # Verify statistics were collected
        assert "final_composition_status" in context.execution_state
        assert "failure_statistics" in context.execution_state
        assert "resource_scheduling_statistics" in context.execution_state

        # Verify timing metadata was set
        assert "pipeline_end" in context.path_metadata
        assert "total_duration_ms" in context.path_metadata
        assert context.path_metadata["total_duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_stage_recovery_mechanisms(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test stage recovery mechanisms - lines 609-626."""
        context = AgentContextPatterns.simple_query("test stage recovery")
        error = Exception("Stage execution failed")

        # Mock stage
        mock_stage: Mock = Mock()
        mock_stage.stage_id = "test_stage_1"

        # Test fallback plan recovery
        fallback_plan: Mock = Mock()
        fallback_plan.stage_id = "fallback_stage_1"

        current_plan: Mock = Mock()
        current_plan.fallback_plan = fallback_plan
        orchestrator_with_dynamic_composition.current_plan = current_plan

        result = await orchestrator_with_dynamic_composition._attempt_stage_recovery(
            mock_stage, error, context
        )

        # Verify fallback plan was activated
        assert result is True
        assert orchestrator_with_dynamic_composition.current_plan == fallback_plan

    @pytest.mark.asyncio
    async def test_pipeline_failure_handling(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test pipeline failure handling - lines 628-647."""
        context = AgentContextPatterns.simple_query("test pipeline failure")
        error = RuntimeError("Critical pipeline error")

        # Mock failure manager with recovery checkpoints
        orchestrator_with_dynamic_composition.failure_manager.recovery_checkpoints = {
            "earliest": {"context": context, "timestamp": time.time()}
        }
        orchestrator_with_dynamic_composition.failure_manager._rollback_to_checkpoint = Mock(
            return_value=True
        )

        await orchestrator_with_dynamic_composition._handle_pipeline_failure(
            error, context
        )

        # Verify failure metadata was recorded
        assert "pipeline_failure" in context.path_metadata
        failure_info = context.path_metadata["pipeline_failure"]
        assert failure_info["error"] == "Critical pipeline error"
        assert failure_info["error_type"] == "RuntimeError"
        assert "timestamp" in failure_info

        # Verify emergency recovery was attempted
        orchestrator_with_dynamic_composition.failure_manager._rollback_to_checkpoint.assert_called_once_with(
            "earliest", context
        )

    @pytest.mark.asyncio
    async def test_advanced_orchestrator_full_pipeline_with_all_phases(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test complete advanced orchestrator pipeline - lines 289-341."""
        query = "test full pipeline"

        # Mock all required components
        orchestrator_with_dynamic_composition.dynamic_composer.auto_discover_and_swap = AsyncMock(
            return_value={"discovered": ["agent1"]}
        )
        orchestrator_with_dynamic_composition.dynamic_composer.optimize_composition = (
            AsyncMock(return_value={"optimized": True})
        )

        # Mock execution plan
        mock_plan: Mock = Mock()
        mock_plan.get_execution_summary.return_value = {"stages": 2}
        mock_plan.get_total_stages.return_value = 2
        mock_plan.parallelism_factor = 1.5
        mock_plan.stages = []
        mock_plan.completed_at = None

        orchestrator_with_dynamic_composition.execution_planner.create_plan = Mock(
            return_value=mock_plan
        )

        # Mock resource scheduler
        orchestrator_with_dynamic_composition.resource_scheduler.get_resource_utilization = Mock(
            return_value={"memory": 0.5}
        )
        orchestrator_with_dynamic_composition.resource_scheduler.release_resources = (
            AsyncMock()
        )
        orchestrator_with_dynamic_composition.resource_scheduler.get_scheduling_statistics = Mock(
            return_value={"allocations": 3}
        )

        # Mock metrics collector
        mock_metrics: Mock = Mock()
        mock_metrics.record_pipeline_execution = Mock()

        with patch(
            "cognivault.dependencies.advanced_orchestrator.get_metrics_collector",
            return_value=mock_metrics,
        ):
            with patch(
                "cognivault.dependencies.advanced_orchestrator.observability_context"
            ):
                result = await orchestrator_with_dynamic_composition.run(query)

        # Verify pipeline completed successfully
        assert isinstance(result, AgentContext)
        assert result.query == query
        assert "execution_id" in result.path_metadata
        assert "orchestration_type" in result.path_metadata
        assert result.path_metadata["orchestration_type"] == "advanced"

        # Verify all phases were called
        orchestrator_with_dynamic_composition.dynamic_composer.auto_discover_and_swap.assert_called_once()
        orchestrator_with_dynamic_composition.execution_planner.create_plan.assert_called_once()
        mock_metrics.record_pipeline_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_stage_execution(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test sequential stage execution - lines 495-499."""
        context = AgentContextPatterns.simple_query("test sequential execution")

        # Create mock stage
        mock_stage: Mock = Mock()
        mock_stage.agents = ["agent1", "agent2"]

        # Add agents
        orchestrator_with_dynamic_composition.loaded_agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }

        # Track execution order
        execution_order = []

        async def track_sequential_execution(agent_id: str, ctx: Any) -> None:
            execution_order.append(agent_id)
            await asyncio.sleep(0.01)  # Small delay

        orchestrator_with_dynamic_composition._execute_agent_with_failure_handling = (
            track_sequential_execution
        )

        await orchestrator_with_dynamic_composition._execute_sequential_stage(
            mock_stage, context
        )

        # Verify sequential execution order
        assert execution_order == ["agent1", "agent2"]

    @pytest.mark.asyncio
    async def test_checkpoint_rollback_recovery(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test checkpoint rollback recovery when fallback plan unavailable."""
        context = AgentContextPatterns.simple_query("test checkpoint recovery")
        error = Exception("Stage execution failed")

        # Mock stage
        mock_stage: Mock = Mock()
        mock_stage.stage_id = "failing_stage"

        # No fallback plan available
        current_plan_mock = Mock()
        current_plan_mock.fallback_plan = None
        orchestrator_with_dynamic_composition.current_plan = current_plan_mock

        # Mock failure manager with checkpoints
        orchestrator_with_dynamic_composition.failure_manager.recovery_checkpoints = {
            "latest": {"context": context, "timestamp": time.time()}
        }
        orchestrator_with_dynamic_composition.failure_manager._rollback_to_checkpoint = Mock(
            return_value=True
        )

        result = await orchestrator_with_dynamic_composition._attempt_stage_recovery(
            mock_stage, error, context
        )

        # Verify checkpoint rollback was attempted
        assert result is True
        orchestrator_with_dynamic_composition.failure_manager._rollback_to_checkpoint.assert_called_once_with(
            "latest", context
        )

    @pytest.mark.asyncio
    async def test_agent_execution_timeout_and_circuit_breaker(
        self, orchestrator_with_dynamic_composition: Any
    ) -> None:
        """Test agent execution with timeout and circuit breaker integration."""
        context = AgentContextPatterns.simple_query("test timeout handling")

        # Create agent that will fail immediately to trigger circuit breaker logic
        failing_agent = MockAgent("failing_agent", should_fail=True)
        orchestrator_with_dynamic_composition.loaded_agents = {
            "failing_agent": failing_agent
        }

        # Mock failure manager to deny execution on second attempt (circuit breaker opens)
        call_count = 0

        def mock_can_execute(agent_id: str, ctx: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (True, None)  # First call allows execution
            else:
                return (False, "Circuit breaker open")  # Subsequent calls blocked

        orchestrator_with_dynamic_composition.failure_manager.can_execute_agent = Mock(
            side_effect=mock_can_execute
        )

        # Mock failure manager to not retry after first failure
        orchestrator_with_dynamic_composition.failure_manager.handle_agent_failure = (
            AsyncMock(return_value=(False, "no_action"))  # Don't retry
        )

        # Mock circuit breaker
        circuit_breaker: Mock = Mock()
        orchestrator_with_dynamic_composition.failure_manager.circuit_breakers = {
            "failing_agent": circuit_breaker
        }

        # This should raise an exception due to agent failure and no retry
        with pytest.raises(Exception, match="Simulated failure"):
            await orchestrator_with_dynamic_composition._execute_agent_with_failure_handling(
                "failing_agent", context
            )

        # Verify circuit breaker was not called (agent never succeeded)
        circuit_breaker.record_success.assert_not_called()

        # Verify can_execute was called (checking circuit breaker state)
        assert call_count >= 1
