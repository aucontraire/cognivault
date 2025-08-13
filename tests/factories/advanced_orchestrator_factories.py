"""Factory functions for creating advanced orchestrator test data objects.

This module provides factory functions for creating test data objects that relate to
advanced orchestration, dependency management, resource allocation, and pipeline execution
in CogniVault. These factories eliminate parameter unfilled warnings and reduce test code
duplication for orchestration-related testing.

Design Principles:
- Factory methods with sensible defaults for common test scenarios
- Specialized factory methods for edge cases and invalid data
- Type-safe factory returns matching schema definitions
- Easy override of specific fields for test customization

Convenience Methods:
All factories include three convenience methods to reduce verbose parameter passing:

- generate_valid_data(**overrides) - Standard valid object for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid object with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp instead of fixed dates

Usage Examples:
    # Simple usage - zero parameters
    config = OrchestratorConfigFactory.generate_valid_data()

    # With customization - only specify what you need
    config = OrchestratorConfigFactory.generate_valid_data(
        max_concurrent_agents=8,
        enable_dynamic_composition=True
    )

    # Minimal for lightweight tests
    result = ResourceAllocationResultFactory.generate_minimal_data()
"""

from typing import List, Any, Optional, TypedDict, cast

from cognivault.dependencies.advanced_orchestrator import (
    OrchestratorConfig,
    ExecutionResults,
    ResourceAllocationResult,
    PipelineStage,
    ExecutionPhase,
)
from cognivault.dependencies.graph_engine import (
    ResourceConstraint,
    DependencyNode,
    ExecutionPriority,
)
from cognivault.dependencies.execution_planner import ExecutionStrategy
from cognivault.dependencies.failure_manager import CascadePreventionStrategy
from cognivault.dependencies.resource_scheduler import ResourceType
from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext


class TestAgent(BaseAgent):
    """Minimal test agent for factory testing purposes."""

    def __init__(self, name: str = "test_agent") -> None:
        super().__init__(name)

    async def run(self, context: AgentContext) -> AgentContext:
        """Simple test implementation."""
        return context


class OrchestratorConfigParams(TypedDict, total=False):
    """TypedDict for OrchestratorConfig factory method parameters."""

    max_concurrent_agents: int
    enable_failure_recovery: bool
    enable_resource_scheduling: bool
    enable_dynamic_composition: bool
    default_execution_strategy: ExecutionStrategy
    cascade_prevention_strategy: CascadePreventionStrategy
    pipeline_timeout_ms: int
    resource_allocation_timeout_ms: int


class AdvancedResourceConstraintFactory:
    """Factory for creating ResourceConstraint test objects."""

    @staticmethod
    def basic(
        resource_type: str = "cpu",
        max_usage: float = 50.0,
        units: str = "percentage",
        shared: bool = False,
        renewable: bool = True,
    ) -> ResourceConstraint:
        """Create basic ResourceConstraint with sensible defaults."""
        return ResourceConstraint(
            resource_type=resource_type,
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=renewable,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ResourceConstraint:
        """Generate standard valid ResourceConstraint for most test scenarios."""
        return AdvancedResourceConstraintFactory.basic(
            resource_type=overrides.get("resource_type", "memory"),
            max_usage=overrides.get("max_usage", 512.0),
            units=overrides.get("units", "MB"),
            shared=overrides.get("shared", False),
            renewable=overrides.get("renewable", True),
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> ResourceConstraint:
        """Generate minimal valid ResourceConstraint for lightweight test scenarios."""
        return AdvancedResourceConstraintFactory.basic(
            resource_type=overrides.get("resource_type", "cpu"),
            max_usage=overrides.get("max_usage", 25.0),
            units=overrides.get("units", "percentage"),
            shared=overrides.get("shared", False),
            renewable=overrides.get("renewable", True),
        )

    @staticmethod
    def memory_constraint(**overrides: Any) -> ResourceConstraint:
        """Create memory-specific resource constraint."""
        return AdvancedResourceConstraintFactory.basic(
            resource_type=overrides.get("resource_type", "memory"),
            max_usage=overrides.get("max_usage", 1024.0),
            units=overrides.get("units", "MB"),
            shared=overrides.get("shared", False),
            renewable=overrides.get("renewable", True),
        )

    @staticmethod
    def cpu_constraint(**overrides: Any) -> ResourceConstraint:
        """Create CPU-specific resource constraint."""
        return AdvancedResourceConstraintFactory.basic(
            resource_type=overrides.get("resource_type", "cpu"),
            max_usage=overrides.get("max_usage", 75.0),
            units=overrides.get("units", "percentage"),
            shared=overrides.get("shared", False),
            renewable=overrides.get("renewable", True),
        )


class OrchestratorConfigFactory:
    """Factory for creating OrchestratorConfig test objects."""

    @staticmethod
    def basic(
        max_concurrent_agents: int = 4,
        enable_failure_recovery: bool = True,
        enable_resource_scheduling: bool = True,
        enable_dynamic_composition: bool = False,
        default_execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
        cascade_prevention_strategy: CascadePreventionStrategy = CascadePreventionStrategy.GRACEFUL_DEGRADATION,
        pipeline_timeout_ms: int = 60000,
        resource_allocation_timeout_ms: int = 10000,
    ) -> OrchestratorConfig:
        """Create basic OrchestratorConfig with sensible defaults."""
        return OrchestratorConfig(
            max_concurrent_agents=max_concurrent_agents,
            enable_failure_recovery=enable_failure_recovery,
            enable_resource_scheduling=enable_resource_scheduling,
            enable_dynamic_composition=enable_dynamic_composition,
            default_execution_strategy=default_execution_strategy,
            cascade_prevention_strategy=cascade_prevention_strategy,
            pipeline_timeout_ms=pipeline_timeout_ms,
            resource_allocation_timeout_ms=resource_allocation_timeout_ms,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> OrchestratorConfig:
        """Generate standard valid OrchestratorConfig for most test scenarios."""
        defaults: OrchestratorConfigParams = {
            "max_concurrent_agents": 3,
            "enable_failure_recovery": True,
            "enable_resource_scheduling": True,
            "enable_dynamic_composition": False,
            "default_execution_strategy": ExecutionStrategy.PARALLEL_BATCHED,
            "cascade_prevention_strategy": CascadePreventionStrategy.GRACEFUL_DEGRADATION,
            "pipeline_timeout_ms": 30000,
            "resource_allocation_timeout_ms": 5000,
        }
        # Apply overrides to defaults (cast needed for TypedDict compatibility)
        defaults.update(cast(OrchestratorConfigParams, overrides))
        return OrchestratorConfigFactory.basic(
            max_concurrent_agents=defaults["max_concurrent_agents"],
            enable_failure_recovery=defaults["enable_failure_recovery"],
            enable_resource_scheduling=defaults["enable_resource_scheduling"],
            enable_dynamic_composition=defaults["enable_dynamic_composition"],
            default_execution_strategy=defaults["default_execution_strategy"],
            cascade_prevention_strategy=defaults["cascade_prevention_strategy"],
            pipeline_timeout_ms=defaults["pipeline_timeout_ms"],
            resource_allocation_timeout_ms=defaults["resource_allocation_timeout_ms"],
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> OrchestratorConfig:
        """Generate minimal valid OrchestratorConfig for lightweight test scenarios."""
        defaults: OrchestratorConfigParams = {
            "max_concurrent_agents": 2,
            "enable_failure_recovery": False,
            "enable_resource_scheduling": False,
            "enable_dynamic_composition": False,
            "default_execution_strategy": ExecutionStrategy.SEQUENTIAL,
        }
        # Apply overrides to defaults (cast needed for TypedDict compatibility)
        defaults.update(cast(OrchestratorConfigParams, overrides))
        return OrchestratorConfigFactory.basic(
            max_concurrent_agents=defaults["max_concurrent_agents"],
            enable_failure_recovery=defaults["enable_failure_recovery"],
            enable_resource_scheduling=defaults["enable_resource_scheduling"],
            enable_dynamic_composition=defaults["enable_dynamic_composition"],
            default_execution_strategy=defaults["default_execution_strategy"],
            cascade_prevention_strategy=defaults.get(
                "cascade_prevention_strategy",
                CascadePreventionStrategy.GRACEFUL_DEGRADATION,
            ),
            pipeline_timeout_ms=defaults.get("pipeline_timeout_ms", 60000),
            resource_allocation_timeout_ms=defaults.get(
                "resource_allocation_timeout_ms", 10000
            ),
        )

    @staticmethod
    def high_performance(**overrides: Any) -> OrchestratorConfig:
        """Create high-performance orchestrator configuration."""
        defaults: OrchestratorConfigParams = {
            "max_concurrent_agents": 8,
            "enable_failure_recovery": True,
            "enable_resource_scheduling": True,
            "enable_dynamic_composition": True,
            "default_execution_strategy": ExecutionStrategy.ADAPTIVE,
        }
        # Apply overrides to defaults (cast needed for TypedDict compatibility)
        defaults.update(cast(OrchestratorConfigParams, overrides))
        return OrchestratorConfigFactory.basic(
            max_concurrent_agents=defaults["max_concurrent_agents"],
            enable_failure_recovery=defaults["enable_failure_recovery"],
            enable_resource_scheduling=defaults["enable_resource_scheduling"],
            enable_dynamic_composition=defaults["enable_dynamic_composition"],
            default_execution_strategy=defaults["default_execution_strategy"],
            cascade_prevention_strategy=defaults.get(
                "cascade_prevention_strategy",
                CascadePreventionStrategy.GRACEFUL_DEGRADATION,
            ),
            pipeline_timeout_ms=defaults.get("pipeline_timeout_ms", 60000),
            resource_allocation_timeout_ms=defaults.get(
                "resource_allocation_timeout_ms", 10000
            ),
        )


class ExecutionResultsFactory:
    """Factory for creating ExecutionResults test objects."""

    @staticmethod
    def basic(
        success: bool = True,
        total_agents_executed: int = 3,
        successful_agents: int = 3,
        failed_agents: int = 0,
        execution_time_ms: float = 1500.0,
        pipeline_stages: Optional[List[PipelineStage]] = None,
        resource_allocation_results: Optional[List[ResourceAllocationResult]] = None,
        failure_recovery_actions: Optional[List[str]] = None,
        **overrides: Any,
    ) -> ExecutionResults:
        """Create basic ExecutionResults with sensible defaults."""
        if pipeline_stages is None:
            pipeline_stages = []
        if resource_allocation_results is None:
            resource_allocation_results = []
        if failure_recovery_actions is None:
            failure_recovery_actions = []

        return ExecutionResults(
            success=success,
            total_agents_executed=total_agents_executed,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            execution_time_ms=execution_time_ms,
            pipeline_stages=pipeline_stages,
            resource_allocation_results=resource_allocation_results,
            failure_recovery_actions=failure_recovery_actions,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ExecutionResults:
        """Generate standard valid ExecutionResults for most test scenarios."""
        # Create realistic pipeline stages
        pipeline_stages = [
            PipelineStageFactory.basic(
                stage_id="preparation",
                phase=ExecutionPhase.PREPARATION,
                agents_executed=[],
                stage_duration_ms=200.0,
                success=True,
            ),
            PipelineStageFactory.basic(
                stage_id="execution",
                phase=ExecutionPhase.EXECUTION,
                agents_executed=["refiner", "critic", "historian"],
                stage_duration_ms=1200.0,
                success=True,
            ),
            PipelineStageFactory.basic(
                stage_id="cleanup",
                phase=ExecutionPhase.CLEANUP,
                agents_executed=[],
                stage_duration_ms=100.0,
                success=True,
            ),
        ]

        return ExecutionResultsFactory.basic(
            success=overrides.get("success", True),
            total_agents_executed=overrides.get("total_agents_executed", 3),
            successful_agents=overrides.get("successful_agents", 3),
            failed_agents=overrides.get("failed_agents", 0),
            execution_time_ms=overrides.get("execution_time_ms", 1500.0),
            pipeline_stages=overrides.get("pipeline_stages", pipeline_stages),
            resource_allocation_results=overrides.get("resource_allocation_results"),
            failure_recovery_actions=overrides.get("failure_recovery_actions"),
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> ExecutionResults:
        """Generate minimal valid ExecutionResults for lightweight test scenarios."""
        return ExecutionResultsFactory.basic(
            success=overrides.get("success", True),
            total_agents_executed=overrides.get("total_agents_executed", 1),
            successful_agents=overrides.get("successful_agents", 1),
            failed_agents=overrides.get("failed_agents", 0),
            execution_time_ms=overrides.get("execution_time_ms", 500.0),
        )

    @staticmethod
    def with_failures(**overrides: Any) -> ExecutionResults:
        """Create ExecutionResults with some failures."""
        return ExecutionResultsFactory.basic(
            success=overrides.get("success", False),
            total_agents_executed=overrides.get("total_agents_executed", 4),
            successful_agents=overrides.get("successful_agents", 2),
            failed_agents=overrides.get("failed_agents", 2),
            execution_time_ms=overrides.get("execution_time_ms", 2000.0),
            failure_recovery_actions=overrides.get(
                "failure_recovery_actions", ["retry", "graceful_degradation"]
            ),
        )


class ResourceAllocationResultFactory:
    """Factory for creating ResourceAllocationResult test objects."""

    @staticmethod
    def basic(
        agent_id: str = "test_agent",
        resource_type: ResourceType = ResourceType.CPU,
        requested_amount: float = 50.0,
        allocated_amount: float = 50.0,
        allocation_time_ms: float = 100.0,
        success: bool = True,
        **overrides: Any,
    ) -> ResourceAllocationResult:
        """Create basic ResourceAllocationResult with sensible defaults."""
        return ResourceAllocationResult(
            agent_id=agent_id,
            resource_type=resource_type,
            requested_amount=requested_amount,
            allocated_amount=allocated_amount,
            allocation_time_ms=allocation_time_ms,
            success=success,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ResourceAllocationResult:
        """Generate standard valid ResourceAllocationResult for most test scenarios."""
        return ResourceAllocationResultFactory.basic(
            agent_id=overrides.get("agent_id", "refiner_agent"),
            resource_type=overrides.get("resource_type", ResourceType.MEMORY),
            requested_amount=overrides.get("requested_amount", 512.0),
            allocated_amount=overrides.get("allocated_amount", 512.0),
            allocation_time_ms=overrides.get("allocation_time_ms", 150.0),
            success=overrides.get("success", True),
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> ResourceAllocationResult:
        """Generate minimal valid ResourceAllocationResult for lightweight test scenarios."""
        return ResourceAllocationResultFactory.basic(
            agent_id=overrides.get("agent_id", "minimal_agent"),
            resource_type=overrides.get("resource_type", ResourceType.CPU),
            requested_amount=overrides.get("requested_amount", 25.0),
            allocated_amount=overrides.get("allocated_amount", 25.0),
            allocation_time_ms=overrides.get("allocation_time_ms", 50.0),
            success=overrides.get("success", True),
        )

    @staticmethod
    def partial_allocation(**overrides: Any) -> ResourceAllocationResult:
        """Create ResourceAllocationResult with partial allocation."""
        return ResourceAllocationResultFactory.basic(
            agent_id=overrides.get("agent_id", "resource_heavy_agent"),
            resource_type=overrides.get("resource_type", ResourceType.MEMORY),
            requested_amount=overrides.get("requested_amount", 2048.0),
            allocated_amount=overrides.get(
                "allocated_amount", 1536.0
            ),  # Only got 75% of requested
            allocation_time_ms=overrides.get("allocation_time_ms", 300.0),
            success=overrides.get("success", True),
        )

    @staticmethod
    def failed_allocation(**overrides: Any) -> ResourceAllocationResult:
        """Create ResourceAllocationResult for failed allocation."""
        return ResourceAllocationResultFactory.basic(
            agent_id=overrides.get("agent_id", "failed_agent"),
            resource_type=overrides.get("resource_type", ResourceType.MEMORY),
            requested_amount=overrides.get("requested_amount", 4096.0),
            allocated_amount=overrides.get("allocated_amount", 0.0),
            allocation_time_ms=overrides.get("allocation_time_ms", 50.0),
            success=overrides.get("success", False),
        )


class PipelineStageFactory:
    """Factory for creating PipelineStage test objects."""

    @staticmethod
    def basic(
        stage_id: str = "test_stage",
        phase: ExecutionPhase = ExecutionPhase.EXECUTION,
        agents_executed: Optional[List[str]] = None,
        stage_duration_ms: float = 1000.0,
        success: bool = True,
        **overrides: Any,
    ) -> PipelineStage:
        """Create basic PipelineStage with sensible defaults."""
        if agents_executed is None:
            agents_executed = ["test_agent"]

        return PipelineStage(
            stage_id=stage_id,
            phase=phase,
            agents_executed=agents_executed,
            stage_duration_ms=stage_duration_ms,
            success=success,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> PipelineStage:
        """Generate standard valid PipelineStage for most test scenarios."""
        return PipelineStageFactory.basic(
            stage_id=overrides.get("stage_id", "execution_stage_001"),
            phase=overrides.get("phase", ExecutionPhase.EXECUTION),
            agents_executed=overrides.get(
                "agents_executed", ["refiner", "critic", "historian"]
            ),
            stage_duration_ms=overrides.get("stage_duration_ms", 1200.0),
            success=overrides.get("success", True),
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> PipelineStage:
        """Generate minimal valid PipelineStage for lightweight test scenarios."""
        return PipelineStageFactory.basic(
            stage_id=overrides.get("stage_id", "minimal_stage"),
            phase=overrides.get("phase", ExecutionPhase.EXECUTION),
            agents_executed=overrides.get("agents_executed", ["single_agent"]),
            stage_duration_ms=overrides.get("stage_duration_ms", 500.0),
            success=overrides.get("success", True),
        )

    @staticmethod
    def preparation_stage(**overrides: Any) -> PipelineStage:
        """Create preparation phase pipeline stage."""
        return PipelineStageFactory.basic(
            stage_id=overrides.get("stage_id", "preparation"),
            phase=overrides.get("phase", ExecutionPhase.PREPARATION),
            agents_executed=overrides.get("agents_executed", []),
            stage_duration_ms=overrides.get("stage_duration_ms", 200.0),
            success=overrides.get("success", True),
        )

    @staticmethod
    def resource_allocation_stage(**overrides: Any) -> PipelineStage:
        """Create resource allocation phase pipeline stage."""
        return PipelineStageFactory.basic(
            stage_id=overrides.get("stage_id", "resource_allocation"),
            phase=overrides.get("phase", ExecutionPhase.RESOURCE_ALLOCATION),
            agents_executed=overrides.get("agents_executed", ["refiner", "critic"]),
            stage_duration_ms=overrides.get("stage_duration_ms", 300.0),
            success=overrides.get("success", True),
        )

    @staticmethod
    def cleanup_stage(**overrides: Any) -> PipelineStage:
        """Create cleanup phase pipeline stage."""
        return PipelineStageFactory.basic(
            stage_id=overrides.get("stage_id", "cleanup"),
            phase=overrides.get("phase", ExecutionPhase.CLEANUP),
            agents_executed=overrides.get("agents_executed", []),
            stage_duration_ms=overrides.get("stage_duration_ms", 100.0),
            success=overrides.get("success", True),
        )


class DependencyNodeFactory:
    """Factory for creating DependencyNode test objects."""

    @staticmethod
    def basic(
        agent_id: str = "test_agent",
        agent: Any = None,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        resource_constraints: Optional[List[ResourceConstraint]] = None,
        max_retries: int = 3,
        timeout_ms: int = 5000,
        can_run_parallel: bool = True,
        requires_exclusive_access: bool = False,
        **overrides: Any,
    ) -> DependencyNode:
        """Create basic DependencyNode with sensible defaults."""
        if agent is None:
            # Create a test agent for testing
            agent = TestAgent(agent_id)

        if resource_constraints is None:
            resource_constraints = []

        return DependencyNode(
            agent_id=agent_id,
            agent=agent,
            priority=priority,
            resource_constraints=resource_constraints,
            max_retries=max_retries,
            timeout_ms=timeout_ms,
            can_run_parallel=can_run_parallel,
            requires_exclusive_access=requires_exclusive_access,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> DependencyNode:
        """Generate standard valid DependencyNode for most test scenarios."""
        resource_constraints = [
            AdvancedResourceConstraintFactory.cpu_constraint(),
            AdvancedResourceConstraintFactory.memory_constraint(),
        ]

        return DependencyNodeFactory.basic(
            agent_id=overrides.get("agent_id", "refiner_agent"),
            agent=overrides.get("agent"),
            priority=overrides.get("priority", ExecutionPriority.NORMAL),
            resource_constraints=overrides.get(
                "resource_constraints", resource_constraints
            ),
            max_retries=overrides.get("max_retries", 3),
            timeout_ms=overrides.get("timeout_ms", 10000),
            can_run_parallel=overrides.get("can_run_parallel", True),
            requires_exclusive_access=overrides.get("requires_exclusive_access", False),
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> DependencyNode:
        """Generate minimal valid DependencyNode for lightweight test scenarios."""
        return DependencyNodeFactory.basic(
            agent_id=overrides.get("agent_id", "minimal_agent"),
            agent=overrides.get("agent"),
            priority=overrides.get("priority", ExecutionPriority.NORMAL),
            resource_constraints=overrides.get("resource_constraints"),
            max_retries=overrides.get("max_retries", 3),
            timeout_ms=overrides.get("timeout_ms", 3000),
            can_run_parallel=overrides.get("can_run_parallel", True),
            requires_exclusive_access=overrides.get("requires_exclusive_access", False),
        )

    @staticmethod
    def high_priority(**overrides: Any) -> DependencyNode:
        """Create high priority dependency node."""
        return DependencyNodeFactory.basic(
            agent_id=overrides.get("agent_id", "critical_agent"),
            agent=overrides.get("agent"),
            priority=overrides.get("priority", ExecutionPriority.HIGH),
            resource_constraints=overrides.get("resource_constraints"),
            max_retries=overrides.get("max_retries", 3),
            timeout_ms=overrides.get("timeout_ms", 15000),
            can_run_parallel=overrides.get("can_run_parallel", True),
            requires_exclusive_access=overrides.get("requires_exclusive_access", False),
        )

    @staticmethod
    def with_exclusive_access(**overrides: Any) -> DependencyNode:
        """Create dependency node requiring exclusive access."""
        return DependencyNodeFactory.basic(
            agent_id=overrides.get("agent_id", "exclusive_agent"),
            agent=overrides.get("agent"),
            priority=overrides.get("priority", ExecutionPriority.HIGH),
            resource_constraints=overrides.get("resource_constraints"),
            max_retries=overrides.get("max_retries", 3),
            timeout_ms=overrides.get("timeout_ms", 5000),
            can_run_parallel=overrides.get("can_run_parallel", False),
            requires_exclusive_access=overrides.get("requires_exclusive_access", True),
        )

    @staticmethod
    def resource_intensive(**overrides: Any) -> DependencyNode:
        """Create resource-intensive dependency node."""
        high_resource_constraints = [
            AdvancedResourceConstraintFactory.basic(
                resource_type="memory", max_usage=4096.0, units="MB"
            ),
            AdvancedResourceConstraintFactory.basic(
                resource_type="cpu", max_usage=90.0, units="percentage"
            ),
        ]

        return DependencyNodeFactory.basic(
            agent_id=overrides.get("agent_id", "resource_intensive_agent"),
            agent=overrides.get("agent"),
            priority=overrides.get("priority", ExecutionPriority.HIGH),
            resource_constraints=overrides.get(
                "resource_constraints", high_resource_constraints
            ),
            max_retries=overrides.get("max_retries", 3),
            timeout_ms=overrides.get("timeout_ms", 20000),
            can_run_parallel=overrides.get("can_run_parallel", True),
            requires_exclusive_access=overrides.get("requires_exclusive_access", False),
        )


# Convenience patterns for common test scenarios
class AdvancedOrchestratorTestPatterns:
    """Common patterns for advanced orchestrator testing."""

    @staticmethod
    def simple_pipeline_execution() -> ExecutionResults:
        """Create results for simple successful pipeline execution."""
        stages = [
            PipelineStageFactory.preparation_stage(),
            PipelineStageFactory.generate_valid_data(),
            PipelineStageFactory.cleanup_stage(),
        ]

        return ExecutionResultsFactory.basic(
            success=True,
            total_agents_executed=3,
            successful_agents=3,
            failed_agents=0,
            execution_time_ms=1500.0,
            pipeline_stages=stages,
        )

    @staticmethod
    def failed_pipeline_execution() -> ExecutionResults:
        """Create results for failed pipeline execution with recovery."""
        stages = [
            PipelineStageFactory.preparation_stage(),
            PipelineStageFactory.basic(
                stage_id="failed_execution",
                phase=ExecutionPhase.EXECUTION,
                agents_executed=["refiner", "failing_agent"],
                stage_duration_ms=2000.0,
                success=False,
            ),
            PipelineStageFactory.cleanup_stage(),
        ]

        return ExecutionResultsFactory.basic(
            success=False,
            total_agents_executed=2,
            successful_agents=1,
            failed_agents=1,
            execution_time_ms=2300.0,
            pipeline_stages=stages,
            failure_recovery_actions=["retry", "graceful_degradation"],
        )

    @staticmethod
    def high_performance_config() -> OrchestratorConfig:
        """Create configuration optimized for high performance."""
        return OrchestratorConfigFactory.high_performance(
            max_concurrent_agents=6,
            pipeline_timeout_ms=120000,
            resource_allocation_timeout_ms=15000,
        )

    @staticmethod
    def resource_constrained_nodes() -> List[DependencyNode]:
        """Create a list of nodes with various resource constraints."""
        return [
            DependencyNodeFactory.generate_valid_data(agent_id="lightweight_agent"),
            DependencyNodeFactory.resource_intensive(agent_id="heavy_agent"),
            DependencyNodeFactory.with_exclusive_access(agent_id="exclusive_agent"),
        ]
