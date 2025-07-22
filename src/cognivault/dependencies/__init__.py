"""
Advanced dependency management system for CogniVault agents.

This package provides sophisticated dependency graph execution with topological
ordering, circular dependency detection, failure propagation, and resource scheduling.
"""

from .graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
    DependencyEdge,
    DependencyType,
    ExecutionPriority,
    ResourceConstraint,
)
from .execution_planner import (
    ExecutionPlanner,
    ExecutionPlan,
    ExecutionStage,
    ParallelGroup,
    ExecutionStrategy,
)
from .failure_manager import (
    FailureManager,
    CascadePreventionStrategy,
    RetryConfiguration,
    FailureImpactAnalysis,
)
from .resource_scheduler import (
    ResourceScheduler,
    ResourcePool,
    ResourceRequest,
    SchedulingPolicy,
)
from .dynamic_composition import (
    DynamicAgentComposer,
    DiscoveredAgentInfo,
    CompositionRule,
    DiscoveryStrategy,
)

__all__ = [
    "DependencyGraphEngine",
    "DependencyNode",
    "DependencyEdge",
    "DependencyType",
    "ExecutionPriority",
    "ResourceConstraint",
    "ExecutionPlanner",
    "ExecutionPlan",
    "ExecutionStage",
    "ParallelGroup",
    "ExecutionStrategy",
    "FailureManager",
    "CascadePreventionStrategy",
    "RetryConfiguration",
    "FailureImpactAnalysis",
    "ResourceScheduler",
    "ResourcePool",
    "ResourceRequest",
    "SchedulingPolicy",
    "DynamicAgentComposer",
    "DiscoveredAgentInfo",
    "CompositionRule",
    "DiscoveryStrategy",
]
