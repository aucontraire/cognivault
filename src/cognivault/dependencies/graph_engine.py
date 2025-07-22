"""
Advanced dependency graph engine with topological ordering and circular dependency detection.

This module provides sophisticated dependency management that extends beyond simple
sequential execution to support complex dependency graphs with validation, optimization,
and advanced routing capabilities.
"""

from collections import defaultdict, deque
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from pydantic import BaseModel, Field, ConfigDict
from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.observability import get_logger

logger = get_logger(__name__)


class DependencyType(Enum):
    """Types of dependencies between agents."""

    HARD = "hard"  # Agent cannot run without this dependency
    SOFT = "soft"  # Agent can run with degraded functionality
    CONDITIONAL = "conditional"  # Dependency based on runtime conditions
    RESOURCE = "resource"  # Shared resource dependency
    DATA = "data"  # Data dependency (output required as input)
    TIMING = "timing"  # Timing constraint dependency


class ExecutionPriority(Enum):
    """Execution priority levels for agents."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ResourceConstraint(BaseModel):
    """
    Resource constraint for agent execution.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    resource_type: str = Field(
        ...,
        description="Type of resource being constrained (e.g., memory, cpu, llm_tokens)",
        min_length=1,
        max_length=50,
        json_schema_extra={"example": "memory"},
    )
    max_usage: float = Field(
        ...,
        description="Maximum usage allowed for this resource",
        gt=0.0,
        json_schema_extra={"example": 8192.0},
    )
    units: str = Field(
        ...,
        description="Units for the resource measurement",
        min_length=1,
        max_length=20,
        json_schema_extra={"example": "MB"},
    )
    shared: bool = Field(
        False,
        description="Whether resource can be shared with other agents",
        json_schema_extra={"example": False},
    )
    renewable: bool = Field(
        True,
        description="Whether resource renews after agent completes execution",
        json_schema_extra={"example": True},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects
    )


class DependencyEdge(BaseModel):
    """
    An edge in the dependency graph representing a dependency relationship.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    from_agent: str = Field(
        ...,
        description="Source agent in the dependency relationship",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "refiner"},
    )
    to_agent: str = Field(
        ...,
        description="Target agent in the dependency relationship",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "historian"},
    )
    dependency_type: DependencyType = Field(
        ...,
        description="Type of dependency relationship",
        json_schema_extra={"example": "hard"},
    )
    condition: Optional[Callable[[AgentContext], bool]] = Field(
        default=None,
        description="Condition function to evaluate dependency satisfaction",
        exclude=True,  # Don't serialize functions
    )
    condition_name: Optional[str] = Field(
        default=None,
        description="Human-readable name for the condition",
        max_length=200,
        json_schema_extra={"example": "check_query_complexity"},
    )
    weight: float = Field(
        default=1.0,
        description="Weight for prioritization (higher = more important)",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 1.0},
    )
    timeout_ms: Optional[int] = Field(
        default=None,
        description="Timeout in milliseconds for dependency evaluation",
        ge=0,
        le=300000,
        json_schema_extra={"example": 30000},
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries attempted for this dependency",
        ge=0,
        le=10,
        json_schema_extra={"example": 0},
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the dependency edge",
        json_schema_extra={"example": {"priority": "high", "source": "user_config"}},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For callable condition
        use_enum_values=False,  # Keep enum objects
    )

    def is_satisfied(self, context: AgentContext) -> bool:
        """Check if dependency condition is satisfied."""
        if self.condition is None:
            return True
        try:
            return self.condition(context)
        except Exception as e:
            logger.warning(f"Error evaluating dependency condition: {e}")
            return False

    def __hash__(self) -> int:
        return hash((self.from_agent, self.to_agent, self.dependency_type))


class DependencyNode(BaseModel):
    """
    A node in the dependency graph representing an agent and its constraints.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    agent_id: str = Field(
        ...,
        description="Unique identifier for the agent",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "refiner"},
    )
    agent: BaseAgent = Field(
        ...,
        description="The agent instance for this node",
        exclude=True,  # Don't serialize agent instances
    )
    priority: ExecutionPriority = Field(
        default=ExecutionPriority.NORMAL,
        description="Execution priority level for this agent",
        json_schema_extra={"example": "normal"},
    )
    resource_constraints: List[ResourceConstraint] = Field(
        default_factory=list,
        description="List of resource constraints for this agent",
        json_schema_extra={"example": []},
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=0,
        le=20,
        json_schema_extra={"example": 3},
    )
    timeout_ms: int = Field(
        default=30000,
        description="Timeout in milliseconds for agent execution",
        ge=100,  # Allow shorter timeouts for testing purposes
        le=600000,
        json_schema_extra={"example": 30000},
    )
    can_run_parallel: bool = Field(
        default=True,
        description="Whether this agent can run in parallel with itself",
        json_schema_extra={"example": True},
    )
    requires_exclusive_access: bool = Field(
        default=False,
        description="Whether this agent requires exclusive system access",
        json_schema_extra={"example": False},
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the dependency node",
        json_schema_extra={"example": {"source": "config", "version": "1.0"}},
    )

    # Runtime state
    execution_count: int = Field(
        default=0,
        description="Number of times this agent has been executed",
        ge=0,
        json_schema_extra={"example": 0},
    )
    last_execution_time_ms: Optional[float] = Field(
        default=None,
        description="Timestamp of last execution in milliseconds",
        json_schema_extra={"example": 1640995200000.0},
    )
    last_error: Optional[Exception] = Field(
        default=None,
        description="Last error encountered during execution",
        exclude=True,  # Don't serialize exceptions
    )
    is_executing: bool = Field(
        default=False,
        description="Whether this agent is currently executing",
        json_schema_extra={"example": False},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For BaseAgent and Exception
        use_enum_values=False,  # Keep enum objects
    )

    def can_execute(self, context: AgentContext) -> bool:
        """Check if node can execute given current context."""
        if self.is_executing and not self.can_run_parallel:
            return False

        if self.execution_count >= self.max_retries:
            return False

        # Check resource constraints
        for constraint in self.resource_constraints:
            if not self._check_resource_constraint(constraint, context):
                return False

        return True

    def _check_resource_constraint(
        self, constraint: ResourceConstraint, context: AgentContext
    ) -> bool:
        """Check if a specific resource constraint is satisfied."""
        # This would integrate with the resource scheduler
        # For now, assume constraints are satisfied
        return True


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected in the graph."""

    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(
            f"Circular dependency detected: {' -> '.join(cycle + [cycle[0]])}"
        )


class DependencyValidationError(Exception):
    """Raised when dependency graph validation fails."""

    pass


class TopologicalSort:
    """Utility class for topological sorting with cycle detection."""

    @staticmethod
    def sort(nodes: List[str], edges: List[DependencyEdge]) -> List[str]:
        """
        Perform topological sort on the dependency graph.

        Parameters
        ----------
        nodes : List[str]
            List of node identifiers
        edges : List[DependencyEdge]
            List of dependency edges

        Returns
        -------
        List[str]
            Topologically sorted list of nodes

        Raises
        ------
        CircularDependencyError
            If circular dependencies are detected
        """
        # Build adjacency list and in-degree count
        adjacency = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize all nodes with 0 in-degree
        for node in nodes:
            in_degree[node] = 0

        # Build graph
        for edge in edges:
            # Only consider hard dependencies for topological ordering
            if edge.dependency_type in [DependencyType.HARD, DependencyType.DATA]:
                adjacency[edge.from_agent].append(edge.to_agent)
                in_degree[edge.to_agent] += 1

        # Kahn's algorithm with cycle detection
        queue = deque([node for node in nodes if in_degree[node] == 0])
        result = []
        processed_count = 0

        while queue:
            current = queue.popleft()
            result.append(current)
            processed_count += 1

            # Process neighbors
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if processed_count != len(nodes):
            # Find cycle for error reporting
            cycle = TopologicalSort._find_cycle(nodes, edges)
            raise CircularDependencyError(cycle)

        return result

    @staticmethod
    def _find_cycle(nodes: List[str], edges: List[DependencyEdge]) -> List[str]:
        """Find a cycle in the graph for error reporting."""
        adjacency = defaultdict(list)
        for edge in edges:
            if edge.dependency_type in [DependencyType.HARD, DependencyType.DATA]:
                adjacency[edge.from_agent].append(edge.to_agent)

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node: WHITE for node in nodes}
        parent: Dict[str, str] = {}

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            if colors[node] == GRAY:
                # Found back edge - extract cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if colors[node] == BLACK:
                return None

            colors[node] = GRAY
            path.append(node)

            for neighbor in adjacency[node]:
                cycle = dfs(neighbor, path.copy())
                if cycle:
                    return cycle

            colors[node] = BLACK
            return None

        for node in nodes:
            if colors[node] == WHITE:
                cycle = dfs(node, [])
                if cycle:
                    return cycle

        return []  # Should not reach here if there's actually a cycle


class DependencyGraphEngine:
    """
    Advanced dependency graph engine for agent execution planning.

    Provides topological ordering, circular dependency detection, conditional dependencies,
    and optimization capabilities for complex agent execution graphs.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: List[DependencyEdge] = []
        self.conditional_dependencies: List[DependencyEdge] = []
        self._execution_cache: Dict[str, List[str]] = {}
        self._optimization_enabled = True

    def add_node(self, node: DependencyNode) -> None:
        """Add a node to the dependency graph."""
        self.nodes[node.agent_id] = node
        self._invalidate_cache()

    def add_edge(self, edge: DependencyEdge) -> None:
        """Add a dependency edge to the graph."""
        if edge.dependency_type == DependencyType.CONDITIONAL:
            self.conditional_dependencies.append(edge)
        else:
            self.edges.append(edge)
        self._invalidate_cache()

    def remove_node(self, agent_id: str) -> None:
        """Remove a node and all its edges from the graph."""
        if agent_id in self.nodes:
            del self.nodes[agent_id]

        # Remove edges involving this node
        self.edges = [
            e for e in self.edges if e.from_agent != agent_id and e.to_agent != agent_id
        ]
        self.conditional_dependencies = [
            e
            for e in self.conditional_dependencies
            if e.from_agent != agent_id and e.to_agent != agent_id
        ]
        self._invalidate_cache()

    def add_dependency(
        self,
        from_agent: str,
        to_agent: str,
        dependency_type: DependencyType = DependencyType.HARD,
        condition: Optional[Callable[[AgentContext], bool]] = None,
        weight: float = 1.0,
        **kwargs,
    ) -> None:
        """Convenience method to add a dependency between two agents."""
        edge = DependencyEdge(
            from_agent=from_agent,
            to_agent=to_agent,
            dependency_type=dependency_type,
            condition=condition,
            weight=weight,
            **kwargs,
        )
        self.add_edge(edge)

    def get_execution_order(self, context: Optional[AgentContext] = None) -> List[str]:
        """
        Get optimal execution order for all agents.

        Parameters
        ----------
        context : Optional[AgentContext]
            Current execution context for evaluating conditional dependencies

        Returns
        -------
        List[str]
            List of agent IDs in optimal execution order
        """
        cache_key = self._get_cache_key(context)
        if cache_key in self._execution_cache and self._optimization_enabled:
            return self._execution_cache[cache_key].copy()

        # Evaluate conditional dependencies
        active_edges = self.edges.copy()
        if context:
            for cond_edge in self.conditional_dependencies:
                if cond_edge.is_satisfied(context):
                    active_edges.append(cond_edge)

        # Perform topological sort
        node_ids = list(self.nodes.keys())
        execution_order = TopologicalSort.sort(node_ids, active_edges)

        # Apply priority-based optimization
        if self._optimization_enabled:
            execution_order = self._optimize_execution_order(
                execution_order, active_edges
            )

        # Cache result
        self._execution_cache[cache_key] = execution_order.copy()

        logger.debug(f"Computed execution order: {execution_order}")
        return execution_order

    def get_parallel_groups(
        self, context: Optional[AgentContext] = None
    ) -> List[List[str]]:
        """
        Get groups of agents that can execute in parallel.

        Returns
        -------
        List[List[str]]
            List of groups, where each group contains agents that can run in parallel
        """
        execution_order = self.get_execution_order(context)

        # Build dependency map
        dependency_map = defaultdict(set)
        active_edges = self.edges.copy()
        if context:
            for cond_edge in self.conditional_dependencies:
                if cond_edge.is_satisfied(context):
                    active_edges.append(cond_edge)

        for edge in active_edges:
            if edge.dependency_type in [DependencyType.HARD, DependencyType.DATA]:
                dependency_map[edge.to_agent].add(edge.from_agent)

        # Group agents by execution level
        levels = []
        remaining = set(execution_order)

        while remaining:
            # Find agents with no remaining dependencies
            current_level = []
            for agent in execution_order:
                if agent in remaining:
                    dependencies = dependency_map[agent]
                    if dependencies.issubset(set(execution_order) - remaining):
                        current_level.append(agent)

            if not current_level:
                # This shouldn't happen if topological sort worked correctly
                logger.error(
                    "Unable to find agents without dependencies - possible cycle"
                )
                break

            levels.append(current_level)
            remaining -= set(current_level)

        logger.debug(f"Parallel execution groups: {levels}")
        return levels

    def validate_graph(self) -> List[str]:
        """
        Validate the dependency graph for consistency and correctness.

        Returns
        -------
        List[str]
            List of validation warnings/errors
        """
        issues = []

        # Check that all referenced agents exist
        all_agent_refs = set()
        for edge in self.edges + self.conditional_dependencies:
            all_agent_refs.add(edge.from_agent)
            all_agent_refs.add(edge.to_agent)

        for agent_ref in all_agent_refs:
            if agent_ref not in self.nodes:
                issues.append(f"Referenced agent '{agent_ref}' not found in nodes")

        # Check for circular dependencies
        try:
            self.get_execution_order()
        except CircularDependencyError as e:
            issues.append(f"Circular dependency detected: {e}")

        # Check for conflicting resource constraints
        resource_conflicts = self._check_resource_conflicts()
        issues.extend(resource_conflicts)

        # Check for isolated nodes (nodes with no edges)
        connected_nodes = set()
        for edge in self.edges + self.conditional_dependencies:
            connected_nodes.add(edge.from_agent)
            connected_nodes.add(edge.to_agent)

        for node_id in self.nodes:
            if node_id not in connected_nodes and len(self.nodes) > 1:
                issues.append(f"Isolated node detected: {node_id}")

        if issues:
            logger.warning(f"Graph validation found {len(issues)} issues: {issues}")
        else:
            logger.info("Dependency graph validation passed")

        return issues

    def get_dependency_impact(self, agent_id: str) -> Dict[str, Any]:
        """
        Analyze the impact of a specific agent on the dependency graph.

        Parameters
        ----------
        agent_id : str
            ID of the agent to analyze

        Returns
        -------
        Dict[str, Any]
            Impact analysis including dependents, dependencies, and criticality
        """
        if agent_id not in self.nodes:
            return {"error": f"Agent {agent_id} not found"}

        # Find direct dependencies (what this agent depends on)
        dependencies = []
        for edge in self.edges:
            if edge.to_agent == agent_id:
                dependencies.append(
                    {
                        "agent": edge.from_agent,
                        "type": edge.dependency_type.value,
                        "weight": edge.weight,
                    }
                )

        # Find dependents (what depends on this agent)
        dependents = []
        for edge in self.edges:
            if edge.from_agent == agent_id:
                dependents.append(
                    {
                        "agent": edge.to_agent,
                        "type": edge.dependency_type.value,
                        "weight": edge.weight,
                    }
                )

        # Calculate criticality score
        criticality = self._calculate_agent_criticality(agent_id)

        # Find transitive dependencies
        transitive_deps = self._get_transitive_dependencies(agent_id)
        transitive_dependents = self._get_transitive_dependents(agent_id)

        return {
            "agent_id": agent_id,
            "direct_dependencies": dependencies,
            "direct_dependents": dependents,
            "transitive_dependencies": transitive_deps,
            "transitive_dependents": transitive_dependents,
            "criticality_score": criticality,
            "is_critical_path": len(transitive_dependents) > 0,
            "isolation_score": len(dependencies) + len(dependents),
        }

    def optimize_for_latency(self) -> None:
        """Optimize execution order to minimize total latency."""
        self._optimization_enabled = True
        self._invalidate_cache()
        logger.info("Enabled latency optimization")

    def optimize_for_reliability(self) -> None:
        """Optimize execution order to maximize reliability."""
        # This could reorder agents to put more reliable ones first
        # or to minimize failure propagation
        self._optimization_enabled = True
        self._invalidate_cache()
        logger.info("Enabled reliability optimization")

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dependency graph and execution patterns."""
        total_nodes = len(self.nodes)
        total_edges = len(self.edges) + len(self.conditional_dependencies)

        # Calculate complexity metrics
        avg_dependencies = (
            sum(
                len([e for e in self.edges if e.to_agent == node_id])
                for node_id in self.nodes
            )
            / total_nodes
            if total_nodes > 0
            else 0
        )

        # Priority distribution
        priority_dist: Dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            priority_dist[node.priority.name] += 1

        # Dependency type distribution
        dep_type_dist: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            dep_type_dist[edge.dependency_type.name] += 1

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "conditional_edges": len(self.conditional_dependencies),
            "average_dependencies_per_node": avg_dependencies,
            "priority_distribution": dict(priority_dist),
            "dependency_type_distribution": dict(dep_type_dist),
            "cache_hits": len(self._execution_cache),
            "optimization_enabled": self._optimization_enabled,
        }

    def _optimize_execution_order(
        self, base_order: List[str], edges: List[DependencyEdge]
    ) -> List[str]:
        """Apply optimization heuristics to the base topological order."""
        if not self._optimization_enabled:
            return base_order

        # Priority-based reordering within topological constraints
        dependency_map = defaultdict(set)
        for edge in edges:
            if edge.dependency_type in [DependencyType.HARD, DependencyType.DATA]:
                dependency_map[edge.to_agent].add(edge.from_agent)

        # Build levels directly to avoid recursion with get_parallel_groups
        levels = []
        remaining = set(base_order)

        while remaining:
            # Find agents with no remaining dependencies
            current_level = []
            for agent in base_order:
                if agent in remaining:
                    dependencies = dependency_map[agent]
                    if dependencies.issubset(set(base_order) - remaining):
                        current_level.append(agent)

            if not current_level:
                # Fallback: add remaining agents to avoid infinite loop
                current_level = list(remaining)

            levels.append(current_level)
            remaining -= set(current_level)

        # Apply priority-based sorting within each level
        result = []
        for level in levels:
            # Sort by priority within each level
            level_nodes = [
                (agent_id, self.nodes[agent_id].priority.value) for agent_id in level
            ]
            level_nodes.sort(key=lambda x: x[1])  # Lower number = higher priority
            result.extend([agent_id for agent_id, _ in level_nodes])

        return result

    def _check_resource_conflicts(self) -> List[str]:
        """Check for resource constraint conflicts between agents."""
        conflicts = []

        # Group agents by exclusive resource requirements
        exclusive_resources = defaultdict(list)
        for node_id, node in self.nodes.items():
            if node.requires_exclusive_access:
                exclusive_resources["exclusive_access"].append(node_id)

        # Check for conflicts in exclusive access
        if len(exclusive_resources["exclusive_access"]) > 1:
            agents = exclusive_resources["exclusive_access"]
            conflicts.append(f"Multiple agents require exclusive access: {agents}")

        return conflicts

    def _calculate_agent_criticality(self, agent_id: str) -> float:
        """Calculate criticality score for an agent (0-1, higher = more critical)."""
        # Base criticality on priority and number of dependents
        node = self.nodes[agent_id]
        priority_score = (6 - node.priority.value) / 5  # Convert to 0-1 scale

        dependents_count = len([e for e in self.edges if e.from_agent == agent_id])
        dependent_score = min(dependents_count / 10, 1.0)  # Normalize to 0-1

        return (priority_score + dependent_score) / 2

    def _get_transitive_dependencies(self, agent_id: str) -> List[str]:
        """Get all transitive dependencies for an agent."""
        visited = set()
        dependencies = []

        def dfs(current_agent: str):
            if current_agent in visited:
                return
            visited.add(current_agent)

            for edge in self.edges:
                if edge.to_agent == current_agent and edge.from_agent != agent_id:
                    dependencies.append(edge.from_agent)
                    dfs(edge.from_agent)

        dfs(agent_id)
        return list(set(dependencies))

    def _get_transitive_dependents(self, agent_id: str) -> List[str]:
        """Get all transitive dependents for an agent."""
        visited = set()
        dependents = []

        def dfs(current_agent: str):
            if current_agent in visited:
                return
            visited.add(current_agent)

            for edge in self.edges:
                if edge.from_agent == current_agent and edge.to_agent != agent_id:
                    dependents.append(edge.to_agent)
                    dfs(edge.to_agent)

        dfs(agent_id)
        return list(set(dependents))

    def _get_cache_key(self, context: Optional[AgentContext]) -> str:
        """Generate cache key for execution order."""
        if context is None:
            return "no_context"

        # Create a simple key based on conditional dependency satisfaction
        satisfied_conditions = []
        for edge in self.conditional_dependencies:
            if edge.is_satisfied(context):
                satisfied_conditions.append(f"{edge.from_agent}->{edge.to_agent}")

        return f"context_{hash(tuple(sorted(satisfied_conditions)))}"

    def _invalidate_cache(self) -> None:
        """Invalidate execution order cache."""
        self._execution_cache.clear()
