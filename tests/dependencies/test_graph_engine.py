"""
Tests for the dependency graph engine.

Covers topological ordering, circular dependency detection, conditional dependencies,
and graph optimization capabilities.
"""

import pytest
from unittest.mock import Mock, MagicMock

from cognivault.context import AgentContext
from cognivault.agents.base_agent import BaseAgent
from cognivault.dependencies.graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
    DependencyEdge,
    DependencyType,
    ExecutionPriority,
    ResourceConstraint,
    TopologicalSort,
    CircularDependencyError,
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.name = name

    async def run(self, context: AgentContext) -> AgentContext:
        context.agent_outputs[self.name] = f"Output from {self.name}"
        return context


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    return {
        "agent_a": MockAgent("agent_a"),
        "agent_b": MockAgent("agent_b"),
        "agent_c": MockAgent("agent_c"),
        "agent_d": MockAgent("agent_d"),
    }


@pytest.fixture
def graph_engine():
    """Create a dependency graph engine for testing."""
    return DependencyGraphEngine()


@pytest.fixture
def sample_nodes(mock_agents):
    """Create sample dependency nodes."""
    nodes = {}
    for agent_id, agent in mock_agents.items():
        nodes[agent_id] = DependencyNode(
            agent_id=agent_id, agent=agent, priority=ExecutionPriority.NORMAL
        )
    return nodes


class TestDependencyNode:
    """Test DependencyNode functionality."""

    def test_node_creation(self, mock_agents):
        """Test creating a dependency node."""
        agent = mock_agents["agent_a"]
        constraint = ResourceConstraint(
            resource_type="memory", max_usage=100.0, units="MB"
        )

        node = DependencyNode(
            agent_id="agent_a",
            agent=agent,
            priority=ExecutionPriority.HIGH,
            resource_constraints=[constraint],
            max_retries=5,
            timeout_ms=60000,
        )

        assert node.agent_id == "agent_a"
        assert node.agent == agent
        assert node.priority == ExecutionPriority.HIGH
        assert len(node.resource_constraints) == 1
        assert node.max_retries == 5
        assert node.timeout_ms == 60000
        assert node.execution_count == 0
        assert not node.is_executing

    def test_can_execute(self, mock_agents):
        """Test node execution eligibility."""
        agent = mock_agents["agent_a"]
        context = AgentContext(query="test")

        node = DependencyNode(agent_id="agent_a", agent=agent, max_retries=3)

        # Should be able to execute initially
        assert node.can_execute(context)

        # Should not be able to execute if already executing and not parallel
        node.is_executing = True
        node.can_run_parallel = False
        assert not node.can_execute(context)

        # Should not be able to execute if max retries exceeded
        node.is_executing = False
        node.execution_count = 5
        assert not node.can_execute(context)


class TestDependencyEdge:
    """Test DependencyEdge functionality."""

    def test_edge_creation(self):
        """Test creating a dependency edge."""
        edge = DependencyEdge(
            from_agent="agent_a",
            to_agent="agent_b",
            dependency_type=DependencyType.HARD,
            weight=1.5,
            timeout_ms=5000,
        )

        assert edge.from_agent == "agent_a"
        assert edge.to_agent == "agent_b"
        assert edge.dependency_type == DependencyType.HARD
        assert edge.weight == 1.5
        assert edge.timeout_ms == 5000

    def test_condition_satisfaction(self):
        """Test conditional dependency evaluation."""
        context = AgentContext(query="test")

        # Edge with no condition (always satisfied)
        edge = DependencyEdge("agent_a", "agent_b", DependencyType.CONDITIONAL)
        assert edge.is_satisfied(context)

        # Edge with true condition
        edge.condition = lambda ctx: True
        assert edge.is_satisfied(context)

        # Edge with false condition
        edge.condition = lambda ctx: False
        assert not edge.is_satisfied(context)

        # Edge with condition that checks context
        edge.condition = lambda ctx: len(ctx.agent_outputs) > 0
        assert not edge.is_satisfied(context)

        context.agent_outputs["test"] = "output"
        assert edge.is_satisfied(context)

    def test_edge_hash(self):
        """Test edge hashing for set operations."""
        edge1 = DependencyEdge("a", "b", DependencyType.HARD)
        edge2 = DependencyEdge("a", "b", DependencyType.HARD)
        edge3 = DependencyEdge("a", "c", DependencyType.HARD)

        assert hash(edge1) == hash(edge2)
        assert hash(edge1) != hash(edge3)

        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 2  # edge1 and edge2 are duplicates


class TestTopologicalSort:
    """Test topological sorting functionality."""

    def test_simple_topological_sort(self):
        """Test basic topological sorting."""
        nodes = ["a", "b", "c"]
        edges = [
            DependencyEdge("a", "b", DependencyType.HARD),
            DependencyEdge("b", "c", DependencyType.HARD),
        ]

        result = TopologicalSort.sort(nodes, edges)
        assert result == ["a", "b", "c"]

    def test_parallel_branches(self):
        """Test sorting with parallel branches."""
        nodes = ["a", "b", "c", "d"]
        edges = [
            DependencyEdge("a", "c", DependencyType.HARD),
            DependencyEdge("b", "d", DependencyType.HARD),
        ]

        result = TopologicalSort.sort(nodes, edges)
        # a and b can be in any order, but a must come before c, b before d
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("d")

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        nodes = ["a", "b", "c"]
        edges = [
            DependencyEdge("a", "b", DependencyType.HARD),
            DependencyEdge("b", "c", DependencyType.HARD),
            DependencyEdge("c", "a", DependencyType.HARD),
        ]

        with pytest.raises(CircularDependencyError) as exc_info:
            TopologicalSort.sort(nodes, edges)

        assert "Circular dependency detected" in str(exc_info.value)
        assert len(exc_info.value.cycle) >= 3

    def test_self_dependency(self):
        """Test detection of self-dependencies."""
        nodes = ["a"]
        edges = [DependencyEdge("a", "a", DependencyType.HARD)]

        with pytest.raises(CircularDependencyError):
            TopologicalSort.sort(nodes, edges)

    def test_soft_dependencies_ignored(self):
        """Test that soft dependencies don't affect topological ordering."""
        nodes = ["a", "b"]
        edges = [
            DependencyEdge("b", "a", DependencyType.SOFT)  # Soft dependency reversed
        ]

        result = TopologicalSort.sort(nodes, edges)
        # Should work fine since soft dependencies are ignored
        assert len(result) == 2


class TestDependencyGraphEngine:
    """Test DependencyGraphEngine functionality."""

    def test_engine_initialization(self, graph_engine):
        """Test graph engine initialization."""
        assert len(graph_engine.nodes) == 0
        assert len(graph_engine.edges) == 0
        assert len(graph_engine.conditional_dependencies) == 0

    def test_add_nodes(self, graph_engine, sample_nodes):
        """Test adding nodes to the graph."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        assert len(graph_engine.nodes) == 4
        assert "agent_a" in graph_engine.nodes
        assert graph_engine.nodes["agent_a"] == sample_nodes["agent_a"]

    def test_add_edges(self, graph_engine, sample_nodes):
        """Test adding edges to the graph."""
        # Add nodes first
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        # Add edges
        edge1 = DependencyEdge("agent_a", "agent_b", DependencyType.HARD)
        edge2 = DependencyEdge("agent_b", "agent_c", DependencyType.CONDITIONAL)

        graph_engine.add_edge(edge1)
        graph_engine.add_edge(edge2)

        assert len(graph_engine.edges) == 1  # Hard edge
        assert len(graph_engine.conditional_dependencies) == 1  # Conditional edge

    def test_add_dependency_convenience_method(self, graph_engine, sample_nodes):
        """Test the convenience method for adding dependencies."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        graph_engine.add_dependency(
            "agent_a", "agent_b", DependencyType.HARD, weight=2.0
        )

        assert len(graph_engine.edges) == 1
        edge = graph_engine.edges[0]
        assert edge.from_agent == "agent_a"
        assert edge.to_agent == "agent_b"
        assert edge.weight == 2.0

    def test_remove_node(self, graph_engine, sample_nodes):
        """Test removing nodes and their edges."""
        # Add nodes and edges
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)
        graph_engine.add_dependency("agent_b", "agent_c", DependencyType.HARD)

        # Remove node
        graph_engine.remove_node("agent_b")

        assert "agent_b" not in graph_engine.nodes
        assert len(graph_engine.edges) == 0  # All edges involving agent_b removed

    def test_get_execution_order(self, graph_engine, sample_nodes):
        """Test getting execution order."""
        # Add nodes
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        # Add dependencies: a -> b -> c, a -> d
        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)
        graph_engine.add_dependency("agent_b", "agent_c", DependencyType.HARD)
        graph_engine.add_dependency("agent_a", "agent_d", DependencyType.HARD)

        order = graph_engine.get_execution_order()

        # Verify topological ordering
        assert order.index("agent_a") < order.index("agent_b")
        assert order.index("agent_b") < order.index("agent_c")
        assert order.index("agent_a") < order.index("agent_d")

    def test_conditional_dependencies(self, graph_engine, sample_nodes):
        """Test conditional dependency evaluation."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        # Add conditional dependency that depends on context
        condition = lambda ctx: "trigger" in ctx.agent_outputs
        graph_engine.add_dependency(
            "agent_a", "agent_b", DependencyType.CONDITIONAL, condition=condition
        )

        # Without trigger, should be no dependencies
        context = AgentContext(query="test")
        order = graph_engine.get_execution_order(context)
        # No dependencies means any order is valid
        assert len(order) == 4

        # With trigger, should have dependency
        context.agent_outputs["trigger"] = "present"
        order = graph_engine.get_execution_order(context)
        assert order.index("agent_a") < order.index("agent_b")

    def test_get_parallel_groups(self, graph_engine, sample_nodes):
        """Test getting parallel execution groups."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        # Create dependency: a -> (b, c) -> d
        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)
        graph_engine.add_dependency("agent_a", "agent_c", DependencyType.HARD)
        graph_engine.add_dependency("agent_b", "agent_d", DependencyType.HARD)
        graph_engine.add_dependency("agent_c", "agent_d", DependencyType.HARD)

        groups = graph_engine.get_parallel_groups()

        # Should have 3 levels: [a], [b, c], [d]
        assert len(groups) == 3
        assert groups[0] == ["agent_a"]
        assert set(groups[1]) == {"agent_b", "agent_c"}
        assert groups[2] == ["agent_d"]

    def test_graph_validation(self, graph_engine, sample_nodes):
        """Test graph validation."""
        # Graph with isolated nodes
        for node in sample_nodes.values():
            graph_engine.add_node(node)
        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)

        issues = graph_engine.validate_graph()
        # Should detect isolated nodes (agent_c and agent_d)
        assert len(issues) == 2
        assert all("Isolated node detected" in issue for issue in issues)

        # Invalid graph with circular dependency
        graph_engine.add_dependency("agent_b", "agent_a", DependencyType.HARD)
        issues = graph_engine.validate_graph()
        assert len(issues) > 0
        assert any("Circular dependency" in issue for issue in issues)

    def test_dependency_impact_analysis(self, graph_engine, sample_nodes):
        """Test dependency impact analysis."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        # Create chain: a -> b -> c, a -> d
        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)
        graph_engine.add_dependency("agent_b", "agent_c", DependencyType.HARD)
        graph_engine.add_dependency("agent_a", "agent_d", DependencyType.HARD)

        impact = graph_engine.get_dependency_impact("agent_a")

        assert impact["agent_id"] == "agent_a"
        assert len(impact["direct_dependents"]) == 2  # b and d
        assert (
            "agent_c" in impact["transitive_dependents"]
        )  # c depends on b which depends on a
        assert impact["criticality_score"] > 0

    def test_optimization_modes(self, graph_engine, sample_nodes):
        """Test different optimization modes."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)

        # Test latency optimization
        graph_engine.optimize_for_latency()
        order1 = graph_engine.get_execution_order()

        # Test reliability optimization
        graph_engine.optimize_for_reliability()
        order2 = graph_engine.get_execution_order()

        # Both should produce valid orders
        assert len(order1) == 4
        assert len(order2) == 4
        assert order1.index("agent_a") < order1.index("agent_b")
        assert order2.index("agent_a") < order2.index("agent_b")

    def test_execution_statistics(self, graph_engine, sample_nodes):
        """Test execution statistics generation."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)
        graph_engine.add_dependency("agent_c", "agent_d", DependencyType.SOFT)

        stats = graph_engine.get_execution_statistics()

        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 2
        assert stats["average_dependencies_per_node"] == 0.5  # 2 deps / 4 nodes
        assert "HARD" in stats["dependency_type_distribution"]
        assert "SOFT" in stats["dependency_type_distribution"]

    def test_cache_invalidation(self, graph_engine, sample_nodes):
        """Test that cache is properly invalidated on changes."""
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        # Get initial order (should be cached)
        order1 = graph_engine.get_execution_order()
        order2 = graph_engine.get_execution_order()
        assert order1 == order2

        # Add dependency and verify cache is invalidated
        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)
        order3 = graph_engine.get_execution_order()

        # Should respect new dependency
        assert order3.index("agent_a") < order3.index("agent_b")


class TestResourceConstraint:
    """Test ResourceConstraint functionality."""

    def test_constraint_creation(self):
        """Test creating resource constraints."""
        constraint = ResourceConstraint(
            resource_type="cpu",
            max_usage=80.0,
            units="percentage",
            shared=True,
            renewable=False,
        )

        assert constraint.resource_type == "cpu"
        assert constraint.max_usage == 80.0
        assert constraint.units == "percentage"
        assert constraint.shared is True
        assert constraint.renewable is False


class TestIntegration:
    """Integration tests for the dependency graph engine."""

    def test_complex_dependency_scenario(self, graph_engine, mock_agents):
        """Test a complex dependency scenario."""
        # Create agents with different priorities
        nodes = {
            "preprocessor": DependencyNode(
                agent_id="preprocessor",
                agent=mock_agents["agent_a"],
                priority=ExecutionPriority.HIGH,
            ),
            "analyzer": DependencyNode(
                agent_id="analyzer",
                agent=mock_agents["agent_b"],
                priority=ExecutionPriority.NORMAL,
            ),
            "validator": DependencyNode(
                agent_id="validator",
                agent=mock_agents["agent_c"],
                priority=ExecutionPriority.CRITICAL,
            ),
            "formatter": DependencyNode(
                agent_id="formatter",
                agent=mock_agents["agent_d"],
                priority=ExecutionPriority.LOW,
            ),
        }

        for node in nodes.values():
            graph_engine.add_node(node)

        # Create complex dependencies
        # preprocessor -> analyzer -> formatter
        # preprocessor -> validator -> formatter
        # validator has conditional dependency on analyzer
        graph_engine.add_dependency("preprocessor", "analyzer", DependencyType.HARD)
        graph_engine.add_dependency("preprocessor", "validator", DependencyType.HARD)
        graph_engine.add_dependency("analyzer", "formatter", DependencyType.HARD)
        graph_engine.add_dependency("validator", "formatter", DependencyType.HARD)

        # Conditional dependency
        condition = lambda ctx: len(ctx.agent_outputs) > 1
        graph_engine.add_dependency(
            "analyzer", "validator", DependencyType.CONDITIONAL, condition=condition
        )

        # Test execution order
        context = AgentContext(query="test")
        order = graph_engine.get_execution_order(context)

        # Verify basic dependencies
        assert order.index("preprocessor") < order.index("analyzer")
        assert order.index("preprocessor") < order.index("validator")
        assert order.index("analyzer") < order.index("formatter")
        assert order.index("validator") < order.index("formatter")

        # Test parallel groups
        groups = graph_engine.get_parallel_groups(context)
        assert groups[0] == ["preprocessor"]  # Must be first
        # analyzer and validator can be parallel (in second level)
        assert set(groups[1]) == {"analyzer", "validator"}
        assert groups[2] == ["formatter"]  # Must be last

        # Test impact analysis
        impact = graph_engine.get_dependency_impact("preprocessor")
        assert len(impact["direct_dependents"]) == 2
        assert impact["is_critical_path"] is True

        # Validate graph
        issues = graph_engine.validate_graph()
        assert len(issues) == 0

    def test_dynamic_graph_modification(self, graph_engine, sample_nodes):
        """Test dynamic modification of the dependency graph."""
        # Start with simple graph
        for node in sample_nodes.values():
            graph_engine.add_node(node)

        graph_engine.add_dependency("agent_a", "agent_b", DependencyType.HARD)

        initial_order = graph_engine.get_execution_order()
        assert initial_order.index("agent_a") < initial_order.index("agent_b")

        # Add more dependencies dynamically
        graph_engine.add_dependency("agent_b", "agent_c", DependencyType.HARD)
        graph_engine.add_dependency("agent_a", "agent_d", DependencyType.HARD)

        new_order = graph_engine.get_execution_order()

        # Verify new dependencies are respected
        assert new_order.index("agent_a") < new_order.index("agent_b")
        assert new_order.index("agent_b") < new_order.index("agent_c")
        assert new_order.index("agent_a") < new_order.index("agent_d")

        # Remove a node and verify graph adapts
        graph_engine.remove_node("agent_b")
        final_order = graph_engine.get_execution_order()

        assert "agent_b" not in final_order
        assert len(final_order) == 3
        # agent_c should now be free to execute in parallel with agent_d
        assert final_order.index("agent_a") < final_order.index("agent_d")
