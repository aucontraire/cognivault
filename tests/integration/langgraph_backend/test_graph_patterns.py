"""Tests for langgraph_backend.graph_patterns module."""

import pytest
from typing import Any, List
from unittest.mock import MagicMock, Mock

from cognivault.langgraph_backend.graph_patterns import (
    GraphPattern,
    StandardPattern,
    ParallelPattern,
    ConditionalPattern,
    PatternRegistry,
)


class TestGraphPattern:
    """Test GraphPattern abstract base class."""

    def test_graph_pattern_is_abstract(self) -> None:
        """Test that GraphPattern cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GraphPattern()  # type: ignore[abstract] # Intentionally testing abstract class instantiation


class TestStandardPattern:
    """Test StandardPattern implementation."""

    @pytest.fixture
    def standard_pattern(self) -> Any:
        """Fixture for StandardPattern instance."""
        return StandardPattern()

    def test_standard_pattern_creation(self, standard_pattern: Any) -> None:
        """Test StandardPattern creation."""
        assert isinstance(standard_pattern, GraphPattern)
        assert standard_pattern.name == "standard"
        assert (
            standard_pattern.description
            == "Standard 4-agent pattern: refiner → [critic, historian] → synthesis"
        )

    def test_standard_pattern_all_agents(self, standard_pattern: Any) -> None:
        """Test standard pattern with all agents."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        edges = standard_pattern.get_edges(agents)

        expected_edges = [
            {"from": "refiner", "to": "critic"},
            {"from": "refiner", "to": "historian"},
            {"from": "critic", "to": "synthesis"},
            {"from": "historian", "to": "synthesis"},
        ]

        # Check that all expected internal edges are present
        for expected_edge in expected_edges:
            assert expected_edge in edges

        # Check that synthesis connects to END (terminal node)
        assert {"from": "synthesis", "to": "END"} in edges

    def test_standard_pattern_subset_agents(self, standard_pattern: Any) -> None:
        """Test standard pattern with subset of agents."""
        agents = ["refiner", "synthesis"]
        edges = standard_pattern.get_edges(agents)

        # Fixed: refiner connects directly to synthesis when no intermediates present
        expected_edges = [
            {"from": "refiner", "to": "synthesis"},
            {"from": "synthesis", "to": "END"},
        ]

        assert edges == expected_edges

    def test_standard_pattern_refiner_critic_only(self, standard_pattern: Any) -> None:
        """Test standard pattern with refiner and critic only."""
        agents = ["refiner", "critic"]
        edges = standard_pattern.get_edges(agents)

        expected_edges = [
            {"from": "refiner", "to": "critic"},
            {"from": "critic", "to": "END"},
        ]

        assert edges == expected_edges

    def test_standard_pattern_refiner_historian_only(
        self, standard_pattern: Any
    ) -> None:
        """Test standard pattern with refiner and historian only."""
        agents = ["refiner", "historian"]
        edges = standard_pattern.get_edges(agents)

        expected_edges = [
            {"from": "refiner", "to": "historian"},
            {"from": "historian", "to": "END"},
        ]

        assert edges == expected_edges

    def test_standard_pattern_missing_refiner(self, standard_pattern: Any) -> None:
        """Test standard pattern without refiner agent."""
        agents = ["critic", "historian", "synthesis"]
        edges = standard_pattern.get_edges(agents)

        # Without refiner, critic and historian connect directly to synthesis
        expected_edges = [
            {"from": "critic", "to": "synthesis"},
            {"from": "historian", "to": "synthesis"},
            {"from": "synthesis", "to": "END"},
        ]

        assert edges == expected_edges

    def test_standard_pattern_synthesis_only(self, standard_pattern: Any) -> None:
        """Test standard pattern with synthesis only."""
        agents = ["synthesis"]
        edges = standard_pattern.get_edges(agents)

        # Synthesis-only case
        expected_edges = [{"from": "synthesis", "to": "END"}]
        assert edges == expected_edges

    def test_standard_pattern_empty_agents(self, standard_pattern: Any) -> None:
        """Test standard pattern with empty agent list."""
        agents: List[str] = []
        edges = standard_pattern.get_edges(agents)

        assert edges == []

    def test_standard_pattern_unknown_agents(self, standard_pattern: Any) -> None:
        """Test standard pattern with unknown agents."""
        agents = ["refiner", "unknown_agent", "synthesis"]
        edges = standard_pattern.get_edges(agents)

        # Refiner and synthesis are recognized, so refiner connects to synthesis
        expected_edges = [
            {"from": "refiner", "to": "synthesis"},
            {"from": "synthesis", "to": "END"},
        ]

        assert edges == expected_edges

    def test_standard_pattern_get_entry_point(self, standard_pattern: Any) -> None:
        """Test getting entry points for standard pattern."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        entry_point = standard_pattern.get_entry_point(agents)

        assert entry_point == "refiner"

    def test_standard_pattern_get_entry_point_no_refiner(
        self, standard_pattern: Any
    ) -> None:
        """Test getting entry points when refiner is missing."""
        agents = ["critic", "historian", "synthesis"]
        entry_point = standard_pattern.get_entry_point(agents)

        # When refiner is missing, first available agent becomes entry point
        assert entry_point in ["critic", "historian"]

    def test_standard_pattern_get_exit_points(self, standard_pattern: Any) -> None:
        """Test getting terminal nodes for standard pattern."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        exit_points = standard_pattern.get_exit_points(agents)

        assert exit_points == ["synthesis"]

    def test_standard_pattern_get_exit_points_no_synthesis(
        self, standard_pattern: Any
    ) -> None:
        """Test getting terminal nodes when synthesis is missing."""
        agents = ["refiner", "critic", "historian"]
        exit_points = standard_pattern.get_exit_points(agents)

        # When synthesis is missing, critic and historian become exit points
        assert set(exit_points) == {"critic", "historian"}


class TestParallelPattern:
    """Test ParallelPattern implementation."""

    @pytest.fixture
    def parallel_pattern(self) -> Any:
        """Fixture for ParallelPattern instance."""
        return ParallelPattern()

    def test_parallel_pattern_creation(self, parallel_pattern: Any) -> None:
        """Test ParallelPattern creation."""
        assert isinstance(parallel_pattern, GraphPattern)
        assert parallel_pattern.name == "parallel"
        assert "Maximum parallelization" in parallel_pattern.description

    def test_parallel_pattern_all_agents(self, parallel_pattern: Any) -> None:
        """Test parallel pattern with all agents."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        edges = parallel_pattern.get_edges(agents)

        # In parallel pattern, all non-synthesis agents feed into synthesis
        expected_edges = [
            {"from": "refiner", "to": "synthesis"},
            {"from": "critic", "to": "synthesis"},
            {"from": "historian", "to": "synthesis"},
            {"from": "synthesis", "to": "END"},
        ]

        assert edges == expected_edges

    def test_parallel_pattern_subset_agents(self, parallel_pattern: Any) -> None:
        """Test parallel pattern with subset of agents."""
        agents = ["refiner", "critic"]
        edges = parallel_pattern.get_edges(agents)

        # No synthesis means both agents are terminal
        expected_edges = [
            {"from": "refiner", "to": "END"},
            {"from": "critic", "to": "END"},
        ]
        assert edges == expected_edges

    def test_parallel_pattern_with_synthesis(self, parallel_pattern: Any) -> None:
        """Test parallel pattern including synthesis."""
        agents = ["refiner", "synthesis"]
        edges = parallel_pattern.get_edges(agents)

        expected_edges = [
            {"from": "refiner", "to": "synthesis"},
            {"from": "synthesis", "to": "END"},
        ]

        assert edges == expected_edges

    def test_parallel_pattern_get_entry_point(self, parallel_pattern: Any) -> None:
        """Test getting entry points for parallel pattern."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        entry_point = parallel_pattern.get_entry_point(agents)

        # Parallel pattern has no single entry point (returns None)
        assert entry_point is None

    def test_parallel_pattern_get_exit_points(self, parallel_pattern: Any) -> None:
        """Test getting terminal nodes for parallel pattern."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        exit_points = parallel_pattern.get_exit_points(agents)

        assert exit_points == ["synthesis"]


class TestConditionalPattern:
    """Test ConditionalPattern implementation."""

    @pytest.fixture
    def conditional_pattern(self) -> Any:
        """Fixture for ConditionalPattern instance."""
        return ConditionalPattern()

    def test_conditional_pattern_creation(self, conditional_pattern: Any) -> None:
        """Test ConditionalPattern creation."""
        assert isinstance(conditional_pattern, GraphPattern)
        assert conditional_pattern.name == "conditional"
        assert "routing pattern" in conditional_pattern.description

    def test_conditional_pattern_all_agents(self, conditional_pattern: Any) -> None:
        """Test conditional pattern with all agents."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        edges = conditional_pattern.get_edges(agents)

        # Conditional pattern uses standard pattern logic
        expected_edges = [
            {"from": "refiner", "to": "critic"},
            {"from": "refiner", "to": "historian"},
            {"from": "critic", "to": "synthesis"},
            {"from": "historian", "to": "synthesis"},
            {"from": "synthesis", "to": "END"},
        ]

        assert edges == expected_edges

    def test_conditional_pattern_get_entry_point(
        self, conditional_pattern: Any
    ) -> None:
        """Test getting entry points for conditional pattern."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        entry_point = conditional_pattern.get_entry_point(agents)

        assert entry_point == "refiner"

    def test_conditional_pattern_get_exit_points(
        self, conditional_pattern: Any
    ) -> None:
        """Test getting terminal nodes for conditional pattern."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        exit_points = conditional_pattern.get_exit_points(agents)

        assert exit_points == ["synthesis"]


class TestPatternRegistry:
    """Test PatternRegistry class."""

    @pytest.fixture
    def pattern_registry(self) -> Any:
        """Fixture for PatternRegistry instance."""
        return PatternRegistry()

    def test_pattern_registry_creation(self, pattern_registry: Any) -> None:
        """Test PatternRegistry creation with default patterns."""
        assert isinstance(pattern_registry, PatternRegistry)

        # Check default patterns are registered
        patterns = pattern_registry.get_pattern_names()
        assert "standard" in patterns
        assert "parallel" in patterns
        assert "conditional" in patterns

    def test_get_pattern_standard(self, pattern_registry: Any) -> None:
        """Test getting standard pattern."""
        pattern = pattern_registry.get_pattern("standard")

        assert isinstance(pattern, StandardPattern)
        assert pattern.name == "standard"

    def test_get_pattern_parallel(self, pattern_registry: Any) -> None:
        """Test getting parallel pattern."""
        pattern = pattern_registry.get_pattern("parallel")

        assert isinstance(pattern, ParallelPattern)
        assert pattern.name == "parallel"

    def test_get_pattern_conditional(self, pattern_registry: Any) -> None:
        """Test getting conditional pattern."""
        pattern = pattern_registry.get_pattern("conditional")

        assert isinstance(pattern, ConditionalPattern)
        assert pattern.name == "conditional"

    def test_get_pattern_unknown(self, pattern_registry: Any) -> None:
        """Test getting unknown pattern."""
        pattern = pattern_registry.get_pattern("unknown")

        assert pattern is None

    def test_get_pattern_none(self, pattern_registry: Any) -> None:
        """Test getting pattern with None name."""
        pattern = pattern_registry.get_pattern(None)

        assert pattern is None

    def test_get_pattern_empty_string(self, pattern_registry: Any) -> None:
        """Test getting pattern with empty string."""
        pattern = pattern_registry.get_pattern("")

        assert pattern is None

    def test_register_pattern(self, pattern_registry: Any) -> None:
        """Test registering a custom pattern."""
        # Create a mock custom pattern
        custom_pattern: Mock = Mock()
        custom_pattern.name = "custom"
        custom_pattern.description = "Custom test pattern"

        # Register the pattern
        pattern_registry.register_pattern(custom_pattern)

        # Verify it's available
        patterns = pattern_registry.get_pattern_names()
        assert "custom" in patterns

        # Verify we can retrieve it
        retrieved_pattern = pattern_registry.get_pattern("custom")
        assert retrieved_pattern is custom_pattern

    def test_register_pattern_override(self, pattern_registry: Any) -> None:
        """Test overriding an existing pattern."""
        # Create a custom standard pattern
        custom_standard: Mock = Mock()
        custom_standard.name = "standard"
        custom_standard.description = "Custom standard pattern"

        # Register it (should override the default)
        pattern_registry.register_pattern(custom_standard)

        # Verify the override
        retrieved_pattern = pattern_registry.get_pattern("standard")
        assert retrieved_pattern is custom_standard

    def test_get_pattern_names(self, pattern_registry: Any) -> None:
        """Test getting list of available patterns."""
        patterns = pattern_registry.get_pattern_names()

        assert isinstance(patterns, list)
        assert len(patterns) >= 3  # At least the default patterns
        assert "standard" in patterns
        assert "parallel" in patterns
        assert "conditional" in patterns

    def test_get_pattern_names_after_registration(self, pattern_registry: Any) -> None:
        """Test getting patterns after registering new ones."""
        initial_patterns = pattern_registry.get_pattern_names()
        initial_count = len(initial_patterns)

        # Register two new patterns
        pattern1: Mock = Mock()
        pattern1.name = "test1"
        pattern2: Mock = Mock()
        pattern2.name = "test2"

        pattern_registry.register_pattern(pattern1)
        pattern_registry.register_pattern(pattern2)

        # Check updated list
        updated_patterns = pattern_registry.get_pattern_names()
        assert len(updated_patterns) == initial_count + 2
        assert "test1" in updated_patterns
        assert "test2" in updated_patterns

    def test_pattern_registry_singleton_behavior(self) -> None:
        """Test that PatternRegistry behaves consistently across instances."""
        registry1 = PatternRegistry()
        registry2 = PatternRegistry()

        # Both should have the same default patterns
        patterns1 = registry1.get_pattern_names()
        patterns2 = registry2.get_pattern_names()

        assert set(patterns1) == set(patterns2)

    def test_pattern_validation_in_registry(self, pattern_registry: Any) -> None:
        """Test pattern validation when registering."""
        # Test with None pattern
        # Test with None pattern - should handle gracefully
        try:
            pattern_registry.register_pattern(None)
        except (TypeError, AttributeError):
            pass  # Expected behavior

        # Should not be retrievable
        assert pattern_registry.get_pattern("none_pattern") is None

        # Should not appear in available patterns
        patterns = pattern_registry.get_pattern_names()
        assert "none_pattern" not in patterns


class TestPatternIntegration:
    """Integration tests for pattern functionality."""

    def test_all_patterns_consistency(self) -> None:
        """Test that all patterns work consistently with same agent sets."""
        registry = PatternRegistry()
        agents = ["refiner", "critic", "historian", "synthesis"]

        # Test each pattern
        for pattern_name in ["standard", "parallel", "conditional"]:
            pattern = registry.get_pattern(pattern_name)
            assert pattern is not None

            # Get edges
            edges = pattern.get_edges(agents)
            assert isinstance(edges, list)

            # Get entry point (single value or None)
            entry_point = pattern.get_entry_point(agents)
            assert entry_point is None or isinstance(entry_point, str)

            # Get exit points
            exit_points = pattern.get_exit_points(agents)
            assert isinstance(exit_points, list)
            assert len(exit_points) > 0

    def test_pattern_edge_validation(self) -> None:
        """Test that pattern edges reference valid agents."""
        registry = PatternRegistry()
        agents = ["refiner", "critic", "historian", "synthesis"]

        for pattern_name in ["standard", "parallel", "conditional"]:
            pattern = registry.get_pattern(pattern_name)
            if pattern is not None:
                edges = pattern.get_edges(agents)

                # All edge endpoints should be in the agent list or be END
                for edge in edges:
                    assert edge["from"] in agents
                    assert edge["to"] in agents or edge["to"] == "END"
                    assert edge["from"] != edge["to"]  # No self-loops

    def test_pattern_dag_properties(self) -> None:
        """Test that patterns create valid DAG structures."""
        registry = PatternRegistry()
        agents = ["refiner", "critic", "historian", "synthesis"]

        for pattern_name in ["standard", "parallel", "conditional"]:
            pattern = registry.get_pattern(pattern_name)
            if pattern is not None:
                edges = pattern.get_edges(agents)
                entry_point = pattern.get_entry_point(agents)
                exit_points = pattern.get_exit_points(agents)

                # Entry points should have no incoming edges
                incoming_nodes = {edge["to"] for edge in edges}
                if entry_point is not None:
                    assert entry_point not in incoming_nodes

                # Terminal nodes should have no outgoing edges
                outgoing_nodes = {edge["from"] for edge in edges}
                for exit_point in exit_points:
                    assert (
                        exit_point not in outgoing_nodes or exit_point in incoming_nodes
                    )
