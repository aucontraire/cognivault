"""Integration tests for langgraph_backend module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Any, Dict, List

from cognivault.langgraph_backend import (
    GraphFactory,
    GraphConfig,
    GraphBuildError,
    CacheConfig,
)
from cognivault.langgraph_backend.graph_patterns import (
    StandardPattern,
    ParallelPattern,
    ConditionalPattern,
    PatternRegistry,
)
from cognivault.langgraph_backend.graph_cache import GraphCache


class TestLangGraphBackendIntegration:
    """Integration tests for the complete langgraph_backend module."""

    @pytest.fixture
    def memory_manager(self):
        """Fixture for memory manager mock."""
        manager = Mock()
        manager.memory_saver = Mock()
        return manager

    @pytest.fixture
    def full_config(self, memory_manager):
        """Fixture for complete graph configuration."""
        return GraphConfig(
            agents_to_run=["refiner", "critic", "historian", "synthesis"],
            enable_checkpoints=True,
            memory_manager=memory_manager,
            pattern_name="standard",
            cache_enabled=True,
        )

    @pytest.fixture
    def graph_factory_with_cache(self):
        """Fixture for GraphFactory with caching enabled."""
        cache_config = CacheConfig(max_size=5, ttl_seconds=300, enable_stats=True)
        return GraphFactory(cache_config)

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_end_to_end_graph_creation_with_cache(
        self, mock_state_graph, graph_factory_with_cache, full_config
    ):
        """Test complete end-to-end graph creation with caching."""
        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        # First creation - should hit pattern registry and StateGraph
        result1 = graph_factory_with_cache.create_graph(full_config)

        # Verify first creation
        assert result1 is mock_compiled
        assert mock_state_graph.called
        assert mock_graph_instance.compile.called

        # Reset mocks for second call
        mock_state_graph.reset_mock()
        mock_graph_instance.reset_mock()

        # Second creation with same config - should hit cache
        result2 = graph_factory_with_cache.create_graph(full_config)

        # Verify cache hit
        assert result2 is mock_compiled
        assert not mock_state_graph.called  # Should not create new StateGraph
        assert not mock_graph_instance.compile.called  # Should not compile again

    def test_pattern_registry_integration_with_factory(self):
        """Test PatternRegistry integration with GraphFactory."""
        factory = GraphFactory()

        # Test all default patterns work with factory
        for pattern_name in ["standard", "parallel", "conditional"]:
            config = GraphConfig(
                agents_to_run=["refiner", "synthesis"],
                enable_checkpoints=False,
                pattern_name=pattern_name,
                cache_enabled=False,
            )

            # Should not raise exception for valid patterns
            pattern = factory.pattern_registry.get_pattern(pattern_name)
            assert pattern is not None
            assert pattern.name == pattern_name

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_different_patterns_create_different_graphs(
        self, mock_state_graph, graph_factory_with_cache
    ):
        """Test that different patterns create different cached graphs."""
        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled_standard = Mock()
        mock_compiled_parallel = Mock()

        def compile_side_effect():
            if (
                mock_graph_instance.add_edge.call_count == 4
            ):  # Standard pattern has 4 edges
                return mock_compiled_standard
            else:  # Parallel pattern has 3 edges
                return mock_compiled_parallel

        mock_graph_instance.compile.side_effect = compile_side_effect
        mock_state_graph.return_value = mock_graph_instance

        agents = ["refiner", "critic", "historian", "synthesis"]

        # Create with standard pattern
        config_standard = GraphConfig(
            agents_to_run=agents,
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=True,
        )
        result_standard = graph_factory_with_cache.create_graph(config_standard)

        # Reset for parallel pattern
        mock_graph_instance.reset_mock()

        # Create with parallel pattern
        config_parallel = GraphConfig(
            agents_to_run=agents,
            enable_checkpoints=False,
            pattern_name="parallel",
            cache_enabled=True,
        )
        result_parallel = graph_factory_with_cache.create_graph(config_parallel)

        # Should be different compiled graphs
        assert result_standard is not result_parallel

    def test_cache_behavior_with_different_configurations(
        self, graph_factory_with_cache
    ):
        """Test cache behavior with various configuration combinations."""
        base_agents = ["refiner", "synthesis"]

        configurations = [
            # Different agents
            (base_agents, False, "standard"),
            (["refiner", "critic", "synthesis"], False, "standard"),
            # Different checkpoints
            (base_agents, True, "standard"),
            # Different patterns
            (base_agents, False, "parallel"),
        ]

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # Create graphs for each configuration
            results = []
            for agents, checkpoints, pattern in configurations:
                config = GraphConfig(
                    agents_to_run=agents,
                    enable_checkpoints=checkpoints,
                    pattern_name=pattern,
                    cache_enabled=True,
                )
                result = graph_factory_with_cache.create_graph(config)
                results.append(result)

            # Each configuration should create a separate cache entry
            # (4 different configs = 4 StateGraph compilations)
            assert mock_state_graph.call_count == 4

    def test_cache_ttl_integration(self):
        """Test cache TTL integration with graph creation."""
        # Use very short TTL for testing
        cache_config = CacheConfig(max_size=5, ttl_seconds=0.1, enable_stats=True)
        factory = GraphFactory(cache_config)

        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=True,
        )

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # First creation
            result1 = factory.create_graph(config)
            assert mock_state_graph.call_count == 1

            # Immediate second creation (should hit cache)
            result2 = factory.create_graph(config)
            assert mock_state_graph.call_count == 1  # No additional calls
            assert result1 is result2

            # Wait for TTL to expire
            time.sleep(0.15)

            # Third creation (should miss cache due to TTL)
            result3 = factory.create_graph(config)
            assert mock_state_graph.call_count == 2  # New creation
            assert result3 is mock_compiled  # Same mock, but new creation

    def test_lru_cache_integration(self):
        """Test LRU cache behavior integration."""
        # Use small cache size
        cache_config = CacheConfig(max_size=2, ttl_seconds=300, enable_stats=True)
        factory = GraphFactory(cache_config)

        # Use different valid configurations to test LRU behavior
        configs = [
            GraphConfig(
                agents_to_run=["refiner", "synthesis"],
                enable_checkpoints=False,
                pattern_name="standard",
                cache_enabled=True,
            ),
            GraphConfig(
                agents_to_run=["refiner", "critic"],
                enable_checkpoints=False,
                pattern_name="standard",
                cache_enabled=True,
            ),
            GraphConfig(
                agents_to_run=["refiner", "historian"],
                enable_checkpoints=False,
                pattern_name="standard",
                cache_enabled=True,
            ),
        ]

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # Create all configs (should trigger LRU eviction)
            results = []
            for config in configs:
                result = factory.create_graph(config)
                results.append(result)

            # All should have been created
            assert mock_state_graph.call_count == 3

            # Re-create first config (should be evicted from cache due to LRU)
            result_first_again = factory.create_graph(configs[0])
            assert mock_state_graph.call_count == 4  # New creation due to eviction

    def test_error_handling_integration(self, graph_factory_with_cache):
        """Test error handling integration across components."""
        # Test invalid agents
        invalid_config = GraphConfig(
            agents_to_run=["invalid_agent"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=True,
        )

        with pytest.raises(GraphBuildError, match="Unknown agent"):
            graph_factory_with_cache.create_graph(invalid_config)

        # Test invalid pattern
        invalid_pattern_config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            enable_checkpoints=False,
            pattern_name="invalid_pattern",
            cache_enabled=True,
        )

        with pytest.raises(GraphBuildError, match="Unknown graph pattern"):
            graph_factory_with_cache.create_graph(invalid_pattern_config)

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_checkpointing_integration(
        self, mock_state_graph, graph_factory_with_cache
    ):
        """Test checkpointing integration with memory manager."""
        memory_manager = Mock()
        memory_saver_mock = Mock()
        memory_manager.get_memory_saver.return_value = memory_saver_mock

        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            enable_checkpoints=True,
            memory_manager=memory_manager,
            pattern_name="standard",
            cache_enabled=True,
        )

        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        # Create graph with checkpointing
        result = graph_factory_with_cache.create_graph(config)

        # Verify checkpointer was passed to compile
        compile_call = mock_graph_instance.compile.call_args
        assert "checkpointer" in compile_call[1]
        assert compile_call[1]["checkpointer"] is memory_saver_mock

    def test_pattern_edge_generation_integration(self):
        """Test pattern edge generation integration."""
        factory = GraphFactory()
        agents = ["refiner", "critic", "historian", "synthesis"]

        # Test each pattern generates correct edges (including END nodes)
        pattern_expected_edges = {
            "standard": [
                {"from": "refiner", "to": "critic"},
                {"from": "refiner", "to": "historian"},
                {"from": "critic", "to": "synthesis"},
                {"from": "historian", "to": "synthesis"},
                {"from": "synthesis", "to": "END"},
            ],
            "parallel": [
                {"from": "refiner", "to": "synthesis"},
                {"from": "critic", "to": "synthesis"},
                {"from": "historian", "to": "synthesis"},
                {"from": "synthesis", "to": "END"},
            ],
            "conditional": [
                {"from": "refiner", "to": "critic"},
                {"from": "refiner", "to": "historian"},
                {"from": "critic", "to": "synthesis"},
                {"from": "historian", "to": "synthesis"},
                {"from": "synthesis", "to": "END"},
            ],
        }

        for pattern_name, expected_edges in pattern_expected_edges.items():
            pattern = factory.pattern_registry.get_pattern(pattern_name)
            actual_edges = pattern.get_edges(agents)
            assert actual_edges == expected_edges

    @patch("cognivault.langgraph_backend.build_graph.StateGraph")
    def test_node_addition_integration(
        self, mock_state_graph, graph_factory_with_cache
    ):
        """Test node addition integration with patterns."""
        config = GraphConfig(
            agents_to_run=["refiner", "critic", "synthesis"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=False,
        )

        # Mock StateGraph
        mock_graph_instance = Mock()
        mock_compiled = Mock()
        mock_graph_instance.compile.return_value = mock_compiled
        mock_state_graph.return_value = mock_graph_instance

        # Create graph
        graph_factory_with_cache.create_graph(config)

        # Verify nodes were added
        expected_nodes = ["refiner", "critic", "synthesis"]
        assert mock_graph_instance.add_node.call_count == len(expected_nodes)

        # Verify correct node functions were called
        node_calls = [call[0] for call in mock_graph_instance.add_node.call_args_list]
        for agent_name in expected_nodes:
            assert any(agent_name in str(call) for call in node_calls)

    def test_cache_statistics_integration(self, graph_factory_with_cache):
        """Test cache statistics integration."""
        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=True,
        )

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # Initial stats
            initial_stats = graph_factory_with_cache.get_cache_stats()
            assert initial_stats["hits"] == 0
            assert initial_stats["misses"] == 0
            assert initial_stats["current_size"] == 0

            # First creation (miss)
            graph_factory_with_cache.create_graph(config)
            stats_after_miss = graph_factory_with_cache.get_cache_stats()
            assert stats_after_miss["misses"] == 1
            assert stats_after_miss["current_size"] == 1

            # Second creation (hit)
            graph_factory_with_cache.create_graph(config)
            stats_after_hit = graph_factory_with_cache.get_cache_stats()
            assert stats_after_hit["hits"] == 1
            assert stats_after_hit["misses"] == 1
            assert stats_after_hit["current_size"] == 1

    def test_clear_cache_integration(self, graph_factory_with_cache):
        """Test cache clearing integration."""
        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=True,
        )

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # Create and cache graph
            graph_factory_with_cache.create_graph(config)
            assert graph_factory_with_cache.get_cache_stats()["current_size"] == 1

            # Clear cache
            graph_factory_with_cache.clear_cache()
            assert graph_factory_with_cache.get_cache_stats()["current_size"] == 0

            # Next creation should miss cache
            graph_factory_with_cache.create_graph(config)
            stats = graph_factory_with_cache.get_cache_stats()
            assert stats["misses"] == 2  # Original miss + post-clear miss

    def test_available_patterns_integration(self, graph_factory_with_cache):
        """Test available patterns integration."""
        patterns = graph_factory_with_cache.get_available_patterns()

        # Should include all default patterns
        assert "standard" in patterns
        assert "parallel" in patterns
        assert "conditional" in patterns
        assert len(patterns) >= 3

    def test_cache_disabled_via_config(self):
        """Test integration with cache disabled via config."""
        # Cache is always enabled - the disable is via cache_enabled=False in GraphConfig
        factory = GraphFactory()

        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=False,  # This disables cache usage
        )

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # Multiple creations should all hit StateGraph (no caching)
            factory.create_graph(config)
            factory.create_graph(config)

            assert mock_state_graph.call_count == 2

            # Cache should still exist but not be used
            stats = factory.get_cache_stats()
            assert stats["current_size"] == 0  # No items cached


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_typical_usage_scenario(self):
        """Test typical usage scenario with multiple graph creations."""
        factory = GraphFactory()

        # Simulate CLI usage patterns
        scenarios = [
            # Full pipeline
            {
                "agents": ["refiner", "critic", "historian", "synthesis"],
                "checkpoints": False,
                "pattern": "standard",
            },
            # Subset of agents
            {
                "agents": ["refiner", "synthesis"],
                "checkpoints": False,
                "pattern": "standard",
            },
            # With checkpointing
            {
                "agents": ["refiner", "critic", "historian", "synthesis"],
                "checkpoints": True,
                "pattern": "standard",
            },
            # Different pattern
            {
                "agents": ["refiner", "critic", "historian", "synthesis"],
                "checkpoints": False,
                "pattern": "parallel",
            },
        ]

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            for scenario in scenarios:
                memory_manager = Mock() if scenario["checkpoints"] else None
                if memory_manager:
                    memory_manager.memory_saver = Mock()

                config = GraphConfig(
                    agents_to_run=scenario["agents"],
                    enable_checkpoints=scenario["checkpoints"],
                    memory_manager=memory_manager,
                    pattern_name=scenario["pattern"],
                    cache_enabled=True,
                )

                result = factory.create_graph(config)
                assert result is mock_compiled

            # Should have created 4 different graphs (different configs)
            assert mock_state_graph.call_count == 4

    def test_performance_scenario(self):
        """Test performance scenario with repeated operations."""
        cache_config = CacheConfig(max_size=10, ttl_seconds=60, enable_stats=True)
        factory = GraphFactory(cache_config)

        config = GraphConfig(
            agents_to_run=["refiner", "critic", "synthesis"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=True,
        )

        with patch(
            "cognivault.langgraph_backend.build_graph.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled = Mock()
            mock_graph_instance.compile.return_value = mock_compiled
            mock_state_graph.return_value = mock_graph_instance

            # Simulate repeated operations (like in testing or batch processing)
            results = []
            for _ in range(10):
                result = factory.create_graph(config)
                results.append(result)

            # Should only create graph once due to caching
            assert mock_state_graph.call_count == 1

            # All results should be the same cached object
            assert all(result is mock_compiled for result in results)

            # Cache should show high hit rate
            stats = factory.get_cache_stats()
            assert stats["hits"] == 9  # 9 hits after first miss
            assert stats["misses"] == 1  # 1 initial miss
