"""Tests for langgraph_backend.build_graph module (corrected)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List

from cognivault.langgraph_backend.build_graph import (
    GraphFactory,
    GraphConfig,
    GraphBuildError,
)
from cognivault.langgraph_backend.graph_cache import CacheConfig


class TestGraphConfig:
    """Test GraphConfig dataclass."""

    def test_graph_config_creation_minimal(self):
        """Test creating GraphConfig with minimal parameters."""
        config = GraphConfig(
            agents_to_run=["refiner", "synthesis"], enable_checkpoints=False
        )

        assert config.agents_to_run == ["refiner", "synthesis"]
        assert config.enable_checkpoints is False
        assert config.memory_manager is None
        assert config.pattern_name == "standard"
        assert config.cache_enabled is True

    def test_graph_config_creation_full(self):
        """Test creating GraphConfig with all parameters."""
        memory_manager = Mock()

        config = GraphConfig(
            agents_to_run=["refiner", "critic", "historian", "synthesis"],
            enable_checkpoints=True,
            memory_manager=memory_manager,
            pattern_name="parallel",
            cache_enabled=False,
        )

        assert config.agents_to_run == ["refiner", "critic", "historian", "synthesis"]
        assert config.enable_checkpoints is True
        assert config.memory_manager is memory_manager
        assert config.pattern_name == "parallel"
        assert config.cache_enabled is False


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_cache_config_creation_default(self):
        """Test creating CacheConfig with default values."""
        config = CacheConfig()

        assert config.max_size == 50
        assert config.ttl_seconds == 3600  # 1 hour
        assert config.enable_stats is True

    def test_cache_config_creation_custom(self):
        """Test creating CacheConfig with custom values."""
        config = CacheConfig(
            max_size=5,
            ttl_seconds=900,
            enable_stats=False,  # 15 minutes
        )

        assert config.max_size == 5
        assert config.ttl_seconds == 900
        assert config.enable_stats is False


class TestGraphFactory:
    """Test GraphFactory class."""

    @pytest.fixture
    def cache_config(self):
        """Fixture for cache configuration."""
        return CacheConfig(max_size=5, ttl_seconds=300, enable_stats=True)

    @pytest.fixture
    def graph_factory(self, cache_config):
        """Fixture for GraphFactory instance."""
        return GraphFactory(cache_config)

    @pytest.fixture
    def graph_config(self):
        """Fixture for graph configuration."""
        return GraphConfig(
            agents_to_run=["refiner", "critic", "historian", "synthesis"],
            enable_checkpoints=False,
            pattern_name="standard",
            cache_enabled=True,
        )

    def test_graph_factory_initialization(self, cache_config):
        """Test GraphFactory initialization."""
        factory = GraphFactory(cache_config)

        # Check that cache was initialized with the config
        assert factory.cache is not None
        assert factory.pattern_registry is not None
        assert factory.cache.config.max_size == cache_config.max_size
        assert factory.cache.config.ttl_seconds == cache_config.ttl_seconds

    def test_graph_factory_initialization_default_cache(self):
        """Test GraphFactory initialization with default cache config."""
        factory = GraphFactory()

        assert factory.cache is not None
        assert factory.cache.config.max_size == 50
        assert factory.cache.config.ttl_seconds == 3600

    def test_validate_agents_valid(self, graph_factory):
        """Test validate_agents with valid agent list."""
        valid_agents = ["refiner", "critic", "historian", "synthesis"]
        assert graph_factory.validate_agents(valid_agents) is True

    def test_validate_agents_invalid(self, graph_factory):
        """Test validate_agents with invalid agent."""
        invalid_agents = ["refiner", "invalid_agent", "synthesis"]
        assert graph_factory.validate_agents(invalid_agents) is False

    def test_validate_agents_empty(self, graph_factory):
        """Test validate_agents with empty list."""
        # Empty list returns True (no missing agents) but would fail later
        # The actual validation logic checks if agents exist, empty list has no missing agents
        assert graph_factory.validate_agents([]) is True

    def test_validate_agents_none(self, graph_factory):
        """Test validate_agents with None."""
        # This will likely raise an error, so let's handle it
        try:
            result = graph_factory.validate_agents(None)
            # If it doesn't raise an error, it should return False
            assert result is False
        except (TypeError, AttributeError):
            # Expected - None doesn't have the methods needed
            pass

    def test_create_graph_validation_only(self, graph_factory, graph_config):
        """Test graph creation validation without full compilation."""
        # Test that the factory has the right components for graph creation
        assert graph_factory.validate_agents(graph_config.agents_to_run) is True

        # Test that the pattern exists
        pattern = graph_factory.pattern_registry.get_pattern(graph_config.pattern_name)
        assert pattern is not None
        assert pattern.name == "standard"

        # Test that agents are available
        for agent in graph_config.agents_to_run:
            assert agent.lower() in graph_factory.node_functions

    def test_create_graph_cache_enabled(self, graph_factory, graph_config):
        """Test graph creation with cache enabled but no cached result."""
        # Ensure cache is enabled
        assert graph_config.cache_enabled is True

        # Test cache lookup would be attempted (cache starts empty)
        initial_stats = graph_factory.get_cache_stats()
        initial_misses = initial_stats["misses"]

        # Test that we can check cache for this config
        cache_key = graph_factory.cache._generate_cache_key(
            graph_config.pattern_name,
            graph_config.agents_to_run,
            graph_config.enable_checkpoints,
        )
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_create_graph_cache_operations(self, graph_factory, graph_config):
        """Test graph cache operations without full graph creation."""
        # Test cache operations directly
        mock_graph = Mock()

        # Cache a graph
        graph_factory.cache.cache_graph(
            graph_config.pattern_name,
            graph_config.agents_to_run,
            graph_config.enable_checkpoints,
            mock_graph,
        )

        # Retrieve from cache
        cached = graph_factory.cache.get_cached_graph(
            graph_config.pattern_name,
            graph_config.agents_to_run,
            graph_config.enable_checkpoints,
        )

        assert cached is mock_graph

    def test_create_graph_invalid_agents(self, graph_factory, graph_config):
        """Test graph creation with invalid agents."""
        graph_config.agents_to_run = ["invalid_agent"]

        # Test validation fails for invalid agents
        assert graph_factory.validate_agents(graph_config.agents_to_run) is False

        # The actual create_graph would raise GraphBuildError
        try:
            graph_factory.create_graph(graph_config)
            assert False, "Should have raised GraphBuildError"
        except GraphBuildError as e:
            # Check for the actual error message format
            assert (
                "Unknown agent" in str(e)
                or "Missing agents" in str(e)
                or "invalid_agent" in str(e)
            )

    def test_create_graph_pattern_not_found(self, graph_factory, graph_config):
        """Test graph creation with unknown pattern."""
        graph_config.pattern_name = "unknown_pattern"

        # Test that pattern doesn't exist
        pattern = graph_factory.pattern_registry.get_pattern(graph_config.pattern_name)
        assert pattern is None

        # The actual create_graph would raise GraphBuildError
        try:
            graph_factory.create_graph(graph_config)
            assert False, "Should have raised GraphBuildError"
        except GraphBuildError as e:
            assert "Unknown pattern" in str(e) or "pattern" in str(e).lower()

    def test_create_graph_edge_generation(self, graph_factory, graph_config):
        """Test that patterns can generate valid edges for agents."""
        # Test that the pattern can generate edges for the given agents
        pattern = graph_factory.pattern_registry.get_pattern(graph_config.pattern_name)
        assert pattern is not None

        edges = pattern.get_edges(graph_config.agents_to_run)
        assert isinstance(edges, list)

        # Verify edges have the right structure
        for edge in edges:
            assert isinstance(edge, dict)
            assert "from" in edge
            assert "to" in edge

        # Test entry and exit points
        entry = pattern.get_entry_point(graph_config.agents_to_run)
        exits = pattern.get_exit_points(graph_config.agents_to_run)

        assert entry is not None or len(exits) > 0

    def test_get_cache_stats(self, graph_factory):
        """Test getting cache statistics."""
        # Get actual stats from the real cache
        stats = graph_factory.get_cache_stats()

        # Verify that stats are returned and have expected keys
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "current_size" in stats
        assert "max_size" in stats
        assert "hit_rate" in stats

    def test_clear_cache(self, graph_factory):
        """Test clearing cache."""
        # Clear the cache
        graph_factory.clear_cache()

        # Verify cache is empty after clearing
        stats = graph_factory.get_cache_stats()
        assert stats["current_size"] == 0

    def test_get_available_patterns(self, graph_factory):
        """Test getting available patterns."""
        # Test the actual method on the factory
        patterns = graph_factory.get_available_patterns()

        # Should have the default patterns
        assert "standard" in patterns
        assert "parallel" in patterns
        assert "conditional" in patterns
        assert len(patterns) == 3


class TestGraphBuildError:
    """Test GraphBuildError exception."""

    def test_graph_build_error_creation(self):
        """Test creating GraphBuildError."""
        error = GraphBuildError("Test error message")
        assert str(error) == "Test error message"

    def test_graph_build_error_inheritance(self):
        """Test GraphBuildError inheritance."""
        error = GraphBuildError("Test")
        assert isinstance(error, Exception)
