"""Tests for langgraph_backend.graph_cache module (corrected)."""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from typing import Any, Dict

from cognivault.langgraph_backend.graph_cache import (
    GraphCache,
    CacheConfig,
    CacheEntry,
    CacheStats,
)


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


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating CacheEntry."""
        mock_graph = Mock()
        current_time = time.time()

        entry = CacheEntry(
            compiled_graph=mock_graph,
            created_at=current_time,
            last_accessed=current_time,
            access_count=0,
            size_estimate=1024,
        )

        assert entry.compiled_graph is mock_graph
        assert entry.created_at == current_time
        assert entry.last_accessed == current_time
        assert entry.access_count == 0
        assert entry.size_estimate == 1024

    def test_cache_entry_is_expired(self):
        """Test cache entry expiration check."""
        mock_graph = Mock()
        current_time = time.time()

        entry = CacheEntry(
            compiled_graph=mock_graph,
            created_at=current_time - 3600,  # 1 hour ago
            last_accessed=current_time,
        )

        # Should be expired with 30 minute TTL
        assert entry.is_expired(1800) is True

        # Should not be expired with 2 hour TTL
        assert entry.is_expired(7200) is False

    def test_cache_entry_touch(self):
        """Test cache entry touch method."""
        mock_graph = Mock()
        current_time = time.time()

        entry = CacheEntry(
            compiled_graph=mock_graph,
            created_at=current_time,
            last_accessed=current_time,
            access_count=0,
        )

        original_access_time = entry.last_accessed
        original_count = entry.access_count

        # Small delay to ensure time difference
        time.sleep(0.01)

        entry.touch()

        assert entry.last_accessed > original_access_time
        assert entry.access_count == original_count + 1


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_cache_stats_creation(self):
        """Test creating CacheStats."""
        stats = CacheStats(hits=10, misses=5, evictions=2, current_size=3, max_size=10)

        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2
        assert stats.current_size == 3
        assert stats.max_size == 10

    def test_cache_stats_hit_rate(self):
        """Test cache stats hit rate calculation."""
        stats = CacheStats(hits=10, misses=5)
        assert stats.hit_rate == 10 / 15

        # Test with no requests
        empty_stats = CacheStats()
        assert empty_stats.hit_rate == 0.0

    def test_cache_stats_total_requests(self):
        """Test cache stats total requests calculation."""
        stats = CacheStats(hits=10, misses=5)
        assert stats.total_requests == 15


class TestGraphCache:
    """Test GraphCache class."""

    @pytest.fixture
    def cache_config(self):
        """Fixture for cache configuration."""
        return CacheConfig(
            max_size=3,
            ttl_seconds=1,
            enable_stats=True,  # Short TTL for testing
        )

    @pytest.fixture
    def graph_cache(self, cache_config):
        """Fixture for GraphCache instance."""
        return GraphCache(cache_config)

    def test_graph_cache_initialization(self, cache_config):
        """Test GraphCache initialization."""
        cache = GraphCache(cache_config)

        assert cache.config is cache_config
        assert len(cache._cache) == 0

    def test_graph_cache_initialization_default(self):
        """Test GraphCache initialization with default values."""
        cache = GraphCache()

        assert cache.config.max_size == 50
        assert cache.config.ttl_seconds == 3600
        assert cache.config.enable_stats is True

    def test_cache_graph_basic(self, graph_cache):
        """Test basic graph caching."""
        pattern_name = "standard"
        agents = ["refiner", "synthesis"]
        checkpoints_enabled = False
        mock_graph = Mock()

        # Cache the graph
        graph_cache.cache_graph(pattern_name, agents, checkpoints_enabled, mock_graph)

        # Verify it was cached
        cached_graph = graph_cache.get_cached_graph(
            pattern_name, agents, checkpoints_enabled
        )
        assert cached_graph is mock_graph

    def test_cache_graph_with_checkpoints(self, graph_cache):
        """Test caching graph with checkpoints enabled."""
        pattern_name = "standard"
        agents = ["refiner", "critic", "synthesis"]
        checkpoints_enabled = True
        mock_graph = Mock()

        # Cache the graph
        graph_cache.cache_graph(pattern_name, agents, checkpoints_enabled, mock_graph)

        # Verify it was cached
        cached_graph = graph_cache.get_cached_graph(
            pattern_name, agents, checkpoints_enabled
        )
        assert cached_graph is mock_graph

    def test_cache_graph_different_keys(self, graph_cache):
        """Test caching graphs with different keys."""
        mock_graph1 = Mock()
        mock_graph2 = Mock()
        mock_graph3 = Mock()

        # Cache different graphs
        graph_cache.cache_graph("standard", ["refiner"], False, mock_graph1)
        graph_cache.cache_graph("standard", ["refiner"], True, mock_graph2)
        graph_cache.cache_graph("parallel", ["refiner"], False, mock_graph3)

        # Verify each is cached separately
        assert (
            graph_cache.get_cached_graph("standard", ["refiner"], False) is mock_graph1
        )
        assert (
            graph_cache.get_cached_graph("standard", ["refiner"], True) is mock_graph2
        )
        assert (
            graph_cache.get_cached_graph("parallel", ["refiner"], False) is mock_graph3
        )

    def test_get_cached_graph_miss(self, graph_cache):
        """Test cache miss."""
        cached_graph = graph_cache.get_cached_graph("standard", ["refiner"], False)
        assert cached_graph is None

    def test_cache_key_generation(self, graph_cache):
        """Test cache key generation consistency."""
        pattern = "standard"
        agents1 = ["refiner", "critic", "synthesis"]
        agents2 = ["critic", "refiner", "synthesis"]  # Different order
        checkpoints = False

        # Generate keys
        key1 = graph_cache._generate_cache_key(pattern, agents1, checkpoints)
        key2 = graph_cache._generate_cache_key(pattern, agents2, checkpoints)

        # Should be the same (normalized)
        assert key1 == key2

    def test_cache_key_generation_empty_agents(self, graph_cache):
        """Test cache key generation with empty agents list."""
        pattern = "standard"
        agents = []
        checkpoints = False

        # Should handle empty agents list
        key = graph_cache._generate_cache_key(pattern, agents, checkpoints)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_lru_eviction(self, graph_cache):
        """Test LRU eviction when cache is full."""
        mock_graphs = [Mock() for _ in range(4)]  # More than max_size (3)

        # Fill cache beyond capacity
        for i, graph in enumerate(mock_graphs):
            graph_cache.cache_graph("standard", [f"agent_{i}"], False, graph)

        # First graph should be evicted (LRU)
        assert graph_cache.get_cached_graph("standard", ["agent_0"], False) is None

        # Others should still be cached
        assert (
            graph_cache.get_cached_graph("standard", ["agent_1"], False)
            is mock_graphs[1]
        )
        assert (
            graph_cache.get_cached_graph("standard", ["agent_2"], False)
            is mock_graphs[2]
        )
        assert (
            graph_cache.get_cached_graph("standard", ["agent_3"], False)
            is mock_graphs[3]
        )

    def test_ttl_expiration(self, graph_cache):
        """Test TTL-based expiration."""
        mock_graph = Mock()

        # Cache graph
        graph_cache.cache_graph("standard", ["refiner"], False, mock_graph)

        # Should be cached immediately
        assert (
            graph_cache.get_cached_graph("standard", ["refiner"], False) is mock_graph
        )

        # Wait for TTL to expire
        time.sleep(1.1)  # TTL is 1 second

        # Should be expired now
        assert graph_cache.get_cached_graph("standard", ["refiner"], False) is None

    def test_clear_cache(self, graph_cache):
        """Test clearing the cache."""
        mock_graphs = [Mock() for _ in range(3)]

        # Fill cache
        for i, graph in enumerate(mock_graphs):
            graph_cache.cache_graph("standard", [f"agent_{i}"], False, graph)

        # Verify items are cached
        assert len(graph_cache._cache) == 3

        # Clear cache
        graph_cache.clear()

        # Verify cache is empty
        assert len(graph_cache._cache) == 0

        # Verify items are no longer accessible
        for i in range(3):
            assert (
                graph_cache.get_cached_graph("standard", [f"agent_{i}"], False) is None
            )

    def test_get_stats(self, graph_cache):
        """Test getting cache statistics."""
        mock_graph = Mock()

        # Cache miss
        graph_cache.get_cached_graph("standard", ["refiner"], False)

        # Cache and hit
        graph_cache.cache_graph("standard", ["refiner"], False, mock_graph)
        graph_cache.get_cached_graph("standard", ["refiner"], False)

        stats = graph_cache.get_stats()

        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "current_size" in stats
        assert "max_size" in stats
        assert "hit_rate" in stats

    def test_thread_safety_basic(self, graph_cache):
        """Test basic thread safety of cache operations."""
        results = []

        def cache_operation(thread_id):
            mock_graph = Mock()
            mock_graph.thread_id = thread_id

            # Cache and retrieve
            graph_cache.cache_graph(
                "standard", [f"agent_{thread_id}"], False, mock_graph
            )
            cached = graph_cache.get_cached_graph(
                "standard", [f"agent_{thread_id}"], False
            )
            results.append(cached is mock_graph)

        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert all(results)

    def test_cache_none_value(self, graph_cache):
        """Test that None values are not cached."""
        # Try to cache None
        graph_cache.cache_graph("standard", ["refiner"], False, None)

        # Should get cache miss (None not cached)
        cached = graph_cache.get_cached_graph("standard", ["refiner"], False)
        assert cached is None

    def test_cache_with_duplicate_agents(self, graph_cache):
        """Test caching with duplicate agents in list."""
        agents_with_duplicates = ["refiner", "critic", "refiner", "synthesis"]
        mock_graph = Mock()

        # Cache with duplicates
        graph_cache.cache_graph("standard", agents_with_duplicates, False, mock_graph)

        # Retrieve with same duplicates (should match)
        cached = graph_cache.get_cached_graph("standard", agents_with_duplicates, False)
        assert cached is mock_graph

        # Different agent order should NOT match (current implementation sorts but doesn't dedupe)
        unique_agents = ["refiner", "critic", "synthesis"]
        cached_unique = graph_cache.get_cached_graph("standard", unique_agents, False)
        assert cached_unique is None  # Different because duplicates aren't normalized

    def test_cache_key_case_sensitivity(self, graph_cache):
        """Test that cache keys are case sensitive for pattern names."""
        mock_graph1 = Mock()
        mock_graph2 = Mock()

        # Cache with different cases
        graph_cache.cache_graph("Standard", ["refiner"], False, mock_graph1)
        graph_cache.cache_graph("standard", ["refiner"], False, mock_graph2)

        # Should be different cache entries
        assert (
            graph_cache.get_cached_graph("Standard", ["refiner"], False) is mock_graph1
        )
        assert (
            graph_cache.get_cached_graph("standard", ["refiner"], False) is mock_graph2
        )

    @patch("time.time")
    def test_ttl_with_mocked_time(self, mock_time, graph_cache):
        """Test TTL functionality with mocked time."""
        # Start at time 0
        mock_time.return_value = 0

        mock_graph = Mock()
        graph_cache.cache_graph("standard", ["refiner"], False, mock_graph)

        # Should be cached at time 0
        assert (
            graph_cache.get_cached_graph("standard", ["refiner"], False) is mock_graph
        )

        # Advance time beyond TTL
        mock_time.return_value = 2  # TTL is 1 second

        # Should be expired
        assert graph_cache.get_cached_graph("standard", ["refiner"], False) is None
