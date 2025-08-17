"""
Graph compilation caching for CogniVault LangGraph backend.

This module provides caching functionality for compiled StateGraphs to improve
performance by avoiding repeated graph compilation for the same configurations.
"""

import hashlib
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import OrderedDict

from cognivault.observability import get_logger


@dataclass
class CacheConfig:
    """Configuration for graph caching."""

    max_size: int = 50  # Maximum number of cached graphs
    ttl_seconds: int = 3600  # Time to live in seconds (1 hour)
    enable_stats: bool = True  # Whether to track cache statistics


@dataclass
class CacheEntry:
    """Entry in the graph cache."""

    compiled_graph: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_estimate: int = 0  # Estimated memory size in bytes

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if this cache entry has expired."""
        return time.time() - self.created_at > ttl_seconds

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired_evictions: int = 0
    size_evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hits + self.misses


class GraphCache:
    """
    LRU cache for compiled StateGraphs with TTL and size management.

    This cache stores compiled graphs to avoid repeated compilation overhead.
    It supports:
    - LRU eviction policy
    - TTL-based expiration
    - Size-based eviction
    - Thread-safe operations
    - Performance statistics
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """
        Initialize the graph cache.

        Parameters
        ----------
        config : CacheConfig, optional
            Cache configuration. If None, default config is used.
        """
        self.config = config or CacheConfig()
        self.logger = get_logger(f"{__name__}.GraphCache")

        # Thread-safe ordered dictionary for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics tracking
        self._stats = CacheStats(max_size=self.config.max_size)

        self.logger.info(
            f"GraphCache initialized with max_size={self.config.max_size}, "
            f"ttl={self.config.ttl_seconds}s"
        )

    def _generate_cache_key(
        self, pattern_name: str, agents: List[str], checkpoints_enabled: bool
    ) -> str:
        """
        Generate a cache key for the given parameters.

        Parameters
        ----------
        pattern_name : str
            Graph pattern name
        agents : List[str]
            List of agent names
        checkpoints_enabled : bool
            Whether checkpointing is enabled

        Returns
        -------
        str
            Cache key
        """
        # Sort agents for consistent ordering
        sorted_agents = sorted(agent.lower() for agent in agents)

        # Create key components
        key_data = f"{pattern_name}:{':'.join(sorted_agents)}:{checkpoints_enabled}"

        # Generate hash for consistent key length
        key_hash = hashlib.md5(key_data.encode()).hexdigest()

        return f"{pattern_name}_{key_hash[:8]}"

    def get_cached_graph(
        self, pattern_name: str, agents: List[str], checkpoints_enabled: bool
    ) -> Optional[Any]:
        """
        Retrieve a cached compiled graph.

        Parameters
        ----------
        pattern_name : str
            Graph pattern name
        agents : List[str]
            List of agent names
        checkpoints_enabled : bool
            Whether checkpointing is enabled

        Returns
        -------
        Optional[Any]
            Cached compiled graph or None if not found/expired
        """
        cache_key = self._generate_cache_key(pattern_name, agents, checkpoints_enabled)

        with self._lock:
            # Check if key exists
            if cache_key not in self._cache:
                self._stats.misses += 1
                self.logger.debug(f"Cache miss for key: {cache_key}")
                return None

            entry = self._cache[cache_key]

            # Check if entry has expired
            if entry.is_expired(self.config.ttl_seconds):
                self.logger.debug(f"Cache entry expired for key: {cache_key}")
                del self._cache[cache_key]
                self._stats.misses += 1
                self._stats.expired_evictions += 1
                self._stats.current_size -= 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            entry.touch()

            self._stats.hits += 1
            self.logger.debug(f"Cache hit for key: {cache_key}")

            return entry.compiled_graph

    def cache_graph(
        self,
        pattern_name: str,
        agents: List[str],
        checkpoints_enabled: bool,
        compiled_graph: Any,
    ) -> None:
        """
        Cache a compiled graph.

        Parameters
        ----------
        pattern_name : str
            Graph pattern name
        agents : List[str]
            List of agent names
        checkpoints_enabled : bool
            Whether checkpointing is enabled
        compiled_graph : Any
            Compiled graph to cache
        """
        cache_key = self._generate_cache_key(pattern_name, agents, checkpoints_enabled)

        with self._lock:
            # Remove expired entries first
            self._cleanup_expired()

            # Check if we need to evict for space
            if len(self._cache) >= self.config.max_size:
                self._evict_lru()

            # Create cache entry
            now = time.time()
            entry = CacheEntry(
                compiled_graph=compiled_graph,
                created_at=now,
                last_accessed=now,
                access_count=1,
                size_estimate=self._estimate_graph_size(compiled_graph),
            )

            # Add to cache
            self._cache[cache_key] = entry
            self._stats.current_size += 1

            self.logger.debug(f"Cached graph for key: {cache_key}")

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        expired_keys = []
        current_time = time.time()

        for key, entry in self._cache.items():
            if current_time - entry.created_at > self.config.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            self._stats.expired_evictions += 1
            self._stats.current_size -= 1
            self.logger.debug(f"Removed expired cache entry: {key}")

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if self._cache:
            # OrderedDict maintains insertion/access order
            key, entry = self._cache.popitem(last=False)  # Remove first (oldest)
            self._stats.evictions += 1
            self._stats.size_evictions += 1
            self._stats.current_size -= 1
            self.logger.debug(f"Evicted LRU cache entry: {key}")

    def _estimate_graph_size(self, compiled_graph: Any) -> int:
        """
        Estimate the memory size of a compiled graph.

        Parameters
        ----------
        compiled_graph : Any
            Compiled graph to estimate size for

        Returns
        -------
        int
            Estimated size in bytes
        """
        # Simple estimation - in practice, this could be more sophisticated
        try:
            import sys

            return sys.getsizeof(compiled_graph)
        except Exception:
            # Fallback to a reasonable estimate
            return 1024  # 1KB default estimate

    def clear(self) -> None:
        """Clear all cached graphs."""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._stats.current_size = 0
            self.logger.info(f"Cleared {cleared_count} cached graphs")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        with self._lock:
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate": self._stats.hit_rate,
                "total_requests": self._stats.total_requests,
                "current_size": self._stats.current_size,
                "max_size": self._stats.max_size,
                "evictions": self._stats.evictions,
                "expired_evictions": self._stats.expired_evictions,
                "size_evictions": self._stats.size_evictions,
                "ttl_seconds": self.config.ttl_seconds,
                "oldest_entry_age": self._get_oldest_entry_age(),
                "newest_entry_age": self._get_newest_entry_age(),
                "total_access_count": self._get_total_access_count(),
            }

    def _get_oldest_entry_age(self) -> Optional[float]:
        """Get age of oldest cache entry in seconds."""
        if not self._cache:
            return None

        current_time = time.time()
        oldest_entry = next(iter(self._cache.values()))
        return current_time - oldest_entry.created_at

    def _get_newest_entry_age(self) -> Optional[float]:
        """Get age of newest cache entry in seconds."""
        if not self._cache:
            return None

        current_time = time.time()
        newest_entry = next(reversed(self._cache.values()))
        return current_time - newest_entry.created_at

    def _get_total_access_count(self) -> int:
        """Get total access count across all entries."""
        return sum(entry.access_count for entry in self._cache.values())

    def get_cache_keys(self) -> List[str]:
        """
        Get all current cache keys.

        Returns
        -------
        List[str]
            List of cache keys
        """
        with self._lock:
            return list(self._cache.keys())

    def remove_pattern(self, pattern_name: str) -> int:
        """
        Remove all cached graphs for a specific pattern.

        Parameters
        ----------
        pattern_name : str
            Pattern name to remove

        Returns
        -------
        int
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys() if key.startswith(f"{pattern_name}_")
            ]

            for key in keys_to_remove:
                del self._cache[key]
                self._stats.current_size -= 1

            removed_count = len(keys_to_remove)
            if removed_count > 0:
                self.logger.info(
                    f"Removed {removed_count} cached graphs for pattern: {pattern_name}"
                )

            return removed_count

    def optimize(self) -> Dict[str, Any]:
        """
        Optimize the cache by removing expired entries and defragmenting.

        Returns
        -------
        Dict[str, int]
            Optimization statistics
        """
        with self._lock:
            initial_size = len(self._cache)

            # Remove expired entries
            self._cleanup_expired()

            # Could add more optimization logic here
            # (e.g., removing least accessed entries if over certain thresholds)

            final_size = len(self._cache)
            removed = initial_size - final_size

            self.logger.info(f"Cache optimization removed {removed} entries")

            return {
                "initial_size": initial_size,
                "final_size": final_size,
                "removed_entries": removed,
                "current_hit_rate": self._stats.hit_rate,
            }
