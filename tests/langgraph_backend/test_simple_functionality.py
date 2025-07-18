"""Simple tests to verify basic langgraph_backend functionality."""

import pytest
from unittest.mock import Mock, patch

from cognivault.langgraph_backend import GraphFactory, GraphConfig, CacheConfig
from cognivault.langgraph_backend.graph_patterns import PatternRegistry, StandardPattern
from cognivault.langgraph_backend.graph_cache import GraphCache


def test_imports_work():
    """Test that all imports work correctly."""
    # If we get here, imports worked
    assert GraphFactory is not None
    assert GraphConfig is not None
    assert CacheConfig is not None
    assert PatternRegistry is not None
    assert StandardPattern is not None
    assert GraphCache is not None


def test_graph_config_basic():
    """Test basic GraphConfig functionality."""
    config = GraphConfig(
        agents_to_run=["refiner", "synthesis"], enable_checkpoints=False
    )

    assert config.agents_to_run == ["refiner", "synthesis"]
    assert config.enable_checkpoints is False


def test_cache_config_basic():
    """Test basic CacheConfig functionality."""
    config = CacheConfig()

    assert config.max_size == 50
    assert config.ttl_seconds == 3600


def test_pattern_registry_basic():
    """Test basic PatternRegistry functionality."""
    registry = PatternRegistry()

    # Should have default patterns
    standard_pattern = registry.get_pattern("standard")
    assert standard_pattern is not None
    assert standard_pattern.name == "standard"


def test_standard_pattern_basic():
    """Test basic StandardPattern functionality."""
    pattern = StandardPattern()

    assert pattern.name == "standard"
    assert "refiner" in pattern.description

    # Test with simple agent list
    agents = ["refiner", "synthesis"]
    edges = pattern.get_edges(agents)

    assert isinstance(edges, list)
    # Should have at least one edge
    assert len(edges) > 0


def test_graph_cache_basic():
    """Test basic GraphCache functionality."""
    cache = GraphCache()

    assert cache.config.max_size == 50
    assert len(cache._cache) == 0

    # Test cache miss
    result = cache.get_cached_graph("standard", ["refiner"], False)
    assert result is None


@patch("cognivault.langgraph_backend.build_graph.StateGraph")
def test_graph_factory_basic(mock_state_graph):
    """Test basic GraphFactory functionality."""
    # Mock StateGraph
    mock_graph_instance = Mock()
    mock_compiled = Mock()
    mock_graph_instance.compile.return_value = mock_compiled
    mock_state_graph.return_value = mock_graph_instance

    factory = GraphFactory()
    config = GraphConfig(
        agents_to_run=["refiner", "synthesis"],
        enable_checkpoints=False,
        cache_enabled=False,  # Disable cache for simple test
    )

    result = factory.create_graph(config)

    # Should return the compiled graph
    assert result is mock_compiled

    # StateGraph should have been called
    assert mock_state_graph.called


def test_integration_basic():
    """Test basic integration between components."""
    # Create components
    factory = GraphFactory()
    registry = PatternRegistry()

    # Test that factory has access to patterns
    assert factory.pattern_registry is not None

    # Test that registry has patterns
    patterns = registry.get_pattern_names()
    assert "standard" in patterns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
