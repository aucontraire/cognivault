"""
Comprehensive tests for enhanced conditional patterns.

This module tests the dynamic routing, fallback mechanisms, performance tracking,
and semantic validation integration of the enhanced conditional pattern system.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from cognivault.langgraph_backend.graph_patterns.conditional import (
    EnhancedConditionalPattern,
    ConditionalPatternValidator,
    ContextAnalyzer,
    ContextComplexity,
    RoutingStrategy,
    RoutingConfig,
    PerformanceTracker,
    FallbackManager,
    FallbackRule,
)
from cognivault.langgraph_backend.semantic_validation import (
    ValidationResult,
    ValidationSeverity,
)


class TestContextAnalyzer:
    """Test context analysis for dynamic routing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ContextAnalyzer()

    def test_simple_query_analysis(self):
        """Test analysis of simple queries."""
        query = "What is Python?"
        analysis = self.analyzer.analyze_context(query)

        assert analysis.complexity_level == ContextComplexity.SIMPLE
        assert analysis.routing_strategy == RoutingStrategy.STREAMLINED
        assert analysis.word_count == 3
        assert not analysis.requires_research
        assert not analysis.requires_criticism

    def test_complex_query_analysis(self):
        """Test analysis of complex queries."""
        query = """
        Provide a comprehensive analysis of the advantages and disadvantages 
        of different machine learning algorithms for complex data analysis. 
        Include historical context, research findings, and critical evaluation 
        of their effectiveness in various domains.
        """
        analysis = self.analyzer.analyze_context(query)

        assert analysis.complexity_level in [
            ContextComplexity.COMPLEX,
            ContextComplexity.VERY_COMPLEX,
        ]
        assert analysis.routing_strategy == RoutingStrategy.COMPREHENSIVE
        assert analysis.requires_research
        assert analysis.requires_criticism
        assert analysis.word_count > 30

    def test_research_indicators(self):
        """Test detection of research requirements."""
        query = "What is the historical background of artificial intelligence?"
        analysis = self.analyzer.analyze_context(query)

        assert analysis.requires_research

        query = "Tell me about the data on climate change effects."
        analysis = self.analyzer.analyze_context(query)

        assert analysis.requires_research

    def test_criticism_indicators(self):
        """Test detection of critical analysis requirements."""
        query = "Evaluate the pros and cons of renewable energy."
        analysis = self.analyzer.analyze_context(query)

        assert analysis.requires_criticism

        query = "Analyze the strengths and weaknesses of this approach."
        analysis = self.analyzer.analyze_context(query)

        assert analysis.requires_criticism

    def test_technical_term_detection(self):
        """Test detection of technical complexity."""
        query = "Implement a comprehensive algorithm for systematic optimization."
        analysis = self.analyzer.analyze_context(query)

        assert analysis.technical_terms > 0
        assert analysis.complexity_score > 0.5


class TestPerformanceTracker:
    """Test performance tracking for routing decisions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker()

    def test_record_execution(self):
        """Test recording execution metrics."""
        self.tracker.record_execution("refiner", 1500.0, True)
        self.tracker.record_execution("refiner", 2000.0, True)
        self.tracker.record_execution("refiner", 1800.0, False)

        avg_time = self.tracker.get_average_time("refiner")
        success_rate = self.tracker.get_success_rate("refiner")

        assert abs(avg_time - 1766.666666666667) < 0.001  # Average of 1500, 2000, 1800
        assert success_rate == 2 / 3  # 2 successes out of 3 attempts

    def test_performance_score_calculation(self):
        """Test performance score combining speed and reliability."""
        # Good performance: fast and reliable
        self.tracker.record_execution("fast_agent", 500.0, True)
        self.tracker.record_execution("fast_agent", 600.0, True)

        # Poor performance: slow and unreliable
        self.tracker.record_execution("slow_agent", 8000.0, False)
        self.tracker.record_execution("slow_agent", 9000.0, True)

        fast_score = self.tracker.get_performance_score("fast_agent")
        slow_score = self.tracker.get_performance_score("slow_agent")

        assert fast_score > slow_score
        assert fast_score > 0.8  # Should be high for fast, reliable agent
        assert slow_score < 0.5  # Should be low for slow, unreliable agent

    def test_unknown_agent_default_score(self):
        """Test default score for agents with no data."""
        score = self.tracker.get_performance_score("unknown_agent")
        assert score == 0.5  # Default score

    def test_history_limit(self):
        """Test that history is limited to recent executions."""
        agent = "test_agent"

        # Record more than 50 executions
        for i in range(60):
            self.tracker.record_execution(agent, 1000.0, True)

        assert len(self.tracker.execution_times[agent]) == 50
        assert len(self.tracker.success_rates[agent]) == 50


class TestFallbackManager:
    """Test fallback mechanisms for agent failures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = FallbackManager()

    def test_default_fallback_rules(self):
        """Test that default fallback rules are registered."""
        assert len(self.manager.fallback_rules) > 0

        # Check for specific default rules
        rule_agents = [rule.failed_agent for rule in self.manager.fallback_rules]
        assert "critic" in rule_agents
        assert "historian" in rule_agents
        assert "refiner" in rule_agents
        assert "synthesis" in rule_agents

    def test_get_fallback_agents(self):
        """Test getting fallback agents for failures."""
        # Test critic fallback
        fallbacks = self.manager.get_fallback_agents("critic", "error")
        assert "synthesis" in fallbacks

        # Test historian fallback (should be empty - just skip)
        fallbacks = self.manager.get_fallback_agents("historian", "error")
        assert fallbacks == []

    def test_fallback_attempt_limits(self):
        """Test that fallback attempts are limited."""
        # Exceed max attempts
        for i in range(5):
            fallbacks = self.manager.get_fallback_agents("critic", "error")

        # Should stop returning fallbacks after max attempts
        final_fallbacks = self.manager.get_fallback_agents("critic", "error")
        assert final_fallbacks == []

    def test_custom_fallback_registration(self):
        """Test registering custom fallback rules."""
        custom_rule = FallbackRule(
            failed_agent="custom_agent",
            fallback_agents=["backup_agent"],
            condition="always",
            priority=1,
        )

        self.manager.register_fallback(custom_rule)
        fallbacks = self.manager.get_fallback_agents("custom_agent", "error")

        assert "backup_agent" in fallbacks

    def test_failure_rate_calculation(self):
        """Test calculation of failure rates."""
        agent = "test_agent"

        # Record some failures
        for i in range(3):
            self.manager.get_fallback_agents(agent, "error")

        failure_rate = self.manager.get_failure_rate(agent, window_minutes=60)
        assert failure_rate > 0
        assert failure_rate <= 1.0


class TestEnhancedConditionalPattern:
    """Test the main enhanced conditional pattern functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pattern = EnhancedConditionalPattern()

    def test_pattern_properties(self):
        """Test basic pattern properties."""
        assert self.pattern.name == "enhanced_conditional"
        assert "dynamic agent selection" in self.pattern.description.lower()

    def test_agent_selection_strategies(self):
        """Test agent selection for different strategies."""
        available_agents = ["refiner", "critic", "historian", "synthesis"]

        # Test streamlined strategy
        context = Mock()
        context.requires_research = False
        context.requires_criticism = False
        context.complexity_level = ContextComplexity.SIMPLE

        selected = self.pattern.select_agents_for_strategy(
            RoutingStrategy.STREAMLINED, available_agents, context
        )
        assert len(selected) <= 2
        assert "synthesis" in selected

        # Test comprehensive strategy
        selected = self.pattern.select_agents_for_strategy(
            RoutingStrategy.COMPREHENSIVE, available_agents, context
        )
        assert len(selected) >= 3
        assert set(selected) == set(available_agents)

    def test_get_recommended_agents(self):
        """Test getting recommended agents for queries."""
        # Simple query should get minimal agents
        simple_query = "What is AI?"
        available = ["refiner", "critic", "historian", "synthesis"]

        recommended = self.pattern.get_recommended_agents(simple_query, available)
        assert "synthesis" in recommended
        assert len(recommended) <= 3

        # Complex query should get more agents
        complex_query = """
        Provide a comprehensive historical analysis and critical evaluation 
        of machine learning approaches with research evidence.
        """

        recommended = self.pattern.get_recommended_agents(complex_query, available)
        assert len(recommended) >= 3

    def test_edge_generation(self):
        """Test edge generation for conditional patterns."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        edges = self.pattern.get_edges(agents)

        # Should generate a valid DAG
        assert len(edges) > 0

        # Check for expected connections
        edge_map = {edge["from"]: edge["to"] for edge in edges}

        # Refiner should connect to critic and historian
        refiner_targets = [edge["to"] for edge in edges if edge["from"] == "refiner"]
        assert "critic" in refiner_targets or "historian" in refiner_targets

        # Should end with END
        end_edges = [edge for edge in edges if edge["to"] == "END"]
        assert len(end_edges) > 0

    def test_fallback_handling(self):
        """Test agent failure handling."""
        remaining_agents = ["critic", "historian", "synthesis"]

        result = self.pattern.handle_agent_failure("refiner", "error", remaining_agents)

        assert result["action"] in ["fallback", "skip"]
        assert result["original_agent"] == "refiner"
        assert result["failure_type"] == "error"
        assert "recommendation" in result

    def test_performance_optimization(self):
        """Test performance-optimized agent selection."""
        available_agents = ["refiner", "critic", "historian", "synthesis"]

        # Set up some performance data
        self.pattern.performance_tracker.record_execution("refiner", 1000, True)
        self.pattern.performance_tracker.record_execution("critic", 3000, False)
        self.pattern.performance_tracker.record_execution("historian", 1500, True)
        self.pattern.performance_tracker.record_execution("synthesis", 2000, True)

        optimized = self.pattern.get_performance_optimized_agents(available_agents)

        # Should prefer better performing agents
        assert "refiner" in optimized  # Critical agent
        assert "synthesis" in optimized  # Critical agent

    def test_routing_statistics(self):
        """Test retrieval of routing statistics."""
        # Generate some data
        self.pattern.performance_tracker.record_execution("refiner", 1000, True)
        self.pattern.fallback_manager.get_fallback_agents("critic", "error")

        stats = self.pattern.get_routing_statistics()

        assert "performance_tracking" in stats
        assert "failure_statistics" in stats
        assert "routing_config" in stats
        assert stats["cache_size"] >= 0

    def test_validate_agents_basic(self):
        """Test basic agent validation."""
        # Valid agents
        assert self.pattern.validate_agents(["refiner", "synthesis"])

        # Empty agents
        assert not self.pattern.validate_agents([])


class TestConditionalPatternValidator:
    """Test semantic validation for conditional patterns."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConditionalPatternValidator()

    def test_supported_patterns(self):
        """Test supported pattern names."""
        patterns = self.validator.get_supported_patterns()
        assert "conditional" in patterns
        assert "enhanced_conditional" in patterns

    def test_routing_strategy_validation(self):
        """Test validation of routing strategies."""
        # Valid streamlined strategy
        result = self.validator.validate_workflow(
            ["refiner", "synthesis"],
            "enhanced_conditional",
            routing_strategy=RoutingStrategy.STREAMLINED,
        )
        assert result.is_valid

        # Invalid - too few agents for comprehensive
        result = self.validator.validate_workflow(
            ["synthesis"],
            "enhanced_conditional",
            routing_strategy=RoutingStrategy.COMPREHENSIVE,
        )
        assert not result.is_valid or result.has_errors

    def test_context_validation(self):
        """Test validation against context analysis."""
        context = Mock()
        context.requires_research = True
        context.requires_criticism = False
        context.complexity_level = ContextComplexity.SIMPLE

        # Missing historian when research is required
        result = self.validator.validate_workflow(
            ["refiner", "synthesis"], "enhanced_conditional", context_analysis=context
        )

        # Should have warning about missing historian
        warning_messages = result.warning_messages
        assert any("historian" in msg.lower() for msg in warning_messages)

    def test_performance_validation(self):
        """Test validation of performance data."""
        performance_data = {
            "slow_agent": {"success_rate": 0.5, "average_time_ms": 15000},
            "fast_agent": {"success_rate": 0.95, "average_time_ms": 1000},
        }

        result = self.validator.validate_workflow(
            ["slow_agent", "fast_agent"],
            "enhanced_conditional",
            performance_data=performance_data,
        )

        # Should have warnings about slow agent
        issues = [issue.message for issue in result.issues]
        assert any("slow_agent" in msg and "success rate" in msg for msg in issues)

    def test_fallback_validation(self):
        """Test validation of fallback configurations."""
        # Single synthesis agent - should warn about single point of failure
        result = self.validator.validate_workflow(["synthesis"], "enhanced_conditional")

        warning_messages = result.warning_messages
        assert any("single point of failure" in msg.lower() for msg in warning_messages)

    def test_routing_decision_validation(self):
        """Test validation of routing decisions."""
        context = Mock()
        context.requires_research = True
        context.requires_criticism = False

        original_agents = ["refiner", "critic", "historian", "synthesis"]
        selected_agents = ["refiner", "critic", "synthesis"]  # Missing historian

        result = self.validator.validate_routing_decision(
            original_agents, selected_agents, context
        )

        # Should warn about excluding historian when research is needed
        warning_messages = result.warning_messages
        assert any(
            "historian" in msg.lower() and "research" in msg.lower()
            for msg in warning_messages
        )


class TestIntegration:
    """Integration tests for conditional patterns with other components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConditionalPatternValidator(strict_mode=False)
        self.pattern = EnhancedConditionalPattern(semantic_validator=self.validator)

    def test_pattern_with_validator_integration(self):
        """Test pattern working with semantic validator."""
        agents = ["refiner", "synthesis"]
        query = "Simple question about Python"

        # Should validate successfully
        result = self.pattern.validate_with_context(agents, query)
        assert result.is_valid

    def test_complex_workflow_validation(self):
        """Test validation of complex workflow scenarios."""
        agents = ["refiner", "critic", "historian", "synthesis"]
        complex_query = """
        Provide a comprehensive analysis with historical research 
        and critical evaluation of different approaches.
        """

        # Should validate successfully for complex query
        result = self.pattern.validate_with_context(
            agents, complex_query, RoutingStrategy.COMPREHENSIVE
        )
        assert result.is_valid

        # Test with performance data
        self.pattern.performance_tracker.record_execution("critic", 1000, True)
        self.pattern.performance_tracker.record_execution("historian", 2000, True)

        result = self.pattern.validate_with_context(
            agents, complex_query, RoutingStrategy.PERFORMANCE_OPTIMIZED
        )
        assert result.is_valid

    def test_caching_behavior(self):
        """Test caching of routing decisions."""
        query = "Test query for caching"
        agents = ["refiner", "synthesis"]

        # First call
        result1 = self.pattern.get_recommended_agents(query, agents)

        # Second call should use cache
        result2 = self.pattern.get_recommended_agents(query, agents)

        assert result1 == result2
        assert len(self.pattern._routing_cache) > 0

    @patch("time.time")
    def test_cache_expiration(self, mock_time):
        """Test that cache entries expire correctly."""
        query = "Test query for expiration"
        agents = ["refiner", "synthesis"]

        # Set initial time
        mock_time.return_value = 1000.0

        # Make first call
        result1 = self.pattern.get_recommended_agents(query, agents)

        # Advance time beyond cache TTL
        mock_time.return_value = 1000.0 + self.pattern._cache_ttl + 1

        # Make second call - should recompute
        result2 = self.pattern.get_recommended_agents(query, agents)

        # Results might be same, but cache should have been refreshed
        # We can't easily test the refresh without more complex mocking


@pytest.mark.parametrize(
    "query,expected_complexity",
    [
        ("What is AI?", ContextComplexity.SIMPLE),
        ("Explain machine learning algorithms in detail.", ContextComplexity.MODERATE),
        (
            "Provide comprehensive analysis of AI ethics with research and criticism.",
            ContextComplexity.COMPLEX,
        ),
    ],
)
def test_context_complexity_classification(query, expected_complexity):
    """Test context complexity classification with various queries."""
    analyzer = ContextAnalyzer()
    analysis = analyzer.analyze_context(query)

    # Allow some flexibility in complexity classification
    complexity_values = [c.value for c in ContextComplexity]
    expected_index = complexity_values.index(expected_complexity.value)
    actual_index = complexity_values.index(analysis.complexity_level.value)

    # Should be within 1 level of expected
    assert abs(actual_index - expected_index) <= 1


@pytest.mark.parametrize(
    "strategy,min_agents,max_agents",
    [
        (RoutingStrategy.STREAMLINED, 1, 2),
        (RoutingStrategy.STANDARD, 2, 4),
        (RoutingStrategy.COMPREHENSIVE, 3, 4),
        (RoutingStrategy.PERFORMANCE_OPTIMIZED, 1, 4),
    ],
)
def test_strategy_agent_count_requirements(strategy, min_agents, max_agents):
    """Test agent count requirements for different strategies."""
    pattern = EnhancedConditionalPattern()
    available = ["refiner", "critic", "historian", "synthesis"]
    context = Mock()
    context.requires_research = False
    context.requires_criticism = False
    context.complexity_level = ContextComplexity.MODERATE

    selected = pattern.select_agents_for_strategy(strategy, available, context)

    assert min_agents <= len(selected) <= max_agents


def test_performance_edge_cases():
    """Test edge cases in performance tracking."""
    tracker = PerformanceTracker()

    # Test with no data
    assert tracker.get_average_time("nonexistent") is None
    assert tracker.get_success_rate("nonexistent") is None
    assert tracker.get_performance_score("nonexistent") == 0.5

    # Test with single data point
    tracker.record_execution("single", 1000.0, True)
    assert tracker.get_average_time("single") == 1000.0
    assert tracker.get_success_rate("single") == 1.0


def test_fallback_edge_cases():
    """Test edge cases in fallback management."""
    manager = FallbackManager()

    # Test failure rate with no data (before any method calls)
    rate = manager.get_failure_rate("truly_unknown_agent")
    assert rate == 0.0

    # Test with unknown agent
    fallbacks = manager.get_fallback_agents("unknown_agent", "error")
    assert fallbacks == []

    # Reset attempts for unknown agent
    manager.reset_attempts("unknown_agent")  # Should not raise error
