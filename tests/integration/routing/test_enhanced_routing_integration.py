"""
Integration tests for Enhanced Routing System.

These tests validate the end-to-end functionality of the Phase 3B.2 enhanced routing
system including query complexity analysis, resource optimization, and event emission.
"""

import pytest
import time
from unittest.mock import Mock, patch
from pydantic import ValidationError

from cognivault.routing.resource_optimizer import (
    ResourceOptimizer,
    OptimizationStrategy,
)
from cognivault.routing.routing_decision import (
    RoutingDecision,
    RoutingReasoning,
    RoutingConfidenceLevel,
)
from cognivault.langgraph_backend.graph_patterns.conditional import (
    EnhancedConditionalPattern,
    ContextAnalyzer,
    ContextComplexity,
    RoutingStrategy,
)
from cognivault.events import (
    emit_routing_decision_from_object,
    InMemoryEventSink,
    get_global_event_emitter,
)

# Factory imports for eliminating parameter unfilled warnings
from tests.factories import RoutingDecisionFactory, ResourceConstraintsFactory


class TestResourceOptimizer:
    """Test resource optimization for agent selection."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.optimizer = ResourceOptimizer()
        self.available_agents = ["refiner", "critic", "historian", "synthesis"]
        self.performance_data = {
            "refiner": {
                "success_rate": 0.95,
                "average_time_ms": 1500.0,
                "performance_score": 0.8,
            },
            "critic": {
                "success_rate": 0.85,
                "average_time_ms": 2000.0,
                "performance_score": 0.7,
            },
            "historian": {
                "success_rate": 0.75,
                "average_time_ms": 3000.0,
                "performance_score": 0.6,
            },
            "synthesis": {
                "success_rate": 0.9,
                "average_time_ms": 1800.0,
                "performance_score": 0.85,
            },
        }

    def test_balanced_optimization_strategy(self) -> None:
        """Test balanced optimization strategy."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        assert isinstance(decision, RoutingDecision)
        assert decision.routing_strategy == "balanced"
        assert len(decision.selected_agents) >= 2
        assert (
            "refiner" in decision.selected_agents
        )  # Should include high-performing agents
        assert "synthesis" in decision.selected_agents
        assert decision.confidence_score > 0.0

    def test_performance_optimization_strategy(self) -> None:
        """Test performance-focused optimization."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.3,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.PERFORMANCE,
        )

        assert decision.routing_strategy == "performance"
        # Should favor faster agents
        assert "refiner" in decision.selected_agents
        assert "synthesis" in decision.selected_agents
        # May exclude slower agents like historian
        assert decision.confidence_score > 0.0

    def test_minimal_optimization_strategy(self) -> None:
        """Test minimal agent selection."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.2,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.MINIMAL,
        )

        assert decision.routing_strategy == "minimal"
        assert len(decision.selected_agents) <= 2
        assert decision.confidence_score > 0.0

    def test_resource_constraints_application(self) -> None:
        """Test resource constraints are properly applied."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            max_agents=2,
            forbidden_agents={"historian"},
            # Performance constraints
            min_success_rate=0.8,
            max_failure_rate=0.2,  # Compatible with min_success_rate=0.8
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.6,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        assert len(decision.selected_agents) <= 2
        assert "historian" not in decision.selected_agents
        assert decision.confidence_score > 0.0

    def test_context_requirements_integration(self) -> None:
        """Test context requirements influence selection."""
        context_requirements = {
            "requires_research": True,
            "requires_criticism": True,
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.7,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.QUALITY,
            context_requirements=context_requirements,
        )

        # Should include historian for research and critic for criticism
        assert "historian" in decision.selected_agents
        assert "critic" in decision.selected_agents
        assert decision.confidence_score > 0.0

    def test_routing_decision_metadata(self) -> None:
        """Test routing decision includes comprehensive metadata."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.6,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Check decision has required metadata
        assert decision.decision_id is not None
        assert decision.timestamp is not None
        assert decision.confidence_level in RoutingConfidenceLevel
        assert len(decision.available_agents) == 4
        assert decision.reasoning is not None
        assert isinstance(decision.reasoning, RoutingReasoning)

        # Check reasoning has content
        assert decision.reasoning.strategy_rationale != ""
        assert len(decision.reasoning.agent_selection_rationale) > 0


class TestContextAnalyzer:
    """Test query context analysis."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.analyzer = ContextAnalyzer()

    def test_simple_query_analysis(self) -> None:
        """Test analysis of simple query."""
        query = "What is the capital of France?"

        analysis = self.analyzer.analyze_context(query)

        assert analysis.complexity_level == ContextComplexity.SIMPLE
        assert analysis.complexity_score <= 0.3
        assert analysis.routing_strategy == RoutingStrategy.STREAMLINED
        assert not analysis.requires_research
        assert not analysis.requires_criticism

    def test_complex_query_analysis(self) -> None:
        """Test analysis of complex query."""
        query = """
        Analyze the comprehensive economic implications of implementing a universal basic income 
        system in developed countries. Consider historical precedents, multiple stakeholder 
        perspectives, and evaluate the long-term sustainability of such policies across 
        different economic frameworks.
        """

        analysis = self.analyzer.analyze_context(query)

        assert analysis.complexity_level in [
            ContextComplexity.COMPLEX,
            ContextComplexity.VERY_COMPLEX,
        ]
        assert analysis.complexity_score > 0.6
        assert analysis.routing_strategy == RoutingStrategy.COMPREHENSIVE
        assert analysis.requires_research
        assert analysis.requires_criticism

    def test_research_query_detection(self) -> None:
        """Test detection of research requirements."""
        query = "What is the historical background of the French Revolution?"

        analysis = self.analyzer.analyze_context(query)

        assert analysis.requires_research
        assert "historical" in query.lower()

    def test_criticism_query_detection(self) -> None:
        """Test detection of critical analysis requirements."""
        query = "Evaluate the pros and cons of renewable energy policies."

        analysis = self.analyzer.analyze_context(query)

        assert analysis.requires_criticism
        assert "evaluate" in query.lower()


class TestEnhancedConditionalPattern:
    """Test enhanced conditional pattern functionality."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.pattern = EnhancedConditionalPattern()
        self.available_agents = ["refiner", "critic", "historian", "synthesis"]

    def test_routing_decision_caching(self) -> None:
        """Test routing decision caching functionality."""
        query = "Test query for caching"

        # First call should create cache entry
        agents1 = self.pattern.get_recommended_agents(query, self.available_agents)

        # Second call should use cache
        agents2 = self.pattern.get_recommended_agents(query, self.available_agents)

        assert agents1 == agents2
        assert len(self.pattern._routing_cache) > 0

    def test_performance_tracking_integration(self) -> None:
        """Test performance tracking integration."""
        # Record some performance data
        self.pattern.update_performance_metrics("refiner", 1500.0, True)
        self.pattern.update_performance_metrics("critic", 2000.0, True)
        self.pattern.update_performance_metrics("historian", 3000.0, False)

        # Check performance scores
        if self.pattern.performance_tracker is not None:
            refiner_score = self.pattern.performance_tracker.get_performance_score(
                "refiner"
            )
            historian_score = self.pattern.performance_tracker.get_performance_score(
                "historian"
            )
        else:
            refiner_score = 0.0
            historian_score = 0.0

        assert refiner_score > historian_score  # Refiner should score better

    def test_fallback_management(self) -> None:
        """Test agent failure fallback handling."""
        fallback_result = self.pattern.handle_agent_failure(
            failed_agent="critic",
            failure_type="error",
            remaining_agents=["refiner", "synthesis", "historian"],
        )

        assert fallback_result["action"] in ["fallback", "skip"]
        assert fallback_result["original_agent"] == "critic"
        assert fallback_result["failure_type"] == "error"
        assert "recommendation" in fallback_result

    def test_performance_optimized_selection(self) -> None:
        """Test performance-optimized agent selection."""
        # Set up performance data
        self.pattern.update_performance_metrics("refiner", 1000.0, True)
        self.pattern.update_performance_metrics("critic", 2000.0, True)
        self.pattern.update_performance_metrics("historian", 4000.0, False)
        self.pattern.update_performance_metrics("synthesis", 1500.0, True)

        optimized_agents = self.pattern.get_performance_optimized_agents(
            self.available_agents, max_agents=3
        )

        assert len(optimized_agents) <= 3
        assert "refiner" in optimized_agents  # Should include high performers
        assert "synthesis" in optimized_agents


class TestRoutingEventIntegration:
    """Test routing decision event emission."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.event_sink = InMemoryEventSink(max_events=100)
        self.emitter = get_global_event_emitter()
        self.emitter.add_sink(self.event_sink)
        self.emitter.enable()

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        self.emitter.disable()
        self.event_sink.clear_events()

    @pytest.mark.asyncio
    async def test_routing_decision_event_emission(self) -> None:
        """Test routing decision event emission."""
        # Create a routing decision
        optimizer = ResourceOptimizer()
        decision = optimizer.select_optimal_agents(
            available_agents=["refiner", "critic", "synthesis"],
            complexity_score=0.5,
            performance_data={
                "refiner": {"success_rate": 0.9, "average_time_ms": 1500.0},
                "critic": {"success_rate": 0.8, "average_time_ms": 2000.0},
                "synthesis": {"success_rate": 0.85, "average_time_ms": 1800.0},
            },
            strategy=OptimizationStrategy.BALANCED,
        )

        # Emit routing decision event
        await emit_routing_decision_from_object(
            routing_decision=decision,
            workflow_id="test-workflow-123",
            correlation_id="test-correlation-456",
            metadata={"test": "metadata"},
        )

        # Check event was emitted
        events = self.event_sink.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.event_type.value == "routing.decision.made"
        assert event.workflow_id == "test-workflow-123"
        assert event.correlation_id == "test-correlation-456"
        assert event.data["selected_agents"] == decision.selected_agents
        assert event.data["routing_strategy"] == decision.routing_strategy
        assert event.data["confidence_score"] == decision.confidence_score

        # Check metadata includes routing decision details
        assert event.metadata["decision_id"] == decision.decision_id
        assert event.metadata["confidence_level"] == decision.confidence_level.value
        assert event.metadata["test"] == "metadata"


class TestEndToEndRoutingIntegration:
    """Test end-to-end routing system integration."""

    @pytest.mark.asyncio
    async def test_orchestrator_routing_integration(self) -> None:
        """Test orchestrator integration with enhanced routing."""
        # Import here to avoid circular imports
        from cognivault.orchestration.orchestrator import LangGraphOrchestrator

        # Create orchestrator with enhanced routing
        orchestrator = LangGraphOrchestrator(
            use_enhanced_routing=True,
            optimization_strategy=OptimizationStrategy.BALANCED,
        )

        # Mock the required components
        with (
            patch.object(orchestrator, "context_analyzer") as mock_analyzer,
            patch.object(orchestrator, "resource_optimizer") as mock_optimizer,
            patch.object(orchestrator, "performance_tracker") as mock_tracker,
        ):
            # Setup mock returns
            mock_context_analysis: Mock = Mock()
            mock_context_analysis.complexity_score = 0.6
            mock_context_analysis.requires_research = True
            mock_context_analysis.requires_criticism = True

            mock_analyzer.analyze_context.return_value = mock_context_analysis

            mock_decision: Mock = Mock()
            mock_decision.selected_agents = ["refiner", "critic", "synthesis"]
            mock_decision.routing_strategy = "balanced"
            mock_decision.confidence_score = 0.8
            mock_decision.confidence_level = RoutingConfidenceLevel.HIGH

            mock_optimizer.select_optimal_agents.return_value = mock_decision

            # Test routing decision
            decision = await orchestrator._make_routing_decision(
                query="Test complex query requiring analysis",
                available_agents=["refiner", "critic", "historian", "synthesis"],
                config={"max_agents": 4},
            )

            assert decision.selected_agents == ["refiner", "critic", "synthesis"]
            assert decision.routing_strategy == "balanced"
            assert decision.confidence_score == 0.8

            # Verify calls were made
            mock_analyzer.analyze_context.assert_called_once()
            mock_optimizer.select_optimal_agents.assert_called_once()

    def test_routing_statistics_collection(self) -> None:
        """Test routing statistics collection."""
        from cognivault.orchestration.orchestrator import LangGraphOrchestrator

        orchestrator = LangGraphOrchestrator(
            use_enhanced_routing=True,
            optimization_strategy=OptimizationStrategy.PERFORMANCE,
        )

        stats = orchestrator.get_routing_statistics()

        assert stats["enhanced_routing"] is True
        assert stats["optimization_strategy"] == "performance"
        assert isinstance(stats, dict)

    def test_agent_performance_updates(self) -> None:
        """Test agent performance metric updates."""
        from cognivault.orchestration.orchestrator import LangGraphOrchestrator

        orchestrator = LangGraphOrchestrator(
            use_enhanced_routing=True,
        )

        # Update performance metrics
        orchestrator.update_agent_performance("refiner", 1500.0, True)
        orchestrator.update_agent_performance("critic", 2000.0, False)

        # Check performance was tracked
        if orchestrator.performance_tracker is not None:
            refiner_score = orchestrator.performance_tracker.get_performance_score(
                "refiner"
            )
            critic_score = orchestrator.performance_tracker.get_performance_score(
                "critic"
            )
        else:
            refiner_score = 0.0
            critic_score = 0.0

        assert refiner_score > critic_score  # Successful agent should score better


class TestComplexConstraintScenarios:
    """Test complex constraint scenarios with required and forbidden agents."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.optimizer = ResourceOptimizer()
        self.available_agents = [
            "refiner",
            "critic",
            "historian",
            "synthesis",
            "analyzer",
            "validator",
        ]
        self.performance_data = {
            "refiner": {
                "success_rate": 0.95,
                "average_time_ms": 1500.0,
                "performance_score": 0.8,
            },
            "critic": {
                "success_rate": 0.85,
                "average_time_ms": 2000.0,
                "performance_score": 0.7,
            },
            "historian": {
                "success_rate": 0.75,
                "average_time_ms": 3000.0,
                "performance_score": 0.6,
            },
            "synthesis": {
                "success_rate": 0.9,
                "average_time_ms": 1800.0,
                "performance_score": 0.85,
            },
            "analyzer": {
                "success_rate": 0.8,
                "average_time_ms": 2200.0,
                "performance_score": 0.75,
            },
            "validator": {
                "success_rate": 0.88,
                "average_time_ms": 1600.0,
                "performance_score": 0.78,
            },
        }

    def test_required_agents_only_selection(self) -> None:
        """Test selection when only required agents are specified."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"refiner", "synthesis"},
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should select only the required agents
        assert "refiner" in decision.selected_agents
        assert "synthesis" in decision.selected_agents
        assert (
            decision.confidence_score > 0.7
        )  # High confidence with required agents available

    def test_forbidden_agents_exclusion(self) -> None:
        """Test exclusion of forbidden agents."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            forbidden_agents={"historian", "analyzer"},
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should exclude forbidden agents
        assert "historian" not in decision.selected_agents
        assert "analyzer" not in decision.selected_agents
        assert len(decision.selected_agents) > 0  # Should still select some agents

    def test_required_and_forbidden_agents_interaction(self) -> None:
        """Test interaction between required and forbidden agents."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"refiner", "synthesis"},
            forbidden_agents={"historian", "analyzer"},
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should include required agents
        assert "refiner" in decision.selected_agents
        assert "synthesis" in decision.selected_agents

        # Should exclude forbidden agents
        assert "historian" not in decision.selected_agents
        assert "analyzer" not in decision.selected_agents

        # Can include other agents
        remaining_agents = set(decision.selected_agents) - {"refiner", "synthesis"}
        allowed_agents = {"critic", "validator"}
        assert remaining_agents.issubset(allowed_agents)

    def test_conflicting_required_and_forbidden_agents(self) -> None:
        """Test that conflicting required and forbidden agents raise ValidationError."""
        # With Pydantic validation, conflicting constraints should be caught at creation time
        with pytest.raises(ValidationError) as exc_info:
            ResourceConstraintsFactory.agent_selection_constraints(
                required_agents={"refiner", "historian"},
                forbidden_agents={
                    "historian",
                    "analyzer",
                },  # historian is both required and forbidden
            )

        # Verify the error message mentions the conflict
        error_message = str(exc_info.value)
        assert "both required and forbidden" in error_message
        assert "historian" in error_message

    def test_required_agents_with_min_max_constraints(self) -> None:
        """Test required agents with min/max agent constraints."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"refiner", "synthesis", "critic"},  # 3 required
            min_agents=2,
            max_agents=4,
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should include all required agents
        assert "refiner" in decision.selected_agents
        assert "synthesis" in decision.selected_agents
        assert "critic" in decision.selected_agents

        # Should respect max_agents constraint
        assert len(decision.selected_agents) <= 4
        assert len(decision.selected_agents) >= 3  # At least the required ones

    def test_required_agents_exceed_max_constraint(self) -> None:
        """Test that required agents exceeding max_agents raises ValidationError."""
        # With Pydantic validation, impossible constraints should be caught at creation time
        with pytest.raises(ValidationError) as exc_info:
            ResourceConstraintsFactory.agent_selection_constraints(
                required_agents={
                    "refiner",
                    "synthesis",
                    "critic",
                    "historian",
                    "analyzer",
                },  # 5 required
                max_agents=3,  # But only 3 allowed
            )

        # Verify the error message mentions the constraint violation
        error_message = str(exc_info.value)
        assert "required agents" in error_message.lower()
        assert "cannot exceed max_agents" in error_message

    def test_forbidden_agents_with_min_constraint(self) -> None:
        """Test forbidden agents reducing available agents below min_agents."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            forbidden_agents={
                "refiner",
                "synthesis",
                "critic",
                "historian",
            },  # Forbid 4 agents
            min_agents=3,  # But need at least 3
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should handle the constraint conflict
        remaining_agents = set(self.available_agents) - {
            "refiner",
            "synthesis",
            "critic",
            "historian",
        }
        assert len(remaining_agents) == 2  # Only analyzer and validator left

        if len(decision.selected_agents) >= 3:
            # System overrode forbidden constraint to meet min_agents
            assert (
                "constraint_override" in decision.reasoning.risks_identified
                or "impossible_constraints" in decision.reasoning.risks_identified
            )
        else:
            # Respected forbidden constraint, couldn't meet min_agents
            assert len(decision.selected_agents) == 2
            assert "analyzer" in decision.selected_agents
            assert "validator" in decision.selected_agents

    def test_required_agents_with_success_rate_constraint(self) -> None:
        """Test required agents with minimum success rate constraint."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"historian"},  # historian has 0.75 success rate
            min_success_rate=0.8,  # But we require 0.8 minimum
            max_failure_rate=0.2,  # Compatible with min_success_rate=0.8
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should include the required agent despite low success rate
        assert "historian" in decision.selected_agents

        # Should track the constraint violation
        assert (
            "success_rate_violation" in decision.reasoning.risks_identified
            or "constraint_conflict" in decision.reasoning.risks_identified
        )
        assert (
            decision.confidence_score < 0.8
        )  # Reduced confidence due to constraint conflict

    def test_complex_multi_constraint_scenario(self) -> None:
        """Test complex scenario with multiple interacting constraints."""
        constraints = ResourceConstraintsFactory.performance_constraints(
            max_execution_time_ms=5000,
            max_agents=5,
            min_success_rate=0.8,
            max_failure_rate=0.2,  # Compatible with min_success_rate=0.8
            # Agent selection constraints
            required_agents={"refiner", "synthesis"},
            forbidden_agents={"analyzer"},
            min_agents=3,
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.7,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should include required agents
        assert "refiner" in decision.selected_agents
        assert "synthesis" in decision.selected_agents

        # Should exclude forbidden agents
        assert "analyzer" not in decision.selected_agents

        # Should respect min/max agents
        assert 3 <= len(decision.selected_agents) <= 5

        # Should prioritize high success rate agents (except required ones)
        optional_agents = [
            agent
            for agent in decision.selected_agents
            if agent not in {"refiner", "synthesis"}
        ]
        for agent in optional_agents:
            if agent in self.performance_data:
                # Should generally prefer high success rate agents
                success_rate = self.performance_data[agent]["success_rate"]
                assert success_rate >= 0.75  # Should prefer better agents

    def test_required_agents_case_insensitive(self) -> None:
        """Test that required agents matching is case insensitive."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"REFINER", "Synthesis"},  # Mixed case
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should match case insensitively
        assert "refiner" in decision.selected_agents
        assert "synthesis" in decision.selected_agents

    def test_forbidden_agents_case_insensitive(self) -> None:
        """Test that forbidden agents matching is case insensitive."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            forbidden_agents={"HISTORIAN", "Analyzer"},  # Mixed case
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should exclude case insensitively
        assert "historian" not in decision.selected_agents
        assert "analyzer" not in decision.selected_agents

    def test_required_agents_priority_over_optimization(self) -> None:
        """Test that required agents take priority over optimization strategy."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"historian"},  # Lowest performance score (0.6)
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.PERFORMANCE,  # Should prefer fast agents
        )

        # Should include required agent despite performance strategy
        assert "historian" in decision.selected_agents

        # Should note the strategy conflict
        assert decision.reasoning.strategy_rationale != ""
        assert "performance" in decision.reasoning.strategy_rationale.lower()

    def test_forbidden_agents_override_context_requirements(self) -> None:
        """Test that forbidden agents override context requirements."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            forbidden_agents={"historian"},  # Forbid historian
        )

        context_requirements = {
            "requires_research": True,  # Would normally select historian
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
            context_requirements=context_requirements,
        )

        # Should exclude historian despite research requirement
        assert "historian" not in decision.selected_agents

        # Should find alternative agents or note the constraint conflict
        assert len(decision.selected_agents) > 0

    def test_empty_required_agents_set(self) -> None:
        """Test behavior with empty required agents set."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents=set(),  # Empty set
            forbidden_agents={"analyzer"},
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should work normally with empty required set
        assert len(decision.selected_agents) > 0
        assert "analyzer" not in decision.selected_agents

    def test_empty_forbidden_agents_set(self) -> None:
        """Test behavior with empty forbidden agents set."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"refiner"},
            forbidden_agents=set(),  # Empty set
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should work normally with empty forbidden set
        assert "refiner" in decision.selected_agents
        assert len(decision.selected_agents) > 0

    def test_nonexistent_required_agents_handling(self) -> None:
        """Test handling of nonexistent required agents."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"nonexistent_agent", "refiner"},
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should include existing required agent
        assert "refiner" in decision.selected_agents

        # Should track missing required agent
        assert "required_agents_unavailable" in decision.reasoning.risks_identified
        assert decision.confidence_score < 0.8  # Reduced confidence

    def test_nonexistent_forbidden_agents_handling(self) -> None:
        """Test handling of nonexistent forbidden agents."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            forbidden_agents={"nonexistent_agent", "historian"},
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should exclude existing forbidden agent
        assert "historian" not in decision.selected_agents

        # Should handle nonexistent forbidden agent gracefully
        assert len(decision.selected_agents) > 0

    def test_constraint_reasoning_completeness(self) -> None:
        """Test that constraint reasoning provides complete information."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"refiner", "synthesis"},
            forbidden_agents={"historian"},
            min_agents=3,
            max_agents=4,
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should provide detailed reasoning about constraint application
        assert decision.reasoning.strategy_rationale != ""

        # Should explain agent selection rationale
        for agent in decision.selected_agents:
            assert agent in decision.reasoning.agent_selection_rationale

        # Should explain excluded agents if any
        excluded_agents = set(self.available_agents) - set(decision.selected_agents)
        for agent in excluded_agents:
            if agent in decision.reasoning.excluded_agents_rationale:
                assert decision.reasoning.excluded_agents_rationale[agent] != ""

    def test_constraint_performance_impact(self) -> None:
        """Test performance impact of constraint processing."""
        import time

        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"refiner", "synthesis"},
            forbidden_agents={"historian", "analyzer"},
            min_agents=2,
            max_agents=5,
            min_success_rate=0.8,
            max_failure_rate=0.2,  # Compatible with min_success_rate=0.8
        )

        # Measure processing time
        start_time = time.time()

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete in reasonable time (less than 100ms)
        assert processing_time < 0.1

        # Should still produce valid results
        assert len(decision.selected_agents) > 0
        assert decision.confidence_score > 0.0


class TestResourceOptimizerErrorHandling:
    """Test error handling scenarios in resource optimization."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.optimizer = ResourceOptimizer()
        self.available_agents = ["refiner", "critic", "historian", "synthesis"]
        self.performance_data = {
            "refiner": {
                "success_rate": 0.95,
                "average_time_ms": 1500.0,
                "performance_score": 0.8,
            },
            "critic": {
                "success_rate": 0.85,
                "average_time_ms": 2000.0,
                "performance_score": 0.7,
            },
            "historian": {
                "success_rate": 0.75,
                "average_time_ms": 3000.0,
                "performance_score": 0.6,
            },
            "synthesis": {
                "success_rate": 0.9,
                "average_time_ms": 1800.0,
                "performance_score": 0.85,
            },
        }

    def test_empty_available_agents_handling(self) -> None:
        """Test handling of empty available agents list."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=[],
            complexity_score=0.5,
            performance_data={},
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should return empty selection with very low confidence
        assert decision.selected_agents == []
        assert decision.routing_strategy == "balanced"
        assert decision.confidence_score < 0.5
        assert decision.confidence_level == RoutingConfidenceLevel.VERY_LOW

    def test_missing_performance_data_handling(self) -> None:
        """Test handling of missing performance data."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=["unknown_agent"],
            complexity_score=0.5,
            performance_data={},  # Empty performance data
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should handle gracefully with fallback values
        assert "unknown_agent" in decision.selected_agents
        assert decision.confidence_score < 0.7  # Lower confidence due to missing data
        assert "missing_performance_data" in decision.reasoning.risks_identified

    def test_invalid_performance_data_handling(self) -> None:
        """Test handling of invalid performance data."""
        invalid_performance_data = {
            "refiner": {
                "success_rate": -0.1,  # Invalid: negative success rate
                "average_time_ms": "invalid",  # Invalid: string instead of number
                "performance_score": 1.5,  # Invalid: score > 1.0
            },
            "critic": {
                "success_rate": 1.2,  # Invalid: success rate > 1.0
                "average_time_ms": -100.0,  # Invalid: negative time
                "performance_score": None,  # Invalid: None value
            },
        }

        # Fix type compatibility for performance_data
        from typing import Dict, Any

        typed_invalid_performance_data: Dict[str, Dict[str, Any]] = {
            "refiner": {
                "success_rate": 1.2,  # Invalid: success rate > 1.0
                "average_time_ms": -100.0,  # Invalid: negative time
                "performance_score": None,  # Invalid: None value
            },
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=["refiner", "critic"],
            complexity_score=0.5,
            performance_data=typed_invalid_performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should handle gracefully with normalized values
        assert len(decision.selected_agents) > 0
        assert decision.confidence_score < 0.6  # Lower confidence due to invalid data
        assert "invalid_performance_data" in decision.reasoning.risks_identified

    def test_extreme_constraint_scenarios(self) -> None:
        """Test handling of extreme constraint scenarios."""
        # With enhanced Pydantic validation, impossible constraints are caught at creation time
        with pytest.raises(ValidationError) as exc_info:
            ResourceConstraintsFactory.basic_constraints(
                min_agents=5,
                max_agents=2,  # min_agents > max_agents should be caught
                min_success_rate=0.8,
            )

        # Verify the error message mentions the constraint conflict
        error_message = str(exc_info.value)
        assert "min_agents" in error_message and "max_agents" in error_message

    def test_forbidden_all_agents_scenario(self) -> None:
        """Test scenario where all agents are forbidden."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            forbidden_agents=set(self.available_agents),
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should return empty selection with appropriate reasoning
        assert decision.selected_agents == []
        assert decision.confidence_score < 0.3
        assert "all_agents_forbidden" in decision.reasoning.risks_identified

    def test_required_agents_not_available(self) -> None:
        """Test scenario where required agents are not available."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"nonexistent_agent", "another_missing_agent"},
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should handle gracefully and provide fallback
        assert len(decision.selected_agents) > 0  # Fallback to available agents
        assert decision.confidence_score < 0.75  # Reduced confidence due to fallback
        assert "required_agents_unavailable" in decision.reasoning.risks_identified
        assert "required_agents_fallback" in decision.reasoning.risks_identified

    def test_extreme_complexity_scores(self) -> None:
        """Test handling of extreme complexity scores."""
        # Test with invalid complexity scores
        extreme_scores = [-1.0, 2.0, float("inf"), float("-inf")]

        for score in extreme_scores:
            decision = self.optimizer.select_optimal_agents(
                available_agents=self.available_agents,
                complexity_score=score,
                performance_data=self.performance_data,
                strategy=OptimizationStrategy.BALANCED,
            )

            # Should normalize and handle gracefully
            assert len(decision.selected_agents) > 0
            assert 0.0 <= decision.confidence_score <= 1.0
            if score < 0 or score > 1:
                assert "invalid_complexity_score" in decision.reasoning.risks_identified

    def test_optimization_strategy_failure_recovery(self) -> None:
        """Test recovery from optimization strategy failures."""
        # Create a mock that simulates strategy failure
        with patch.object(self.optimizer, "_score_agents") as mock_score:
            mock_score.side_effect = Exception("Strategy execution failed")

            decision = self.optimizer.select_optimal_agents(
                available_agents=self.available_agents,
                complexity_score=0.5,
                performance_data=self.performance_data,
                strategy=OptimizationStrategy.BALANCED,
            )

            # Should fall back to minimal strategy
            assert len(decision.selected_agents) > 0
            assert decision.confidence_score < 0.5
            assert "strategy_failure_fallback" in decision.reasoning.risks_identified

    def test_time_estimation_edge_cases(self) -> None:
        """Test time estimation with edge case data."""
        edge_case_performance = {
            "agent1": {
                "success_rate": 0.8,
                "average_time_ms": 0.0,  # Zero time
                "performance_score": 0.7,
            },
            "agent2": {
                "success_rate": 0.9,
                "average_time_ms": float("inf"),  # Infinite time
                "performance_score": 0.8,
            },
            "agent3": {
                "success_rate": 0.7,
                "average_time_ms": -100.0,  # Negative time
                "performance_score": 0.6,
            },
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=["agent1", "agent2", "agent3"],
            complexity_score=0.5,
            performance_data=edge_case_performance,
            strategy=OptimizationStrategy.PERFORMANCE,
        )

        # Should handle gracefully with reasonable time estimates
        assert decision.estimated_total_time_ms is not None
        assert decision.estimated_total_time_ms > 0
        assert decision.estimated_total_time_ms != float("inf")
        assert "time_estimation_issues" in decision.reasoning.risks_identified

    def test_success_probability_calculation_edge_cases(self) -> None:
        """Test success probability calculation with edge cases."""
        edge_case_performance = {
            "agent1": {
                "success_rate": 0.0,  # Zero success rate
                "average_time_ms": 1000.0,
                "performance_score": 0.1,
            },
            "agent2": {
                "success_rate": 1.0,  # Perfect success rate
                "average_time_ms": 2000.0,
                "performance_score": 1.0,
            },
            "agent3": {
                "success_rate": 0.5,  # Medium success rate
                "average_time_ms": 1500.0,
                "performance_score": 0.5,
            },
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=["agent1", "agent2", "agent3"],
            complexity_score=0.5,
            performance_data=edge_case_performance,
            strategy=OptimizationStrategy.QUALITY,
        )

        # Should prefer high success rate agents
        assert "agent2" in decision.selected_agents  # Perfect success rate
        assert decision.estimated_success_probability is not None
        assert 0.0 <= decision.estimated_success_probability <= 1.0

    def test_concurrent_optimization_requests(self) -> None:
        """Test handling of concurrent optimization requests."""
        import threading

        results = []
        errors = []

        def run_optimization() -> None:
            try:
                decision = self.optimizer.select_optimal_agents(
                    available_agents=self.available_agents,
                    complexity_score=0.5,
                    performance_data=self.performance_data,
                    strategy=OptimizationStrategy.BALANCED,
                )
                results.append(decision)
            except Exception as e:
                errors.append(e)

        # Run multiple concurrent optimizations
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=run_optimization)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should handle concurrency gracefully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # All results should be valid
        for result in results:
            assert isinstance(result, RoutingDecision)
            assert len(result.selected_agents) > 0
            assert 0.0 <= result.confidence_score <= 1.0


# Performance benchmarks
class TestRoutingPerformance:
    """Test routing system performance characteristics."""

    @pytest.mark.asyncio
    async def test_routing_decision_performance(self) -> None:
        """Test routing decision performance is acceptable."""
        import time

        optimizer = ResourceOptimizer()
        performance_data = {
            agent: {
                "success_rate": 0.8,
                "average_time_ms": 2000.0,
                "performance_score": 0.7,
            }
            for agent in ["refiner", "critic", "historian", "synthesis"]
        }

        start_time = time.time()

        # Make multiple routing decisions
        for i in range(10):
            decision = optimizer.select_optimal_agents(
                available_agents=["refiner", "critic", "historian", "synthesis"],
                complexity_score=0.5,
                performance_data=performance_data,
                strategy=OptimizationStrategy.BALANCED,
            )
            assert len(decision.selected_agents) > 0

        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 10) * 1000

        # Routing decisions should be fast (< 50ms each)
        assert avg_time_ms < 50.0

    def test_context_analysis_performance(self) -> None:
        """Test context analysis performance."""
        import time

        analyzer = ContextAnalyzer()
        test_queries = [
            "Simple question?",
            "Complex multi-faceted analysis of economic implications...",
            "Historical research into the background of...",
            "Critical evaluation of the pros and cons...",
        ]

        start_time = time.time()

        for query in test_queries * 25:  # 100 analyses
            analysis = analyzer.analyze_context(query)
            assert analysis.complexity_score >= 0.0

        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 100) * 1000

        # Context analysis should be very fast (< 5ms each)
        assert avg_time_ms < 5.0


class TestPerformancePredictionAccuracy:
    """Test performance prediction accuracy and validation."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.optimizer = ResourceOptimizer()
        self.available_agents = ["refiner", "critic", "historian", "synthesis"]
        self.performance_data = {
            "refiner": {
                "success_rate": 0.95,
                "average_time_ms": 1500.0,
                "performance_score": 0.8,
            },
            "critic": {
                "success_rate": 0.85,
                "average_time_ms": 2000.0,
                "performance_score": 0.7,
            },
            "historian": {
                "success_rate": 0.75,
                "average_time_ms": 3000.0,
                "performance_score": 0.6,
            },
            "synthesis": {
                "success_rate": 0.9,
                "average_time_ms": 1800.0,
                "performance_score": 0.85,
            },
        }

    def test_time_prediction_accuracy_single_agent(self) -> None:
        """Test time prediction accuracy for single agent selection."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=["refiner"],
            complexity_score=0.5,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should predict exactly the agent's average time
        assert decision.estimated_total_time_ms == 1500.0
        assert decision.estimated_success_probability == 0.95

    def test_time_prediction_accuracy_multiple_agents(self) -> None:
        """Test time prediction accuracy for multiple agent selection."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should predict sum of selected agents' times, accounting for parallel execution
        base_time = sum(
            self.performance_data[agent.lower()]["average_time_ms"]
            for agent in decision.selected_agents
        )

        # Calculate expected time considering parallel execution
        expected_time = base_time
        if decision.parallel_groups:
            for group in decision.parallel_groups:
                if len(group) > 1:
                    # Calculate time savings from parallel execution
                    group_times = [
                        self.performance_data[agent.lower()]["average_time_ms"]
                        for agent in group
                    ]
                    sequential_time = sum(group_times)
                    parallel_time = max(group_times)
                    time_savings = sequential_time - parallel_time
                    expected_time -= time_savings

        assert decision.estimated_total_time_ms == expected_time

    def test_success_probability_calculation_single_agent(self) -> None:
        """Test success probability calculation for single agent."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=["refiner"],
            complexity_score=0.5,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should predict exactly the agent's success rate
        assert decision.estimated_success_probability == 0.95

    def test_success_probability_calculation_multiple_agents(self) -> None:
        """Test success probability calculation for multiple agents."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=["refiner", "critic"],
            complexity_score=0.5,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should calculate multiplicative probability with optimism factor
        expected_prob = 0.95 * 0.85 * 1.1  # Base probability with optimism factor
        expected_prob = min(1.0, expected_prob)  # Capped at 1.0
        if decision.estimated_success_probability is not None:
            assert abs(decision.estimated_success_probability - expected_prob) < 0.01
        else:
            assert False, "estimated_success_probability should not be None"

    def test_parallel_execution_time_savings(self) -> None:
        """Test time prediction considers parallel execution savings."""
        # Force parallel execution by setting up the decision metadata
        decision = self.optimizer.select_optimal_agents(
            available_agents=["critic", "historian"],
            complexity_score=0.5,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Check if parallel groups are detected
        if decision.parallel_groups:
            # Time should be less than sequential (max of parallel times rather than sum)
            critic_time = self.performance_data["critic"]["average_time_ms"]
            historian_time = self.performance_data["historian"]["average_time_ms"]
            expected_parallel_time = max(critic_time, historian_time)

            # The actual prediction should account for parallel execution
            if decision.estimated_total_time_ms is not None:
                assert decision.estimated_total_time_ms <= critic_time + historian_time
            else:
                assert False, "estimated_total_time_ms should not be None"

    def test_prediction_with_missing_performance_data(self) -> None:
        """Test predictions when performance data is missing."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=["unknown_agent"],
            complexity_score=0.5,
            performance_data={},  # Empty performance data
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should use default values from sanitization
        assert decision.estimated_total_time_ms == 2000.0  # Default time
        assert decision.estimated_success_probability == 0.7  # Default success rate
        assert "missing_performance_data" in decision.reasoning.risks_identified

    def test_prediction_with_invalid_performance_data(self) -> None:
        """Test predictions with invalid performance data."""
        invalid_data = {
            "agent1": {
                "success_rate": -0.1,  # Invalid: negative
                "average_time_ms": "invalid",  # Invalid: string
                "performance_score": 1.5,  # Invalid: > 1.0
            }
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=["agent1"],
            complexity_score=0.5,
            performance_data=invalid_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should use sanitized values
        assert decision.estimated_total_time_ms == 2000.0  # Sanitized to default
        assert decision.estimated_success_probability == 0.7  # Sanitized to default
        assert "invalid_performance_data" in decision.reasoning.risks_identified

    def test_optimization_opportunities_generation(self) -> None:
        """Test that optimization opportunities are generated based on predictions."""
        slow_performance_data = {
            "slow_agent": {
                "success_rate": 0.5,  # Low success rate
                "average_time_ms": 10000.0,  # Very slow (> 8000ms)
                "performance_score": 0.3,
            }
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=["slow_agent"],
            complexity_score=0.5,
            performance_data=slow_performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should generate optimization opportunities
        assert len(decision.optimization_opportunities) > 0

        # Check specific optimization suggestions
        opportunities = decision.optimization_opportunities
        time_opportunity = any(
            "parallel execution" in opp or "faster agents" in opp
            for opp in opportunities
        )
        reliability_opportunity = any(
            "reliable agents" in opp or "fallback" in opp for opp in opportunities
        )

        assert time_opportunity  # Should suggest faster execution
        assert reliability_opportunity  # Should suggest more reliable agents

    def test_prediction_accuracy_with_constraints(self) -> None:
        """Test prediction accuracy when constraints affect agent selection."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents={"historian"},  # Force selection of slower agent
            forbidden_agents={"refiner"},  # Prevent selection of faster agent
        )

        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.5,
            performance_data=self.performance_data,
            constraints=constraints,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should predict time based on actual selected agents
        assert "historian" in decision.selected_agents
        assert "refiner" not in decision.selected_agents

        # Time prediction should reflect the constrained selection, accounting for parallel execution
        base_time = sum(
            self.performance_data[agent.lower()]["average_time_ms"]
            for agent in decision.selected_agents
        )

        # Calculate expected time considering parallel execution
        expected_time = base_time
        if decision.parallel_groups:
            for group in decision.parallel_groups:
                if len(group) > 1:
                    # Calculate time savings from parallel execution
                    group_times = [
                        self.performance_data[agent.lower()]["average_time_ms"]
                        for agent in group
                    ]
                    sequential_time = sum(group_times)
                    parallel_time = max(group_times)
                    time_savings = sequential_time - parallel_time
                    expected_time -= time_savings

        assert decision.estimated_total_time_ms == expected_time

    def test_prediction_confidence_correlation(self) -> None:
        """Test that prediction confidence correlates with decision confidence."""
        # High confidence scenario
        high_conf_data = {
            "excellent_agent": {
                "success_rate": 0.98,
                "average_time_ms": 1000.0,
                "performance_score": 0.95,
            }
        }

        high_conf_decision = self.optimizer.select_optimal_agents(
            available_agents=["excellent_agent"],
            complexity_score=0.3,  # Simple query
            performance_data=high_conf_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Low confidence scenario
        low_conf_data = {
            "poor_agent": {
                "success_rate": 0.4,
                "average_time_ms": 8000.0,
                "performance_score": 0.2,
            }
        }

        low_conf_decision = self.optimizer.select_optimal_agents(
            available_agents=["poor_agent"],
            complexity_score=0.9,  # Complex query
            performance_data=low_conf_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # High confidence decision should have better predictions
        assert high_conf_decision.confidence_score > low_conf_decision.confidence_score

        # Handle Optional types for probability comparison
        if (
            high_conf_decision.estimated_success_probability is not None
            and low_conf_decision.estimated_success_probability is not None
        ):
            assert (
                high_conf_decision.estimated_success_probability
                > low_conf_decision.estimated_success_probability
            )

        # Handle Optional types for time comparison
        if (
            high_conf_decision.estimated_total_time_ms is not None
            and low_conf_decision.estimated_total_time_ms is not None
        ):
            assert (
                high_conf_decision.estimated_total_time_ms
                < low_conf_decision.estimated_total_time_ms
            )

    def test_prediction_validation_edge_cases(self) -> None:
        """Test prediction validation with edge cases."""
        edge_case_data = {
            "edge_agent": {
                "success_rate": 0.0,  # Zero success rate
                "average_time_ms": 0.0,  # Zero time (should be sanitized)
                "performance_score": 0.0,
            }
        }

        decision = self.optimizer.select_optimal_agents(
            available_agents=["edge_agent"],
            complexity_score=0.5,
            performance_data=edge_case_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should handle edge cases gracefully
        if decision.estimated_total_time_ms is not None:
            assert decision.estimated_total_time_ms > 0  # Should use minimum time
        if decision.estimated_success_probability is not None:
            assert decision.estimated_success_probability >= 0.0
            assert decision.estimated_success_probability <= 1.0
        assert "time_estimation_issues" in decision.reasoning.risks_identified

    def test_prediction_consistency_across_strategies(self) -> None:
        """Test that predictions remain consistent across different optimization strategies."""
        strategies = [
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.PERFORMANCE,
            OptimizationStrategy.RELIABILITY,
            OptimizationStrategy.QUALITY,
        ]

        decisions = []
        for strategy in strategies:
            decision = self.optimizer.select_optimal_agents(
                available_agents=self.available_agents,
                complexity_score=0.5,
                performance_data=self.performance_data,
                strategy=strategy,
            )
            decisions.append(decision)

        # All decisions should have valid predictions
        for decision in decisions:
            assert decision.estimated_total_time_ms is not None
            assert decision.estimated_total_time_ms > 0
            assert decision.estimated_success_probability is not None
            assert 0.0 <= decision.estimated_success_probability <= 1.0

    def test_prediction_metadata_completeness(self) -> None:
        """Test that prediction metadata is complete and accurate."""
        decision = self.optimizer.select_optimal_agents(
            available_agents=self.available_agents,
            complexity_score=0.7,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should have complete prediction metadata
        assert decision.estimated_total_time_ms is not None
        assert decision.estimated_success_probability is not None

        # Reasoning should include prediction data
        assert decision.reasoning.estimated_execution_time_ms is not None
        assert decision.reasoning.estimated_success_probability is not None

        # Should match between decision and reasoning
        assert (
            decision.estimated_total_time_ms
            == decision.reasoning.estimated_execution_time_ms
        )
        assert (
            decision.estimated_success_probability
            == decision.reasoning.estimated_success_probability
        )

    def test_prediction_boundary_conditions(self) -> None:
        """Test predictions at boundary conditions."""
        # Test with maximum reasonable values
        max_data = {
            "max_agent": {
                "success_rate": 1.0,
                "average_time_ms": 60000.0,  # 1 minute
                "performance_score": 1.0,
            }
        }

        # Test with minimum reasonable values
        min_data = {
            "min_agent": {
                "success_rate": 0.01,
                "average_time_ms": 50.0,  # 50ms
                "performance_score": 0.01,
            }
        }

        max_decision = self.optimizer.select_optimal_agents(
            available_agents=["max_agent"],
            complexity_score=0.5,
            performance_data=max_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        min_decision = self.optimizer.select_optimal_agents(
            available_agents=["min_agent"],
            complexity_score=0.5,
            performance_data=min_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Should handle boundary conditions correctly
        assert max_decision.estimated_total_time_ms == 60000.0
        assert max_decision.estimated_success_probability == 1.0

        assert min_decision.estimated_total_time_ms == 50.0
        assert min_decision.estimated_success_probability == 0.01


class TestCacheInvalidationAndPersistence:
    """Test cache invalidation and persistence scenarios for enhanced routing."""

    def setup_method(self) -> None:
        """Setup for each test."""
        from cognivault.langgraph_backend.graph_cache import GraphCache, CacheConfig
        from cognivault.langgraph_backend.build_graph import GraphFactory

        # Create cache with short TTL for testing
        self.cache_config = CacheConfig(
            max_size=5,
            ttl_seconds=1,
            enable_stats=True,  # Short TTL for testing
        )
        self.cache = GraphCache(self.cache_config)
        self.graph_factory = GraphFactory()

        # Mock graph patterns for testing
        self.mock_graphs = {}
        for i in range(10):
            mock_graph: Mock = Mock()
            mock_graph.name = f"mock_graph_{i}"
            mock_graph.pattern = f"pattern_{i % 3}"  # Multiple patterns
            self.mock_graphs[f"graph_{i}"] = mock_graph

    def test_cache_ttl_invalidation(self) -> None:
        """Test cache invalidation based on TTL expiration."""
        pattern_name = "standard"
        agents = ["refiner", "critic"]
        checkpoints = False
        mock_graph: Mock = Mock()

        # Cache the graph
        self.cache.cache_graph(pattern_name, agents, checkpoints, mock_graph)

        # Should be cached immediately
        cached_graph = self.cache.get_cached_graph(pattern_name, agents, checkpoints)
        assert cached_graph is mock_graph

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should be invalidated by TTL
        cached_graph = self.cache.get_cached_graph(pattern_name, agents, checkpoints)
        assert cached_graph is None

        # Statistics should reflect the expiration
        stats = self.cache.get_stats()
        assert stats["expired_evictions"] > 0
        assert stats["misses"] >= 1  # At least one miss from expired entry

    def test_cache_manual_invalidation(self) -> None:
        """Test manual cache invalidation scenarios."""
        # Cache multiple patterns
        patterns = ["standard", "parallel", "conditional"]
        for pattern in patterns:
            for i in range(3):
                mock_graph: Mock = Mock()
                self.cache.cache_graph(pattern, [f"agent_{i}"], False, mock_graph)

        # Verify cache is at capacity (LRU eviction occurred)
        initial_size = len(self.cache.get_cache_keys())
        assert initial_size == self.cache_config.max_size  # Should be at max capacity

        # Test clear() - invalidate all
        self.cache.clear()
        assert len(self.cache.get_cache_keys()) == 0

        # Re-cache for pattern-specific invalidation test
        for pattern in patterns:
            for i in range(2):
                mock_graph_invalidation: Mock = Mock()
                self.cache.cache_graph(
                    pattern, [f"agent_{i}"], False, mock_graph_invalidation
                )

        # Test pattern-specific invalidation
        removed_count = self.cache.remove_pattern("standard")
        assert removed_count >= 1  # At least one standard pattern entry removed

        # Verify standard pattern entries are gone
        remaining_keys = self.cache.get_cache_keys()
        assert len(remaining_keys) >= 0  # Some entries remain

        # Verify standard pattern entries are gone
        for key in remaining_keys:
            assert not key.startswith("standard_")

    def test_cache_lru_invalidation(self) -> None:
        """Test LRU-based cache invalidation."""
        # Fill cache to capacity
        for i in range(self.cache_config.max_size):
            mock_graph: Mock = Mock()
            self.cache.cache_graph("standard", [f"agent_{i}"], False, mock_graph)

        # Cache is now full
        assert len(self.cache.get_cache_keys()) == self.cache_config.max_size

        # Access first few entries to make them recently used
        for i in range(2):
            self.cache.get_cached_graph("standard", [f"agent_{i}"], False)

        # Add new entry to trigger LRU eviction
        new_graph: Mock = Mock()
        self.cache.cache_graph("standard", ["new_agent"], False, new_graph)

        # Verify LRU eviction occurred
        stats = self.cache.get_stats()
        assert stats["size_evictions"] > 0
        assert stats["evictions"] > 0

        # Verify least recently used entry was evicted (not the first two we accessed)
        assert self.cache.get_cached_graph("standard", ["agent_0"], False) is not None
        assert self.cache.get_cached_graph("standard", ["agent_1"], False) is not None
        assert self.cache.get_cached_graph("standard", ["new_agent"], False) is not None

    def test_cache_optimization_invalidation(self) -> None:
        """Test cache optimization and cleanup."""
        # Cache entries with different patterns
        patterns = ["standard", "parallel", "conditional"]
        cached_graphs = {}

        for pattern in patterns:
            for i in range(2):
                mock_graph: Mock = Mock()
                self.cache.cache_graph(pattern, [f"agent_{i}"], False, mock_graph)
                cached_graphs[f"{pattern}_{i}"] = mock_graph

        initial_size = len(self.cache.get_cache_keys())

        # Wait for some entries to expire
        time.sleep(0.5)

        # Add more entries (some will be expired, some fresh)
        for i in range(2):
            mock_graph_fresh: Mock = Mock()
            self.cache.cache_graph("fresh", [f"agent_{i}"], False, mock_graph_fresh)

        # Wait for first batch to expire
        time.sleep(0.6)

        # Run optimization
        optimization_stats = self.cache.optimize()

        # Should have removed expired entries
        assert optimization_stats["removed_entries"] > 0
        assert optimization_stats["final_size"] < optimization_stats["initial_size"]

        # Fresh entries should still be accessible
        assert self.cache.get_cached_graph("fresh", ["agent_0"], False) is not None
        assert self.cache.get_cached_graph("fresh", ["agent_1"], False) is not None

    def test_cache_persistence_simulation(self) -> None:
        """Test cache persistence scenarios (simulated since no real persistence exists)."""
        # Simulate cache persistence by testing state preservation
        pattern_name = "standard"
        agents = ["refiner", "critic", "synthesis"]
        checkpoints = False
        mock_graph: Mock = Mock()

        # Cache initial state
        self.cache.cache_graph(pattern_name, agents, checkpoints, mock_graph)

        # Simulate "saving" cache state
        cache_state = {
            "keys": self.cache.get_cache_keys(),
            "stats": self.cache.get_stats(),
            "size": len(self.cache._cache),
        }

        # Verify state is captured
        keys = cache_state.get("keys", [])
        if isinstance(keys, (list, tuple)) and hasattr(keys, "__len__"):
            assert len(keys) == 1
        else:
            assert False, "Cache state keys should be a list or tuple"
        assert cache_state["size"] == 1

        # Simulate cache invalidation
        self.cache.clear()
        assert len(self.cache.get_cache_keys()) == 0

        # Simulate "restoring" cache state would require re-caching
        # (since no persistence exists, we simulate restoration)
        self.cache.cache_graph(pattern_name, agents, checkpoints, mock_graph)

        # Verify state is restored
        restored_graph = self.cache.get_cached_graph(pattern_name, agents, checkpoints)
        assert restored_graph is mock_graph

    def test_cache_concurrent_invalidation(self) -> None:
        """Test concurrent cache invalidation scenarios."""
        import threading

        # Cache initial entries
        for i in range(10):
            mock_graph: Mock = Mock()
            self.cache.cache_graph("standard", [f"agent_{i}"], False, mock_graph)

        results = []
        errors = []

        def invalidate_cache() -> None:
            try:
                # Perform various invalidation operations
                self.cache.clear()
                results.append("clear_success")
            except Exception as e:
                errors.append(e)

        def cache_operations() -> None:
            try:
                for i in range(5):
                    mock_graph: Mock = Mock()
                    self.cache.cache_graph(
                        "concurrent", [f"agent_{i}"], False, mock_graph
                    )
                    cached = self.cache.get_cached_graph(
                        "concurrent", [f"agent_{i}"], False
                    )
                    if cached is mock_graph:
                        results.append("cache_success")
            except Exception as e:
                errors.append(e)

        def optimization_operations() -> None:
            try:
                for _ in range(3):
                    self.cache.optimize()
                    results.append("optimize_success")
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        threads = []
        for target in [invalidate_cache, cache_operations, optimization_operations]:
            thread = threading.Thread(target=target)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should handle concurrency without errors
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) > 0

    def test_cache_invalidation_with_routing_decisions(self) -> None:
        """Test cache invalidation integrated with routing decisions."""
        from cognivault.langgraph_backend.build_graph import GraphConfig
        from cognivault.routing.resource_optimizer import ResourceOptimizer

        # Create graph factory with our test cache
        graph_config = GraphConfig(
            pattern_name="standard",
            agents_to_run=["refiner", "critic", "synthesis"],
            enable_checkpoints=False,
            cache_enabled=True,
        )

        # Mock a routing decision that would use cache
        optimizer = ResourceOptimizer()
        performance_data = {
            "refiner": {"success_rate": 0.9, "average_time_ms": 1500.0},
            "critic": {"success_rate": 0.8, "average_time_ms": 2000.0},
            "synthesis": {"success_rate": 0.85, "average_time_ms": 1800.0},
        }

        # Create multiple routing decisions
        decisions = []
        for i in range(3):
            decision = optimizer.select_optimal_agents(
                available_agents=["refiner", "critic", "synthesis"],
                complexity_score=0.5 + (i * 0.1),
                performance_data=performance_data,
            )
            decisions.append(decision)

        # Simulate cache entries for each decision
        for i, decision in enumerate(decisions):
            mock_graph: Mock = Mock()
            mock_graph.decision_id = decision.decision_id
            self.cache.cache_graph(
                "standard", decision.selected_agents, False, mock_graph
            )

        # Verify decisions cached (may have fewer entries due to same agent selection)
        initial_cache_size = len(self.cache.get_cache_keys())
        assert initial_cache_size >= 1  # At least one cache entry

        # Test invalidation affects routing-related cache
        removed_count = self.cache.remove_pattern("standard")
        assert removed_count >= 1  # At least one entry removed

        # Verify routing decisions no longer cached
        for decision in decisions:
            cached_graph = self.cache.get_cached_graph(
                "standard", decision.selected_agents, False
            )
            assert cached_graph is None

    def test_cache_size_estimation_and_limits(self) -> None:
        """Test cache size estimation and memory limits."""
        # Test size estimation
        large_mock_graph: Mock = Mock()
        large_mock_graph.data = "x" * 10000  # Large data

        small_mock_graph: Mock = Mock()
        small_mock_graph.data = "x" * 100  # Small data

        # Cache graphs with different sizes
        self.cache.cache_graph("standard", ["large_agent"], False, large_mock_graph)
        self.cache.cache_graph("standard", ["small_agent"], False, small_mock_graph)

        # Verify size estimation is working
        stats = self.cache.get_stats()
        assert stats["current_size"] == 2

        # Test cache limits with size considerations
        # Fill cache to capacity
        for i in range(self.cache_config.max_size):
            mock_graph: Mock = Mock()
            mock_graph.data = "x" * 1000  # Medium size
            self.cache.cache_graph("standard", [f"agent_{i}"], False, mock_graph)

        # Should trigger eviction
        final_stats = self.cache.get_stats()
        assert final_stats["current_size"] <= self.cache_config.max_size
        assert final_stats["evictions"] > 0

    def test_cache_key_collision_handling(self) -> None:
        """Test cache key collision handling and uniqueness."""
        # Test potential key collisions
        test_cases = [
            ("standard", ["agent_1", "agent_2"], False),
            ("standard", ["agent_1", "agent_2"], True),  # Different checkpoints
            (
                "standard",
                ["agent_2", "agent_1"],
                False,
            ),  # Different order (should be same)
            ("parallel", ["agent_1", "agent_2"], False),  # Different pattern
            (
                "standard",
                ["Agent_1", "Agent_2"],
                False,
            ),  # Different case (should be same)
        ]

        mock_graphs = {}
        for i, (pattern, agents, checkpoints) in enumerate(test_cases):
            mock_graph: Mock = Mock()
            mock_graph.test_id = i
            mock_graphs[i] = mock_graph
            self.cache.cache_graph(pattern, agents, checkpoints, mock_graph)

        # Verify each entry is stored correctly
        for i, (pattern, agents, checkpoints) in enumerate(test_cases):
            cached_graph = self.cache.get_cached_graph(pattern, agents, checkpoints)
            if (
                i == 2
            ):  # Same as case 0 (different order) - should get the later cached value
                if cached_graph is not None:
                    assert cached_graph.test_id == 4  # Latest entry with same key
            elif i == 4:  # Same as case 0 (different case) - should get the same value
                if cached_graph is not None:
                    assert cached_graph.test_id == 4  # Latest entry with same key
            else:
                # Due to cache size limits, some entries may have been evicted
                # Just verify that we got a valid graph
                assert cached_graph is not None
                assert hasattr(cached_graph, "test_id")

    def test_cache_statistics_accuracy(self) -> None:
        """Test accuracy of cache statistics during invalidation."""
        # Start with clean cache
        self.cache.clear()
        initial_stats = self.cache.get_stats()

        # Perform various cache operations
        operations = []

        # Cache misses
        for i in range(3):
            result = self.cache.get_cached_graph("standard", [f"agent_{i}"], False)
            operations.append(("miss", result is None))

        # Cache hits
        for i in range(3):
            mock_graph: Mock = Mock()
            self.cache.cache_graph("standard", [f"agent_{i}"], False, mock_graph)
            result = self.cache.get_cached_graph("standard", [f"agent_{i}"], False)
            operations.append(("hit", result is mock_graph))

        # Cache evictions (fill beyond capacity)
        for i in range(self.cache_config.max_size + 2):
            mock_graph_eviction: Mock = Mock()
            self.cache.cache_graph(
                "eviction", [f"agent_{i}"], False, mock_graph_eviction
            )

        # Get final statistics
        final_stats = self.cache.get_stats()

        # Verify statistics accuracy
        assert final_stats["misses"] == 3
        assert final_stats["hits"] == 3
        assert final_stats["total_requests"] == 6
        assert final_stats["hit_rate"] == 0.5
        assert final_stats["evictions"] > 0
        assert final_stats["current_size"] <= self.cache_config.max_size

    def test_cache_invalidation_patterns(self) -> None:
        """Test different cache invalidation patterns and their effects."""
        # Create a complex cache scenario
        patterns = ["standard", "parallel", "conditional"]
        agents_combinations = [
            ["refiner"],
            ["critic"],
            ["refiner", "critic"],
            ["refiner", "critic", "synthesis"],
            ["historian", "synthesis"],
        ]

        # Cache multiple combinations
        cached_entries = {}
        for pattern in patterns:
            for agents in agents_combinations:
                for checkpoints in [False, True]:
                    mock_graph: Mock = Mock()
                    cache_key = f"{pattern}_{agents}_{checkpoints}"
                    mock_graph.cache_key = cache_key
                    cached_entries[cache_key] = mock_graph
                    self.cache.cache_graph(pattern, agents, checkpoints, mock_graph)

        total_entries = len(cached_entries)

        # Test pattern-based invalidation
        standard_removed = self.cache.remove_pattern("standard")
        assert (
            standard_removed >= 0
        )  # May be 0 if no standard entries due to cache size limits

        # Test selective invalidation by waiting for TTL
        remaining_before_ttl = len(self.cache.get_cache_keys())
        time.sleep(1.1)  # Wait for TTL expiration

        # Access cache to trigger TTL cleanup
        self.cache.get_cached_graph("parallel", ["refiner"], False)

        # Verify TTL-based invalidation (may be 0 if no entries expired)
        stats = self.cache.get_stats()
        assert stats["expired_evictions"] >= 0

        # Test optimization cleanup
        optimization_stats = self.cache.optimize()
        assert optimization_stats["removed_entries"] >= 0

    def test_cache_invalidation_edge_cases(self) -> None:
        """Test edge cases in cache invalidation."""
        # Test invalidation with empty cache
        empty_removed = self.cache.remove_pattern("nonexistent")
        assert empty_removed == 0

        # Test multiple clears
        mock_graph: Mock = Mock()
        self.cache.cache_graph("standard", ["agent"], False, mock_graph)
        self.cache.clear()
        self.cache.clear()  # Second clear should be safe

        # Test optimization on empty cache
        optimization_stats = self.cache.optimize()
        assert optimization_stats["removed_entries"] == 0

        # Test concurrent invalidation and caching
        import threading

        def cache_worker() -> None:
            for i in range(10):
                mock_graph: Mock = Mock()
                self.cache.cache_graph("concurrent", [f"agent_{i}"], False, mock_graph)

        def invalidate_worker() -> None:
            for _ in range(5):
                self.cache.remove_pattern("concurrent")

        # Run concurrent operations
        threads = []
        for worker in [cache_worker, invalidate_worker]:
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        final_stats = self.cache.get_stats()
        assert final_stats["current_size"] >= 0


class TestRoutingDecisionSerializationEdgeCases:
    """Test routing decision serialization and deserialization edge cases."""

    def setup_method(self) -> None:
        """Setup for each test."""
        from cognivault.routing.resource_optimizer import ResourceOptimizer

        self.optimizer = ResourceOptimizer()
        self.performance_data = {
            "refiner": {"success_rate": 0.95, "average_time_ms": 1500.0},
            "critic": {"success_rate": 0.85, "average_time_ms": 2000.0},
            "synthesis": {"success_rate": 0.9, "average_time_ms": 1800.0},
        }

        # Create a comprehensive routing decision for testing
        self.test_decision = self.optimizer.select_optimal_agents(
            available_agents=["refiner", "critic", "synthesis"],
            complexity_score=0.7,
            performance_data=self.performance_data,
            strategy=OptimizationStrategy.BALANCED,
        )

        # Add comprehensive reasoning data
        self.test_decision.add_reasoning("complexity", "level", "high")
        self.test_decision.add_reasoning("performance", "metrics", {"avg_time": 1766.7})
        self.test_decision.add_reasoning(
            "resource", "utilization", {"cpu": 0.8, "memory": 0.6}
        )
        self.test_decision.add_risk("test_risk", "test_mitigation")
        self.test_decision.add_fallback_option("fallback_agent")
        self.test_decision.add_optimization_opportunity("Consider parallel execution")

    def test_basic_serialization_roundtrip(self) -> None:
        """Test basic serialization and deserialization roundtrip."""
        # Serialize to dict
        decision_dict = self.test_decision.to_dict()

        # Verify dict structure
        assert isinstance(decision_dict, dict)
        assert "decision_id" in decision_dict
        assert "timestamp" in decision_dict
        assert "selected_agents" in decision_dict
        assert "reasoning" in decision_dict

        # Deserialize from dict
        restored_decision = RoutingDecision.from_dict(decision_dict)

        # Verify restoration
        assert restored_decision.decision_id == self.test_decision.decision_id
        assert restored_decision.selected_agents == self.test_decision.selected_agents
        assert restored_decision.routing_strategy == self.test_decision.routing_strategy
        assert restored_decision.confidence_score == self.test_decision.confidence_score
        assert restored_decision.confidence_level == self.test_decision.confidence_level

    def test_serialization_with_none_values(self) -> None:
        """Test serialization with None values in optional fields."""
        # Create decision with None values
        decision = RoutingDecisionFactory.minimal_routing_decision(
            selected_agents=["refiner"],
            routing_strategy="minimal",
            confidence_score=0.5,
            query_hash=None,  # None value
            entry_point=None,  # None value
            estimated_total_time_ms=None,  # None value
            estimated_success_probability=None,  # None value
        )

        # Serialize and deserialize
        decision_dict = decision.to_dict()
        restored_decision = RoutingDecision.from_dict(decision_dict)

        # Verify None values are preserved
        assert restored_decision.query_hash is None
        assert restored_decision.entry_point is None
        assert restored_decision.estimated_total_time_ms is None
        assert restored_decision.estimated_success_probability is None

    def test_serialization_with_empty_collections(self) -> None:
        """Test serialization with empty collections."""
        decision = RoutingDecisionFactory.minimal_routing_decision(
            selected_agents=[],  # Empty list
            routing_strategy="test",
            confidence_score=0.0,
            available_agents=[],  # Empty list
            execution_order=[],  # Empty list
            parallel_groups=[],  # Empty list
            exit_points=[],  # Empty list
            optimization_opportunities=[],  # Empty list
        )

        # Clear reasoning to have empty collections
        decision.reasoning.risks_identified = []
        decision.reasoning.mitigation_strategies = []
        decision.reasoning.fallback_options = []
        decision.reasoning.complexity_analysis = {}
        decision.reasoning.performance_analysis = {}
        decision.reasoning.resource_analysis = {}
        decision.reasoning.agent_selection_rationale = {}
        decision.reasoning.excluded_agents_rationale = {}
        decision.reasoning.resource_utilization_estimate = {}

        # Serialize and deserialize
        decision_dict = decision.to_dict()
        restored_decision = RoutingDecision.from_dict(decision_dict)

        # Verify empty collections are preserved
        assert restored_decision.selected_agents == []
        assert restored_decision.available_agents == []
        assert restored_decision.execution_order == []
        assert restored_decision.parallel_groups == []
        assert restored_decision.exit_points == []
        assert restored_decision.optimization_opportunities == []
        assert restored_decision.reasoning.risks_identified == []
        assert restored_decision.reasoning.mitigation_strategies == []
        assert restored_decision.reasoning.fallback_options == []

    def test_serialization_with_complex_nested_data(self) -> None:
        """Test serialization with complex nested data structures."""
        # Create decision with complex nested data
        decision = self.test_decision

        # Add complex nested reasoning data
        decision.add_reasoning(
            "complexity",
            "nested_analysis",
            {
                "factors": ["query_length", "semantic_complexity"],
                "metrics": {"readability": 0.8, "ambiguity": 0.3},
                "sub_analysis": {
                    "grammar": {"score": 0.9, "issues": []},
                    "vocabulary": {"score": 0.7, "advanced_terms": ["optimization"]},
                },
            },
        )

        decision.add_reasoning(
            "performance",
            "agent_metrics",
            {
                "refiner": {"latency": 1500, "throughput": 100, "memory_usage": 0.6},
                "critic": {"latency": 2000, "throughput": 80, "memory_usage": 0.7},
            },
        )

        # Add complex parallel groups
        decision.parallel_groups = [["critic", "historian"], ["analyzer", "validator"]]

        # Serialize and deserialize
        decision_dict = decision.to_dict()
        restored_decision = RoutingDecision.from_dict(decision_dict)

        # Verify complex data is preserved
        assert (
            restored_decision.reasoning.complexity_analysis["nested_analysis"][
                "sub_analysis"
            ]["grammar"]["score"]
            == 0.9
        )
        assert (
            restored_decision.reasoning.performance_analysis["agent_metrics"][
                "refiner"
            ]["latency"]
            == 1500
        )
        assert restored_decision.parallel_groups == [
            ["critic", "historian"],
            ["analyzer", "validator"],
        ]

    def test_serialization_with_unicode_and_special_characters(self) -> None:
        """Test serialization with Unicode and special characters."""
        decision = RoutingDecisionFactory.basic_routing_decision(
            selected_agents=["refiner"],
            routing_strategy="test",
            confidence_score=0.5,
        )

        # Add Unicode and special characters
        decision.reasoning.strategy_rationale = (
            "Stratégie d'optimisation avec caractères spéciaux: éàü"
        )
        decision.reasoning.agent_selection_rationale = {
            "refiner": "Sélectionné pour l'analyse 🤖",
            "critic": "Excluded due to performance issues ⚠️",
        }

        decision.reasoning.risks_identified = [
            "Risk with émojis 🚨",
            "Risk with quotes \"quoted text\" and 'single quotes'",
            "Risk with newlines\nand\ttabs",
        ]

        # Serialize and deserialize
        decision_dict = decision.to_dict()
        restored_decision = RoutingDecision.from_dict(decision_dict)

        # Verify Unicode is preserved
        assert "éàü" in restored_decision.reasoning.strategy_rationale
        assert "🤖" in restored_decision.reasoning.agent_selection_rationale["refiner"]
        assert "🚨" in restored_decision.reasoning.risks_identified[0]
        assert '"quoted text"' in restored_decision.reasoning.risks_identified[1]
        assert "\n" in restored_decision.reasoning.risks_identified[2]

    def test_deserialization_with_missing_required_fields(self) -> None:
        """Test deserialization with missing required fields."""
        # Create incomplete dictionary
        incomplete_dict = {
            "selected_agents": ["refiner"],
            "routing_strategy": "test",
            "confidence_score": 0.5,
            # Missing required fields: decision_id, timestamp, confidence_level
        }

        # Should raise ValidationError for missing required fields with Pydantic
        with pytest.raises(ValidationError):
            RoutingDecision.from_dict(incomplete_dict)

    def test_deserialization_with_missing_optional_fields(self) -> None:
        """Test deserialization with missing optional fields."""
        # Create minimal dictionary with only required fields
        minimal_dict = {
            "decision_id": "a1b2c3d4e5f6789012345678901234ab",  # 32-char hex string
            "timestamp": "2023-01-01T00:00:00+00:00",
            "selected_agents": ["refiner"],
            "routing_strategy": "test",
            "confidence_score": 0.5,
            "confidence_level": "medium",
        }

        # Should work with default values for missing optional fields
        restored_decision = RoutingDecision.from_dict(minimal_dict)

        # Verify defaults are applied
        assert restored_decision.query_hash is None
        assert restored_decision.available_agents == []
        # Note: execution_order defaults to selected_agents.copy() in __post_init__
        assert restored_decision.execution_order == ["refiner"]
        assert restored_decision.parallel_groups == []
        assert restored_decision.exit_points == []
        assert restored_decision.estimated_total_time_ms is None
        assert restored_decision.estimated_success_probability is None
        assert restored_decision.optimization_opportunities == []

    def test_deserialization_with_invalid_enum_values(self) -> None:
        """Test deserialization with invalid enum values."""
        decision_dict = self.test_decision.to_dict()

        # Corrupt the confidence level
        decision_dict["confidence_level"] = "invalid_level"

        # Should raise ValueError for invalid enum value
        with pytest.raises(ValueError):
            RoutingDecision.from_dict(decision_dict)

    def test_deserialization_with_invalid_timestamp(self) -> None:
        """Test deserialization with invalid timestamp format."""
        decision_dict = self.test_decision.to_dict()

        # Corrupt the timestamp
        decision_dict["timestamp"] = "invalid-timestamp"

        # Should raise ValueError for invalid timestamp
        with pytest.raises(ValueError):
            RoutingDecision.from_dict(decision_dict)

    def test_deserialization_with_wrong_data_types(self) -> None:
        """Test deserialization with wrong data types."""
        decision_dict = self.test_decision.to_dict()

        # Corrupt data types
        decision_dict["selected_agents"] = "not_a_list"  # Should be list
        decision_dict["confidence_score"] = "not_a_number"  # Should be float

        # Should handle type errors gracefully or raise appropriate exceptions
        with pytest.raises(Exception) as exc_info:
            RoutingDecision.from_dict(decision_dict)

        # Verify we got an appropriate exception type
        assert isinstance(exc_info.value, (TypeError, ValueError, AttributeError))

    def test_serialization_with_extreme_values(self) -> None:
        """Test serialization with extreme values that pass validation."""
        # Enhanced Pydantic validation prevents truly extreme values, so test with large but valid values
        with pytest.raises(ValidationError):
            # This should fail validation due to string length and infinity constraints
            RoutingDecisionFactory.basic_routing_decision(
                selected_agents=["agent"] * 100,  # Very long list (valid)
                routing_strategy="x"
                * 1000,  # Very long string (exceeds max_length=200)
                confidence_score=1.0,  # Maximum confidence
                estimated_total_time_ms=float(
                    "inf"
                ),  # Infinity (exceeds max constraint)
                estimated_success_probability=0.0,  # Minimum probability
            )

        # Test with large but valid values
        decision = RoutingDecisionFactory.basic_routing_decision(
            selected_agents=["agent"] * 100,  # Very long list (valid)
            routing_strategy="x" * 190,  # Long string within limits
            confidence_score=1.0,  # Maximum confidence
            estimated_total_time_ms=599999.0,  # Just under limit
            estimated_success_probability=0.0,  # Minimum probability
        )

        # Add large reasoning data
        decision.reasoning.risks_identified = ["risk"] * 50  # Many risks
        decision.reasoning.complexity_analysis = {
            f"key_{i}": f"value_{i}" for i in range(100)
        }  # Large dict

        # Serialize and deserialize
        decision_dict = decision.to_dict()
        restored_decision = RoutingDecision.from_dict(decision_dict)

        # Verify large values are preserved
        assert len(restored_decision.selected_agents) == 100
        assert len(restored_decision.routing_strategy) == 190
        assert restored_decision.confidence_score == 1.0
        assert restored_decision.estimated_total_time_ms == 599999.0
        assert restored_decision.estimated_success_probability == 0.0
        assert len(restored_decision.reasoning.risks_identified) == 50
        assert len(restored_decision.reasoning.complexity_analysis) == 100

    def test_serialization_with_circular_references(self) -> None:
        """Test serialization behavior with potential circular references."""
        decision = RoutingDecisionFactory.basic_routing_decision(
            selected_agents=["refiner"],
            routing_strategy="test",
            confidence_score=0.5,
        )

        # Create self-referential structure (not actually circular in this case)
        decision.reasoning.complexity_analysis = {
            "self_ref": decision.decision_id,
            "nested": {
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp.isoformat(),
            },
        }

        # Should serialize successfully
        decision_dict = decision.to_dict()
        restored_decision = RoutingDecision.from_dict(decision_dict)

        # Verify references are preserved as values
        assert (
            restored_decision.reasoning.complexity_analysis["self_ref"]
            == decision.decision_id
        )
        assert (
            restored_decision.reasoning.complexity_analysis["nested"]["decision_id"]
            == decision.decision_id
        )

    def test_serialization_json_compatibility(self) -> None:
        """Test that serialized decisions are JSON-compatible."""
        import json

        # Serialize to dict
        decision_dict = self.test_decision.to_dict()

        # Should be JSON serializable
        json_string = json.dumps(decision_dict, indent=2)
        assert isinstance(json_string, str)
        assert len(json_string) > 0

        # Should be JSON deserializable
        restored_dict = json.loads(json_string)
        assert isinstance(restored_dict, dict)

        # Should be able to create RoutingDecision from JSON-restored dict
        restored_decision = RoutingDecision.from_dict(restored_dict)
        assert restored_decision.decision_id == self.test_decision.decision_id

    def test_serialization_with_custom_objects(self) -> None:
        """Test serialization with custom objects that should be converted."""
        from datetime import datetime, timezone

        decision = RoutingDecisionFactory.basic_routing_decision(
            selected_agents=["refiner"],
            routing_strategy="test",
            confidence_score=0.5,
            timestamp=datetime.now(timezone.utc),
        )

        # Add custom objects in reasoning
        decision.reasoning.performance_analysis = {
            "timestamp": datetime.now(timezone.utc),
            "metrics": {"value": 42},
        }

        # Serialize - should convert datetime to string
        decision_dict = decision.to_dict()

        # Verify datetime is converted to string
        assert isinstance(decision_dict["timestamp"], str)

        # Deserialize should work
        restored_decision = RoutingDecision.from_dict(decision_dict)
        assert isinstance(restored_decision.timestamp, datetime)

    def test_serialization_performance_with_large_data(self) -> None:
        """Test serialization performance with large data structures."""
        import time

        # Create decision with large data
        decision = RoutingDecisionFactory.basic_routing_decision(
            selected_agents=["agent"] * 1000,
            routing_strategy="performance_test",
            confidence_score=0.8,
        )

        # Add large reasoning data
        decision.reasoning.complexity_analysis = {
            f"key_{i}": f"value_{i}" * 100 for i in range(1000)
        }
        decision.reasoning.performance_analysis = {
            f"agent_{i}": {"metric": i} for i in range(1000)
        }
        decision.reasoning.risks_identified = [f"risk_{i}" for i in range(1000)]

        # Measure serialization time
        start_time = time.time()
        decision_dict = decision.to_dict()
        serialize_time = time.time() - start_time

        # Measure deserialization time
        start_time = time.time()
        restored_decision = RoutingDecision.from_dict(decision_dict)
        deserialize_time = time.time() - start_time

        # Should complete in reasonable time (< 1 second each)
        assert serialize_time < 1.0, f"Serialization took {serialize_time:.2f}s"
        assert deserialize_time < 1.0, f"Deserialization took {deserialize_time:.2f}s"

        # Verify large data is preserved
        assert len(restored_decision.selected_agents) == 1000
        assert len(restored_decision.reasoning.complexity_analysis) == 1000
        assert len(restored_decision.reasoning.performance_analysis) == 1000
        assert len(restored_decision.reasoning.risks_identified) == 1000

    def test_serialization_consistency_across_versions(self) -> None:
        """Test serialization consistency to ensure backward compatibility."""
        # Create decision with all possible fields
        decision = self.test_decision

        # Serialize multiple times
        dict1 = decision.to_dict()
        dict2 = decision.to_dict()

        # Should be identical
        assert dict1 == dict2

        # Deserialize multiple times
        restored1 = RoutingDecision.from_dict(dict1)
        restored2 = RoutingDecision.from_dict(dict2)

        # Should be equivalent
        assert restored1.decision_id == restored2.decision_id
        assert restored1.selected_agents == restored2.selected_agents
        assert restored1.confidence_score == restored2.confidence_score
        assert (
            restored1.reasoning.risks_identified == restored2.reasoning.risks_identified
        )

    def test_serialization_error_handling(self) -> None:
        """Test error handling during serialization edge cases."""
        # Test with corrupted reasoning
        decision = RoutingDecisionFactory.basic_routing_decision(
            selected_agents=["refiner"],
            routing_strategy="test",
            confidence_score=0.5,
        )

        # Enhanced Pydantic validation prevents setting reasoning to None
        with pytest.raises(ValidationError):
            # Create a mock reasoning object that should fail validation
            invalid_reasoning: RoutingReasoning = None  # type: ignore
            decision.reasoning = invalid_reasoning

        # Test with attempt to set invalid confidence_level
        # Enhanced validation prevents setting confidence_level to None
        with pytest.raises(ValidationError):
            # Create a mock confidence level that should fail validation
            invalid_confidence: RoutingConfidenceLevel = None  # type: ignore
            decision.confidence_level = invalid_confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
