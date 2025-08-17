"""
Routing Object Factory Functions for Testing.

Provides factory functions for creating routing decision and resource constraint
objects with sensible defaults to eliminate parameter unfilled warnings in tests
and improve maintainability.

This implements the Routing Object Factory Pattern for Testing as specified
in the API Service Architecture Specialist role requirements.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
import uuid

from cognivault.routing.routing_decision import (
    RoutingDecision,
    RoutingReasoning,
    RoutingConfidenceLevel,
)
from cognivault.routing.resource_optimizer import (
    ResourceConstraints,
    OptimizationStrategy,
)


class RoutingDecisionFactory:
    """Factory for creating RoutingDecision instances with sensible defaults."""

    @staticmethod
    def basic_routing_decision(
        selected_agents: Optional[List[str]] = None,
        routing_strategy: str = "balanced",
        confidence_score: float = 0.75,
        confidence_level: Optional[RoutingConfidenceLevel] = None,
        **overrides: Any,
    ) -> RoutingDecision:
        """Create basic routing decision with minimal required parameters."""
        if selected_agents is None:
            selected_agents = ["refiner", "critic"]

        if confidence_level is None:
            # Auto-determine confidence level from score
            if confidence_score >= 0.8:
                confidence_level = RoutingConfidenceLevel.VERY_HIGH
            elif confidence_score >= 0.6:
                confidence_level = RoutingConfidenceLevel.HIGH
            elif confidence_score >= 0.4:
                confidence_level = RoutingConfidenceLevel.MEDIUM
            elif confidence_score >= 0.2:
                confidence_level = RoutingConfidenceLevel.LOW
            else:
                confidence_level = RoutingConfidenceLevel.VERY_LOW

        return RoutingDecision(
            selected_agents=selected_agents,
            routing_strategy=routing_strategy,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            **overrides,
        )

    @staticmethod
    def comprehensive_routing_decision(
        selected_agents: Optional[List[str]] = None,
        routing_strategy: str = "capability_based",
        confidence_score: float = 0.85,
        available_agents: Optional[List[str]] = None,
        query_hash: str = "sha256:test123",
        **overrides: Any,
    ) -> RoutingDecision:
        """Create comprehensive routing decision with full metadata."""
        if selected_agents is None:
            selected_agents = ["refiner", "critic", "historian", "synthesis"]

        if available_agents is None:
            available_agents = [
                "refiner",
                "critic",
                "historian",
                "synthesis",
                "analyzer",
            ]

        # Auto-determine confidence level from score
        if confidence_score >= 0.8:
            confidence_level = RoutingConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            confidence_level = RoutingConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            confidence_level = RoutingConfidenceLevel.MEDIUM
        elif confidence_score >= 0.2:
            confidence_level = RoutingConfidenceLevel.LOW
        else:
            confidence_level = RoutingConfidenceLevel.VERY_LOW

        return RoutingDecision(
            selected_agents=selected_agents,
            routing_strategy=routing_strategy,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            available_agents=available_agents,
            query_hash=query_hash,
            execution_order=selected_agents.copy(),
            entry_point=selected_agents[0] if selected_agents else None,
            exit_points=[selected_agents[-1]] if selected_agents else [],
            **overrides,
        )

    @staticmethod
    def minimal_routing_decision(
        selected_agents: Optional[List[str]] = None,
        routing_strategy: str = "minimal",
        confidence_score: float = 0.5,
        **overrides: Any,
    ) -> RoutingDecision:
        """Create minimal routing decision for edge case testing."""
        if selected_agents is None:
            selected_agents = ["refiner"]

        return RoutingDecision(
            selected_agents=selected_agents,
            routing_strategy=routing_strategy,
            confidence_score=confidence_score,
            confidence_level=RoutingConfidenceLevel.MEDIUM,
            **overrides,
        )

    @staticmethod
    def performance_optimized_decision(
        selected_agents: Optional[List[str]] = None,
        routing_strategy: str = "performance",
        confidence_score: float = 0.9,
        estimated_total_time_ms: float = 2500.0,
        **overrides: Any,
    ) -> RoutingDecision:
        """Create performance-optimized routing decision with timing estimates."""
        if selected_agents is None:
            selected_agents = ["refiner", "synthesis"]  # Fast agents

        return RoutingDecision(
            selected_agents=selected_agents,
            routing_strategy=routing_strategy,
            confidence_score=confidence_score,
            confidence_level=RoutingConfidenceLevel.VERY_HIGH,
            estimated_total_time_ms=estimated_total_time_ms,
            estimated_success_probability=0.95,
            execution_order=selected_agents.copy(),
            **overrides,
        )


class RoutingReasoningFactory:
    """Factory for creating RoutingReasoning instances with sensible defaults."""

    @staticmethod
    def basic_reasoning(
        strategy_rationale: str = "Selected agents based on task analysis",
        **overrides: Any,
    ) -> RoutingReasoning:
        """Create basic routing reasoning with default analysis."""
        return RoutingReasoning(
            strategy_rationale=strategy_rationale,
            complexity_analysis={
                "complexity_score": 0.7,
                "factors": ["multi_step", "reasoning_required"],
            },
            performance_analysis={
                "latency_estimate_ms": 2500,
                "throughput_score": 0.8,
            },
            resource_analysis={
                "memory_requirement": "medium",
                "compute_intensity": 0.6,
            },
            agent_selection_rationale={
                "refiner": "Query needs clarification",
                "critic": "Analysis requires validation",
            },
            **overrides,
        )

    @staticmethod
    def comprehensive_reasoning(
        strategy_rationale: str = "Multi-agent approach for complex analysis",
        risks: Optional[List[str]] = None,
        mitigation_strategies: Optional[List[str]] = None,
        **overrides: Any,
    ) -> RoutingReasoning:
        """Create comprehensive reasoning with risk analysis."""
        if risks is None:
            risks = ["High latency possible", "Resource contention risk"]

        if mitigation_strategies is None:
            mitigation_strategies = [
                "Implement timeout fallback",
                "Use resource throttling",
            ]

        return RoutingReasoning(
            strategy_rationale=strategy_rationale,
            complexity_analysis={
                "complexity_score": 0.8,
                "factors": ["multi_step", "reasoning_required", "context_sensitive"],
                "domain_complexity": "high",
            },
            performance_analysis={
                "latency_estimate_ms": 3500,
                "throughput_score": 0.7,
                "resource_utilization": 0.75,
            },
            resource_analysis={
                "memory_requirement": "high",
                "compute_intensity": 0.8,
                "network_usage": 0.3,
            },
            agent_selection_rationale={
                "refiner": "Query needs clarification and preprocessing",
                "historian": "Context retrieval required from knowledge base",
                "critic": "Analysis requires critical evaluation",
                "synthesis": "Final integration and response generation needed",
            },
            risks_identified=risks,
            mitigation_strategies=mitigation_strategies,
            fallback_options=[
                "Fall back to sequential execution",
                "Use cached response",
            ],
            estimated_execution_time_ms=3500.0,
            estimated_success_probability=0.85,
            resource_utilization_estimate={"cpu": 0.7, "memory": 0.8, "network": 0.3},
            **overrides,
        )

    @staticmethod
    def minimal_reasoning(
        strategy_rationale: str = "Simple agent selection",
        **overrides: Any,
    ) -> RoutingReasoning:
        """Create minimal reasoning for basic scenarios."""
        return RoutingReasoning(
            strategy_rationale=strategy_rationale,
            **overrides,
        )


class ResourceConstraintsFactory:
    """Factory for creating ResourceConstraints instances with sensible defaults."""

    @staticmethod
    def basic_constraints(
        max_agents: Optional[int] = 4,
        min_agents: Optional[int] = 2,
        min_success_rate: float = 0.7,
        max_failure_rate: float = 0.3,
        **overrides: Any,
    ) -> ResourceConstraints:
        """Create basic resource constraints for typical scenarios."""
        return ResourceConstraints(
            max_agents=max_agents,
            min_agents=min_agents,
            min_success_rate=min_success_rate,
            max_failure_rate=max_failure_rate,
            **overrides,
        )

    @staticmethod
    def performance_constraints(
        max_execution_time_ms: float = 30000.0,
        max_agent_time_ms: float = 5000.0,
        max_agents: int = 3,
        min_success_rate: float = 0.8,
        max_failure_rate: float = 0.2,
        **overrides: Any,
    ) -> ResourceConstraints:
        """Create performance-focused resource constraints."""
        return ResourceConstraints(
            max_execution_time_ms=max_execution_time_ms,
            max_agent_time_ms=max_agent_time_ms,
            max_agents=max_agents,
            min_success_rate=min_success_rate,
            max_failure_rate=max_failure_rate,
            **overrides,
        )

    @staticmethod
    def agent_selection_constraints(
        required_agents: Optional[Set[str]] = None,
        forbidden_agents: Optional[Set[str]] = None,
        max_agents: Optional[int] = 5,
        **overrides: Any,
    ) -> ResourceConstraints:
        """Create constraints focused on agent selection rules."""
        if required_agents is None:
            required_agents = set()

        if forbidden_agents is None:
            forbidden_agents = set()

        return ResourceConstraints(
            required_agents=required_agents,
            forbidden_agents=forbidden_agents,
            max_agents=max_agents,
            **overrides,
        )

    @staticmethod
    def strict_constraints(
        required_agents: Optional[Set[str]] = None,
        max_agents: int = 2,
        min_agents: int = 2,
        min_success_rate: float = 0.9,
        max_failure_rate: float = 0.1,
        max_execution_time_ms: float = 15000.0,
        **overrides: Any,
    ) -> ResourceConstraints:
        """Create strict resource constraints for high-quality scenarios."""
        if required_agents is None:
            required_agents = {"refiner", "synthesis"}

        return ResourceConstraints(
            required_agents=required_agents,
            max_agents=max_agents,
            min_agents=min_agents,
            min_success_rate=min_success_rate,
            max_failure_rate=max_failure_rate,
            max_execution_time_ms=max_execution_time_ms,
            **overrides,
        )

    @staticmethod
    def minimal_constraints(**overrides: Any) -> ResourceConstraints:
        """Create minimal constraints with defaults only."""
        return ResourceConstraints(**overrides)


class RoutingTestPatterns:
    """Common routing test patterns combining multiple factory objects."""

    @staticmethod
    def complete_routing_scenario(
        query_complexity: float = 0.7,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ) -> tuple[RoutingDecision, ResourceConstraints]:
        """Create a complete routing scenario with decision and constraints."""
        constraints = ResourceConstraintsFactory.basic_constraints()

        # Select agents based on complexity
        if query_complexity > 0.8:
            selected_agents = ["refiner", "critic", "historian", "synthesis"]
            routing_strategy = "comprehensive"
            confidence_score = 0.9
        elif query_complexity > 0.5:
            selected_agents = ["refiner", "critic", "synthesis"]
            routing_strategy = "balanced"
            confidence_score = 0.8
        else:
            selected_agents = ["refiner", "synthesis"]
            routing_strategy = "minimal"
            confidence_score = 0.7

        decision = RoutingDecisionFactory.comprehensive_routing_decision(
            selected_agents=selected_agents,
            routing_strategy=routing_strategy,
            confidence_score=confidence_score,
        )

        return decision, constraints

    @staticmethod
    def performance_scenario() -> tuple[RoutingDecision, ResourceConstraints]:
        """Create a performance-focused routing scenario."""
        constraints = ResourceConstraintsFactory.performance_constraints()
        decision = RoutingDecisionFactory.performance_optimized_decision()
        return decision, constraints

    @staticmethod
    def constrained_scenario(
        required_agents: Set[str],
        forbidden_agents: Optional[Set[str]] = None,
    ) -> tuple[RoutingDecision, ResourceConstraints]:
        """Create a scenario with specific agent constraints."""
        constraints = ResourceConstraintsFactory.agent_selection_constraints(
            required_agents=required_agents,
            forbidden_agents=forbidden_agents or set(),
        )

        decision = RoutingDecisionFactory.basic_routing_decision(
            selected_agents=list(required_agents),
            routing_strategy="constraint_based",
        )

        return decision, constraints
