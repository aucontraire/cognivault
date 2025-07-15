"""
Routing Decision Data Structures.

This module defines the data structures for routing decisions,
providing comprehensive reasoning and confidence tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any


class ConfidenceLevel(Enum):
    """Confidence levels for routing decisions."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


@dataclass
class RoutingReasoning:
    """Detailed reasoning for routing decisions."""

    # Primary reasoning factors
    complexity_analysis: Dict[str, Any] = field(default_factory=dict)
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    resource_analysis: Dict[str, Any] = field(default_factory=dict)

    # Decision factors
    strategy_rationale: str = ""
    agent_selection_rationale: Dict[str, str] = field(default_factory=dict)
    excluded_agents_rationale: Dict[str, str] = field(default_factory=dict)

    # Risk assessment
    risks_identified: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    fallback_options: List[str] = field(default_factory=list)

    # Performance predictions
    estimated_execution_time_ms: Optional[float] = None
    estimated_success_probability: Optional[float] = None
    resource_utilization_estimate: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "complexity_analysis": self.complexity_analysis,
            "performance_analysis": self.performance_analysis,
            "resource_analysis": self.resource_analysis,
            "strategy_rationale": self.strategy_rationale,
            "agent_selection_rationale": self.agent_selection_rationale,
            "excluded_agents_rationale": self.excluded_agents_rationale,
            "risks_identified": self.risks_identified,
            "mitigation_strategies": self.mitigation_strategies,
            "fallback_options": self.fallback_options,
            "estimated_execution_time_ms": self.estimated_execution_time_ms,
            "estimated_success_probability": self.estimated_success_probability,
            "resource_utilization_estimate": self.resource_utilization_estimate,
        }


@dataclass
class RoutingDecision:
    """
    Comprehensive routing decision with reasoning and metadata.

    This class captures all aspects of an intelligent routing decision,
    including the selected agents, reasoning, confidence, and predictions.
    """

    # Core decision
    selected_agents: List[str]
    routing_strategy: str
    confidence_score: float
    confidence_level: ConfidenceLevel

    # Decision context
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_hash: Optional[str] = None
    available_agents: List[str] = field(default_factory=list)

    # Reasoning and analysis
    reasoning: RoutingReasoning = field(default_factory=RoutingReasoning)

    # Execution metadata
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    entry_point: Optional[str] = None
    exit_points: List[str] = field(default_factory=list)

    # Performance predictions
    estimated_total_time_ms: Optional[float] = None
    estimated_success_probability: Optional[float] = None
    optimization_opportunities: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization to set derived fields."""
        if not self.confidence_level:
            self.confidence_level = self._calculate_confidence_level()

        if not self.execution_order:
            self.execution_order = self.selected_agents.copy()

    def _calculate_confidence_level(self) -> ConfidenceLevel:
        """Calculate confidence level from numeric score."""
        if self.confidence_score <= 0.2:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence_score <= 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence_score <= 0.6:
            return ConfidenceLevel.MEDIUM
        elif self.confidence_score <= 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def add_reasoning(self, category: str, key: str, value: Any) -> None:
        """Add reasoning information to the decision."""
        if category == "complexity":
            self.reasoning.complexity_analysis[key] = value
        elif category == "performance":
            self.reasoning.performance_analysis[key] = value
        elif category == "resource":
            self.reasoning.resource_analysis[key] = value

    def add_agent_rationale(
        self, agent: str, rationale: str, included: bool = True
    ) -> None:
        """Add rationale for including or excluding an agent."""
        if included:
            self.reasoning.agent_selection_rationale[agent] = rationale
        else:
            self.reasoning.excluded_agents_rationale[agent] = rationale

    def add_risk(self, risk: str, mitigation: Optional[str] = None) -> None:
        """Add identified risk and optional mitigation strategy."""
        self.reasoning.risks_identified.append(risk)
        if mitigation:
            self.reasoning.mitigation_strategies.append(mitigation)

    def add_fallback_option(self, fallback: str) -> None:
        """Add fallback option for failure scenarios."""
        self.reasoning.fallback_options.append(fallback)

    def set_performance_prediction(
        self,
        total_time_ms: float,
        success_probability: float,
        resource_utilization: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set performance predictions for the routing decision."""
        self.estimated_total_time_ms = total_time_ms
        self.estimated_success_probability = success_probability
        self.reasoning.estimated_execution_time_ms = total_time_ms
        self.reasoning.estimated_success_probability = success_probability

        if resource_utilization:
            self.reasoning.resource_utilization_estimate = resource_utilization

    def add_optimization_opportunity(self, opportunity: str) -> None:
        """Add identified optimization opportunity."""
        self.optimization_opportunities.append(opportunity)

    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence decision."""
        return self.confidence_level in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
        ]

    def is_risky(self) -> bool:
        """Check if this decision has identified risks."""
        return len(self.reasoning.risks_identified) > 0

    def has_fallbacks(self) -> bool:
        """Check if fallback options are available."""
        return len(self.reasoning.fallback_options) > 0

    def get_excluded_agents(self) -> List[str]:
        """Get list of agents that were available but not selected."""
        return [
            agent
            for agent in self.available_agents
            if agent not in self.selected_agents
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert routing decision to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "selected_agents": self.selected_agents,
            "routing_strategy": self.routing_strategy,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "query_hash": self.query_hash,
            "available_agents": self.available_agents,
            "reasoning": self.reasoning.to_dict(),
            "execution_order": self.execution_order,
            "parallel_groups": self.parallel_groups,
            "entry_point": self.entry_point,
            "exit_points": self.exit_points,
            "estimated_total_time_ms": self.estimated_total_time_ms,
            "estimated_success_probability": self.estimated_success_probability,
            "optimization_opportunities": self.optimization_opportunities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """Create routing decision from dictionary."""
        reasoning_data = data.get("reasoning", {})
        reasoning = RoutingReasoning(
            complexity_analysis=reasoning_data.get("complexity_analysis", {}),
            performance_analysis=reasoning_data.get("performance_analysis", {}),
            resource_analysis=reasoning_data.get("resource_analysis", {}),
            strategy_rationale=reasoning_data.get("strategy_rationale", ""),
            agent_selection_rationale=reasoning_data.get(
                "agent_selection_rationale", {}
            ),
            excluded_agents_rationale=reasoning_data.get(
                "excluded_agents_rationale", {}
            ),
            risks_identified=reasoning_data.get("risks_identified", []),
            mitigation_strategies=reasoning_data.get("mitigation_strategies", []),
            fallback_options=reasoning_data.get("fallback_options", []),
            estimated_execution_time_ms=reasoning_data.get(
                "estimated_execution_time_ms"
            ),
            estimated_success_probability=reasoning_data.get(
                "estimated_success_probability"
            ),
            resource_utilization_estimate=reasoning_data.get(
                "resource_utilization_estimate", {}
            ),
        )

        return cls(
            decision_id=data["decision_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            selected_agents=data["selected_agents"],
            routing_strategy=data["routing_strategy"],
            confidence_score=data["confidence_score"],
            confidence_level=ConfidenceLevel(data["confidence_level"]),
            query_hash=data.get("query_hash"),
            available_agents=data.get("available_agents", []),
            reasoning=reasoning,
            execution_order=data.get("execution_order", []),
            parallel_groups=data.get("parallel_groups", []),
            entry_point=data.get("entry_point"),
            exit_points=data.get("exit_points", []),
            estimated_total_time_ms=data.get("estimated_total_time_ms"),
            estimated_success_probability=data.get("estimated_success_probability"),
            optimization_opportunities=data.get("optimization_opportunities", []),
        )
