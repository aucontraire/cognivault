"""
Resource Optimizer for Intelligent Agent Selection.

This module provides sophisticated resource optimization capabilities for
agent selection, considering performance, cost, and resource constraints.
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from cognivault.observability import get_logger
from .routing_decision import RoutingDecision, RoutingConfidenceLevel

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for agent selection."""

    PERFORMANCE = "performance"  # Optimize for execution speed
    RELIABILITY = "reliability"  # Optimize for success rate
    COST = "cost"  # Optimize for resource cost
    BALANCED = "balanced"  # Balance all factors
    QUALITY = "quality"  # Optimize for output quality
    MINIMAL = "minimal"  # Use minimal agents


class ResourceConstraints(BaseModel):
    """
    Resource constraints for optimization decisions.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    type safety, and integration with the CogniVault Pydantic ecosystem.
    """

    # Time constraints
    max_execution_time_ms: Optional[float] = Field(
        None,
        description="Maximum allowed execution time in milliseconds",
        ge=0.0,
        le=3600000.0,  # 1 hour max
        json_schema_extra={"example": 30000.0},
    )
    max_agent_time_ms: Optional[float] = Field(
        None,
        description="Maximum allowed time per agent in milliseconds",
        ge=0.0,
        le=600000.0,  # 10 minutes max per agent
        json_schema_extra={"example": 5000.0},
    )

    # Agent constraints
    max_agents: Optional[int] = Field(
        None,
        description="Maximum number of agents to select",
        ge=1,
        le=10,  # Reasonable upper limit
        json_schema_extra={"example": 4},
    )
    min_agents: Optional[int] = Field(
        None,
        description="Minimum number of agents to select",
        ge=1,
        le=10,
        json_schema_extra={"example": 2},
    )
    required_agents: Set[str] = Field(
        default_factory=set,
        description="Set of agent names that must be included",
        json_schema_extra={"example": ["refiner", "synthesis"]},
    )
    forbidden_agents: Set[str] = Field(
        default_factory=set,
        description="Set of agent names that must be excluded",
        json_schema_extra={"example": ["legacy_agent"]},
    )

    # Performance constraints
    min_success_rate: float = Field(
        0.7,
        description="Minimum required success rate for agent selection (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    max_failure_rate: float = Field(
        0.3,
        description="Maximum allowed failure rate for agent selection (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.15},
    )

    # Cost constraints (future extensibility)
    max_cost_per_request: Optional[float] = Field(
        None,
        description="Maximum cost allowed per request (future feature)",
        ge=0.0,
        json_schema_extra={"example": 5.00},
    )
    cost_per_agent: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost mapping per agent (future feature)",
        json_schema_extra={
            "example": {"refiner": 0.5, "critic": 0.3, "historian": 0.4}
        },
    )

    # Quality constraints
    min_quality_score: float = Field(
        0.6,
        description="Minimum quality score required for agent selection (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.75},
    )
    quality_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality weight mapping per agent",
        json_schema_extra={
            "example": {"refiner": 0.9, "critic": 0.8, "historian": 0.7}
        },
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    @field_validator("cost_per_agent", "quality_weights")
    @classmethod
    def validate_agent_mappings(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that agent mapping values are within valid ranges."""
        for agent_name, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Agent mapping value for '{agent_name}' must be numeric"
                )
            if value < 0.0:
                raise ValueError(
                    f"Agent mapping value for '{agent_name}' cannot be negative"
                )
            # Quality weights should be between 0 and 1
            if "quality" in str(cls) and value > 1.0:
                raise ValueError(f"Quality weight for '{agent_name}' cannot exceed 1.0")
        return v

    @field_validator("required_agents", "forbidden_agents")
    @classmethod
    def validate_agent_sets(cls, v: Set[str]) -> Set[str]:
        """Validate that agent names are non-empty strings."""
        if not isinstance(v, set):
            raise ValueError("Agent sets must be sets of strings")

        validated_agents = set()
        for agent in v:
            if not isinstance(agent, str):
                raise ValueError("Agent names must be strings")
            agent_clean = agent.strip()
            if not agent_clean:
                raise ValueError("Agent names cannot be empty")
            validated_agents.add(agent_clean)

        return validated_agents

    @model_validator(mode="after")
    def validate_constraint_consistency(self) -> "ResourceConstraints":
        """Validate that constraints are internally consistent."""
        # Check min/max agent constraints
        if self.min_agents is not None and self.max_agents is not None:
            if self.min_agents > self.max_agents:
                raise ValueError(
                    f"min_agents ({self.min_agents}) cannot exceed max_agents ({self.max_agents})"
                )

        # Check that required agents don't exceed max_agents
        if self.required_agents and self.max_agents is not None:
            if len(self.required_agents) > self.max_agents:
                raise ValueError(
                    f"Number of required agents ({len(self.required_agents)}) cannot exceed max_agents ({self.max_agents})"
                )

        # Check for conflicts between required and forbidden agents
        if self.required_agents and self.forbidden_agents:
            conflicts = self.required_agents.intersection(self.forbidden_agents)
            if conflicts:
                raise ValueError(
                    f"Agents cannot be both required and forbidden: {conflicts}"
                )

        # Check success/failure rate consistency
        if self.min_success_rate + self.max_failure_rate > 1.0:
            raise ValueError(
                f"min_success_rate ({self.min_success_rate}) + max_failure_rate ({self.max_failure_rate}) cannot exceed 1.0"
            )

        # Validate time constraints
        if (
            self.max_execution_time_ms is not None
            and self.max_agent_time_ms is not None
            and self.max_agent_time_ms > self.max_execution_time_ms
        ):
            raise ValueError("max_agent_time_ms cannot exceed max_execution_time_ms")

        return self

    def is_agent_allowed(self, agent: str, strict_required: bool = True) -> bool:
        """Check if an agent is allowed by constraints.

        Args:
            agent: The agent name to check
            strict_required: If True, check if required agents are present before allowing others.
                           If False, allow any agent that's not forbidden.
        """
        agent_lower = agent.lower()

        # Convert forbidden agents to lowercase for case-insensitive comparison
        forbidden_lower = {a.lower() for a in self.forbidden_agents}
        if agent_lower in forbidden_lower:
            return False

        # For normal operation, allow any agent that's not forbidden
        # strict_required is only used for fallback logic detection
        return True

    def validate_agent_count(self, count: int) -> bool:
        """Validate agent count against constraints."""
        if self.min_agents and count < self.min_agents:
            return False
        if self.max_agents and count > self.max_agents:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Maintained for backward compatibility. Uses Pydantic's model_dump()
        internally for consistent serialization with set handling.
        """
        data = self.model_dump(mode="json")

        # Convert sets to lists for JSON serialization compatibility
        if isinstance(data.get("required_agents"), list):
            data["required_agents"] = list(self.required_agents)
        if isinstance(data.get("forbidden_agents"), list):
            data["forbidden_agents"] = list(self.forbidden_agents)

        return data


class ResourceOptimizer:
    """
    Advanced resource optimizer for intelligent agent selection.

    This optimizer considers multiple factors including performance metrics,
    resource constraints, cost optimization, and quality requirements to
    make optimal agent selection decisions.
    """

    def __init__(self):
        """Initialize the resource optimizer."""
        self.logger = get_logger(f"{__name__}.ResourceOptimizer")

        # Agent capability matrix (can be extended)
        self.agent_capabilities = {
            "refiner": {
                "quality_weight": 0.9,
                "preprocessing": True,
                "critical": True,
                "parallel_safe": False,
            },
            "critic": {
                "quality_weight": 0.8,
                "analysis": True,
                "critical": False,
                "parallel_safe": True,
            },
            "historian": {
                "quality_weight": 0.7,
                "research": True,
                "critical": False,
                "parallel_safe": True,
            },
            "synthesis": {
                "quality_weight": 0.9,
                "synthesis": True,
                "critical": True,
                "parallel_safe": False,
            },
        }

        # Default optimization weights for different strategies
        self.strategy_weights = {
            OptimizationStrategy.PERFORMANCE: {
                "execution_time": 0.6,
                "success_rate": 0.3,
                "quality": 0.1,
            },
            OptimizationStrategy.RELIABILITY: {
                "success_rate": 0.7,
                "execution_time": 0.2,
                "quality": 0.1,
            },
            OptimizationStrategy.QUALITY: {
                "quality": 0.6,
                "success_rate": 0.3,
                "execution_time": 0.1,
            },
            OptimizationStrategy.BALANCED: {
                "execution_time": 0.3,
                "success_rate": 0.4,
                "quality": 0.3,
            },
            OptimizationStrategy.COST: {
                "cost": 0.6,
                "execution_time": 0.2,
                "success_rate": 0.2,
            },
            OptimizationStrategy.MINIMAL: {
                "agent_count": 0.5,
                "execution_time": 0.3,
                "success_rate": 0.2,
            },
        }

    def select_optimal_agents(
        self,
        available_agents: List[str],
        complexity_score: float,
        performance_data: Dict[str, Dict[str, Any]],
        constraints: Optional[ResourceConstraints] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        context_requirements: Optional[Dict[str, bool]] = None,
    ) -> RoutingDecision:
        """
        Select optimal agents based on multiple optimization criteria.

        Parameters
        ----------
        available_agents : List[str]
            Available agents to choose from
        complexity_score : float
            Query complexity score (0.0 to 1.0)
        performance_data : Dict[str, Dict[str, Any]]
            Historical performance data for agents
        constraints : Optional[ResourceConstraints]
            Resource constraints for optimization
        strategy : OptimizationStrategy
            Optimization strategy to use
        context_requirements : Optional[Dict[str, bool]]
            Context-specific requirements (research, criticism, etc.)

        Returns
        -------
        RoutingDecision
            Comprehensive routing decision with reasoning
        """
        start_time = time.time()
        constraints = constraints or ResourceConstraints()
        context_requirements = context_requirements or {}

        # Initialize decision
        decision = RoutingDecision(
            selected_agents=[],
            routing_strategy=strategy.value,
            confidence_score=0.0,
            confidence_level=RoutingConfidenceLevel.MEDIUM,
            available_agents=available_agents.copy(),
        )

        # Error handling: Check for empty available agents
        if not available_agents:
            decision.add_risk("empty_agents_list", "No agents available for selection")
            decision.confidence_score = 0.1
            decision.confidence_level = RoutingConfidenceLevel.VERY_LOW
            return decision

        # Error handling: Validate and normalize complexity score
        original_complexity = complexity_score
        if (
            complexity_score < 0
            or complexity_score > 1
            or not isinstance(complexity_score, (int, float))
        ):
            decision.add_risk(
                "invalid_complexity_score",
                f"Invalid complexity score: {complexity_score}",
            )
            complexity_score = max(
                0.0,
                min(
                    1.0,
                    (
                        float(complexity_score)
                        if isinstance(complexity_score, (int, float))
                        else 0.5
                    ),
                ),
            )
            if complexity_score != original_complexity:
                decision.add_risk(
                    "complexity_score_normalized",
                    f"Normalized from {original_complexity} to {complexity_score}",
                )

        # Error handling: Validate and sanitize performance data
        sanitized_performance_data = self._sanitize_performance_data(
            performance_data, available_agents, decision
        )

        # Error handling: Validate constraints
        if constraints:
            constraint_issues = self._validate_constraints(
                constraints, available_agents, decision
            )
            # Additional validation for success rate constraints
            self._validate_success_rate_constraints(
                constraints, sanitized_performance_data, decision
            )

        # Filter agents by constraints
        candidate_agents = self._filter_agents_by_constraints(
            available_agents,
            constraints,
            sanitized_performance_data,
            strict_required=False,
        )

        # Check if any required agents are missing and add fallback risk
        if constraints.required_agents:
            available_set = set(agent.lower() for agent in available_agents)
            required_set = set(agent.lower() for agent in constraints.required_agents)
            selected_set = set(agent.lower() for agent in candidate_agents)

            # Check if any required agents are missing from available agents
            missing_required = required_set - available_set
            if missing_required:
                decision.add_risk(
                    "required_agents_fallback",
                    "Using available agents as fallback for missing required agents",
                )

            # Check if any required agents are missing from selected candidates
            missing_from_selected = required_set - selected_set
            if missing_from_selected:
                # Some required agents didn't make it through filtering, add them back
                for agent in available_agents:
                    if (
                        agent.lower() in missing_from_selected
                        and agent not in candidate_agents
                    ):
                        candidate_agents.append(agent)
                        decision.add_risk(
                            "required_agent_override",
                            f"Including required agent {agent} despite constraints",
                        )

        if not candidate_agents:
            # Final fallback: use basic filtering without performance constraints
            candidate_agents = self._filter_agents_by_constraints(
                available_agents,
                constraints,
                performance_data=None,
                strict_required=False,
            )
            if candidate_agents:
                decision.add_risk(
                    "performance_fallback",
                    "Using basic filtering due to performance constraints",
                )

            if not candidate_agents:
                decision.add_reasoning(
                    "resource", "error", "No agents satisfy constraints"
                )
                decision.confidence_score = 0.0
                decision.confidence_level = RoutingConfidenceLevel.VERY_LOW
                return decision

        try:
            # Score all candidate agents
            agent_scores = self._score_agents(
                candidate_agents,
                complexity_score,
                sanitized_performance_data,
                strategy,
                context_requirements,
            )

            # Select optimal combination
            selected_agents, selection_confidence = self._select_agent_combination(
                agent_scores,
                complexity_score,
                constraints,
                strategy,
                context_requirements,
            )
        except Exception as e:
            # Strategy failure fallback
            decision.add_risk(
                "strategy_failure_fallback", f"Strategy execution failed: {e}"
            )
            selected_agents = (
                candidate_agents[:2] if len(candidate_agents) >= 2 else candidate_agents
            )
            selection_confidence = 0.3

        # Build comprehensive decision
        decision.selected_agents = selected_agents
        decision.confidence_score = selection_confidence

        # Apply confidence penalties for data quality issues
        risk_count = len(decision.reasoning.risks_identified)
        if risk_count > 0:
            # Reduce confidence based on number of risks - higher penalty for constraint violations
            base_penalty = risk_count * 0.05
            # Extra penalty for constraint violations
            constraint_risks = [
                r
                for r in decision.reasoning.risks_identified
                if any(
                    keyword in r
                    for keyword in [
                        "constraint",
                        "unavailable",
                        "success_rate_violation",
                    ]
                )
            ]
            constraint_penalty = len(constraint_risks) * 0.1
            total_penalty = min(0.5, base_penalty + constraint_penalty)
            decision.confidence_score = max(
                0.0, decision.confidence_score - total_penalty
            )

        decision.confidence_level = decision._calculate_confidence_level()

        # Add detailed reasoning
        try:
            self._build_reasoning(
                decision,
                agent_scores if "agent_scores" in locals() else {},
                complexity_score,
                sanitized_performance_data,
                constraints,
                strategy,
                context_requirements,
            )
        except Exception as e:
            decision.add_risk(
                "reasoning_failure", f"Failed to build detailed reasoning: {e}"
            )

        # Set execution metadata
        self._set_execution_metadata(decision)

        # Calculate performance predictions
        try:
            self._calculate_predictions(decision, sanitized_performance_data)
        except Exception as e:
            decision.add_risk(
                "prediction_failure", f"Failed to calculate predictions: {e}"
            )

        execution_time = (time.time() - start_time) * 1000
        self.logger.debug(
            f"Resource optimization completed in {execution_time:.1f}ms, "
            f"selected {len(selected_agents)} agents with confidence {selection_confidence:.2f}"
        )

        return decision

    def _filter_agents_by_constraints(
        self,
        agents: List[str],
        constraints: ResourceConstraints,
        performance_data: Optional[Dict[str, Dict[str, Any]]] = None,
        strict_required: bool = True,
    ) -> List[str]:
        """Filter agents based on resource constraints."""
        filtered = []

        for agent in agents:
            if not constraints.is_agent_allowed(agent, strict_required=strict_required):
                continue

            # Check success rate constraints (except for required agents)
            if performance_data and agent.lower() not in {
                req.lower() for req in constraints.required_agents
            }:
                agent_data = performance_data.get(agent.lower(), {})
                success_rate = agent_data.get("success_rate", 0.8)

                # Filter out agents that don't meet success rate constraints
                if success_rate < constraints.min_success_rate:
                    continue

                # Filter out agents that exceed failure rate constraints
                if (1 - success_rate) > constraints.max_failure_rate:
                    continue

            filtered.append(agent)

        return filtered

    def _score_agents(
        self,
        agents: List[str],
        complexity_score: float,
        performance_data: Dict[str, Dict[str, Any]],
        strategy: OptimizationStrategy,
        context_requirements: Dict[str, bool],
    ) -> Dict[str, float]:
        """Score agents based on optimization strategy."""
        scores = {}
        weights = self.strategy_weights.get(
            strategy, self.strategy_weights[OptimizationStrategy.BALANCED]
        )

        for agent in agents:
            agent_lower = agent.lower()
            perf_data = performance_data.get(agent_lower, {})

            # Base performance metrics
            success_rate = perf_data.get("success_rate", 0.8)  # Default assumption
            avg_time_ms = perf_data.get("average_time_ms", 2000.0)  # Default assumption

            # Calculate component scores
            time_score = self._calculate_time_score(avg_time_ms)
            reliability_score = success_rate
            quality_score = self._calculate_quality_score(agent_lower, complexity_score)
            cost_score = self._calculate_cost_score(agent_lower)
            context_score = self._calculate_context_score(
                agent_lower, context_requirements
            )

            # Weighted combination based on strategy
            final_score = (
                weights.get("execution_time", 0.0) * time_score
                + weights.get("success_rate", 0.0) * reliability_score
                + weights.get("quality", 0.0) * quality_score
                + weights.get("cost", 0.0) * cost_score
                + 0.1 * context_score  # Context relevance bonus
            )

            # Normalize to 0-1 range
            scores[agent] = min(1.0, max(0.0, final_score))

        return scores

    def _calculate_time_score(self, avg_time_ms: float) -> float:
        """Calculate time performance score (higher is better)."""
        # Assume 5000ms is poor (score 0.0), 500ms is excellent (score 1.0)
        if avg_time_ms <= 500:
            return 1.0
        elif avg_time_ms >= 5000:
            return 0.0
        else:
            return (5000 - avg_time_ms) / 4500

    def _calculate_quality_score(self, agent: str, complexity_score: float) -> float:
        """Calculate quality score based on agent capabilities and query complexity."""
        capabilities = self.agent_capabilities.get(agent, {})
        base_quality = capabilities.get("quality_weight", 0.5)

        # Adjust for complexity - some agents perform better on complex queries
        if agent in ["critic", "synthesis"] and complexity_score > 0.7:
            return min(1.0, base_quality + 0.1)
        elif agent == "historian" and complexity_score > 0.5:
            return min(1.0, base_quality + 0.1)

        return base_quality

    def _calculate_cost_score(self, agent: str) -> float:
        """Calculate cost score (placeholder for future cost optimization)."""
        # Placeholder - could be extended with actual cost data
        return 0.5

    def _calculate_context_score(
        self, agent: str, requirements: Dict[str, bool]
    ) -> float:
        """Calculate context relevance score."""
        score = 0.0

        if requirements.get("requires_research", False) and agent == "historian":
            score += 0.3
        if requirements.get("requires_criticism", False) and agent == "critic":
            score += 0.3
        if requirements.get("requires_synthesis", False) and agent == "synthesis":
            score += 0.3
        if requirements.get("requires_refinement", False) and agent == "refiner":
            score += 0.3

        return score

    def _select_agent_combination(
        self,
        agent_scores: Dict[str, float],
        complexity_score: float,
        constraints: ResourceConstraints,
        strategy: OptimizationStrategy,
        context_requirements: Dict[str, bool],
    ) -> Tuple[List[str], float]:
        """Select optimal combination of agents."""
        if not agent_scores:
            return [], 0.0

        # Sort agents by score
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        selected = []
        confidence_factors = []

        # Always include required agents first
        for agent, score in sorted_agents:
            agent_lower = agent.lower()
            if agent_lower in constraints.required_agents:
                selected.append(agent)
                confidence_factors.append(score)

        # Strategy-specific selection logic
        if strategy == OptimizationStrategy.MINIMAL:
            # Select minimal high-performing agents
            target_count = max(1, constraints.min_agents or 1)
            while len(selected) < target_count and len(selected) < len(sorted_agents):
                for agent, score in sorted_agents:
                    if agent not in selected and score > 0.6:
                        selected.append(agent)
                        confidence_factors.append(score)
                        break
                else:
                    break

        elif strategy in [
            OptimizationStrategy.PERFORMANCE,
            OptimizationStrategy.RELIABILITY,
        ]:
            # Select top performing agents up to constraints
            max_agents = constraints.max_agents or 4
            for agent, score in sorted_agents:
                if agent not in selected and len(selected) < max_agents:
                    if score > 0.5:  # Only select decent performers
                        selected.append(agent)
                        confidence_factors.append(score)

        elif strategy == OptimizationStrategy.QUALITY:
            # Select agents that contribute to quality, considering complexity
            quality_agents = ["refiner", "synthesis"]
            if complexity_score > 0.6:
                quality_agents.extend(["critic", "historian"])

            for agent, score in sorted_agents:
                agent_lower = agent.lower()
                if agent not in selected and (
                    agent_lower in quality_agents or score > 0.7
                ):
                    selected.append(agent)
                    confidence_factors.append(score)

        else:  # BALANCED or other strategies
            # Balanced selection based on scores and constraints
            max_agents = constraints.max_agents or 4
            min_score = 0.4 if complexity_score > 0.7 else 0.3

            for agent, score in sorted_agents:
                if (
                    agent not in selected
                    and len(selected) < max_agents
                    and score > min_score
                ):
                    selected.append(agent)
                    confidence_factors.append(score)

        # Ensure minimum agent count
        min_agents = constraints.min_agents or 1
        while len(selected) < min_agents and len(selected) < len(sorted_agents):
            for agent, score in sorted_agents:
                if agent not in selected:
                    selected.append(agent)
                    confidence_factors.append(score)
                    break

        # Calculate overall confidence
        if confidence_factors:
            avg_confidence = sum(confidence_factors) / len(confidence_factors)
            # Adjust confidence based on selection completeness
            completeness = len(selected) / max(1, len(agent_scores))
            final_confidence = avg_confidence * (0.7 + 0.3 * completeness)
        else:
            final_confidence = 0.0

        return selected, final_confidence

    def _build_reasoning(
        self,
        decision: RoutingDecision,
        agent_scores: Dict[str, float],
        complexity_score: float,
        performance_data: Dict[str, Dict[str, Any]],
        constraints: ResourceConstraints,
        strategy: OptimizationStrategy,
        context_requirements: Dict[str, bool],
    ) -> None:
        """Build comprehensive reasoning for the decision."""

        # Strategy rationale
        decision.reasoning.strategy_rationale = f"Selected {strategy.value} optimization strategy for complexity score {complexity_score:.2f}"

        # Agent selection rationale
        for agent in decision.selected_agents:
            score = agent_scores.get(agent, 0.0)
            perf_data = performance_data.get(agent.lower(), {})

            rationale_parts = []
            rationale_parts.append(f"Score: {score:.2f}")

            if perf_data.get("success_rate"):
                rationale_parts.append(f"Success rate: {perf_data['success_rate']:.1%}")
            if perf_data.get("average_time_ms"):
                rationale_parts.append(
                    f"Avg time: {perf_data['average_time_ms']:.0f}ms"
                )

            # Context-specific rationale
            agent_lower = agent.lower()
            if (
                context_requirements.get("requires_research")
                and agent_lower == "historian"
            ):
                rationale_parts.append("Research capability required")
            if (
                context_requirements.get("requires_criticism")
                and agent_lower == "critic"
            ):
                rationale_parts.append("Critical analysis required")

            decision.add_agent_rationale(agent, "; ".join(rationale_parts))

        # Excluded agents rationale
        excluded = decision.get_excluded_agents()
        for agent in excluded:
            score = agent_scores.get(agent, 0.0)

            if score < 0.3:
                decision.add_agent_rationale(
                    agent, f"Low performance score: {score:.2f}", included=False
                )
            elif agent.lower() in constraints.forbidden_agents:
                decision.add_agent_rationale(
                    agent, "Forbidden by constraints", included=False
                )
            else:
                decision.add_agent_rationale(
                    agent,
                    f"Score {score:.2f} below selection threshold",
                    included=False,
                )

        # Risk assessment
        if decision.confidence_score < 0.5:
            decision.add_risk(
                "Low confidence in agent selection",
                "Monitor execution closely and consider fallback options",
            )

        if complexity_score > 0.8 and len(decision.selected_agents) < 3:
            decision.add_risk(
                "High complexity query with limited agent coverage",
                "Consider adding more specialized agents",
            )

        # Fallback options
        if excluded:
            top_excluded = sorted(
                [(agent, agent_scores.get(agent, 0.0)) for agent in excluded],
                key=lambda x: x[1],
                reverse=True,
            )[:2]

            for agent, score in top_excluded:
                if score > 0.4:
                    decision.add_fallback_option(f"{agent} (score: {score:.2f})")

        # Analysis data
        decision.add_reasoning("complexity", "score", complexity_score)
        decision.add_reasoning(
            "complexity", "level", self._get_complexity_level(complexity_score)
        )
        decision.add_reasoning("performance", "agent_scores", agent_scores)
        decision.add_reasoning("resource", "constraints_applied", True)
        decision.add_reasoning("resource", "strategy", strategy.value)

    def _get_complexity_level(self, score: float) -> str:
        """Get complexity level description."""
        if score <= 0.3:
            return "simple"
        elif score <= 0.6:
            return "moderate"
        elif score <= 0.8:
            return "complex"
        else:
            return "very_complex"

    def _set_execution_metadata(self, decision: RoutingDecision) -> None:
        """Set execution metadata for the decision."""
        agents = decision.selected_agents
        agents_lower = [agent.lower() for agent in agents]

        # Determine entry point
        if "refiner" in agents_lower:
            decision.entry_point = "refiner"
        elif agents:
            decision.entry_point = agents[0]

        # Determine exit points
        if "synthesis" in agents_lower:
            decision.exit_points = ["synthesis"]
        else:
            # Last agents are exit points
            decision.exit_points = agents[-1:] if agents else []

        # Determine parallel groups
        parallel_candidates = ["critic", "historian"]
        parallel_group = [
            agent for agent in agents_lower if agent in parallel_candidates
        ]

        if len(parallel_group) > 1:
            decision.parallel_groups = [parallel_group]

    def _calculate_predictions(
        self, decision: RoutingDecision, performance_data: Dict[str, Dict[str, Any]]
    ) -> None:
        """Calculate performance predictions for the decision."""
        if not decision.selected_agents:
            return

        # Estimate total execution time
        total_time = 0.0
        success_probs = []

        for agent in decision.selected_agents:
            agent_data = performance_data.get(agent.lower(), {})
            agent_time = agent_data.get("average_time_ms", 2000.0)
            agent_success = agent_data.get("success_rate", 0.8)

            total_time += agent_time
            success_probs.append(agent_success)

        # Adjust for parallel execution
        if decision.parallel_groups:
            # Estimate parallel execution savings
            for group in decision.parallel_groups:
                if len(group) > 1:
                    # Assume parallel execution saves time
                    group_times = [
                        performance_data.get(agent.lower(), {}).get(
                            "average_time_ms", 2000.0
                        )
                        for agent in group
                    ]
                    sequential_time = sum(group_times)
                    parallel_time = max(group_times)
                    time_savings = sequential_time - parallel_time
                    total_time -= time_savings

        # Calculate overall success probability (multiplicative for now)
        if success_probs:
            overall_success = 1.0
            for prob in success_probs:
                overall_success *= prob
            # Apply optimism factor for good agent combinations
            if len(success_probs) > 1:
                overall_success = min(1.0, overall_success * 1.1)
        else:
            overall_success = 0.5

        decision.set_performance_prediction(total_time, overall_success)

        # Identify optimization opportunities
        if total_time > 8000:  # More than 8 seconds
            decision.add_optimization_opportunity(
                "Consider parallel execution or faster agents"
            )

        if overall_success < 0.6:
            decision.add_optimization_opportunity(
                "Consider more reliable agents or fallback strategies"
            )

        if len(decision.selected_agents) > 3:
            decision.add_optimization_opportunity(
                "Evaluate if all agents are necessary"
            )

    def _sanitize_performance_data(
        self,
        performance_data: Dict[str, Dict[str, Any]],
        available_agents: List[str],
        decision: RoutingDecision,
    ) -> Dict[str, Dict[str, Any]]:
        """Sanitize and validate performance data."""
        sanitized = {}

        for agent in available_agents:
            agent_lower = agent.lower()
            agent_data = performance_data.get(agent_lower, {})

            # Default values for missing data
            if not agent_data:
                decision.add_risk(
                    "missing_performance_data", f"No performance data for {agent}"
                )
                sanitized[agent_lower] = {
                    "success_rate": 0.7,
                    "average_time_ms": 2000.0,
                    "performance_score": 0.5,
                }
                continue

            # Sanitize success rate
            success_rate = agent_data.get("success_rate", 0.7)
            if (
                not isinstance(success_rate, (int, float))
                or success_rate < 0
                or success_rate > 1
            ):
                decision.add_risk(
                    "invalid_performance_data",
                    f"Invalid success rate for {agent}: {success_rate}",
                )
                success_rate = max(0.0, min(1.0, 0.7))

            # Sanitize average time
            avg_time = agent_data.get("average_time_ms", 2000.0)
            if not isinstance(avg_time, (int, float)) or avg_time < 0:
                decision.add_risk(
                    "invalid_performance_data",
                    f"Invalid average time for {agent}: {avg_time}",
                )
                avg_time = 2000.0
            elif avg_time == 0.0:
                decision.add_risk(
                    "time_estimation_issues",
                    f"Zero time for {agent}, using minimum estimate",
                )
                avg_time = 100.0  # Minimum reasonable time
            elif avg_time == float("inf"):
                decision.add_risk(
                    "time_estimation_issues", f"Infinite time for {agent}"
                )
                avg_time = 10000.0  # Cap at 10 seconds

            # Sanitize performance score
            perf_score = agent_data.get("performance_score", 0.5)
            if (
                not isinstance(perf_score, (int, float))
                or perf_score < 0
                or perf_score > 1
            ):
                decision.add_risk(
                    "invalid_performance_data",
                    f"Invalid performance score for {agent}: {perf_score}",
                )
                perf_score = max(0.0, min(1.0, 0.5))

            sanitized[agent_lower] = {
                "success_rate": success_rate,
                "average_time_ms": avg_time,
                "performance_score": perf_score,
            }

        return sanitized

    def _validate_constraints(
        self,
        constraints: ResourceConstraints,
        available_agents: List[str],
        decision: RoutingDecision,
    ) -> bool:
        """Validate constraints and add risk information."""
        issues = False

        # Check impossible constraints
        if constraints.min_agents and constraints.max_agents:
            if constraints.min_agents > constraints.max_agents:
                decision.add_risk(
                    "impossible_constraints",
                    f"min_agents ({constraints.min_agents}) > max_agents ({constraints.max_agents})",
                )
                issues = True

        # Check if required agents exceed max_agents
        if constraints.required_agents and constraints.max_agents:
            if len(constraints.required_agents) > constraints.max_agents:
                decision.add_risk(
                    "impossible_constraints",
                    f"Required agents ({len(constraints.required_agents)}) exceed max_agents ({constraints.max_agents})",
                )
                issues = True

        # Check if all agents are forbidden
        if constraints.forbidden_agents:
            available_set = set(agent.lower() for agent in available_agents)
            forbidden_set = set(agent.lower() for agent in constraints.forbidden_agents)
            if available_set.issubset(forbidden_set):
                decision.add_risk(
                    "all_agents_forbidden", "All available agents are forbidden"
                )
                issues = True

        # Check required agents availability
        if constraints.required_agents:
            available_set = set(agent.lower() for agent in available_agents)
            required_set = set(agent.lower() for agent in constraints.required_agents)
            missing_required = required_set - available_set
            if missing_required:
                decision.add_risk(
                    "required_agents_unavailable",
                    f"Required agents not available: {missing_required}",
                )
                issues = True

        # Check for conflicts between required and forbidden agents
        if constraints.required_agents and constraints.forbidden_agents:
            required_set = set(agent.lower() for agent in constraints.required_agents)
            forbidden_set = set(agent.lower() for agent in constraints.forbidden_agents)
            conflicts = required_set.intersection(forbidden_set)
            if conflicts:
                decision.add_risk(
                    "constraint_conflict",
                    f"Agents both required and forbidden: {conflicts}",
                )
                issues = True

        # Check if forbidden agents would prevent meeting min_agents
        if constraints.forbidden_agents and constraints.min_agents:
            available_set = set(agent.lower() for agent in available_agents)
            forbidden_set = set(agent.lower() for agent in constraints.forbidden_agents)
            remaining_agents = available_set - forbidden_set
            if len(remaining_agents) < constraints.min_agents:
                decision.add_risk(
                    "impossible_constraints",
                    f"Forbidden agents leave only {len(remaining_agents)} agents, but min_agents is {constraints.min_agents}",
                )
                issues = True

        return issues

    def _validate_success_rate_constraints(
        self,
        constraints: ResourceConstraints,
        performance_data: Dict[str, Dict[str, Any]],
        decision: RoutingDecision,
    ) -> None:
        """Validate success rate constraints for required agents."""
        if not constraints.required_agents:
            return

        for agent in constraints.required_agents:
            agent_lower = agent.lower()
            agent_data = performance_data.get(agent_lower, {})
            success_rate = agent_data.get("success_rate", 0.8)

            # Check if required agent violates minimum success rate
            if (
                constraints.min_success_rate
                and success_rate < constraints.min_success_rate
            ):
                decision.add_risk(
                    "success_rate_violation",
                    f"Required agent {agent} has success rate {success_rate:.2f} below minimum {constraints.min_success_rate:.2f}",
                )

            # Check if required agent violates maximum failure rate
            if (
                constraints.max_failure_rate
                and (1 - success_rate) > constraints.max_failure_rate
            ):
                decision.add_risk(
                    "failure_rate_violation",
                    f"Required agent {agent} has failure rate {1 - success_rate:.2f} above maximum {constraints.max_failure_rate:.2f}",
                )
