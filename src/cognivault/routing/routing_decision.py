"""
Routing Decision Data Structures.

This module defines the data structures for routing decisions,
providing comprehensive reasoning and confidence tracking.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class ConfidenceLevel(Enum):
    """Confidence levels for routing decisions."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


class RoutingReasoning(BaseModel):
    """
    Detailed reasoning for routing decisions.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    # Primary reasoning factors
    complexity_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis of query/task complexity factors",
        json_schema_extra={
            "example": {
                "complexity_score": 0.7,
                "factors": ["multi_step", "reasoning_required"],
            }
        },
    )
    performance_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance analysis and optimization considerations",
        json_schema_extra={
            "example": {"latency_estimate_ms": 2500, "throughput_score": 0.8}
        },
    )
    resource_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource utilization and constraint analysis",
        json_schema_extra={
            "example": {"memory_requirement": "medium", "compute_intensity": 0.6}
        },
    )

    # Decision factors
    strategy_rationale: str = Field(
        default="",
        description="Primary rationale for the chosen routing strategy",
        max_length=1000,
        json_schema_extra={
            "example": "Selected parallel execution to optimize response time"
        },
    )
    agent_selection_rationale: Dict[str, str] = Field(
        default_factory=dict,
        description="Rationale for why specific agents were selected",
        json_schema_extra={
            "example": {
                "refiner": "Query needs clarification",
                "historian": "Context retrieval required",
            }
        },
    )
    excluded_agents_rationale: Dict[str, str] = Field(
        default_factory=dict,
        description="Rationale for why specific agents were excluded",
        json_schema_extra={
            "example": {"critic": "Simple query doesn't require critical analysis"}
        },
    )

    # Risk assessment
    risks_identified: List[str] = Field(
        default_factory=list,
        description="List of identified risks and potential failure points",
        json_schema_extra={
            "example": ["High latency possible", "Resource contention risk"]
        },
    )
    mitigation_strategies: List[str] = Field(
        default_factory=list,
        description="Strategies to mitigate identified risks",
        json_schema_extra={
            "example": ["Implement timeout fallback", "Use resource throttling"]
        },
    )
    fallback_options: List[str] = Field(
        default_factory=list,
        description="Alternative approaches if primary strategy fails",
        json_schema_extra={
            "example": ["Fall back to sequential execution", "Use cached response"]
        },
    )

    # Performance predictions
    estimated_execution_time_ms: Optional[float] = Field(
        default=None,
        description="Estimated execution time in milliseconds",
        ge=0,
        le=300000,  # 5 minutes max
        json_schema_extra={"example": 2500.0},
    )
    estimated_success_probability: Optional[float] = Field(
        default=None,
        description="Estimated probability of successful execution (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    resource_utilization_estimate: Dict[str, float] = Field(
        default_factory=dict,
        description="Estimated resource utilization by category",
        json_schema_extra={"example": {"cpu": 0.6, "memory": 0.4, "network": 0.2}},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Maintained for backward compatibility. Uses Pydantic's model_dump()
        internally for consistent serialization.
        """
        return self.model_dump()


class RoutingDecision(BaseModel):
    """
    Comprehensive routing decision with reasoning and metadata.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.

    This class captures all aspects of an intelligent routing decision,
    including the selected agents, reasoning, confidence, and predictions.
    """

    # Core decision - required fields first
    selected_agents: List[str] = Field(
        ...,
        description="List of selected agent names for execution",
        min_length=0,
        json_schema_extra={"example": ["refiner", "critic", "historian"]},
    )
    routing_strategy: str = Field(
        ...,
        description="Strategy used for agent selection and routing",
        min_length=1,
        max_length=200,
        json_schema_extra={"example": "capability_based"},
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score for the routing decision (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Categorical confidence level derived from numeric score",
        json_schema_extra={"example": "high"},
    )

    # Decision context - optional fields with defaults
    decision_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this routing decision",
        min_length=32,
        max_length=32,
        json_schema_extra={"example": "a1b2c3d4e5f6789012345678901234ab"},
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this routing decision was made (UTC)",
        json_schema_extra={"example": "2024-01-01T12:00:00Z"},
    )
    query_hash: Optional[str] = Field(
        None,
        description="Hash of the query that triggered this routing decision",
        max_length=128,
        json_schema_extra={"example": "sha256:abc123..."},
    )
    available_agents: List[str] = Field(
        default_factory=list,
        description="List of agent names that were available for selection",
        json_schema_extra={"example": ["refiner", "critic", "historian", "synthesis"]},
    )

    # Reasoning and analysis
    reasoning: RoutingReasoning = Field(
        default_factory=RoutingReasoning,
        description="Comprehensive reasoning and analysis for this decision",
    )

    # Execution metadata
    execution_order: List[str] = Field(
        default_factory=list,
        description="Ordered list of agents for execution",
        json_schema_extra={"example": ["refiner", "historian", "critic", "synthesis"]},
    )
    parallel_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups of agents that can be executed in parallel",
        json_schema_extra={"example": [["critic", "historian"]]},
    )
    entry_point: Optional[str] = Field(
        None,
        description="The first agent in the execution flow",
        max_length=100,
        json_schema_extra={"example": "refiner"},
    )
    exit_points: List[str] = Field(
        default_factory=list,
        description="List of agents that conclude the execution flow",
        json_schema_extra={"example": ["synthesis"]},
    )

    # Performance predictions
    estimated_total_time_ms: Optional[float] = Field(
        None,
        description="Estimated total execution time in milliseconds",
        ge=0.0,
        le=600000.0,  # 10 minutes max
        json_schema_extra={"example": 5500.0},
    )
    estimated_success_probability: Optional[float] = Field(
        None,
        description="Estimated probability of successful execution (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92},
    )
    optimization_opportunities: List[str] = Field(
        default_factory=list,
        description="List of identified optimization opportunities",
        json_schema_extra={
            "example": ["Consider parallel execution", "Cache intermediate results"]
        },
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,  # Keep enum objects, not string values
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization to set derived fields."""
        # Calculate confidence level if not explicitly set
        if not self.confidence_level:
            self.confidence_level = self._calculate_confidence_level()

        # Set execution order to selected agents if not specified
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
        """
        Convert to dictionary for serialization.

        Maintained for backward compatibility. Uses Pydantic's model_dump()
        internally for consistent serialization with datetime handling.
        """
        # Use model_dump with mode='json' to properly serialize datetime and enums
        data = self.model_dump(mode="json")

        # Ensure datetime is serialized as ISO format string for compatibility
        data["timestamp"] = self.timestamp.isoformat()

        # Ensure confidence_level is serialized as string value
        data["confidence_level"] = self.confidence_level.value

        # Ensure reasoning uses the to_dict method for consistency
        data["reasoning"] = self.reasoning.to_dict() if self.reasoning else {}

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """
        Create routing decision from dictionary.

        Uses Pydantic's model_validate() for cleaner deserialization
        and automatic type conversion.
        """
        # Handle nested reasoning object if it's a dict
        if "reasoning" in data and isinstance(data["reasoning"], dict):
            data = data.copy()  # Don't modify original
            data["reasoning"] = RoutingReasoning.model_validate(data["reasoning"])

        # Use Pydantic's model_validate for automatic type conversion
        return cls.model_validate(data)
