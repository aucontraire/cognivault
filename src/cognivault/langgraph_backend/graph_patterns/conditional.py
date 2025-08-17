"""
Enhanced conditional pattern implementation with dynamic routing.

This module implements intelligent routing patterns that adapt based on:
- Query complexity analysis
- Agent performance metrics
- Context requirements
- Failure scenarios and fallbacks
"""

import re
import time
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .base import GraphPattern
from ..semantic_validation import (
    WorkflowSemanticValidator,
    SemanticValidationResult,
    ValidationSeverity,
)


class ContextComplexity(Enum):
    """Context complexity levels for routing decisions."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class RoutingStrategy(Enum):
    """Available routing strategies."""

    STREAMLINED = "streamlined"  # Minimal agents for simple queries
    STANDARD = "standard"  # Default CogniVault workflow
    COMPREHENSIVE = "comprehensive"  # All agents for complex analysis
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Based on performance metrics


@dataclass
class RoutingConfig:
    """Configuration for conditional routing decisions."""

    # Complexity thresholds
    simple_threshold: float = 0.3
    moderate_threshold: float = 0.6
    complex_threshold: float = 0.8

    # Performance settings
    performance_weight: float = 0.4
    reliability_weight: float = 0.6

    # Fallback settings
    enable_fallbacks: bool = True
    max_fallback_depth: int = 2

    # Agent preferences
    preferred_agents: List[str] = field(
        default_factory=lambda: ["refiner", "synthesis"]
    )
    optional_agents: List[str] = field(default_factory=lambda: ["critic", "historian"])

    # Performance tracking
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    failure_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ContextAnalysis:
    """Analysis results for query context."""

    complexity_score: float
    complexity_level: ContextComplexity
    word_count: int
    question_count: int
    technical_terms: int
    domain_indicators: List[str]
    requires_research: bool
    requires_criticism: bool
    routing_strategy: RoutingStrategy


class ContextAnalyzer:
    """Analyzes query context to determine complexity and routing needs."""

    def __init__(self) -> None:
        """Initialize the context analyzer with domain knowledge."""
        self.technical_patterns = [
            r"\b(?:algorithm|implementation|architecture|framework|methodology)\b",
            r"\b(?:analysis|evaluation|assessment|comparison|optimization)\b",
            r"\b(?:strategy|approach|solution|technique|process)\b",
            r"\b(?:research|study|investigation|exploration|examination)\b",
        ]

        self.complexity_indicators = [
            r"\b(?:comprehensive|detailed|thorough|in-depth|extensive)\b",
            r"\b(?:complex|complicated|sophisticated|advanced|nuanced)\b",
            r"\b(?:multiple|various|several|numerous|different)\b",
            r"\b(?:interdisciplinary|multifaceted|holistic|systematic)\b",
        ]

        self.research_indicators = [
            r"\b(?:history|historical|background|context|origin)\b",
            r"\b(?:evidence|data|statistics|facts|information)\b",
            r"\b(?:sources|references|citations|literature|studies)\b",
        ]

        self.criticism_indicators = [
            r"\b(?:critique|criticism|evaluate|assess|analyze)\b",
            r"\b(?:pros and cons|advantages|disadvantages|limitations)\b",
            r"\b(?:strengths|weaknesses|flaws|problems|issues)\b",
            r"\b(?:validity|reliability|accuracy|effectiveness)\b",
        ]

    def analyze_context(self, query: str) -> ContextAnalysis:
        """
        Analyze query context to determine routing strategy.

        Parameters
        ----------
        query : str
            The user query to analyze

        Returns
        -------
        ContextAnalysis
            Detailed analysis of the query context
        """
        # Basic metrics
        word_count = len(query.split())
        question_count = query.count("?")

        # Technical term detection
        technical_score = sum(
            len(re.findall(pattern, query, re.IGNORECASE))
            for pattern in self.technical_patterns
        )

        # Complexity indicators
        complexity_score = sum(
            len(re.findall(pattern, query, re.IGNORECASE))
            for pattern in self.complexity_indicators
        )

        # Domain analysis
        research_needed = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.research_indicators
        )

        criticism_needed = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.criticism_indicators
        )

        # Calculate overall complexity score (0.0 to 1.0)
        base_score = min(word_count / 100.0, 1.0)  # Word count component
        technical_component = min(technical_score / 5.0, 0.3)  # Technical terms
        complexity_component = min(complexity_score / 3.0, 0.3)  # Complexity indicators
        question_component = min(question_count / 3.0, 0.2)  # Multiple questions

        overall_complexity = min(
            base_score
            + technical_component
            + complexity_component
            + question_component,
            1.0,
        )

        # Determine complexity level
        if overall_complexity <= 0.3:
            complexity_level = ContextComplexity.SIMPLE
        elif overall_complexity <= 0.6:
            complexity_level = ContextComplexity.MODERATE
        elif overall_complexity <= 0.8:
            complexity_level = ContextComplexity.COMPLEX
        else:
            complexity_level = ContextComplexity.VERY_COMPLEX

        # Determine routing strategy
        if complexity_level == ContextComplexity.SIMPLE and not (
            research_needed or criticism_needed
        ):
            strategy = RoutingStrategy.STREAMLINED
        elif complexity_level in [
            ContextComplexity.COMPLEX,
            ContextComplexity.VERY_COMPLEX,
        ]:
            strategy = RoutingStrategy.COMPREHENSIVE
        else:
            strategy = RoutingStrategy.STANDARD

        return ContextAnalysis(
            complexity_score=overall_complexity,
            complexity_level=complexity_level,
            word_count=word_count,
            question_count=question_count,
            technical_terms=technical_score,
            domain_indicators=[],  # Could be enhanced with specific domain detection
            requires_research=research_needed,
            requires_criticism=criticism_needed,
            routing_strategy=strategy,
        )


class PerformanceTracker:
    """Tracks agent performance metrics for routing decisions."""

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self.execution_times: Dict[str, List[float]] = {}
        self.success_rates: Dict[str, List[bool]] = {}
        self.last_updated: Dict[str, float] = {}

    def record_execution(self, agent: str, duration_ms: float, success: bool) -> None:
        """
        Record agent execution metrics.

        Parameters
        ----------
        agent : str
            Agent name
        duration_ms : float
            Execution duration in milliseconds
        success : bool
            Whether execution was successful
        """
        if agent not in self.execution_times:
            self.execution_times[agent] = []
            self.success_rates[agent] = []

        self.execution_times[agent].append(duration_ms)
        self.success_rates[agent].append(success)
        self.last_updated[agent] = time.time()

        # Keep only recent history (last 50 executions)
        if len(self.execution_times[agent]) > 50:
            self.execution_times[agent] = self.execution_times[agent][-50:]
            self.success_rates[agent] = self.success_rates[agent][-50:]

    def get_average_time(self, agent: str) -> Optional[float]:
        """Get average execution time for agent."""
        if agent not in self.execution_times or not self.execution_times[agent]:
            return None
        return sum(self.execution_times[agent]) / len(self.execution_times[agent])

    def get_success_rate(self, agent: str) -> Optional[float]:
        """Get success rate for agent."""
        if agent not in self.success_rates or not self.success_rates[agent]:
            return None
        return sum(self.success_rates[agent]) / len(self.success_rates[agent])

    def get_performance_score(self, agent: str) -> float:
        """
        Get overall performance score combining speed and reliability.

        Returns
        -------
        float
            Performance score (0.0 to 1.0, higher is better)
        """
        avg_time = self.get_average_time(agent)
        success_rate = self.get_success_rate(agent)

        if avg_time is None or success_rate is None:
            return 0.5  # Default score for unknown agents

        # Normalize execution time (assume 5000ms is poor, 500ms is excellent)
        time_score = max(0.0, min(1.0, (5000 - avg_time) / 4500))

        # Combine time and success rate
        return 0.4 * time_score + 0.6 * success_rate


@dataclass
class FallbackRule:
    """Defines fallback behavior for agent failures."""

    failed_agent: str
    fallback_agents: List[str]
    condition: str  # "always", "performance_poor", "timeout", "error"
    priority: int = 1  # Lower number = higher priority
    max_attempts: int = 2


class FallbackManager:
    """Manages agent failure fallbacks and recovery strategies."""

    def __init__(self) -> None:
        """Initialize fallback manager with default rules."""
        self.fallback_rules: List[FallbackRule] = []
        self.failure_history: Dict[str, List[float]] = {}  # agent -> [timestamp, ...]
        self.fallback_attempts: Dict[str, int] = {}  # agent -> attempt count

        # Register default fallback rules
        self._register_default_fallbacks()

    def _register_default_fallbacks(self) -> None:
        """Register default fallback rules for CogniVault agents."""
        # Critic fallbacks - if critic fails, can try synthesis directly
        self.fallback_rules.append(
            FallbackRule(
                failed_agent="critic",
                fallback_agents=["synthesis"],
                condition="always",
                priority=1,
            )
        )

        # Historian fallbacks - if historian fails, can continue without it
        self.fallback_rules.append(
            FallbackRule(
                failed_agent="historian",
                fallback_agents=[],  # No replacement, just skip
                condition="always",
                priority=1,
            )
        )

        # Refiner fallbacks - critical agent, try limited fallback
        self.fallback_rules.append(
            FallbackRule(
                failed_agent="refiner",
                fallback_agents=[
                    "synthesis"
                ],  # Skip refinement, go straight to synthesis
                condition="error",
                priority=2,
            )
        )

        # Synthesis fallbacks - if synthesis fails, try critic for final output
        self.fallback_rules.append(
            FallbackRule(
                failed_agent="synthesis",
                fallback_agents=["critic"],
                condition="error",
                priority=2,
            )
        )

    def register_fallback(self, rule: FallbackRule) -> None:
        """Register a custom fallback rule."""
        self.fallback_rules.append(rule)
        # Sort by priority (lower number = higher priority)
        self.fallback_rules.sort(key=lambda r: r.priority)

    def get_fallback_agents(
        self, failed_agent: str, failure_type: str = "error"
    ) -> List[str]:
        """
        Get fallback agents for a failed agent.

        Parameters
        ----------
        failed_agent : str
            Name of the failed agent
        failure_type : str
            Type of failure ("error", "timeout", "performance_poor")

        Returns
        -------
        List[str]
            List of fallback agents to try
        """
        # Record failure
        current_time = time.time()
        if failed_agent not in self.failure_history:
            self.failure_history[failed_agent] = []
        self.failure_history[failed_agent].append(current_time)

        # Check if we've exceeded max attempts
        attempts = self.fallback_attempts.get(failed_agent, 0)

        # Find applicable fallback rules
        for rule in self.fallback_rules:
            if (
                rule.failed_agent == failed_agent.lower()
                and (rule.condition == "always" or rule.condition == failure_type)
                and attempts < rule.max_attempts
            ):
                # Increment attempt counter
                self.fallback_attempts[failed_agent] = attempts + 1
                return rule.fallback_agents.copy()

        return []  # No fallback available

    def reset_attempts(self, agent: str) -> None:
        """Reset fallback attempt counter for an agent."""
        if agent in self.fallback_attempts:
            del self.fallback_attempts[agent]

    def get_failure_rate(self, agent: str, window_minutes: int = 60) -> float:
        """
        Get recent failure rate for an agent.

        Parameters
        ----------
        agent : str
            Agent name
        window_minutes : int
            Time window in minutes

        Returns
        -------
        float
            Failure rate (0.0 to 1.0)
        """
        if agent not in self.failure_history:
            return 0.0

        cutoff_time = time.time() - (window_minutes * 60)
        recent_failures = [t for t in self.failure_history[agent] if t > cutoff_time]

        # Simple heuristic: if more than 2 failures in the window, high failure rate
        return min(len(recent_failures) / 5.0, 1.0)


class EnhancedConditionalPattern(GraphPattern):
    """
    Enhanced conditional pattern with intelligent routing.

    This pattern implements dynamic agent selection based on:
    - Query complexity analysis
    - Agent performance metrics
    - Context requirements
    - Failure recovery strategies
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        semantic_validator: Optional[WorkflowSemanticValidator] = None,
    ) -> None:
        """
        Initialize enhanced conditional pattern.

        Parameters
        ----------
        config : Optional[RoutingConfig]
            Routing configuration, defaults to standard config
        semantic_validator : Optional[WorkflowSemanticValidator]
            Semantic validator for domain-specific rules
        """
        self.config = config or RoutingConfig()
        self.semantic_validator = semantic_validator
        self.context_analyzer = ContextAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.fallback_manager = FallbackManager()

        # Cache for routing decisions
        self._routing_cache: Dict[str, Tuple[List[str], float]] = {}
        self._cache_ttl = 300  # 5 minutes

    @property
    def name(self) -> str:
        """Pattern name identifier."""
        return "enhanced_conditional"

    @property
    def description(self) -> str:
        """Human-readable pattern description."""
        return "Enhanced conditional routing with dynamic agent selection based on context and performance"

    def analyze_query_context(self, query: str) -> ContextAnalysis:
        """
        Analyze query to determine routing strategy.

        Parameters
        ----------
        query : str
            User query to analyze

        Returns
        -------
        ContextAnalysis
            Analysis results for routing decisions
        """
        return self.context_analyzer.analyze_context(query)

    def select_agents_for_strategy(
        self,
        strategy: RoutingStrategy,
        available_agents: List[str],
        context: ContextAnalysis,
    ) -> List[str]:
        """
        Select agents based on routing strategy.

        Parameters
        ----------
        strategy : RoutingStrategy
            The routing strategy to implement
        available_agents : List[str]
            Available agents to choose from
        context : ContextAnalysis
            Analysis context for decision making

        Returns
        -------
        List[str]
            Selected agents for execution
        """
        available_lower = [agent.lower() for agent in available_agents]
        selected: List[str] = []

        if strategy == RoutingStrategy.STREAMLINED:
            # Minimal agents for simple queries
            if "refiner" in available_lower:
                selected.append("refiner")
            if "synthesis" in available_lower:
                selected.append("synthesis")

        elif strategy == RoutingStrategy.COMPREHENSIVE:
            # All available agents for complex analysis
            selected = available_agents.copy()

        elif strategy == RoutingStrategy.STANDARD:
            # Default CogniVault workflow
            if "refiner" in available_lower:
                selected.append("refiner")

            # Add critic if criticism is needed or complexity is moderate+
            if "critic" in available_lower and (
                context.requires_criticism
                or context.complexity_level
                in [ContextComplexity.MODERATE, ContextComplexity.COMPLEX]
            ):
                selected.append("critic")

            # Add historian if research is needed
            if "historian" in available_lower and context.requires_research:
                selected.append("historian")

            if "synthesis" in available_lower:
                selected.append("synthesis")

        elif strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            # Select based on performance metrics
            agent_scores = {}
            for agent in available_agents:
                agent_scores[agent] = self.performance_tracker.get_performance_score(
                    agent.lower()
                )

            # Sort by performance and select top agents
            sorted_agents = sorted(
                agent_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Always include refiner and synthesis if available
            if "refiner" in available_lower:
                selected.append("refiner")
            if "synthesis" in available_lower:
                selected.append("synthesis")

            # Add best performing optional agents
            for agent, score in sorted_agents:
                if (
                    agent.lower() not in ["refiner", "synthesis"]
                    and len(selected) < 4
                    and score > 0.6
                ):
                    selected.append(agent)

        return selected

    def get_edges(self, agents: List[str]) -> List[Dict[str, str]]:
        """
        Get edge definitions with conditional routing logic.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the graph

        Returns
        -------
        List[Dict[str, str]]
            List of edge dictionaries with conditional routing
        """
        edges = []
        agents_lower = [agent.lower() for agent in agents]

        # For now, implement enhanced standard pattern logic
        # Future enhancement: add conditional routing based on runtime decisions

        if "refiner" in agents_lower:
            # Refiner to other agents (parallel where possible)
            if "critic" in agents_lower:
                edges.append({"from": "refiner", "to": "critic"})
            if "historian" in agents_lower:
                edges.append({"from": "refiner", "to": "historian"})

            # Handle synthesis connections
            if "synthesis" in agents_lower:
                # Collect all intermediate agents
                intermediates = [
                    agent for agent in agents_lower if agent in ["critic", "historian"]
                ]

                if intermediates:
                    # Intermediate agents feed into synthesis
                    for intermediate in intermediates:
                        edges.append({"from": intermediate, "to": "synthesis"})
                else:
                    # Direct refiner to synthesis if no intermediates
                    edges.append({"from": "refiner", "to": "synthesis"})

                # Synthesis to END
                edges.append({"from": "synthesis", "to": "END"})
            else:
                # No synthesis, intermediates are terminal
                for agent in ["critic", "historian"]:
                    if agent in agents_lower:
                        edges.append({"from": agent, "to": "END"})

        # Handle cases without refiner
        elif "critic" in agents_lower or "historian" in agents_lower:
            if "synthesis" in agents_lower:
                for agent in ["critic", "historian"]:
                    if agent in agents_lower:
                        edges.append({"from": agent, "to": "synthesis"})
                edges.append({"from": "synthesis", "to": "END"})
            else:
                # Terminal nodes
                for agent in ["critic", "historian"]:
                    if agent in agents_lower:
                        edges.append({"from": agent, "to": "END"})

        # Handle synthesis-only case
        elif "synthesis" in agents_lower:
            edges.append({"from": "synthesis", "to": "END"})

        return edges

    def get_entry_point(self, agents: List[str]) -> Optional[str]:
        """Get entry point for conditional pattern."""
        agents_lower = [agent.lower() for agent in agents]

        # Prefer refiner as entry point
        if "refiner" in agents_lower:
            return "refiner"

        # Fallback to first available agent
        return agents_lower[0] if agents_lower else None

    def get_exit_points(self, agents: List[str]) -> List[str]:
        """Get exit points for conditional pattern."""
        agents_lower = [agent.lower() for agent in agents]

        # Prefer synthesis as exit point
        if "synthesis" in agents_lower:
            return ["synthesis"]

        # Otherwise, return all non-refiner agents as potential exit points
        exit_points = [agent for agent in agents_lower if agent != "refiner"]
        return exit_points if exit_points else agents_lower

    def get_parallel_groups(self, agents: List[str]) -> List[List[str]]:
        """Get parallel execution groups for conditional pattern."""
        agents_lower = [agent.lower() for agent in agents]

        # Standard parallel group: critic and historian after refiner
        parallel_group = []
        if "critic" in agents_lower:
            parallel_group.append("critic")
        if "historian" in agents_lower:
            parallel_group.append("historian")

        return [parallel_group] if len(parallel_group) > 1 else []

    def validate_agents(self, agents: List[str]) -> bool:
        """
        Validate agents with semantic validation if available.

        Parameters
        ----------
        agents : List[str]
            List of agent names to validate

        Returns
        -------
        bool
            True if agents are compatible
        """
        if self.semantic_validator:
            try:
                result = self.semantic_validator.validate_workflow(agents, self.name)
                return result.is_valid
            except Exception:
                # Fall back to basic validation if semantic validation fails
                pass

        # Basic validation: ensure we have at least one agent
        return len(agents) > 0

    def validate_with_context(
        self,
        agents: List[str],
        query: str,
        routing_strategy: Optional[RoutingStrategy] = None,
    ) -> SemanticValidationResult:
        """
        Comprehensive validation with conditional pattern context.

        Parameters
        ----------
        agents : List[str]
            List of agent names to validate
        query : str
            User query for context analysis
        routing_strategy : Optional[RoutingStrategy]
            The routing strategy being used

        Returns
        -------
        SemanticValidationResult
            Detailed validation result with suggestions
        """
        # Use conditional validator if available, fallback to semantic validator
        validator = None
        if isinstance(self.semantic_validator, ConditionalPatternValidator):
            validator = self.semantic_validator
        elif self.semantic_validator:
            # Create a conditional validator with same settings
            validator = ConditionalPatternValidator(strict_mode=False)
        else:
            # Create default conditional validator
            validator = ConditionalPatternValidator(strict_mode=False)

        # Analyze query context
        context_analysis = self.analyze_query_context(query)

        # Get performance data for validation
        performance_data = {}
        for agent in agents:
            agent_lower = agent.lower()
            performance_data[agent_lower] = {
                "success_rate": self.performance_tracker.get_success_rate(agent_lower)
                or 1.0,
                "average_time_ms": self.performance_tracker.get_average_time(
                    agent_lower
                )
                or 0,
                "performance_score": self.performance_tracker.get_performance_score(
                    agent_lower
                ),
            }

        # Perform validation with conditional context
        return validator.validate_workflow(
            agents,
            self.name,
            routing_strategy=routing_strategy,
            context_analysis=context_analysis,
            performance_data=performance_data,
        )

    def update_performance_metrics(
        self, agent: str, duration_ms: float, success: bool
    ) -> None:
        """
        Update performance metrics for an agent.

        Parameters
        ----------
        agent : str
            Agent name
        duration_ms : float
            Execution duration in milliseconds
        success : bool
            Whether execution was successful
        """
        self.performance_tracker.record_execution(agent, duration_ms, success)

    def get_recommended_agents(
        self, query: str, available_agents: List[str]
    ) -> List[str]:
        """
        Get recommended agents for a query based on analysis.

        Parameters
        ----------
        query : str
            User query
        available_agents : List[str]
            Available agents to choose from

        Returns
        -------
        List[str]
            Recommended agents for the query
        """
        # Check cache first
        cache_key = f"{query[:100]}:{','.join(sorted(available_agents))}"
        if cache_key in self._routing_cache:
            cached_agents, timestamp = self._routing_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_agents

        # Analyze query context
        context = self.analyze_query_context(query)

        # Select agents based on strategy
        selected_agents = self.select_agents_for_strategy(
            context.routing_strategy, available_agents, context
        )

        # Cache the result
        self._routing_cache[cache_key] = (selected_agents, time.time())

        return selected_agents

    def handle_agent_failure(
        self,
        failed_agent: str,
        failure_type: str = "error",
        remaining_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Handle agent failure and determine fallback strategy.

        Parameters
        ----------
        failed_agent : str
            Name of the failed agent
        failure_type : str
            Type of failure ("error", "timeout", "performance_poor")
        remaining_agents : Optional[List[str]]
            Remaining agents available for fallback

        Returns
        -------
        Dict[str, Any]
            Fallback strategy with action and agents
        """
        fallback_agents = self.fallback_manager.get_fallback_agents(
            failed_agent, failure_type
        )

        # Filter fallback agents to only include available ones
        if remaining_agents:
            available_fallbacks = [
                agent
                for agent in fallback_agents
                if agent in [a.lower() for a in remaining_agents]
            ]
        else:
            available_fallbacks = fallback_agents

        failure_rate = self.fallback_manager.get_failure_rate(failed_agent)

        return {
            "action": "fallback" if available_fallbacks else "skip",
            "fallback_agents": available_fallbacks,
            "original_agent": failed_agent,
            "failure_type": failure_type,
            "failure_rate": failure_rate,
            "recommendation": self._get_failure_recommendation(
                failed_agent, failure_type, failure_rate, bool(available_fallbacks)
            ),
        }

    def _get_failure_recommendation(
        self, agent: str, failure_type: str, failure_rate: float, has_fallback: bool
    ) -> str:
        """Get human-readable recommendation for failure handling."""
        if failure_rate > 0.6:
            return f"High failure rate for {agent} ({failure_rate:.1%}). Consider removing from pipeline."
        elif failure_type == "timeout":
            return (
                f"{agent} is experiencing timeouts. Consider performance optimization."
            )
        elif has_fallback:
            return f"Using fallback strategy for {agent} failure."
        else:
            return f"No fallback available for {agent}. Continuing with degraded functionality."

    def get_performance_optimized_agents(
        self, available_agents: List[str], max_agents: int = 4
    ) -> List[str]:
        """
        Get performance-optimized agent selection.

        Parameters
        ----------
        available_agents : List[str]
            Available agents to choose from
        max_agents : int
            Maximum number of agents to select

        Returns
        -------
        List[str]
            Performance-optimized agent selection
        """
        # Score all agents
        agent_scores = {}
        for agent in available_agents:
            agent_lower = agent.lower()
            performance_score = self.performance_tracker.get_performance_score(
                agent_lower
            )
            failure_rate = self.fallback_manager.get_failure_rate(agent_lower)

            # Combine performance and reliability (penalize high failure rates)
            combined_score = performance_score * (1.0 - failure_rate)
            agent_scores[agent] = combined_score

        # Sort by combined score
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        # Always include critical agents if available
        critical_agents = ["refiner", "synthesis"]
        selected: List[str] = []

        # Add critical agents first
        for agent in available_agents:
            if agent.lower() in critical_agents and len(selected) < max_agents:
                selected.append(agent)

        # Add best performing optional agents
        for agent, score in sorted_agents:
            if (
                agent not in selected
                and agent.lower() not in critical_agents
                and len(selected) < max_agents
                and score > 0.5
            ):  # Only add agents with decent performance
                selected.append(agent)

        return selected

    def register_custom_fallback(self, fallback_rule: FallbackRule) -> None:
        """Register a custom fallback rule."""
        self.fallback_manager.register_fallback(fallback_rule)

    def reset_performance_metrics(self) -> None:
        """Reset all performance tracking data."""
        self.performance_tracker = PerformanceTracker()
        self.fallback_manager = FallbackManager()
        self._routing_cache.clear()

    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing and performance statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics about routing decisions and performance
        """
        stats: Dict[str, Any] = {
            "cache_size": len(self._routing_cache),
            "performance_tracking": {},
            "failure_statistics": {},
            "routing_config": {
                "simple_threshold": self.config.simple_threshold,
                "moderate_threshold": self.config.moderate_threshold,
                "complex_threshold": self.config.complex_threshold,
                "performance_weight": self.config.performance_weight,
                "reliability_weight": self.config.reliability_weight,
            },
        }

        # Performance statistics
        for agent in self.performance_tracker.execution_times:
            stats["performance_tracking"][agent] = {
                "average_time_ms": self.performance_tracker.get_average_time(agent),
                "success_rate": self.performance_tracker.get_success_rate(agent),
                "performance_score": self.performance_tracker.get_performance_score(
                    agent
                ),
                "execution_count": len(self.performance_tracker.execution_times[agent]),
            }

        # Failure statistics
        for agent in self.fallback_manager.failure_history:
            stats["failure_statistics"][agent] = {
                "total_failures": len(self.fallback_manager.failure_history[agent]),
                "recent_failure_rate": self.fallback_manager.get_failure_rate(agent),
                "fallback_attempts": self.fallback_manager.fallback_attempts.get(
                    agent, 0
                ),
            }

        return stats


class ConditionalPatternValidator(WorkflowSemanticValidator):
    """
    Specialized semantic validator for conditional patterns.

    This validator understands the dynamic nature of conditional patterns
    and provides contextual validation based on routing strategies.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """
        Initialize conditional pattern validator.

        Parameters
        ----------
        strict_mode : bool
            If True, enforces stricter validation rules
        """
        self.strict_mode = strict_mode

        # Define agent roles and capabilities
        self.agent_roles = {
            "refiner": {
                "role": "preprocessor",
                "critical": True,
                "parallel_safe": False,
            },
            "critic": {"role": "analyzer", "critical": False, "parallel_safe": True},
            "historian": {
                "role": "researcher",
                "critical": False,
                "parallel_safe": True,
            },
            "synthesis": {
                "role": "synthesizer",
                "critical": True,
                "parallel_safe": False,
            },
        }

        # Define valid routing strategies for different contexts
        self.strategy_requirements: Dict[RoutingStrategy, Dict[str, Any]] = {
            RoutingStrategy.STREAMLINED: {
                "min_agents": 1,
                "max_agents": 2,
                "requires": ["synthesis"],
            },
            RoutingStrategy.STANDARD: {
                "min_agents": 2,
                "max_agents": 4,
                "requires": ["refiner", "synthesis"],
            },
            RoutingStrategy.COMPREHENSIVE: {
                "min_agents": 3,
                "max_agents": 4,
                "requires": ["refiner", "synthesis"],
            },
            RoutingStrategy.PERFORMANCE_OPTIMIZED: {
                "min_agents": 1,
                "max_agents": 4,
                "requires": [],
            },
        }

    def validate_workflow(
        self, agents: List[str], pattern: str, **kwargs: Any
    ) -> SemanticValidationResult:
        """
        Validate conditional pattern workflow.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the workflow
        pattern : str
            Name of the graph pattern being used
        **kwargs : Any
            Additional context (routing_strategy, context_analysis, etc.)

        Returns
        -------
        SemanticValidationResult
            Detailed validation result with conditional pattern insights
        """
        result = SemanticValidationResult(is_valid=True, issues=[])
        agents_lower = [agent.lower() for agent in agents]

        # Basic agent validation
        base_result = self.validate_agents(agents)
        result.issues.extend(base_result.issues)
        if not base_result.is_valid:
            result.is_valid = False

        # Skip conditional validation if not a conditional pattern
        if pattern not in ["conditional", "enhanced_conditional"]:
            return result

        # Extract conditional pattern context
        routing_strategy = kwargs.get("routing_strategy")
        context_analysis = kwargs.get("context_analysis")
        performance_data = kwargs.get("performance_data", {})

        # Validate routing strategy
        if routing_strategy:
            self._validate_routing_strategy(result, agents_lower, routing_strategy)

        # Validate context-based selection
        if context_analysis:
            self._validate_context_selection(result, agents_lower, context_analysis)

        # Validate performance considerations
        if performance_data:
            self._validate_performance_selection(result, agents_lower, performance_data)

        # Validate fallback configuration
        self._validate_fallback_viability(result, agents_lower)

        return result

    def _validate_routing_strategy(
        self,
        result: SemanticValidationResult,
        agents: List[str],
        strategy: RoutingStrategy,
    ) -> None:
        """Validate agents against routing strategy requirements."""
        if strategy not in self.strategy_requirements:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Unknown routing strategy: {strategy}",
                suggestion="Use a recognized routing strategy",
            )
            return

        requirements = self.strategy_requirements[strategy]

        # Check agent count
        if len(agents) < requirements["min_agents"]:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Strategy {strategy.value} requires at least {requirements['min_agents']} agents, got {len(agents)}",
                suggestion="Add more agents or use a different strategy",
            )

        if len(agents) > requirements["max_agents"]:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Strategy {strategy.value} typically uses at most {requirements['max_agents']} agents, got {len(agents)}",
                suggestion="Consider using COMPREHENSIVE strategy for more agents",
            )

        # Check required agents
        for required_agent in requirements["requires"]:
            if required_agent not in agents:
                severity = (
                    ValidationSeverity.ERROR
                    if self.strict_mode
                    else ValidationSeverity.WARNING
                )
                result.add_issue(
                    severity,
                    f"Strategy {strategy.value} typically requires {required_agent} agent",
                    suggestion=f"Add {required_agent} agent or use a different strategy",
                )

    def _validate_context_selection(
        self,
        result: SemanticValidationResult,
        agents: List[str],
        context: ContextAnalysis,
    ) -> None:
        """Validate agent selection against context analysis."""
        # Check if research capability is available when needed
        if context.requires_research and "historian" not in agents:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Query appears to require research but historian agent not included",
                suggestion="Consider adding historian agent for better research capabilities",
            )

        # Check if critical analysis is available when needed
        if context.requires_criticism and "critic" not in agents:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Query appears to require critical analysis but critic agent not included",
                suggestion="Consider adding critic agent for thorough analysis",
            )

        # Check agent selection against complexity
        if context.complexity_level == ContextComplexity.SIMPLE and len(agents) > 2:
            result.add_issue(
                ValidationSeverity.INFO,
                f"Simple query using {len(agents)} agents - consider streamlined approach",
                suggestion="For simple queries, refiner + synthesis may be sufficient",
            )

        if (
            context.complexity_level == ContextComplexity.VERY_COMPLEX
            and len(agents) < 3
        ):
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Very complex query using only {len(agents)} agents",
                suggestion="Complex queries typically benefit from multiple specialized agents",
            )

    def _validate_performance_selection(
        self,
        result: SemanticValidationResult,
        agents: List[str],
        performance_data: Dict[str, Any],
    ) -> None:
        """Validate agent selection against performance data."""
        for agent in agents:
            agent_perf = performance_data.get(agent, {})

            # Check for poor performing agents
            success_rate = agent_perf.get("success_rate", 1.0)
            if success_rate < 0.7:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Agent {agent} has low success rate ({success_rate:.1%})",
                    agent=agent,
                    suggestion="Consider fallback configuration or agent optimization",
                )

            # Check for slow agents
            avg_time = agent_perf.get("average_time_ms", 0)
            if avg_time > 10000:  # More than 10 seconds
                result.add_issue(
                    ValidationSeverity.INFO,
                    f"Agent {agent} has high average execution time ({avg_time:.0f}ms)",
                    agent=agent,
                    suggestion="Monitor performance or consider optimization",
                )

    def _validate_fallback_viability(
        self, result: SemanticValidationResult, agents: List[str]
    ) -> None:
        """Validate that meaningful fallbacks exist for critical agents."""
        critical_agents = [
            agent
            for agent in agents
            if self.agent_roles.get(agent, {}).get("critical", False)
        ]

        for critical_agent in critical_agents:
            if critical_agent == "refiner" and "synthesis" not in agents:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Critical refiner agent has no viable fallback without synthesis",
                    agent=critical_agent,
                    suggestion="Include synthesis agent to enable fallback paths",
                )

            if critical_agent == "synthesis" and len(agents) == 1:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Synthesis agent is single point of failure",
                    agent=critical_agent,
                    suggestion="Include additional agents to provide fallback options",
                )

    def get_supported_patterns(self) -> Set[str]:
        """Get supported pattern names."""
        return {"conditional", "enhanced_conditional"}

    def validate_routing_decision(
        self,
        original_agents: List[str],
        selected_agents: List[str],
        context: ContextAnalysis,
    ) -> SemanticValidationResult:
        """
        Validate a routing decision made by the conditional pattern.

        Parameters
        ----------
        original_agents : List[str]
            Original available agents
        selected_agents : List[str]
            Agents selected by routing logic
        context : ContextAnalysis
            Context analysis used for routing

        Returns
        -------
        SemanticValidationResult
            Validation of the routing decision
        """
        result = SemanticValidationResult(is_valid=True, issues=[])

        # Check if important agents were excluded
        excluded = [agent for agent in original_agents if agent not in selected_agents]

        for agent in excluded:
            agent_lower = agent.lower()

            # Check if critical capabilities were dropped
            if agent_lower == "historian" and context.requires_research:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Historian excluded despite research requirements in query",
                    suggestion="Verify routing logic for research-intensive queries",
                )

            if agent_lower == "critic" and context.requires_criticism:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Critic excluded despite critical analysis requirements",
                    suggestion="Verify routing logic for analytical queries",
                )

        # Provide optimization suggestions
        if len(selected_agents) == len(original_agents):
            result.add_issue(
                ValidationSeverity.INFO,
                "All available agents selected - consider optimization opportunities",
                suggestion="Use context analysis to selectively include agents",
            )

        return result
