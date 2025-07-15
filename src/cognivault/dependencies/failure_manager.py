"""
Advanced failure management with cascade prevention and retry strategies.

This module provides sophisticated failure handling capabilities including
cascade prevention, intelligent retry strategies, failure impact analysis,
and recovery mechanisms for agent execution failures.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple

from cognivault.context import AgentContext
from cognivault.observability import get_logger
from .graph_engine import DependencyGraphEngine, DependencyType

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of failures that can occur during agent execution."""

    TIMEOUT = "timeout"
    LLM_ERROR = "llm_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class CascadePreventionStrategy(Enum):
    """Strategies for preventing failure cascades."""

    CIRCUIT_BREAKER = "circuit_breaker"  # Stop execution after threshold
    ISOLATION = "isolation"  # Isolate failed components
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Continue with reduced functionality
    FALLBACK_CHAIN = "fallback_chain"  # Try alternative execution paths
    CHECKPOINT_ROLLBACK = "checkpoint_rollback"  # Rollback to stable state


class RetryStrategy(Enum):
    """Retry strategies for failed agents."""

    FIXED_INTERVAL = "fixed_interval"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    ADAPTIVE = "adaptive"
    NO_RETRY = "no_retry"


@dataclass
class RetryConfiguration:
    """Configuration for retry behavior."""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    base_delay_ms: float = 1000.0
    max_delay_ms: float = 30000.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    reset_on_success: bool = True

    # Adaptive retry parameters
    success_rate_threshold: float = 0.7
    failure_window_size: int = 10

    def calculate_delay(self, attempt: int, recent_failures: int = 0) -> float:
        """Calculate delay for the given attempt number."""
        if self.strategy == RetryStrategy.NO_RETRY:
            return 0

        if self.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = self.base_delay_ms

        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay_ms * attempt

        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay_ms * (self.backoff_multiplier ** (attempt - 1))

        elif self.strategy == RetryStrategy.ADAPTIVE:
            # Adapt based on recent failure rate
            failure_rate = recent_failures / self.failure_window_size
            if failure_rate > (1 - self.success_rate_threshold):
                # High failure rate - increase delay
                delay = self.base_delay_ms * (self.backoff_multiplier**attempt)
            else:
                # Low failure rate - use fixed interval
                delay = self.base_delay_ms

        else:
            delay = self.base_delay_ms

        # Apply jitter to prevent thundering herd
        if self.jitter:
            import random

            jitter_factor = random.uniform(0.8, 1.2)
            delay = delay * jitter_factor

        return min(delay, self.max_delay_ms)


@dataclass
class FailureRecord:
    """Record of a failure event."""

    agent_id: str
    failure_type: FailureType
    error_message: str
    timestamp: float
    attempt_number: int
    context_snapshot: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    impact_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "failure_type": self.failure_type.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "attempt_number": self.attempt_number,
            "context_snapshot": self.context_snapshot,
            "stack_trace": self.stack_trace,
            "recovery_action": self.recovery_action,
            "impact_score": self.impact_score,
        }


@dataclass
class FailureImpactAnalysis:
    """Analysis of failure impact on the execution graph."""

    failed_agent: str
    directly_affected: List[str]  # Agents that depend on failed agent
    transitively_affected: List[str]  # All downstream agents affected
    critical_path_affected: bool
    estimated_delay_ms: float
    alternative_paths: List[List[str]]  # Alternative execution paths
    recovery_options: List[str]
    severity_score: float  # 0-1 scale

    def get_total_affected_count(self) -> int:
        """Get total number of affected agents."""
        return len(set(self.directly_affected + self.transitively_affected))

    def has_recovery_options(self) -> bool:
        """Check if recovery options are available."""
        return len(self.recovery_options) > 0 or len(self.alternative_paths) > 0


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_ms: float = 60000.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_attempts = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed through the circuit breaker."""
        current_time = time.time() * 1000

        if self.state == "CLOSED":
            return True

        elif self.state == "OPEN":
            if current_time - self.last_failure_time > self.recovery_timeout_ms:
                self.state = "HALF_OPEN"
                self.half_open_attempts = 1  # Count this call
                return True
            return False

        elif self.state == "HALF_OPEN":
            if self.half_open_attempts < self.half_open_max_calls:
                self.half_open_attempts += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record a successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = float(time.time() * 1000)

        if self.state == "HALF_OPEN":
            self.state = "OPEN"
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


class FailureManager:
    """
    Advanced failure manager with cascade prevention and recovery strategies.

    Provides comprehensive failure handling including impact analysis,
    cascade prevention, retry management, and recovery coordination.
    """

    def __init__(self, graph_engine: DependencyGraphEngine):
        self.graph_engine = graph_engine
        self.failure_history: List[FailureRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_configs: Dict[str, RetryConfiguration] = {}
        self.cascade_prevention = CascadePreventionStrategy.CIRCUIT_BREAKER

        # Failure tracking
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.recent_failures: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.blocked_agents: Set[str] = set()

        # Recovery state
        self.recovery_checkpoints: Dict[str, AgentContext] = {}
        self.fallback_chains: Dict[str, List[str]] = {}

    def configure_retry(self, agent_id: str, config: RetryConfiguration) -> None:
        """Configure retry behavior for a specific agent."""
        self.retry_configs[agent_id] = config
        logger.debug(f"Configured retry for {agent_id}: {config.strategy.value}")

    def configure_circuit_breaker(
        self,
        agent_id: str,
        failure_threshold: int = 5,
        recovery_timeout_ms: int = 60000,
    ) -> None:
        """Configure circuit breaker for a specific agent."""
        self.circuit_breakers[agent_id] = CircuitBreaker(
            failure_threshold=failure_threshold, recovery_timeout_ms=recovery_timeout_ms
        )
        logger.debug(f"Configured circuit breaker for {agent_id}")

    def set_cascade_prevention(self, strategy: CascadePreventionStrategy) -> None:
        """Set the cascade prevention strategy."""
        self.cascade_prevention = strategy
        logger.info(f"Set cascade prevention strategy: {strategy.value}")

    def add_fallback_chain(self, agent_id: str, fallback_agents: List[str]) -> None:
        """Add fallback chain for an agent."""
        self.fallback_chains[agent_id] = fallback_agents
        logger.debug(f"Added fallback chain for {agent_id}: {fallback_agents}")

    def can_execute_agent(
        self, agent_id: str, context: AgentContext
    ) -> Tuple[bool, str]:
        """
        Check if an agent can execute given current failure state.

        Returns
        -------
        Tuple[bool, str]
            (can_execute, reason)
        """
        if agent_id in self.blocked_agents:
            return False, "Agent is blocked due to repeated failures"

        # Check circuit breaker
        if agent_id in self.circuit_breakers:
            if not self.circuit_breakers[agent_id].can_execute():
                return (
                    False,
                    f"Circuit breaker is {self.circuit_breakers[agent_id].get_state()}",
                )

        # Check dependency failures
        impact_analysis = self.analyze_dependency_failures(agent_id, context)
        if impact_analysis and impact_analysis.severity_score > 0.8:
            return False, "Critical dependencies have failed"

        return True, "OK"

    async def handle_agent_failure(
        self,
        agent_id: str,
        error: Exception,
        context: AgentContext,
        attempt_number: int = 1,
    ) -> Tuple[bool, Optional[str]]:
        """
        Handle an agent failure with comprehensive recovery strategies.

        Parameters
        ----------
        agent_id : str
            ID of the failed agent
        error : Exception
            The error that occurred
        context : AgentContext
            Current execution context
        attempt_number : int
            Current attempt number

        Returns
        -------
        Tuple[bool, Optional[str]]
            (should_retry, recovery_action)
        """
        # Classify failure type
        failure_type = self._classify_failure(error)

        # Record failure
        failure_record = FailureRecord(
            agent_id=agent_id,
            failure_type=failure_type,
            error_message=str(error),
            timestamp=time.time(),
            attempt_number=attempt_number,
            context_snapshot=self._create_context_snapshot(context),
            stack_trace=self._extract_stack_trace(error),
        )

        self.failure_history.append(failure_record)
        self.failure_counts[agent_id] += 1
        self.recent_failures[agent_id].append(time.time())

        # Update circuit breaker
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id].record_failure()

        # Analyze failure impact
        impact_analysis = self.analyze_failure_impact(agent_id, context)
        failure_record.impact_score = impact_analysis.severity_score

        logger.warning(
            f"Agent {agent_id} failed (attempt {attempt_number}): {error}. "
            f"Impact score: {impact_analysis.severity_score:.2f}"
        )

        # Determine if retry is appropriate
        should_retry = self._should_retry(
            agent_id, attempt_number, failure_type, impact_analysis
        )

        # Apply cascade prevention strategy
        recovery_action = await self._apply_cascade_prevention(
            agent_id, impact_analysis, context
        )

        # Update failure record with recovery action
        failure_record.recovery_action = recovery_action

        return should_retry, recovery_action

    def analyze_failure_impact(
        self, failed_agent: str, context: AgentContext
    ) -> FailureImpactAnalysis:
        """Analyze the impact of an agent failure on the execution graph."""
        # Find directly affected agents (immediate dependents)
        directly_affected = []
        for edge in self.graph_engine.edges:
            if (
                edge.from_agent == failed_agent
                and edge.dependency_type == DependencyType.HARD
            ):
                directly_affected.append(edge.to_agent)

        # Find transitively affected agents
        transitively_affected = self._find_transitive_dependents(failed_agent)

        # Check if critical path is affected
        critical_path_affected = self._is_critical_path_affected(failed_agent, context)

        # Estimate delay impact
        estimated_delay = self._estimate_failure_delay(failed_agent, directly_affected)

        # Find alternative execution paths
        alternative_paths = self._find_alternative_paths(failed_agent, context)

        # Determine recovery options
        recovery_options = self._get_recovery_options(failed_agent)

        # Calculate severity score
        severity_score = self._calculate_severity_score(
            len(directly_affected),
            len(transitively_affected),
            critical_path_affected,
            estimated_delay,
        )

        return FailureImpactAnalysis(
            failed_agent=failed_agent,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            critical_path_affected=critical_path_affected,
            estimated_delay_ms=estimated_delay,
            alternative_paths=alternative_paths,
            recovery_options=recovery_options,
            severity_score=severity_score,
        )

    def analyze_dependency_failures(
        self, agent_id: str, context: AgentContext
    ) -> Optional[FailureImpactAnalysis]:
        """Analyze how dependency failures affect a specific agent."""
        # Check if any dependencies have failed
        failed_dependencies = []
        for edge in self.graph_engine.edges:
            if edge.to_agent == agent_id:
                dependency = edge.from_agent
                if any(
                    f.agent_id == dependency for f in self.failure_history[-10:]
                ):  # Recent failures
                    failed_dependencies.append(dependency)

        if not failed_dependencies:
            return None

        # For now, analyze the first failed dependency
        # In a full implementation, we'd analyze combined impact
        return self.analyze_failure_impact(failed_dependencies[0], context)

    async def attempt_recovery(
        self, agent_id: str, recovery_action: str, context: AgentContext
    ) -> bool:
        """Attempt to recover from a failure using the specified recovery action."""
        logger.info(f"Attempting recovery for {agent_id}: {recovery_action}")

        if recovery_action == "fallback_chain":
            return await self._execute_fallback_chain(agent_id, context)

        elif recovery_action == "checkpoint_rollback":
            return self._rollback_to_checkpoint(agent_id, context)

        elif recovery_action == "graceful_degradation":
            return self._apply_graceful_degradation(agent_id, context)

        elif recovery_action == "isolation":
            return self._isolate_failed_component(agent_id, context)

        else:
            logger.warning(f"Unknown recovery action: {recovery_action}")
            return False

    def create_checkpoint(self, checkpoint_id: str, context: AgentContext) -> None:
        """Create a checkpoint for recovery purposes."""
        # Create a deep copy of the context
        import copy

        self.recovery_checkpoints[checkpoint_id] = copy.deepcopy(context)
        logger.debug(f"Created checkpoint: {checkpoint_id}")

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics."""
        # Calculate failure rates by agent
        agent_failure_rates = {}
        for agent_id in self.failure_counts:
            total_failures = self.failure_counts[agent_id]
            recent_failures = len(
                [
                    f
                    for f in self.failure_history
                    if f.agent_id == agent_id
                    and time.time() - f.timestamp < 3600  # Last hour
                ]
            )
            agent_failure_rates[agent_id] = {
                "total_failures": total_failures,
                "recent_failures": recent_failures,
                "failure_rate": (
                    recent_failures / 10 if recent_failures > 0 else 0
                ),  # Rough estimate
            }

        # Calculate failure type distribution
        failure_type_dist: Dict[str, int] = defaultdict(int)
        for failure in self.failure_history:
            failure_type_dist[failure.failure_type.value] += 1

        # Circuit breaker states
        circuit_breaker_states = {
            agent_id: cb.get_state() for agent_id, cb in self.circuit_breakers.items()
        }

        return {
            "total_failures": len(self.failure_history),
            "unique_failed_agents": len(self.failure_counts),
            "blocked_agents": list(self.blocked_agents),
            "agent_failure_rates": agent_failure_rates,
            "failure_type_distribution": dict(failure_type_dist),
            "circuit_breaker_states": circuit_breaker_states,
            "cascade_prevention_strategy": self.cascade_prevention.value,
            "active_fallback_chains": len(self.fallback_chains),
        }

    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify the type of failure based on the exception."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if "timeout" in error_str or "timeout" in error_type:
            return FailureType.TIMEOUT
        elif "openai" in error_str or "llm" in error_str or "api" in error_str:
            return FailureType.LLM_ERROR
        elif "config" in error_str or "configuration" in error_str:
            return FailureType.CONFIGURATION_ERROR
        elif "validation" in error_str or "invalid" in error_str:
            return FailureType.VALIDATION_ERROR
        elif "memory" in error_str or "resource" in error_str:
            return FailureType.RESOURCE_EXHAUSTION
        elif "network" in error_str or "connection" in error_str:
            return FailureType.NETWORK_ERROR
        else:
            return FailureType.UNKNOWN

    def _should_retry(
        self,
        agent_id: str,
        attempt_number: int,
        failure_type: FailureType,
        impact_analysis: FailureImpactAnalysis,
    ) -> bool:
        """Determine if an agent should be retried."""
        config = self.retry_configs.get(agent_id, RetryConfiguration())

        # Check max attempts
        if attempt_number >= config.max_attempts:
            return False

        # Check circuit breaker
        if agent_id in self.circuit_breakers:
            if not self.circuit_breakers[agent_id].can_execute():
                return False

        # Consider failure type
        if failure_type in [
            FailureType.CONFIGURATION_ERROR,
            FailureType.VALIDATION_ERROR,
        ]:
            return False  # These likely won't resolve with retry

        # Consider impact severity
        if impact_analysis.severity_score > 0.9:
            return False  # Too risky to retry high-impact failures

        return True

    async def _apply_cascade_prevention(
        self,
        agent_id: str,
        impact_analysis: FailureImpactAnalysis,
        context: AgentContext,
    ) -> str:
        """Apply cascade prevention strategy."""
        if self.cascade_prevention == CascadePreventionStrategy.CIRCUIT_BREAKER:
            if impact_analysis.severity_score > 0.7:
                self.blocked_agents.add(agent_id)
                return "circuit_breaker_activated"

        elif self.cascade_prevention == CascadePreventionStrategy.ISOLATION:
            if impact_analysis.get_total_affected_count() > 3:
                return "isolation"

        elif self.cascade_prevention == CascadePreventionStrategy.GRACEFUL_DEGRADATION:
            if impact_analysis.has_recovery_options():
                return "graceful_degradation"

        elif self.cascade_prevention == CascadePreventionStrategy.FALLBACK_CHAIN:
            if agent_id in self.fallback_chains:
                return "fallback_chain"

        elif self.cascade_prevention == CascadePreventionStrategy.CHECKPOINT_ROLLBACK:
            if self.recovery_checkpoints:
                return "checkpoint_rollback"

        return "no_action"

    async def _execute_fallback_chain(
        self, agent_id: str, context: AgentContext
    ) -> bool:
        """Execute fallback chain for a failed agent."""
        if agent_id not in self.fallback_chains:
            return False

        fallback_agents = self.fallback_chains[agent_id]
        logger.info(f"Executing fallback chain for {agent_id}: {fallback_agents}")

        for fallback_agent_id in fallback_agents:
            try:
                if fallback_agent_id in self.graph_engine.nodes:
                    fallback_node = self.graph_engine.nodes[fallback_agent_id]
                    agent = fallback_node.agent

                    # Check if fallback agent can execute
                    can_execute, reason = self.can_execute_agent(
                        fallback_agent_id, context
                    )
                    if not can_execute:
                        logger.warning(
                            f"Fallback agent {fallback_agent_id} cannot execute: {reason}"
                        )
                        continue

                    # Execute fallback agent
                    await agent.run(context)
                    logger.info(
                        f"Fallback agent {fallback_agent_id} executed successfully"
                    )
                    return True

            except Exception as e:
                logger.warning(f"Fallback agent {fallback_agent_id} also failed: {e}")
                continue

        return False

    def _rollback_to_checkpoint(self, agent_id: str, context: AgentContext) -> bool:
        """Rollback to the most recent checkpoint."""
        if not self.recovery_checkpoints:
            return False

        # Find the most recent checkpoint
        latest_checkpoint = max(self.recovery_checkpoints.keys())
        checkpoint_context = self.recovery_checkpoints[latest_checkpoint]

        # Restore context state
        context.agent_outputs = checkpoint_context.agent_outputs.copy()
        context.execution_state = checkpoint_context.execution_state.copy()

        logger.info(f"Rolled back to checkpoint: {latest_checkpoint}")
        return True

    def _apply_graceful_degradation(self, agent_id: str, context: AgentContext) -> bool:
        """Apply graceful degradation for failed agent."""
        # Mark agent as degraded and continue execution
        context.execution_state.setdefault("degraded_agents", []).append(agent_id)

        # Add placeholder output to prevent downstream failures
        context.agent_outputs[agent_id] = (
            f"[DEGRADED] Agent {agent_id} failed but execution continued"
        )

        logger.info(f"Applied graceful degradation for {agent_id}")
        return True

    def _isolate_failed_component(self, agent_id: str, context: AgentContext) -> bool:
        """Isolate failed component to prevent cascade failures."""
        self.blocked_agents.add(agent_id)

        # Remove agent from future execution plans
        context.execution_state.setdefault("isolated_agents", []).append(agent_id)

        logger.info(f"Isolated failed component: {agent_id}")
        return True

    def _find_transitive_dependents(self, agent_id: str) -> List[str]:
        """Find all agents transitively dependent on the given agent."""
        dependents = set()
        to_visit = [agent_id]
        visited = set()

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            # Find direct dependents
            for edge in self.graph_engine.edges:
                if edge.from_agent == current and edge.to_agent not in visited:
                    dependents.add(edge.to_agent)
                    to_visit.append(edge.to_agent)

        return list(dependents)

    def _is_critical_path_affected(self, agent_id: str, context: AgentContext) -> bool:
        """Check if the failed agent is on the critical execution path."""
        # For now, consider agents with many dependents as critical
        dependent_count = len(
            [edge for edge in self.graph_engine.edges if edge.from_agent == agent_id]
        )
        return dependent_count > 2

    def _estimate_failure_delay(
        self, agent_id: str, affected_agents: List[str]
    ) -> float:
        """Estimate delay caused by the failure."""
        node = self.graph_engine.nodes.get(agent_id)
        if not node:
            return 0

        # Base delay from agent timeout
        base_delay = node.timeout_ms or 30000

        # Additional delay from affected agents
        affected_delay = len(affected_agents) * 5000  # 5s per affected agent

        return base_delay + affected_delay

    def _find_alternative_paths(
        self, failed_agent: str, context: AgentContext
    ) -> List[List[str]]:
        """Find alternative execution paths around the failed agent."""
        # Simplified implementation - in practice this would be more sophisticated
        alternatives = []

        if failed_agent in self.fallback_chains:
            alternatives.append(self.fallback_chains[failed_agent])

        return alternatives

    def _get_recovery_options(self, agent_id: str) -> List[str]:
        """Get available recovery options for a failed agent."""
        options = []

        if agent_id in self.fallback_chains:
            options.append("fallback_chain")

        if self.recovery_checkpoints:
            options.append("checkpoint_rollback")

        options.append("graceful_degradation")
        options.append("isolation")

        return options

    def _calculate_severity_score(
        self,
        direct_affected: int,
        transitive_affected: int,
        critical_path: bool,
        delay_ms: float,
    ) -> float:
        """Calculate failure severity score (0-1)."""
        # Weight different factors
        affected_score = min((direct_affected + transitive_affected) / 10, 1.0)
        critical_score = 0.3 if critical_path else 0.0
        delay_score = min(delay_ms / 60000, 0.4)  # Max 0.4 for 1 minute delay

        return min(affected_score + critical_score + delay_score, 1.0)

    def _create_context_snapshot(self, context: AgentContext) -> Dict[str, Any]:
        """Create a snapshot of context for failure analysis."""
        return {
            "agent_count": len(context.agent_outputs),
            "query_length": len(context.query),
            "execution_state_keys": list(context.execution_state.keys()),
            "timestamp": time.time(),
        }

    def _extract_stack_trace(self, error: Exception) -> Optional[str]:
        """Extract stack trace from exception."""
        import traceback

        try:
            return traceback.format_exc()
        except Exception:
            return None
