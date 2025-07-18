"""
Dynamic agent composition with runtime discovery and hot-swapping.

This module provides capabilities for discovering, loading, and dynamically
composing agents at runtime, including hot-swapping of failed agents and
dynamic graph reconfiguration.
"""

import importlib
import inspect
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod

from cognivault.agents.base_agent import BaseAgent
from cognivault.context import AgentContext
from cognivault.observability import get_logger
from .graph_engine import (
    DependencyGraphEngine,
    DependencyNode,
)

logger = get_logger(__name__)


class DiscoveryStrategy(Enum):
    """Strategies for discovering agents."""

    FILESYSTEM = "filesystem"  # Scan filesystem for agent modules
    REGISTRY = "registry"  # Use agent registry
    NETWORK = "network"  # Discover over network (future)
    PLUGIN = "plugin"  # Plugin-based discovery
    HYBRID = "hybrid"  # Combination of strategies


class CompositionEvent(Enum):
    """Events in the dynamic composition lifecycle."""

    AGENT_DISCOVERED = "agent_discovered"
    AGENT_LOADED = "agent_loaded"
    AGENT_REGISTERED = "agent_registered"
    AGENT_SWAPPED = "agent_swapped"
    AGENT_UNLOADED = "agent_unloaded"
    GRAPH_RECONFIGURED = "graph_reconfigured"
    COMPOSITION_OPTIMIZED = "composition_optimized"


@dataclass
class AgentMetadata:
    """Metadata about a discovered agent."""

    agent_id: str
    agent_class: str
    module_path: str
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    compatibility: Dict[str, str] = field(default_factory=dict)  # version constraints

    # Discovery metadata
    discovered_at: float = field(default_factory=time.time)
    discovery_strategy: Optional[DiscoveryStrategy] = None
    file_path: Optional[Path] = None
    checksum: Optional[str] = None

    # Runtime metadata
    load_count: int = 0
    last_loaded: Optional[float] = None
    load_errors: List[str] = field(default_factory=list)
    is_loaded: bool = False

    def can_replace(self, other: "AgentMetadata") -> bool:
        """Check if this agent can replace another agent."""
        # Basic compatibility check
        if self.agent_id != other.agent_id:
            return False

        # Check version compatibility
        if "min_version" in other.compatibility:
            # Simple version comparison (in practice, use proper version parsing)
            min_version = other.compatibility["min_version"]
            if self.version < min_version:
                return False

        # Check capabilities
        for required_cap in other.capabilities:
            if required_cap not in self.capabilities:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "agent_class": self.agent_class,
            "module_path": self.module_path,
            "version": self.version,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "resource_requirements": self.resource_requirements,
            "compatibility": self.compatibility,
            "discovered_at": self.discovered_at,
            "discovery_strategy": (
                self.discovery_strategy.value if self.discovery_strategy else None
            ),
            "file_path": str(self.file_path) if self.file_path else None,
            "load_count": self.load_count,
            "last_loaded": self.last_loaded,
            "is_loaded": self.is_loaded,
            "load_errors": self.load_errors,
        }


@dataclass
class CompositionRule:
    """Rule for dynamic composition decisions."""

    rule_id: str
    name: str
    condition: Callable[[AgentContext, Dict[str, Any]], bool]
    action: Callable[[AgentContext, Dict[str, Any]], Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    description: str = ""

    def evaluate(self, context: AgentContext, metadata: Dict[str, Any]) -> bool:
        """Evaluate if this rule should be applied."""
        if not self.enabled:
            return False

        try:
            return self.condition(context, metadata)
        except Exception as e:
            logger.warning(f"Error evaluating composition rule {self.rule_id}: {e}")
            return False

    def apply(self, context: AgentContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the rule's action."""
        try:
            return self.action(context, metadata)
        except Exception as e:
            logger.error(f"Error applying composition rule {self.rule_id}: {e}")
            return {}


class AgentDiscoverer(ABC):
    """Abstract base for agent discovery implementations."""

    @abstractmethod
    async def discover_agents(self) -> List[AgentMetadata]:
        """Discover available agents."""
        pass

    @abstractmethod
    def can_hot_reload(self) -> bool:
        """Check if this discoverer supports hot reloading."""
        pass


class FilesystemDiscoverer(AgentDiscoverer):
    """Discover agents by scanning the filesystem."""

    def __init__(self, search_paths: List[Path], patterns: Optional[List[str]] = None):
        self.search_paths = [Path(p) for p in search_paths]
        self.patterns = patterns or ["*agent*.py", "*_agent.py"]
        self._file_checksums: Dict[Path, str] = {}

    async def discover_agents(self) -> List[AgentMetadata]:
        """Discover agents by scanning filesystem."""
        discovered = []

        for search_path in self.search_paths:
            if not search_path.exists():
                continue

            # Scan for Python files matching patterns
            for pattern in self.patterns:
                for file_path in search_path.rglob(pattern):
                    if file_path.is_file() and file_path.suffix == ".py":
                        metadata = await self._analyze_agent_file(file_path)
                        if metadata:
                            discovered.append(metadata)

        logger.info(f"Discovered {len(discovered)} agents via filesystem scan")
        return discovered

    def can_hot_reload(self) -> bool:
        """Filesystem discoverer supports hot reloading."""
        return True

    async def _analyze_agent_file(self, file_path: Path) -> Optional[AgentMetadata]:
        """Analyze a Python file to extract agent metadata."""
        try:
            # Calculate file checksum
            checksum = self._calculate_checksum(file_path)

            # Skip if file hasn't changed
            if (
                file_path in self._file_checksums
                and self._file_checksums[file_path] == checksum
            ):
                return None

            self._file_checksums[file_path] = checksum

            # Convert path to module path
            module_path = self._path_to_module(file_path)

            # Dynamically import and inspect
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find BaseAgent subclasses
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseAgent)
                    and obj != BaseAgent
                    and obj.__module__ == module.__name__
                ):
                    # Extract metadata from class
                    agent_id = getattr(obj, "agent_id", name.lower())
                    capabilities = getattr(obj, "capabilities", [])
                    dependencies = getattr(obj, "dependencies", [])
                    version = getattr(obj, "version", "1.0.0")

                    return AgentMetadata(
                        agent_id=agent_id,
                        agent_class=f"{module_path}.{name}",
                        module_path=module_path,
                        version=version,
                        capabilities=capabilities,
                        dependencies=dependencies,
                        discovery_strategy=DiscoveryStrategy.FILESYSTEM,
                        file_path=file_path,
                        checksum=checksum,
                    )

        except Exception as e:
            logger.warning(f"Error analyzing agent file {file_path}: {e}")

        return None

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        import hashlib

        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _path_to_module(self, file_path: Path) -> str:
        """Convert file path to Python module path."""
        # This is a simplified implementation
        # In practice, would need more sophisticated path resolution
        parts = file_path.with_suffix("").parts
        return ".".join(parts)


class RegistryDiscoverer(AgentDiscoverer):
    """Discover agents using the agent registry."""

    def __init__(self, registry):
        self.registry = registry

    async def discover_agents(self) -> List[AgentMetadata]:
        """Discover agents from registry."""
        discovered = []

        for agent_id, agent_metadata in self.registry._agents.items():
            metadata = AgentMetadata(
                agent_id=agent_id,
                agent_class=agent_metadata.agent_class.__name__,
                module_path=agent_metadata.agent_class.__module__,
                capabilities=getattr(agent_metadata.agent_class, "capabilities", []),
                dependencies=agent_metadata.dependencies or [],
                discovery_strategy=DiscoveryStrategy.REGISTRY,
            )
            discovered.append(metadata)

        logger.info(f"Discovered {len(discovered)} agents via registry")
        return discovered

    def can_hot_reload(self) -> bool:
        """Registry discoverer has limited hot reload support."""
        return False


class DynamicAgentComposer:
    """
    Dynamic agent composer with runtime discovery and hot-swapping.

    Provides capabilities for discovering agents at runtime, dynamically
    composing execution graphs, and hot-swapping failed or outdated agents.
    """

    def __init__(self, graph_engine: DependencyGraphEngine):
        self.graph_engine = graph_engine
        self.discoverers: List[AgentDiscoverer] = []
        self.discovered_agents: Dict[str, AgentMetadata] = {}
        self.loaded_agents: Dict[str, BaseAgent] = {}
        self.composition_rules: List[CompositionRule] = []

        # Hot-swapping state
        self.swap_candidates: Dict[str, List[AgentMetadata]] = defaultdict(list)
        self.swap_history: List[Dict[str, Any]] = []
        self.auto_discovery_enabled = False
        self.auto_swap_enabled = False

        # Event tracking
        self.composition_events: List[Dict[str, Any]] = []
        self.event_handlers: Dict[CompositionEvent, List[Callable]] = defaultdict(list)

        # Performance tracking
        self.discovery_stats = {
            "total_discoveries": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "swaps_performed": 0,
        }

    def add_discoverer(self, discoverer: AgentDiscoverer) -> None:
        """Add an agent discoverer."""
        self.discoverers.append(discoverer)
        logger.info(f"Added discoverer: {type(discoverer).__name__}")

    def add_composition_rule(self, rule: CompositionRule) -> None:
        """Add a dynamic composition rule."""
        self.composition_rules.append(rule)
        # Sort by priority (higher priority first)
        self.composition_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added composition rule: {rule.name}")

    def add_event_handler(self, event: CompositionEvent, handler: Callable) -> None:
        """Add an event handler for composition events."""
        self.event_handlers[event].append(handler)

    async def discover_agents(
        self, force_rediscovery: bool = False
    ) -> Dict[str, AgentMetadata]:
        """Run discovery across all configured discoverers."""
        if not force_rediscovery and self.discovered_agents:
            return self.discovered_agents

        all_discovered: Dict[str, AgentMetadata] = {}

        for discoverer in self.discoverers:
            try:
                discovered = await discoverer.discover_agents()
                for metadata in discovered:
                    # Merge with existing metadata if present
                    if metadata.agent_id in all_discovered:
                        existing = all_discovered[metadata.agent_id]
                        if metadata.version > existing.version:
                            all_discovered[metadata.agent_id] = metadata
                    else:
                        all_discovered[metadata.agent_id] = metadata

                    self._emit_event(
                        CompositionEvent.AGENT_DISCOVERED,
                        {
                            "agent_id": metadata.agent_id,
                            "discoverer": type(discoverer).__name__,
                            "metadata": metadata.to_dict(),
                        },
                    )

                self.discovery_stats["total_discoveries"] += len(discovered)

            except Exception as e:
                logger.error(f"Error in discoverer {type(discoverer).__name__}: {e}")

        self.discovered_agents.update(all_discovered)
        logger.info(f"Total discovered agents: {len(self.discovered_agents)}")

        return all_discovered

    async def load_agent(
        self, agent_id: str, force_reload: bool = False
    ) -> Optional[BaseAgent]:
        """Load an agent by ID."""
        if agent_id in self.loaded_agents and not force_reload:
            return self.loaded_agents[agent_id]

        if agent_id not in self.discovered_agents:
            logger.warning(f"Agent {agent_id} not found in discovered agents")
            return None

        metadata = self.discovered_agents[agent_id]

        try:
            # Dynamic import and instantiation
            module_path, class_name = metadata.agent_class.rsplit(".", 1)
            module = importlib.import_module(module_path)

            # Reload module if force_reload
            if force_reload:
                importlib.reload(module)

            agent_class = getattr(module, class_name)

            # Create agent instance
            agent = agent_class()

            # Update metadata
            metadata.load_count += 1
            metadata.last_loaded = time.time()
            metadata.is_loaded = True

            self.loaded_agents[agent_id] = agent
            self.discovery_stats["successful_loads"] += 1

            self._emit_event(
                CompositionEvent.AGENT_LOADED,
                {"agent_id": agent_id, "metadata": metadata.to_dict()},
            )

            logger.info(f"Loaded agent: {agent_id}")
            return agent

        except Exception as e:
            error_msg = f"Failed to load agent {agent_id}: {e}"
            logger.error(error_msg)

            metadata.load_errors.append(error_msg)
            self.discovery_stats["failed_loads"] += 1

            return None

    async def hot_swap_agent(
        self, old_agent_id: str, new_agent_id: str, context: AgentContext
    ) -> bool:
        """Hot-swap one agent for another."""
        logger.info(f"Attempting hot swap: {old_agent_id} -> {new_agent_id}")

        # Load new agent
        new_agent = await self.load_agent(new_agent_id)
        if new_agent is None:
            return False

        # Check compatibility
        old_metadata = self.discovered_agents.get(old_agent_id)
        new_metadata = self.discovered_agents.get(new_agent_id)

        if old_metadata and new_metadata:
            if not new_metadata.can_replace(old_metadata):
                logger.warning(
                    f"Agent {new_agent_id} is not compatible with {old_agent_id}"
                )
                return False

        # Update graph engine
        if old_agent_id in self.graph_engine.nodes:
            old_node = self.graph_engine.nodes[old_agent_id]

            # Create new node with updated agent
            new_node = DependencyNode(
                agent_id=new_agent_id,
                agent=new_agent,
                priority=old_node.priority,
                resource_constraints=old_node.resource_constraints,
                max_retries=old_node.max_retries,
                timeout_ms=old_node.timeout_ms,
            )

            # Update graph
            self.graph_engine.remove_node(old_agent_id)
            self.graph_engine.add_node(new_node)

            # Update edges
            for edge in self.graph_engine.edges.copy():
                if edge.from_agent == old_agent_id:
                    edge.from_agent = new_agent_id
                elif edge.to_agent == old_agent_id:
                    edge.to_agent = new_agent_id

        # Update loaded agents
        if old_agent_id in self.loaded_agents:
            del self.loaded_agents[old_agent_id]

        # Record swap
        swap_record = {
            "old_agent": old_agent_id,
            "new_agent": new_agent_id,
            "timestamp": time.time(),
            "context_snapshot": self._create_context_snapshot(context),
        }
        self.swap_history.append(swap_record)
        self.discovery_stats["swaps_performed"] += 1

        self._emit_event(CompositionEvent.AGENT_SWAPPED, swap_record)

        logger.info(f"Successfully swapped {old_agent_id} -> {new_agent_id}")
        return True

    async def auto_discover_and_swap(self, context: AgentContext) -> Dict[str, Any]:
        """Automatically discover new agents and perform beneficial swaps."""
        if not self.auto_discovery_enabled:
            return {"auto_discovery_disabled": True}

        # Discover new agents
        await self.discover_agents(force_rediscovery=True)

        # Find swap opportunities
        swap_opportunities = await self._find_swap_opportunities(context)

        results = {
            "swaps_attempted": 0,
            "swaps_successful": 0,
            "opportunities_found": len(swap_opportunities),
        }

        if self.auto_swap_enabled:
            for opportunity in swap_opportunities:
                old_agent = opportunity["old_agent"]
                new_agent = opportunity["new_agent"]

                results["swaps_attempted"] += 1
                if await self.hot_swap_agent(old_agent, new_agent, context):
                    results["swaps_successful"] += 1

        return results

    async def optimize_composition(self, context: AgentContext) -> Dict[str, Any]:
        """Optimize the current agent composition using composition rules."""
        optimization_results: Dict[str, Any] = {
            "rules_evaluated": 0,
            "rules_applied": 0,
            "changes_made": [],
        }

        metadata = {
            "discovered_agents": self.discovered_agents,
            "loaded_agents": self.loaded_agents,
            "graph_stats": self.graph_engine.get_execution_statistics(),
        }

        for rule in self.composition_rules:
            optimization_results["rules_evaluated"] += 1

            if rule.evaluate(context, metadata):
                try:
                    changes = rule.apply(context, metadata)
                    if changes:
                        optimization_results["rules_applied"] += 1
                        optimization_results["changes_made"].append(
                            {"rule": rule.name, "changes": changes}
                        )

                        logger.info(f"Applied composition rule: {rule.name}")

                except Exception as e:
                    logger.error(f"Error applying rule {rule.name}: {e}")

        if optimization_results["changes_made"]:
            self._emit_event(
                CompositionEvent.COMPOSITION_OPTIMIZED, optimization_results
            )

        return optimization_results

    def get_composition_status(self) -> Dict[str, Any]:
        """Get comprehensive composition status."""
        # Calculate swap candidate statistics
        swap_candidate_stats = {}
        for agent_id, candidates in self.swap_candidates.items():
            swap_candidate_stats[agent_id] = {
                "candidate_count": len(candidates),
                "latest_version": (
                    max((c.version for c in candidates), default=None)
                    if candidates
                    else None
                ),
            }

        return {
            "discovered_agents": len(self.discovered_agents),
            "loaded_agents": len(self.loaded_agents),
            "composition_rules": len(self.composition_rules),
            "discoverers": len(self.discoverers),
            "auto_discovery_enabled": self.auto_discovery_enabled,
            "auto_swap_enabled": self.auto_swap_enabled,
            "discovery_stats": self.discovery_stats.copy(),
            "swap_candidates": swap_candidate_stats,
            "recent_events": self.composition_events[-10:],  # Last 10 events
            "swap_history_count": len(self.swap_history),
        }

    def enable_auto_discovery(self, interval_seconds: int = 300) -> None:
        """Enable automatic agent discovery."""
        self.auto_discovery_enabled = True
        logger.info(f"Enabled auto-discovery with {interval_seconds}s interval")

        # In a full implementation, this would start a background task
        # asyncio.create_task(self._auto_discovery_loop(interval_seconds))

    def enable_auto_swap(self) -> None:
        """Enable automatic agent swapping."""
        self.auto_swap_enabled = True
        logger.info("Enabled automatic agent swapping")

    async def _find_swap_opportunities(
        self, context: AgentContext
    ) -> List[Dict[str, Any]]:
        """Find opportunities for beneficial agent swaps."""
        opportunities = []

        for agent_id, metadata in self.discovered_agents.items():
            if agent_id in self.loaded_agents:
                # Look for newer versions or better alternatives
                for alt_id, alt_metadata in self.discovered_agents.items():
                    if (
                        alt_id != agent_id
                        and alt_metadata.can_replace(metadata)
                        and alt_metadata.version > metadata.version
                    ):
                        opportunities.append(
                            {
                                "old_agent": agent_id,
                                "new_agent": alt_id,
                                "reason": "version_upgrade",
                                "old_version": metadata.version,
                                "new_version": alt_metadata.version,
                            }
                        )

        return opportunities

    def _emit_event(self, event: CompositionEvent, data: Dict[str, Any]) -> None:
        """Emit a composition event."""
        event_record = {"event": event.value, "timestamp": time.time(), "data": data}

        self.composition_events.append(event_record)

        # Keep only recent events (last 100)
        if len(self.composition_events) > 100:
            self.composition_events = self.composition_events[-100:]

        # Call event handlers
        for handler in self.event_handlers[event]:
            try:
                handler(event_record)
            except Exception as e:
                logger.warning(f"Error in event handler for {event}: {e}")

    def _create_context_snapshot(self, context: AgentContext) -> Dict[str, Any]:
        """Create a snapshot of execution context."""
        return {
            "agent_outputs_count": len(context.agent_outputs),
            "query_length": len(context.query),
            "execution_state_keys": list(context.execution_state.keys()),
            "timestamp": time.time(),
        }


# Predefined composition rules
def create_version_upgrade_rule() -> CompositionRule:
    """Create a rule that swaps agents when newer versions are available."""

    def condition(context: AgentContext, metadata: Dict[str, Any]) -> bool:
        discovered = metadata["discovered_agents"]
        loaded = metadata["loaded_agents"]

        # Check if any loaded agent has a newer version available
        for agent_id in loaded:
            if agent_id in discovered:
                current_version = discovered[agent_id].version
                # Look for newer versions
                for alt_id, alt_metadata in discovered.items():
                    if (
                        alt_metadata.agent_id == agent_id
                        and alt_metadata.version > current_version
                    ):
                        return True
        return False

    def action(context: AgentContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"strategy": "version_upgrade", "trigger": "newer_version_available"}

    return CompositionRule(
        rule_id="version_upgrade",
        name="Version Upgrade Rule",
        condition=condition,
        action=action,
        priority=10,
        description="Swap agents when newer versions are discovered",
    )


def create_failure_recovery_rule() -> CompositionRule:
    """Create a rule that swaps failing agents."""

    def condition(context: AgentContext, metadata: Dict[str, Any]) -> bool:
        # Check if any agents have failed recently
        failed_agents = context.execution_state.get("failed_agents", [])
        return len(failed_agents) > 0

    def action(context: AgentContext, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"strategy": "failure_recovery", "trigger": "agent_failure"}

    return CompositionRule(
        rule_id="failure_recovery",
        name="Failure Recovery Rule",
        condition=condition,
        action=action,
        priority=20,
        description="Swap agents that have failed with alternatives",
    )
