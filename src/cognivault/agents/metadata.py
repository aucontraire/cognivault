"""
Unified Agent Metadata with Multi-Axis Classification.

This module consolidates agent metadata from registry.py and dynamic_composition.py
while adding the multi-axis classification system needed for event-driven architecture
and future utility agent integration.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Literal, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from cognivault.agents.base_agent import BaseAgent
from cognivault.exceptions import FailurePropagationStrategy


class DiscoveryStrategy(Enum):
    """Strategies for discovering agents."""

    FILESYSTEM = "filesystem"
    REGISTRY = "registry"
    NETWORK = "network"
    PLUGIN = "plugin"
    HYBRID = "hybrid"


@dataclass
class AgentMetadata:
    """
    Unified agent metadata with multi-axis classification.

    Combines functionality from registry-based agent management with
    dynamic discovery capabilities and adds multi-axis classification
    for intelligent event routing and service extraction.
    """

    # Core identification (from registry.py)
    name: str
    agent_class: Type["BaseAgent"]
    description: str = ""

    # Multi-axis classification (new for event-driven architecture)
    cognitive_speed: Literal["fast", "slow", "adaptive"] = "adaptive"
    cognitive_depth: Literal["shallow", "deep", "variable"] = "variable"
    processing_pattern: Literal["atomic", "composite", "chain"] = "atomic"
    primary_capability: str = ""  # "critical_analysis", "translation", "summarization"
    secondary_capabilities: List[str] = field(default_factory=list)
    pipeline_role: Literal["entry", "intermediate", "terminal", "standalone"] = (
        "standalone"
    )
    bounded_context: str = "reflection"  # "reflection", "transformation", "retrieval"

    # Runtime and execution (from registry.py)
    requires_llm: bool = False
    dependencies: List[str] = field(default_factory=list)
    is_critical: bool = True
    failure_strategy: FailurePropagationStrategy = FailurePropagationStrategy.FAIL_FAST
    fallback_agents: List[str] = field(default_factory=list)
    health_checks: List[str] = field(default_factory=list)

    # Capabilities and versioning (from dynamic_composition.py)
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)  # Technical capabilities
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    compatibility: Dict[str, str] = field(default_factory=dict)

    # Discovery metadata (from dynamic_composition.py)
    agent_id: Optional[str] = None
    module_path: Optional[str] = None
    discovered_at: float = field(default_factory=time.time)
    discovery_strategy: Optional[DiscoveryStrategy] = None
    file_path: Optional[Path] = None
    checksum: Optional[str] = None

    # Runtime state (from dynamic_composition.py)
    load_count: int = 0
    last_loaded: Optional[float] = None
    load_errors: List[str] = field(default_factory=list)
    is_loaded: bool = False

    def __post_init__(self):
        """Initialize derived fields and defaults."""
        # Set agent_id from name if not provided
        if self.agent_id is None:
            self.agent_id = self.name

        # Set module_path from agent_class if not provided
        if self.module_path is None and self.agent_class:
            self.module_path = (
                f"{self.agent_class.__module__}.{self.agent_class.__name__}"
            )

        # Derive primary_capability from name if not set
        if not self.primary_capability:
            self.primary_capability = self._derive_capability_from_name()

        # Initialize capabilities list if empty
        if not self.capabilities:
            self.capabilities = self._derive_capabilities()

    def _derive_capability_from_name(self) -> str:
        """Derive primary capability from agent name."""
        capability_map = {
            "refiner": "intent_clarification",
            "critic": "critical_analysis",
            "historian": "context_retrieval",
            "synthesis": "multi_perspective_synthesis",
            "translator": "translation",
            "summarizer": "summarization",
            "formatter": "output_formatting",
        }
        return capability_map.get(self.name.lower(), self.name.lower())

    def _derive_capabilities(self) -> List[str]:
        """Derive technical capabilities list from agent characteristics."""
        caps = [self.primary_capability]

        # Add capabilities based on agent characteristics
        if self.requires_llm:
            caps.append("llm_integration")

        if self.processing_pattern == "composite":
            caps.append("multi_step_processing")

        if self.pipeline_role == "entry":
            caps.append("input_processing")
        elif self.pipeline_role == "terminal":
            caps.append("output_generation")

        return caps

    def can_replace(self, other: "AgentMetadata") -> bool:
        """
        Check if this agent can replace another agent.

        Uses both capability matching and version compatibility.
        """
        if self.agent_id != other.agent_id:
            return False

        # Check version compatibility
        if "min_version" in other.compatibility:
            min_version = other.compatibility["min_version"]
            if self.version < min_version:
                return False

        # Check primary capability match
        if self.primary_capability != other.primary_capability:
            return False

        # Check that all required capabilities are present
        for required_cap in other.capabilities:
            if required_cap not in self.capabilities:
                return False

        return True

    def is_compatible_with_task(
        self, task_type: str, domain: Optional[str] = None
    ) -> bool:
        """
        Check if this agent is compatible with a specific task type.

        Args:
            task_type: Type of task ("transform", "evaluate", "retrieve", etc.)
            domain: Optional domain specialization

        Returns:
            True if agent can handle this task type
        """
        # Map task types to capabilities
        task_capability_map = {
            "transform": ["translation", "summarization", "output_formatting"],
            "evaluate": [
                "critical_analysis",
                "bias_detection",
                "assumption_identification",
            ],
            "retrieve": ["context_retrieval", "memory_search", "information_gathering"],
            "synthesize": ["multi_perspective_synthesis", "conflict_resolution"],
            "clarify": ["intent_clarification", "prompt_structuring"],
            "format": ["output_formatting", "structure_generation"],
        }

        compatible_capabilities = task_capability_map.get(task_type, [])

        # Check if agent has any compatible capability
        agent_capabilities = [self.primary_capability] + self.secondary_capabilities
        return any(cap in agent_capabilities for cap in compatible_capabilities)

    def get_performance_tier(self) -> str:
        """
        Get performance tier based on cognitive characteristics.

        Returns:
            Performance tier: "fast", "balanced", "thorough"
        """
        if self.cognitive_speed == "fast" and self.cognitive_depth == "shallow":
            return "fast"
        elif self.cognitive_speed == "slow" and self.cognitive_depth == "deep":
            return "thorough"
        else:
            return "balanced"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            # Core identification
            "name": self.name,
            "agent_class": f"{self.agent_class.__module__}.{self.agent_class.__name__}",
            "description": self.description,
            # Multi-axis classification
            "cognitive_speed": self.cognitive_speed,
            "cognitive_depth": self.cognitive_depth,
            "processing_pattern": self.processing_pattern,
            "primary_capability": self.primary_capability,
            "secondary_capabilities": self.secondary_capabilities,
            "pipeline_role": self.pipeline_role,
            "bounded_context": self.bounded_context,
            # Runtime and execution
            "requires_llm": self.requires_llm,
            "dependencies": self.dependencies,
            "is_critical": self.is_critical,
            "failure_strategy": self.failure_strategy.value,
            "fallback_agents": self.fallback_agents,
            "health_checks": self.health_checks,
            # Capabilities and versioning
            "version": self.version,
            "capabilities": self.capabilities,
            "resource_requirements": self.resource_requirements,
            "compatibility": self.compatibility,
            # Discovery metadata
            "agent_id": self.agent_id,
            "module_path": self.module_path,
            "discovered_at": self.discovered_at,
            "discovery_strategy": (
                self.discovery_strategy.value if self.discovery_strategy else None
            ),
            "file_path": str(self.file_path) if self.file_path else None,
            "checksum": self.checksum,
            # Runtime state
            "load_count": self.load_count,
            "last_loaded": self.last_loaded,
            "is_loaded": self.is_loaded,
            "load_errors": self.load_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMetadata":
        """Create AgentMetadata from dictionary representation."""
        # Handle agent_class reconstruction (simplified for now)
        agent_class_path = data.get("agent_class", "")
        # In production, this would use importlib to reconstruct the class
        # For now, we'll use a placeholder
        # Import BaseAgent at runtime to avoid circular import
        import importlib

        try:
            base_agent_module = importlib.import_module("cognivault.agents.base_agent")
            agent_class = base_agent_module.BaseAgent  # Placeholder
        except ImportError:
            agent_class = None  # type: ignore

        # Handle enum reconstruction
        discovery_strategy = None
        if data.get("discovery_strategy"):
            discovery_strategy = DiscoveryStrategy(data["discovery_strategy"])

        failure_strategy = FailurePropagationStrategy.FAIL_FAST
        if data.get("failure_strategy"):
            failure_strategy = FailurePropagationStrategy(data["failure_strategy"])

        return cls(
            name=data["name"],
            agent_class=agent_class,
            description=data.get("description", ""),
            cognitive_speed=data.get("cognitive_speed", "adaptive"),
            cognitive_depth=data.get("cognitive_depth", "variable"),
            processing_pattern=data.get("processing_pattern", "atomic"),
            primary_capability=data.get("primary_capability", ""),
            secondary_capabilities=data.get("secondary_capabilities", []),
            pipeline_role=data.get("pipeline_role", "standalone"),
            bounded_context=data.get("bounded_context", "reflection"),
            requires_llm=data.get("requires_llm", False),
            dependencies=data.get("dependencies", []),
            is_critical=data.get("is_critical", True),
            failure_strategy=failure_strategy,
            fallback_agents=data.get("fallback_agents", []),
            health_checks=data.get("health_checks", []),
            version=data.get("version", "1.0.0"),
            capabilities=data.get("capabilities", []),
            resource_requirements=data.get("resource_requirements", {}),
            compatibility=data.get("compatibility", {}),
            agent_id=data.get("agent_id"),
            module_path=data.get("module_path"),
            discovered_at=data.get("discovered_at", time.time()),
            discovery_strategy=discovery_strategy,
            file_path=Path(data["file_path"]) if data.get("file_path") else None,
            checksum=data.get("checksum"),
            load_count=data.get("load_count", 0),
            last_loaded=data.get("last_loaded"),
            is_loaded=data.get("is_loaded", False),
            load_errors=data.get("load_errors", []),
        )


@dataclass
class TaskClassification:
    """
    Classification of work being performed for semantic event routing.

    This enables intelligent routing based on work intent rather than
    just agent pipeline position.
    """

    task_type: Literal[
        "transform",  # Data/format transformation
        "evaluate",  # Critical analysis and assessment
        "retrieve",  # Information and context retrieval
        "synthesize",  # Multi-perspective integration
        "summarize",  # Content condensation
        "format",  # Output formatting and structuring
        "filter",  # Content filtering and selection
        "rank",  # Prioritization and ordering
        "compare",  # Comparative analysis
        "explain",  # Explanatory and educational content
        "clarify",  # Intent clarification and refinement
    ]

    domain: Optional[str] = None  # "economics", "code", "policy", "medical"
    intent: Optional[str] = (
        None  # "help me decide", "convert to JSON", "explain concepts"
    )
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    urgency: Literal["low", "normal", "high"] = "normal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_type": self.task_type,
            "domain": self.domain,
            "intent": self.intent,
            "complexity": self.complexity,
            "urgency": self.urgency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskClassification":
        """Create TaskClassification from dictionary."""
        return cls(
            task_type=data["task_type"],
            domain=data.get("domain"),
            intent=data.get("intent"),
            complexity=data.get("complexity", "moderate"),
            urgency=data.get("urgency", "normal"),
        )


def classify_query_task(query: str) -> TaskClassification:
    """
    Classify a user query into task type and characteristics.

    This is a simplified rule-based implementation. In production,
    this could use LLM-powered classification or ML models.

    Args:
        query: User query to classify

    Returns:
        TaskClassification with inferred task type and characteristics
    """
    query_lower = query.lower()

    # Simple keyword-based classification
    task_type: Literal[
        "transform",
        "evaluate",
        "retrieve",
        "synthesize",
        "summarize",
        "format",
        "filter",
        "rank",
        "compare",
        "explain",
        "clarify",
    ]
    if any(word in query_lower for word in ["translate", "convert", "transform"]):
        task_type = "transform"
    elif any(
        word in query_lower for word in ["analyze", "evaluate", "critique", "assess"]
    ):
        task_type = "evaluate"
    elif any(word in query_lower for word in ["find", "search", "retrieve", "lookup"]):
        task_type = "retrieve"
    elif any(
        word in query_lower for word in ["combine", "synthesize", "merge", "integrate"]
    ):
        task_type = "synthesize"
    elif any(word in query_lower for word in ["summarize", "condense", "shorten"]):
        task_type = "summarize"
    elif any(word in query_lower for word in ["format", "structure", "organize"]):
        task_type = "format"
    elif any(
        word in query_lower for word in ["explain", "clarify", "help me understand"]
    ):
        task_type = "explain"
    else:
        task_type = "evaluate"  # Default to evaluation for complex queries

    # Determine complexity based on query length and keywords
    complexity: Literal["simple", "moderate", "complex"] = "simple"
    if len(query) > 100:
        complexity = "moderate"
    if len(query) > 300 or any(
        word in query_lower for word in ["complex", "detailed", "comprehensive"]
    ):
        complexity = "complex"

    # Determine urgency based on keywords
    urgency: Literal["low", "normal", "high"] = "normal"
    if any(
        word in query_lower for word in ["urgent", "asap", "quickly", "immediately"]
    ):
        urgency = "high"
    elif any(
        word in query_lower for word in ["when convenient", "no rush", "eventually"]
    ):
        urgency = "low"

    # Extract domain hints
    domain = None
    domain_keywords = {
        "economics": ["economic", "financial", "market", "trade", "economy"],
        "code": ["code", "programming", "software", "function", "class"],
        "policy": ["policy", "government", "regulation", "law", "legal"],
        "medical": ["medical", "health", "disease", "treatment", "clinical"],
        "science": ["research", "study", "experiment", "hypothesis", "data"],
    }

    for domain_name, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            domain = domain_name
            break

    return TaskClassification(
        task_type=task_type,
        domain=domain,
        intent=query[:50] + "..." if len(query) > 50 else query,
        complexity=complexity,
        urgency=urgency,
    )
