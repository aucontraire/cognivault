"""
Enhanced frontmatter schema for CogniVault notes.

This module provides comprehensive frontmatter structures for rich metadata
including topics, agent status tracking, domain classification, and more.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class AgentStatus(Enum):
    """Status values for individual agent processing."""

    # Refiner Agent Statuses
    REFINED = "refined"  # Successfully refined the query
    PASSTHROUGH = "passthrough"  # Passed query through unchanged
    FAILED = "failed"  # Processing failed

    # Critic Agent Statuses
    ANALYZED = "analyzed"  # Successfully analyzed content
    INSUFFICIENT_CONTENT = "insufficient_content"  # Not enough content to critique
    SKIPPED = "skipped"  # Skipped due to configuration

    # Historian Agent Statuses
    FOUND_MATCHES = "found_matches"  # Found relevant historical content
    NO_MATCHES = "no_matches"  # No relevant content found
    SEARCH_FAILED = "search_failed"  # Search operation failed

    # Synthesis Agent Statuses
    INTEGRATED = "integrated"  # Successfully synthesized all inputs
    PARTIAL = "partial"  # Partial synthesis due to missing inputs
    CONFLICTS_UNRESOLVED = "conflicts_unresolved"  # Unable to resolve conflicts


class DifficultyLevel(Enum):
    """Difficulty levels for content classification."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ConfidenceLevel(Enum):
    """Confidence levels for agent outputs."""

    LOW = "low"  # 0.0 - 0.4
    MODERATE = "moderate"  # 0.4 - 0.7
    HIGH = "high"  # 0.7 - 0.9
    VERY_HIGH = "very_high"  # 0.9 - 1.0


class AgentExecutionResult(BaseModel):
    """Detailed execution result for a single agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: AgentStatus
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_level: Optional[ConfidenceLevel] = None
    processing_time_ms: Optional[int] = None
    changes_made: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate confidence level from numeric confidence."""
        if self.confidence_level is None:
            if self.confidence < 0.4:
                self.confidence_level = ConfidenceLevel.LOW
            elif self.confidence < 0.7:
                self.confidence_level = ConfidenceLevel.MODERATE
            elif self.confidence < 0.9:
                self.confidence_level = ConfidenceLevel.HIGH
            else:
                self.confidence_level = ConfidenceLevel.VERY_HIGH


class EnhancedFrontmatter(BaseModel):
    """
    Enhanced frontmatter schema for CogniVault notes.

    Provides comprehensive metadata including topics, agent status tracking,
    domain classification, and relationship mapping.
    """

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, validate_assignment=True
    )

    # Core Metadata (existing fields)
    title: str
    date: str  # ISO format timestamp
    filename: str
    source: str = "cli"
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Agent Execution Results
    agents: Dict[str, AgentExecutionResult] = Field(default_factory=dict)

    # Topic and Classification
    topics: List[str] = Field(default_factory=list)
    domain: Optional[str] = None  # Primary domain (e.g., "psychology", "technology")
    subdomain: Optional[str] = None  # More specific classification
    difficulty: Optional[DifficultyLevel] = None

    # Content Relationships
    related_queries: List[str] = Field(default_factory=list)
    related_notes: List[str] = Field(default_factory=list)  # UUIDs or filenames
    parent_topics: List[str] = Field(default_factory=list)  # Broader topics
    child_topics: List[str] = Field(default_factory=list)  # More specific topics

    # Quality and Processing Metadata
    summary: str = "Generated response from CogniVault agents"
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    completeness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    synthesis_quality: Optional[str] = None  # "high", "moderate", "low"

    # Extended Metadata
    language: str = "en"
    word_count: Optional[int] = None
    reading_time_minutes: Optional[int] = None
    last_updated: Optional[str] = None
    version: int = 1

    # Future Extensions
    external_sources: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)  # Free-form tags vs structured topics

    def add_agent_result(self, agent_name: str, result: AgentExecutionResult) -> None:
        """Add or update an agent execution result."""
        self.agents[agent_name] = result

    def add_topic(self, topic: str) -> None:
        """Add a topic if not already present."""
        if topic not in self.topics:
            self.topics.append(topic)

    def add_related_query(self, query: str) -> None:
        """Add a related query if not already present."""
        if query not in self.related_queries:
            self.related_queries.append(query)

    def update_last_modified(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now().isoformat()

    def calculate_reading_time(self, text_content: str) -> None:
        """Calculate and set reading time based on word count."""
        words = len(text_content.split())
        self.word_count = words
        # Average reading speed: 200-250 words per minute
        self.reading_time_minutes = max(1, words // 225)


def create_basic_frontmatter(
    title: str,
    agent_outputs: Dict[str, str],
    timestamp: Optional[str] = None,
    filename: Optional[str] = None,
) -> EnhancedFrontmatter:
    """
    Create basic frontmatter for backward compatibility.

    This provides a migration path from the old simple frontmatter
    to the new enhanced schema.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    if filename is None:
        filename = f"{timestamp.replace(':', '-')}-generated.md"

    frontmatter = EnhancedFrontmatter(
        title=title,
        date=timestamp,
        filename=filename,
    )

    # Add basic agent results
    for agent_name in agent_outputs.keys():
        result = AgentExecutionResult(
            status=AgentStatus.INTEGRATED,  # Default status
            confidence=0.8,  # Default confidence
            changes_made=True,
        )
        frontmatter.add_agent_result(agent_name, result)

    return frontmatter


def frontmatter_to_yaml_dict(frontmatter: EnhancedFrontmatter) -> Dict[str, Any]:
    """
    Convert EnhancedFrontmatter to a YAML-serializable dictionary.

    This handles enum conversion and nested structures for YAML export.
    """
    data = frontmatter.model_dump()

    # Convert agent results to serializable format
    if "agents" in data:
        converted_agents = {}
        for agent_name, result_data in data["agents"].items():
            if isinstance(result_data, dict):
                # Handle already dumped agent result data
                converted_result = {}
                for key, value in result_data.items():
                    if key == "status" and hasattr(value, "value"):
                        converted_result[key] = value.value
                    elif (
                        key == "confidence_level" and value and hasattr(value, "value")
                    ):
                        converted_result[key] = value.value
                    elif value is not None:
                        converted_result[key] = value
                converted_agents[agent_name] = converted_result
            else:
                # Handle raw AgentExecutionResult objects (shouldn't happen with model_dump)
                converted_agents[agent_name] = result_data
        data["agents"] = converted_agents

    # Convert enums to values
    if "difficulty" in data and data["difficulty"]:
        data["difficulty"] = (
            data["difficulty"].value
            if hasattr(data["difficulty"], "value")
            else data["difficulty"]
        )

    # Remove None values and empty lists/dicts for cleaner output
    return {k: v for k, v in data.items() if v is not None and v != [] and v != {}}


# Topic taxonomy helpers
class TopicTaxonomy:
    """Predefined topic taxonomy for consistent tagging."""

    DOMAINS = {
        "technology": ["ai", "machine_learning", "programming", "software", "hardware"],
        "psychology": [
            "cognitive_science",
            "behavior",
            "neuroscience",
            "mental_health",
        ],
        "philosophy": ["ethics", "logic", "metaphysics", "epistemology"],
        "science": ["physics", "chemistry", "biology", "mathematics"],
        "society": [
            "politics",
            "economics",
            "culture",
            "education",
            "democracy",
            "government",
            "elections",
            "policy",
            "voting",
            "citizenship",
            "political",
        ],
        "business": ["management", "strategy", "marketing", "finance"],
        "health": ["medicine", "nutrition", "fitness", "wellness"],
        "creative": ["art", "music", "writing", "design"],
    }

    @classmethod
    def suggest_domain(cls, topics: List[str]) -> Optional[str]:
        """Suggest primary domain based on topics."""
        domain_scores: Dict[str, int] = {}
        for topic in topics:
            for domain, subtopics in cls.DOMAINS.items():
                if topic.lower() in subtopics:
                    domain_scores[domain] = domain_scores.get(domain, 0) + 1

        if domain_scores:
            return max(domain_scores, key=lambda x: domain_scores[x])
        return None

    @classmethod
    def get_related_topics(cls, topic: str) -> List[str]:
        """Get related topics within the same domain."""
        topic_lower = topic.lower()
        for domain, subtopics in cls.DOMAINS.items():
            if topic_lower in subtopics:
                return [t for t in subtopics if t != topic_lower]
        return []
