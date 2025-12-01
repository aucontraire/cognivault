"""
Pydantic models for structured agent outputs.

This module defines the data structures that agents return when using Pydantic AI
for structured response validation. These models ensure consistent data shapes
in the execution_metadata JSONB field while maintaining agent swapping flexibility.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union, cast
from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing_extensions import Self


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent outputs."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BiasType(str, Enum):
    """Types of biases that can be identified by the Critic agent."""

    TEMPORAL = "temporal"
    CULTURAL = "cultural"
    METHODOLOGICAL = "methodological"
    SCALE = "scale"
    CONFIRMATION = "confirmation"
    AVAILABILITY = "availability"
    ANCHORING = "anchoring"


class BiasDetail(BaseModel):
    """Structured detail about a specific bias identified by the Critic agent.

    This model replaces the Dict[str, str] bias_details field to ensure
    compatibility with OpenAI's structured output API, which requires
    all fields to have predefined schemas (additionalProperties: false).
    """

    bias_type: BiasType = Field(..., description="Type of bias identified")
    explanation: str = Field(
        ...,
        min_length=10,
        max_length=200,
        description="Detailed explanation of how this bias manifests in the query",
    )


class ProcessingMode(str, Enum):
    """Processing modes for agents."""

    ACTIVE = "active"
    PASSIVE = "passive"
    FALLBACK = "fallback"


class BaseAgentOutput(BaseModel):
    """Base class for all agent outputs with common metadata."""

    agent_name: str = Field(
        ..., description="Name of the agent that produced this output"
    )
    processing_mode: ProcessingMode = Field(..., description="Mode used for processing")
    confidence: ConfidenceLevel = Field(
        ..., description="Confidence level of the output"
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the output was generated (ISO format)",
    )

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, use_enum_values=True
    )


class RefinerOutput(BaseAgentOutput):
    """Structured output from the Refiner agent."""

    refined_query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The refined and clarified query - content only, no meta-commentary",
    )
    original_query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The original input query as received",
    )
    changes_made: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="List of specific changes made to improve the query",
    )
    was_unchanged: bool = Field(
        default=False,
        description="True if query was returned unchanged with [Unchanged] tag",
    )
    fallback_used: bool = Field(
        default=False, description="True if fallback mode was used for malformed input"
    )
    ambiguities_resolved: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="List of ambiguities that were resolved",
    )

    @field_validator("refined_query")
    @classmethod
    def validate_no_meta_commentary(cls, v: str) -> str:
        """
        Prevent content pollution by ensuring refined_query contains only content.

        Based on LangChain article patterns for field validation.
        """
        # Meta-commentary phrases that indicate pollution
        pollution_markers = [
            "I refined",
            "I changed",
            "I modified",
            "I updated",
            "I improved",
            "The query was",
            "After analysis",
            "Upon review",
            "Changes made:",
            "To clarify",
            "To improve",
            "I suggest",
            "I recommend",
            "This is better because",
            "The refined version",
        ]

        v_lower = v.lower()
        for marker in pollution_markers:
            if marker.lower() in v_lower:
                raise ValueError(
                    f"Content pollution detected: refined_query contains meta-commentary '{marker}'. "
                    f"Only the refined content should be included, not commentary about the refinement."
                )

        return v.strip()

    @field_validator("changes_made")
    @classmethod
    def validate_changes_format(cls, v: List[str]) -> List[str]:
        """Ensure changes are concise and properly formatted.

        Character limit increased from 100 → 150 chars (2025-01-26) to accommodate
        natural LLM language patterns when describing query refinements.
        """
        if not v:
            return v

        validated_changes = []
        for change in v:
            change = change.strip()
            if len(change) < 5:
                raise ValueError(f"Change description too short: '{change}'")
            if len(change) > 150:  # Increased to accommodate LLM natural language
                raise ValueError(
                    f"Change description too long (max 150 chars): '{change[:50]}...'"
                )
            validated_changes.append(change)

        return validated_changes

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "agent_name": "refiner",
                "processing_mode": "active",
                "confidence": "high",
                "refined_query": "What are the potential positive and negative impacts of artificial intelligence on social structures, employment, and human relationships over the next decade?",
                "original_query": "What about AI and society?",
                "changes_made": [
                    "Clarified scope to include positive and negative impacts",
                    "Specified timeframe as next decade",
                    "Added specific domains: social structures, employment, relationships",
                ],
                "was_unchanged": False,
                "fallback_used": False,
                "ambiguities_resolved": ["Unclear scope of 'AI and society'"],
            }
        },
    )


class CriticOutput(BaseAgentOutput):
    """Structured output from the Critic agent."""

    assumptions: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Implicit assumptions identified in the query - analytical content only",
    )
    logical_gaps: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Logical gaps or under-specified concepts - analytical content only",
    )
    biases: List[BiasType] = Field(
        default_factory=list,
        max_length=7,
        description="Types of biases identified in the framing",
    )
    bias_details: List[BiasDetail] = Field(
        default_factory=list,
        max_length=7,
        description="Detailed explanations for each bias type identified - structured for OpenAI compatibility",
    )
    alternate_framings: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Suggested alternate ways to frame the query - content only",
    )
    critique_summary: str = Field(
        ...,
        min_length=20,
        max_length=300,
        description="Overall critique summary - analytical content only, no process description",
    )
    issues_detected: int = Field(
        ..., ge=0, le=100, description="Number of issues detected"
    )
    no_issues_found: bool = Field(
        default=False, description="True if query is well-scoped and neutral"
    )

    @field_validator("critique_summary")
    @classmethod
    def validate_critique_content_only(cls, v: str) -> str:
        """
        Prevent content pollution in critique summary.

        Based on LangChain article validation patterns.
        """
        # Process-oriented phrases that indicate meta-commentary pollution
        process_markers = [
            "I analyzed",
            "I found",
            "I identified",
            "I discovered",
            "I noticed",
            "My analysis shows",
            "Upon examination",
            "After reviewing",
            "The analysis reveals",
            "I conclude",
            "My assessment",
            "Processing this query",
            "In my evaluation",
        ]

        v_lower = v.lower()
        for marker in process_markers:
            if marker.lower() in v_lower:
                raise ValueError(
                    f"Content pollution detected: critique_summary contains process description '{marker}'. "
                    f"Only analytical insights should be included, not descriptions of the analysis process."
                )

        return v.strip()

    @field_validator("assumptions", "logical_gaps", "alternate_framings")
    @classmethod
    def validate_analysis_items(cls, v: List[str]) -> List[str]:
        """Validate analysis items contain content only, not process descriptions."""
        if not v:
            return v

        validated_items = []
        for item in v:
            item = item.strip()
            if len(item) < 10:
                raise ValueError(f"Analysis item too short: '{item}'")
            if len(item) > 250:
                raise ValueError(
                    f"Analysis item too long (max 250 chars): '{item[:50]}...'"
                )

            # Check for process pollution
            if any(
                phrase in item.lower()
                for phrase in ["i found", "i noticed", "my analysis"]
            ):
                raise ValueError(
                    f"Process description in analysis item: '{item[:30]}...'"
                )

            validated_items.append(item)

        return validated_items

    @model_validator(mode="after")
    def validate_issues_consistency(self) -> Self:
        """Ensure issues_detected count is within reasonable bounds.

        NOTE: We do NOT enforce strict matching with actual array lengths because:
        1. LLM counts semantically (conceptual issues) vs mechanically (array items)
        2. Some issues may be mentioned in summary but not detailed
        3. Some array items may combine multiple related issues

        We only validate the count is reasonable (0-100 range).
        """
        # Validate reasonable range instead of strict matching
        if self.issues_detected < 0 or self.issues_detected > 100:
            raise ValueError(
                f"issues_detected out of reasonable range (0-100): {self.issues_detected}"
            )

        return self

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "agent_name": "critic",
                "processing_mode": "active",
                "confidence": "medium",
                "assumptions": ["Presumes AI will have significant social impact"],
                "logical_gaps": ["No definition of 'societal impacts' scope"],
                "biases": ["temporal"],
                "bias_details": [
                    {
                        "bias_type": "temporal",
                        "explanation": "Assumes current AI trajectory will continue without considering potential disruptions",
                    }
                ],
                "alternate_framings": [
                    "Consider both positive and negative impacts separately"
                ],
                "critique_summary": "Query assumes AI impact without specifying direction or scope",
                "issues_detected": 3,
                "no_issues_found": False,
            }
        },
    )


class HistoricalReference(BaseModel):
    """Reference to a historical document or context."""

    source_id: Optional[str] = Field(
        None, description="ID, filename, or URL of the source document"
    )
    title: Optional[str] = Field(None, description="Title of the historical source")
    relevance_score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    content_snippet: str = Field(..., description="Relevant snippet from the source")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class HistorianOutput(BaseAgentOutput):
    """Structured output from the Historian agent."""

    relevant_sources: List[HistoricalReference] = Field(
        default_factory=list,
        max_length=20,
        description="List of relevant historical sources found",
    )
    historical_synthesis: str = Field(
        ...,
        min_length=50,
        max_length=5000,  # Increased to accommodate comprehensive LLM synthesis
        description="Synthesized historical context - content only, no process description",
    )
    themes_identified: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Historical themes relevant to the query - themes only, no analysis process",
    )
    time_periods_covered: List[str] = Field(
        default_factory=list,
        max_length=8,
        description="Time periods covered in the historical context",
    )
    contextual_connections: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Connections between historical context and current query - insights only",
    )
    sources_searched: int = Field(
        ..., ge=0, le=1000, description="Number of sources searched"
    )
    relevant_sources_found: int = Field(
        ..., ge=0, le=100, description="Number of relevant sources found"
    )
    no_relevant_context: bool = Field(
        default=False, description="True if no relevant historical context was found"
    )

    @field_validator("historical_synthesis")
    @classmethod
    def validate_historical_content_only(cls, v: str) -> str:
        """
        Prevent content pollution in historical synthesis.

        Based on LangChain article validation patterns for content purity.
        """
        # Process-oriented phrases that indicate meta-commentary pollution
        process_markers = [
            "I searched",
            "I found",
            "I analyzed",
            "I discovered",
            "I examined",
            "My research shows",
            "My analysis reveals",
            "After searching",
            "Upon investigation",
            "I conclude",
            "My findings suggest",
            "Database search revealed",
            "In my historical analysis",
            "Processing historical data",
            "Through my research",
        ]

        v_lower = v.lower()
        for marker in process_markers:
            if marker.lower() in v_lower:
                raise ValueError(
                    f"Content pollution detected: historical_synthesis contains process description '{marker}'. "
                    f"Only historical insights and context should be included, not descriptions of the research process."
                )

        return v.strip()

    @field_validator("themes_identified", "contextual_connections")
    @classmethod
    def validate_historical_items(cls, v: List[str]) -> List[str]:
        """Validate historical items contain insights only, not process descriptions."""
        if not v:
            return v

        validated_items = []
        for item in v:
            item = item.strip()
            if len(item) < 5:
                raise ValueError(f"Historical item too short: '{item}'")
            if len(item) > 200:
                raise ValueError(
                    f"Historical item too long (max 200 chars): '{item[:50]}...'"
                )

            # Check for process pollution
            if any(
                phrase in item.lower()
                for phrase in [
                    "i found",
                    "my research",
                    "database shows",
                    "search results",
                ]
            ):
                raise ValueError(
                    f"Process description in historical item: '{item[:30]}...'"
                )

            validated_items.append(item)

        return validated_items

    @model_validator(mode="after")
    def validate_source_counts(self) -> Self:
        """Ensure source counts are consistent."""
        if self.relevant_sources_found > self.sources_searched:
            raise ValueError(
                f"Inconsistent source counts: relevant_sources_found={self.relevant_sources_found} "
                f"cannot exceed sources_searched={self.sources_searched}"
            )

        # Check consistency with actual relevant_sources list
        actual_sources = len(self.relevant_sources)
        if abs(self.relevant_sources_found - actual_sources) > 1:
            raise ValueError(
                f"Inconsistent source list: relevant_sources_found={self.relevant_sources_found}, "
                f"actual relevant_sources list has {actual_sources} items"
            )

        return self

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "agent_name": "historian",
                "processing_mode": "active",
                "confidence": "high",
                "historical_synthesis": "Historical analysis shows recurring patterns in technology adoption...",
                "themes_identified": [
                    "Technology adoption cycles",
                    "Social resistance to change",
                ],
                "time_periods_covered": ["Industrial Revolution", "Digital Revolution"],
                "contextual_connections": [
                    "Similar patterns of initial resistance followed by widespread adoption"
                ],
                "sources_searched": 15,
                "relevant_sources_found": 5,
                "no_relevant_context": False,
            }
        },
    )


class SynthesisTheme(BaseModel):
    """A synthesized theme with supporting evidence."""

    theme_name: str = Field(..., description="Name of the synthesized theme")
    description: str = Field(..., description="Detailed description of the theme")
    supporting_agents: List[str] = Field(
        ..., description="Agents that contributed to this theme"
    )
    confidence: ConfidenceLevel = Field(..., description="Confidence in this theme")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SynthesisOutput(BaseAgentOutput):
    """Structured output from the Synthesis agent."""

    final_synthesis: str = Field(
        ...,
        min_length=100,
        max_length=5000,
        description="Final synthesized wiki content - pure content only, no synthesis process description",
    )
    key_themes: List[SynthesisTheme] = Field(
        default_factory=list,
        max_length=8,
        description="Key themes identified across all agent outputs",
    )
    conflicts_resolved: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Conflicts between agents that were resolved - resolution content only",
    )
    complementary_insights: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Insights that build on each other across agents - insights only, no process",
    )
    knowledge_gaps: List[str] = Field(
        default_factory=list,
        max_length=8,
        description="Important aspects not covered by any agent - gaps only, no analysis process",
    )
    meta_insights: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Higher-level insights about the analysis process - insights only",
    )
    contributing_agents: List[str] = Field(
        ..., max_length=10, description="List of agents that contributed to synthesis"
    )
    word_count: int = Field(
        ..., ge=50, le=10000, description="Word count of the final synthesis"
    )
    topics_extracted: List[str] = Field(
        default_factory=list,
        max_length=30,
        description="Key topics/concepts mentioned in synthesis (up to 30 topics)",
    )

    @field_validator("final_synthesis")
    @classmethod
    def validate_synthesis_content_only(cls, v: str) -> str:
        """
        Prevent content pollution in final synthesis.

        Based on LangChain article validation patterns for content purity.
        """
        # Process-oriented phrases that indicate meta-commentary pollution
        process_markers = [
            "I synthesized",
            "I combined",
            "I analyzed",
            "I integrated",
            "I found",
            "My analysis shows",
            "My synthesis reveals",
            "After analyzing",
            "Upon examination",
            "I conclude",
            "My assessment shows",
            "Processing all inputs",
            "Synthesis process shows",
            "Integration reveals",
            "Combining the outputs",
            "Analysis of agent outputs",
            "My final synthesis",
        ]

        v_lower = v.lower()
        for marker in process_markers:
            if marker.lower() in v_lower:
                raise ValueError(
                    f"Content pollution detected: final_synthesis contains process description '{marker}'. "
                    f"Only the synthesized content should be included, not descriptions of the synthesis process."
                )

        return v.strip()

    @field_validator(
        "conflicts_resolved",
        "complementary_insights",
        "knowledge_gaps",
        "meta_insights",
    )
    @classmethod
    def validate_synthesis_items(cls, v: List[str]) -> List[str]:
        """Validate synthesis items contain content only, not process descriptions."""
        if not v:
            return v

        validated_items = []
        for item in v:
            item = item.strip()
            if len(item) < 8:
                raise ValueError(f"Synthesis item too short: '{item}'")
            if len(item) > 500:
                raise ValueError(
                    f"Synthesis item too long (max 500 chars): '{item[:50]}...'"
                )

            # Check for process pollution
            if any(
                phrase in item.lower()
                for phrase in [
                    "i found",
                    "my analysis",
                    "synthesis shows",
                    "combining outputs",
                ]
            ):
                raise ValueError(
                    f"Process description in synthesis item: '{item[:30]}...'"
                )

            validated_items.append(item)

        return validated_items

    @field_validator("topics_extracted")
    @classmethod
    def validate_topics_format(cls, v: List[str]) -> List[str]:
        """Validate and normalize topic formats.

        Character limit increased from 50 → 100 chars (2025-01-27) to accommodate
        LLM natural language topic extraction. Topics exceeding 100 chars are
        truncated rather than rejected to prevent validation retry cascades.
        """
        if not v:
            return v

        validated_topics = []
        for topic in v:
            topic = topic.strip().lower()
            if len(topic) < 2:
                raise ValueError(f"Topic too short: '{topic}'")

            # Truncate instead of rejecting to avoid retry cascade
            if len(topic) > 100:
                topic = topic[:97] + "..."

            # Check for process pollution in topics
            if any(
                phrase in topic
                for phrase in ["analysis of", "synthesis of", "processing", "combining"]
            ):
                raise ValueError(f"Process description in topic: '{topic}'")

            validated_topics.append(topic)

        return validated_topics

    @model_validator(mode="after")
    def validate_synthesis_consistency(self) -> Self:
        """Ensure synthesis consistency and completeness."""
        # Validate word count matches actual content
        actual_word_count = len(self.final_synthesis.split())
        if abs(self.word_count - actual_word_count) > 50:  # Allow some tolerance
            raise ValueError(
                f"Inconsistent word count: reported={self.word_count}, "
                f"actual={actual_word_count}"
            )

        # Ensure contributing agents list is not empty
        if not self.contributing_agents:
            raise ValueError("contributing_agents cannot be empty")

        # Validate known agent names
        valid_agents = {"refiner", "critic", "historian", "synthesis"}
        invalid_agents = [
            agent
            for agent in self.contributing_agents
            if agent.lower() not in valid_agents
        ]
        if invalid_agents:
            raise ValueError(
                f"Invalid agent names: {invalid_agents}. Valid agents: {valid_agents}"
            )

        return self

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "agent_name": "synthesis",
                "processing_mode": "active",
                "confidence": "high",
                "final_synthesis": "# Artificial Intelligence and Society\\n\\nArtificial intelligence represents...",
                "key_themes": [
                    {
                        "theme_name": "Social Transformation",
                        "description": "AI's role in changing social structures",
                        "supporting_agents": ["refiner", "critic", "historian"],
                        "confidence": "high",
                    }
                ],
                "conflicts_resolved": [
                    "Disagreement on timeline between historian and critic"
                ],
                "contributing_agents": ["refiner", "critic", "historian"],
                "word_count": 450,
                "topics_extracted": [
                    "artificial intelligence",
                    "employment",
                    "social structures",
                ],
            }
        },
    )


class DatabaseExecutionMetadata(BaseModel):
    """Complete execution metadata structure for JSONB database storage.

    This model defines the schema for storing execution metadata in PostgreSQL
    JSONB columns. It's designed for database persistence, not for LangGraph
    state management (which uses the TypedDict ExecutionMetadata).
    """

    execution_id: str = Field(..., description="Unique execution identifier")
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for tracking"
    )
    total_execution_time_ms: float = Field(
        ..., description="Total execution time in milliseconds"
    )
    nodes_executed: List[str] = Field(..., description="List of agent names executed")
    parallel_execution: bool = Field(
        default=False, description="Whether agents ran in parallel"
    )

    # Agent outputs
    agent_outputs: Dict[
        str, Union[RefinerOutput, CriticOutput, HistorianOutput, SynthesisOutput]
    ] = Field(default_factory=dict, description="Structured outputs from each agent")

    # LLM usage metadata
    total_tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    total_cost_usd: Optional[float] = Field(None, description="Total cost in USD")
    model_used: Optional[str] = Field(None, description="Primary LLM model used")

    # Error and retry information
    errors_encountered: List[str] = Field(
        default_factory=list, description="Errors encountered during execution"
    )
    retries_attempted: int = Field(default=0, description="Number of retries attempted")

    # Workflow metadata
    workflow_version: str = Field(
        default="1.0", description="Version of the workflow executed"
    )
    success: bool = Field(..., description="Whether execution completed successfully")

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for extensibility
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "execution_id": "exec_123abc",
                "correlation_id": "corr_456def",
                "total_execution_time_ms": 3500.0,
                "nodes_executed": ["refiner", "critic", "historian", "synthesis"],
                "parallel_execution": False,
                "agent_outputs": {
                    "refiner": {"agent_name": "refiner", "refined_query": "..."},
                    "synthesis": {"agent_name": "synthesis", "final_synthesis": "..."},
                },
                "total_tokens_used": 2500,
                "total_cost_usd": 0.05,
                "model_used": "gpt-4",
                "success": True,
            }
        },
    )


# Type aliases for convenience
AgentStructuredOutput = Union[
    RefinerOutput, CriticOutput, HistorianOutput, SynthesisOutput
]


# Factory function for creating agent outputs
def create_agent_output(agent_name: str, **kwargs: Any) -> AgentStructuredOutput:
    """
    Factory function to create the appropriate agent output based on agent name.

    Args:
        agent_name: Name of the agent ("refiner", "critic", "historian", "synthesis")
        **kwargs: Agent-specific output data

    Returns:
        Appropriate agent output model instance

    Raises:
        ValueError: If agent_name is not recognized
    """
    output_classes = {
        "refiner": RefinerOutput,
        "critic": CriticOutput,
        "historian": HistorianOutput,
        "synthesis": SynthesisOutput,
    }

    if agent_name not in output_classes:
        raise ValueError(
            f"Unknown agent name: {agent_name}. Must be one of {list(output_classes.keys())}"
        )

    output_class = output_classes[agent_name]
    return cast(AgentStructuredOutput, output_class(agent_name=agent_name, **kwargs))
