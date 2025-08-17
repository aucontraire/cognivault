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

from pydantic import BaseModel, Field, ConfigDict


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

    refined_query: str = Field(..., description="The refined and clarified query")
    original_query: str = Field(..., description="The original input query")
    changes_made: List[str] = Field(
        default_factory=list,
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
        default_factory=list, description="List of ambiguities that were resolved"
    )

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
        default_factory=list, description="Implicit assumptions identified in the query"
    )
    logical_gaps: List[str] = Field(
        default_factory=list, description="Logical gaps or under-specified concepts"
    )
    biases: List[BiasType] = Field(
        default_factory=list, description="Types of biases identified in the framing"
    )
    bias_details: Dict[str, str] = Field(
        default_factory=dict,
        description="Detailed explanations for each bias type identified",
    )
    alternate_framings: List[str] = Field(
        default_factory=list, description="Suggested alternate ways to frame the query"
    )
    critique_summary: str = Field(..., description="Overall critique summary")
    issues_detected: int = Field(..., description="Number of issues detected")
    no_issues_found: bool = Field(
        default=False, description="True if query is well-scoped and neutral"
    )

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
                "bias_details": {
                    "temporal": "Assumes current AI trajectory will continue"
                },
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

    source_id: Optional[UUID] = Field(None, description="ID of the source document")
    title: Optional[str] = Field(None, description="Title of the historical source")
    relevance_score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    content_snippet: str = Field(..., description="Relevant snippet from the source")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class HistorianOutput(BaseAgentOutput):
    """Structured output from the Historian agent."""

    relevant_sources: List[HistoricalReference] = Field(
        default_factory=list, description="List of relevant historical sources found"
    )
    historical_synthesis: str = Field(..., description="Synthesized historical context")
    themes_identified: List[str] = Field(
        default_factory=list, description="Historical themes relevant to the query"
    )
    time_periods_covered: List[str] = Field(
        default_factory=list,
        description="Time periods covered in the historical context",
    )
    contextual_connections: List[str] = Field(
        default_factory=list,
        description="Connections between historical context and current query",
    )
    sources_searched: int = Field(..., description="Number of sources searched")
    relevant_sources_found: int = Field(
        ..., description="Number of relevant sources found"
    )
    no_relevant_context: bool = Field(
        default=False, description="True if no relevant historical context was found"
    )

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

    final_synthesis: str = Field(..., description="Final synthesized wiki content")
    key_themes: List[SynthesisTheme] = Field(
        default_factory=list,
        description="Key themes identified across all agent outputs",
    )
    conflicts_resolved: List[str] = Field(
        default_factory=list, description="Conflicts between agents that were resolved"
    )
    complementary_insights: List[str] = Field(
        default_factory=list,
        description="Insights that build on each other across agents",
    )
    knowledge_gaps: List[str] = Field(
        default_factory=list, description="Important aspects not covered by any agent"
    )
    meta_insights: List[str] = Field(
        default_factory=list,
        description="Higher-level insights about the analysis process",
    )
    contributing_agents: List[str] = Field(
        ..., description="List of agents that contributed to synthesis"
    )
    word_count: int = Field(..., description="Word count of the final synthesis")
    topics_extracted: List[str] = Field(
        default_factory=list, description="Key topics/concepts mentioned in synthesis"
    )

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


class ExecutionMetadata(BaseModel):
    """Complete execution metadata structure for JSONB storage."""

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
AgentOutputType = Union[RefinerOutput, CriticOutput, HistorianOutput, SynthesisOutput]


# Factory function for creating agent outputs
def create_agent_output(agent_name: str, **kwargs: Any) -> AgentOutputType:
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
    return cast(AgentOutputType, output_class(agent_name=agent_name, **kwargs))
