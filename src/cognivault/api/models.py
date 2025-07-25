"""
Centralized schema definitions for API contracts.

Schemas tagged with # EXTERNAL SCHEMA require special handling for changes.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import re


# =============================================================================
# EXTERNAL SCHEMAS - Breaking changes require migration path
# =============================================================================


# EXTERNAL SCHEMA
class WorkflowRequest(BaseModel):
    """External workflow execution request - v1.0.0"""

    query: str = Field(
        ...,
        description="The query or prompt to execute",
        min_length=1,
        max_length=10000,
        json_schema_extra={
            "example": "Analyze the impact of climate change on agriculture"
        },
    )
    agents: Optional[List[str]] = Field(
        None,
        description="List of agent names to execute (default: all available)",
        json_schema_extra={"example": ["refiner", "historian", "critic", "synthesis"]},
    )
    execution_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional execution configuration parameters",
        json_schema_extra={
            "example": {"timeout_seconds": 30, "parallel_execution": True}
        },
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Unique identifier for request correlation",
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=100,
        json_schema_extra={"example": "req-12345-abcdef"},
    )

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, v):
        """Validate agent names."""
        if v is not None:
            if not v:  # Empty list
                raise ValueError("agents list cannot be empty if provided")

            valid_agents = {"refiner", "historian", "critic", "synthesis"}
            invalid_agents = set(v) - valid_agents
            if invalid_agents:
                raise ValueError(
                    f"Invalid agents: {invalid_agents}. Valid agents: {valid_agents}"
                )

            # Check for duplicates
            if len(v) != len(set(v)):
                raise ValueError("Duplicate agents are not allowed")

        return v

    @field_validator("execution_config")
    @classmethod
    def validate_execution_config(cls, v):
        """Validate execution configuration."""
        if v is not None:
            # Validate timeout if provided
            if "timeout_seconds" in v:
                timeout = v["timeout_seconds"]
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    raise ValueError("timeout_seconds must be a positive number")
                if timeout > 600:  # 10 minutes max
                    raise ValueError("timeout_seconds cannot exceed 600 seconds")

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,  # Prevent additional fields
    )


# EXTERNAL SCHEMA
class WorkflowResponse(BaseModel):
    """External workflow execution response - v1.0.0"""

    workflow_id: str = Field(
        ...,
        description="Unique identifier for the workflow execution",
        pattern=r"^[a-f0-9-]{36}$",  # UUID format
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    status: str = Field(
        ...,
        description="Execution status",
        pattern=r"^(completed|failed|running|cancelled)$",
        json_schema_extra={"example": "completed"},
    )
    agent_outputs: Dict[str, str] = Field(
        ...,
        description="Outputs from each executed agent",
        json_schema_extra={
            "example": {
                "refiner": "Refined and clarified query",
                "historian": "Relevant historical context",
                "critic": "Critical analysis and evaluation",
                "synthesis": "Comprehensive synthesis of insights",
            }
        },
    )
    execution_time_seconds: float = Field(
        ...,
        description="Total execution time in seconds",
        ge=0.0,
        json_schema_extra={"example": 42.5},
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Request correlation identifier (if provided in request)",
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=100,
        json_schema_extra={"example": "req-12345-abcdef"},
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if execution failed",
        max_length=5000,
        json_schema_extra={"example": "Agent 'historian' failed: timeout exceeded"},
    )

    @model_validator(mode="after")
    def validate_status_consistency(self):
        """Validate status consistency with other fields."""
        if self.status == "failed" and not self.error_message:
            raise ValueError("error_message is required when status is 'failed'")

        if self.status == "completed" and not self.agent_outputs:
            raise ValueError("agent_outputs cannot be empty when status is 'completed'")

        return self

    @field_validator("agent_outputs")
    @classmethod
    def validate_agent_outputs(cls, v):
        """Validate agent outputs."""
        for agent_name, output in v.items():
            if not isinstance(output, str):
                raise ValueError(f"Output for agent '{agent_name}' must be a string")
            if len(output.strip()) == 0:
                raise ValueError(f"Output for agent '{agent_name}' cannot be empty")

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# EXTERNAL SCHEMA
class StatusResponse(BaseModel):
    """External status query response - v1.0.0"""

    workflow_id: str = Field(
        ...,
        description="Unique identifier for the workflow execution",
        pattern=r"^[a-f0-9-]{36}$",  # UUID format
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    status: str = Field(
        ...,
        description="Current execution status",
        pattern=r"^(completed|failed|running|cancelled)$",
        json_schema_extra={"example": "running"},
    )
    progress_percentage: float = Field(
        ...,
        description="Execution progress as percentage (0-100)",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 75.0},
    )
    current_agent: Optional[str] = Field(
        None,
        description="Currently executing agent (if status is 'running')",
        json_schema_extra={"example": "critic"},
    )
    estimated_completion_seconds: Optional[float] = Field(
        None,
        description="Estimated time to completion in seconds",
        ge=0.0,
        json_schema_extra={"example": 15.5},
    )

    @model_validator(mode="after")
    def validate_status_consistency(self):
        """Validate status consistency with other fields."""
        # Validate current_agent consistency
        if self.status == "running" and self.current_agent is None:
            # Allow None for running status (agent may not be determinable)
            pass
        elif self.status != "running" and self.current_agent is not None:
            raise ValueError(
                "current_agent should only be set when status is 'running'"
            )

        # Validate progress consistency
        if self.status == "completed" and self.progress_percentage != 100.0:
            raise ValueError(
                "progress_percentage must be 100.0 when status is 'completed'"
            )
        elif self.status == "failed" and self.progress_percentage == 100.0:
            raise ValueError(
                "progress_percentage should not be 100.0 when status is 'failed'"
            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# EXTERNAL SCHEMA
class CompletionRequest(BaseModel):
    """External LLM completion request - v1.0.0"""

    prompt: str = Field(
        ...,
        description="The prompt to send to the LLM",
        min_length=1,
        max_length=50000,
        json_schema_extra={
            "example": "Explain the concept of machine learning in simple terms"
        },
    )
    model: Optional[str] = Field(
        None,
        description="LLM model to use (default: configured model)",
        pattern=r"^[a-zA-Z0-9._-]+$",
        json_schema_extra={"example": "gpt-4"},
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate",
        ge=1,
        le=32000,
        json_schema_extra={"example": 1000},
    )
    temperature: Optional[float] = Field(
        None,
        description="Sampling temperature (0.0-2.0)",
        ge=0.0,
        le=2.0,
        json_schema_extra={"example": 0.7},
    )
    agent_context: Optional[str] = Field(
        None,
        description="Additional context from agent execution",
        max_length=10000,
        json_schema_extra={"example": "Previous agent outputs and workflow context"},
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# EXTERNAL SCHEMA
class CompletionResponse(BaseModel):
    """External LLM completion response - v1.0.0"""

    completion: str = Field(
        ...,
        description="Generated completion text",
        min_length=1,
        json_schema_extra={
            "example": "Machine learning is a subset of artificial intelligence..."
        },
    )
    model_used: str = Field(
        ...,
        description="The actual model that generated the completion",
        json_schema_extra={"example": "gpt-4"},
    )
    token_usage: Dict[str, int] = Field(
        ...,
        description="Token usage statistics",
        json_schema_extra={
            "example": {
                "prompt_tokens": 25,
                "completion_tokens": 150,
                "total_tokens": 175,
            }
        },
    )
    response_time_ms: float = Field(
        ...,
        description="Response time in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 1250.5},
    )
    request_id: str = Field(
        ...,
        description="Unique identifier for this completion request",
        pattern=r"^[a-f0-9-]{36}$",  # UUID format
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )

    @field_validator("token_usage")
    @classmethod
    def validate_token_usage(cls, v):
        """Validate token usage structure."""
        required_keys = {"prompt_tokens", "completion_tokens", "total_tokens"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"token_usage must contain keys: {required_keys}")

        for key, value in v.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"token_usage['{key}'] must be a non-negative integer")

        # Validate total tokens calculation
        if v["total_tokens"] != v["prompt_tokens"] + v["completion_tokens"]:
            raise ValueError(
                "total_tokens must equal prompt_tokens + completion_tokens"
            )

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# EXTERNAL SCHEMA
class LLMProviderInfo(BaseModel):
    """External LLM provider information - v1.0.0"""

    name: str = Field(
        ...,
        description="Provider name",
        pattern=r"^[a-zA-Z0-9_-]+$",
        json_schema_extra={"example": "openai"},
    )
    models: List[str] = Field(
        ...,
        description="List of available models",
        min_length=1,
        json_schema_extra={"example": ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]},
    )
    available: bool = Field(
        ...,
        description="Whether the provider is currently available",
        json_schema_extra={"example": True},
    )
    cost_per_token: Optional[float] = Field(
        None,
        description="Cost per token in USD (if available)",
        ge=0.0,
        json_schema_extra={"example": 0.00003},
    )

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        """Validate model names."""
        for model in v:
            if not isinstance(model, str) or len(model.strip()) == 0:
                raise ValueError("All model names must be non-empty strings")
            if not re.match(r"^[a-zA-Z0-9._-]+$", model):
                raise ValueError(f"Invalid model name format: {model}")

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate model names are not allowed")

        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# =============================================================================
# INTERNAL SCHEMAS - Subject to change without notice
# =============================================================================


class InternalExecutionGraph(BaseModel):
    """Internal execution graph representation - v0.1.0"""

    nodes: List[Dict[str, Any]] = Field(
        ..., description="Graph nodes representing execution units"
    )
    edges: List[Dict[str, Any]] = Field(
        ..., description="Graph edges representing execution dependencies"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional graph metadata"
    )

    model_config = ConfigDict(extra="allow")  # Internal schemas can be more flexible


class InternalAgentMetrics(BaseModel):
    """Internal agent performance metrics - v0.1.0"""

    agent_name: str = Field(..., description="Name of the agent")
    execution_time_ms: float = Field(
        ..., description="Execution time in milliseconds", ge=0.0
    )
    token_usage: Dict[str, int] = Field(..., description="Token usage statistics")
    success: bool = Field(..., description="Whether the agent execution was successful")
    timestamp: datetime = Field(..., description="Timestamp of the execution")

    model_config = ConfigDict(extra="allow")  # Internal schemas can be more flexible


# EXTERNAL SCHEMA
class WorkflowHistoryItem(BaseModel):
    """Individual workflow history entry - v1.0.0"""

    workflow_id: str = Field(
        ...,
        description="Unique identifier for the workflow execution",
        pattern=r"^[a-f0-9-]{36}$",  # UUID format
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    status: str = Field(
        ...,
        description="Workflow execution status",
        pattern=r"^(completed|failed|running|cancelled)$",
        json_schema_extra={"example": "completed"},
    )
    query: str = Field(
        ...,
        description="Original query (truncated for display)",
        max_length=200,
        json_schema_extra={"example": "Analyze the impact of climate change..."},
    )
    start_time: float = Field(
        ...,
        description="Workflow start time as Unix timestamp",
        ge=0.0,
        json_schema_extra={"example": 1703097600.0},
    )
    execution_time_seconds: float = Field(
        ...,
        description="Total execution time in seconds",
        ge=0.0,
        json_schema_extra={"example": 12.5},
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate workflow status values."""
        valid_statuses = {"completed", "failed", "running", "cancelled"}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "query": self.query,
            "start_time": self.start_time,
            "execution_time_seconds": self.execution_time_seconds,
        }


# EXTERNAL SCHEMA
class WorkflowHistoryResponse(BaseModel):
    """External workflow history response - v1.0.0"""

    workflows: List[WorkflowHistoryItem] = Field(
        ...,
        description="List of workflow execution history items",
        json_schema_extra={
            "example": [
                {
                    "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed",
                    "query": "Analyze the impact of climate change...",
                    "start_time": 1703097600.0,
                    "execution_time_seconds": 12.5,
                }
            ]
        },
    )
    total: int = Field(
        ...,
        description="Total number of workflows available (not just returned)",
        ge=0,
        json_schema_extra={"example": 150},
    )
    limit: int = Field(
        ...,
        description="Maximum number of results requested",
        ge=1,
        le=100,
        json_schema_extra={"example": 10},
    )
    offset: int = Field(
        ...,
        description="Number of results skipped",
        ge=0,
        json_schema_extra={"example": 0},
    )
    has_more: bool = Field(
        ...,
        description="Whether there are more results beyond this page",
        json_schema_extra={"example": True},
    )

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Validate limit is within acceptable range."""
        if v < 1 or v > 100:
            raise ValueError("Limit must be between 1 and 100")
        return v

    @model_validator(mode="after")
    def validate_pagination_consistency(self) -> "WorkflowHistoryResponse":
        """Validate pagination parameters are consistent."""
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")

        # Check has_more consistency
        expected_has_more = (self.offset + len(self.workflows)) < self.total
        if self.has_more != expected_has_more:
            # Fix has_more to be consistent
            self.has_more = expected_has_more

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflows": [wf.to_dict() for wf in self.workflows],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_more": self.has_more,
        }


# EXTERNAL SCHEMA
class TopicSummary(BaseModel):
    """Individual topic summary entry - v1.0.0"""

    topic_id: str = Field(
        ...,
        description="Unique identifier for the topic",
        pattern=r"^[a-f0-9-]{36}$",  # UUID format
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    name: str = Field(
        ...,
        description="Human-readable topic name",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "Machine Learning Fundamentals"},
    )
    description: str = Field(
        ...,
        description="Brief topic description",
        max_length=500,
        json_schema_extra={
            "example": "Core concepts and principles of machine learning algorithms"
        },
    )
    query_count: int = Field(
        ...,
        description="Number of queries related to this topic",
        ge=0,
        json_schema_extra={"example": 15},
    )
    last_updated: float = Field(
        ...,
        description="Last update time as Unix timestamp",
        ge=0.0,
        json_schema_extra={"example": 1703097600.0},
    )
    similarity_score: Optional[float] = Field(
        None,
        description="Similarity score for search results (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate topic name format."""
        if not v.strip():
            raise ValueError("Topic name cannot be empty or whitespace")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "description": self.description,
            "query_count": self.query_count,
            "last_updated": self.last_updated,
            "similarity_score": self.similarity_score,
        }


# EXTERNAL SCHEMA
class TopicsResponse(BaseModel):
    """External topics discovery response - v1.0.0"""

    topics: List[TopicSummary] = Field(
        ...,
        description="List of topic summaries",
        json_schema_extra={
            "example": [
                {
                    "topic_id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "Machine Learning Fundamentals",
                    "description": "Core concepts and principles of machine learning",
                    "query_count": 15,
                    "last_updated": 1703097600.0,
                    "similarity_score": 0.85,
                }
            ]
        },
    )
    total: int = Field(
        ...,
        description="Total number of topics available (not just returned)",
        ge=0,
        json_schema_extra={"example": 42},
    )
    limit: int = Field(
        ...,
        description="Maximum number of results requested",
        ge=1,
        le=100,
        json_schema_extra={"example": 10},
    )
    offset: int = Field(
        ...,
        description="Number of results skipped",
        ge=0,
        json_schema_extra={"example": 0},
    )
    has_more: bool = Field(
        ...,
        description="Whether there are more results beyond this page",
        json_schema_extra={"example": True},
    )
    search_query: Optional[str] = Field(
        None,
        description="Search query used for filtering (if any)",
        max_length=200,
        json_schema_extra={"example": "machine learning"},
    )

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Validate limit is within acceptable range."""
        if v < 1 or v > 100:
            raise ValueError("Limit must be between 1 and 100")
        return v

    @model_validator(mode="after")
    def validate_pagination_consistency(self) -> "TopicsResponse":
        """Validate pagination parameters are consistent."""
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")

        # Check has_more consistency
        expected_has_more = (self.offset + len(self.topics)) < self.total
        if self.has_more != expected_has_more:
            # Fix has_more to be consistent
            self.has_more = expected_has_more

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topics": [topic.to_dict() for topic in self.topics],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_more": self.has_more,
            "search_query": self.search_query,
        }


# EXTERNAL SCHEMA
class TopicWikiResponse(BaseModel):
    """External topic wiki knowledge response - v1.0.0"""

    topic_id: str = Field(
        ...,
        description="Unique identifier for the topic",
        pattern=r"^[a-f0-9-]{36}$",  # UUID format
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    topic_name: str = Field(
        ...,
        description="Human-readable topic name",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "Machine Learning Fundamentals"},
    )
    content: str = Field(
        ...,
        description="Synthesized knowledge content for the topic",
        min_length=1,
        max_length=10000,
        json_schema_extra={
            "example": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed..."
        },
    )
    last_updated: float = Field(
        ...,
        description="Last update time as Unix timestamp",
        ge=0.0,
        json_schema_extra={"example": 1703097600.0},
    )
    sources: List[str] = Field(
        ...,
        description="List of source workflow IDs that contributed to this knowledge",
        max_length=50,  # Limit number of sources
        json_schema_extra={
            "example": [
                "550e8400-e29b-41d4-a716-446655440001",
                "550e8400-e29b-41d4-a716-446655440002",
            ]
        },
    )
    query_count: int = Field(
        ...,
        description="Number of queries that contributed to this topic knowledge",
        ge=0,
        json_schema_extra={"example": 15},
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score for the synthesized content (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92},
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: List[str]) -> List[str]:
        """Validate source workflow IDs."""
        for source_id in v:
            if not re.match(r"^[a-f0-9-]{36}$", source_id):
                raise ValueError(f"Invalid source workflow ID format: {source_id}")

        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in v:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)

        return unique_sources

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content format."""
        content = v.strip()
        if not content:
            raise ValueError("Content cannot be empty or whitespace")
        return content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "content": self.content,
            "last_updated": self.last_updated,
            "sources": self.sources,
            "query_count": self.query_count,
            "confidence_score": self.confidence_score,
        }


# EXTERNAL SCHEMA
class WorkflowMetadata(BaseModel):
    """Individual workflow metadata entry - v1.0.0"""

    workflow_id: str = Field(
        ...,
        description="Unique identifier for the workflow",
        pattern=r"^[a-zA-Z0-9_-]+$",
        json_schema_extra={"example": "academic_research"},
    )
    name: str = Field(
        ...,
        description="Human-readable workflow name",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "Academic Research Analysis"},
    )
    description: str = Field(
        ...,
        description="Detailed workflow description",
        max_length=1000,
        json_schema_extra={
            "example": "Comprehensive academic research workflow with peer-review standards"
        },
    )
    version: str = Field(
        ...,
        description="Workflow version",
        pattern=r"^\d+\.\d+\.\d+$",
        json_schema_extra={"example": "1.0.0"},
    )
    category: str = Field(
        ...,
        description="Primary workflow category",
        min_length=1,
        max_length=50,
        json_schema_extra={"example": "academic"},
    )
    tags: List[str] = Field(
        ...,
        description="Workflow tags for filtering and search",
        max_length=20,  # Limit number of tags
        json_schema_extra={
            "example": ["academic", "research", "scholarly", "analysis"]
        },
    )
    created_by: str = Field(
        ...,
        description="Workflow author or creator",
        max_length=100,
        json_schema_extra={"example": "CogniVault Team"},
    )
    created_at: float = Field(
        ...,
        description="Creation time as Unix timestamp",
        ge=0.0,
        json_schema_extra={"example": 1703097600.0},
    )
    estimated_execution_time: str = Field(
        ...,
        description="Estimated execution time range",
        max_length=50,
        json_schema_extra={"example": "45-60 seconds"},
    )
    complexity_level: str = Field(
        ...,
        description="Workflow complexity level",
        pattern=r"^(low|medium|high|expert)$",
        json_schema_extra={"example": "high"},
    )
    node_count: int = Field(
        ...,
        description="Number of nodes in the workflow",
        ge=1,
        json_schema_extra={"example": 7},
    )
    use_cases: List[str] = Field(
        ...,
        description="Common use cases for this workflow",
        max_length=10,  # Limit number of use cases
        json_schema_extra={"example": ["dissertation_research", "literature_review"]},
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate workflow tags."""
        if not v:
            raise ValueError("At least one tag must be provided")

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError("All tags must be strings")
            tag_clean = tag.strip().lower()
            if not tag_clean:
                raise ValueError("Tags cannot be empty or whitespace")
            if len(tag_clean) > 30:
                raise ValueError("Tags cannot exceed 30 characters")
            if tag_clean not in seen:
                seen.add(tag_clean)
                unique_tags.append(tag_clean)

        return unique_tags

    @field_validator("use_cases")
    @classmethod
    def validate_use_cases(cls, v: List[str]) -> List[str]:
        """Validate use cases."""
        for use_case in v:
            if not isinstance(use_case, str):
                raise ValueError("All use cases must be strings")
            if not use_case.strip():
                raise ValueError("Use cases cannot be empty or whitespace")
            if len(use_case) > 100:
                raise ValueError("Use cases cannot exceed 100 characters")

        return [uc.strip() for uc in v]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "estimated_execution_time": self.estimated_execution_time,
            "complexity_level": self.complexity_level,
            "node_count": self.node_count,
            "use_cases": self.use_cases,
        }


# EXTERNAL SCHEMA
class WorkflowsResponse(BaseModel):
    """External workflows discovery response - v1.0.0"""

    workflows: List[WorkflowMetadata] = Field(
        ...,
        description="List of available workflow metadata",
        json_schema_extra={
            "example": [
                {
                    "workflow_id": "academic_research",
                    "name": "Academic Research Analysis",
                    "description": "Comprehensive academic research workflow",
                    "version": "1.0.0",
                    "category": "academic",
                    "tags": ["academic", "research", "scholarly"],
                    "created_by": "CogniVault Team",
                    "created_at": 1703097600.0,
                    "estimated_execution_time": "45-60 seconds",
                    "complexity_level": "high",
                    "node_count": 7,
                    "use_cases": ["dissertation_research", "literature_review"],
                }
            ]
        },
    )
    categories: List[str] = Field(
        ...,
        description="Available workflow categories for filtering",
        json_schema_extra={"example": ["academic", "legal", "business", "general"]},
    )
    total: int = Field(
        ...,
        description="Total number of workflows available (not just returned)",
        ge=0,
        json_schema_extra={"example": 25},
    )
    limit: int = Field(
        ...,
        description="Maximum number of results requested",
        ge=1,
        le=100,
        json_schema_extra={"example": 10},
    )
    offset: int = Field(
        ...,
        description="Number of results skipped",
        ge=0,
        json_schema_extra={"example": 0},
    )
    has_more: bool = Field(
        ...,
        description="Whether there are more results beyond this page",
        json_schema_extra={"example": True},
    )
    search_query: Optional[str] = Field(
        None,
        description="Search query used for filtering (if any)",
        max_length=200,
        json_schema_extra={"example": "academic research"},
    )
    category_filter: Optional[str] = Field(
        None,
        description="Category filter applied (if any)",
        max_length=50,
        json_schema_extra={"example": "academic"},
    )
    complexity_filter: Optional[str] = Field(
        None,
        description="Complexity filter applied (if any)",
        pattern=r"^(low|medium|high|expert)$",
        json_schema_extra={"example": "high"},
    )

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Validate limit is within acceptable range."""
        if v < 1 or v > 100:
            raise ValueError("Limit must be between 1 and 100")
        return v

    @model_validator(mode="after")
    def validate_pagination_consistency(self) -> "WorkflowsResponse":
        """Validate pagination parameters are consistent."""
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")

        # Check has_more consistency
        expected_has_more = (self.offset + len(self.workflows)) < self.total
        if self.has_more != expected_has_more:
            # Fix has_more to be consistent
            self.has_more = expected_has_more

        return self

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: List[str]) -> List[str]:
        """Validate and normalize categories."""
        if not v:
            return []

        # Remove duplicates and normalize
        unique_categories = list(set(cat.strip().lower() for cat in v if cat.strip()))
        return sorted(unique_categories)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflows": [wf.to_dict() for wf in self.workflows],
            "categories": self.categories,
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_more": self.has_more,
            "search_query": self.search_query,
            "category_filter": self.category_filter,
            "complexity_filter": self.complexity_filter,
        }
