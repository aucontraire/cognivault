"""
Centralized schema definitions for API contracts.

Schemas tagged with # EXTERNAL SCHEMA require special handling for changes.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


# =============================================================================
# EXTERNAL SCHEMAS - Breaking changes require migration path
# =============================================================================


# EXTERNAL SCHEMA
@dataclass
class WorkflowRequest:
    """External workflow execution request - v1.0.0"""

    query: str
    agents: Optional[List[str]] = None
    execution_config: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None


# EXTERNAL SCHEMA
@dataclass
class WorkflowResponse:
    """External workflow execution response - v1.0.0"""

    workflow_id: str
    status: str
    agent_outputs: Dict[str, str]
    execution_time_seconds: float
    correlation_id: Optional[str] = None
    error_message: Optional[str] = None


# EXTERNAL SCHEMA
@dataclass
class StatusResponse:
    """External status query response - v1.0.0"""

    workflow_id: str
    status: str
    progress_percentage: float
    current_agent: Optional[str] = None
    estimated_completion_seconds: Optional[float] = None


# EXTERNAL SCHEMA
@dataclass
class CompletionRequest:
    """External LLM completion request - v1.0.0"""

    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    agent_context: Optional[str] = None


# EXTERNAL SCHEMA
@dataclass
class CompletionResponse:
    """External LLM completion response - v1.0.0"""

    completion: str
    model_used: str
    token_usage: Dict[str, int]
    response_time_ms: float
    request_id: str


# EXTERNAL SCHEMA
@dataclass
class LLMProvider:
    """External LLM provider information - v1.0.0"""

    name: str
    models: List[str]
    available: bool
    cost_per_token: Optional[float] = None


# =============================================================================
# INTERNAL SCHEMAS - Subject to change without notice
# =============================================================================


@dataclass
class InternalExecutionGraph:
    """Internal execution graph representation - v0.1.0"""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class InternalAgentMetrics:
    """Internal agent performance metrics - v0.1.0"""

    agent_name: str
    execution_time_ms: float
    token_usage: Dict[str, int]
    success: bool
    timestamp: datetime
