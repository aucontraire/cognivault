"""
LangGraph state schemas for CogniVault agents.

This module provides TypedDict definitions for type-safe state management
in LangGraph StateGraph execution. Each agent output is strictly typed
to ensure consistency and enable proper validation.

Design Principles:
- Type safety through TypedDict definitions
- Clear separation of agent outputs
- Validation helpers for state integrity
- Comprehensive documentation for maintainability
"""

from typing import TypedDict, List, Dict, Any, Optional, Union, Annotated
from datetime import datetime, timezone
from dataclasses import dataclass
import operator


class RefinerOutput(TypedDict):
    """
    Output schema for the RefinerAgent.

    The RefinerAgent transforms raw user queries into structured,
    clarified prompts for downstream processing.
    """

    refined_question: str
    """The clarified and structured version of the original query."""

    topics: List[str]
    """List of identified topics and themes in the query."""

    confidence: float
    """Confidence score (0.0-1.0) in the refinement quality."""

    processing_notes: Optional[str]
    """Optional notes about the refinement process."""

    timestamp: str
    """ISO timestamp when refinement was completed."""


class CriticOutput(TypedDict):
    """
    Output schema for the CriticAgent.

    The CriticAgent provides analytical critique and evaluation
    of the refined query and suggests improvements.
    """

    critique: str
    """Detailed critique of the refined question and approach."""

    suggestions: List[str]
    """List of specific improvement suggestions."""

    severity: str
    """Severity level: 'low', 'medium', 'high', 'critical'."""

    strengths: List[str]
    """Identified strengths in the current approach."""

    weaknesses: List[str]
    """Identified weaknesses or gaps in the approach."""

    confidence: float
    """Confidence score (0.0-1.0) in the critique accuracy."""

    timestamp: str
    """ISO timestamp when critique was completed."""


class HistorianOutput(TypedDict):
    """
    Output schema for the HistorianAgent.

    The HistorianAgent retrieves and analyzes historical context
    using intelligent search and LLM-powered relevance analysis.
    """

    historical_summary: str
    """Synthesized historical context relevant to the current query."""

    retrieved_notes: List[str]
    """List of filepaths for notes that were retrieved and used."""

    search_results_count: int
    """Number of search results found before filtering."""

    filtered_results_count: int
    """Number of results after relevance filtering."""

    search_strategy: str
    """Type of search strategy used (e.g., 'hybrid', 'tag-based', 'keyword')."""

    topics_found: List[str]
    """List of topics identified in the retrieved historical content."""

    confidence: float
    """Confidence score (0.0-1.0) in the historical context relevance."""

    llm_analysis_used: bool
    """Whether LLM was used for relevance analysis and synthesis."""

    metadata: Dict[str, Any]
    """Additional metadata about the historical search process."""

    timestamp: str
    """ISO timestamp when historical analysis was completed."""


class SynthesisOutput(TypedDict):
    """
    Output schema for the SynthesisAgent.

    The SynthesisAgent generates final synthesis from multiple
    agent outputs, creating coherent and comprehensive analysis.
    """

    final_analysis: str
    """Comprehensive final analysis integrating all inputs."""

    key_insights: List[str]
    """List of key insights derived from the analysis."""

    sources_used: List[str]
    """List of source agents/outputs used in synthesis."""

    themes_identified: List[str]
    """Major themes identified across all inputs."""

    conflicts_resolved: int
    """Number of conflicts resolved during synthesis."""

    confidence: float
    """Confidence score (0.0-1.0) in the synthesis quality."""

    metadata: Dict[str, Any]
    """Additional metadata about the synthesis process."""

    timestamp: str
    """ISO timestamp when synthesis was completed."""


class ExecutionMetadata(TypedDict):
    """
    Metadata about the LangGraph execution process.
    """

    execution_id: str
    """Unique identifier for this execution."""

    correlation_id: Optional[str]
    """Correlation ID for event tracking and WebSocket filtering."""

    start_time: str
    """ISO timestamp when execution started."""

    orchestrator_type: str
    """Type of orchestrator: 'langgraph-real'."""

    agents_requested: List[str]
    """List of agents requested for execution."""

    execution_mode: str
    """Execution mode: 'langgraph-real'."""

    phase: str
    """Implementation phase: 'phase2_0'."""


class CogniVaultState(TypedDict):
    """
    Master state schema for CogniVault LangGraph execution.

    This represents the complete state that flows through the
    LangGraph StateGraph during execution. Each agent contributes
    its typed output to this shared state.

    State Flow:
    1. Initial state created with query and metadata
    2. Refiner adds RefinerOutput to state["refiner"]
    3. Critic and Historian run in parallel after refiner
    4. Critic adds CriticOutput to state["critic"]
    5. Historian adds HistorianOutput to state["historian"]
    6. Synthesis adds SynthesisOutput to state["synthesis"]
    7. Final state contains all agent outputs
    """

    # Core input
    query: str
    """The original user query to process."""

    # Agent outputs (populated during execution)
    refiner: Optional[RefinerOutput]
    """Output from the RefinerAgent (populated after refiner node)."""

    critic: Optional[CriticOutput]
    """Output from the CriticAgent (populated after critic node)."""

    historian: Optional[HistorianOutput]
    """Output from the HistorianAgent (populated after historian node)."""

    synthesis: Optional[SynthesisOutput]
    """Output from the SynthesisAgent (populated after synthesis node)."""

    # Execution tracking
    execution_metadata: ExecutionMetadata
    """Metadata about the current execution."""

    # Error handling
    errors: Annotated[List[Dict[str, Any]], operator.add]
    """List of errors encountered during execution."""

    # Success tracking
    successful_agents: Annotated[List[str], operator.add]
    """List of agents that completed successfully."""

    failed_agents: Annotated[List[str], operator.add]
    """List of agents that failed during execution."""


# Type aliases for improved clarity
LangGraphState = CogniVaultState
"""Alias for CogniVaultState to improve code readability."""

AgentOutput = Union[RefinerOutput, CriticOutput, HistorianOutput, SynthesisOutput]
"""Union type for any agent output schema."""


def create_initial_state(
    query: str, execution_id: str, correlation_id: Optional[str] = None
) -> CogniVaultState:
    """
    Create initial LangGraph state for execution.

    Parameters
    ----------
    query : str
        The user query to process
    execution_id : str
        Unique identifier for this execution

    Returns
    -------
    CogniVaultState
        Initial state ready for LangGraph execution
    """
    now = datetime.now(timezone.utc).isoformat()

    return CogniVaultState(
        query=query,
        refiner=None,
        critic=None,
        historian=None,
        synthesis=None,
        execution_metadata=ExecutionMetadata(
            execution_id=execution_id,
            correlation_id=correlation_id,
            start_time=now,
            orchestrator_type="langgraph-real",
            agents_requested=["refiner", "critic", "historian", "synthesis"],
            execution_mode="langgraph-real",
            phase="phase2_1",
        ),
        errors=[],
        successful_agents=[],
        failed_agents=[],
    )


def validate_state_integrity(state: CogniVaultState) -> bool:
    """
    Validate LangGraph state integrity.

    Parameters
    ----------
    state : CogniVaultState
        State to validate

    Returns
    -------
    bool
        True if state is valid, False otherwise
    """
    try:
        # Check that state is a dict-like object
        if not isinstance(state, dict):
            return False

        # Check required fields
        if not state.get("query"):
            return False

        if not state.get("execution_metadata"):
            return False

        # Check metadata integrity
        metadata = state["execution_metadata"]
        if not metadata.get("execution_id"):
            return False

        # Validate agent outputs if present
        if state.get("refiner"):
            refiner: Optional[RefinerOutput] = state["refiner"]
            if (
                refiner is None
                or not refiner.get("refined_question")
                or not refiner.get("timestamp")
            ):
                return False

        if state.get("critic"):
            critic: Optional[CriticOutput] = state["critic"]
            if (
                critic is None
                or not critic.get("critique")
                or not critic.get("timestamp")
            ):
                return False

        if state.get("historian"):
            historian: Optional[HistorianOutput] = state["historian"]
            if (
                historian is None
                or not historian.get("historical_summary")
                or not historian.get("timestamp")
            ):
                return False

        if state.get("synthesis"):
            synthesis: Optional[SynthesisOutput] = state["synthesis"]
            if (
                synthesis is None
                or not synthesis.get("final_analysis")
                or not synthesis.get("timestamp")
            ):
                return False

        return True

    except (KeyError, TypeError):
        return False


def get_agent_output(state: CogniVaultState, agent_name: str) -> Optional[AgentOutput]:
    """
    Get typed agent output from state.

    Parameters
    ----------
    state : CogniVaultState
        Current state
    agent_name : str
        Name of agent ('refiner', 'critic', 'historian', 'synthesis')

    Returns
    -------
    AgentOutput or None
        Typed agent output if available
    """
    agent_name = agent_name.lower()

    if agent_name == "refiner":
        return state.get("refiner")
    elif agent_name == "critic":
        return state.get("critic")
    elif agent_name == "historian":
        return state.get("historian")
    elif agent_name == "synthesis":
        return state.get("synthesis")
    else:
        return None


def set_agent_output(
    state: CogniVaultState, agent_name: str, output: AgentOutput
) -> CogniVaultState:
    """
    Set typed agent output in state.

    Parameters
    ----------
    state : CogniVaultState
        Current state
    agent_name : str
        Name of agent ('refiner', 'critic', 'historian', 'synthesis')
    output : AgentOutput
        Typed agent output to set

    Returns
    -------
    CogniVaultState
        Updated state with agent output
    """
    # Create a deep copy to avoid mutations
    new_state = state.copy()
    # Deep copy mutable lists
    new_state["successful_agents"] = state["successful_agents"].copy()
    new_state["failed_agents"] = state["failed_agents"].copy()
    new_state["errors"] = state["errors"].copy()

    agent_name = agent_name.lower()

    if agent_name == "refiner" and isinstance(output, dict):
        new_state["refiner"] = output  # type: ignore
    elif agent_name == "critic" and isinstance(output, dict):
        new_state["critic"] = output  # type: ignore
    elif agent_name == "historian" and isinstance(output, dict):
        new_state["historian"] = output  # type: ignore
    elif agent_name == "synthesis" and isinstance(output, dict):
        new_state["synthesis"] = output  # type: ignore

    # Track successful completion - append single item for LangGraph reducer
    if agent_name not in new_state["successful_agents"]:
        new_state["successful_agents"].append(agent_name)

    return new_state


def record_agent_error(
    state: CogniVaultState, agent_name: str, error: Exception
) -> CogniVaultState:
    """
    Record agent execution error in state.

    Handles both complete and partial state objects gracefully.

    Parameters
    ----------
    state : CogniVaultState
        Current state (may be partial)
    agent_name : str
        Name of failed agent
    error : Exception
        Error that occurred

    Returns
    -------
    CogniVaultState
        Updated state with error recorded
    """
    new_state = state.copy()
    # Deep copy mutable lists - handle partial state
    new_state["successful_agents"] = (
        state.get("successful_agents", []).copy()
        if isinstance(state.get("successful_agents"), list)
        else []
    )
    new_state["failed_agents"] = (
        state.get("failed_agents", []).copy()
        if isinstance(state.get("failed_agents"), list)
        else []
    )
    new_state["errors"] = (
        state.get("errors", []).copy() if isinstance(state.get("errors"), list) else []
    )

    error_record = {
        "agent": agent_name,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    new_state["errors"].append(error_record)

    if agent_name not in new_state["failed_agents"]:
        new_state["failed_agents"].append(agent_name)

    return new_state


@dataclass
class CogniVaultContext:
    """
    Context schema for CogniVault LangGraph execution.

    This context is passed to all nodes during graph execution,
    providing thread-scoped information and configuration.
    """

    thread_id: str
    execution_id: str
    query: str
    correlation_id: Optional[str] = None
    enable_checkpoints: bool = False


# Export commonly used types for convenience
__all__ = [
    "CogniVaultState",
    "LangGraphState",
    "RefinerOutput",
    "CriticOutput",
    "HistorianOutput",
    "SynthesisOutput",
    "AgentOutput",
    "ExecutionMetadata",
    "CogniVaultContext",
    "create_initial_state",
    "validate_state_integrity",
    "get_agent_output",
    "set_agent_output",
    "record_agent_error",
]
